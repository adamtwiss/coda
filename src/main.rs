#![allow(dead_code)]

mod types;
mod bitboard;
mod zobrist;
mod attacks;
mod board;
mod movegen;
mod eval;
mod see;
mod tt;
mod movepicker;
mod search;
mod uci;
mod epd;
pub mod nnue;
pub mod book;
pub mod tb;
pub mod binpack;
pub mod datagen;
pub mod nnue_export;
pub mod bullet_convert;
mod cuckoo;

use board::Board;
use movegen::{perft, perft_divide};
use clap::{Parser, Subcommand};

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        attacks::init_attacks();
        bitboard::init_bitboards();
        zobrist::init_zobrist();
        board::init_castle_masks();
        search::init_lmr();
        cuckoo::init_cuckoo();
        nnue::init_nnue();
    });
}

/// Coda Chess Engine — Chess Optimised, Developed Agentically
#[derive(Parser)]
#[command(name = "coda", about = "Coda Chess Engine", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// NNUE network file path
    #[arg(long = "nnue", short = 'n', global = true)]
    nnue: Option<String>,

    /// Opening book file path
    #[arg(long = "book", short = 'b')]
    book: Option<String>,

    /// Use classical (PeSTO) eval instead of NNUE
    #[arg(long = "classical")]
    classical: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Search benchmark (default depth 13)
    Bench {
        /// Search depth
        #[arg(default_value_t = 13)]
        depth: i32,
        /// Number of threads
        #[arg(long = "threads", short = 't', default_value_t = 1)]
        threads: usize,
    },
    /// Run EPD test suite
    Epd {
        /// EPD file path
        #[arg(default_value = "testdata/wac.epd")]
        path: String,
        /// Time per position (ms)
        #[arg(default_value_t = 5000)]
        time: u64,
        /// Maximum positions (0 = all)
        #[arg(default_value_t = 0)]
        max: usize,
    },
    /// Perft with divide
    Perft {
        /// Search depth
        #[arg(default_value_t = 5)]
        depth: u32,
        /// FEN position (default: startpos)
        #[arg(trailing_var_arg = true)]
        fen: Vec<String>,
    },
    /// Perft benchmark suite (6 standard positions)
    PerftBench,
    /// Download NNUE net from net.txt URL
    FetchNet,
    /// NNUE network health check
    CheckNet {
        /// Path to .nnue file
        path: String,
    },
    /// Generate training data (SF binpack format)
    Datagen {
        /// Output binpack file
        #[arg(long, short = 'o', default_value = "data.binpack")]
        output: String,
        /// Search depth
        #[arg(long, short = 'd', default_value_t = 8)]
        depth: i32,
        /// Number of games
        #[arg(long, short = 'g', default_value_t = 1000)]
        games: usize,
        /// Number of threads
        #[arg(long, short = 't', default_value_t = 1)]
        threads: usize,
        /// Hash table size (MB)
        #[arg(long, default_value_t = 16)]
        hash: usize,
        /// Blunder rate (0.0 - 1.0)
        #[arg(long, default_value_t = 0.0)]
        blunder: f64,
        /// Source EPD for material removal mode
        #[arg(long)]
        epd: Option<String>,
    },
    /// Sample positions from binpack to EPD
    SamplePositions {
        /// Input binpack file
        #[arg(long, short = 'i')]
        input: String,
        /// Output EPD file
        #[arg(long, short = 'o', default_value = "positions.epd")]
        output: String,
        /// Number of positions to sample
        #[arg(long, short = 'n', default_value_t = 1_000_000)]
        count: usize,
        /// Sample rate (0.0 = keep all until count)
        #[arg(long, default_value_t = 0.0)]
        rate: f64,
    },
    /// Convert Bullet quantised.bin to .nnue
    ConvertBullet {
        /// Input quantised.bin path
        #[arg(long, short = 'i')]
        input: String,
        /// Output .nnue path
        #[arg(long, short = 'o', default_value = "net.nnue")]
        output: String,
        /// Use SCReLU activation
        #[arg(long)]
        screlu: bool,
        /// Use pairwise multiplication
        #[arg(long)]
        pairwise: bool,
        /// L1 hidden layer size (0 = no hidden layers, v5)
        #[arg(long, default_value_t = 0)]
        hidden: usize,
        /// L2 hidden layer size
        #[arg(long, default_value_t = 0)]
        hidden2: usize,
        /// Use int8 quantization for L1
        #[arg(long)]
        int8l1: bool,
    },
    /// Convert .nnue to Bullet checkpoint (for transfer learning)
    ConvertCheckpoint {
        /// Input .nnue path
        #[arg(long)]
        nnue: String,
        /// Output checkpoint directory
        #[arg(long, short = 'o', default_value = "v7_checkpoint")]
        output: String,
        /// Feature transformer size
        #[arg(long, default_value_t = 1024)]
        ft: usize,
        /// L1 hidden layer size
        #[arg(long, default_value_t = 16)]
        l1: usize,
        /// L2 hidden layer size
        #[arg(long, default_value_t = 32)]
        l2: usize,
    },
}

fn main() {
    init();

    let cli = Cli::parse();

    match cli.command {
        None => {
            // UCI mode (default)
            let nnue_ref = cli.nnue.as_deref();
            let book_ref = cli.book.as_deref();
            uci::uci_loop_with_nnue(nnue_ref, book_ref, cli.classical);
        }

        Some(Commands::Bench { depth, threads }) => {
            let nnue_path = cli.nnue.as_deref();
            let start = std::time::Instant::now();
            if threads <= 1 {
                let nodes = search::bench(depth, nnue_path);
                let elapsed = start.elapsed();
                let nps = if elapsed.as_secs_f64() > 0.0 {
                    (nodes as f64 / elapsed.as_secs_f64()) as u64
                } else { 0 };
                println!("\n{} nodes {} nps", nodes, nps);
            } else {
                let nnue_owned = nnue_path.map(|s| s.to_string());
                let handles: Vec<_> = (0..threads).map(|_| {
                    let np = nnue_owned.clone();
                    std::thread::spawn(move || {
                        search::bench_silent(depth, np.as_deref())
                    })
                }).collect();
                let mut total_nodes = 0u64;
                for h in handles {
                    total_nodes += h.join().unwrap();
                }
                let elapsed = start.elapsed();
                let nps = if elapsed.as_secs_f64() > 0.0 {
                    (total_nodes as f64 / elapsed.as_secs_f64()) as u64
                } else { 0 };
                println!("\n{} nodes {} nps", total_nodes, nps);
            }
        }

        Some(Commands::Epd { path, time, max }) => {
            epd::run_epd(&path, time, max, cli.nnue.as_deref());
        }

        Some(Commands::Perft { depth, fen }) => {
            let fen_str = if fen.is_empty() {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()
            } else {
                fen.join(" ")
            };
            let mut board = Board::from_fen(&fen_str);
            println!("Position: {}", fen_str);
            println!("Depth: {}", depth);
            println!();
            let start = std::time::Instant::now();
            let nodes = perft_divide(&mut board, depth);
            let elapsed = start.elapsed();
            let nps = if elapsed.as_secs_f64() > 0.0 {
                (nodes as f64 / elapsed.as_secs_f64()) as u64
            } else { 0 };
            println!("\nNodes: {}", nodes);
            println!("Time: {:.3}s", elapsed.as_secs_f64());
            println!("NPS: {}", nps);
        }

        Some(Commands::PerftBench) => {
            run_perft_bench();
        }

        Some(Commands::FetchNet) => {
            run_fetch_net();
        }

        Some(Commands::CheckNet { path }) => {
            run_check_net(&path);
        }

        Some(Commands::Datagen { output, depth, games, threads, hash, blunder, epd }) => {
            let nnue_path = cli.nnue.unwrap_or_default();
            let mode = if let Some(epd_path) = epd {
                datagen::DatagenMode::Material { source_epd: epd_path }
            } else {
                datagen::DatagenMode::SelfPlay { blunder_rate: blunder }
            };
            let config = datagen::DatagenConfig {
                nnue_path,
                output_path: output,
                mode,
                depth,
                num_games: games,
                threads,
                hash_mb: hash,
            };
            datagen::run_datagen(&config);
        }

        Some(Commands::SamplePositions { input, output, count, rate }) => {
            run_sample_positions(&input, &output, count, rate);
        }

        Some(Commands::ConvertBullet { input, output, screlu, pairwise, hidden, hidden2, int8l1 }) => {
            let result = if hidden > 0 {
                bullet_convert::convert_v7(&input, &output, screlu, pairwise, hidden, hidden2, int8l1)
            } else {
                bullet_convert::convert_v5(&input, &output, screlu, pairwise)
            };
            if let Err(e) = result {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Some(Commands::ConvertCheckpoint { nnue, output, ft, l1, l2 }) => {
            if let Err(e) = nnue_export::nnue_to_bullet_checkpoint(&nnue, &output, ft, l1, l2) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn run_perft_bench() {
    let positions = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324u64),
        ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
        ("pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083),
        ("pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292),
        ("pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194),
        ("pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551),
    ];

    let start = std::time::Instant::now();
    let mut total_nodes = 0u64;
    let mut all_passed = true;

    for (name, fen, depth, expected) in &positions {
        let mut board = Board::from_fen(fen);
        let nodes = perft(&mut board, *depth);
        total_nodes += nodes;
        let passed = nodes == *expected;
        if !passed { all_passed = false; }
        println!("{:10} depth {} : {:>12} {} (expected {})",
            name, depth, nodes, if passed { "OK" } else { "FAIL" }, expected);
    }

    let elapsed = start.elapsed();
    let nps = (total_nodes as f64 / elapsed.as_secs_f64()) as u64;
    println!("\nTotal: {} nodes in {:.3}s = {} NPS", total_nodes, elapsed.as_secs_f64(), nps);
    if all_passed {
        println!("All perft tests PASSED");
    } else {
        println!("Some perft tests FAILED");
        std::process::exit(1);
    }
}

fn run_fetch_net() {
    let net_txt = std::path::Path::new("net.txt");
    if !net_txt.exists() {
        eprintln!("Error: net.txt not found in current directory");
        std::process::exit(1);
    }
    let url = std::fs::read_to_string(net_txt).unwrap().trim().to_string();
    if url.is_empty() {
        eprintln!("Error: net.txt is empty");
        std::process::exit(1);
    }
    let fname = url.rsplit('/').next().unwrap_or("net.nnue");
    let out_path = std::path::Path::new(fname);
    if out_path.exists() {
        println!("{} already exists, skipping download", fname);
        return;
    }
    println!("Downloading {} ...", url);
    let output = std::process::Command::new("curl")
        .args(["-sL", &url, "-o", fname])
        .status();
    match output {
        Ok(status) if status.success() => {
            let size = std::fs::metadata(fname).map(|m| m.len()).unwrap_or(0);
            println!("Downloaded {} ({} bytes)", fname, size);
        }
        _ => {
            eprintln!("Error: failed to download {}", url);
            std::process::exit(1);
        }
    }
}

fn run_check_net(net_path: &str) {
    use nnue::{NNUENet, NNUEAccumulator};

    let net = match NNUENet::load(net_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Error loading net: {}", e);
            std::process::exit(1);
        }
    };

    let mut arch = format!("FT={}", net.hidden_size);
    if net.l1_size > 0 { arch += &format!(" L1={}", net.l1_size); }
    if net.l2_size > 0 { arch += &format!(" L2={}", net.l2_size); }
    if net.use_screlu { arch += " SCReLU"; }
    if net.use_pairwise { arch += " pairwise"; }
    if net.l1_scale == 64 { arch += " int8L1"; }
    println!("Net: {} ({})", net_path, arch);
    println!();

    let eval_fen = |fen: &str| -> i32 {
        let board = Board::from_fen(fen);
        let h = net.hidden_size;
        let mut acc = NNUEAccumulator::new(h);
        // Full recompute for both perspectives
        let piece_count = (board.pieces[types::PAWN as usize]
            | board.pieces[types::KNIGHT as usize]
            | board.pieces[types::BISHOP as usize]
            | board.pieces[types::ROOK as usize]
            | board.pieces[types::QUEEN as usize]).count_ones();
        acc.force_recompute(&net, &board);
        net.forward(&acc, board.side_to_move, piece_count)
    };

    let test_positions = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -50, 50),
        ("miss pawn", "rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -200, -20),
        ("miss knight", "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -500, -80),
        ("miss bishop", "rnbqk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -500, -80),
        ("miss rook", "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -800, -120),
        ("miss queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -1500, -200),
        ("EG rook up", "4k3/8/8/8/8/8/PPPPPPPP/R3K3 w Q - 0 1", 400, 2500),
        ("EG queen up", "4k3/8/8/8/8/8/PPPPPPPP/3QK3 w - - 0 1", 500, 3000),
    ];

    let mut issues = 0;
    println!("{:<18} {:>8} {:>9} {:>9}  {}", "Position", "Score", "Min", "Max", "Status");
    println!("{:<18} {:>8} {:>9} {:>9}  {}", "--------", "--------", "--------", "--------", "------");
    for (name, fen, min, max) in &test_positions {
        let score = eval_fen(fen);
        let status = if score >= *min && score <= *max { "OK" }
            else if score < *min { issues += 1; "LOW" }
            else { issues += 1; "HIGH" };
        println!("{:<18} {:>8} {:>9} {:>9}  {}", name, score, min, max, status);
    }

    println!();
    if issues == 0 {
        println!("All checks passed.");
    } else {
        println!("{} issue(s) found — eval scale may be collapsed or miscalibrated.", issues);
    }
}

fn run_sample_positions(input: &str, output: &str, n: usize, sample_rate: f64) {
    use sfbinpack::CompressedTrainingDataEntryReader;
    use std::io::Write;

    println!("Sampling ~{} positions from {}", n, input);
    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(output).unwrap_or_else(|_| panic!("Failed to create {}", output))
    );

    let mut count = 0usize;
    let mut total = 0usize;
    let mut rng_state = 0xDEADBEEF42u64;
    let mut reader = reader;

    while reader.has_next() {
        let entry = reader.next();
        total += 1;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        if entry.score.unsigned_abs() > 2000 { continue; }
        if entry.pos.is_checked(entry.pos.side_to_move()) { continue; }

        let keep = if sample_rate > 0.0 {
            let rand_val = (rng_state >> 11) as f64 / ((1u64 << 53) as f64);
            rand_val < sample_rate
        } else {
            true
        };

        if keep {
            let fen = entry.pos.fen().unwrap_or_default();
            writeln!(out, "{}", fen).expect("write failed");
            count += 1;
            if count >= n { break; }
        }

        if total % 10_000_000 == 0 {
            println!("  scanned {}M positions, sampled {}", total / 1_000_000, count);
        }
    }

    println!("Sampled {} positions from {} scanned → {}", count, total, output);
}
