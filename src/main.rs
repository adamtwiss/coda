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
        /// Force-capture rate (0.0 - 1.0): probability of playing a random capture
        /// instead of best move, creating material-imbalanced games for training diversity
        #[arg(long, default_value_t = 0.0)]
        force_captures: f64,
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
    /// Evaluate positions from binpack to measure eval scale distribution
    EvalDist {
        /// Input binpack file
        #[arg(long, short = 'i')]
        input: String,
        /// Number of positions to evaluate
        #[arg(long, short = 'c', default_value_t = 1_000_000)]
        count: usize,
    },
    /// Show statistics for a binpack file
    BinpackStats {
        /// Input binpack file
        #[arg(long, short = 'i')]
        input: String,
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
        /// Bucketed hidden layers (output buckets baked into L1/L2)
        #[arg(long)]
        bucketed_hidden: bool,
        /// Source output bucket count (default 8, set to 2 for 2-bucket nets)
        #[arg(long, default_value_t = 8)]
        output_buckets: usize,
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

        Some(Commands::Datagen { output, depth, games, threads, hash, blunder, force_captures, epd }) => {
            let nnue_path = cli.nnue.unwrap_or_default();
            let mode = if let Some(epd_path) = epd {
                datagen::DatagenMode::Material { source_epd: epd_path }
            } else {
                datagen::DatagenMode::SelfPlay { blunder_rate: blunder, force_capture_rate: force_captures }
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

        Some(Commands::BinpackStats { input }) => {
            run_binpack_stats(&input);
        }

        Some(Commands::SamplePositions { input, output, count, rate }) => {
            run_sample_positions(&input, &output, count, rate);
        }

        Some(Commands::EvalDist { input, count }) => {
            run_eval_dist(&input, count, &cli.nnue);
        }

        Some(Commands::ConvertBullet { input, output, screlu, pairwise, hidden, hidden2, int8l1, bucketed_hidden, output_buckets }) => {
            let result = if hidden > 0 {
                bullet_convert::convert_v7(&input, &output, screlu, pairwise, hidden, hidden2, int8l1, bucketed_hidden)
            } else {
                bullet_convert::convert_v5(&input, &output, screlu, pairwise, output_buckets)
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
        let piece_count = board.occupied().count_ones();
        acc.force_recompute(&net, &board);
        net.forward(&acc, board.side_to_move, piece_count)
    };

    let test_positions = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -50, 50),
        // White missing a piece — score should be negative (White is worse)
        ("miss pawn", "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1", -200, -20),
        ("miss knight", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1", -500, -80),
        ("miss bishop", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1", -500, -80),
        ("miss rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1", -800, -120),
        ("miss queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", -1500, -200),
        // White up material — score should be positive
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

fn run_binpack_stats(input: &str) {
    use sfbinpack::CompressedTrainingDataEntryReader;

    let file_size = std::fs::metadata(input)
        .unwrap_or_else(|_| panic!("Failed to stat {}", input))
        .len();
    println!("File: {} ({:.2} GB)", input, file_size as f64 / 1_073_741_824.0);

    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let mut total_positions = 0u64;
    let mut score_sum = 0i64;
    let mut score_abs_sum = 0u64;
    let mut score_max = 0i16;
    let mut score_min = 0i16;
    let mut results = [0u64; 3]; // loss/draw/win
    let mut in_check = 0u64;
    let mut score_buckets = [0u64; 10]; // <100, <200, <500, <1000, <2000, <5000, <10000, <20000, <30000, 30000+
    let start = std::time::Instant::now();

    while reader.has_next() {
        let entry = reader.next();
        total_positions += 1;

        let score = entry.score;
        score_sum += score as i64;
        score_abs_sum += score.unsigned_abs() as u64;
        if score > score_max { score_max = score; }
        if score < score_min { score_min = score; }

        match entry.result {
            -1 => results[0] += 1,
            0 => results[1] += 1,
            1 => results[2] += 1,
            _ => {}
        }

        if entry.pos.is_checked(entry.pos.side_to_move()) {
            in_check += 1;
        }

        let abs = score.unsigned_abs();
        let bucket = if abs < 100 { 0 } else if abs < 200 { 1 } else if abs < 500 { 2 }
            else if abs < 1000 { 3 } else if abs < 2000 { 4 } else if abs < 5000 { 5 }
            else if abs < 10000 { 6 } else if abs < 20000 { 7 } else if abs < 30000 { 8 }
            else { 9 };
        score_buckets[bucket] += 1;

        if total_positions % 10_000_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            eprint!("\r{:.0}M positions scanned ({:.0}M/s)...",
                total_positions as f64 / 1e6, total_positions as f64 / elapsed / 1e6);
        }
    }
    eprintln!();

    let elapsed = start.elapsed().as_secs_f64();
    let bytes_per_pos = file_size as f64 / total_positions as f64;
    let avg_score = score_sum as f64 / total_positions as f64;
    let avg_abs_score = score_abs_sum as f64 / total_positions as f64;

    println!("\n=== Binpack Statistics ===");
    println!("Positions:      {:>12} ({:.2}M)", total_positions, total_positions as f64 / 1e6);
    println!("Bytes/position: {:>12.1}", bytes_per_pos);
    println!("Scan time:      {:>12.1}s ({:.1}M pos/s)", elapsed, total_positions as f64 / elapsed / 1e6);
    println!();
    println!("Score range:    {} to {}", score_min, score_max);
    println!("Avg score:      {:.1}", avg_score);
    println!("Avg |score|:    {:.1}", avg_abs_score);
    println!("In check:       {:>12} ({:.1}%)", in_check, in_check as f64 / total_positions as f64 * 100.0);
    println!();
    println!("Results: W {:.1}%  D {:.1}%  L {:.1}%",
        results[2] as f64 / total_positions as f64 * 100.0,
        results[1] as f64 / total_positions as f64 * 100.0,
        results[0] as f64 / total_positions as f64 * 100.0);
    println!();
    println!("Score distribution:");
    let labels = ["<100", "<200", "<500", "<1000", "<2000", "<5000", "<10000", "<20000", "<30000", "30000+"];
    for (i, label) in labels.iter().enumerate() {
        let pct = score_buckets[i] as f64 / total_positions as f64 * 100.0;
        let bar = "#".repeat((pct * 2.0) as usize);
        println!("  {:>8}: {:>10} ({:>5.1}%) {}", label, score_buckets[i], pct, bar);
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

fn run_eval_dist(input: &str, n: usize, nnue_path: &Option<String>) {
    use sfbinpack::CompressedTrainingDataEntryReader;

    // Load NNUE
    let mut info = search::SearchInfo::new(1);
    info.silent = true;
    if let Some(path) = nnue_path {
        info.load_nnue(path).expect("Failed to load NNUE");
    } else if !info.auto_discover_nnue() {
        eprintln!("Error: No NNUE net. Use -n <path>");
        return;
    }

    println!("Evaluating {} positions from {}", n, input);
    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let mut scores: Vec<i32> = Vec::with_capacity(n);
    let mut results: Vec<f64> = Vec::with_capacity(n); // 1.0=white win, 0.5=draw, 0.0=black win
    let mut total = 0usize;
    let mut reader = reader;

    while reader.has_next() && scores.len() < n {
        let entry = reader.next();
        total += 1;

        // Skip checks and extreme scores (same filter as training)
        if entry.pos.is_checked(entry.pos.side_to_move()) { continue; }
        if entry.score.unsigned_abs() > 10000 { continue; }

        // Convert to our Board and evaluate
        let fen = match entry.pos.fen() {
            Ok(f) => f,
            Err(_) => continue,
        };
        let board = board::Board::from_fen(&fen);

        let score = if let (Some(net), Some(acc)) = (&info.nnue_net, &mut info.nnue_acc) {
            eval::evaluate_nnue(&board, net, acc)
        } else {
            continue;
        };

        // Game result from STM perspective: 1.0=stm wins, 0.0=stm loses
        // entry.result: 1=white win, 0=draw, -1=black win
        let stm_is_white = entry.pos.side_to_move() == sfbinpack::chess::color::Color::White;
        let result_f = match entry.result {
            1 => if stm_is_white { 1.0 } else { 0.0 },
            -1 => if stm_is_white { 0.0 } else { 1.0 },
            _ => 0.5, // draw
        };

        scores.push(score);
        results.push(result_f);
        // Also track search score from binpack for comparison
        // (uncomment to use search score instead of static eval for WDL fit)

        if scores.len() % 100_000 == 0 {
            eprint!("\r  {} / {} evaluated ({} scanned)", scores.len(), n, total);
        }
    }
    eprintln!();

    let count = scores.len();
    if count == 0 {
        println!("No positions evaluated");
        return;
    }

    // Debug: result distribution
    let wins = results.iter().filter(|&&r| r > 0.7).count();
    let draws = results.iter().filter(|&&r| r > 0.3 && r < 0.7).count();
    let losses = results.iter().filter(|&&r| r < 0.3).count();
    println!("  Results: {} wins, {} draws, {} losses", wins, draws, losses);

    // Statistics
    let sum: f64 = scores.iter().map(|&s| s as f64).sum();
    let sum_sq: f64 = scores.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let abs_sum: f64 = scores.iter().map(|&s| (s as f64).abs()).sum();
    let mean = sum / count as f64;
    let rms = (sum_sq / count as f64).sqrt();
    let abs_mean = abs_sum / count as f64;

    let mut sorted = scores.clone();
    sorted.sort();

    println!("=== Eval Distribution ({} positions, {} scanned) ===", count, total);
    println!("  Mean:     {:+.1}", mean);
    println!("  Abs mean: {:.1}", abs_mean);
    println!("  RMS:      {:.1}", rms);
    println!("  Ratio to baseline (580): {:.2}x", rms / 580.0);
    println!();
    println!("  Percentiles:");
    println!("    p1:  {:+}", sorted[count / 100]);
    println!("    p5:  {:+}", sorted[count * 5 / 100]);
    println!("    p10: {:+}", sorted[count / 10]);
    println!("    p25: {:+}", sorted[count / 4]);
    println!("    p50: {:+}", sorted[count / 2]);
    println!("    p75: {:+}", sorted[count * 3 / 4]);
    println!("    p90: {:+}", sorted[count * 9 / 10]);
    println!("    p95: {:+}", sorted[count * 95 / 100]);
    println!("    p99: {:+}", sorted[count * 99 / 100]);
    println!();

    // === WDL Logistic Fit (SF approach) ===
    // Fit: win_rate(eval) = 1 / (1 + exp(-eval / K))
    // Find K (NormalizeToPawnValue) where EVAL_SCALE should be set so that
    // 100cp of NNUE output ≈ 50% win probability.
    //
    // Uses our static NNUE eval paired with game outcomes from the binpack.
    // NOTE: For best results, use data generated by the SAME net being evaluated,
    // or T80 data with a net trained on T80. Mismatched eval/results gives poor fit.
    println!("=== WDL Logistic Fit (NNUE eval vs game outcome) ===");

    let mut best_k = 200.0f64;
    let mut best_err = f64::MAX;

    // Binary search over K from 50 to 2000
    for k_int in 50..=2000 {
        let k = k_int as f64;
        let mut err_sum = 0.0f64;
        for i in 0..count {
            let eval = scores[i] as f64;
            let predicted = 1.0 / (1.0 + (-eval / k).exp());
            let actual = results[i];
            let diff = predicted - actual;
            err_sum += diff * diff;
        }
        if err_sum < best_err {
            best_err = err_sum;
            best_k = k;
        }
    }

    let mse = best_err / count as f64;
    println!("  Best K (NormalizeToPawnValue): {:.0}", best_k);
    println!("  MSE: {:.6}", mse);
    println!("  → EVAL_SCALE should be {:.0} so that 100cp ≈ 50% win", best_k);
    println!();

    // Show calibration: predicted win% at key eval values
    println!("  Calibration (eval → predicted win%):");
    for &ev in &[25, 50, 100, 150, 200, 300, 500] {
        let p = 1.0 / (1.0 + (-(ev as f64) / best_k).exp());
        println!("    {:+4}cp → {:.1}% win", ev, p * 100.0);
    }
    println!();

    // Show actual win rates in eval buckets for validation
    println!("  Actual win rates by eval bucket:");
    let buckets: Vec<(i32, i32)> = vec![
        (-500, -200), (-200, -100), (-100, -50), (-50, -25),
        (-25, 0), (0, 25), (25, 50), (50, 100),
        (100, 200), (200, 500),
    ];
    for (lo, hi) in &buckets {
        let mut wins = 0.0f64;
        let mut n_bucket = 0usize;
        for i in 0..count {
            if scores[i] >= *lo && scores[i] < *hi {
                wins += results[i];
                n_bucket += 1;
            }
        }
        if n_bucket > 100 {
            println!("    [{:+4}, {:+4}): {:.1}% win (n={})",
                lo, hi, wins / n_bucket as f64 * 100.0, n_bucket);
        }
    }
}
