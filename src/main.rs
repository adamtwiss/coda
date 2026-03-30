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

use board::Board;
use movegen::{perft, perft_divide};

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        attacks::init_attacks();
        bitboard::init_bitboards();
        zobrist::init_zobrist();
        board::init_castle_masks();
        search::init_lmr();
        nnue::init_nnue();
    });
}

/// Extract a flag value: -flag <value>
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|s| s == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() {
    init();

    let args: Vec<String> = std::env::args().collect();
    let subcmd = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match subcmd {
        "perft" => {
            let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let fen = if args.len() > 3 {
                args[3..].join(" ")
            } else {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()
            };

            let mut board = Board::from_fen(&fen);
            println!("Position: {}", fen);
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

        "perft-bench" => {
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
                    name, depth, nodes,
                    if passed { "OK" } else { "FAIL" },
                    expected);
            }

            let elapsed = start.elapsed();
            let nps = (total_nodes as f64 / elapsed.as_secs_f64()) as u64;
            println!("\nTotal: {} nodes in {:.3}s = {} NPS",
                total_nodes, elapsed.as_secs_f64(), nps);
            if all_passed {
                println!("All perft tests PASSED");
            } else {
                println!("Some perft tests FAILED");
                std::process::exit(1);
            }
        }

        "bench" => {
            let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(13);
            let nnue_path = flag_value(&args, "-nnue");
            let start = std::time::Instant::now();
            let nodes = search::bench(depth, nnue_path);
            let elapsed = start.elapsed();
            let nps = if elapsed.as_secs_f64() > 0.0 {
                (nodes as f64 / elapsed.as_secs_f64()) as u64
            } else { 0 };
            println!("\n{} nodes {} nps", nodes, nps);
        }

        "epd" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("testdata/wac.epd");
            let time: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5000);
            let max: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let nnue_path = flag_value(&args, "-nnue");
            epd::run_epd(path, time, max, nnue_path);
        }

        "datagen" => {
            let nnue_path = flag_value(&args, "-nnue").unwrap_or("").to_string();
            let output = flag_value(&args, "-output").unwrap_or("data.binpack").to_string();
            let depth: i32 = flag_value(&args, "-depth").and_then(|s| s.parse().ok()).unwrap_or(8);
            let games: usize = flag_value(&args, "-games").and_then(|s| s.parse().ok()).unwrap_or(1000);
            let hash: usize = flag_value(&args, "-hash").and_then(|s| s.parse().ok()).unwrap_or(16);
            let threads: usize = flag_value(&args, "-threads").and_then(|s| s.parse().ok()).unwrap_or(1);
            let blunder: f64 = flag_value(&args, "-blunder").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let epd_path = flag_value(&args, "-epd");

            let mode = if let Some(epd) = epd_path {
                datagen::DatagenMode::Material { source_epd: epd.to_string() }
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

        "convert-checkpoint" => {
            let nnue_path = flag_value(&args, "-nnue")
                .expect("Usage: coda convert-checkpoint -nnue <net.nnue> -output <dir> [-ft N] [-l1 N] [-l2 N]");
            let output = flag_value(&args, "-output").unwrap_or("v7_checkpoint");
            let ft: usize = flag_value(&args, "-ft").and_then(|s| s.parse().ok()).unwrap_or(1024);
            let l1: usize = flag_value(&args, "-l1").and_then(|s| s.parse().ok()).unwrap_or(16);
            let l2: usize = flag_value(&args, "-l2").and_then(|s| s.parse().ok()).unwrap_or(32);
            if let Err(e) = nnue_export::nnue_to_bullet_checkpoint(nnue_path, output, ft, l1, l2) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        "convert-bullet" => {
            let input = flag_value(&args, "-input")
                .expect("Usage: coda convert-bullet -input <quantised.bin> -output <net.nnue> [options]");
            let output = flag_value(&args, "-output").unwrap_or("net.nnue");
            let screlu = args.iter().any(|a| a == "-screlu");
            let pairwise = args.iter().any(|a| a == "-pairwise");
            let l1: usize = flag_value(&args, "-hidden").and_then(|s| s.parse().ok()).unwrap_or(0);
            let l2: usize = flag_value(&args, "-hidden2").and_then(|s| s.parse().ok()).unwrap_or(0);
            let int8l1 = args.iter().any(|a| a == "-int8l1");

            let result = if l1 > 0 {
                bullet_convert::convert_v7(input, output, screlu, pairwise, l1, l2, int8l1)
            } else {
                bullet_convert::convert_v5(input, output, screlu, pairwise)
            };
            if let Err(e) = result {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        "check-net" => {
            let net_path = args.get(2).expect("Usage: coda check-net <net.nnue>");
            // TODO: port check-net from GoChess tuner
            println!("check-net not yet implemented — use GoChess tuner for now");
            println!("  ./tuner check-net {}", net_path);
        }

        "help" | "--help" | "-h" => {
            print_usage();
        }

        _ => {
            // Validate any dash-prefixed args are known flags
            let known_flags = ["-nnue", "-book", "-classical", "-h", "--help"];
            for arg in &args[1..] {
                if arg.starts_with('-') && !known_flags.contains(&arg.as_str()) {
                    eprintln!("Unknown option: {}", arg);
                    print_usage();
                    std::process::exit(1);
                }
            }

            // UCI mode (default)
            let nnue_path = flag_value(&args, "-nnue");
            let book_path = flag_value(&args, "-book");
            let classical = args.iter().any(|a| a == "-classical");

            if subcmd.starts_with('-') || subcmd.is_empty() {
                uci::uci_loop_with_nnue(nnue_path, book_path, classical);
            } else {
                eprintln!("Unknown command: {}", subcmd);
                print_usage();
                std::process::exit(1);
            }
        }
    }
}

fn print_usage() {
    println!("Coda Chess Engine — Chess Optimised, Developed Agentically");
    println!();
    println!("Usage:");
    println!("  coda                              UCI mode (auto-discovers NNUE from net.txt)");
    println!("  coda -nnue <net.nnue>             UCI with specific NNUE network");
    println!("  coda -classical                   UCI with PeSTO eval (no NNUE required)");
    println!("  coda -book <book.bin>             UCI with Polyglot opening book");
    println!("  coda bench [depth] [-nnue <net>]  Search benchmark");
    println!("  coda epd <file> [time] [max] [-nnue <net>]");
    println!("                                    Run EPD test suite");
    println!("  coda perft [depth] [fen...]       Perft with divide");
    println!("  coda perft-bench                  Perft benchmark suite");
    println!("  coda datagen -nnue <net> -output <file.binpack> [options]");
    println!("                                    Generate training data (SF binpack format)");
    println!("  coda convert-bullet -input <quantised.bin> -output <net.nnue> [options]");
    println!("                                    Convert Bullet quantised.bin to .nnue");
    println!("    -screlu                         SCReLU activation");
    println!("    -pairwise                       Pairwise multiplication");
    println!("    -hidden <N>                     L1 hidden layer (v7)");
    println!("    -hidden2 <N>                    L2 hidden layer (v7)");
    println!("    -int8l1                         L1 weights are int8 (QA=64)");
    println!("  coda convert-checkpoint -nnue <v5.nnue> -output <dir> [-ft 1024] [-l1 16] [-l2 32]");
    println!("                                    Convert .nnue to Bullet checkpoint for v7 transfer learning");
    println!("  coda check-net <net.nnue>         NNUE health check (TODO)");
    println!("    -depth <N>                      Search depth per position (default 8)");
    println!("    -games <N>                      Number of self-play games (default 1000)");
    println!("    -threads <N>                    Worker threads (default 1)");
    println!("    -hash <MB>                      Hash table size per thread (default 16)");
    println!("    -blunder <rate>                 Random move rate 0.0-1.0 (default 0.0)");
    println!("    -epd <file>                     Material mode: remove pieces from EPD positions");
    println!("  coda help                         Show this help");
    println!();
    println!("UCI options:");
    println!("  Hash          (spin, 1-4096, default 64)");
    println!("  NNUEFile      (string)  Path to .nnue network file");
    println!("  OwnBook       (check)   Use opening book");
    println!("  BookFile      (string)  Path to Polyglot .bin book");
    println!("  MoveOverhead  (spin, 0-5000, default 100)");
    println!("  Ponder        (check)   Enable pondering");
}
