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

        "help" | "--help" | "-h" => {
            print_usage();
        }

        _ => {
            // Validate any dash-prefixed args are known flags
            let known_flags = ["-nnue", "-book", "-h", "--help"];
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

            if subcmd.starts_with('-') || subcmd.is_empty() {
                uci::uci_loop_with_nnue(nnue_path, book_path);
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
    println!("  coda                              UCI mode (default)");
    println!("  coda -nnue <net.nnue>             UCI with NNUE evaluation");
    println!("  coda -book <book.bin>             UCI with Polyglot opening book");
    println!("  coda bench [depth] [-nnue <net>]  Search benchmark");
    println!("  coda epd <file> [time] [max] [-nnue <net>]");
    println!("                                    Run EPD test suite");
    println!("  coda perft [depth] [fen...]       Perft with divide");
    println!("  coda perft-bench                  Perft benchmark suite");
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
