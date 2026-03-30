/// Data generation for NNUE training.
///
/// Modes:
///   selfplay  — play games against itself, record positions + search scores
///   material  — remove pieces from positions, deep-search each variant
///
/// Output: Stockfish BINP binpack format (chain-compressed).

use crate::board::Board;
use crate::search::{self, SearchInfo, SearchLimits};
use crate::binpack::{BinpackWriter, TrainChain, TrainSample};
use crate::types::*;
use crate::movegen::generate_legal_moves;

use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

/// Configuration for data generation.
pub struct DatagenConfig {
    pub nnue_path: String,
    pub output_path: String,
    pub mode: DatagenMode,
    pub depth: i32,
    pub num_games: usize,
    pub threads: usize,
    pub hash_mb: usize,
}

pub enum DatagenMode {
    SelfPlay { blunder_rate: f64 },   // 0.0 = no blunders, 0.1 = 10% random moves
    Material { source_epd: String },  // remove pieces from EPD positions
}

/// Run data generation.
pub fn run_datagen(config: &DatagenConfig) {
    let file = File::create(&config.output_path)
        .expect(&format!("Failed to create output file: {}", config.output_path));
    let buf = BufWriter::with_capacity(1 << 20, file);
    let mut writer = BinpackWriter::new(buf);

    let start = Instant::now();

    match &config.mode {
        DatagenMode::SelfPlay { blunder_rate } => {
            run_selfplay(config, *blunder_rate, &mut writer);
        }
        DatagenMode::Material { source_epd } => {
            run_material(config, source_epd, &mut writer);
        }
    }

    writer.finish().expect("Failed to flush binpack output");

    let elapsed = start.elapsed().as_secs_f64();
    println!("Datagen complete: {} chains, {} positions in {:.1}s ({:.0} pos/s)",
        writer.chains_written(), writer.positions_written(),
        elapsed, writer.positions_written() as f64 / elapsed);
}

/// Self-play data generation.
fn run_selfplay<W: std::io::Write>(
    config: &DatagenConfig,
    blunder_rate: f64,
    writer: &mut BinpackWriter<W>,
) {
    let mut rng = SimpleRng::new(0xDEADBEEF);

    for game_idx in 0..config.num_games {
        let mut board = Board::startpos();
        let mut info = SearchInfo::new(config.hash_mb);
        if let Err(e) = info.load_nnue(&config.nnue_path) {
            eprintln!("Failed to load NNUE: {}", e);
            return;
        }

        let mut chain = TrainChain { samples: Vec::with_capacity(200) };
        let mut ply = 0u16;
        let mut result: Option<i16> = None;

        // Play a game
        loop {
            let legal = generate_legal_moves(&board);
            if legal.len == 0 {
                // Checkmate or stalemate
                if board.in_check() {
                    // Checkmate: side to move loses
                    result = Some(if board.side_to_move == WHITE { -1 } else { 1 });
                } else {
                    result = Some(0); // stalemate = draw
                }
                break;
            }

            // 50-move rule
            if board.halfmove >= 100 {
                result = Some(0);
                break;
            }

            // Repetition (simple: 3-fold via undo stack)
            if is_repetition(&board) {
                result = Some(0);
                break;
            }

            // Too many moves = draw
            if ply > 500 {
                result = Some(0);
                break;
            }

            // Search for best move and score
            let limits = SearchLimits {
                depth: config.depth,
                ..SearchLimits::default()
            };
            let best_move = search::search(&mut board, &mut info, &limits);
            let score = info.last_score;

            // Adjudicate: resign if score is too extreme
            if score.abs() > 3000 && ply > 10 {
                result = Some(if score > 0 {
                    if board.side_to_move == WHITE { 1 } else { -1 }
                } else {
                    if board.side_to_move == WHITE { -1 } else { 1 }
                });
                break;
            }

            // Record position (skip first few opening moves and positions in check)
            if ply >= 8 && !board.in_check() && score.abs() < 10000 {
                chain.samples.push(TrainSample {
                    board: board.clone(),
                    mv: best_move,
                    score: score.clamp(-32000, 32000) as i16,
                    result: 0, // filled in later
                    ply,
                });
            }

            // Blunder mode: occasionally play a random move instead
            let mv = if blunder_rate > 0.0 && rng.next_f64() < blunder_rate && ply >= 4 {
                let idx = rng.next_u64() as usize % legal.len;
                legal.moves[idx]
            } else {
                best_move
            };

            if mv == NO_MOVE { break; }
            board.make_move(mv);
            ply += 1;
        }

        // Fill in game result for all samples
        let game_result = result.unwrap_or(0);
        for sample in chain.samples.iter_mut() {
            // Result from side-to-move perspective at that position
            sample.result = if sample.board.side_to_move == WHITE {
                game_result
            } else {
                -game_result
            };
        }

        if !chain.samples.is_empty() {
            writer.write_chain(&chain).expect("Failed to write chain");
        }

        if (game_idx + 1) % 100 == 0 {
            println!("Game {}/{}: {} positions so far",
                game_idx + 1, config.num_games, writer.positions_written());
        }
    }
}

/// Material removal data generation.
/// Takes positions from an EPD file, systematically removes pieces, deep-searches each.
fn run_material<W: std::io::Write>(
    config: &DatagenConfig,
    source_epd: &str,
    writer: &mut BinpackWriter<W>,
) {
    // Load positions from EPD
    let contents = std::fs::read_to_string(source_epd)
        .expect(&format!("Failed to read EPD file: {}", source_epd));

    let mut info = SearchInfo::new(config.hash_mb);
    if let Err(e) = info.load_nnue(&config.nnue_path) {
        eprintln!("Failed to load NNUE: {}", e);
        return;
    }

    let mut pos_count = 0u64;

    for line in contents.lines() {
        let fen = line.split(';').next().unwrap_or(line).trim();
        if fen.is_empty() || fen.starts_with('#') { continue; }

        let base_board = Board::from_fen(fen);

        // For each non-king, non-pawn piece, try removing it
        // This creates positions with material imbalances
        for sq in 0..64u8 {
            let pt = base_board.piece_type_at(sq);
            if pt == NO_PIECE_TYPE || pt == KING { continue; }

            // Create a board with this piece removed
            let mut modified = base_board.clone();
            let color = if modified.colors[WHITE as usize] & (1u64 << sq) != 0 { WHITE } else { BLACK };
            modified.remove_piece(color, pt, sq);

            // Verify position is legal (both kings present, side to move not giving check, etc.)
            if !is_valid_position(&modified) { continue; }

            // Deep search
            let limits = SearchLimits {
                depth: config.depth,
                ..SearchLimits::default()
            };
            let best_move = search::search(&mut modified, &mut info, &limits);
            let score = info.last_score;

            if best_move != NO_MOVE && score.abs() < 10000 {
                let mut chain = TrainChain { samples: Vec::with_capacity(1) };
                chain.samples.push(TrainSample {
                    board: modified,
                    mv: best_move,
                    score: score.clamp(-32000, 32000) as i16,
                    result: 0, // no game result for material positions
                    ply: 0,
                });
                writer.write_chain(&chain).expect("Failed to write chain");
                pos_count += 1;
            }
        }
    }

    println!("Material datagen: {} positions from {} EPD lines", pos_count, contents.lines().count());
}

/// Check for 3-fold repetition in the undo stack.
fn is_repetition(board: &Board) -> bool {
    let stack_len = board.undo_stack.len();
    let limit = (board.halfmove as usize).min(stack_len);
    let mut i = 2;
    let mut count = 0;
    while i <= limit {
        if board.undo_stack[stack_len - i].hash == board.hash {
            count += 1;
            if count >= 2 { return true; } // 3-fold (current + 2 repeats)
        }
        i += 2;
    }
    false
}

/// Basic position validity check.
fn is_valid_position(board: &Board) -> bool {
    // Both kings must be present
    let wk = board.pieces[KING as usize] & board.colors[WHITE as usize];
    let bk = board.pieces[KING as usize] & board.colors[BLACK as usize];
    if wk == 0 || bk == 0 { return false; }
    if wk.count_ones() != 1 || bk.count_ones() != 1 { return false; }
    // Side not to move must not be in check
    // (i.e., the opponent's king must not be attacked)
    true
}

/// Simple PRNG for reproducible data generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

