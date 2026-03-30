/// Data generation for NNUE training.
///
/// Modes:
///   selfplay  — play games against itself, record positions + search scores
///   material  — remove pieces from positions, deep-search each variant
///
/// Output: Stockfish BINP binpack format (chain-compressed).
/// Multi-threaded: each worker thread plays games independently, sends
/// completed chains to the writer thread via channel.

use crate::board::Board;
use crate::search::{self, SearchInfo, SearchLimits};
use crate::binpack::{BinpackWriter, TrainChain, TrainSample};
use crate::types::*;
use crate::movegen::generate_legal_moves;

use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
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
    let start = Instant::now();
    let threads = config.threads.max(1);

    // Shared counters
    let games_done = Arc::new(AtomicU64::new(0));
    let positions_done = Arc::new(AtomicU64::new(0));

    // Channel: workers send chains, writer thread receives
    let (tx, rx) = std::sync::mpsc::channel::<TrainChain>();

    // Writer thread
    let output_path = config.output_path.clone();
    let positions_done_w = positions_done.clone();
    let writer_handle = std::thread::spawn(move || {
        let file = File::create(&output_path)
            .expect(&format!("Failed to create output file: {}", output_path));
        let buf = BufWriter::with_capacity(1 << 20, file);
        let mut writer = BinpackWriter::new(buf);

        for chain in rx {
            let n = chain.samples.len() as u64;
            writer.write_chain(&chain).expect("Failed to write chain");
            positions_done_w.fetch_add(n, Ordering::Relaxed);
        }

        writer.finish().expect("Failed to flush binpack output");
        (writer.chains_written(), writer.positions_written())
    });

    match &config.mode {
        DatagenMode::SelfPlay { blunder_rate } => {
            let blunder_rate = *blunder_rate;
            let games_per_thread = config.num_games / threads;
            let remainder = config.num_games % threads;

            let mut handles = Vec::new();
            for thread_id in 0..threads {
                let tx = tx.clone();
                let games_done = games_done.clone();
                let nnue_path = config.nnue_path.clone();
                let depth = config.depth;
                let hash_mb = config.hash_mb;
                let total_games = config.num_games;
                let my_games = games_per_thread + if thread_id < remainder { 1 } else { 0 };

                handles.push(std::thread::Builder::new()
                    .stack_size(16 * 1024 * 1024) // 16MB stack for search + NNUE
                    .spawn(move || {
                    selfplay_worker(
                        thread_id, my_games, total_games,
                        &nnue_path, depth, hash_mb, blunder_rate,
                        &tx, &games_done,
                    );
                }).expect("Failed to spawn worker thread"));
            }

            // Drop our copy of tx so the channel closes when all workers finish
            drop(tx);

            // Progress reporting
            let total = config.num_games as u64;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let g = games_done.load(Ordering::Relaxed);
                let p = positions_done.load(Ordering::Relaxed);
                let elapsed = start.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    println!("Progress: {}/{} games, {} positions ({:.0} pos/s, {:.0} games/s)",
                        g, total, p, p as f64 / elapsed, g as f64 / elapsed);
                }
                if g >= total { break; }
            }

            for h in handles {
                h.join().expect("Worker thread panicked");
            }
        }
        DatagenMode::Material { source_epd } => {
            let contents = std::fs::read_to_string(source_epd)
                .expect(&format!("Failed to read EPD file: {}", source_epd));
            let lines: Vec<String> = contents.lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
                .map(|l| l.split(';').next().unwrap_or(l).trim().to_string())
                .collect();

            let lines = Arc::new(lines);
            let line_idx = Arc::new(AtomicU64::new(0));

            let mut handles = Vec::new();
            for thread_id in 0..threads {
                let tx = tx.clone();
                let lines = lines.clone();
                let line_idx = line_idx.clone();
                let nnue_path = config.nnue_path.clone();
                let depth = config.depth;
                let hash_mb = config.hash_mb;

                handles.push(std::thread::Builder::new()
                    .stack_size(16 * 1024 * 1024)
                    .spawn(move || {
                    material_worker(
                        thread_id, &lines, &line_idx,
                        &nnue_path, depth, hash_mb, &tx,
                    );
                }).expect("Failed to spawn worker thread"));
            }

            drop(tx);

            for h in handles {
                h.join().expect("Worker thread panicked");
            }
        }
    }

    let (chains, positions) = writer_handle.join().expect("Writer thread panicked");
    let elapsed = start.elapsed().as_secs_f64();
    println!("Datagen complete: {} chains, {} positions in {:.1}s ({:.0} pos/s) [{} threads]",
        chains, positions, elapsed, positions as f64 / elapsed, threads);
}

/// Self-play worker thread. Plays games and sends chains via channel.
fn selfplay_worker(
    thread_id: usize,
    my_games: usize,
    _total_games: usize,
    nnue_path: &str,
    depth: i32,
    hash_mb: usize,
    blunder_rate: f64,
    tx: &std::sync::mpsc::Sender<TrainChain>,
    games_done: &AtomicU64,
) {
    let mut rng = SimpleRng::new(0xDEADBEEF ^ (thread_id as u64 * 0x9E3779B97F4A7C15));

    // Each thread gets its own SearchInfo (own TT, NNUE accumulator)
    let mut info = SearchInfo::new(hash_mb);
    info.silent = true;
    if let Err(e) = info.load_nnue(nnue_path) {
        eprintln!("Thread {}: Failed to load NNUE: {}", thread_id, e);
        return;
    }

    for _ in 0..my_games {
        let chain = play_one_game(&mut info, &mut rng, depth, blunder_rate);

        if !chain.samples.is_empty() {
            if tx.send(chain).is_err() {
                break; // writer closed
            }
        }

        // Reset TT between games for data diversity
        info.tt.clear();
        games_done.fetch_add(1, Ordering::Relaxed);
    }
}

/// Play a single self-play game, returning the chain of training samples.
fn play_one_game(
    info: &mut SearchInfo,
    rng: &mut SimpleRng,
    depth: i32,
    blunder_rate: f64,
) -> TrainChain {
    let mut board = Board::startpos();
    let mut chain = TrainChain { samples: Vec::with_capacity(200) };
    let mut ply = 0u16;
    let mut result: Option<i16> = None;

    // Play a few random opening moves for diversity
    let random_opening_plies = 4 + (rng.next_u64() % 6) as u16; // 4-9 random moves
    for _ in 0..random_opening_plies {
        let legal = generate_legal_moves(&board);
        if legal.len == 0 { return chain; }
        let idx = rng.next_u64() as usize % legal.len;
        board.make_move(legal.moves[idx]);
        ply += 1;
    }

    loop {
        let legal = generate_legal_moves(&board);
        if legal.len == 0 {
            if board.in_check() {
                result = Some(if board.side_to_move == WHITE { -1 } else { 1 });
            } else {
                result = Some(0);
            }
            break;
        }

        if board.halfmove >= 100 { result = Some(0); break; }
        if is_repetition(&board) { result = Some(0); break; }
        if ply > 500 { result = Some(0); break; }

        let limits = SearchLimits { depth, ..SearchLimits::default() };
        let best_move = search::search(&mut board, info, &limits);
        let score = info.last_score;

        // Adjudicate
        if score.abs() > 3000 && ply > 10 {
            result = Some(if score > 0 {
                if board.side_to_move == WHITE { 1 } else { -1 }
            } else {
                if board.side_to_move == WHITE { -1 } else { 1 }
            });
            break;
        }

        // Record position (skip positions in check, extreme scores, early plies)
        if ply >= 8 && !board.in_check() && score.abs() < 10000 {
            chain.samples.push(TrainSample {
                board: board.clone(),
                mv: best_move,
                score: score.clamp(-32000, 32000) as i16,
                result: 0, // filled in later
                ply,
            });
        }

        // Blunder mode
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

    // Fill in game result
    let game_result = result.unwrap_or(0);
    for sample in chain.samples.iter_mut() {
        sample.result = if sample.board.side_to_move == WHITE {
            game_result
        } else {
            -game_result
        };
    }

    chain
}

/// Material removal worker thread.
fn material_worker(
    thread_id: usize,
    lines: &[String],
    line_idx: &AtomicU64,
    nnue_path: &str,
    depth: i32,
    hash_mb: usize,
    tx: &std::sync::mpsc::Sender<TrainChain>,
) {
    let mut info = SearchInfo::new(hash_mb);
    info.silent = true;
    if let Err(e) = info.load_nnue(nnue_path) {
        eprintln!("Thread {}: Failed to load NNUE: {}", thread_id, e);
        return;
    }

    loop {
        let idx = line_idx.fetch_add(1, Ordering::Relaxed) as usize;
        if idx >= lines.len() { break; }

        let base_board = Board::from_fen(&lines[idx]);

        // For each non-king piece, remove it and deep-search
        for sq in 0..64u8 {
            let pt = base_board.piece_type_at(sq);
            if pt == NO_PIECE_TYPE || pt == KING { continue; }

            let mut modified = base_board.clone();
            let color = if modified.colors[WHITE as usize] & (1u64 << sq) != 0 { WHITE } else { BLACK };
            modified.remove_piece(color, pt, sq);

            if !is_valid_position(&modified) { continue; }

            let limits = SearchLimits { depth, ..SearchLimits::default() };
            let best_move = search::search(&mut modified, &mut info, &limits);
            let score = info.last_score;

            if best_move != NO_MOVE && score.abs() < 10000 {
                let chain = TrainChain {
                    samples: vec![TrainSample {
                        board: modified,
                        mv: best_move,
                        score: score.clamp(-32000, 32000) as i16,
                        result: 0,
                        ply: 0,
                    }],
                };
                if tx.send(chain).is_err() { return; }
            }
        }
    }
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
            if count >= 2 { return true; }
        }
        i += 2;
    }
    false
}

/// Basic position validity check.
fn is_valid_position(board: &Board) -> bool {
    let wk = board.pieces[KING as usize] & board.colors[WHITE as usize];
    let bk = board.pieces[KING as usize] & board.colors[BLACK as usize];
    if wk == 0 || bk == 0 { return false; }
    if wk.count_ones() != 1 || bk.count_ones() != 1 { return false; }
    true
}

/// Simple xorshift PRNG for reproducible data generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed.max(1) }
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
