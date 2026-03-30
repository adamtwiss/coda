/// Data generation for NNUE training.
///
/// Modes:
///   selfplay  — play games against itself, record positions + search scores
///   material  — remove pieces from positions, deep-search each variant
///
/// Output: Stockfish BINP binpack format via sfbinpack crate.
/// Multi-threaded: each worker plays games independently, sends entries
/// to writer thread via channel.

use crate::board::Board;
use crate::search::{self, SearchInfo, SearchLimits};
use crate::types::*;
use crate::movegen::generate_legal_moves;

use sfbinpack::chess::position::Position as SfPosition;
use sfbinpack::chess::r#move::{Move as SfMove, MoveType as SfMoveType};
use sfbinpack::chess::coords::Square as SfSquare;
use sfbinpack::chess::piece::Piece as SfPiece;
use sfbinpack::chess::color::Color as SfColor;
use sfbinpack::chess::piecetype::PieceType as SfPieceType;
use sfbinpack::TrainingDataEntry;
use sfbinpack::CompressedTrainingDataEntryWriter;

use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    SelfPlay { blunder_rate: f64 },
    Material { source_epd: String },
}

/// Convert Coda board + move + score to sfbinpack TrainingDataEntry.
fn to_sf_entry(board: &Board, mv: Move, score: i16, result: i16, ply: u16) -> Option<TrainingDataEntry> {
    let fen = board.to_fen();
    let pos = SfPosition::from_fen(&fen).ok()?;

    let from = SfSquare::new(move_from(mv) as u32);
    let to = SfSquare::new(move_to(mv) as u32);
    let flags = move_flags(mv);

    let sf_move = if flags == FLAG_EN_PASSANT {
        SfMove::new(from, to, SfMoveType::EnPassant, SfPiece::none())
    } else if flags == FLAG_CASTLE {
        // sfbinpack expects king→rook (not king→destination)
        let rook_sq = match move_to(mv) {
            6 => SfSquare::new(7),   // g1 → h1
            2 => SfSquare::new(0),   // c1 → a1
            62 => SfSquare::new(63), // g8 → h8
            58 => SfSquare::new(56), // c8 → a8
            _ => to,
        };
        SfMove::new(from, rook_sq, SfMoveType::Castle, SfPiece::none())
    } else if is_promotion(mv) {
        let sf_color = if board.side_to_move == WHITE { SfColor::White } else { SfColor::Black };
        let promo_piece = match promotion_piece_type(mv) {
            KNIGHT => SfPiece::new(SfPieceType::Knight, sf_color),
            BISHOP => SfPiece::new(SfPieceType::Bishop, sf_color),
            ROOK => SfPiece::new(SfPieceType::Rook, sf_color),
            _ => SfPiece::new(SfPieceType::Queen, sf_color),
        };
        SfMove::new(from, to, SfMoveType::Promotion, promo_piece)
    } else {
        SfMove::new(from, to, SfMoveType::Normal, SfPiece::none())
    };

    Some(TrainingDataEntry {
        pos,
        mv: sf_move,
        score,
        ply,
        result,
    })
}

/// Run data generation.
pub fn run_datagen(config: &DatagenConfig) {
    let start = Instant::now();
    let threads = config.threads.max(1);

    let games_done = Arc::new(AtomicU64::new(0));
    let positions_done = Arc::new(AtomicU64::new(0));

    let (tx, rx) = std::sync::mpsc::channel::<Vec<TrainingDataEntry>>();

    // Writer thread
    let output_path = config.output_path.clone();
    let positions_done_w = positions_done.clone();
    let writer_handle = std::thread::spawn(move || {
        let file = File::create(&output_path)
            .expect(&format!("Failed to create output file: {}", output_path));
        let buf = BufWriter::with_capacity(1 << 20, file);
        let mut writer = CompressedTrainingDataEntryWriter::new(buf)
            .expect("Failed to create binpack writer");
        let mut total = 0u64;

        for entries in rx {
            for entry in &entries {
                writer.write_entry(entry).expect("Failed to write entry");
                total += 1;
            }
            positions_done_w.fetch_add(entries.len() as u64, Ordering::Relaxed);
        }
        // writer drops here, flushing automatically
        total
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
                let my_games = games_per_thread + if thread_id < remainder { 1 } else { 0 };

                handles.push(std::thread::Builder::new()
                    .stack_size(16 * 1024 * 1024)
                    .spawn(move || {
                    selfplay_worker(thread_id, my_games, &nnue_path, depth, hash_mb,
                        blunder_rate, &tx, &games_done);
                }).expect("Failed to spawn worker thread"));
            }
            drop(tx);

            // Progress reporting
            let total = config.num_games as u64;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let g = games_done.load(Ordering::Relaxed);
                let p = positions_done.load(Ordering::Relaxed);
                let elapsed = start.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    println!("Progress: {}/{} games, {} positions ({:.0} pos/s)",
                        g, total, p, p as f64 / elapsed);
                }
                if g >= total { break; }
            }
            for h in handles { h.join().expect("Worker panicked"); }
        }
        DatagenMode::Material { source_epd } => {
            let contents = std::fs::read_to_string(source_epd)
                .expect(&format!("Failed to read EPD: {}", source_epd));
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
                    material_worker(thread_id, &lines, &line_idx, &nnue_path, depth, hash_mb, &tx);
                }).expect("Failed to spawn worker thread"));
            }
            drop(tx);
            for h in handles { h.join().expect("Worker panicked"); }
        }
    }

    let total = writer_handle.join().expect("Writer panicked");
    let elapsed = start.elapsed().as_secs_f64();
    println!("Datagen complete: {} positions in {:.1}s ({:.0} pos/s) [{} threads]",
        total, elapsed, total as f64 / elapsed, threads);
}

fn selfplay_worker(
    thread_id: usize, my_games: usize, nnue_path: &str,
    depth: i32, hash_mb: usize, blunder_rate: f64,
    tx: &std::sync::mpsc::Sender<Vec<TrainingDataEntry>>,
    games_done: &AtomicU64,
) {
    let mut rng = SimpleRng::new(0xDEADBEEF ^ (thread_id as u64 * 0x9E3779B97F4A7C15));
    let mut info = SearchInfo::new(hash_mb);
    info.silent = true;
    if let Err(e) = info.load_nnue(nnue_path) {
        eprintln!("Thread {}: Failed to load NNUE: {}", thread_id, e);
        return;
    }

    for _ in 0..my_games {
        let entries = play_one_game(&mut info, &mut rng, depth, blunder_rate);
        if !entries.is_empty() {
            if tx.send(entries).is_err() { break; }
        }
        info.tt.clear();
        games_done.fetch_add(1, Ordering::Relaxed);
    }
}

fn play_one_game(info: &mut SearchInfo, rng: &mut SimpleRng, depth: i32, blunder_rate: f64)
    -> Vec<TrainingDataEntry>
{
    let mut board = Board::startpos();
    let mut entries: Vec<(Board, Move, i16, u16)> = Vec::with_capacity(200);
    let mut ply = 0u16;
    let mut result: Option<i16> = None;

    // Random opening for diversity
    let random_plies = 4 + (rng.next_u64() % 6) as u16;
    for _ in 0..random_plies {
        let legal = generate_legal_moves(&board);
        if legal.len == 0 { return Vec::new(); }
        let idx = rng.next_u64() as usize % legal.len;
        board.make_move(legal.moves[idx]);
        ply += 1;
    }

    loop {
        let legal = generate_legal_moves(&board);
        if legal.len == 0 {
            result = Some(if board.in_check() {
                if board.side_to_move == WHITE { -1 } else { 1 }
            } else { 0 });
            break;
        }
        if board.halfmove >= 100 { result = Some(0); break; }
        if is_repetition(&board) { result = Some(0); break; }
        if ply > 500 { result = Some(0); break; }

        let limits = SearchLimits { depth, ..SearchLimits::default() };
        let best_move = search::search(&mut board, info, &limits);
        let score = info.last_score;

        if score.abs() > 3000 && ply > 10 {
            result = Some(if score > 0 {
                if board.side_to_move == WHITE { 1 } else { -1 }
            } else {
                if board.side_to_move == WHITE { -1 } else { 1 }
            });
            break;
        }

        if ply >= 8 && !board.in_check() && score.abs() < 10000 {
            entries.push((board.clone(), best_move, score.clamp(-32000, 32000) as i16, ply));
        }

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

    let game_result = result.unwrap_or(0);
    let mut sf_entries = Vec::with_capacity(entries.len());
    for (board, mv, score, ply) in &entries {
        let r = if board.side_to_move == WHITE { game_result } else { -game_result };
        if let Some(entry) = to_sf_entry(&board, *mv, *score, r, *ply) {
            sf_entries.push(entry);
        }
    }
    sf_entries
}

fn material_worker(
    thread_id: usize, lines: &[String], line_idx: &AtomicU64,
    nnue_path: &str, depth: i32, hash_mb: usize,
    tx: &std::sync::mpsc::Sender<Vec<TrainingDataEntry>>,
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
        let mut batch = Vec::new();

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
                if let Some(entry) = to_sf_entry(&modified, best_move,
                    score.clamp(-32000, 32000) as i16, 0, 0)
                {
                    batch.push(entry);
                }
            }
        }

        if !batch.is_empty() {
            if tx.send(batch).is_err() { return; }
        }
    }
}

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

fn is_valid_position(board: &Board) -> bool {
    let wk = board.pieces[KING as usize] & board.colors[WHITE as usize];
    let bk = board.pieces[KING as usize] & board.colors[BLACK as usize];
    wk != 0 && bk != 0 && wk.count_ones() == 1 && bk.count_ones() == 1
}

struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { SimpleRng { state: seed.max(1) } }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
}
