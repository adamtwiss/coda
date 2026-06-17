//! Data generation for NNUE training.
//!
//! Modes:
//!   selfplay  — play games against itself, record positions + search scores
//!   material  — remove pieces from positions, deep-search each variant
//!
//! Output: Stockfish BINP binpack format via sfbinpack crate.
//! Multi-threaded: each worker plays games independently, sends entries
//! to writer thread via channel.

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
    SelfPlay { blunder_rate: f64, force_capture_rate: f64 },
    Material { source_epd: String },
}

/// Convert Coda board + move + score to sfbinpack entry (standalone, no chaining).
/// Used by material_worker where positions aren't sequential.
fn to_sf_entry(board: &Board, mv: Move, score: i16, result: i16, ply: u16) -> Option<TrainingDataEntry> {
    let pos = SfPosition::from_fen(&board.to_fen()).ok()?;
    let sf_move = make_sf_move(&pos, mv, board)?;
    Some(TrainingDataEntry { pos, mv: sf_move, score, ply, result })
}

/// Import a TSV of (fen, score) rows into an sfbinpack binpack for fine-tuning.
/// `score` is STM-POV cp (the SF deep eval = the corrective target). The WDL
/// `result` is DERIVED FROM THE SCORE (STM-POV: win if >200, loss if <-200,
/// else draw) because harvested-loss corpora have a uniform/dead `res` column.
/// Positions with |score| > `max_abs` are dropped (mate-ish — not the subtle
/// over-valuation signal). `repeat` upsamples each kept position (for a small
/// targeted fine-tune set). Standalone entries (no chaining).
pub fn import_tsv(tsv_path: &str, out_path: &str, fen_col: usize, score_col: usize,
                  max_abs: i32, repeat: usize) {
    use std::io::{BufRead, BufReader, BufWriter};
    let f = std::fs::File::open(tsv_path)
        .unwrap_or_else(|e| panic!("open {}: {}", tsv_path, e));
    let outf = std::fs::File::create(out_path)
        .unwrap_or_else(|e| panic!("create {}: {}", out_path, e));
    let buf = BufWriter::with_capacity(1 << 20, outf);
    let mut writer = CompressedTrainingDataEntryWriter::new(buf)
        .expect("Failed to create binpack writer");
    let (mut written, mut unique, mut skipped) = (0u64, 0u64, 0u64);
    let mut lines = BufReader::new(f).lines();
    lines.next(); // skip header
    let need = fen_col.max(score_col);
    for line in lines {
        let line = match line { Ok(l) => l, Err(_) => { skipped += 1; continue } };
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() <= need { skipped += 1; continue; }
        let fen = cols[fen_col].trim();
        let score: i32 = match cols[score_col].trim().parse() {
            Ok(s) => s, Err(_) => { skipped += 1; continue }
        };
        if score.abs() > max_abs { skipped += 1; continue; }
        let board = Board::from_fen(fen);
        let legal = generate_legal_moves(&board);
        if legal.len == 0 { skipped += 1; continue; }
        let mv = legal.get(0);
        let result: i16 = if score > 200 { 1 } else if score < -200 { -1 } else { 0 };
        match to_sf_entry(&board, mv, score.clamp(-32000, 32000) as i16, result, 0) {
            Some(entry) => {
                unique += 1;
                for _ in 0..repeat.max(1) {
                    writer.write_entry(&entry).expect("Failed to write entry");
                    written += 1;
                }
            }
            None => { skipped += 1; }
        }
    }
    drop(writer); // flush
    println!("import-tsv: {} entries written ({} unique x{} repeat), {} skipped \
              (mate-ish/parse/illegal)", written, unique, repeat.max(1), skipped);
}

/// Find the legal move matching a UCI string (e.g. "g1f3", "e7e8q") by formatting
/// each legal move and comparing — robust to flag/encoding details (EP, castle,
/// promotion) since it uses the same `move_to_uci` the engine emits.
fn find_move_by_uci(board: &Board, uci: &str) -> Option<Move> {
    let legal = generate_legal_moves(board);
    for i in 0..legal.len {
        let m = legal.get(i);
        if crate::types::move_to_uci(m) == uci {
            return Some(m);
        }
    }
    None
}

/// Import a per-position game TSV into an sfbinpack binpack, writing EVERY
/// position in game order with its ACTUAL move played (so consecutive same-game
/// positions chain -> the binpack delta-compresses them), the ACTUAL game WDL
/// result, and the ACTUAL ply. NO filtering: the trainer's loader does that, and
/// keeping all positions in order is what preserves chain compression. This is
/// the full-game complement to `import_tsv` (which writes isolated positions).
///
/// Columns (TAB-separated, NO header — feed `pgn_to_game_tsv.py` output, file or
/// "-" for stdin): 0 fen | 1 uci_move | 2 score(STM-POV cp) | 3 result(+1/0/-1
/// STM-POV) | 4 ply.
pub fn import_game_tsv(tsv_path: &str, out_path: &str) {
    use std::io::{BufRead, BufReader, BufWriter, Read};
    let reader: Box<dyn Read> = if tsv_path == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(tsv_path).unwrap_or_else(|e| panic!("open {}: {}", tsv_path, e)))
    };
    let outf = std::fs::File::create(out_path)
        .unwrap_or_else(|e| panic!("create {}: {}", out_path, e));
    let buf = BufWriter::with_capacity(1 << 20, outf);
    let mut writer = CompressedTrainingDataEntryWriter::new(buf)
        .expect("Failed to create binpack writer");
    let (mut written, mut skipped) = (0u64, 0u64);
    for line in BufReader::new(reader).lines() {
        let line = match line { Ok(l) => l, Err(_) => { skipped += 1; continue } };
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 { skipped += 1; continue; }
        let board = Board::from_fen(cols[0].trim());
        let mv = match find_move_by_uci(&board, cols[1].trim()) {
            Some(m) => m,
            None => { skipped += 1; continue }
        };
        let score: i32 = cols[2].trim().parse().unwrap_or(0);
        let result: i16 = cols[3].trim().parse().unwrap_or(0);
        let ply: u16 = cols[4].trim().parse().unwrap_or(0);
        match to_sf_entry(&board, mv, score.clamp(-32000, 32000) as i16, result, ply) {
            Some(entry) => { writer.write_entry(&entry).expect("Failed to write entry"); written += 1; }
            None => { skipped += 1; }
        }
        if written != 0 && written % 5_000_000 == 0 {
            eprintln!("  ...{} positions written", written);
        }
    }
    drop(writer); // flush
    println!("import-game-tsv: {} positions written, {} skipped (parse/illegal-move)",
             written, skipped);
}

/// Convert a Coda move to sfbinpack Move format.
fn make_sf_move(_pos: &SfPosition, mv: Move, board: &Board) -> Option<SfMove> {
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
    Some(sf_move)
}

/// Run data generation.
pub fn run_datagen(config: &DatagenConfig) {
    let start = Instant::now();
    let threads = config.threads.max(1);

    // Verify NNUE loading on thread 0 before spawning workers
    {
        let mut test_info = SearchInfo::new(1);
        test_info.silent = true;
        let loaded = if !config.nnue_path.is_empty() {
            test_info.load_nnue(&config.nnue_path).is_ok()
        } else {
            test_info.auto_discover_nnue()
        };
        if !loaded {
            eprintln!("Error: No NNUE net found. Use --nnue or build with 'make' for embedded net.");
            return;
        }
        eprintln!("Datagen using NNUE: {} threads, depth {}, {} games",
            threads, config.depth, config.num_games);
    }

    let games_done = Arc::new(AtomicU64::new(0));
    let positions_done = Arc::new(AtomicU64::new(0));

    let (tx, rx) = std::sync::mpsc::channel::<Vec<TrainingDataEntry>>();

    // Writer thread
    let output_path = config.output_path.clone();
    let positions_done_w = positions_done.clone();
    let writer_handle = std::thread::spawn(move || {
        let file = std::fs::OpenOptions::new()
            .create(true).append(true).open(&output_path)
            .unwrap_or_else(|_| panic!("Failed to open output file: {}", output_path));
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
        DatagenMode::SelfPlay { blunder_rate, force_capture_rate } => {
            let blunder_rate = *blunder_rate;
            let force_capture_rate = *force_capture_rate;
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
                        blunder_rate, force_capture_rate, &tx, &games_done);
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
                .unwrap_or_else(|_| panic!("Failed to read EPD: {}", source_epd));
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
    depth: i32, hash_mb: usize, blunder_rate: f64, force_capture_rate: f64,
    tx: &std::sync::mpsc::Sender<Vec<TrainingDataEntry>>,
    games_done: &AtomicU64,
) {
    let mut rng = SimpleRng::new(0xDEADBEEF ^ (thread_id as u64 * 0x9E3779B97F4A7C15));
    let mut info = SearchInfo::new(hash_mb);
    info.silent = true;
    if !nnue_path.is_empty() {
        if let Err(e) = info.load_nnue(nnue_path) {
            eprintln!("Thread {}: Failed to load NNUE: {}", thread_id, e);
            return;
        }
    } else if !info.auto_discover_nnue() {
        eprintln!("Thread {}: No NNUE net found", thread_id);
        return;
    }

    for _ in 0..my_games {
        let entries = play_one_game(&mut info, &mut rng, depth, blunder_rate, force_capture_rate);
        if !entries.is_empty()
            && tx.send(entries).is_err() { break; }
        info.tt.clear();
        games_done.fetch_add(1, Ordering::Relaxed);
    }
}

fn play_one_game(info: &mut SearchInfo, rng: &mut SimpleRng, depth: i32, blunder_rate: f64, force_capture_rate: f64)
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
        board.make_move(legal.get(idx));
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
        info.tt.new_search();
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

        let mv = if force_capture_rate > 0.0 && rng.next_f64() < force_capture_rate && ply >= 4 {
            // Force a random capture if any are available
            let captures: Vec<Move> = (0..legal.len)
                .map(|i| legal.get(i))
                .filter(|&m| {
                    board.piece_type_at(move_to(m)) != NO_PIECE_TYPE
                    || move_flags(m) == FLAG_EN_PASSANT
                })
                .collect();
            if captures.is_empty() {
                best_move // no captures available, play normally
            } else {
                captures[rng.next_u64() as usize % captures.len()]
            }
        } else if blunder_rate > 0.0 && rng.next_f64() < blunder_rate && ply >= 4 {
            let idx = rng.next_u64() as usize % legal.len;
            legal.get(idx)
        } else {
            best_move
        };

        // Write ALL positions for chain continuity (matches T80 format).
        // Store the ACTUALLY PLAYED move (not best_move) so pos.after_move(mv)
        // matches the next entry's position — required for binpack chain compression.
        // Score is still from search (best_move's score). Filtering done at training time.
        entries.push((board.clone(), mv, score.clamp(-32000, 32000) as i16, ply));

        if mv == NO_MOVE { break; }
        board.make_move(mv);
        ply += 1;
    }

    let game_result = result.unwrap_or(0);
    let mut sf_entries = Vec::with_capacity(entries.len());

    // Build chained SfPositions: first from FEN, then via after_move().
    // This guarantees is_continuation() passes for the compressed writer,
    // avoiding 32-byte packed entries where 2-3 byte continuations should be.
    let mut running_pos: Option<SfPosition> = None;
    for (board, mv, score, ply) in &entries {
        let r = if board.side_to_move == WHITE { game_result } else { -game_result };

        let pos = if let Some(ref prev_pos) = running_pos {
            // Use chained position from previous after_move
            *prev_pos
        } else {
            // First entry: create from FEN
            match SfPosition::from_fen(&board.to_fen()) {
                Ok(p) => p,
                Err(_) => continue,
            }
        };

        // Convert our move to sfbinpack move
        let sf_mv = match make_sf_move(&pos, *mv, board) {
            Some(m) => m,
            None => {
                running_pos = None; // break chain on conversion failure
                continue;
            }
        };

        sf_entries.push(TrainingDataEntry {
            pos,
            mv: sf_mv,
            score: *score,
            ply: *ply,
            result: r,
        });

        // Advance running position for next entry's chain
        running_pos = Some(sf_entries.last().unwrap().pos.after_move(sf_mv));
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
    if !nnue_path.is_empty() {
        if let Err(e) = info.load_nnue(nnue_path) {
            eprintln!("Thread {}: Failed to load NNUE: {}", thread_id, e);
            return;
        }
    } else if !info.auto_discover_nnue() {
        eprintln!("Thread {}: No NNUE net found", thread_id);
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

            // C8 audit LIKELY #47: `remove_piece` doesn't touch castling or
            // ep_square. Removing a corner rook (A1/H1/A8/H8) leaves the
            // castling bit set with no rook; removing a pawn that had just
            // double-pushed leaves the EP square valid but pointing at
            // empty. Clean these up so the produced position is legal.
            // Removing a rook from a corner: clear that castling right.
            if pt == ROOK {
                match (color, sq) {
                    (WHITE, 0) => modified.castling &= !0b0010,  // white queenside (A1)
                    (WHITE, 7) => modified.castling &= !0b0001,  // white kingside (H1)
                    (BLACK, 56) => modified.castling &= !0b1000, // black queenside (A8)
                    (BLACK, 63) => modified.castling &= !0b0100, // black kingside (H8)
                    _ => {}
                }
            }
            // Removing a pawn invalidates any EP square that's the
            // double-push destination of that pawn. Safest to clear on
            // any pawn removal.
            if pt == PAWN {
                modified.ep_square = crate::types::NO_SQUARE;
            }

            if !is_valid_position(&modified) { continue; }

            let limits = SearchLimits { depth, ..SearchLimits::default() };
            info.tt.new_search();
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

        if !batch.is_empty()
            && tx.send(batch).is_err() { return; }
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

/// genfens — OpenBench datagen opening-book generation.
///
/// OB invokes the *dev* engine as `./coda "genfens N seed S book B" quit`, i.e.
/// argv[1] is the entire genfens command string (with spaces). For each of N
/// openings we play a short run of random legal moves from the start position
/// and print the resulting FEN as `info string genfens <FEN>`, which OB's
/// `genfens.py` collects into the datagen opening book. Pure movegen — never
/// touches search or eval, so it cannot affect playing strength. The `book`
/// token (e.g. "None") is accepted and ignored; we always start from the
/// initial position. `seed` makes the output reproducible (OB runs several
/// processes, each with a distinct seed, for parallelism + diversity).
pub fn run_genfens(arg: &str) {
    let toks: Vec<&str> = arg.split_whitespace().collect();
    let mut n: usize = 0;
    let mut seed: u64 = 1;
    let mut i = 0;
    while i < toks.len() {
        match toks[i] {
            "genfens" => { if i + 1 < toks.len() { n = toks[i + 1].parse().unwrap_or(0); i += 1; } }
            "seed"    => { if i + 1 < toks.len() { seed = toks[i + 1].parse().unwrap_or(1); i += 1; } }
            _ => {} // "book <name>" and any extra tokens are ignored
        }
        i += 1;
    }
    genfens(n, seed);
}

fn genfens(n: usize, seed: u64) {
    use std::io::Write;
    let mut rng = SimpleRng::new(seed);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut generated = 0usize;
    let mut attempts = 0usize;
    // Safety cap so a pathological seed can never spin forever (random 6–9 ply
    // walks dead-end into mate/stalemate only very rarely, so this is slack).
    let attempt_cap = n.saturating_mul(1000).saturating_add(1000);
    while generated < n {
        attempts += 1;
        if attempts > attempt_cap { break; }
        let mut board = Board::startpos();
        let plies = 6 + (rng.next_u64() % 4); // 6–9 random plies (PlentyChess pattern)
        let mut dead = false;
        for _ in 0..plies {
            let legal = generate_legal_moves(&board);
            if legal.len == 0 { dead = true; break; } // mate/stalemate mid-walk → retry
            let idx = rng.next_u64() as usize % legal.len;
            board.make_move(legal.get(idx));
        }
        if dead { continue; }
        // The opening must itself be playable (the last random move could have
        // delivered mate/stalemate) — otherwise OB can't start a game from it.
        if generate_legal_moves(&board).len == 0 { continue; }
        let _ = writeln!(out, "info string genfens {}", board.to_fen());
        let _ = out.flush();
        generated += 1;
    }
}
