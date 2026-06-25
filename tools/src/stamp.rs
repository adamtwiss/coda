//! coda-stamp — neutralize Coda-to-move positions in a Coda-vs-SF datagen PGN
//! by stamping their score with a FILTERED SENTINEL (32000, i.e. |score|>10000),
//! while keeping SF-to-move positions' real PGN eval. Output is a
//! chain-preserving binpack: every position is written in game order so the
//! binpack chain-compresses, but Bullet's quality filter
//! (`entry.score.unsigned_abs() <= 10000`) drops the sentinel positions at
//! train time. Net effect at training: only the SF-labelled half is used, with
//! zero Coda-eval poison — and NO Stockfish search needed (pure CPU, fast).
//!
//! This replaces the SF-rescore pipeline: instead of re-searching every
//! Coda-to-move position with SF@15000 nodes (~months at scale), we stamp them
//! in one cheap pass and just generate ~2× more raw data on the fleet.
//!
//!   bzcat shard.pgn.bz2 | coda-stamp -o shard.binpack
//!
//! Sentinel is 32000 (distinct from the ±30000 mate value, so post-hoc we can
//! tell "Coda-neutralised" apart from "real mate" — both are filtered anyway).

use std::io::BufWriter;

use clap::Parser;
use pgn_reader::{BufferedReader, RawComment, RawTag, SanPlus, Skip, Visitor};
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position, Move, Square};
use shakmaty::fen::Fen;

use sfbinpack::chess::position::Position as SfPosition;
use sfbinpack::chess::r#move::{Move as SfMove, MoveType as SfMoveType};
use sfbinpack::chess::coords::Square as SfSquare;
use sfbinpack::chess::piece::Piece as SfPiece;
use sfbinpack::chess::piecetype::PieceType as SfPieceType;
use sfbinpack::chess::color::Color as SfColor;
use sfbinpack::{TrainingDataEntry, CompressedTrainingDataEntryWriter};

#[derive(Parser)]
#[command(about = "Stamp Coda-to-move positions with a filtered sentinel -> chained binpack (no SF)")]
struct Args {
    /// Input PGN ("-" for stdin)
    #[arg(short = 'i', long, default_value = "-")]
    input: String,
    /// Output binpack
    #[arg(short = 'o', long, default_value = "stamped.binpack")]
    output: String,
    /// Substring of the SF player name (its moves keep their PGN eval; everyone
    /// else's moves — i.e. Coda's — get the sentinel)
    #[arg(long, default_value = "Stockfish")]
    sf_player: String,
    /// Sentinel score for Coda-to-move positions (must be |v|>10000 to be
    /// dropped by Bullet's filter; default 32000, distinct from ±30000 mate)
    #[arg(long, default_value_t = 32000)]
    sentinel: i16,
    /// Log progress every N games
    #[arg(long, default_value_t = 5000)]
    progress_every: u64,
}

/// shakmaty Move -> sfbinpack Move (king->rook castling, EP, promotion).
fn to_sf_move(mv: &Move, stm: Color) -> SfMove {
    let sq = |s: Square| SfSquare::new(u32::from(s as u8));
    match mv {
        Move::Castle { king, rook } => SfMove::new(sq(*king), sq(*rook), SfMoveType::Castle, SfPiece::none()),
        Move::EnPassant { from, to } => SfMove::new(sq(*from), sq(*to), SfMoveType::EnPassant, SfPiece::none()),
        Move::Normal { from, to, promotion: Some(role), .. } => {
            let c = if stm == Color::White { SfColor::White } else { SfColor::Black };
            let pt = match role {
                shakmaty::Role::Knight => SfPieceType::Knight,
                shakmaty::Role::Bishop => SfPieceType::Bishop,
                shakmaty::Role::Rook => SfPieceType::Rook,
                _ => SfPieceType::Queen,
            };
            SfMove::new(sq(*from), sq(*to), SfMoveType::Promotion, SfPiece::new(pt, c))
        }
        Move::Normal { from, to, .. } => SfMove::new(sq(*from), sq(*to), SfMoveType::Normal, SfPiece::none()),
        _ => SfMove::new(SfSquare::new(0), SfSquare::new(0), SfMoveType::Normal, SfPiece::none()),
    }
}

/// Port of the eval parse in coda-rescore: `([+-]?)(M)?(\d+(?:\.\d+)?)/` -> cp.
fn parse_eval(c: &[u8]) -> Option<i32> {
    let n = c.len();
    let mut i = 0;
    while i < n {
        let mut j = i;
        let mut sign = 1i32;
        if c[j] == b'+' || c[j] == b'-' { if c[j] == b'-' { sign = -1; } j += 1; }
        let is_mate = j < n && c[j] == b'M';
        if is_mate { j += 1; }
        let ds = j;
        while j < n && c[j].is_ascii_digit() { j += 1; }
        if j > ds {
            if j < n && c[j] == b'.' {
                let mut k = j + 1;
                while k < n && c[k].is_ascii_digit() { k += 1; }
                if k > j + 1 { j = k; }
            }
            if j < n && c[j] == b'/' {
                if is_mate { return Some(sign * 30000); }
                let v: f64 = std::str::from_utf8(&c[ds..j]).ok()?.parse().ok()?;
                return Some((sign as f64 * v * 100.0).round() as i32);
            }
        }
        i += 1;
    }
    None
}

struct Stamper {
    pos: Chess,
    white: String,
    black: String,
    white_res: Option<i16>,
    sf_player: String,
    sentinel: i16,
    writer: CompressedTrainingDataEntryWriter<BufWriter<std::fs::File>>,
    pending: Option<(String, SfMove, u16, i16, bool)>, // fen, sfmove, ply, result, is_sf_mover
    pub games: u64,
    pub positions: u64,
    pub stamped: u64, // Coda-mover -> sentinel
    pub kept: u64,    // SF-mover -> real eval
    // CHAIN FIX: the position chained forward via after_move, so the writer's
    // is_continuation always holds (re-deriving each position from FEN gives a
    // mismatched halfmove clock and fragments chains ~14x/game -> ~2x bloat).
    cur_sf: Option<SfPosition>,
}

impl Visitor for Stamper {
    type Result = ();
    fn begin_game(&mut self) {
        self.pos = Chess::default();
        self.white_res = None;
        self.white.clear(); self.black.clear();
        self.pending = None;
        self.cur_sf = None; // a chain always restarts at a new game
    }
    fn tag(&mut self, name: &[u8], value: RawTag<'_>) {
        match name {
            b"FEN" => {
                if let Ok(f) = Fen::from_ascii(value.as_bytes()) {
                    if let Ok(p) = f.into_position::<Chess>(CastlingMode::Standard) { self.pos = p; }
                }
            }
            b"White" => self.white = String::from_utf8_lossy(value.as_bytes()).into_owned(),
            b"Black" => self.black = String::from_utf8_lossy(value.as_bytes()).into_owned(),
            b"Result" => self.white_res = match value.as_bytes() {
                b"1-0" => Some(1), b"0-1" => Some(-1), b"1/2-1/2" => Some(0), _ => None,
            },
            _ => {}
        }
    }
    fn end_tags(&mut self) -> Skip { Skip(self.white_res.is_none()) }
    fn begin_variation(&mut self) -> Skip { Skip(true) }
    fn san(&mut self, san_plus: SanPlus) {
        self.pending = None;
        let m = match san_plus.san.to_move(&self.pos) { Ok(m) => m, Err(_) => return };
        let stm_white = self.pos.turn() == Color::White;
        let mover = if stm_white { &self.white } else { &self.black };
        let is_sf = mover.contains(&self.sf_player);
        let fen = Fen::from_position(&self.pos, EnPassantMode::Legal).to_string();
        let ply = (2 * (self.pos.fullmoves().get() - 1) + if stm_white { 0 } else { 1 }) as u16;
        let result = self.white_res.unwrap_or(0) * if stm_white { 1 } else { -1 };
        let smv = to_sf_move(&m, self.pos.turn());
        self.pending = Some((fen, smv, ply, result, is_sf));
        self.pos.play_unchecked(m);
    }
    fn comment(&mut self, comment: RawComment<'_>) {
        let Some((fen, smv, ply, result, is_sf)) = self.pending.take() else { return };
        // CHAIN FIX: chain from the prior position via after_move; take() => any
        // early return below resets to from_fen at the next ply (correct on drops).
        let chained = self.cur_sf.take();
        // SF-mover keeps its parsed eval; Coda-mover gets the filtered sentinel
        // (no eval parse needed for the Coda half — it's discarded at train time).
        let score = if is_sf {
            match parse_eval(comment.as_bytes()) { Some(c) => { self.kept += 1; c } None => return }
        } else {
            self.stamped += 1;
            self.sentinel as i32
        };
        // Use the exact after_move(prev) position when chaining (so is_continuation
        // holds); only at a game start (or after a drop) fall back to from_fen.
        let maybe_pos = chained.or_else(|| SfPosition::from_fen(&fen).ok());
        if let Some(pos) = maybe_pos {
            let entry = TrainingDataEntry {
                pos, mv: smv,
                score: score.clamp(-32000, 32000) as i16,
                ply, result,
            };
            let _ = self.writer.write_entry(&entry);
            self.positions += 1;
            self.cur_sf = Some(pos.after_move(smv)); // advance the chain for the next ply
        }
    }
    fn end_game(&mut self) { self.games += 1; }
}

fn main() {
    let a = Args::parse();
    assert!(a.sentinel.unsigned_abs() > 10000,
            "sentinel {} must have |value|>10000 to be dropped by Bullet's filter", a.sentinel);
    let reader: Box<dyn std::io::Read> = if a.input == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&a.input).expect("open input"))
    };
    let outf = std::fs::File::create(&a.output).expect("create output");
    let writer = CompressedTrainingDataEntryWriter::new(BufWriter::with_capacity(1 << 20, outf))
        .expect("binpack writer");
    let mut v = Stamper {
        pos: Chess::default(), white: String::new(), black: String::new(),
        white_res: None, sf_player: a.sf_player.clone(), sentinel: a.sentinel,
        writer, pending: None,
        games: 0, positions: 0, stamped: 0, kept: 0,
        cur_sf: None,
    };
    let mut br = BufferedReader::new(reader);
    let mut last_log = 0u64;
    while br.read_game(&mut v).expect("read").is_some() {
        if v.games - last_log >= a.progress_every {
            last_log = v.games;
            eprintln!("  ...{} games, {} pos ({} SF-kept, {} Coda-stamped)",
                      v.games, v.positions, v.kept, v.stamped);
        }
    }
    drop(v.writer);
    eprintln!("done: {} games -> {} pos ({} SF-kept, {} Coda-stamped@{})",
              v.games, v.positions, v.kept, v.stamped, a.sentinel);
}
