//! coda-rescore — SF-relabel a Coda-vs-SF datagen PGN into a chain-preserving
//! binpack. Coda-to-move positions are re-searched with Stockfish at fixed nodes
//! (the datagen budget) so every label is SF's; SF-to-move positions keep their
//! PGN eval. Emitted in game order so the binpack chain-compresses.
//!
//! Single SF engine per process (one UCI subprocess). Parallelism = shard the
//! input across N processes (each writes its own binpack; the loader reads many).
//! Rust replaces the python version whose PGN parse + GIL capped SF at ~50%.
//!
//!   bzcat shard.pgn.bz2 | coda-rescore --sf <sf> -o shard.binpack

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

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
#[command(about = "SF-relabel a Coda-vs-SF datagen PGN -> chained binpack")]
struct Args {
    /// Input PGN file ("-" for stdin)
    #[arg(short = 'i', long, default_value = "-")]
    input: String,
    /// Output binpack
    #[arg(short = 'o', long, default_value = "rescored.binpack")]
    output: String,
    /// Stockfish binary (must be the ob_17.1 + 1C000000 build for label consistency)
    #[arg(long)]
    sf: String,
    /// Fixed nodes per SF search (match the datagen)
    #[arg(long, default_value_t = 15000)]
    nodes: u64,
    /// Substring of the SF player name (its moves keep their PGN eval)
    #[arg(long, default_value = "Stockfish")]
    sf_player: String,
    /// Log progress every N games
    #[arg(long, default_value_t = 1000)]
    progress_every: u64,
}

/// Minimal persistent UCI client around one Stockfish subprocess.
struct Sf {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}
impl Sf {
    fn new(path: &str) -> Self {
        let mut child = Command::new(path)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
            .spawn().expect("spawn stockfish");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut sf = Sf { _child: child, stdin, stdout };
        sf.cmd("uci"); sf.wait("uciok");
        sf.cmd("isready"); sf.wait("readyok");
        sf
    }
    fn cmd(&mut self, s: &str) { writeln!(self.stdin, "{s}").unwrap(); self.stdin.flush().unwrap(); }
    fn wait(&mut self, tok: &str) {
        let mut line = String::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line).unwrap() == 0 { break; }
            if line.starts_with(tok) { break; }
        }
    }
    /// STM-POV cp at fixed nodes; mate -> ±30000.
    fn score(&mut self, fen: &str, nodes: u64) -> Option<i32> {
        self.cmd(&format!("position fen {fen}"));
        self.cmd(&format!("go nodes {nodes}"));
        let mut last: Option<i32> = None;
        let mut line = String::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line).unwrap() == 0 { return last; }
            if line.starts_with("bestmove") { return last; }
            if let Some(i) = line.find("score cp ") {
                if let Some(v) = line[i + 9..].split_whitespace().next() {
                    last = v.parse::<i32>().ok();
                }
            } else if let Some(i) = line.find("score mate ") {
                if let Some(v) = line[i + 11..].split_whitespace().next() {
                    last = v.parse::<i32>().ok().map(|m| if m >= 0 { 30000 } else { -30000 });
                }
            }
        }
    }
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

fn parse_eval(c: &[u8]) -> Option<i32> {
    // faithful to ([+-]?)(M)?(\d+(?:\.\d+)?)/  — eval before /depth
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

struct Rescorer {
    pos: Chess,
    white: String,
    black: String,
    white_res: Option<i16>,
    sf_player: String,
    nodes: u64,
    sf: Sf,
    writer: CompressedTrainingDataEntryWriter<BufWriter<std::fs::File>>,
    // pending move awaiting its trailing eval comment:
    pending: Option<(String, SfMove, u16, i16, bool)>, // fen, sfmove, ply, result, is_sf_mover
    pub games: u64,
    pub positions: u64,
    pub rescored: u64,
    pub kept: u64,
}

impl Visitor for Rescorer {
    type Result = ();
    fn begin_game(&mut self) {
        self.pos = Chess::default();
        self.white_res = None;
        self.white.clear(); self.black.clear();
        self.pending = None;
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
        let cp = match parse_eval(comment.as_bytes()) { Some(c) => c, None => return };
        let score = if is_sf { cp } else {
            self.rescored += 1;
            self.sf.score(&fen, self.nodes).unwrap_or(cp)
        };
        if is_sf { self.kept += 1; }
        if let Ok(pos) = SfPosition::from_fen(&fen) {
            let entry = TrainingDataEntry {
                pos, mv: smv,
                score: score.clamp(-32000, 32000) as i16,
                ply, result,
            };
            let _ = self.writer.write_entry(&entry);
            self.positions += 1;
        }
    }
    fn end_game(&mut self) {
        self.games += 1;
    }
}

fn main() {
    let a = Args::parse();
    let reader: Box<dyn std::io::Read> = if a.input == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&a.input).expect("open input"))
    };
    let outf = std::fs::File::create(&a.output).expect("create output");
    let writer = CompressedTrainingDataEntryWriter::new(BufWriter::with_capacity(1 << 20, outf))
        .expect("binpack writer");
    let mut v = Rescorer {
        pos: Chess::default(), white: String::new(), black: String::new(),
        white_res: None, sf_player: a.sf_player.clone(), nodes: a.nodes,
        sf: Sf::new(&a.sf), writer, pending: None,
        games: 0, positions: 0, rescored: 0, kept: 0,
    };
    let mut br = BufferedReader::new(reader);
    // read_all, logging progress periodically
    let mut last_log = 0u64;
    while br.read_game(&mut v).expect("read").is_some() {
        if v.games - last_log >= a.progress_every {
            last_log = v.games;
            eprintln!("  ...{} games, {} pos ({} SF-rescored, {} SF-kept)",
                      v.games, v.positions, v.rescored, v.kept);
        }
    }
    drop(v.writer); // flush
    eprintln!("done: {} games -> {} pos ({} rescored, {} kept)",
              v.games, v.positions, v.rescored, v.kept);
}
