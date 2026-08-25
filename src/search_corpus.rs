//! Deterministic construction of the search-audit corpus.
//!
//! This is a diagnostic CLI backend rather than a playing feature. Mate roots
//! come from Coda's existing tactical suites and must produce a mate score at a
//! fixed depth. Tablebase roots are generated locally, checked for basic legal
//! reachability, and labelled with the installed Syzygy tables.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use crate::board::Board;
use crate::epd;
use crate::movegen::generate_legal_moves;
use crate::search::{search, SearchInfo, SearchLimits};
use crate::tb::SyzygyTB;
use crate::tt::is_mate_score;
use crate::types::*;

const DEFAULT_EPD_SOURCES: &[&str] = &[
    "testdata/wac.epd",
    "testdata/wac2018.epd",
    "testdata/ecm.epd",
    "testdata/sbd.epd",
    "testdata/arasan.epd",
];

const CLOCKS: &[u16] = &[0, 50, 90, 99];

// Promoted from a deterministic corpus run when sampled decline-and-search
// disagreed with RFP. Keeping the exact interior FEN makes the observation
// reproducible even if later generator changes alter the parent corpus path.
const RFP_TB_COUNTEREXAMPLES: &[&str] = &["8/8/6qn/8/5b2/7K/2k5/8 w - - 0 8"];

pub fn resolve_syzygy_path(requested: &str) -> String {
    if !requested.is_empty() {
        return requested.to_string();
    }
    if let Ok(path) = std::env::var("CODA_SYZYGY_PATH") {
        if std::path::Path::new(&path).is_dir() {
            return path;
        }
    }
    let mut candidates = vec!["/tablebases".to_string()];
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(format!("{}/chess/tablebases", home));
    }
    candidates
        .into_iter()
        .find(|path| std::path::Path::new(path).is_dir())
        .unwrap_or_else(|| {
            panic!("no Syzygy directory found; pass --syzygy or set CODA_SYZYGY_PATH")
        })
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn below(&mut self, limit: usize) -> usize {
        (self.next() as usize) % limit
    }
}

fn placement_fen(squares: &[(u8, char)], stm: Color, halfmove: u16) -> String {
    let mut cells = [' '; 64];
    for &(sq, piece) in squares {
        cells[sq as usize] = piece;
    }

    let mut placement = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let piece = cells[rank * 8 + file];
            if piece == ' ' {
                empty += 1;
            } else {
                if empty > 0 {
                    placement.push(char::from_digit(empty, 10).unwrap());
                    empty = 0;
                }
                placement.push(piece);
            }
        }
        if empty > 0 {
            placement.push(char::from_digit(empty, 10).unwrap());
        }
        if rank > 0 {
            placement.push('/');
        }
    }
    format!(
        "{} {} - - {} 1",
        placement,
        if stm == WHITE { 'w' } else { 'b' },
        halfmove
    )
}

fn random_board(rng: &mut XorShift64, pieces: usize) -> Option<Board> {
    debug_assert!((3..=6).contains(&pieces));
    let mut occupied = BTreeSet::new();
    let white_king = rng.below(64) as u8;
    occupied.insert(white_king);
    let black_king = loop {
        let sq = rng.below(64) as u8;
        let file_gap = (sq % 8).abs_diff(white_king % 8);
        let rank_gap = (sq / 8).abs_diff(white_king / 8);
        if !occupied.contains(&sq) && (file_gap > 1 || rank_gap > 1) {
            break sq;
        }
    };
    occupied.insert(black_king);

    let mut placed = vec![(white_king, 'K'), (black_king, 'k')];
    let piece_chars = ['P', 'N', 'B', 'R', 'Q'];
    while placed.len() < pieces {
        let mut sq = rng.below(64) as u8;
        while occupied.contains(&sq) {
            sq = rng.below(64) as u8;
        }
        let mut piece = piece_chars[rng.below(piece_chars.len())];
        if piece == 'P' && (sq / 8 == 0 || sq / 8 == 7) {
            piece = piece_chars[1 + rng.below(piece_chars.len() - 1)];
        }
        if rng.next() & 1 != 0 {
            piece = piece.to_ascii_lowercase();
        }
        occupied.insert(sq);
        placed.push((sq, piece));
    }

    let stm = if rng.next() & 1 == 0 { WHITE } else { BLACK };
    let board = Board::from_fen(&placement_fen(&placed, stm, 0));
    if board.occupied().count_ones() as usize != pieces {
        return None;
    }

    // A reachable position may check the side to move, but the side that just
    // moved cannot have left its own king attacked.
    let mut previous_mover = board.clone();
    previous_mover.side_to_move = flip_color(stm);
    if previous_mover.in_check() || generate_legal_moves(&board).len == 0 {
        return None;
    }
    Some(board)
}

fn reset_search(info: &mut SearchInfo) {
    info.nodes = 0;
    info.stop.store(false, Ordering::Relaxed);
    info.clear_persistent_histories();
    info.tt.clear();
}

fn safe_tag(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn mate_lines(
    sources: &[String],
    depth: i32,
    target: usize,
    nnue_path: Option<&str>,
) -> Vec<String> {
    if target == 0 {
        return Vec::new();
    }
    let source_paths: Vec<String> = if sources.is_empty() {
        DEFAULT_EPD_SOURCES
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    } else {
        sources.to_vec()
    };
    let suites: Vec<Vec<epd::EpdPosition>> =
        source_paths.iter().map(|p| epd::parse_epd(p)).collect();
    let max_len = suites.iter().map(Vec::len).max().unwrap_or(0);

    let mut info = SearchInfo::new(16);
    info.silent = true;
    epd::resolve_net(&mut info, nnue_path).unwrap_or_else(|e| panic!("mate corpus: {}", e));
    let limits = SearchLimits {
        depth,
        fixed_depth: true,
        infinite: true,
        ..SearchLimits::new()
    };

    let mut lines = Vec::new();
    let mut seen = BTreeSet::new();
    'positions: for row in 0..max_len {
        for suite in &suites {
            let Some(pos) = suite.get(row) else { continue };
            let key = pos
                .fen
                .split_whitespace()
                .take(4)
                .collect::<Vec<_>>()
                .join(" ");
            if !seen.insert(key) {
                continue;
            }
            let mut board = Board::from_fen(&pos.fen);
            reset_search(&mut info);
            let best = search(&mut board, &mut info, &limits);
            let score = info.last_score;
            if !is_mate_score(score) || best == NO_MOVE {
                continue;
            }
            let id = safe_tag(&pos.id);
            lines.push(format!(
                "{} ; kind=mate-root score={} source={}",
                board.to_fen(),
                score,
                id
            ));
            if lines.len() >= target {
                break 'positions;
            }

            // The child supplies the opposite score direction and is more
            // useful than synthetically flipping side-to-move, which can make
            // an unreachable position. Immediate checkmates are omitted because
            // they never enter the pruning code under study.
            let best_uci = move_to_uci(best);
            if board.make_move(best) && generate_legal_moves(&board).len > 0 {
                lines.push(format!(
                    "{} ; kind=mate-child parent_score={} move={} source={}",
                    board.to_fen(),
                    score,
                    best_uci,
                    id
                ));
                if lines.len() >= target {
                    break 'positions;
                }
            }
        }
    }
    lines.truncate(target);
    lines
}

fn tb_lines(tb: &SyzygyTB, target: usize) -> Vec<String> {
    if target == 0 {
        return Vec::new();
    }
    assert!(
        tb.max_pieces() >= 5,
        "search corpus requires complete five-piece tables"
    );

    let direct_target = target / 2;
    let transition_target = target - direct_target;
    let mut direct = Vec::new();
    let mut transitions = Vec::new();
    let mut seen_direct = BTreeSet::new();
    let mut seen_transition = BTreeSet::new();
    let mut rng = XorShift64::new(0xC0DA_5EED_D1A6_2026);
    let mut attempts = 0usize;

    while (direct.len() < direct_target || transitions.len() < transition_target)
        && attempts < 2_000_000
    {
        attempts += 1;
        if direct.len() < direct_target {
            if let Some(board) = random_board(&mut rng, 5) {
                let key = board
                    .to_fen()
                    .split_whitespace()
                    .take(4)
                    .collect::<Vec<_>>()
                    .join(" ");
                if seen_direct.insert(key) {
                    if let Some(wdl0) = tb.probe_wdl(&board) {
                        if wdl0.abs() > 1 {
                            for &clock in CLOCKS {
                                if direct.len() >= direct_target {
                                    break;
                                }
                                let mut variant = board.clone();
                                variant.halfmove = clock;
                                let wdl = tb.probe_wdl(&variant).unwrap_or(0);
                                direct.push(format!(
                                    "{} ; kind=tb-direct wdl={} wdl0={} clock={}",
                                    variant.to_fen(),
                                    wdl,
                                    wdl0,
                                    clock
                                ));
                            }
                        }
                    }
                }
            }
        }

        if transitions.len() < transition_target {
            if let Some(board) = random_board(&mut rng, 6) {
                let key = board
                    .to_fen()
                    .split_whitespace()
                    .take(4)
                    .collect::<Vec<_>>()
                    .join(" ");
                if !seen_transition.insert(key) {
                    continue;
                }
                let mut win_entries = 0usize;
                let mut loss_entries = 0usize;
                for &mv in generate_legal_moves(&board).as_slice() {
                    if board.piece_type_at(move_to(mv)) == NO_PIECE_TYPE
                        && move_flags(mv) != FLAG_EN_PASSANT
                    {
                        continue;
                    }
                    let mut child = board.clone();
                    if !child.make_move(mv) || child.occupied().count_ones() != 5 {
                        continue;
                    }
                    if let Some(wdl) = tb.probe_wdl(&child) {
                        if wdl < -1 {
                            win_entries += 1;
                        }
                        if wdl > 1 {
                            loss_entries += 1;
                        }
                    }
                }
                if win_entries + loss_entries > 0 {
                    for &clock in CLOCKS {
                        if transitions.len() >= transition_target {
                            break;
                        }
                        let mut variant = board.clone();
                        variant.halfmove = clock;
                        transitions.push(format!(
                            "{} ; kind=tb-transition winning_entries={} losing_entries={} clock={}",
                            variant.to_fen(),
                            win_entries,
                            loss_entries,
                            clock
                        ));
                    }
                }
            }
        }
    }

    assert_eq!(
        direct.len(),
        direct_target,
        "could not generate requested direct TB positions"
    );
    assert_eq!(
        transitions.len(),
        transition_target,
        "could not generate requested TB transitions"
    );
    direct.extend(transitions);
    direct
}

pub fn build(
    output: &str,
    epd_sources: &[String],
    depth: i32,
    mate_target: usize,
    tb_target: usize,
    syzygy_path: &str,
    nnue_path: Option<&str>,
) {
    let tb = SyzygyTB::new(syzygy_path)
        .unwrap_or_else(|e| panic!("failed to load Syzygy tables '{}': {}", syzygy_path, e));
    let mates = mate_lines(epd_sources, depth, mate_target, nnue_path);
    let tablebases = tb_lines(&tb, tb_target);
    let counterexamples: Vec<String> = RFP_TB_COUNTEREXAMPLES
        .iter()
        .map(|fen| {
            let board = Board::from_fen(fen);
            let wdl = tb.probe_wdl(&board).expect("counterexample must be covered by Syzygy");
            format!(
                "{} ; kind=tb-counterexample wdl={} observed_depth=2 observed_ply=13 observed_beta=-28796 observed_verified=-28985",
                board.to_fen(), wdl
            )
        })
        .collect();

    let mut text = String::new();
    text.push_str("# Coda search audit corpus v1\n");
    text.push_str("# Six-field FEN is mandatory; metadata follows ';'.\n");
    text.push_str(&format!(
        "# mate classification depth={} requested={} retained={}\n",
        depth,
        mate_target,
        mates.len()
    ));
    text.push_str(&format!(
        "# tablebase positions={} generator_seed=0xC0DA5EEDD1A62026\n",
        tablebases.len()
    ));
    text.push_str(&format!(
        "# retained TB/RFP counterexamples={}\n",
        counterexamples.len()
    ));
    for line in mates
        .iter()
        .chain(tablebases.iter())
        .chain(counterexamples.iter())
    {
        text.push_str(line);
        text.push('\n');
    }
    std::fs::write(output, text)
        .unwrap_or_else(|e| panic!("failed to write search corpus '{}': {}", output, e));
    println!(
        "wrote {} positions to {} ({} mate, {} generated tablebase, {} retained counterexample)",
        mates.len() + tablebases.len() + counterexamples.len(),
        output,
        mates.len(),
        tablebases.len(),
        counterexamples.len(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_builder_preserves_clock_and_piece_count() {
        crate::init();
        let fen = placement_fen(&[(4, 'K'), (60, 'k'), (8, 'P'), (55, 'r')], BLACK, 99);
        let board = Board::from_fen(&fen);
        assert_eq!(board.occupied().count_ones(), 4);
        assert_eq!(board.side_to_move, BLACK);
        assert_eq!(board.halfmove, 99);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn random_generator_is_deterministic_and_basically_legal() {
        crate::init();
        let mut a = XorShift64::new(12345);
        let mut b = XorShift64::new(12345);
        let board_a = (0..1000).find_map(|_| random_board(&mut a, 6)).unwrap();
        let board_b = (0..1000).find_map(|_| random_board(&mut b, 6)).unwrap();
        assert_eq!(board_a.to_fen(), board_b.to_fen());
        assert_eq!(board_a.occupied().count_ones(), 6);
        assert!(generate_legal_moves(&board_a).len > 0);
    }

    #[test]
    fn checked_in_corpus_has_expected_shape_and_full_fens() {
        crate::init();
        let text = std::fs::read_to_string("testdata/search_audit.epd").unwrap();
        let lines: Vec<&str> = text
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .collect();
        assert_eq!(lines.len(), 193);
        for line in &lines {
            let fen = line
                .split_whitespace()
                .take(6)
                .collect::<Vec<_>>()
                .join(" ");
            assert_eq!(fen.split_whitespace().count(), 6);
            let board = Board::from_fen(&fen);
            assert_eq!(board.to_fen(), fen);
            assert_eq!(board.pieces[KING as usize].count_ones(), 2);
        }
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.contains("kind=mate-root"))
                .count(),
            32
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.contains("kind=mate-child"))
                .count(),
            32
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.contains("kind=tb-direct"))
                .count(),
            64
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.contains("kind=tb-transition"))
                .count(),
            64
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.contains("kind=tb-counterexample"))
                .count(),
            1
        );
        for clock in CLOCKS {
            let tag = format!("clock={}", clock);
            assert_eq!(lines.iter().filter(|l| l.contains(&tag)).count(), 32);
        }
    }
}
