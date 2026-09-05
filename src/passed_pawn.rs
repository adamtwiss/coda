//! Passed-pawn input features.
//!
//! Passed-pawn inputs are a cross-engine pattern; we believe the idea
//! originates from Triumviratus.
//!
//! # Encoding (96 features)
//!
//! One feature per passed pawn, indexed by its SQUARE and whether it is ours:
//!
//! * 48 squares (ranks 2..7 — a pawn cannot stand elsewhere)
//! * x 2 colours (ours / theirs), STM-relative
//! * = **96 features**
//!
//! Square-indexed for the same reason the pawn-pair block is: a passer on the
//! 7th rank and one on the 3rd are different things, and an encoding that
//! cannot tell them apart carries influence without carrying information.
//!
//! This is a *relational* property — whether a pawn is passed depends on every
//! enemy pawn on three files — so the king-bucketed piece-square features
//! cannot express it directly however wide the net gets.
//!
//! Cost is negligible: a typical position has 0-3 passers, against the
//! pawn-pair block's ~18 active features.

use crate::types::{Color, Move, WHITE};
use crate::bitboard::{Bitboard, FILE_A, FILE_H};

/// Squares a pawn can occupy (ranks 2..7).
const PAWN_SQUARES: usize = 48;
/// (ours, theirs), STM-relative.
const PASSED_PAWN_COLOURS: usize = 2;
/// Total passed-pawn input features.
pub const PASSED_PAWN_FEATURES: usize = PAWN_SQUARES * PASSED_PAWN_COLOURS;

/// Upper bound on passed-pawn deltas from a single move.
///
/// A move can flip at most every pawn's passed status (removing a lone sentry
/// can pass a whole phalanx), so the bound is "every pawn off, every pawn on":
/// 16 removals + 16 additions.
pub const MAX_PASSED_PAWN_DELTAS: usize = 32;

#[inline]
fn north_fill(mut b: Bitboard) -> Bitboard {
    b |= b << 8;
    b |= b << 16;
    b |= b << 32;
    b
}

#[inline]
fn south_fill(mut b: Bitboard) -> Bitboard {
    b |= b >> 8;
    b |= b >> 16;
    b |= b >> 32;
    b
}

/// Widen a mask to the two adjacent files.
#[inline]
fn widen(b: Bitboard) -> Bitboard {
    b | ((b & !FILE_H) << 1) | ((b & !FILE_A) >> 1)
}

/// Passed pawns for both colours, as `[white, black]` bitboards.
///
/// A pawn is passed when no enemy pawn stands on its own or an adjacent file
/// on any square strictly ahead of it. Computed as a span rather than
/// per-pawn: shift the enemy pawns one square toward us, fill away from them,
/// widen to adjacent files, and subtract.
#[inline]
pub fn passed_pawns(white_pawns: Bitboard, black_pawns: Bitboard) -> [Bitboard; 2] {
    // Everything strictly south of a black pawn, on its file or a neighbour:
    // the squares from which a white pawn could never outrun it.
    let blocked_for_white = widen(south_fill(black_pawns >> 8));
    let blocked_for_black = widen(north_fill(white_pawns << 8));
    [white_pawns & !blocked_for_white, black_pawns & !blocked_for_black]
}

/// Feature index for one passed pawn, or `None` if the square cannot hold one.
///
/// `sq` is already perspective-transformed.
#[inline]
pub fn passer_feature(sq: u8, mine: bool) -> Option<usize> {
    if !(8..56).contains(&sq) {
        return None;
    }
    Some((sq as usize - 8) * PASSED_PAWN_COLOURS + (mine as usize))
}

/// Enumerate active passed-pawn features from `pov`'s perspective.
pub fn enumerate_passed_pawns<F: FnMut(usize)>(
    pawns_bb: Bitboard,
    colors_bb: &[Bitboard; 2],
    pov: Color,
    mirrored: bool,
    mut callback: F,
) {
    let white = pawns_bb & colors_bb[WHITE as usize];
    let black = pawns_bb & !colors_bb[WHITE as usize];
    let passers = passed_pawns(white, black);
    for c in 0..2 {
        let mine = (c as Color) == pov;
        let mut bb = passers[c];
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            let mut s = sq;
            if pov != WHITE { s ^= 56; }
            if mirrored { s ^= 7; }
            if let Some(i) = passer_feature(s, mine) {
                callback(i);
            }
        }
    }
}

/// One passed-pawn change: a square, its colour, and whether the pawn became
/// passed or stopped being passed. Physical square and absolute colour, like
/// the other delta types, so one delta serves both perspectives.
/// Packed into 8 bits: `sq(6) | colour(1) | add(1)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PassedPawnDelta(u8);

impl PassedPawnDelta {
    pub const ZERO: Self = Self(0);
    #[inline]
    pub fn new(sq: u8, col: Color, add: bool) -> Self {
        Self(sq | ((col as u8) << 6) | ((add as u8) << 7))
    }
    #[inline] pub fn sq(self) -> u8 { self.0 & 63 }
    #[inline] pub fn col(self) -> Color { ((self.0 >> 6) & 1) as Color }
    #[inline] pub fn add(self) -> bool { (self.0 >> 7) & 1 == 1 }
}

/// Feature index for one delta from one perspective.
#[inline]
pub fn passed_index_for(d: PassedPawnDelta, pov: Color, mirrored: bool) -> Option<usize> {
    let mut s = d.sq();
    if pov != WHITE { s ^= 56; }
    if mirrored { s ^= 7; }
    passer_feature(s, d.col() == pov)
}

/// Emit the passed-pawn deltas for `mv`, given the pawn state BEFORE it.
///
/// **This cannot be a local computation, unlike the pawn-pair block.** Moving
/// one pawn can flip the passed status of several OTHERS — capturing a lone
/// sentry can pass a whole phalanx, and advancing a pawn can un-pass an enemy
/// one. So both passer sets are recomputed before and after and diffed. That
/// is about a dozen bitboard operations, cheaper than reasoning about which
/// pawns could have been affected, and correct by construction.
///
/// Uses the SAME `pawn_set_change` derivation as the pawn-pair block, so the
/// two blocks cannot disagree about what a move did to the pawn set.
pub fn push_passed_pawn_deltas(
    out: &mut Vec<PassedPawnDelta>,
    pawns_bb: Bitboard,
    colors_bb: &[Bitboard; 2],
    us: Color,
    mv: Move,
    captured: u8,
    pt: u8,
) {
    out.clear();
    let ch = crate::pawn_pair::pawn_set_change(us, mv, captured, pt);
    if ch.n_removed == 0 && ch.added.is_none() {
        return; // no pawn moved and none was captured
    }

    let mut w = pawns_bb & colors_bb[WHITE as usize];
    let mut b = pawns_bb & !colors_bb[WHITE as usize];
    let before = passed_pawns(w, b);

    for k in 0..ch.n_removed {
        let (sq, col) = ch.removed[k];
        let bit = 1u64 << sq;
        if col == WHITE { w &= !bit; } else { b &= !bit; }
    }
    if let Some((sq, col)) = ch.added {
        let bit = 1u64 << sq;
        if col == WHITE { w |= bit; } else { b |= bit; }
    }
    let after = passed_pawns(w, b);

    for c in 0..2 {
        let col = c as Color;
        let mut gone = before[c] & !after[c];
        while gone != 0 {
            let sq = gone.trailing_zeros() as u8;
            gone &= gone - 1;
            out.push(PassedPawnDelta::new(sq, col, false));
        }
        let mut born = after[c] & !before[c];
        while born != 0 {
            let sq = born.trailing_zeros() as u8;
            born &= born - 1;
            out.push(PassedPawnDelta::new(sq, col, true));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::types::{BLACK, PAWN};

    fn passers_of(fen: &str) -> ([Vec<String>; 2], Board) {
        let mut b = Board::new();
        b.set_fen(fen);
        let white = b.pieces[PAWN as usize] & b.colors[WHITE as usize];
        let black = b.pieces[PAWN as usize] & b.colors[BLACK as usize];
        let p = passed_pawns(white, black);
        let name = |mut bb: Bitboard| {
            let mut v = Vec::new();
            while bb != 0 {
                let s = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                v.push(format!("{}{}", (b'a' + (s % 8) as u8) as char, s / 8 + 1));
            }
            v.sort();
            v
        };
        ([name(p[0]), name(p[1])], b)
    }

    fn feats(fen: &str, pov: Color, mirrored: bool) -> Vec<usize> {
        let mut b = Board::new();
        b.set_fen(fen);
        let mut v = Vec::new();
        enumerate_passed_pawns(b.pieces[PAWN as usize], &b.colors, pov, mirrored, |i| v.push(i));
        v.sort_unstable();
        v
    }

    /// The detection itself, against positions where the answer is unambiguous.
    #[test]
    fn detection_is_correct() {
        // Lone white pawn, empty board ahead: passed.
        let (p, _) = passers_of("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1");
        assert_eq!(p[0], vec!["e4"], "lone pawn must be passed");

        // Enemy pawn directly in front on the same file: NOT passed.
        let (p, _) = passers_of("4k3/8/4p3/8/4P3/8/8/4K3 w - - 0 1");
        assert!(p[0].is_empty(), "pawn faced on its own file is not passed");

        // Enemy pawn ahead on an ADJACENT file: NOT passed.
        let (p, _) = passers_of("4k3/8/3p4/8/4P3/8/8/4K3 w - - 0 1");
        assert!(p[0].is_empty(), "pawn with a sentry on an adjacent file is not passed");

        // Enemy pawn BEHIND on an adjacent file: passed (it can never catch it).
        let (p, _) = passers_of("4k3/8/8/8/4P3/3p4/8/4K3 w - - 0 1");
        assert_eq!(p[0], vec!["e4"], "an enemy pawn behind does not stop a passer");

        // Enemy pawn two files away ahead: passed.
        let (p, _) = passers_of("4k3/8/2p5/8/4P3/8/8/4K3 w - - 0 1");
        assert_eq!(p[0], vec!["e4"], "a pawn two files away is not a sentry");

        // Both sides can have passers at once.
        let (p, _) = passers_of("4k3/p7/8/8/8/8/7P/4K3 w - - 0 1");
        assert_eq!(p[0], vec!["h2"]);
        assert_eq!(p[1], vec!["a7"]);
    }

    /// No file wraparound: an a-file pawn must not be stopped by an h-file one.
    #[test]
    fn no_file_wraparound() {
        let (p, _) = passers_of("4k3/7p/8/8/P7/8/8/4K3 w - - 0 1");
        assert_eq!(p[0], vec!["a4"], "h-file enemy pawn must not block the a-file");
        let (p, _) = passers_of("4k3/p7/8/8/7P/8/8/4K3 w - - 0 1");
        assert_eq!(p[0], vec!["h4"], "a-file enemy pawn must not block the h-file");
    }

    /// Same square, different rank, must be a different feature — the property
    /// whose absence made the first pawn-pair encoding inert.
    #[test]
    fn rank_and_file_are_encoded() {
        let low  = feats("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", WHITE, false); // e2
        let high = feats("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1", WHITE, false); // e6
        let far  = feats("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", WHITE, false);  // a2
        assert_eq!(low.len(), 1);
        assert_eq!(high.len(), 1);
        assert_eq!(far.len(), 1);
        assert_ne!(low[0], high[0], "different ranks must differ");
        assert_ne!(low[0], far[0], "different files must differ");
    }

    /// Ours vs theirs must differ.
    #[test]
    fn ownership_is_encoded() {
        let f = feats("4k3/p7/8/8/8/8/7P/4K3 w - - 0 1", WHITE, false);
        assert_eq!(f.len(), 2, "one passer each side");
        assert_ne!(f[0], f[1]);
        // Same position from black's view: the same two pawns, ownership swapped.
        let b = feats("4k3/p7/8/8/8/8/7P/4K3 w - - 0 1", BLACK, false);
        assert_eq!(b.len(), 2);
        assert_ne!(f, b, "ownership must flip with perspective");
    }

    /// Indices in range, over a broad corpus.
    #[test]
    fn indices_in_range() {
        let text = std::fs::read_to_string("positions.epd").unwrap_or_default();
        let mut n = 0;
        for line in text.lines().take(2000) {
            let fen = line.split('\t').next().unwrap_or("");
            if fen.split_whitespace().count() < 4 { continue; }
            for pov in [WHITE, BLACK] {
                for m in [false, true] {
                    for i in feats(fen, pov, m) {
                        assert!(i < PASSED_PAWN_FEATURES, "index {i} out of range");
                        n += 1;
                    }
                }
            }
        }
        assert!(n > 100, "corpus produced too few passers to be a real check: {n}");
    }

    /// THE invariant: applying a move's deltas to the parent's feature set must
    /// reproduce a fresh enumeration exactly, for both perspectives. This block
    /// is the harder case — a single move can flip several pawns at once — so
    /// the counter tracks how often that actually happened, and the test fails
    /// if the corpus never exercised it.
    #[test]
    fn deltas_reproduce_full_enumeration() {
        use crate::movegen::generate_legal_moves;
        use crate::types::{move_from, move_to, move_flags, is_promotion, FLAG_EN_PASSANT, Move};
        crate::init();

        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "8/P6P/4k3/8/8/4K3/p6p/8 w - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "4k3/8/8/1pP5/1P6/8/8/4K3 w - - 0 1",
            "4k3/pp4pp/8/8/8/8/PP4PP/4K3 w - - 0 1",
        ];
        fn next_u32(st: &mut u32) -> u32 { let mut x=*st; x^=x<<13; x^=x>>17; x^=x<<5; *st=x; x }
        fn mirror_of(b: &Board, pov: Color) -> bool { (b.king_sq(pov) % 8) >= 4 }
        fn set_of(b: &Board, pov: Color) -> [i32; PASSED_PAWN_FEATURES] {
            let mut c = [0i32; PASSED_PAWN_FEATURES];
            enumerate_passed_pawns(b.pieces[PAWN as usize], &b.colors, pov,
                                   mirror_of(b, pov), |i| c[i] += 1);
            c
        }

        let (mut checked, mut nonempty, mut multi, mut third_party) = (0, 0, 0, 0);
        for (fi, fen) in START_FENS.iter().enumerate() {
            for game in 0..20 {
                let mut rng = 0x2545_F491u32
                    .wrapping_add(fi as u32 * 7919).wrapping_add(game * 104_729) | 1;
                let mut board = Board::new();
                board.set_fen(fen);
                for _ply in 0..120 {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mut pawnish: Vec<Move> = Vec::new();
                    for i in 0..legal.len {
                        let m = legal.get(i);
                        if board.mailbox[move_from(m) as usize] == PAWN
                            || board.mailbox[move_to(m) as usize] == PAWN { pawnish.push(m); }
                    }
                    let mv = if !pawnish.is_empty() && next_u32(&mut rng) % 4 != 0 {
                        pawnish[(next_u32(&mut rng) as usize) % pawnish.len()]
                    } else { legal.get((next_u32(&mut rng) as usize) % legal.len) };

                    let us = board.side_to_move;
                    let is_ep = move_flags(mv) == FLAG_EN_PASSANT;
                    let captured = if is_ep { PAWN } else { board.mailbox[move_to(mv) as usize] };
                    let pt = board.mailbox[move_from(mv) as usize];
                    let before = [set_of(&board, WHITE), set_of(&board, 1 - WHITE)];
                    let mirror_before = [mirror_of(&board, WHITE), mirror_of(&board, 1 - WHITE)];

                    let mut d = Vec::new();
                    push_passed_pawn_deltas(&mut d, board.pieces[PAWN as usize],
                                            &board.colors, us, mv, captured, pt);
                    assert!(d.len() <= MAX_PASSED_PAWN_DELTAS,
                            "delta overflow {} > {}", d.len(), MAX_PASSED_PAWN_DELTAS);
                    if !d.is_empty() { nonempty += 1; }
                    if d.len() > 2 { multi += 1; }
                    // A delta on a square that is neither the from nor the to
                    // square proves the non-local case actually occurred.
                    if d.iter().any(|x| x.sq() != move_from(mv) && x.sq() != move_to(mv)) {
                        third_party += 1;
                    }

                    if !board.make_move(mv) { break; }
                    for pov in [WHITE, 1 - WHITE] {
                        let p = pov as usize;
                        if mirror_of(&board, pov) != mirror_before[p] { continue; }
                        let mut got = before[p];
                        for x in &d {
                            if let Some(i) = passed_index_for(*x, pov, mirror_before[p]) {
                                got[i] += if x.add() { 1 } else { -1 };
                            }
                        }
                        assert_eq!(got, set_of(&board, pov),
                            "delta replay != enumeration, fen {fi} game {game} pov {pov} mv {mv:#06x}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 5000, "too few checks: {checked}");
        assert!(nonempty > 200, "too few passer changes: {nonempty}");
        assert!(third_party > 20,
            "the NON-LOCAL case never occurred ({third_party}); this test would not \
             have caught a local-only implementation");
        println!("passed-pawn deltas: {checked} checks, {nonempty} changing moves, \
                  {multi} multi-pawn flips, {third_party} third-party flips");
    }

    /// Cross-check dump: `stm|ntm` indices per FEN, for diffing against the
    /// trainer's enumeration. Env-gated.
    #[test]
    #[ignore]
    fn dump_indices() {
        use std::io::Write;
        let inp = std::env::var("PA_DUMP_IN").expect("PA_DUMP_IN");
        let outp = std::env::var("PA_DUMP_OUT").expect("PA_DUMP_OUT");
        let text = std::fs::read_to_string(&inp).unwrap();
        let mut out = std::fs::File::create(&outp).unwrap();
        for fen in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut b = Board::new();
            b.set_fen(fen);
            let stm = b.side_to_move;
            let mut row: Vec<String> = Vec::new();
            for pov in [stm, 1 - stm] {
                let ks = b.king_sq(pov) as usize ^ if pov != WHITE { 56 } else { 0 };
                let mirrored = (ks & 7) >= 4;
                let mut v = Vec::new();
                enumerate_passed_pawns(b.pieces[PAWN as usize], &b.colors, pov,
                                       mirrored, |i| v.push(i));
                v.sort_unstable();
                row.push(v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            }
            writeln!(out, "{}|{}", row[0], row[1]).unwrap();
        }
    }

    /// Mirror invariance, as for pawn-pair.
    #[test]
    fn mirror_invariance() {
        let a = feats("4k3/8/8/8/8/8/P6P/4K3 w - - 0 1", WHITE, false);
        let b = feats("4k3/8/8/8/8/8/P6P/4K3 w - - 0 1", WHITE, true);
        assert_eq!(a, b, "a file-symmetric position must be mirror-invariant");
    }
}
