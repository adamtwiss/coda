//! Passed-pawn input features.
//!
//! **Attribution — the idea is not ours.** A passed-pawn NNUE input block is
//! Triumviratus's, who claim it as original to that project and measured it at
//! +6.96 ±6.56. Unlike the pawn-pair block it has, as far as our engine notes
//! record, no independent replication — so the prior here is weaker and the
//! measurement is what decides. The detection and encoding below are written in
//! Coda's own bitboard idiom; no foreign code, formula or constant is used.
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
//! cannot tell them apart carries influence without carrying information. That
//! failure mode cost a full cycle on pawn-pair and is not worth repeating.
//!
//! This is a *relational* property — whether a pawn is passed depends on every
//! enemy pawn on three files — so the king-bucketed piece-square features
//! cannot express it directly however wide the net gets.
//!
//! Cost is negligible: a typical position has 0-3 passers, against the
//! pawn-pair block's ~18 active features.

use crate::types::{Color, WHITE};
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

    /// Mirror invariance, as for pawn-pair.
    #[test]
    fn mirror_invariance() {
        let a = feats("4k3/8/8/8/8/8/P6P/4K3 w - - 0 1", WHITE, false);
        let b = feats("4k3/8/8/8/8/8/P6P/4K3 w - - 0 1", WHITE, true);
        assert_eq!(a, b, "a file-symmetric position must be mirror-invariant");
    }
}
