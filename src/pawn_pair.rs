//! Pawn-pair input features.
//!
//! **Attribution.** The idea of a pawn-pair input block — and specifically the
//! empirical observation that, of all pawn pairs, only those at most one file
//! apart carry signal — is **Jonathan Hallström's, introduced in Pawnocchio**.
//! It has since been adopted independently by Stormphrax, Viridithas and
//! Stockfish. The idea is his; the encoding below is derived from Coda's own
//! board representation and was written without any other engine's source open.
//!
//! # Encoding (64 features)
//!
//! Pawns occupy ranks 2..7 only, so a file holds at most 6 pawn squares. For an
//! unordered pair of pawns whose files differ by at most one, canonicalise so
//! `a` is the lower square (smaller index) and describe the pair by its
//! geometry plus which colours occupy the two squares:
//!
//! * **same file** (`file_delta == 0`): rank difference `d` in `1..=5`
//!   → 5 shapes. These are doubled pawns.
//! * **adjacent file** (`file_delta == 1` after mirroring): rank difference
//!   `d` in `-5..=5` → 11 shapes. `d == 0` is a phalanx; `|d| == 1` includes
//!   the chain/attack relations; larger `|d|` are the loose/backward shapes.
//!
//! = **16 geometric shapes**, crossed with **4 colour pairings** of
//! `(colour at the lower square, colour at the upper square)` → **64 features**.
//!
//! Four pairings rather than three: distinguishing (ours, theirs) from
//! (theirs, ours) records which colour is *in front*, which is what separates a
//! supported pawn from a blockaded one. Collapsing them to an unordered
//! {ours, theirs} would merge those.
//!
//! All indices are STM-relative and mirror with the rest of the net, so the
//! block behaves like the existing feature spaces under `pov` / `mirrored`.
//!
//! Cost: 64 inputs is ~64 x ft_size extra FT weights — about 120 KB at
//! ft=1024 against a 65 MB threat block, i.e. negligible. Pawn structure also
//! changes only on pawn moves, captures, promotions and en passant, so the
//! incremental update rate is far below the threat block's.

use crate::types::{Color, WHITE};
use crate::bitboard::Bitboard;

/// Geometric shapes: 5 same-file (d in 1..=5) + 11 adjacent-file (d in -5..=5).
pub const PAWN_PAIR_SHAPES: usize = 16;
/// (lower-square colour, upper-square colour), STM-relative.
pub const PAWN_PAIR_COLOURS: usize = 4;
/// Total pawn-pair input features.
pub const PAWN_PAIR_FEATURES: usize = PAWN_PAIR_SHAPES * PAWN_PAIR_COLOURS;

/// Feature index in `0..PAWN_PAIR_FEATURES` for one pawn pair, or `None` when
/// the pair is more than one file apart.
///
/// The two cases canonicalise on DIFFERENT keys, and that is the point:
///
/// * **same file** — order by rank, so `d` in `1..=5` names the gap.
/// * **adjacent file** — order by FILE, so `d = rank(right) - rank(left)` keeps
///   its SIGN and spans `-5..=5`.
///
/// Ordering the adjacent case by square index instead would force `d >= 0`
/// (if the lower-index square had the higher rank it would not be the lower
/// index), collapsing the two chain directions onto one shape and leaving 20
/// of the 64 features unreachable. The direction is meaningful because the
/// frame is already king-mirrored, so "leaning toward the king side" and
/// "leaning away" are genuinely different structures.
#[inline]
fn pair_feature(sq_a: u8, a_mine: bool, sq_b: u8, b_mine: bool) -> Option<usize> {
    let (fa, ra) = ((sq_a % 8) as i32, (sq_a / 8) as i32);
    let (fb, rb) = ((sq_b % 8) as i32, (sq_b / 8) as i32);
    if (fb - fa).abs() > 1 {
        return None;
    }
    let (shape, lo_mine, hi_mine) = if fa == fb {
        let (d, lo_mine, hi_mine) = if ra < rb {
            (rb - ra, a_mine, b_mine)
        } else {
            (ra - rb, b_mine, a_mine)
        };
        if !(1..=5).contains(&d) {
            return None;
        }
        ((d - 1) as usize, lo_mine, hi_mine)
    } else {
        let (d, l_mine, r_mine) = if fa < fb {
            (rb - ra, a_mine, b_mine)
        } else {
            (ra - rb, b_mine, a_mine)
        };
        if !(-5..=5).contains(&d) {
            return None;
        }
        (5 + (d + 5) as usize, l_mine, r_mine)
    };
    Some(shape * PAWN_PAIR_COLOURS + (lo_mine as usize) * 2 + (hi_mine as usize))
}

/// Enumerate active pawn-pair features from `pov`'s perspective.
///
/// Mirrors the `enumerate_threats` contract: pushes each active feature index
/// to `callback`. Indices are in `0..PAWN_PAIR_FEATURES`; callers offset them
/// into whatever combined space they use.
pub fn enumerate_pawn_pairs<F: FnMut(usize)>(
    pawns_bb: Bitboard,
    colors_bb: &[Bitboard; 2],
    pov: Color,
    mirrored: bool,
    mut callback: F,
) {
    // Collect pawn squares once, STM-relative. Vertical flip for a black POV
    // and horizontal mirror for a king on the queenside both preserve pair
    // geometry, so applying them per-square before pairing is sound.
    let mut sqs: [u8; 16] = [0; 16];
    let mut n = 0usize;
    let mut bb = pawns_bb;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if n == sqs.len() { break; } // 16 pawns is the legal maximum
        let mut s = sq;
        if pov != WHITE { s ^= 56; }      // vertical flip
        if mirrored { s ^= 7; }           // horizontal mirror
        sqs[n] = s;
        n += 1;
    }
    // Colour of each collected square, in the same STM-relative order.
    let ours = colors_bb[pov as usize];
    let mut mine: [bool; 16] = [false; 16];
    let mut bb2 = pawns_bb;
    let mut i = 0usize;
    while bb2 != 0 && i < n {
        let sq = bb2.trailing_zeros();
        bb2 &= bb2 - 1;
        mine[i] = (ours >> sq) & 1 == 1;
        i += 1;
    }

    for a in 0..n {
        for b in (a + 1)..n {
            if let Some(idx) = pair_feature(sqs[a], mine[a], sqs[b], mine[b]) {
                callback(idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::types::{BLACK, PAWN};

    fn feats(fen: &str, pov: Color, mirrored: bool) -> Vec<usize> {
        let mut b = Board::new();
        b.set_fen(fen);
        let mut v = Vec::new();
        enumerate_pawn_pairs(b.pieces[PAWN as usize], &b.colors, pov, mirrored, |i| v.push(i));
        v.sort_unstable();
        v
    }

    /// Every emitted index must be in range — a violation would corrupt the FT.
    #[test]
    fn indices_in_range() {
        for fen in [
            "8/pppppppp/8/8/8/8/PPPPPPPP/8 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/1p1p1p1p/p1p1p1p1/8/8/P1P1P1P1/1P1P1P1P/8 w - - 0 1",
        ] {
            for pov in [WHITE, BLACK] {
                for m in [false, true] {
                    for i in feats(fen, pov, m) {
                        assert!(i < PAWN_PAIR_FEATURES, "index {i} out of range for {fen}");
                    }
                }
            }
        }
    }

    /// Only pairs within one file may be emitted. Two pawns three files apart
    /// must produce nothing.
    #[test]
    fn distant_files_excluded() {
        assert!(feats("8/8/8/8/8/8/P3P3/8 w - - 0 1", WHITE, false).is_empty(),
                "pawns 4 files apart must not pair");
        assert_eq!(feats("8/8/8/8/8/8/PP6/8 w - - 0 1", WHITE, false).len(), 1,
                   "adjacent-file pawns must pair exactly once");
        assert_eq!(feats("8/8/8/8/8/P7/P7/8 w - - 0 1", WHITE, false).len(), 1,
                   "doubled pawns must pair exactly once");
    }

    /// A phalanx (side by side, same rank) and a doubled pair must land on
    /// DIFFERENT shapes — collapsing them would lose the distinction the block
    /// exists to capture.
    #[test]
    fn phalanx_and_doubled_are_distinct() {
        let phalanx = feats("8/8/8/8/8/8/PP6/8 w - - 0 1", WHITE, false);
        let doubled = feats("8/8/8/8/8/P7/P7/8 w - - 0 1", WHITE, false);
        assert_eq!(phalanx.len(), 1);
        assert_eq!(doubled.len(), 1);
        assert_ne!(phalanx[0], doubled[0], "phalanx and doubled must differ");
    }

    /// Which colour is IN FRONT must change the feature — this is the whole
    /// reason for 4 colour pairings rather than 3.
    #[test]
    fn front_colour_matters() {
        // white pawn a2, black pawn a3  vs  black pawn a2, white pawn a3
        let wb = feats("8/8/8/8/8/p7/P7/8 w - - 0 1", WHITE, false);
        let bw = feats("8/8/8/8/8/P7/p7/8 w - - 0 1", WHITE, false);
        assert_eq!(wb.len(), 1);
        assert_eq!(bw.len(), 1);
        assert_ne!(wb[0], bw[0], "swapping which colour is in front must change the feature");
    }

    /// Mirroring the board horizontally and asking for the mirrored view must
    /// give the identical feature multiset — otherwise the block does not
    /// mirror with the rest of the net and king-side/queen-side positions
    /// train as unrelated.
    #[test]
    fn mirror_invariance() {
        let a = feats("8/8/8/8/8/8/PP4P1/8 w - - 0 1", WHITE, false);
        let b = feats("8/8/8/8/8/8/1P4PP/8 w - - 0 1", WHITE, true);
        assert_eq!(a, b, "mirrored position under mirrored=true must match");
    }

    /// The two chain directions must be DIFFERENT features. Canonicalising
    /// adjacent-file pairs by square index instead of by file collapses them
    /// and leaves the 20 `d < 0` features permanently unreachable, which is
    /// exactly the bug the feature-usage histogram caught.
    #[test]
    fn chain_directions_are_distinct() {
        let up_right = feats("8/8/8/8/8/2P5/1P6/8 w - - 0 1", WHITE, false); // b2,c3
        let up_left  = feats("8/8/8/8/8/1P6/2P5/8 w - - 0 1", WHITE, false); // c2,b3
        assert_eq!(up_right.len(), 1);
        assert_eq!(up_left.len(), 1);
        assert_ne!(up_right[0], up_left[0],
                   "a chain leaning right must differ from one leaning left");
    }

    /// Every one of the 16 shapes must be reachable. An unreachable shape is
    /// 4 dead features x ft_size weights and a silently mis-specified block.
    #[test]
    fn all_shapes_reachable() {
        let mut seen = [false; PAWN_PAIR_SHAPES];
        // Walk every legal pawn-square pair directly through the encoder.
        for a in 8u8..56 {
            for b in 8u8..56 {
                if a == b { continue; }
                if let Some(idx) = pair_feature(a, true, b, true) {
                    seen[idx / PAWN_PAIR_COLOURS] = true;
                }
            }
        }
        let dead: Vec<usize> = (0..PAWN_PAIR_SHAPES).filter(|&s| !seen[s]).collect();
        assert!(dead.is_empty(), "unreachable shapes: {dead:?}");
    }

    /// Cross-check dump: writes `stm-indices | ntm-indices` per FEN so the
    /// bullet trainer's enumeration can be diffed against this one. A silent
    /// index mismatch between trainer and engine is the expensive failure
    /// mode here, so it is checked mechanically rather than reasoned about.
    /// Env-gated: `PP_DUMP_IN=<fens> PP_DUMP_OUT=<path> cargo test --release
    /// pawn_pair::tests::dump_indices -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_indices() {
        use std::io::Write;
        let inp = std::env::var("PP_DUMP_IN").expect("PP_DUMP_IN");
        let outp = std::env::var("PP_DUMP_OUT").expect("PP_DUMP_OUT");
        let text = std::fs::read_to_string(&inp).unwrap();
        let mut out = std::fs::File::create(&outp).unwrap();
        for fen in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut b = Board::new();
            b.set_fen(fen);
            let stm = b.side_to_move;
            let ntm = 1 - stm;
            let mut row: Vec<String> = Vec::new();
            for pov in [stm, ntm] {
                // Mirror is decided by the perspective's OWN king, after the
                // vertical flip — same rule as `halfka_index`.
                let ks = b.king_sq(pov) as usize ^ if pov != WHITE { 56 } else { 0 };
                let mirrored = (ks & 7) >= 4;
                let mut v = Vec::new();
                enumerate_pawn_pairs(
                    b.pieces[PAWN as usize], &b.colors, pov, mirrored, |i| v.push(i));
                v.sort_unstable();
                row.push(v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            }
            writeln!(out, "{}|{}", row[0], row[1]).unwrap();
        }
    }

    /// Same structure with colours swapped, viewed from the other POV, must
    /// give the identical multiset — the block must be STM-relative.
    #[test]
    fn pov_symmetry() {
        let w = feats("8/8/8/8/8/8/PP6/8 w - - 0 1", WHITE, false);
        let b = feats("8/6pp/8/8/8/8/8/8 w - - 0 1", BLACK, false);
        assert_eq!(w, b, "colour-and-rank-flipped position from BLACK must match WHITE's");
    }
}
