//! Pawn-pair input features.
//!
//! Pairwise pawn-structure inputs are now a common cross-engine pattern. We
//! believe the idea originates from Jonathan Hallström's Pawnocchio engine —
//! the concept also appeared in Stockfish, Triumviratus and Stormphrax before
//! Coda.
//!
//! # Encoding (1,488 features)
//!
//! Every unordered pair of pawn SQUARES whose files differ by at most one gets
//! its own learned weight, crossed with which colour occupies each square:
//!
//! * 372 square pairs over ranks 2..7 within one file
//! * x 4 colour combinations of `(colour at lower square, colour at upper
//!   square)`, STM-relative
//! * = **1,488 features**, 1.5 MB of int8 weights at ft=1024
//!
//! Four colour combinations rather than three: separating (ours, theirs) from
//! (theirs, ours) records which colour is IN FRONT, which is what distinguishes
//! a supported pawn from a blockaded one.
//!
//! All indices are STM-relative and mirror with the rest of the net, so the
//! block behaves like the existing feature spaces under `pov` / `mirrored`.
//!
//! Cost is set by the number of ACTIVE features, not the table size: about 18
//! pairs are live in a typical position and a pawn move changes roughly four of
//! them, so the per-node work is independent of how finely the pairs are
//! indexed. Pawn structure also only changes on pawn moves, captures,
//! promotions and en passant.

use crate::types::{Color, Move, WHITE, PAWN, FLAG_EN_PASSANT,
                   move_from, move_to, move_flags, is_promotion, flip_color};
use crate::bitboard::Bitboard;

/// Colour combinations per square pair: (colour at the lower square, colour at
/// the upper square), STM-relative.
pub const PAWN_PAIR_COLOURS: usize = 4;

/// Slot table: `PAIR_SLOT[a][b]` is the square-pair ordinal for pawn squares
/// `a` and `b`, or -1 when the pair is not within one file (or either square
/// cannot hold a pawn).
///
/// Built over ranks 2..7 only, so the table is dense over exactly the legal
/// pawn placements.
const PAIR_SLOT: [[i16; 64]; 64] = build_slots(false);

/// Slot table for the TOUCHING encoding: only pawns a king's step apart
/// (`|dfile| <= 1 && |drank| <= 1`), ranks 2..7. Same lower-square-first
/// ordering as `PAIR_SLOT`, so the two encodings differ only in which pairs
/// exist.
const PAIR_SLOT_TOUCH: [[i16; 64]; 64] = build_slots(true);

const fn build_slots(touching: bool) -> [[i16; 64]; 64] {
    let mut t = [[-1i16; 64]; 64];
    let mut n = 0i16;
    let mut a = 8usize;
    while a < 56 {
        let mut b = a + 1;
        while b < 56 {
            let fa = (a % 8) as i32;
            let fb = (b % 8) as i32;
            let df = if fa > fb { fa - fb } else { fb - fa };
            let dr = (b / 8) as i32 - (a / 8) as i32; // b > a, so >= 0
            if df <= 1 && (!touching || dr <= 1) {
                t[a][b] = n;
                t[b][a] = n;
                n += 1;
            }
            b += 1;
        }
        a += 1;
    }
    t
}

/// Distinct square pairs within one file, over ranks 2..7.
pub const PAWN_PAIR_SQUARE_PAIRS: usize = 372;

/// Total pawn-pair input features (the default, all-distances encoding).
pub const PAWN_PAIR_FEATURES: usize = PAWN_PAIR_SQUARE_PAIRS * PAWN_PAIR_COLOURS;

/// Square pairs a king's step apart over ranks 2..7: 42 side by side, 40 one
/// rank apart on a file, 70 diagonal.
pub const PAWN_PAIR_SQUARE_PAIRS_TOUCHING: usize = 152;

/// Total pawn-pair input features in the touching encoding.
pub const PAWN_PAIR_FEATURES_TOUCHING: usize = PAWN_PAIR_SQUARE_PAIRS_TOUCHING * PAWN_PAIR_COLOURS;

/// Which pawn-pair encoding the loaded net uses: 0 = all distances (1,488
/// features), 1 = touching only (608). Process-wide, like the threat feature
/// space: set once by the net loader from the feature count the net declares,
/// so the count is the single source of truth and no header flag is needed.
#[cfg(not(test))]
static ENCODING_TOUCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// Tests run on parallel threads, and a process-wide switch would let a
// touching-encoding test change the encoding under every other pawn-pair test.
// Each test thread gets its own copy instead (default: all distances).
#[cfg(test)]
thread_local! {
    static ENCODING_TOUCHING_TL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Select the encoding for a net that declares `n` pawn-pair features.
/// Errors on a count that matches neither encoding.
pub fn set_encoding_for_count(n: usize) -> Result<(), String> {
    let touching = match n {
        PAWN_PAIR_FEATURES => false,
        PAWN_PAIR_FEATURES_TOUCHING => true,
        _ => return Err(format!(
            "net declares {} pawn-pair features; this build implements {} (all distances) \
             or {} (touching)", n, PAWN_PAIR_FEATURES, PAWN_PAIR_FEATURES_TOUCHING)),
    };
    #[cfg(not(test))]
    ENCODING_TOUCHING.store(touching, std::sync::atomic::Ordering::Relaxed);
    #[cfg(test)]
    ENCODING_TOUCHING_TL.with(|c| c.set(touching));
    Ok(())
}

#[inline]
fn touching() -> bool {
    #[cfg(not(test))]
    { ENCODING_TOUCHING.load(std::sync::atomic::Ordering::Relaxed) }
    #[cfg(test)]
    { ENCODING_TOUCHING_TL.with(|c| c.get()) }
}

/// Feature index in `0..PAWN_PAIR_FEATURES` for one pawn pair, or `None` when
/// the pair is more than one file apart.
///
/// Indexed by the two pawns' ACTUAL SQUARES, not by the pair's relative shape.
/// A shape-only encoding cannot separate a phalanx on rank 2 from one on rank
/// 6. Measured here, such a block was active in every position and carried
/// ~105cp of eval influence while improving eval accuracy by nothing (#3381,
/// #3382). Square indexing costs no extra work per node -- the same pairs are
/// active, only the weight table is larger -- and measured +2.6 STC (#3397) /
/// +9.3 LTC (#3399).
///
/// Ordering by square index needs no special case for the two-file geometry:
/// (a2,b3) and (b2,a3) are different square pairs, so a chain leaning one way
/// is naturally distinct from one leaning the other.
#[inline]
fn pair_feature(sq_a: u8, a_mine: bool, sq_b: u8, b_mine: bool) -> Option<usize> {
    pair_feature_in(touching(), sq_a, a_mine, sq_b, b_mine)
}

#[inline]
fn pair_feature_in(touching: bool, sq_a: u8, a_mine: bool, sq_b: u8, b_mine: bool) -> Option<usize> {
    let (lo, hi, lo_mine, hi_mine) = if sq_a < sq_b {
        (sq_a, sq_b, a_mine, b_mine)
    } else {
        (sq_b, sq_a, b_mine, a_mine)
    };
    let table = if touching { &PAIR_SLOT_TOUCH } else { &PAIR_SLOT };
    let slot = table[lo as usize][hi as usize];
    if slot < 0 {
        return None;
    }
    Some(slot as usize * PAWN_PAIR_COLOURS + (lo_mine as usize) * 2 + (hi_mine as usize))
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

/// One pawn-pair change: a pair of pawn squares with their colours, and
/// whether the pair is being added or removed.
///
/// Stored in PHYSICAL squares and ABSOLUTE colours, exactly like
/// `RawThreatDelta`, so one delta serves both perspectives — the transform and
/// the ours/theirs decision happen at apply time. Packed into 15 bits:
/// `a_sq(6) | b_sq(6) | a_col(1) | b_col(1) | add(1)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PawnPairDelta(u32);

impl PawnPairDelta {
    /// Filler for fixed-size delta buffers. Never applied: buffers are read
    /// only up to their recorded length.
    pub const ZERO: Self = Self(0);

    #[inline]
    pub fn new(a_sq: u8, a_col: Color, b_sq: u8, b_col: Color, add: bool) -> Self {
        Self((a_sq as u32)
            | ((b_sq as u32) << 6)
            | ((a_col as u32) << 12)
            | ((b_col as u32) << 13)
            | ((add as u32) << 14))
    }
    #[inline] pub fn a_sq(self) -> u8 { (self.0 & 63) as u8 }
    #[inline] pub fn b_sq(self) -> u8 { ((self.0 >> 6) & 63) as u8 }
    #[inline] pub fn a_col(self) -> Color { ((self.0 >> 12) & 1) as Color }
    #[inline] pub fn b_col(self) -> Color { ((self.0 >> 13) & 1) as Color }
    #[inline] pub fn add(self) -> bool { (self.0 >> 14) & 1 == 1 }
}

/// Upper bound on pawn-pair deltas from a single move.
///
/// A move removes at most two pawns (the mover and an captured pawn) and adds
/// at most one. Squares within one file of a given square span three files of
/// six pawn ranks, so each changed pawn pairs with at most 17 others:
/// `3 * 17 = 51`. Rounded to 64.
pub const MAX_PAWN_PAIR_DELTAS: usize = 64;

/// Files within one of file `f`, as a bitboard mask.
///
/// Generation uses this instead of walking all 16 pawns: a changed pawn can
/// only pair with pawns on three files, which is typically four or five pawns
/// rather than sixteen, and needs no array at all. Walking the full set cost
/// ~4.5% NPS on its own.
const FILE3: [Bitboard; 8] = {
    const FILE_A: Bitboard = 0x0101_0101_0101_0101;
    let mut t = [0u64; 8];
    let mut f = 0usize;
    while f < 8 {
        let mut m = FILE_A << f;
        if f > 0 { m |= FILE_A << (f - 1); }
        if f < 7 { m |= FILE_A << (f + 1); }
        t[f] = m;
        f += 1;
    }
    t
};

/// King-step neighbourhood of each square, as a bitboard. The touching
/// encoding generates candidates from this instead of `FILE3`, so a changed
/// pawn is paired with at most 8 others rather than up to 17 -- the row
/// saving is the point of the encoding, so it must not be spent re-filtering.
const NEIGHBOURS: [Bitboard; 64] = {
    let mut t = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let f = (sq % 8) as i32;
        let r = (sq / 8) as i32;
        let mut m = 0u64;
        let mut dr = -1i32;
        while dr <= 1 {
            let mut df = -1i32;
            while df <= 1 {
                let (nf, nr) = (f + df, r + dr);
                if (dr != 0 || df != 0) && nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    m |= 1u64 << (nr * 8 + nf);
                }
                df += 1;
            }
            dr += 1;
        }
        t[sq] = m;
        sq += 1;
    }
    t
};

/// Candidate partners of a pawn on `sq` under the active encoding.
#[inline]
fn partner_mask(sq: u8) -> Bitboard {
    if touching() { NEIGHBOURS[sq as usize] } else { FILE3[(sq % 8) as usize] }
}

/// Emit the pawn-pair deltas for `mv`, given the pawn state BEFORE it.
///
/// Called from two places that must agree byte-for-byte: eagerly in
/// `Board::make_move` before any piece mutation, and lazily from the
/// accumulator's backward walk over `PieceState`. Sharing one generator is what
/// makes those two paths identical by construction rather than by testing.
///
/// `pt` is the moving piece type, read from the pre-move mailbox.
pub fn push_pawn_pair_deltas(
    out: &mut Vec<PawnPairDelta>,
    pawns_bb: Bitboard,
    colors_bb: &[Bitboard; 2],
    us: Color,
    mv: Move,
    captured: u8,
    pt: u8,
) {
    out.clear();
    let from = move_from(mv);
    let to = move_to(mv);
    let them = flip_color(us);
    let is_ep = move_flags(mv) == FLAG_EN_PASSANT;

    // Which pawns leave the pawn set, and which join it. A promotion removes
    // the pawn from `from` and adds NOTHING — the piece that lands on `to` is
    // no longer a pawn.
    let mut removed: [(u8, Color); 2] = [(0, 0); 2];
    let mut n_removed = 0usize;
    if is_ep {
        let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
        removed[n_removed] = (cap_sq, them);
        n_removed += 1;
    } else if captured == PAWN {
        removed[n_removed] = (to, them);
        n_removed += 1;
    }
    if pt == PAWN {
        removed[n_removed] = (from, us);
        n_removed += 1;
    }
    let added: Option<(u8, Color)> =
        if pt == PAWN && !is_promotion(mv) { Some((to, us)) } else { None };

    if n_removed == 0 && added.is_none() {
        return; // no pawn moved and no pawn was captured -- the block is untouched
    }

    // Working pawn set, mutated as pawns leave and join. Processing removals
    // against a SHRINKING set is what keeps pawn-takes-pawn correct: the pair
    // between the two departing pawns is emitted exactly once, when the first
    // of them is removed.
    let mut pawns = pawns_bb;
    let white = colors_bb[WHITE as usize];

    for k in 0..n_removed {
        let (rs, rc) = removed[k];
        pawns &= !(1u64 << rs);
        let mut cand = pawns & partner_mask(rs);
        while cand != 0 {
            let s = cand.trailing_zeros() as u8;
            cand &= cand - 1;
            let sc = if (white >> s) & 1 == 1 { WHITE } else { 1 - WHITE };
            out.push(PawnPairDelta::new(rs, rc, s, sc, false));
        }
    }

    if let Some((a_sq, a_col)) = added {
        let mut cand = pawns & partner_mask(a_sq);
        while cand != 0 {
            let s = cand.trailing_zeros() as u8;
            cand &= cand - 1;
            let sc = if (white >> s) & 1 == 1 { WHITE } else { 1 - WHITE };
            out.push(PawnPairDelta::new(a_sq, a_col, s, sc, true));
        }
    }
}

#[inline]
pub fn pp_index_for(d: PawnPairDelta, pov: Color, mirrored: bool) -> Option<usize> {
    let (mut a, mut b) = (d.a_sq(), d.b_sq());
    if pov != WHITE { a ^= 56; b ^= 56; }
    if mirrored { a ^= 7; b ^= 7; }
    pair_feature(a, d.a_col() == pov, b, d.b_col() == pov)
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

    /// Every square-pair slot must be reachable, and the declared count must
    /// match what the table actually builds. A mismatch would silently leave
    /// dead weight rows or index past the table.
    #[test]
    fn every_square_pair_reachable() {
        let mut seen = [false; PAWN_PAIR_SQUARE_PAIRS];
        for a in 8u8..56 {
            for b in 8u8..56 {
                if a == b { continue; }
                if let Some(idx) = pair_feature(a, true, b, true) {
                    seen[idx / PAWN_PAIR_COLOURS] = true;
                }
            }
        }
        let dead: Vec<usize> =
            (0..PAWN_PAIR_SQUARE_PAIRS).filter(|&s| !seen[s]).collect();
        assert!(dead.is_empty(), "unreachable square pairs: {} of {}",
                dead.len(), PAWN_PAIR_SQUARE_PAIRS);
    }

    /// The square pair must encode POSITION, not just relative shape. This is
    /// the property whose absence made the first version of this block inert:
    /// a phalanx on rank 2 and one on rank 6 must be different features.
    #[test]
    fn same_shape_different_rank_is_a_different_feature() {
        let low  = feats("4k3/8/8/8/8/8/PP6/4K3 w - - 0 1", WHITE, false); // a2,b2
        let high = feats("4k3/8/PP6/8/8/8/8/4K3 w - - 0 1", WHITE, false); // a6,b6
        assert_eq!(low.len(), 1);
        assert_eq!(high.len(), 1);
        assert_ne!(low[0], high[0],
                   "same shape on different ranks must be different features");

        // ... and the same shape on a different FILE too.
        let far = feats("4k3/8/8/8/8/8/5PP1/4K3 w - - 0 1", WHITE, false); // f2,g2
        assert_ne!(low[0], far[0],
                   "same shape on different files must be different features");
    }

    /// THE invariant: applying a move's deltas to the parent's feature
    /// multiset must reproduce a full fresh enumeration, exactly, for both
    /// perspectives. Everything else about the block can be right and this
    /// still wrong, and the failure is silent — a net that evaluates a
    /// slightly wrong position on most nodes.
    ///
    /// Modelled on `threats::lazy_deltas_match_eager_generation`, including
    /// its corpus and its bias toward promotions/EP, which are exactly the
    /// moves where the pawn SET changes in a way a naive "move one pawn" delta
    /// would get wrong.
    #[test]
    fn deltas_reproduce_full_enumeration() {
        use crate::movegen::generate_legal_moves;
        use crate::types::{move_flags, is_promotion, FLAG_EN_PASSANT, Move};
        crate::init();

        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 3",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }
        fn mirror_of(b: &Board, pov: Color) -> bool { (b.king_sq(pov) % 8) >= 4 }
        fn counts(b: &Board, pov: Color) -> [i32; PAWN_PAIR_FEATURES] {
            let mut c = [0i32; PAWN_PAIR_FEATURES];
            enumerate_pawn_pairs(b.pieces[PAWN as usize], &b.colors, pov,
                                 mirror_of(b, pov), |i| c[i] += 1);
            c
        }

        let (mut checked, mut with_deltas, mut promos, mut eps, mut pawn_caps) = (0, 0, 0, 0, 0);
        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..20 {
                let seed = 0x9E37_79B9u32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };
                let mut board = Board::new();
                board.set_fen(fen);

                for _ply in 0..120 {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    // Bias toward pawn moves: uniform play spends most of its
                    // time on piece shuffles, where the delta list is empty and
                    // the test proves nothing.
                    let mut interesting: Vec<Move> = Vec::new();
                    for i in 0..legal.len {
                        let m = legal.get(i);
                        let f = move_from(m);
                        if board.mailbox[f as usize] == PAWN
                            || board.mailbox[move_to(m) as usize] == PAWN
                        {
                            interesting.push(m);
                        }
                    }
                    let mv = if !interesting.is_empty() && next_u32(&mut rng) % 4 != 0 {
                        interesting[(next_u32(&mut rng) as usize) % interesting.len()]
                    } else {
                        legal.get((next_u32(&mut rng) as usize) % legal.len)
                    };

                    let us = board.side_to_move;
                    let to = move_to(mv);
                    let is_ep = move_flags(mv) == FLAG_EN_PASSANT;
                    let captured = if is_ep { PAWN } else { board.mailbox[to as usize] };
                    let pt = board.mailbox[move_from(mv) as usize];
                    if is_promotion(mv) { promos += 1; }
                    if is_ep { eps += 1; }
                    if captured == PAWN { pawn_caps += 1; }

                    let before: [[i32; PAWN_PAIR_FEATURES]; 2] =
                        [counts(&board, WHITE), counts(&board, 1 - WHITE)];
                    let mirror_before = [mirror_of(&board, WHITE), mirror_of(&board, 1 - WHITE)];

                    let mut deltas = Vec::new();
                    push_pawn_pair_deltas(&mut deltas, board.pieces[PAWN as usize],
                                          &board.colors, us, mv, captured, pt);
                    assert!(deltas.len() <= MAX_PAWN_PAIR_DELTAS,
                        "delta overflow: {} > {}", deltas.len(), MAX_PAWN_PAIR_DELTAS);
                    if !deltas.is_empty() { with_deltas += 1; }

                    if !board.make_move(mv) { break; }

                    for pov in [WHITE, 1 - WHITE] {
                        let p = pov as usize;
                        // A king crossing the file midline changes the mirror,
                        // which invalidates deltas for that perspective — the
                        // accumulator refreshes instead. Skip, as it would.
                        if mirror_of(&board, pov) != mirror_before[p] { continue; }
                        let mirrored = mirror_before[p];
                        let mut got = before[p];
                        for d in &deltas {
                            let (mut a, mut b) = (d.a_sq(), d.b_sq());
                            if pov != WHITE { a ^= 56; b ^= 56; }
                            if mirrored { a ^= 7; b ^= 7; }
                            if let Some(i) = pair_feature(a, d.a_col() == pov,
                                                          b, d.b_col() == pov) {
                                got[i] += if d.add() { 1 } else { -1 };
                            }
                        }
                        let want = counts(&board, pov);
                        assert_eq!(got, want,
                            "delta replay != enumeration, fen {fen_idx} game {game} \
                             pov {pov} mv {mv:#06x} pt {pt} captured {captured} ep {is_ep}");
                        checked += 1;
                    }
                }
            }
        }
        // The test is worthless if the interesting cases never occurred.
        assert!(checked > 5000, "too few checks: {checked}");
        assert!(with_deltas > 2000, "too few moves actually changed pawns: {with_deltas}");
        assert!(promos > 20, "too few promotions: {promos}");
        assert!(eps > 0, "no en-passant captures exercised");
        assert!(pawn_caps > 100, "too few pawn captures: {pawn_caps}");
        println!("pawn-pair deltas verified: {checked} checks, {with_deltas} changing moves, \
                  {promos} promos, {eps} EP, {pawn_caps} pawn captures");
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
        // Black's pawns must sit on the squares that map ONTO white's after
        // the vertical flip -- a7/b7 -> a2/b2. The earlier version used g7/h7,
        // which flips to g2/h2: a different square pair, and it only passed
        // because the old encoding could not tell the two apart.
        let w = feats("8/8/8/8/8/8/PP6/8 w - - 0 1", WHITE, false);
        let b = feats("8/pp6/8/8/8/8/8/8 w - - 0 1", BLACK, false);
        assert_eq!(w, b, "colour-and-rank-flipped position from BLACK must match WHITE's");
    }

    /// The touching table holds exactly the 152 king-step square pairs over
    /// ranks 2..7, densely numbered, and is a subset of the full table.
    #[test]
    fn touching_table_shape() {
        let mut seen = vec![false; PAWN_PAIR_SQUARE_PAIRS_TOUCHING];
        let mut count = 0;
        for a in 0..64usize {
            for b in (a + 1)..64usize {
                let t = PAIR_SLOT_TOUCH[a][b];
                let df = ((a % 8) as i32 - (b % 8) as i32).abs();
                let dr = ((a / 8) as i32 - (b / 8) as i32).abs();
                let legal = (8..56).contains(&a) && (8..56).contains(&b);
                if legal && df <= 1 && dr <= 1 {
                    assert!(t >= 0, "touching pair {a},{b} missing");
                    assert!(!seen[t as usize], "slot {t} reused");
                    seen[t as usize] = true;
                    count += 1;
                    assert!(PAIR_SLOT[a][b] >= 0, "touching pair must also be in the full table");
                } else {
                    assert_eq!(t, -1, "non-touching pair {a},{b} has a slot");
                }
                assert_eq!(PAIR_SLOT_TOUCH[a][b], PAIR_SLOT_TOUCH[b][a]);
            }
        }
        assert_eq!(count, PAWN_PAIR_SQUARE_PAIRS_TOUCHING);
    }

    #[test]
    fn encoding_selected_by_feature_count() {
        assert!(set_encoding_for_count(PAWN_PAIR_FEATURES_TOUCHING).is_ok());
        assert!(touching());
        assert!(set_encoding_for_count(PAWN_PAIR_FEATURES).is_ok());
        assert!(!touching());
        assert!(set_encoding_for_count(1000).is_err());
    }

    /// Under the touching encoding, an isolated-file pair two ranks apart is
    /// not a feature, while a diagonal chain link is.
    #[test]
    fn touching_enumeration() {
        set_encoding_for_count(PAWN_PAIR_FEATURES_TOUCHING).unwrap();
        // a2 + a4: same file, two ranks apart -> nothing.
        assert!(feats("8/8/8/8/P7/8/P7/8 w - - 0 1", WHITE, false).is_empty());
        // a2 + b3: diagonal -> one feature, in range.
        let f = feats("8/8/8/8/8/1P6/P7/8 w - - 0 1", WHITE, false);
        assert_eq!(f.len(), 1);
        assert!(f[0] < PAWN_PAIR_FEATURES_TOUCHING);
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/1p1p1p1p/p1p1p1p1/8/8/P1P1P1P1/1P1P1P1P/8 w - - 0 1",
        ] {
            for pov in [WHITE, BLACK] {
                for m in [false, true] {
                    for i in feats(fen, pov, m) {
                        assert!(i < PAWN_PAIR_FEATURES_TOUCHING);
                    }
                }
            }
        }
        set_encoding_for_count(PAWN_PAIR_FEATURES).unwrap();
    }

    /// Incremental deltas must reproduce a full enumeration after every move,
    /// for both encodings, including captures, en passant and promotion.
    #[test]
    fn deltas_match_enumeration_both_encodings() {
        use crate::movegen::generate_legal_moves;
        crate::init();
        for &count in &[PAWN_PAIR_FEATURES, PAWN_PAIR_FEATURES_TOUCHING] {
            set_encoding_for_count(count).unwrap();
            for fen in [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/1pp2ppp/p1np4/3Pp3/2P1P3/1P3N2/P4PPP/R3K2R w KQkq e6 0 12",
                "8/1P1p1p2/2p1p3/pP1P1Pp1/P5P1/2P1P3/3p4/K6k w - a6 0 40",
            ] {
                let mut b = Board::new();
                b.set_fen(fen);
                let moves = generate_legal_moves(&b);
                for &mv in moves.as_slice() {
                    let us = b.side_to_move;
                    let from = crate::types::move_from(mv);
                    let to = crate::types::move_to(mv);
                    let pt = b.piece_type_at(from);
                    let captured = if crate::types::move_flags(mv) == FLAG_EN_PASSANT { PAWN }
                                   else { b.piece_type_at(to) };
                    let mut deltas = Vec::new();
                    push_pawn_pair_deltas(&mut deltas, b.pieces[PAWN as usize], &b.colors,
                                          us, mv, captured, pt);
                    for pov in [WHITE, BLACK] {
                        for m in [false, true] {
                            let mut before: Vec<i64> = Vec::new();
                            enumerate_pawn_pairs(b.pieces[PAWN as usize], &b.colors, pov, m,
                                                 |x| before.push(x as i64));
                            let mut after_b = b.clone();
                            after_b.make_move(mv);
                            let mut after: Vec<usize> = Vec::new();
                            enumerate_pawn_pairs(after_b.pieces[PAWN as usize], &after_b.colors, pov, m,
                                                 |x| after.push(x));
                            // Apply deltas to the `before` multiset.
                            let mut cur = before.clone();
                            for d in &deltas {
                                if let Some(ix) = pp_index_for(*d, pov, m) {
                                    if d.add() { cur.push(ix as i64); }
                                    else {
                                        let pos = cur.iter().position(|&y| y == ix as i64)
                                            .unwrap_or_else(|| panic!("removed absent feature {ix} ({fen}, enc {count})"));
                                        cur.swap_remove(pos);
                                    }
                                }
                            }
                            cur.sort_unstable();
                            let mut want: Vec<i64> = after.iter().map(|&x| x as i64).collect();
                            want.sort_unstable();
                            assert_eq!(cur, want, "delta mismatch: {fen} move {} enc {count}",
                                       crate::types::move_to_uci(mv));
                        }
                    }
                }
            }
        }
        set_encoding_for_count(PAWN_PAIR_FEATURES).unwrap();
    }
}
