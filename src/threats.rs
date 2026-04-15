/// Threat feature computation for NNUE.
///
/// Encodes (attacker_piece, attacker_sq, victim_piece, victim_sq) relationships.
/// Each active threat on the board contributes one feature index into the threat
/// accumulator. Feature indices are perspective-relative with king-file mirroring.
///
/// Reference: Reckless engine (src/nnue/threats.rs) — same encoding pattern.
/// Total features: ~66,864 (depends on piece-pair filtering).

use crate::attacks::*;
use crate::bitboard::*;
use crate::types::*;

/// Piece interaction map: which attacker×victim pairs are tracked.
/// Rows = attacker piece type (P/N/B/R/Q/K), columns = victim piece type.
/// -1 = excluded (not tracked). Non-negative values index into target buckets.
/// Symmetric pairs (same piece type) are semi-excluded: only one ordering kept.
const PIECE_INTERACTION_MAP: [[i32; 6]; 6] = [
    [0,  1, -1,  2, -1, -1],  // Pawn attacks:   P, N, _, R, _, _
    [0,  1,  2,  3,  4, -1],  // Knight attacks:  P, N, B, R, Q, _
    [0,  1,  2,  3, -1, -1],  // Bishop attacks:  P, N, B, R, _, _
    [0,  1,  2,  3, -1, -1],  // Rook attacks:    P, N, B, R, _, _
    [0,  1,  2,  3,  4, -1],  // Queen attacks:   P, N, B, R, Q, _
    [0,  1,  2,  3, -1, -1],  // King attacks:    P, N, B, R, _, _
];

/// Per-attacker target count (friendly + enemy combined).
/// Pawn: 3 target types × 2 sides = 6, Knight: 5 × 2 = 10, etc.
const PIECE_TARGET_COUNT: [i32; 6] = [6, 10, 8, 8, 10, 8];

/// Number of colored pieces (white P, white N, ..., black K).
const NUM_COLORED_PIECES: usize = 12;

/// Compact encoding of a piece-pair interaction.
/// Bit layout: bits 0..23 = base index, bit 30 = semi-excluded, bit 31 = excluded.
#[derive(Copy, Clone)]
struct PiecePair {
    inner: u32,
}

impl PiecePair {
    const fn new(excluded: bool, semi_excluded: bool, base: i32) -> Self {
        Self {
            inner: (((semi_excluded && !excluded) as u32) << 30)
                | ((excluded as u32) << 31)
                | ((base & 0x3FFFFFFF) as u32),
        }
    }

    /// Returns the base index with ordering correction.
    /// If semi-excluded and attacker_sq >= attacked_sq, returns negative (skip).
    const fn base(self, attacking_sq: u32, attacked_sq: u32) -> i32 {
        let below = ((attacking_sq as u8) < (attacked_sq as u8)) as u32;
        ((self.inner.wrapping_add(below << 30)) & 0x80FFFFFF) as i32
    }
}

// Static lookup tables — initialised once at startup.
static mut PIECE_PAIR_LOOKUP: [[PiecePair; NUM_COLORED_PIECES]; NUM_COLORED_PIECES] =
    [[PiecePair { inner: 0 }; NUM_COLORED_PIECES]; NUM_COLORED_PIECES];
static mut PIECE_OFFSET_LOOKUP: [[i32; 64]; NUM_COLORED_PIECES] = [[0; 64]; NUM_COLORED_PIECES];
static mut ATTACK_INDEX_LOOKUP: [[[u8; 64]; 64]; NUM_COLORED_PIECES] = [[[0; 64]; 64]; NUM_COLORED_PIECES];

/// Total number of threat features. Set during init_threats().
static mut TOTAL_THREAT_FEATURES: usize = 0;

/// Get the total threat feature count (call after init_threats).
pub fn num_threat_features() -> usize {
    unsafe { TOTAL_THREAT_FEATURES }
}

/// Colored piece index: 0=WP, 1=WN, 2=WB, 3=WR, 4=WQ, 5=WK, 6=BP, ..., 11=BK
#[inline]
pub fn colored_piece(color: Color, piece_type: u8) -> usize {
    color as usize * 6 + piece_type as usize
}

/// Piece type from colored piece index.
#[inline]
fn piece_type_of(cp: usize) -> usize {
    cp % 6
}

/// Color from colored piece index.
#[inline]
fn color_of(cp: usize) -> usize {
    cp / 6
}

/// Compute attack bitboard for a colored piece on a square (empty board for init).
fn piece_attacks_empty(cp: usize, sq: u32) -> Bitboard {
    let pt = piece_type_of(cp);
    match pt {
        0 => pawn_attacks(if color_of(cp) == 0 { WHITE } else { BLACK }, sq),
        1 => knight_attacks(sq),
        2 => bishop_attacks(sq, 0), // empty board
        3 => rook_attacks(sq, 0),
        4 => queen_attacks(sq, 0),
        5 => king_attacks(sq),
        _ => 0,
    }
}

/// Compute attack bitboard for a piece type on a square with given occupancy.
pub fn piece_attacks_occ(piece_type: u8, color: Color, sq: u32, occ: Bitboard) -> Bitboard {
    match piece_type {
        PAWN => pawn_attacks(color, sq),
        KNIGHT => knight_attacks(sq),
        BISHOP => bishop_attacks(sq, occ),
        ROOK => rook_attacks(sq, occ),
        QUEEN => queen_attacks(sq, occ),
        KING => king_attacks(sq),
        _ => 0,
    }
}

/// Initialise threat feature lookup tables. Must be called at startup.
pub fn init_threats() {
    let mut offset: i32 = 0;
    let mut piece_offset = [0i32; NUM_COLORED_PIECES];
    let mut offset_table = [0i32; NUM_COLORED_PIECES];

    // Build PIECE_OFFSET_LOOKUP: for each colored piece, for each square,
    // how many attack squares exist below this square (cumulative count).
    for color in 0..2usize {
        for pt in 0..6usize {
            let cp = color * 6 + pt;
            let mut count: i32 = 0;

            for sq in 0..64u32 {
                unsafe { PIECE_OFFSET_LOOKUP[cp][sq as usize] = count; }

                // Pawns on ranks 1 and 8 have no attacks (can't exist there)
                if pt == 0 && (sq < 8 || sq >= 56) {
                    continue;
                }

                let attacks = piece_attacks_empty(cp, sq);
                count += popcount(attacks) as i32;
            }

            piece_offset[cp] = count;
            offset_table[cp] = offset;
            offset += PIECE_TARGET_COUNT[pt] * count;
        }
    }

    unsafe { TOTAL_THREAT_FEATURES = offset as usize; }

    // Build PIECE_PAIR_LOOKUP: for each (attacker, victim) pair,
    // compute the base index and exclusion flags.
    for attacking_cp in 0..NUM_COLORED_PIECES {
        let attacking_pt = piece_type_of(attacking_cp);
        let attacking_color = color_of(attacking_cp);

        for attacked_cp in 0..NUM_COLORED_PIECES {
            let attacked_pt = piece_type_of(attacked_cp);
            let attacked_color = color_of(attacked_cp);

            let map = PIECE_INTERACTION_MAP[attacking_pt][attacked_pt];
            let base = offset_table[attacking_cp]
                + (attacked_color as i32 * (PIECE_TARGET_COUNT[attacking_pt] / 2) + map)
                    * piece_offset[attacking_cp];

            let enemy = attacking_color != attacked_color;
            let semi_excluded = attacking_pt == attacked_pt
                && (enemy || attacking_pt != 0); // pawn-pawn same color not semi-excluded
            let excluded = map < 0;

            unsafe {
                PIECE_PAIR_LOOKUP[attacking_cp][attacked_cp] =
                    PiecePair::new(excluded, semi_excluded, base);
            }
        }
    }

    // Build ATTACK_INDEX_LOOKUP: for each piece and source square,
    // the ray index of each target square within the attack set.
    for cp in 0..NUM_COLORED_PIECES {
        for from in 0..64u32 {
            let attacks = piece_attacks_empty(cp, from);

            for to in 0..64u32 {
                let below_mask = if to > 0 { (1u64 << to) - 1 } else { 0 };
                unsafe {
                    ATTACK_INDEX_LOOKUP[cp][from as usize][to as usize] =
                        popcount(below_mask & attacks) as u8;
                }
            }
        }
    }

    eprintln!("Threat features initialised: {} total", offset);
}

/// Compute a single threat feature index.
///
/// Returns negative if this pair is excluded (should be skipped).
/// `pov` is the perspective (WHITE or BLACK).
/// `mirrored` is true when the perspective king is on files e-h.
#[inline]
pub fn threat_index(
    attacker_cp: usize, // colored piece index of attacker
    mut from: u32,      // attacker square
    victim_cp: usize,   // colored piece index of victim
    mut to: u32,        // victim square
    mirrored: bool,
    pov: Color,
) -> i32 {
    // Perspective flip: rank flip for black, file flip if mirrored
    let flip = (7 * mirrored as u32) ^ (56 * pov as u32);
    from ^= flip;
    to ^= flip;

    // Remap piece colors relative to POV
    let attacking = if pov == BLACK {
        // Swap color: WP(0)↔BP(6), WN(1)↔BN(7), etc.
        (attacker_cp + 6) % 12
    } else {
        attacker_cp
    };
    let attacked = if pov == BLACK {
        (victim_cp + 6) % 12
    } else {
        victim_cp
    };

    unsafe {
        let pair = PIECE_PAIR_LOOKUP[attacking][attacked];
        let base = pair.base(from, to);
        if base < 0 {
            return base; // excluded pair
        }
        base + PIECE_OFFSET_LOOKUP[attacking][from as usize]
            + ATTACK_INDEX_LOOKUP[attacking][from as usize][to as usize] as i32
    }
}

/// Enumerate all threat features active in a position.
/// Calls `callback(feature_index)` for each active threat.
pub fn enumerate_threats<F: FnMut(usize)>(
    pieces_bb: &[Bitboard; 6],  // by piece type
    colors_bb: &[Bitboard; 2],  // by color
    mailbox: &[u8; 64],         // square → piece type (NO_PIECE_TYPE for empty)
    occ: Bitboard,
    pov: Color,
    mirrored: bool,
    mut callback: F,
) {
    let white_bb = colors_bb[WHITE as usize];

    for color in [WHITE, BLACK] {
        for pt in 0..6u8 {
            let mut piece_bb = pieces_bb[pt as usize] & colors_bb[color as usize];
            let cp = colored_piece(color, pt);

            while piece_bb != 0 {
                let sq = piece_bb.trailing_zeros();
                piece_bb &= piece_bb - 1; // clear LSB

                // Compute attacks for this piece (with occupancy for sliders)
                let attacks = piece_attacks_occ(pt, color, sq, occ);

                // Find attacked occupied squares
                let mut attacked_occ = attacks & occ;
                while attacked_occ != 0 {
                    let target_sq = attacked_occ.trailing_zeros();
                    attacked_occ &= attacked_occ - 1;

                    let victim_pt = mailbox[target_sq as usize];
                    if victim_pt >= 6 { continue; }
                    let victim_color = if white_bb & (1u64 << target_sq) != 0 { WHITE } else { BLACK };
                    let victim_cp = colored_piece(victim_color, victim_pt);

                    let idx = threat_index(cp, sq, victim_cp, target_sq, mirrored, pov);
                    if idx >= 0 {
                        callback(idx as usize);
                    }
                }
            }
        }
    }
}

/// Maximum threat deltas per ply. Reckless uses 80; we use 128 for safety.
pub const MAX_THREAT_DELTAS: usize = 128;

/// Raw threat delta: stores piece data, not pre-computed indices.
/// Indices are computed per-perspective during accumulator update.
#[derive(Copy, Clone)]
pub struct RawThreatDelta {
    pub attacker_cp: u8,
    pub from_sq: u8,
    pub victim_cp: u8,
    pub to_sq: u8,
    pub add: bool,
}

/// Compute raw threat deltas when a piece moves from `from` to `to`.
/// Must be called BEFORE the move is applied on the board (board still has old state).
/// `occ_without_dest` = occupancy with `from` removed but `to` not yet occupied.
pub fn push_threats_on_move(
    deltas: &mut Vec<RawThreatDelta>,
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    piece_color: Color,
    piece_type: u8,
    from: u32,
    to: u32,
) {
    let white_bb = colors_bb[WHITE as usize];
    let cp = colored_piece(piece_color, piece_type);
    // Use occupancy with the moving piece removed from `from` but not yet at `to`
    // This matches how Reckless handles it: occ ^ to_bb
    let occ_transit = occ ^ (1u64 << to);

    // Remove threats from old square
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ_transit, white_bb, cp, piece_color, piece_type, from, false);
    // Add threats at new square
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ_transit, white_bb, cp, piece_color, piece_type, to, true);
}

/// Compute raw threat deltas when a piece appears or disappears.
pub fn push_threats_on_change(
    deltas: &mut Vec<RawThreatDelta>,
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    piece_color: Color,
    piece_type: u8,
    square: u32,
    add: bool,
) {
    let white_bb = colors_bb[WHITE as usize];
    let cp = colored_piece(piece_color, piece_type);
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ, white_bb, cp, piece_color, piece_type, square, add);
}

/// Core: compute all threat deltas for a piece on a square.
/// Matches Reckless's push_threats_single exactly:
/// 1. Threats FROM this piece to occupied squares
/// 2. Sliders that see this square + x-ray targets behind it
/// 3. Non-sliders (pawns, knights, kings) that attack this square
fn push_threats_for_piece(
    deltas: &mut Vec<RawThreatDelta>,
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    white_bb: Bitboard,
    cp: usize,
    piece_color: Color,
    piece_type: u8,
    square: u32,
    add: bool,
) {
    // 1. Threats FROM this piece to occupied squares
    let my_attacks = piece_attacks_occ(piece_type, piece_color, square, occ);
    let mut attacked_occ = my_attacks & occ;
    while attacked_occ != 0 {
        let target_sq = attacked_occ.trailing_zeros();
        attacked_occ &= attacked_occ - 1;
        let victim_pt = mailbox[target_sq as usize];
        if victim_pt >= 6 { continue; }
        let victim_color = if white_bb & (1u64 << target_sq) != 0 { WHITE } else { BLACK };
        deltas.push(RawThreatDelta {
            attacker_cp: cp as u8, from_sq: square as u8,
            victim_cp: colored_piece(victim_color, victim_pt) as u8, to_sq: target_sq as u8, add,
        });
    }

    // 2. Sliding pieces that see this square (Reckless pattern)
    // Compute rook/bishop attacks FROM this square to find which sliders can reach it
    let rook_att = rook_attacks(square, occ);
    let bishop_att = bishop_attacks(square, occ);
    let queen_att = rook_att | bishop_att;

    let diagonal_sliders = (pieces_bb[BISHOP as usize] | pieces_bb[QUEEN as usize]) & bishop_att;
    let orthogonal_sliders = (pieces_bb[ROOK as usize] | pieces_bb[QUEEN as usize]) & rook_att;

    let mut sliders = (diagonal_sliders | orthogonal_sliders) & occ;
    while sliders != 0 {
        let slider_sq = sliders.trailing_zeros();
        sliders &= sliders - 1;
        let slider_pt = mailbox[slider_sq as usize];
        if slider_pt >= 6 { continue; }
        let slider_color = if white_bb & (1u64 << slider_sq) != 0 { WHITE } else { BLACK };
        let slider_cp = colored_piece(slider_color, slider_pt);

        // X-ray: compare slider's attacks WITH vs WITHOUT piece on this square.
        // occ may or may not include the piece at `square` (depends on transit state),
        // so always explicitly construct both cases.
        let occ_with = occ | (1u64 << square);
        let occ_without = occ & !(1u64 << square);
        let slider_att_through = piece_attacks_occ(slider_pt, slider_color, slider_sq, occ_without);
        let slider_att_blocked = piece_attacks_occ(slider_pt, slider_color, slider_sq, occ_with);
        let revealed = slider_att_through & !slider_att_blocked & occ & queen_att;

        if revealed != 0 {
            // Take the first revealed piece (closest on the ray)
            // Use the ray direction: if slider < square, revealed pieces are > square
            let xray_sq = if slider_sq < square {
                // Ray goes upward, take lowest revealed square above `square`
                let above = revealed & !((1u64 << (square + 1)) - 1);
                if above != 0 { above.trailing_zeros() } else { 64 }
            } else {
                // Ray goes downward, take highest revealed square below `square`
                let below = revealed & ((1u64 << square) - 1);
                if below != 0 { 63 - below.leading_zeros() } else { 64 }
            };

            if xray_sq < 64 {
                let xpt = mailbox[xray_sq as usize];
                if xpt < 6 {
                    let xcolor = if white_bb & (1u64 << xray_sq) != 0 { WHITE } else { BLACK };
                    deltas.push(RawThreatDelta {
                        attacker_cp: slider_cp as u8, from_sq: slider_sq as u8,
                        victim_cp: colored_piece(xcolor, xpt) as u8, to_sq: xray_sq as u8, add: !add,
                    });
                }
            }
        }

        // The slider itself attacks/no longer attacks this square
        deltas.push(RawThreatDelta {
            attacker_cp: slider_cp as u8, from_sq: slider_sq as u8,
            victim_cp: cp as u8, to_sq: square as u8, add,
        });
    }

    // 3. Non-sliding pieces that attack this square
    let black_pawns = pieces_bb[PAWN as usize] & colors_bb[BLACK as usize] & pawn_attacks(WHITE, square);
    let white_pawns = pieces_bb[PAWN as usize] & colors_bb[WHITE as usize] & pawn_attacks(BLACK, square);
    let knights = pieces_bb[KNIGHT as usize] & knight_attacks(square);
    let kings = pieces_bb[KING as usize] & king_attacks(square);

    let mut non_sliders = (black_pawns | white_pawns | knights | kings) & occ;
    while non_sliders != 0 {
        let ns_sq = non_sliders.trailing_zeros();
        non_sliders &= non_sliders - 1;
        let ns_pt = mailbox[ns_sq as usize];
        if ns_pt >= 6 { continue; }
        let ns_color = if white_bb & (1u64 << ns_sq) != 0 { WHITE } else { BLACK };
        deltas.push(RawThreatDelta {
            attacker_cp: colored_piece(ns_color, ns_pt) as u8, from_sq: ns_sq as u8,
            victim_cp: cp as u8, to_sq: square as u8, add,
        });
    }
}

#[allow(dead_code)]
fn ray_between(_from: u32, _to: u32) -> Bitboard { 0 }
#[allow(dead_code)]
fn ray_beyond(_from: u32, _to: u32) -> Bitboard { 0 }

/// Compute threat deltas from post-move board state + undo info.
/// This reconstructs what changed without needing pre-move board access.
///
/// For each perspective (pov), computes adds/subs to the threat accumulator:
/// - Threats from/to the moved piece at old vs new square
/// - Threats from/to the captured piece (removed)
/// - Threats from sliders whose rays are affected by the from/to squares
/// - Threats from non-sliders that attack the from/to squares
///
/// `board` is the POST-move state. `from`, `to`, `moved_pt`, `moved_color` describe
/// the move that was just made. `captured_pt` is NO_PIECE_TYPE if no capture.
pub fn compute_move_deltas(
    deltas: &mut Vec<RawThreatDelta>,
    board_pieces: &[Bitboard; 6],
    board_colors: &[Bitboard; 2],
    board_mailbox: &[u8; 64],
    moved_color: Color,
    moved_pt: u8,
    from: u32,
    to: u32,
    captured_pt: u8,
    captured_color: Color,
) {
    let post_occ = board_colors[0] | board_colors[1];
    // Pre-move occupancy: piece was at `from` not `to`, captured piece was at `to` (for captures)
    let pre_occ = (post_occ | (1u64 << from)) & !(1u64 << to)
        | if captured_pt != NO_PIECE_TYPE { 1u64 << to } else { 0 };

    let moved_cp = colored_piece(moved_color, moved_pt);
    let white_bb_post = board_colors[WHITE as usize];

    // Helper: get victim info at a square in the POST-move board
    let victim_at_post = |sq: u32| -> Option<(u8, Color, usize)> {
        let pt = board_mailbox[sq as usize];
        if pt >= 6 { return None; }
        let color = if white_bb_post & (1u64 << sq) != 0 { WHITE } else { BLACK };
        Some((pt, color, colored_piece(color, pt)))
    };

    // 1. Remove threats FROM moved piece at old square (using pre-move occ)
    let old_attacks = piece_attacks_occ(moved_pt, moved_color, from, pre_occ);
    let mut old_targets = old_attacks & pre_occ & !(1u64 << from);
    while old_targets != 0 {
        let tsq = old_targets.trailing_zeros();
        old_targets &= old_targets - 1;
        // Target might have been the captured piece (at `to`) or a piece still on the board
        if tsq == to && captured_pt != NO_PIECE_TYPE {
            // Threat was to the captured piece
            let vcp = colored_piece(captured_color, captured_pt);
            deltas.push(RawThreatDelta { attacker_cp: moved_cp as u8, from_sq: from as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: false });
        } else if let Some((_, _, vcp)) = victim_at_post(tsq) {
            deltas.push(RawThreatDelta { attacker_cp: moved_cp as u8, from_sq: from as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: false });
        }
    }

    // 2. Add threats FROM moved piece at new square (using post-move occ)
    let new_attacks = piece_attacks_occ(moved_pt, moved_color, to, post_occ);
    let mut new_targets = new_attacks & post_occ & !(1u64 << to);
    while new_targets != 0 {
        let tsq = new_targets.trailing_zeros();
        new_targets &= new_targets - 1;
        if let Some((_, _, vcp)) = victim_at_post(tsq) {
            deltas.push(RawThreatDelta { attacker_cp: moved_cp as u8, from_sq: to as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: true });
        }
    }

    // 3. Pieces that attacked the FROM square (they lose a threat)
    // Non-sliders: pawns, knights, kings
    let from_pawn_attackers_w = board_pieces[PAWN as usize] & board_colors[WHITE as usize] & pawn_attacks(BLACK, from);
    let from_pawn_attackers_b = board_pieces[PAWN as usize] & board_colors[BLACK as usize] & pawn_attacks(WHITE, from);
    let from_knight_attackers = board_pieces[KNIGHT as usize] & knight_attacks(from);
    let from_king_attackers = board_pieces[KING as usize] & king_attacks(from);
    let mut from_ns = (from_pawn_attackers_w | from_pawn_attackers_b | from_knight_attackers | from_king_attackers) & post_occ;
    // Remove the moved piece itself (it's no longer at from)
    from_ns &= !(1u64 << to); // moved piece is now at `to`, not relevant for `from` attackers
    while from_ns != 0 {
        let asq = from_ns.trailing_zeros();
        from_ns &= from_ns - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        // This piece used to attack the moved piece at `from` — remove that threat
        deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: moved_cp as u8, to_sq: from as u8, add: false });
    }

    // Sliders that attacked through from (using pre-move occ to find them)
    let from_rook_att = rook_attacks(from, pre_occ);
    let from_bishop_att = bishop_attacks(from, pre_occ);
    let from_diag = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & from_bishop_att & post_occ;
    let from_orth = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & from_rook_att & post_occ;
    let mut from_sliders = from_diag | from_orth;
    while from_sliders != 0 {
        let asq = from_sliders.trailing_zeros();
        from_sliders &= from_sliders - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        // Remove: this slider attacked the moved piece at `from`
        deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: moved_cp as u8, to_sq: from as u8, add: false });
    }

    // 4. Pieces that attack the TO square (they gain a threat to the moved piece)
    let to_pawn_attackers_w = board_pieces[PAWN as usize] & board_colors[WHITE as usize] & pawn_attacks(BLACK, to);
    let to_pawn_attackers_b = board_pieces[PAWN as usize] & board_colors[BLACK as usize] & pawn_attacks(WHITE, to);
    let to_knight_attackers = board_pieces[KNIGHT as usize] & knight_attacks(to);
    let to_king_attackers = board_pieces[KING as usize] & king_attacks(to);
    let mut to_ns = (to_pawn_attackers_w | to_pawn_attackers_b | to_knight_attackers | to_king_attackers) & post_occ;
    to_ns &= !(1u64 << to); // exclude the moved piece attacking itself
    while to_ns != 0 {
        let asq = to_ns.trailing_zeros();
        to_ns &= to_ns - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: moved_cp as u8, to_sq: to as u8, add: true });
    }

    // Sliders that attack to (using post-move occ)
    let to_rook_att = rook_attacks(to, post_occ);
    let to_bishop_att = bishop_attacks(to, post_occ);
    let to_diag = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & to_bishop_att & post_occ;
    let to_orth = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & to_rook_att & post_occ;
    let mut to_sliders = to_diag | to_orth;
    to_sliders &= !(1u64 << to); // exclude the moved piece
    while to_sliders != 0 {
        let asq = to_sliders.trailing_zeros();
        to_sliders &= to_sliders - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: moved_cp as u8, to_sq: to as u8, add: true });
    }

    // 5. Captured piece: remove all its threats
    if captured_pt != NO_PIECE_TYPE {
        let cap_cp = colored_piece(captured_color, captured_pt);
        // Remove threats FROM captured piece
        let cap_attacks = piece_attacks_occ(captured_pt, captured_color, to, pre_occ);
        let mut cap_targets = cap_attacks & pre_occ & !(1u64 << to);
        while cap_targets != 0 {
            let tsq = cap_targets.trailing_zeros();
            cap_targets &= cap_targets - 1;
            if tsq == from {
                // Was attacking the moved piece at its old square
                deltas.push(RawThreatDelta { attacker_cp: cap_cp as u8, from_sq: to as u8, victim_cp: moved_cp as u8, to_sq: from as u8, add: false });
            } else if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: cap_cp as u8, from_sq: to as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: false });
            }
        }
        // Remove threats TO captured piece (from non-sliders)
        let cap_pawn_w = board_pieces[PAWN as usize] & board_colors[WHITE as usize] & pawn_attacks(BLACK, to);
        let cap_pawn_b = board_pieces[PAWN as usize] & board_colors[BLACK as usize] & pawn_attacks(WHITE, to);
        let cap_knights = board_pieces[KNIGHT as usize] & knight_attacks(to);
        let cap_kings = board_pieces[KING as usize] & king_attacks(to);
        let mut cap_attackers = (cap_pawn_w | cap_pawn_b | cap_knights | cap_kings) & post_occ;
        cap_attackers &= !(1u64 << to);
        while cap_attackers != 0 {
            let asq = cap_attackers.trailing_zeros();
            cap_attackers &= cap_attackers - 1;
            let apt = board_mailbox[asq as usize];
            if apt >= 6 { continue; }
            let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
            let acp = colored_piece(acolor, apt);
            deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: cap_cp as u8, to_sq: to as u8, add: false });
        }
        // Sliders that attacked captured piece
        let cap_rook_att = rook_attacks(to, pre_occ);
        let cap_bishop_att = bishop_attacks(to, pre_occ);
        let cap_diag = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & cap_bishop_att & post_occ;
        let cap_orth = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & cap_rook_att & post_occ;
        let mut cap_sliders = cap_diag | cap_orth;
        cap_sliders &= !(1u64 << to);
        while cap_sliders != 0 {
            let asq = cap_sliders.trailing_zeros();
            cap_sliders &= cap_sliders - 1;
            let apt = board_mailbox[asq as usize];
            if apt >= 6 { continue; }
            let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
            let acp = colored_piece(acolor, apt);
            deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: cap_cp as u8, to_sq: to as u8, add: false });
        }
    }

    // 6. X-ray discoveries: sliders that were blocked by the piece at `from` now
    // see through to new targets. Also, sliders that now see `to` may be blocked
    // by the piece that just arrived there.
    //
    // For the vacated `from` square: find sliders that attack through `from` in the
    // POST-move board (piece gone). Compare their attacks with pre-move attacks.
    // New targets = attacks with piece gone minus attacks with piece present.

    // Sliders that see through `from` in post-move (piece removed)
    let post_rook_from = rook_attacks(from, post_occ);
    let post_bishop_from = bishop_attacks(from, post_occ);
    let pre_rook_from = rook_attacks(from, pre_occ);
    let pre_bishop_from = bishop_attacks(from, pre_occ);

    // Orthogonal sliders (rooks + queens) near `from`
    let orth_sliders = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & post_occ;
    let diag_sliders = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & post_occ;

    // Find sliders on the same rank/file/diagonal as `from` that gain new targets
    let mut orth_near = orth_sliders & (post_rook_from | pre_rook_from);
    while orth_near != 0 {
        let asq = orth_near.trailing_zeros();
        orth_near &= orth_near - 1;
        if asq == to { continue; } // the moved piece, already handled
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = rook_attacks(asq, pre_occ) & pre_occ;
        let new_att = rook_attacks(asq, post_occ) & post_occ;

        // New targets (gained)
        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: true });
            }
        }
        // Lost targets
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: false });
            }
        }
    }

    let mut diag_near = diag_sliders & (post_bishop_from | pre_bishop_from);
    while diag_near != 0 {
        let asq = diag_near.trailing_zeros();
        diag_near &= diag_near - 1;
        if asq == to { continue; }
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = bishop_attacks(asq, pre_occ) & pre_occ;
        let new_att = bishop_attacks(asq, post_occ) & post_occ;

        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: true });
            }
        }
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: false });
            }
        }
    }

    // 7. X-ray changes at `to` square: sliders that used to see through `to`
    // (or see pieces beyond `to`) may now be blocked by the moved piece.
    // Also, if `to` was occupied (capture), sliders might have new visibility.
    // Only process sliders not already handled in sections 3-4.
    let post_rook_to = rook_attacks(to, post_occ);
    let post_bishop_to = bishop_attacks(to, post_occ);
    let pre_rook_to = rook_attacks(to, pre_occ);
    let pre_bishop_to = bishop_attacks(to, pre_occ);

    let mut orth_near_to = orth_sliders & (post_rook_to | pre_rook_to);
    orth_near_to &= !(1u64 << to); // exclude moved piece
    while orth_near_to != 0 {
        let asq = orth_near_to.trailing_zeros();
        orth_near_to &= orth_near_to - 1;
        // Skip if already processed in the `from` section
        if (post_rook_from | pre_rook_from) & (1u64 << asq) != 0 { continue; }
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = rook_attacks(asq, pre_occ) & pre_occ;
        let new_att = rook_attacks(asq, post_occ) & post_occ;

        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: true });
            }
        }
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: false });
            }
        }
    }

    let mut diag_near_to = diag_sliders & (post_bishop_to | pre_bishop_to);
    diag_near_to &= !(1u64 << to);
    while diag_near_to != 0 {
        let asq = diag_near_to.trailing_zeros();
        diag_near_to &= diag_near_to - 1;
        if (post_bishop_from | pre_bishop_from) & (1u64 << asq) != 0 { continue; }
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = bishop_attacks(asq, pre_occ) & pre_occ;
        let new_att = bishop_attacks(asq, post_occ) & post_occ;

        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: true });
            }
        }
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta { attacker_cp: acp as u8, from_sq: asq as u8, victim_cp: vcp as u8, to_sq: tsq as u8, add: false });
            }
        }
    }
}

/// Apply raw threat deltas to update the threat accumulator incrementally.
/// Copies from `prev` and applies all deltas for a specific perspective.
pub fn apply_threat_deltas(
    dst: &mut [i16],           // destination threat accumulator (one perspective)
    src: &[i16],               // source (previous position's threat accumulator)
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],     // [num_threats × hidden_size]
    hidden_size: usize,
    num_threats: usize,
    pov: Color,
    mirrored: bool,
) {
    // Copy from previous position
    dst[..hidden_size].copy_from_slice(&src[..hidden_size]);

    // Collect valid add/sub indices (stack-allocated, no heap)
    let mut adds = [0usize; MAX_THREAT_DELTAS];
    let mut subs = [0usize; MAX_THREAT_DELTAS];
    let mut n_adds = 0usize;
    let mut n_subs = 0usize;
    for delta in deltas {
        let idx = threat_index(
            delta.attacker_cp as usize,
            delta.from_sq as u32,
            delta.victim_cp as usize,
            delta.to_sq as u32,
            mirrored,
            pov,
        );
        if idx < 0 || (idx as usize) >= num_threats { continue; }
        if delta.add { adds[n_adds] = idx as usize; n_adds += 1; }
        else { subs[n_subs] = idx as usize; n_subs += 1; }
    }
    let adds = &adds[..n_adds];
    let subs = &subs[..n_subs];

    // Apply weight rows with SIMD when available
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && hidden_size % 16 == 0 {
            unsafe {
                apply_deltas_avx2(dst, threat_weights, hidden_size, &adds, &subs);
            }
            return;
        }
    }

    // Scalar fallback
    for &idx in adds {
        let w_off = idx * hidden_size;
        for j in 0..hidden_size {
            dst[j] += threat_weights[w_off + j] as i16;
        }
    }
    for &idx in subs {
        let w_off = idx * hidden_size;
        for j in 0..hidden_size {
            dst[j] -= threat_weights[w_off + j] as i16;
        }
    }
}

/// AVX2 SIMD: apply threat weight rows to accumulator using register tiling.
/// Loads accumulator chunk into registers ONCE, applies ALL deltas while in
/// registers, then stores ONCE. This is Reckless's pattern — 21× less memory
/// traffic than per-delta streaming.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_deltas_avx2(
    dst: &mut [i16],
    threat_weights: &[i8],
    hidden_size: usize,
    adds: &[usize],
    subs: &[usize],
) {
    use std::arch::x86_64::*;

    let dst_ptr = dst.as_mut_ptr();
    let w_ptr = threat_weights.as_ptr();

    // 8 AVX2 registers × 16 i16 = 128 elements per chunk
    const REGS: usize = 8;
    const CHUNK: usize = REGS * 16; // 128 elements

    let mut offset = 0;
    while offset < hidden_size {
        let chunk_size = (hidden_size - offset).min(CHUNK);
        let nregs = (chunk_size + 15) / 16;

        // Load accumulator chunk into registers
        let mut regs: [__m256i; REGS] = [_mm256_setzero_si256(); REGS];
        for i in 0..nregs {
            regs[i] = _mm256_loadu_si256(dst_ptr.add(offset + i * 16) as *const __m256i);
        }

        // Apply paired add+sub
        let mut ai = 0;
        let mut si = 0;
        while ai < adds.len() && si < subs.len() {
            let aw = w_ptr.add(adds[ai] * hidden_size + offset);
            let sw = w_ptr.add(subs[si] * hidden_size + offset);
            for i in 0..nregs {
                let add_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(aw.add(i * 16) as *const __m128i));
                let sub_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(sw.add(i * 16) as *const __m128i));
                regs[i] = _mm256_sub_epi16(_mm256_add_epi16(regs[i], add_w), sub_w);
            }
            ai += 1;
            si += 1;
        }

        // Remaining adds
        while ai < adds.len() {
            let aw = w_ptr.add(adds[ai] * hidden_size + offset);
            for i in 0..nregs {
                let add_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(aw.add(i * 16) as *const __m128i));
                regs[i] = _mm256_add_epi16(regs[i], add_w);
            }
            ai += 1;
        }

        // Remaining subs
        while si < subs.len() {
            let sw = w_ptr.add(subs[si] * hidden_size + offset);
            for i in 0..nregs {
                let sub_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(sw.add(i * 16) as *const __m128i));
                regs[i] = _mm256_sub_epi16(regs[i], sub_w);
            }
            si += 1;
        }

        // Store registers back
        for i in 0..nregs {
            _mm256_storeu_si256(dst_ptr.add(offset + i * 16) as *mut __m256i, regs[i]);
        }

        offset += CHUNK;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_threats() {
        crate::init();
        let total = num_threat_features();
        // Reckless has 66,864 — we should match
        assert!(total > 60000, "Expected >60K threat features, got {}", total);
        assert!(total < 70000, "Expected <70K threat features, got {}", total);
        eprintln!("Total threat features: {}", total);
    }

    #[test]
    fn test_threat_index_basic() {
        crate::init();

        // White knight on c3 attacks black pawn on d5
        let wn = colored_piece(WHITE, KNIGHT);
        let bp = colored_piece(BLACK, PAWN);
        let idx = threat_index(wn, 18, bp, 35, false, WHITE); // c3=18, d5=35
        assert!(idx >= 0, "WN c3 × BP d5 should be a valid threat, got {}", idx);

        // Same threat from black's perspective should give different index
        let idx_black = threat_index(wn, 18, bp, 35, false, BLACK);
        assert!(idx_black >= 0, "Should be valid from black POV too");
        assert_ne!(idx, idx_black, "Different POV should give different index");
    }

    #[test]
    fn test_excluded_pairs() {
        crate::init();

        // Pawn attacks bishop: excluded (PIECE_INTERACTION_MAP[0][2] = -1)
        let wp = colored_piece(WHITE, PAWN);
        let bb = colored_piece(BLACK, BISHOP as u8);
        let idx = threat_index(wp, 28, bb, 35, false, WHITE); // e4 → d5
        assert!(idx < 0, "Pawn×Bishop should be excluded, got {}", idx);

        // King attacks queen: excluded (PIECE_INTERACTION_MAP[5][4] = -1)
        let wk = colored_piece(WHITE, KING as u8);
        let bq = colored_piece(BLACK, QUEEN as u8);
        let idx = threat_index(wk, 4, bq, 5, false, WHITE);
        assert!(idx < 0, "King×Queen should be excluded, got {}", idx);
    }

    #[test]
    fn test_mirroring() {
        crate::init();

        // Same attack, mirrored vs not, should differ
        let wn = colored_piece(WHITE, KNIGHT);
        let bp = colored_piece(BLACK, PAWN);
        let idx_normal = threat_index(wn, 18, bp, 35, false, WHITE);
        let idx_mirror = threat_index(wn, 18, bp, 35, true, WHITE);
        assert_ne!(idx_normal, idx_mirror, "Mirrored should differ");
    }

    #[test]
    fn test_enumerate_startpos() {
        crate::init();

        // Standard starting position — count active threat features
        // In startpos, pieces attack each other across the board
        // Pawns attack nothing occupied, knights attack nothing, etc.
        // Only threats should be from pawns that are diagonal to opposing pawns (none in startpos)
        // and any other piece attacking an occupied square

        // Simplified: just verify we get a reasonable count
        let pieces_bb: [Bitboard; 6] = [
            0x00FF00000000FF00, // pawns
            0x4200000000000042, // knights
            0x2400000000000024, // bishops
            0x8100000000000081, // rooks
            0x0800000000000008, // queens
            0x1000000000000010, // kings
        ];
        let colors_bb: [Bitboard; 2] = [
            0x000000000000FFFF, // white
            0xFFFF000000000000, // black
        ];
        let occ = colors_bb[0] | colors_bb[1];

        // Build mailbox from bitboards
        let mut mailbox = [NO_PIECE_TYPE; 64];
        for sq in 0..64u32 {
            let bit = 1u64 << sq;
            if occ & bit == 0 { continue; }
            for pt in 0..6u8 {
                if pieces_bb[pt as usize] & bit != 0 {
                    mailbox[sq as usize] = pt;
                    break;
                }
            }
        }

        let mut count = 0;
        enumerate_threats(
            &pieces_bb, &colors_bb, &mailbox,
            occ, WHITE, false,
            |_idx| { count += 1; },
        );
        eprintln!("Startpos threat count (white POV): {}", count);
        // In startpos, pieces behind pawns don't attack much occupied territory.
        // But knights on b1/g1 attack no occupied squares, etc.
        // Expect a small number — mostly cross-pawn structure.
        assert!(count > 0, "Should have some threats in startpos");
        assert!(count < 200, "Shouldn't have too many threats in startpos");
    }

    #[test]
    fn test_bench_threat_enumeration() {
        crate::init();

        // Test positions: startpos + several middlegame/endgame positions
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "2r3k1/pp3ppp/2n1b3/3pP3/3P4/2NB4/PP3PPP/R4RK1 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];

        let mut total_threats = 0usize;
        let mut total_positions = 0usize;

        for fen in &fens {
            let mut board = crate::board::Board::new();
            board.set_fen(fen);
            let occ = board.colors[0] | board.colors[1];
            let king_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
            let mirrored = (king_sq % 8) >= 4;

            let mut count = 0;
            enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, mirrored,
                |_idx| { count += 1; },
            );
            eprintln!("  {} → {} threats", fen, count);
            total_threats += count;
            total_positions += 1;
        }

        eprintln!("Average threats per position: {}", total_threats / total_positions);

        // Benchmark: enumerate threats 100K times on the complex middlegame position
        let mut board = crate::board::Board::new();
        board.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let occ = board.colors[0] | board.colors[1];
        let king_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
        let mirrored = (king_sq % 8) >= 4;

        let iterations = 100_000;
        let start = std::time::Instant::now();
        let mut total = 0usize;
        for _ in 0..iterations {
            enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, mirrored,
                |_idx| { total += 1; },
            );
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() / iterations as u128;
        let calls_per_sec = if elapsed.as_secs_f64() > 0.0 {
            iterations as f64 / elapsed.as_secs_f64()
        } else { 0.0 };
        eprintln!("Threat enumeration benchmark (kiwipete, {}x):", iterations);
        eprintln!("  Total time: {:?}", elapsed);
        eprintln!("  Per call: {} ns ({:.0} K calls/sec)", per_call_ns, calls_per_sec / 1000.0);
        eprintln!("  Threats per call: {}", total / iterations);

        // Sanity: should complete in reasonable time
        assert!(elapsed.as_secs() < 10, "Benchmark took too long: {:?}", elapsed);
    }
}
