/// Threat accumulator stack (Reckless pattern).
///
/// Separate from the PSQ accumulator. Each ply has:
/// - Per-perspective i16 accumulator values (aligned)
/// - Per-perspective accuracy flags
/// - Threat deltas (ArrayVec, no heap allocation)
/// - Move info for king mirror detection
///
/// The stack is pre-allocated for MAX_PLY entries. Push/pop is trivial.
/// BoardObserver callbacks during make_move push deltas directly.
/// Evaluate walks back to find an accurate ancestor and replays forward.

use crate::threats::{RawThreatDelta, MAX_THREAT_DELTAS};
use crate::types::*;

const MAX_PLY: usize = 256;

/// Fixed-capacity array (no heap, like ArrayVec but simpler).
/// Tracks overflow so callers can force full recompute instead of
/// silently using incomplete deltas.
#[derive(Clone)]
pub struct DeltaVec {
    data: [RawThreatDelta; MAX_THREAT_DELTAS],
    len: usize,
    overflowed: bool,
}

impl DeltaVec {
    pub const fn new() -> Self {
        Self {
            data: [RawThreatDelta::ZERO; MAX_THREAT_DELTAS],
            len: 0,
            overflowed: false,
        }
    }

    #[inline]
    pub fn clear(&mut self) { self.len = 0; self.overflowed = false; }

    #[inline]
    pub fn push(&mut self, d: RawThreatDelta) {
        if self.len < MAX_THREAT_DELTAS {
            self.data[self.len] = d;
            self.len += 1;
        } else {
            self.overflowed = true;
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[RawThreatDelta] { &self.data[..self.len] }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub fn overflowed(&self) -> bool { self.overflowed }
}

/// Single threat accumulator entry (one ply).
#[repr(C, align(64))]
pub struct ThreatEntry {
    /// Per-perspective accumulator values: [WHITE][..h], [BLACK][..h]
    pub values: [[i16; 768]; 2], // max accumulator size for v9
    /// Per-perspective accuracy flags
    pub accurate: [bool; 2],
    /// Threat deltas for the move that produced this ply
    pub delta: DeltaVec,
    /// The move that produced this ply (for king mirror check)
    pub mv: Move,
    /// Piece type that moved (for king mirror detection)
    pub moved_pt: u8,
    /// Color that moved (for per-perspective king mirror check)
    pub moved_color: u8,
}

impl ThreatEntry {
    pub const fn new() -> Self {
        Self {
            values: [[0i16; 768]; 2],
            accurate: [false; 2],
            delta: DeltaVec::new(),
            mv: NO_MOVE,
            moved_pt: NO_PIECE_TYPE,
            moved_color: WHITE,
        }
    }
}

/// The threat accumulator stack.
pub struct ThreatStack {
    stack: Vec<ThreatEntry>,
    index: usize,
    hidden_size: usize,
    /// Whether threat features are active (net has threats)
    pub active: bool,
}

impl ThreatStack {
    pub fn new(hidden_size: usize) -> Self {
        let mut stack = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            stack.push(ThreatEntry::new());
        }
        Self { stack, index: 0, hidden_size, active: false }
    }

    #[inline]
    pub fn index(&self) -> usize { self.index }

    #[inline]
    pub fn current(&self) -> &ThreatEntry { &self.stack[self.index] }

    #[inline]
    pub fn current_mut(&mut self) -> &mut ThreatEntry { &mut self.stack[self.index] }

    /// Push: increment index, reset flags, clear deltas.
    /// Called BEFORE make_move (mirrors Reckless's Network::push).
    pub fn push(&mut self, mv: Move, moved_pt: u8) {
        self.index += 1;
        if self.index >= self.stack.len() {
            self.stack.push(ThreatEntry::new());
        }
        let entry = &mut self.stack[self.index];
        entry.accurate = [false; 2];
        entry.delta.clear();
        entry.mv = mv;
        entry.moved_pt = moved_pt;
    }

    /// Pop: decrement index.
    pub fn pop(&mut self) {
        debug_assert!(self.index > 0);
        self.index -= 1;
    }

    /// Reset: for new positions (between bench positions, new game).
    pub fn reset(&mut self) {
        self.index = 0;
        self.stack[0].accurate = [false; 2];
    }

    /// Force a refresh on the next `ensure_computed` — used by the
    /// eval-bench microbench to isolate threat-refresh cost.
    pub fn reset_for_bench(&mut self) {
        self.stack[self.index].accurate = [false; 2];
    }

    /// Full refresh for one perspective: zero + enumerate all threats.
    /// Collects feature indices first, then applies with SIMD.
    /// `remap` (empty = identity) drops zero-weight rows at load time; any
    /// raw feature mapped to -1 is skipped before we hit the weight matrix.
    pub fn refresh(&mut self, net_weights: &[i8], num_features: usize,
                   remap: &[i32],
                   board: &crate::board::Board, pov: Color) {
        let h = self.hidden_size;
        let entry = &mut self.stack[self.index];
        let p = pov as usize;
        entry.values[p][..h].fill(0);

        let occ = board.colors[0] | board.colors[1];
        let king_sq = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
        let mirrored = (king_sq % 8) >= 4;

        // Collect feature indices, then apply with SIMD
        let mut indices = [0usize; 256]; // max active threat features per position
        let mut n_indices = 0usize;
        // Merged from HEAD + 611ba0f:
        //  - `overflowed`: C8 audit LIKELY #18 guarantee that if the
        //    enumerator produced more features than our 256-slot buffer
        //    can hold, we mark the entry inaccurate so the caller forces
        //    a full refresh (rather than silently dropping deltas and
        //    building on a corrupted baseline).
        //  - `has_remap` / `remap` handling: load-time zero-row compact
        //    for sparse (L1-trained) nets. `remap[raw_idx] == -1` means
        //    the raw feature was dropped at load (its weight row was
        //    all-zero); skip it without touching the weight matrix.
        let mut overflowed = false;
        let has_remap = !remap.is_empty();

        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, pov, mirrored,
            |feat_idx| {
                if feat_idx >= num_features { return; }
                let row = if has_remap {
                    let r = remap[feat_idx];
                    if r < 0 { return; }  // dropped at load (zero row)
                    r as usize
                } else {
                    feat_idx
                };
                if n_indices < indices.len() {
                    indices[n_indices] = row;
                    n_indices += 1;
                } else {
                    overflowed = true;
                }
            },
        );

        // Apply all weight rows with SIMD
        crate::threats::add_weight_rows(
            &mut entry.values[p][..h], net_weights, h, &indices[..n_indices],
        );

        entry.accurate[p] = !overflowed;
    }

    /// Check if we can incrementally update this perspective by walking back.
    /// Returns Some(ancestor_index) or None (need full refresh).
    /// Matches Reckless's can_update_threats.
    #[inline]
    pub fn can_update(&self, pov: Color) -> Option<usize> {
        for i in (0..self.index).rev() {
            // Validate the move that produced entry[i+1] BEFORE accepting
            // entry[i] as an ancestor. If the move at i+1 is a king crossing
            // (changes mirror for this perspective) or has overflowed deltas,
            // we cannot replay from any ancestor at or below i — the stored
            // deltas would apply with the wrong mirror or be incomplete.
            //
            // Earlier code returned Some(i) on `accurate[i]` *before* doing
            // this check, so a king-file-crossing at the current ply slipped
            // through whenever the prior ply was accurate (the common case).
            // Caught by the threat-accumulator fuzzer on 2026-04-17.
            let entry = &self.stack[i + 1];
            if entry.mv != NO_MOVE {
                if entry.delta.overflowed() {
                    return None;
                }
                if entry.moved_pt == KING && entry.moved_color == pov {
                    let from = move_from(entry.mv);
                    let to = move_to(entry.mv);
                    if (from % 8 >= 4) != (to % 8 >= 4) {
                        // This perspective's king crossed e-file — mirroring changed.
                        return None;
                    }
                }
            }

            if self.stack[i].accurate[pov as usize] {
                return Some(i);
            }
        }
        None
    }

    /// Incremental update: replay from ancestor to current index for one perspective.
    /// Uses SIMD apply_threat_deltas for the inner loop (AVX2 register tiling).
    pub fn update(&mut self, ancestor: usize, net_weights: &[i8], num_features: usize,
                  remap: &[i32],
                  board: &crate::board::Board, pov: Color) {
        let h = self.hidden_size;
        let p = pov as usize;
        let king_sq = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
        let mirrored = (king_sq % 8) >= 4;

        for ply in (ancestor + 1)..=self.index {
            let entry_mv = self.stack[ply].mv;

            if entry_mv == NO_MOVE || self.stack[ply].delta.is_empty() {
                // Null move or no deltas: copy from previous
                let (prev, curr) = self.stack.split_at_mut(ply);
                curr[0].values[p][..h].copy_from_slice(&prev[ply - 1].values[p][..h]);
            } else {
                // Copy deltas to local buffer to avoid borrow conflict with split_at_mut
                let n_deltas = self.stack[ply].delta.len;
                let mut local_deltas = [crate::threats::RawThreatDelta::ZERO; 128];
                local_deltas[..n_deltas].copy_from_slice(&self.stack[ply].delta.data[..n_deltas]);
                // Use SIMD apply_threat_deltas (copies src + applies adds/subs)
                let (prev, curr) = self.stack.split_at_mut(ply);
                unsafe {
                    crate::threats::apply_threat_deltas(
                        &mut curr[0].values[p][..h],
                        &prev[ply - 1].values[p][..h],
                        &local_deltas[..n_deltas],
                        net_weights, h, num_features, remap,
                        pov, mirrored,
                    );
                }
            }

            self.stack[ply].accurate[p] = true;
        }
    }

    /// Get the accumulator values for a perspective.
    #[inline]
    pub fn values(&self, pov: Color) -> &[i16] {
        &self.stack[self.index].values[pov as usize][..self.hidden_size]
    }

    /// Ensure both perspectives are computed for the current position.
    /// Matches Reckless's evaluate() pattern.
    #[inline]
    pub fn ensure_computed(&mut self, net_weights: &[i8], num_features: usize,
                          remap: &[i32],
                          board: &crate::board::Board) {
        if !self.active { return; }

        let idx = self.index;
        for pov in [WHITE, BLACK] {
            if self.stack[idx].accurate[pov as usize] {
                continue;
            }

            match self.can_update(pov) {
                Some(ancestor) => self.update(ancestor, net_weights, num_features, remap, board, pov),
                None => self.refresh(net_weights, num_features, remap, board, pov),
            }
        }
    }
}

#[cfg(test)]
mod incremental_tests {
    //! Regression tests: the incremental threat update path must produce
    //! the same per-perspective accumulator as a full re-enumeration
    //! after every move. Drives two ThreatStacks side-by-side along the
    //! same move sequence — one always refreshes, the other relies on
    //! the incremental deltas path (ensure_computed → can_update → update).
    //!
    //! Failure mode this targets: capture moves that remove a blocker
    //! between a slider and a piece behind it should register a new
    //! x-ray feature, but the incremental path may miss it.
    //!
    //! Each FEN → moves sequence is a self-contained scenario. The
    //! deterministic weight pattern makes any single feature-level
    //! divergence show up as an element-wise vector diff.
    use super::*;
    use crate::board::Board;
    use crate::movegen::generate_legal_moves;
    use crate::threats::{num_threat_features, RawThreatDelta};

    const H: usize = 768;

    /// Deterministic weights: each (feature, channel) gets a distinct i8.
    /// Ensures a single-feature multiset divergence produces a visible
    /// element-wise delta in the accumulator.
    fn make_weights(num_features: usize) -> Vec<i8> {
        let mut w = vec![0i8; num_features * H];
        for idx in 0..num_features {
            for j in 0..H {
                // Mix idx and channel with primes. Mod 251 keeps values
                // in i8 range while staying well-distributed.
                let v = ((idx.wrapping_mul(7919)).wrapping_add(j.wrapping_mul(31)) % 251) as i32 - 125;
                w[idx * H + j] = v as i8;
            }
        }
        w
    }

    fn parse_uci(board: &Board, s: &str) -> Move {
        let bytes = s.as_bytes();
        assert!(bytes.len() >= 4, "bad uci: {}", s);
        let from_file = bytes[0] - b'a';
        let from_rank = bytes[1] - b'1';
        let to_file = bytes[2] - b'a';
        let to_rank = bytes[3] - b'1';
        let from = crate::types::square(from_file, from_rank);
        let to = crate::types::square(to_file, to_rank);
        let promo_flag = if bytes.len() > 4 {
            match bytes[4] {
                b'q' => Some(FLAG_PROMOTE_Q),
                b'r' => Some(FLAG_PROMOTE_R),
                b'b' => Some(FLAG_PROMOTE_B),
                b'n' => Some(FLAG_PROMOTE_N),
                _ => None,
            }
        } else { None };
        let legal = generate_legal_moves(board);
        for i in 0..legal.len {
            let mv = legal.moves[i];
            if move_from(mv) == from && move_to(mv) == to {
                if let Some(pf) = promo_flag {
                    if move_flags(mv) == pf { return mv; }
                } else if !is_promotion(mv) {
                    return mv;
                }
            }
        }
        panic!("no legal move {} in position", s);
    }

    /// Copy board.threat_deltas into the current ThreatStack entry and
    /// record the move metadata (replicates what search.rs does post-make_move).
    fn absorb_deltas(ts: &mut ThreatStack, board: &mut Board) {
        let entry = ts.current_mut();
        entry.delta.clear();
        for d in board.threat_deltas.iter() { entry.delta.push(*d); }
        let ul = board.undo_stack.len();
        if ul > 0 {
            let u = &board.undo_stack[ul - 1];
            entry.mv = u.mv;
            if u.mv != NO_MOVE {
                entry.moved_pt = board.mailbox[move_to(u.mv) as usize];
                entry.moved_color = crate::types::flip_color(board.side_to_move);
            }
        }
    }

    /// Run the scenario: play each UCI move, verifying after every ply
    /// that incremental == full-refresh for both perspectives.
    fn run_scenario(name: &str, fen: &str, moves: &[&str]) {
        crate::init();
        let nf = num_threat_features();
        let weights = make_weights(nf);

        let mut board = Board::new();
        board.set_fen(fen);
        board.generate_threat_deltas = true;

        let remap: &[i32] = &[];
        let mut incr = ThreatStack::new(H);
        incr.active = true;
        incr.refresh(&weights, nf, remap, &board, WHITE);
        incr.refresh(&weights, nf, remap, &board, BLACK);

        let mut refs = ThreatStack::new(H);
        refs.active = true;
        refs.refresh(&weights, nf, remap, &board, WHITE);
        refs.refresh(&weights, nf, remap, &board, BLACK);

        // Sanity: both start identical.
        assert_eq!(incr.values(WHITE), refs.values(WHITE), "{}: baseline W mismatch", name);
        assert_eq!(incr.values(BLACK), refs.values(BLACK), "{}: baseline B mismatch", name);

        for (ply, uci) in moves.iter().enumerate() {
            let mv = parse_uci(&board, uci);

            // Incremental side: push before make, absorb deltas after.
            incr.push(NO_MOVE, NO_PIECE_TYPE);
            // Reference side: push too so indices line up; we'll overwrite
            // with a refresh (no delta replay).
            refs.push(NO_MOVE, NO_PIECE_TYPE);

            let ok = board.make_move(mv);
            assert!(ok, "{}: move {} illegal at ply {}", name, uci, ply);

            absorb_deltas(&mut incr, &mut board);
            incr.ensure_computed(&weights, nf, remap, &board);

            refs.refresh(&weights, nf, remap, &board, WHITE);
            refs.refresh(&weights, nf, remap, &board, BLACK);

            // Compare element-wise and surface first divergence.
            for pov in [WHITE, BLACK] {
                let a = incr.values(pov);
                let b = refs.values(pov);
                if a != b {
                    let mut first = None;
                    for j in 0..H {
                        if a[j] != b[j] { first = Some((j, a[j], b[j])); break; }
                    }
                    let (j, av, bv) = first.unwrap();
                    panic!(
                        "{}: ply={} move={} pov={} first diff at channel {} incr={} refresh={} (delta_count={})",
                        name, ply, uci,
                        if pov == WHITE { "W" } else { "B" },
                        j, av, bv,
                        incr.current().delta.len(),
                    );
                }
            }
        }
    }

    #[test]
    fn startpos_quiet_moves() {
        // Sanity: no captures, no x-rays activated.
        run_scenario(
            "startpos_quiet",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
        );
    }

    #[test]
    fn simple_captures_no_xray() {
        // Knight captures with no slider x-ray behind the captured square.
        run_scenario(
            "simple_captures",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            &["b8c6", "f1b5", "a7a6", "b5c6", "d7c6"],
        );
    }

    #[test]
    fn rook_captures_pawn_revealing_xray() {
        // White rook on a1, white pawn gone, black pawn on a7, black king a8.
        // Rook takes pawn on a7 — now rook attacks king (already attacked via a-file).
        // More interesting: set up a rook blocked by enemy piece with enemy piece behind.
        // Position: white rook a1, black pawn a4 (blocker), black rook a8 (x-ray target).
        // Wa1 captures a4 pawn directly; before capture a1→a4 direct, a1→a8 x-ray.
        // After capture (Ra1xa4), rook now on a4; direct Ra4→a8 on a-file.
        run_scenario(
            "rook_xray_capture",
            "r3k3/8/8/8/p7/8/8/R3K3 w Q - 0 1",
            &["a1a4"],
        );
    }

    #[test]
    fn bishop_xray_through_pawn_captured() {
        // White bishop a1, black pawn d4 (blocker), black rook h8 (x-ray target).
        // Before: Ba1 directly attacks pawn d4; x-ray through d4 reveals... nothing
        // (h8 is on the a1-h8 diagonal, pawn d4 is also on it, rook h8 behind).
        // Capture Bxd4 changes geometry.
        run_scenario(
            "bishop_xray_diagonal",
            "7r/8/8/8/3p4/8/8/B3K2k w - - 0 1",
            &["a1d4"],
        );
    }

    #[test]
    fn queen_xray_orthogonal_and_diagonal() {
        // Queen on d1 with pawn on d4 (blocker) and king on d8 (x-ray target).
        // Capture reveals queen → king x-ray on d-file.
        run_scenario(
            "queen_xray",
            "3k4/8/8/8/3p4/8/8/3QK3 w - - 0 1",
            &["d1d4"],
        );
    }

    #[test]
    fn capture_that_opens_third_party_xray() {
        // The tricky case: a capture that doesn't involve the slider at all
        // but removes a piece that was blocking a slider from seeing behind.
        //
        // Setup:
        //   White rook a1, white pawn a4 (its own blocker),
        //   black pawn a5 (gets captured-ish scenario),
        //   black king a8.
        //
        // Better — capture by a different piece:
        //   White rook h1, white bishop c3 (irrelevant),
        //   black rook h8 on open h-file with black pawn h4 blocking (direct & x-ray slot),
        //   black knight g5 that white's bishop will capture.
        //
        // After Bc3xg5 (unrelated capture), the h-file situation is unchanged,
        // so this is a quiet-for-h-file capture. Good negative test.
        //
        // A real third-party x-ray: white rook a1 blocked by white pawn a2
        // from seeing black pawn a7 → black king a8. When something else
        // captures elsewhere, nothing should change on the a-file.
        run_scenario(
            "unrelated_capture",
            "k1b5/pp6/8/8/4n3/2B5/PP6/K1R5 w - - 0 1",
            &["c3e5"],  // unrelated knight takes would be c3xe4; we move bishop c3-e5 then tests after
        );
    }

    #[test]
    fn capture_blocker_between_slider_and_third_piece() {
        // Core x-ray-on-capture bug scenario.
        // White rook a1, black pawn a4 (blocker), black knight a7 (x-ray target behind).
        // Before Rxa4: direct a1→a4 pawn; x-ray a1→a7 knight (through pawn a4).
        // After Rxa4: rook now on a4; direct a4→a7 knight.
        // Incremental path must net out the x-ray-through-pawn feature loss.
        run_scenario(
            "rook_captures_blocker_with_xray_behind",
            "k7/n7/8/8/p7/8/8/R3K3 w Q - 0 1",
            &["a1a4"],
        );
    }

    #[test]
    fn slider_captures_then_moves_away() {
        // Multi-move scenario: capture then retreat. Exercises back-to-back
        // delta application. The BN on a7 is pinned against BK on a8 once
        // the rook lands on a4, so use a black pawn move between the two
        // white moves to test incremental survival across a black move.
        run_scenario(
            "capture_then_retreat",
            "k7/n5p1/8/8/p7/8/8/R3K3 w Q - 0 1",
            &["a1a4", "g7g6", "a4a1"],
        );
    }

    #[test]
    fn kiwipete_tactical_sequence() {
        // Rich middlegame with many sliders and captures.
        // e5g6: Nxg6 (captures BP). f7g6: black pawn recapture.
        run_scenario(
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &["e5g6", "f7g6", "e2f1", "c7c6", "d5c6"],
        );
    }

    #[test]
    fn en_passant_capture() {
        // EP captures remove a piece from a square other than the move's `to`.
        // Tests push_threats_on_change for EP cap_sq + push_threats_on_move for pawn.
        run_scenario(
            "ep_capture",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            &["e5f6"],  // exf6 en passant
        );
    }

    #[test]
    fn castling_kingside() {
        // Castle moves both king and rook — tests back-to-back deltas
        // plus per-perspective king-file-mirror change.
        run_scenario(
            "castle_ks",
            "r3k2r/pppqppbp/2np1np1/4P3/2B5/2NQ1N2/PPP2PPP/R1B1K2R w KQkq - 0 1",
            &["e1g1"],  // O-O
        );
    }

    #[test]
    fn castling_queenside_phantom_xray_regression() {
        // Regression for the 2b slider-iteration rewrite: during the rook
        // leg of O-O-O (a1→d1) the moved rook is in pieces_bb at d1 but
        // occ_transit has d1 cleared. Without `& occ` on section-2b
        // candidates, d1 is iterated as a phantom x-ray candidate from
        // sq=a1 (with king@c1 between d1 and a1 → exactly one blocker)
        // and a spurious (rook@d1, wrook, a1) delta is emitted. Caught
        // by fuzz_random_games seed 0xdebd0132 ply 28.
        run_scenario(
            "castle_qs_phantom",
            "rQr5/p2pkp1n/1n2p1p1/7q/b1P1P3/Np2NB1P/PP3P1P/R3K2R w KQ - 3 15",
            &["e1c1"],  // O-O-O
        );
    }

    #[test]
    fn slider_move_reveals_x_ray_for_other_slider() {
        // WQ on d1, WR on d2 blocking. WR moves to d5 — WQ's rank/file view
        // shifts: gains direct d-file targets, loses the blocker.
        run_scenario(
            "slider_move_reveals_xray",
            "4k3/8/8/3r4/8/8/3R4/3QK3 w - - 0 1",
            &["d2d4"],
        );
    }

    #[test]
    fn chain_of_captures() {
        // Back-to-back captures — pawn trades leaving the incremental
        // state to absorb multiple small deltas in sequence.
        run_scenario(
            "chain_captures",
            "4k3/8/3p4/4p3/3P4/2N5/8/4K3 w - - 0 1",
            &["d4e5", "d6e5", "c3e4", "e8d8"],
        );
    }

    #[test]
    fn promotion_with_capture() {
        // Pawn captures and promotes — double state change.
        run_scenario(
            "promotion_capture",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            &["a7a8q"],
        );
    }

    /// Deterministic fuzzer: plays random legal moves from several
    /// starting positions and asserts incremental == refresh after
    /// every move. Covers move-type combinations that curated scenarios
    /// can miss (EP in slider ray, promotion into pin, castling through
    /// x-ray target, etc.).
    ///
    /// This is the counterpart to the curated scenario list above —
    /// those pin down specific failure modes; this one finds whatever
    /// we haven't thought of. If it fires on a new pattern, add a
    /// curated scenario for that pattern and keep the fuzzer for
    /// regression.
    #[test]
    fn fuzz_random_games() {
        crate::init();
        let nf = num_threat_features();
        let weights = make_weights(nf);

        // Several varied starting positions — opening, kiwipete middle-game,
        // tactical midgame with heavy slider activity, and an endgame.
        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1", // promotion testbed
        ];

        // Deterministic xorshift32 PRNG — reproducible failures.
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        const MAX_PLIES_PER_GAME: usize = 120;
        const GAMES_PER_FEN: usize = 20;

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..GAMES_PER_FEN {
                let seed: u32 = 0xDEADBEEFu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::new();
                board.set_fen(fen);
                board.generate_threat_deltas = true;

                let remap: &[i32] = &[];
                let mut incr = ThreatStack::new(H);
                incr.active = true;
                incr.refresh(&weights, nf, remap, &board, WHITE);
                incr.refresh(&weights, nf, remap, &board, BLACK);

                let mut refs = ThreatStack::new(H);
                refs.active = true;
                refs.refresh(&weights, nf, remap, &board, WHITE);
                refs.refresh(&weights, nf, remap, &board, BLACK);

                for ply in 0..MAX_PLIES_PER_GAME {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 {
                        break; // stalemate or checkmate
                    }
                    let idx = (next_u32(&mut rng) as usize) % legal.len;
                    let mv = legal.moves[idx];

                    incr.push(NO_MOVE, NO_PIECE_TYPE);
                    refs.push(NO_MOVE, NO_PIECE_TYPE);

                    let ok = board.make_move(mv);
                    assert!(ok, "fuzz {} game {} ply {}: move {} illegal?",
                        fen_idx, game, ply, crate::types::move_to_uci(mv));

                    // Absorb deltas into incremental stack.
                    {
                        let entry = incr.current_mut();
                        entry.delta.clear();
                        for d in board.threat_deltas.iter() { entry.delta.push(*d); }
                        let ul = board.undo_stack.len();
                        if ul > 0 {
                            let u = &board.undo_stack[ul - 1];
                            entry.mv = u.mv;
                            if u.mv != NO_MOVE {
                                entry.moved_pt = board.mailbox[move_to(u.mv) as usize];
                                entry.moved_color = crate::types::flip_color(board.side_to_move);
                            }
                        }
                    }

                    incr.ensure_computed(&weights, nf, remap, &board);
                    refs.refresh(&weights, nf, remap, &board, WHITE);
                    refs.refresh(&weights, nf, remap, &board, BLACK);

                    for pov in [WHITE, BLACK] {
                        let a = incr.values(pov);
                        let b = refs.values(pov);
                        if a != b {
                            // Find first divergent channel for a useful panic message.
                            let mut first = None;
                            for j in 0..H {
                                if a[j] != b[j] {
                                    first = Some((j, a[j], b[j]));
                                    break;
                                }
                            }
                            let (j, av, bv) = first.unwrap();
                            panic!(
                                "fuzz divergence: fen_idx={} game={} ply={} move={} pov={} \
                                 channel={} incr={} refresh={} seed={:#x}",
                                fen_idx, game, ply,
                                crate::types::move_to_uci(mv),
                                if pov == WHITE { "W" } else { "B" },
                                j, av, bv, seed,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Sanity: manual i16 value comparison across all 256 channels
    /// proves the weight pattern actually differs between features.
    #[test]
    fn weights_distinguish_features() {
        crate::init();
        let nf = num_threat_features();
        let w = make_weights(nf.min(4));
        // Feature 0's row should differ from feature 1's row.
        let row0 = &w[0..H];
        let row1 = &w[H..2 * H];
        assert_ne!(row0, row1, "weight pattern collides between features");
    }

    /// Diagnostic: print the multiset symmetric diff between the
    /// incremental-applied feature indices and the refresh-enumerated
    /// indices for a single move. Not a pass/fail test — run with
    /// `cargo test dump_diff -- --nocapture` to inspect.
    #[test]
    #[ignore]
    fn dump_diff_b8c6_quiet_knight() {
        crate::init();
        let nf = num_threat_features();
        let mut board = Board::new();
        // Position after 1.e4 e5 — white to move, then black plays Nc6 on next ply.
        // Set up mid-game FEN so it's black-to-move playing b8c6 immediately.
        board.set_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
        board.generate_threat_deltas = true;

        // Multiset diff, same structure as dump_diff_rook_capture_with_xray but parametric on pov.
        for pov in [WHITE, BLACK] {
            let mut pre: Vec<usize> = Vec::new();
            {
                let occ = board.colors[0] | board.colors[1];
                let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
                crate::threats::enumerate_threats(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, (k % 8) >= 4,
                    |idx| if idx < nf { pre.push(idx) },
                );
            }
            eprintln!("[pov={}] pre count={}", if pov == WHITE { "W" } else { "B" }, pre.len());
        }

        let mv = parse_uci(&board, "b8c6");
        board.make_move(mv);

        for pov in [WHITE, BLACK] {
            let mut post: Vec<usize> = Vec::new();
            {
                let occ = board.colors[0] | board.colors[1];
                let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
                crate::threats::enumerate_threats(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, (k % 8) >= 4,
                    |idx| if idx < nf { post.push(idx) },
                );
            }

            let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
            let mirrored = (k % 8) >= 4;

            let mut actual_adds: Vec<usize> = Vec::new();
            let mut actual_subs: Vec<usize> = Vec::new();
            for d in board.threat_deltas.iter() {
                let idx = crate::threats::threat_index(
                    d.attacker_cp() as usize, d.from_sq() as u32,
                    d.victim_cp() as usize, d.to_sq() as u32,
                    mirrored, pov,
                );
                if idx < 0 || (idx as usize) >= nf { continue; }
                if d.add() { actual_adds.push(idx as usize); }
                else { actual_subs.push(idx as usize); }
            }

            // Rebuild pre from the delta'd post to expose divergence.
            // expected_pre = post - adds + subs (treat as multiset)
            // For a matching path: expected_pre == pre.
            let mut reconstructed = post.clone();
            for a in &actual_adds {
                if let Some(p) = reconstructed.iter().position(|x| x == a) {
                    reconstructed.swap_remove(p);
                }
            }
            reconstructed.extend(actual_subs.iter());

            // Sort both for comparison.
            let mut pre: Vec<usize> = Vec::new();
            {
                // recompute pre via a pre-state snapshot via UNDO — but our board
                // already applied the move. Recompute using full pre-move FEN.
                let mut pre_board = Board::new();
                pre_board.set_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
                let occ = pre_board.colors[0] | pre_board.colors[1];
                let k0 = (pre_board.pieces[KING as usize] & pre_board.colors[pov as usize]).trailing_zeros();
                crate::threats::enumerate_threats(
                    &pre_board.pieces, &pre_board.colors, &pre_board.mailbox,
                    occ, pov, (k0 % 8) >= 4,
                    |idx| if idx < nf { pre.push(idx) },
                );
            }

            let mut pre_sorted = pre.clone(); pre_sorted.sort();
            let mut recon_sorted = reconstructed.clone(); recon_sorted.sort();

            let missing: Vec<usize> = pre_sorted.iter().filter(|x| !recon_sorted.contains(x)).cloned().collect();
            let extra: Vec<usize> = recon_sorted.iter().filter(|x| !pre_sorted.contains(x)).cloned().collect();
            eprintln!("[pov={}] pre={} post={} adds={} subs={} missing_from_recon={:?} extra_in_recon={:?}",
                if pov == WHITE { "W" } else { "B" }, pre.len(), post.len(), actual_adds.len(), actual_subs.len(), missing, extra);
            if !missing.is_empty() || !extra.is_empty() {
                // Enumerate post-state tuples that map to the extra index.
                let extra_set = extra.clone();
                eprintln!("  tuples in post that hit each 'extra' index:");
                let occ_post = board.colors[0] | board.colors[1];
                let mailbox_post = &board.mailbox;
                let white_bb = board.colors[0];
                let pieces_bb = &board.pieces;
                for extra_idx in &extra_set {
                    for color in [0u8, 1u8] {
                        for pt in 0..6u8 {
                            let mut bb = pieces_bb[pt as usize] & board.colors[color as usize];
                            while bb != 0 {
                                let sq = bb.trailing_zeros();
                                bb &= bb - 1;
                                let cp_a = crate::threats::colored_piece(color, pt);
                                let atts = crate::threats::piece_attacks_occ(pt, color, sq, occ_post) & occ_post;
                                let mut t = atts;
                                while t != 0 {
                                    let tsq = t.trailing_zeros();
                                    t &= t - 1;
                                    let vpt = mailbox_post[tsq as usize];
                                    if vpt >= 6 { continue; }
                                    let vcol = if white_bb & (1u64 << tsq) != 0 { 0 } else { 1 };
                                    let cp_v = crate::threats::colored_piece(vcol, vpt);
                                    let i = crate::threats::threat_index(
                                        cp_a, sq, cp_v, tsq as u32, mirrored, pov,
                                    );
                                    if i as usize == *extra_idx {
                                        eprintln!("    idx={} direct {}@{} -> {}@{}", i, cp_a, sq, cp_v, tsq);
                                    }
                                }
                                // x-ray
                                if pt == BISHOP || pt == ROOK || pt == QUEEN {
                                    let mut dt = atts;
                                    while dt != 0 {
                                        let bsq = dt.trailing_zeros();
                                        dt &= dt - 1;
                                        let ow = occ_post & !(1u64 << bsq);
                                        let through = crate::threats::piece_attacks_occ(pt, color, sq, ow);
                                        let revealed = through & !atts & ow;
                                        if revealed == 0 { continue; }
                                        let xsq = if sq < bsq {
                                            let a = revealed & !((1u64 << (bsq + 1)) - 1);
                                            if a != 0 { a.trailing_zeros() } else { 64 }
                                        } else {
                                            let b = revealed & ((1u64 << bsq) - 1);
                                            if b != 0 { 63 - b.leading_zeros() } else { 64 }
                                        };
                                        if xsq < 64 {
                                            let xpt = mailbox_post[xsq as usize];
                                            if xpt < 6 {
                                                let xcol = if white_bb & (1u64 << xsq) != 0 { 0 } else { 1 };
                                                let cp_x = crate::threats::colored_piece(xcol, xpt);
                                                let i = crate::threats::threat_index(
                                                    cp_a, sq, cp_x, xsq, mirrored, pov,
                                                );
                                                if i as usize == *extra_idx {
                                                    eprintln!("    idx={} xray  {}@{} -> {}@{}", i, cp_a, sq, cp_x, xsq);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                eprintln!("  raw deltas (idx = threat_index for this pov):");
                for d in board.threat_deltas.iter() {
                    let idx = crate::threats::threat_index(
                        d.attacker_cp() as usize, d.from_sq() as u32,
                        d.victim_cp() as usize, d.to_sq() as u32,
                        mirrored, pov,
                    );
                    eprintln!("    a_cp={} from={} v_cp={} to={} add={} idx={}",
                        d.attacker_cp(), d.from_sq(), d.victim_cp(), d.to_sq(), d.add(), idx);
                }
                // Dump pre and post multisets too (sorted).
                let mut pre_sorted2 = pre.clone(); pre_sorted2.sort();
                let mut post_sorted = post.clone(); post_sorted.sort();
                eprintln!("  pre : {:?}", pre_sorted2);
                eprintln!("  post: {:?}", post_sorted);
            }
        }
    }

    #[test]
    #[ignore]
    fn dump_diff_rook_capture_with_xray() {
        crate::init();
        let nf = num_threat_features();
        let mut board = Board::new();
        board.set_fen("k7/n7/8/8/p7/8/8/R3K3 w Q - 0 1");
        board.generate_threat_deltas = true;

        // Baseline multiset: enumerate_threats at position BEFORE move.
        let mut pre_indices_w: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            let wk = (board.pieces[KING as usize] & board.colors[0]).trailing_zeros();
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, (wk % 8) >= 4,
                |idx| if idx < nf { pre_indices_w.push(idx) },
            );
        }

        let mv = parse_uci(&board, "a1a4");
        board.make_move(mv);

        // Refresh multiset at post-move position.
        let mut post_indices_w: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            let wk = (board.pieces[KING as usize] & board.colors[0]).trailing_zeros();
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, (wk % 8) >= 4,
                |idx| if idx < nf { post_indices_w.push(idx) },
            );
        }

        // Expected delta multiset: (post - pre)_add, (pre - post)_sub.
        let mut expected_adds = post_indices_w.clone();
        let mut expected_subs = pre_indices_w.clone();
        // Remove intersection to get the symmetric difference (net change).
        for i in (0..expected_adds.len()).rev() {
            if let Some(p) = expected_subs.iter().position(|&x| x == expected_adds[i]) {
                expected_subs.swap_remove(p);
                expected_adds.swap_remove(i);
            }
        }
        expected_adds.sort();
        expected_subs.sort();

        // Incremental multiset: walk board.threat_deltas through threat_index.
        let wk = (board.pieces[KING as usize] & board.colors[0]).trailing_zeros();
        let mirrored = (wk % 8) >= 4;
        let mut actual_adds: Vec<usize> = Vec::new();
        let mut actual_subs: Vec<usize> = Vec::new();
        for d in board.threat_deltas.iter() {
            let idx = crate::threats::threat_index(
                d.attacker_cp() as usize, d.from_sq() as u32,
                d.victim_cp() as usize, d.to_sq() as u32,
                mirrored, WHITE,
            );
            if idx < 0 || (idx as usize) >= nf { continue; }
            if d.add() { actual_adds.push(idx as usize); }
            else { actual_subs.push(idx as usize); }
        }
        // Net out the actual delta too (incremental deltas can include
        // add+sub pairs for the same feature that cancel out).
        for i in (0..actual_adds.len()).rev() {
            if let Some(p) = actual_subs.iter().position(|&x| x == actual_adds[i]) {
                actual_subs.swap_remove(p);
                actual_adds.swap_remove(i);
            }
        }
        actual_adds.sort();
        actual_subs.sort();

        eprintln!("=== Rxa4 (a1→a4 capture, knight on a7 behind) WHITE pov ===");
        eprintln!("expected_adds (post - pre): {:?}", expected_adds);
        eprintln!("expected_subs (pre - post): {:?}", expected_subs);
        eprintln!("actual_adds   (from deltas): {:?}", actual_adds);
        eprintln!("actual_subs   (from deltas): {:?}", actual_subs);

        let missing_adds: Vec<_> = expected_adds.iter().filter(|i| !actual_adds.contains(i)).collect();
        let extra_adds: Vec<_> = actual_adds.iter().filter(|i| !expected_adds.contains(i)).collect();
        let missing_subs: Vec<_> = expected_subs.iter().filter(|i| !actual_subs.contains(i)).collect();
        let extra_subs: Vec<_> = actual_subs.iter().filter(|i| !expected_subs.contains(i)).collect();
        eprintln!("missing adds (expected but not emitted): {:?}", missing_adds);
        eprintln!("extra   adds (emitted but not expected): {:?}", extra_adds);
        eprintln!("missing subs (expected but not emitted): {:?}", missing_subs);
        eprintln!("extra   subs (emitted but not expected): {:?}", extra_subs);

        // Print raw deltas too.
        eprintln!("raw deltas:");
        for d in board.threat_deltas.iter() {
            eprintln!("  attacker_cp={} from={} victim_cp={} to={} add={}",
                d.attacker_cp(), d.from_sq(), d.victim_cp(), d.to_sq(), d.add());
        }
    }

    /// Reproduce the fuzz_random_games failure seed 0xdebd0132 at ply 28 (e1c1).
    /// Prints the pre-move FEN, raw deltas, and per-channel threat_index diff
    /// for the BLACK pov where the failure is observed. Ignored by default —
    /// run with `cargo test --release diag_fuzz_ply28_e1c1 -- --nocapture --ignored`.
    #[test]
    #[ignore]
    fn diag_fuzz_ply28_e1c1() {
        crate::init();
        let nf = num_threat_features();

        // Kiwipete + same xorshift32 PRNG as fuzz_random_games (fen_idx=1, game=0).
        const KIWIPETE: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let seed: u32 = 0xdebd0132;
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        let mut rng = seed;
        let mut board = Board::new();
        board.set_fen(KIWIPETE);
        board.generate_threat_deltas = true;

        for ply in 0..28 {
            let legal = generate_legal_moves(&board);
            assert!(legal.len > 0, "no legal at ply {}", ply);
            let idx = (next_u32(&mut rng) as usize) % legal.len;
            let mv = legal.moves[idx];
            assert!(board.make_move(mv), "move illegal at ply {}", ply);
        }

        // Ply 28: pick the move, expect e1c1 (white castling queenside).
        let legal = generate_legal_moves(&board);
        let idx = (next_u32(&mut rng) as usize) % legal.len;
        let mv = legal.moves[idx];
        let uci = crate::types::move_to_uci(mv);
        let pre_fen = board.to_fen();
        eprintln!("=== ply 28 pre-FEN: {}", pre_fen);
        eprintln!("=== ply 28 move: {}", uci);
        assert_eq!(uci, "e1c1", "expected e1c1, got {}", uci);

        // Pre-move enumeration for BLACK pov.
        let pov = BLACK;
        let mirrored_pre = {
            let occ = board.colors[0] | board.colors[1];
            let _ = occ;
            let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
            (k % 8) >= 4
        };
        let mut pre_indices: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, pov, mirrored_pre,
                |i| if i < nf { pre_indices.push(i) },
            );
        }

        // Make the move.
        assert!(board.make_move(mv));

        // Post-move enumeration for BLACK.
        let mirrored_post = {
            let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
            (k % 8) >= 4
        };
        let mut post_indices: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, pov, mirrored_post,
                |i| if i < nf { post_indices.push(i) },
            );
        }

        // If mirror flipped between pre and post, per-feature index
        // comparison isn't meaningful. Skip directly in that case.
        eprintln!("mirrored_pre={} mirrored_post={}", mirrored_pre, mirrored_post);

        // Expected net change: (post - pre)_add, (pre - post)_sub.
        let mut expected_adds = post_indices.clone();
        let mut expected_subs = pre_indices.clone();
        for i in (0..expected_adds.len()).rev() {
            if let Some(p) = expected_subs.iter().position(|&x| x == expected_adds[i]) {
                expected_subs.swap_remove(p);
                expected_adds.swap_remove(i);
            }
        }
        expected_adds.sort(); expected_subs.sort();

        // Actual net: walk raw deltas through threat_index with the POST-move mirror.
        let mut actual_adds: Vec<usize> = Vec::new();
        let mut actual_subs: Vec<usize> = Vec::new();
        for d in board.threat_deltas.iter() {
            let idx = crate::threats::threat_index(
                d.attacker_cp() as usize, d.from_sq() as u32,
                d.victim_cp() as usize, d.to_sq() as u32,
                mirrored_post, pov,
            );
            if idx < 0 || (idx as usize) >= nf { continue; }
            if d.add() { actual_adds.push(idx as usize); }
            else { actual_subs.push(idx as usize); }
        }
        // Net out same-feature add+sub cancellations.
        for i in (0..actual_adds.len()).rev() {
            if let Some(p) = actual_subs.iter().position(|&x| x == actual_adds[i]) {
                actual_subs.swap_remove(p);
                actual_adds.swap_remove(i);
            }
        }
        actual_adds.sort(); actual_subs.sort();

        let missing_adds: Vec<_> = expected_adds.iter().filter(|i| !actual_adds.contains(i)).cloned().collect();
        let extra_adds:   Vec<_> = actual_adds.iter().filter(|i| !expected_adds.contains(i)).cloned().collect();
        let missing_subs: Vec<_> = expected_subs.iter().filter(|i| !actual_subs.contains(i)).cloned().collect();
        let extra_subs:   Vec<_> = actual_subs.iter().filter(|i| !expected_subs.contains(i)).cloned().collect();
        eprintln!("pre count={} post count={}", pre_indices.len(), post_indices.len());
        eprintln!("expected_adds: {:?}", expected_adds);
        eprintln!("expected_subs: {:?}", expected_subs);
        eprintln!("actual_adds:   {:?}", actual_adds);
        eprintln!("actual_subs:   {:?}", actual_subs);
        eprintln!("missing adds (expected but not emitted): {:?}", missing_adds);
        eprintln!("extra   adds (emitted but not expected): {:?}", extra_adds);
        eprintln!("missing subs (expected but not emitted): {:?}", missing_subs);
        eprintln!("extra   subs (emitted but not expected): {:?}", extra_subs);

        eprintln!("raw deltas (ply28 e1c1):");
        for d in board.threat_deltas.iter() {
            let idx = crate::threats::threat_index(
                d.attacker_cp() as usize, d.from_sq() as u32,
                d.victim_cp() as usize, d.to_sq() as u32,
                mirrored_post, pov,
            );
            eprintln!("  a_cp={} from={} v_cp={} to={} add={}  idx(B)={}",
                d.attacker_cp(), d.from_sq(), d.victim_cp(), d.to_sq(), d.add(), idx);
        }
    }

    /// Ensure RawThreatDelta round-trips — if this breaks the whole
    /// Sparsity measurement: feature activation frequency across many
    /// self-play positions. Ignored by default — run with
    /// `cargo test --release measure_feature_sparsity -- --nocapture --ignored`.
    ///
    /// Purpose: inform the "drop cold features" optimization target. If
    /// the bottom X% of features fire <0.001% of positions, dropping
    /// them shrinks the 50 MB weight matrix proportionally with near-
    /// zero Elo cost, improving cache residency on memory-constrained
    /// hardware.
    #[test]
    #[ignore]
    fn measure_feature_sparsity() {
        crate::init();
        let nf = num_threat_features();
        eprintln!("Measuring activation frequency across {} threat features", nf);

        // Deterministic self-play positions from 5 starting FENs * 30 games * 80 plies
        // ≈ 12000 positions. Wider than fuzz_random_games, focused on distribution.
        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
        ];
        const GAMES_PER_FEN: usize = 30;
        const MAX_PLIES: usize = 80;

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        // Histograms: total activations per feature + positions observed
        let mut feature_hits: Vec<u32> = vec![0u32; nf];
        let mut positions = 0u64;
        let mut total_activations = 0u64;

        for (fi, fen) in START_FENS.iter().enumerate() {
            for g in 0..GAMES_PER_FEN {
                let seed: u32 = 0x12345u32
                    .wrapping_add((fi as u32).wrapping_mul(1_000_003))
                    .wrapping_add((g as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::new();
                board.set_fen(fen);

                for _ply in 0..MAX_PLIES {
                    // Record feature activations from both POVs at this position.
                    for pov in [WHITE, BLACK] {
                        let occ = board.colors[0] | board.colors[1];
                        let k = (board.pieces[KING as usize] & board.colors[pov as usize])
                            .trailing_zeros();
                        let mirrored = (k % 8) >= 4;
                        crate::threats::enumerate_threats(
                            &board.pieces, &board.colors, &board.mailbox,
                            occ, pov, mirrored,
                            |idx| {
                                if idx < nf {
                                    feature_hits[idx] = feature_hits[idx].saturating_add(1);
                                    total_activations += 1;
                                }
                            },
                        );
                    }
                    positions += 1;

                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv_idx = (next_u32(&mut rng) as usize) % legal.len;
                    let mv = legal.moves[mv_idx];
                    if !board.make_move(mv) { break; }
                }
            }
        }

        // Distribution buckets
        let mut bucket_0     = 0u64; // never activated
        let mut bucket_1_9   = 0u64;
        let mut bucket_10_99 = 0u64;
        let mut bucket_100_999 = 0u64;
        let mut bucket_1k_plus = 0u64;
        let mut max_hits = 0u32;
        let mut max_idx  = 0usize;
        for (i, &h) in feature_hits.iter().enumerate() {
            if h == 0 { bucket_0 += 1; }
            else if h < 10 { bucket_1_9 += 1; }
            else if h < 100 { bucket_10_99 += 1; }
            else if h < 1000 { bucket_100_999 += 1; }
            else { bucket_1k_plus += 1; }
            if h > max_hits { max_hits = h; max_idx = i; }
        }

        // Top-K features
        let mut indexed: Vec<(usize, u32)> = feature_hits.iter().enumerate()
            .map(|(i, &h)| (i, h)).collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));

        // Coverage: cumulative % of activations captured by top-K features
        let mut cumulative = 0u64;
        let mut cov_10   = 0.0;
        let mut cov_50   = 0.0;
        let mut cov_90   = 0.0;
        let mut cov_99   = 0.0;
        let mut features_for_99 = 0usize;
        let mut features_for_90 = 0usize;
        let mut features_for_50 = 0usize;
        for (i, (_idx, h)) in indexed.iter().enumerate() {
            cumulative += *h as u64;
            let pct = cumulative as f64 / total_activations as f64 * 100.0;
            if i == 10   && cov_10   == 0.0 { cov_10   = pct; }
            if i == 50   && cov_50   == 0.0 { cov_50   = pct; }
            if i == 100  && cov_90   == 0.0 && pct >= 50.0 { cov_50 = pct; features_for_50 = i; }
            if pct >= 50.0 && features_for_50 == 0 { features_for_50 = i + 1; cov_50 = pct; }
            if pct >= 90.0 && features_for_90 == 0 { features_for_90 = i + 1; cov_90 = pct; }
            if pct >= 99.0 && features_for_99 == 0 { features_for_99 = i + 1; cov_99 = pct; break; }
        }

        eprintln!("\n=== Threat feature sparsity measurement ===");
        eprintln!("Total positions sampled:   {}", positions);
        eprintln!("Total activations recorded: {}", total_activations);
        eprintln!("Avg features active per pov-position: {:.1}", total_activations as f64 / (positions * 2) as f64);
        eprintln!("\n--- Feature activation distribution ---");
        eprintln!("  0 hits    (never fired):    {:>7} features ({:.1}%)", bucket_0, bucket_0 as f64 / nf as f64 * 100.0);
        eprintln!("  1-9 hits  (very rare):      {:>7} features ({:.1}%)", bucket_1_9, bucket_1_9 as f64 / nf as f64 * 100.0);
        eprintln!("  10-99     (uncommon):       {:>7} features ({:.1}%)", bucket_10_99, bucket_10_99 as f64 / nf as f64 * 100.0);
        eprintln!("  100-999   (common):         {:>7} features ({:.1}%)", bucket_100_999, bucket_100_999 as f64 / nf as f64 * 100.0);
        eprintln!("  1000+     (hot):            {:>7} features ({:.1}%)", bucket_1k_plus, bucket_1k_plus as f64 / nf as f64 * 100.0);
        eprintln!("\n--- Coverage ---");
        eprintln!("  Top {} features capture 50% of activations", features_for_50);
        eprintln!("  Top {} features capture 90% of activations", features_for_90);
        eprintln!("  Top {} features capture 99% of activations", features_for_99);
        eprintln!("  Max activations on single feature: {} (idx={})", max_hits, max_idx);

        // Memory implications
        let row_bytes = 768;  // v9 accumulator size, i8 weights
        let total_bytes = (nf as u64) * row_bytes;
        let dead_bytes  = bucket_0 * row_bytes;
        let cold_bytes  = (bucket_0 + bucket_1_9) * row_bytes;
        eprintln!("\n--- Memory impact (768-byte rows, i8 weights) ---");
        eprintln!("  Total weight matrix: {:.1} MB", total_bytes as f64 / 1_048_576.0);
        eprintln!("  Dropping dead features (0 hits): save {:.1} MB ({:.1}%)", dead_bytes as f64 / 1_048_576.0, bucket_0 as f64 / nf as f64 * 100.0);
        eprintln!("  Dropping dead+rare (<10 hits):   save {:.1} MB ({:.1}%)", cold_bytes as f64 / 1_048_576.0, (bucket_0 + bucket_1_9) as f64 / nf as f64 * 100.0);
    }

    /// Compact remap: a net with some zero feature rows must produce the
    /// same accumulator as the uncompacted net. Simulates load-time
    /// compaction by zeroing half the feature rows, then mirrors the
    /// load_nnue compaction logic: build remap, drop zero rows. Runs the
    /// same make-moves scenario against both and asserts identical values.
    #[test]
    fn compact_remap_matches_identity() {
        crate::init();
        let nf = num_threat_features();
        let h = H;
        let full_weights = make_weights(nf);

        // Build a "sparsified" copy: zero every other feature row.
        let mut sparse_weights = full_weights.clone();
        let mut is_zero = vec![false; nf];
        for f in 0..nf {
            if f % 2 == 0 {
                for j in 0..h {
                    sparse_weights[f * h + j] = 0;
                }
                is_zero[f] = true;
            }
        }

        // Compact-compact: drop zero rows, build remap.
        let mut compact = Vec::with_capacity(sparse_weights.len());
        let mut remap = vec![-1i32; nf];
        let mut n_compact = 0usize;
        for f in 0..nf {
            if is_zero[f] { continue; }
            compact.extend_from_slice(&sparse_weights[f * h..(f + 1) * h]);
            remap[f] = n_compact as i32;
            n_compact += 1;
        }

        let moves = &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "d2d4", "e5d4"];
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        let mut board_a = Board::new();
        board_a.set_fen(fen);
        board_a.generate_threat_deltas = true;
        let mut board_b = Board::new();
        board_b.set_fen(fen);
        board_b.generate_threat_deltas = true;

        let identity: &[i32] = &[];
        let mut ts_full = ThreatStack::new(h);
        ts_full.active = true;
        ts_full.refresh(&sparse_weights, nf, identity, &board_a, WHITE);
        ts_full.refresh(&sparse_weights, nf, identity, &board_a, BLACK);

        let mut ts_compact = ThreatStack::new(h);
        ts_compact.active = true;
        ts_compact.refresh(&compact, nf, &remap, &board_b, WHITE);
        ts_compact.refresh(&compact, nf, &remap, &board_b, BLACK);

        assert_eq!(ts_full.values(WHITE), ts_compact.values(WHITE),
            "compact W baseline diverges from full");
        assert_eq!(ts_full.values(BLACK), ts_compact.values(BLACK),
            "compact B baseline diverges from full");

        for (ply, uci) in moves.iter().enumerate() {
            let mv_a = parse_uci(&board_a, uci);
            let mv_b = parse_uci(&board_b, uci);
            ts_full.push(NO_MOVE, NO_PIECE_TYPE);
            ts_compact.push(NO_MOVE, NO_PIECE_TYPE);
            assert!(board_a.make_move(mv_a));
            assert!(board_b.make_move(mv_b));
            absorb_deltas(&mut ts_full, &mut board_a);
            absorb_deltas(&mut ts_compact, &mut board_b);
            ts_full.ensure_computed(&sparse_weights, nf, identity, &board_a);
            ts_compact.ensure_computed(&compact, nf, &remap, &board_b);
            for pov in [WHITE, BLACK] {
                let a = ts_full.values(pov);
                let b = ts_compact.values(pov);
                if a != b {
                    let (j, av, bv) = (0..h).find_map(|j| if a[j] != b[j] { Some((j, a[j], b[j])) } else { None }).unwrap();
                    panic!("compact remap diverges at ply={} move={} pov={} chan={} full={} compact={}",
                        ply, uci, if pov == WHITE { "W" } else { "B" }, j, av, bv);
                }
            }
        }
    }

    /// incremental path will silently misapply deltas.
    #[test]
    fn raw_delta_roundtrip() {
        let d = RawThreatDelta::new(5, 28, 11, 63, true);
        assert_eq!(d.attacker_cp(), 5);
        assert_eq!(d.from_sq(), 28);
        assert_eq!(d.victim_cp(), 11);
        assert_eq!(d.to_sq(), 63);
        assert!(d.add());
        let d2 = RawThreatDelta::new(0, 0, 0, 0, false);
        assert!(!d2.add());
    }
}
