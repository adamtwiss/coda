//! EPD file loading and test suite runner.

use std::fs;
use std::time::Instant;

use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::*;
use crate::types::*;

/// A single EPD test position.
pub struct EpdPosition {
    pub fen: String,
    pub best_moves: Vec<String>,  // "bm" field: correct moves in SAN or coordinate
    pub avoid_moves: Vec<String>, // "am" field: moves to avoid
    pub id: String,
}

/// Resolve the NNUE net for an EPD run.
///
/// Mirrors every other subcommand (`bench`, `datagen`, `eval-fens`): an explicit
/// `-n` wins and is fatal if it fails to load, otherwise fall back to the shared
/// `auto_discover_nnue` (embedded net > ./net.nnue > net.txt discovery).
///
/// `epd` used to skip the fallback entirely, so on a `make` build — which HAS a
/// net embedded — `coda epd file.epd` ran with no net at all and panicked deep
/// inside the search ("no NNUE net loaded", search.rs) on the first evaluate.
/// It was the only subcommand missing the fallback.
///
/// Returns Err with a ready-to-print message instead of exiting, so the failure
/// paths are testable.
pub(crate) fn resolve_net(info: &mut SearchInfo, nnue_path: Option<&str>) -> Result<(), String> {
    if let Some(path) = nnue_path {
        // An EXPLICIT override that fails must be fatal: silently falling back
        // to the embedded net produces plausible-looking but wrong suite results.
        return info
            .load_nnue(path)
            .map_err(|e| format!("failed to load NNUE '{}': {}", path, e));
    }
    if !info.auto_discover_nnue() {
        return Err(
            "no NNUE net found (looked for an embedded net, ./net.nnue, and net.txt \
             discovery). Build with `make` to embed one, or pass -n <path>."
                .to_string(),
        );
    }
    Ok(())
}

/// Parse an EPD file into positions.
pub fn parse_epd(path: &str) -> Vec<EpdPosition> {
    let content = fs::read_to_string(path).expect("Failed to read EPD file");
    let mut positions = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(pos) = parse_epd_line(line) {
            positions.push(pos);
        }
    }

    positions
}

fn parse_epd_line(line: &str) -> Option<EpdPosition> {
    // EPD format: <FEN without move counters> <operations>
    // FEN has 4 fields: position, side, castling, ep
    // Then operations like: bm Rg3; id "WAC.003";

    let parts: Vec<&str> = line.splitn(5, ' ').collect();
    if parts.len() < 4 {
        return None;
    }

    // The FEN is the first 4 fields + default move counters
    let fen = format!("{} {} {} {} 0 1", parts[0], parts[1], parts[2], parts[3]);

    // Parse operations (everything after the 4th field)
    let ops_str = if parts.len() > 4 { parts[4] } else { "" };

    let mut best_moves = Vec::new();
    let mut avoid_moves = Vec::new();
    let mut id = String::new();

    // Split by semicolon for separate operations
    for op in ops_str.split(';') {
        let op = op.trim();
        if op.is_empty() { continue; }

        if let Some(moves_str) = op.strip_prefix("bm ") {
            for m in moves_str.split_whitespace() {
                best_moves.push(m.trim_end_matches(';').to_string());
            }
        } else if let Some(moves_str) = op.strip_prefix("am ") {
            for m in moves_str.split_whitespace() {
                avoid_moves.push(m.trim_end_matches(';').to_string());
            }
        } else if op.starts_with("id ") {
            id = op[3..].trim_matches('"').to_string();
        }
    }

    Some(EpdPosition {
        fen,
        best_moves,
        avoid_moves,
        id,
    })
}

/// Convert a move to SAN notation for comparison with EPD best moves.
pub fn move_to_san(board: &Board, mv: Move) -> String {
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);
    let pt = board.piece_type_at(from);

    // Castling
    if flags == FLAG_CASTLE {
        return if to > from { "O-O".to_string() } else { "O-O-O".to_string() };
    }

    let mut san = String::new();

    // Piece letter (not for pawns)
    if pt != PAWN {
        san.push(match pt {
            KNIGHT => 'N',
            BISHOP => 'B',
            ROOK => 'R',
            QUEEN => 'Q',
            KING => 'K',
            _ => '?',
        });
    }

    // Disambiguation: check if another piece of same type can go to the same square
    if pt != PAWN && pt != KING {
        let legal = generate_legal_moves(board);
        let mut same_piece_to = false;
        let mut same_file = false;
        let mut same_rank = false;

        for i in 0..legal.len {
            let other = legal.get(i);
            if other == mv { continue; }
            let other_from = move_from(other);
            let other_to = move_to(other);
            if other_to != to { continue; }
            if board.piece_type_at(other_from) != pt { continue; }

            same_piece_to = true;
            if file_of(other_from) == file_of(from) { same_file = true; }
            if rank_of(other_from) == rank_of(from) { same_rank = true; }
        }

        if same_piece_to {
            if !same_file {
                san.push((b'a' + file_of(from)) as char);
            } else if !same_rank {
                san.push((b'1' + rank_of(from)) as char);
            } else {
                san.push((b'a' + file_of(from)) as char);
                san.push((b'1' + rank_of(from)) as char);
            }
        }
    }

    // Capture
    let is_capture = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
    if is_capture {
        if pt == PAWN {
            san.push((b'a' + file_of(from)) as char);
        }
        san.push('x');
    }

    // Destination square
    san.push((b'a' + file_of(to)) as char);
    san.push((b'1' + rank_of(to)) as char);

    // Promotion
    if is_promotion(mv) {
        san.push('=');
        san.push(match promotion_piece_type(mv) {
            KNIGHT => 'N',
            BISHOP => 'B',
            ROOK => 'R',
            QUEEN => 'Q',
            _ => '?',
        });
    }

    // Check/checkmate suffix
    let mut board_copy = board.clone();
    if !board_copy.make_move(mv) { return san; }
    if board_copy.in_check() {
        // Check if it's checkmate
        let legal_after = generate_legal_moves(&board_copy);
        if legal_after.len == 0 {
            san.push('#');
        } else {
            san.push('+');
        }
    }

    san
}

/// Run an EPD test suite.
pub fn run_epd(path: &str, time_per_pos: u64, max_positions: usize, nnue_path: Option<&str>) {
    let positions = parse_epd(path);
    let total = if max_positions > 0 { max_positions.min(positions.len()) } else { positions.len() };

    println!("Running {} positions from {}", total, path);
    println!("Time per position: {}ms", time_per_pos);
    println!();

    let mut info = SearchInfo::new(64);
    if let Err(e) = resolve_net(&mut info, nnue_path) {
        eprintln!("FATAL: {}", e);
        std::process::exit(2);
    }
    let mut passed = 0;
    let mut failed = 0;
    let suite_start = Instant::now();

    for (i, pos) in positions.iter().enumerate() {
        if i >= total { break; }

        let mut board = Board::from_fen(&pos.fen);
        info.nodes = 0;
        // `search()` deliberately does NOT reset global_nodes (see its comment:
        // callers own it, so SMP helper contributions aren't clobbered). epd
        // calls `search()` directly and single-threaded, so it must do it here
        // or the counter accumulates across the whole suite: every info line
        // after the first reports the running SUITE total as `nodes`, and
        // divides it by the per-position `time`, so `nps` climbs without bound
        // (63M nps by mid-suite on a ~900k nps engine).
        //
        // Second, quieter consequence: global_nodes is also what the max_nodes
        // limit tests against, so a node-limited epd run would have every
        // position after the first trip its budget immediately.
        //
        // `bench` already carries the identical fix (P2.8); epd was missed.
        // (last_flushed_nodes is reset inside search() itself, so only the
        // shared counter needs clearing here.)
        info.global_nodes.store(0, std::sync::atomic::Ordering::Relaxed);
        info.stop.store(false, std::sync::atomic::Ordering::Relaxed);
        reset_epd_position_state(&mut info);

        let limits = SearchLimits {
            movetime: time_per_pos,
            ..SearchLimits::new()
        };

        let best = search(&mut board, &mut info, &limits);
        let best_san = move_to_san(&board, best);
        let best_uci = move_to_uci(best);

        // Move match helper: compare SAN (ignoring +/#) or UCI.
        let matches_move = |mv: &str| {
            let mv_clean = mv.trim_end_matches('+').trim_end_matches('#');
            let san_clean = best_san.trim_end_matches('+').trim_end_matches('#');
            mv_clean == san_clean || mv == best_uci
        };

        // A position is solved when:
        //   - bm set: the engine played one of the best moves
        //   - am set: the engine did NOT play any of the avoid moves
        //   - both set: must satisfy both (strictest)
        //   - neither set: trivially passing (rare in practice)
        let matches_bm = pos.best_moves.iter().any(|bm| matches_move(bm));
        let matches_am = pos.avoid_moves.iter().any(|am| matches_move(am));

        let bm_ok = pos.best_moves.is_empty() || matches_bm;
        let am_ok = pos.avoid_moves.is_empty() || !matches_am;
        let is_correct = bm_ok && am_ok;

        if is_correct {
            passed += 1;
            print!(".");
        } else {
            failed += 1;
            print!("X");
            // Print details for failures — include both bm and am so the
            // classifier is obvious (e.g., "played avoid-move" vs "missed best").
            let mut reason = String::new();
            if !bm_ok { reason.push_str(&format!("not in bm {:?}", pos.best_moves)); }
            if !am_ok {
                if !reason.is_empty() { reason.push_str(" and "); }
                reason.push_str(&format!("matched am {:?}", pos.avoid_moves));
            }
            eprint!("\n  {} FAIL: played {} ({}) — {}",
                pos.id, best_san, best_uci, reason);
        }

        // Flush periodically
        if (i + 1) % 50 == 0 {
            println!(" [{}/{}]", i + 1, total);
        }
    }

    let elapsed = suite_start.elapsed();
    println!("\n\nResults: {}/{} passed ({:.1}%)",
        passed, passed + failed,
        100.0 * passed as f64 / (passed + failed) as f64);
    println!("Total time: {:.1}s", elapsed.as_secs_f64());
}

fn reset_epd_position_state(info: &mut SearchInfo) {
    info.clear_persistent_histories();
    info.tt.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the bug where `coda epd` skipped auto-discovery.
    ///
    /// It was the only subcommand that did not fall back to the shared
    /// `auto_discover_nnue`, so on a `make` build (which embeds a net) it ran
    /// with NO net and panicked inside the search on the first evaluate rather
    /// than reporting anything useful. The contract this locks down: resolution
    /// either loads a net or returns a clean Err — it never leaves `info`
    /// netless while claiming success, and it never panics.
    #[test]
    fn epd_net_resolution_never_leaves_info_netless_on_ok() {
        // Explicit path that cannot load -> clean Err, not a panic.
        let mut info = SearchInfo::new(1);
        let err = resolve_net(&mut info, Some("/nonexistent/definitely-not-a.nnue"));
        assert!(err.is_err(), "a bad explicit -n must be an error");
        assert!(info.nnue_net.is_none(), "failed load must not leave a net behind");

        // No explicit path -> MUST attempt auto-discovery. Whether a net is
        // found depends on the environment, so assert the invariant rather than
        // the outcome: Ok implies a net really is loaded (the old code could
        // reach the search with none), and Err carries a message.
        let mut info2 = SearchInfo::new(1);
        match resolve_net(&mut info2, None) {
            Ok(()) => assert!(
                info2.nnue_net.is_some(),
                "resolve_net returned Ok without actually loading a net"
            ),
            Err(msg) => assert!(!msg.is_empty(), "Err must explain what to do"),
        }
    }

    #[test]
    fn epd_position_reset_clears_persistent_histories_and_tt() {
        let mut info = SearchInfo::new(1);
        info.dirty_persistent_histories_for_test();

        let hash = 0x1234_5678_9abc_def0;
        info.tt.store(hash, 1, 23, crate::tt::TT_FLAG_EXACT, NO_MOVE, 10, false);
        assert!(info.tt.probe(hash).hit, "test setup must seed TT entry");

        reset_epd_position_state(&mut info);

        info.assert_persistent_histories_clear_for_test();
        assert!(!info.tt.probe(hash).hit, "EPD reset must clear TT between positions");
    }
}
