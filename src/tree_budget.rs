//! Diagnostic event ledger. Enable CODA_TREE_BUDGET=1; UCI `treebudget`
//! dumps and clears. Use Threads=1, fixed nodes, not time or NPS comparisons.
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use crate::board::Board;

pub type Key = (u8, u8, u8);
type Ledger = BTreeMap<(Key, &'static str), u64>;
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CODA_TREE_BUDGET").is_some())
}
fn ledger() -> &'static Mutex<Ledger> {
    static DATA: OnceLock<Mutex<Ledger>> = OnceLock::new();
    DATA.get_or_init(|| Mutex::new(BTreeMap::new()))
}
pub fn key(board: &Board, ply: i32, depth: i32) -> Key {
    if !enabled() { return (0,0,0); }
    let pieces = board.occupied().count_ones();
    let phase = if pieces <= 7 { 0 } else if pieces <= 12 { 1 } else if pieces <= 20 { 2 } else { 3 };
    let p = if ply == 0 { 0 } else if ply <= 4 { 1 } else if ply <= 8 { 2 } else if ply <= 16 { 3 } else { 4 };
    let d = if depth <= 0 { 0 } else if depth <= 3 { 1 } else if depth <= 7 { 2 } else if depth <= 15 { 3 } else { 4 };
    (phase,p,d)
}
pub fn add(key: Key, event: &'static str, amount: u64) {
    if enabled() { *ledger().lock().unwrap().entry((key,event)).or_default() += amount; }
}
pub fn dump() {
    for ((key,event), count) in std::mem::take(&mut *ledger().lock().unwrap()) {
        println!("TREEBUDGET {} {} {} {} {}", key.0,key.1,key.2,event,count);
    }
}
