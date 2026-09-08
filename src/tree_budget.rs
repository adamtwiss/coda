//! Diagnostic event ledger. Enable CODA_TREE_BUDGET=1; UCI `treebudget`
//! dumps and clears. Use Threads=1, fixed nodes, not time or NPS comparisons.
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use crate::board::Board;
use std::cell::Cell;
thread_local! { static OWNER: Cell<&'static str> = const { Cell::new("owned_primary") }; }
pub struct Scope(&'static str);
impl Drop for Scope { fn drop(&mut self) { OWNER.with(|o| o.set(self.0)); } }
pub fn scope(name: &'static str) -> Scope {
    let old = OWNER.with(|o| { let old=o.get(); if old=="owned_primary" {o.set(name);} old });
    Scope(old)
}
pub fn charge(key: Key) { if enabled() { OWNER.with(|o| add(key,o.get(),1)); } }

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
// Diagnostic intervention, independent of ledger enablement. Protect only
// eligible PV/TT moves crossing to QS; do not globally extend the frontier.
pub fn pv_depth(key: Key, depth: i32, eligible: bool, decisive: bool) -> i32 {
    if depth > 0 || !eligible { return depth; }
    add(key, "tt_pv_frontier_candidate", 1);
    if decisive { add(key, "tt_pv_frontier_decisive", 1); }
    static FLOOR: OnceLock<bool> = OnceLock::new();
    if *FLOOR.get_or_init(|| std::env::var_os("CODA_PROBE_PV_FLOOR").is_some()) { 1 } else { depth }
}
pub fn dump() {
    for ((key,event), count) in std::mem::take(&mut *ledger().lock().unwrap()) {
        println!("TREEBUDGET {} {} {} {} {}", key.0,key.1,key.2,event,count);
    }
}
