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

thread_local! { static VERIFY_NEXT: Cell<u8> = const { Cell::new(0) }; }
pub fn verify_next(trace: bool) { VERIFY_NEXT.with(|v| v.set(if trace {2} else {1})); }
pub struct ReturnAudit { active: bool, trace: bool, pub source: &'static str, ply: i32, depth: i32 }
pub fn return_audit(ply: i32, depth: i32) -> ReturnAudit {
    let mode = VERIFY_NEXT.with(|v| v.replace(0));
    ReturnAudit { active: mode != 0, trace: mode == 2, source: "other", ply, depth }
}
impl ReturnAudit {
    pub fn skip_rfp(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        if !self.active || !*ON.get_or_init(|| std::env::var_os("CODA_VERIFY_NO_ROOT_RFP").is_some()) { return false; }
        // Optional single-event intervention: preserve the entire baseline
        // prefix up to the first traced verifier at this entry depth.
        static TARGET: OnceLock<Option<i32>> = OnceLock::new();
        thread_local! { static USED: Cell<bool> = const { Cell::new(false) }; }
        match TARGET.get_or_init(|| std::env::var("CODA_VERIFY_ONLY_TRACE_DEPTH").ok().map(|s|s.parse().expect("integer trace depth"))) {
            None => true,
            Some(d) => self.trace && self.depth == *d && USED.with(|v|!v.replace(true)),
        }
    }
}
pub fn verify_depth(depth: i32) -> i32 {
    static ON: OnceLock<bool> = OnceLock::new();
    depth + if *ON.get_or_init(|| std::env::var_os("CODA_VERIFY_PLUS_TWO").is_some()) { 2 } else { 0 }
}
impl Drop for ReturnAudit {
    fn drop(&mut self) {
        if self.trace { eprintln!("TRACE verify_return ply={} depth={} source={}", self.ply,self.depth,self.source); }
    }
}
pub fn pvs(key: Key, alpha: i32, beta: i32, score: i32, stopped: bool, nodes: u64) {
    let outcome = if stopped { "pvs_stopped" } else if score <= alpha { "pvs_reversed" }
        else if score >= beta { "pvs_cutoff" } else { "pvs_confirmed" };
    add(key,outcome,1);
    let cost = match outcome { "pvs_stopped"=>"pvs_stopped_nodes", "pvs_reversed"=>"pvs_reversed_nodes",
        "pvs_cutoff"=>"pvs_cutoff_nodes", _=>"pvs_confirmed_nodes" };
    add(key,cost,nodes);
}

thread_local! {
    static ROOT_MOVE: Cell<u16> = const { Cell::new(0) };
    static EXTENSIONS: Cell<u64> = const { Cell::new(0) };
}
pub fn root_move(mv: u16) { ROOT_MOVE.with(|v| v.set(mv)); }
pub fn extensions(n: i32) { if n>0 { EXTENSIONS.with(|v| v.set(v.get()+n as u64)); } }
pub struct PvsAudit { prefix: Option<String>, nodes: u64, extensions: u64 }
pub fn pvs_begin(hash: u64, mv: u16, root_depth: i32, ply: i32, scout: i32,
    scout_depth: i32, full_depth: i32, alpha: i32, beta: i32, nodes: u64) -> PvsAudit {
    static ON: OnceLock<bool> = OnceLock::new();
    let prefix = if *ON.get_or_init(|| std::env::var_os("CODA_PVS_AUDIT").is_some()) {
        Some(format!("PVSAUDIT child={} move={} rootmove={} iteration={} ply={} scout={} scout_depth={} full_depth={} alpha={} beta={} start={} owner={}",
            hash,mv,ROOT_MOVE.with(|v|v.get()),root_depth,ply,scout,scout_depth,full_depth,alpha,beta,nodes,OWNER.with(|v|v.get())))
    } else { None };
    PvsAudit { prefix, nodes, extensions: EXTENSIONS.with(|v|v.get()) }
}
impl PvsAudit {
    pub fn finish(self, score: i32, nodes: u64, stopped: bool) {
        if let Some(prefix)=self.prefix { eprintln!("{} full={} cost={} extensions={} stopped={}",
            prefix,score,nodes-self.nodes,EXTENSIONS.with(|v|v.get())-self.extensions,stopped); }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verification_marker_belongs_to_exact_next_call() {
        verify_next(false);
        let root=return_audit(3,8);
        assert!(root.active);
        let child=return_audit(4,7);
        assert!(!child.active);
        // Same-ply singular searches must not inherit verifier-root status.
        let singular=return_audit(3,3);
        assert!(!singular.active);
        assert!(root.active);
    }
    #[test]
    fn nested_verifier_does_not_change_outer_provenance() {
        verify_next(false);
        let mut outer=return_audit(3,8);
        outer.source="moves";
        verify_next(false);
        let mut inner=return_audit(9,2);
        inner.source="rfp";
        drop(inner);
        assert_eq!(outer.source,"moves");
        assert!(outer.active);
    }
}
