// Quick std::mem::size_of probe for hot structs.
// Run with: rustc -O scripts/sizeof_probe.rs -o /tmp/sizeof_probe && /tmp/sizeof_probe
// Not part of the build; ad-hoc tool only.
use std::mem::{size_of, align_of};

#[repr(C, align(64))]
struct ThreatEntryLike {
    values: [[i16; 768]; 2],            // 3072
    accurate: [bool; 2],                // 2
    delta_data: [u32; 128],             // 512
    delta_len: usize,                   // 8
    delta_overflowed: bool,             // 1
    mv: u16,                            // 2
    moved_pt: u8,                       // 1
    moved_color: u8,                    // 1
}

#[repr(C)]
struct ThreatMetaProposal {
    accurate: [bool; 2],     // 2
    delta_overflowed: bool,  // 1
    moved_pt: u8,            // 1
    moved_color: u8,         // 1
    _pad1: u8,               // 1
    mv: u16,                 // 2
    delta_len: u32,          // 4 — total 12 bytes, naturally aligned to 4
}

#[repr(C, align(64))]
struct ThreatValuesProposal {
    values: [[i16; 768]; 2],
    delta_data: [u32; 128],
}

fn main() {
    println!("Current ThreatEntry: size={} align={}",
        size_of::<ThreatEntryLike>(), align_of::<ThreatEntryLike>());
    println!("Proposed ThreatMeta:  size={} align={}",
        size_of::<ThreatMetaProposal>(), align_of::<ThreatMetaProposal>());
    println!("Proposed ThreatValues: size={} align={}",
        size_of::<ThreatValuesProposal>(), align_of::<ThreatValuesProposal>());

    let n = 256;
    println!();
    println!("--- 256-ply stack totals ---");
    println!("Today:  {} bytes ({:.1} KB)", n*size_of::<ThreatEntryLike>(), (n*size_of::<ThreatEntryLike>()) as f64 / 1024.0);
    let split_total = n*size_of::<ThreatMetaProposal>() + n*size_of::<ThreatValuesProposal>();
    println!("Split:  {} bytes ({:.1} KB)  metas:{}  values:{}",
        split_total, split_total as f64 / 1024.0,
        n*size_of::<ThreatMetaProposal>(),
        n*size_of::<ThreatValuesProposal>());

    let cur_meta_offset_estimate = 3072 + 512;  // values + delta_data come before metadata
    println!();
    println!("--- can_update walk per ensure_computed (5-14 plies typical) ---");
    println!("Today metadata access stride: {} bytes (full ThreatEntry)", size_of::<ThreatEntryLike>());
    println!("  → 14 plies = {} cache lines minimum (if not already resident)", 14);
    println!("Split metadata access stride: {} bytes per entry", size_of::<ThreatMetaProposal>());
    println!("  → 14 plies = {:.1} cache lines (likely 1 prefetched line)",
        (14 * size_of::<ThreatMetaProposal>()) as f64 / 64.0);
}
