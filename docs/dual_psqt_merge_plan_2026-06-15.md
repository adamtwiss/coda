# Dual-activation + PSQT-outputs merge plan (2026-06-15)

Goal: stack **dual activation** (+2 @ S200) and **PSQT outputs** (neutral @ S200)
on top of the passing **L1=32** path. Both features are conceptually
ORTHOGONAL (dual = L1 output activated both ways `[crelu;screlu]`, doubling L2
input; PSQT = SF-style output skip-connection adding a material/positional term)
— they don't conflict in concept, only their branches touch overlapping files.

## Status snapshot

| Feature | Coda (engine) | Bullet (trainer) |
|---|---|---|
| **Dual activation** | ✅ merged to main (`dual_l1` inference + `--dual` convert flag, end-to-end) | ✅ **MERGED to gpu4 main locally** (`f3019a8`, cherry-picked clean, builds) — **NEEDS PUSH** (HTTPS remote, no creds in headless ssh; `cd ~/code/bullet && git push origin main` from interactive gpu4) |
| **PSQT outputs** | ❌ `feature/psqt-inference` — big STALE fork (26 files, +1243/−2579 vs main; predates dual/L1=32-kernels/threat-work) | ❌ `feature/psqt-outputs` — touches ONLY `coda_v9_768_threats.rs` (+126/−20) + duplicate warmstart commits |

## DONE: dual activation
- Coda main is fully end-to-end (inference `dual_l1`, convert `--dual` → bit-4 v8).
- Bullet: dual commit (`--hidden-activation dual`) cherry-picked onto gpu4 main,
  compiles. **Only remaining step: push** (auth — user/interactive).

## PLAN: PSQT outputs (HOLD — do not merge yet)

PSQT splits into a clean Bullet side and a hard Coda side.

### Bullet side (`feature/psqt-outputs`) — contained, ~easy
The 6 psqt-specific commits (`e1a998e` --psqt flag, `7d134ce` validator,
`1e413f7` recipe fix, `07a65c2` mask-to-PSQ, `d48f7e8` material-prior init,
`115ce4d` concat-of-zeros) touch **only `examples/coda_v9_768_threats.rs`**.
- **Drop** the 2 warmstart commits on the branch (`a1f06ab`, `55bae45`) — they're
  duplicates of what's already on main (`8f4d379`, `493512d`).
- **Conflict to resolve:** the psqt commits and the just-merged dual commit both
  edit `coda_v9_768_threats.rs` (the model-build). Orthogonal but co-located —
  combine the `--psqt` skip-connection and the `--hidden-activation dual` arm.
- **Verify they COMPOSE:** `--psqt` AND `--hidden-activation dual` together in one
  train (the L1=32+dual+psqt target). Confirm the quantised.bin contract handles
  both (dual: l2w is 2*l1×l2; psqt: extra PSQT output head) without colliding.
- Plan: `git rebase --onto main <psqt-base> 115ce4d` keeping only the 6 psqt
  commits; resolve the one-file conflict; `cargo build --example
  coda_v9_768_threats --features cuda`; smoke-train a few SBs.

### Coda side (`feature/psqt-inference`) — the hard part, DON'T rebase the branch
The branch is a stale fork (forked from old main; the −2579 in the diff is
current main's work it LACKS — L1=32 VNNI kernels, dual, the threat-accumulator
work). It touches `nnue.rs` (L1 kernel/dual area), `threat_accum.rs` (the
recapture-combine code), `threats.rs`, `search.rs` — all heavily moved since.
- **Do NOT rebase the whole branch** (conflict storm). Instead **re-apply just the
  3 PSQT-specific commits onto current main**:
  - `1fe2c48` — PSQT skip-connection inference (v11 nets, `--psqt` converter flag,
    reading the PSQT output head in the forward pass). THIS is the bulk — re-implement
    on current `nnue.rs` (load/parse the v11 PSQT bytes + add the PSQT term to the
    output). Current nnue.rs L1/dual/kernel structure differs from the branch's.
  - `8fbd63c` — trainer-parity oracle test (engine PSQT formula == trainer, exact).
    Re-add; it's the correctness gate — must pass on current main.
  - `d6694c5` — #1963 psqt-v3 tunables. **STALE** (calibrated for old main + the
    psqt net). Do NOT apply verbatim; retune after PSQT lands (net-specific tune,
    like #2017 for L1=32).
- The PSQT v11 net format adds a PSQT output head; confirm it composes with v10
  threats + v8 dual (the flags/version byte scheme: threats→v10, dual→bit4; PSQT
  needs its own version/flag that doesn't collide — check `bullet_convert.rs` +
  the version dispatch).

### Sequencing recommendation
1. Push dual (Bullet) — unblocks dual training. (User/interactive.)
2. L1=32 lands (tune #2023 → H1) + Zeus's L1=32 VNNI kernel merges.
3. **Bullet PSQT**: rebase the 6 psqt commits onto main (post-dual), resolve the
   one-file conflict, verify `--psqt`+`--hidden-activation dual` compose.
4. **Coda PSQT**: re-implement `1fe2c48` (+`8fbd63c` oracle) on current main — the
   real work. Skip the stale #1963 tunables.
5. Train **L1=32 + dual + psqt** net; convert (`--dual --psqt` + L1=32 flags);
   SPRT vs prod; net-specific retune; re-SPRT.

### Orthogonality / clash summary
- dual ⟂ psqt ⟂ L1=32 conceptually (L1-activation / output-head / L1-width) —
  all three stackable.
- File clashes are branch-integration only: Bullet `coda_v9_768_threats.rs` (dual
  + psqt co-located); Coda `nnue.rs` version/flag dispatch (v10 threats + v8 dual
  + v11 psqt must coexist) + the stale-fork rebase of psqt-inference.
- Stale-tunables trap: the psqt #1963 tune is calibrated for old trunk — retune,
  don't reuse.
