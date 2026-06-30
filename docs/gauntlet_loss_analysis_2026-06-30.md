# Gauntlet loss analysis — "why does Coda lose?" (2026-06-30)

First-pass mining of the **losing** games in `conversion_gauntlet.pgn` (3800
games, Coda v8s3 vs 18 top-20 engines, STC 10+0.1, no-resign). The earlier
`conversion_failure_study` mined *draws of won positions* (→ the +10 Elo endgame
mop-up); this mines the **295 losses**, an untouched seam.

**Method (first pass):** in-PGN evals only — uses Coda's *own* reported eval
trajectory (reliable for "Coda thinks it blundered"). Mate-scores clamped to
±800 so the inevitable end-of-game collapse doesn't masquerade as a blunder. An
SF-gated deep dive (confirm the refutation, name the missed motif) is the
follow-up, not yet done.

## Headline

Coda's gauntlet record is **net-positive: 415 W / 295 L / 3081 D** vs this
top-20 field (≈+120 decisive) — it is competitive, not outclassed.

**Losses split ~50/50 between two distinct failure modes:**

| mode | count | share | lever |
|---|---|---|---|
| **Tactical crash** (≥300cp single-move eval drop, mate-excluded) | 147 | 49% | search / tactics (depth, pruning, missed shot) |
| **Gradual decline** (no single big drop — slowly outplayed) | 148 | 50% | positional eval blindspot (training) |
| of which: was **winning (+1.5)** then lost it | 12 | — | giveaways |

## The recoverable Elo: losses to *weaker* engines

**140 of 295 losses (47%) are to engines rated BELOW Coda** (Stormphrax,
Viridithas, Astra, Tarnished, Starzix, Motor, Halogen, Quanticade, Hobbes,
Cinder…). Losses to SF / Reckless / PlentyChess (the +100-Elo field) are
expected; losses to weaker engines are where the cheap Elo is. The most diagnostic
(Coda was clearly winning, then a big single-move crash vs a weaker engine):

```
peak +2.1 -> drop 1001cp  vs Astra      (catastrophic give-away of a won game)
peak +2.7 -> drop  359cp  vs Motor
peak +2.6 -> drop  436cp  vs Starzix
peak +1.9 -> drop  391cp  vs Viridithas
peak +1.6 -> drop  400cp  vs Stormphrax
```

## Reading

- The **tactical half (49%)** is a *search* problem — Coda reaches a position,
  plays a move, and its own eval admits a ≥3-pawn refutation one ply later. That
  means the refutation was beyond its effective horizon (depth / over-aggressive
  pruning on the critical line), not an eval-label problem. This is the same class
  as the `RpZ9LbYM`-style quiet-refutation blindspot — and it's *fixable in
  search*, distinct from the eval-flywheel/training work.
- The **gradual half (50%)** is the harder, eval-side problem: no admitted
  blunder, the position just deteriorates — an eval that doesn't perceive slow
  positional deterioration. This feeds the training corpus (Atlas), not a quick fix.

## Next (SF-gated deep dive — proposed)

1. SF-gate the **weaker-engine tactical crashes** (~highest value): confirm the
   refuting move, classify the missed motif (back-rank, fork, pin, deflection,
   undefended-piece). A recurring motif → a targeted search/ordering or eval fix.
2. SF-gate a sample of the **gradual** losses: are they a recurring structure
   (e.g. specific pawn formations, opposite-side attacks) → training-cluster fuel.
3. Re-run on a **fresh current-main gauntlet** (post-mop-up, net E6C62000) to
   confirm the split holds on the shipped engine.

## Caveats
- Opponent/own-eval based, not SF-confirmed — directional, magnitudes approximate.
- v8s3 net + pre-mop-up main (the mop-up only touched lone-king draws, not losses,
  so the loss picture is essentially current — but a fresh gauntlet would confirm).
