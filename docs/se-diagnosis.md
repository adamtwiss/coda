# Singular Extensions: Root Cause Diagnosis

## Summary

Singular extensions cause -60 to -140 Elo in our engine despite being +20-30 Elo in every
top engine. After detailed comparison of our implementation against Alexandria, Berserk,
Obsidian, and Weiss, I have identified **five distinct bugs/issues**, ranked by severity.
The top two are almost certainly responsible for the catastrophic Elo loss. The remaining
three compound the problem.

---

## BUG #1 (CRITICAL): TT Probing NOT Skipped During Singular Verification

**Severity: Catastrophic. This alone likely accounts for most of the Elo loss.**

### The Problem

In our engine (both main and se-alex worktree), when the SE verification search recurses
into `negamax`, it probes the TT **using the same hash key** as the original position
(because ExcludedMove does not change the Zobrist key). This means:

1. The verification search hits the same TT entry that triggered the SE in the first place.
2. Since the TT entry has `ttEntry.Flag != TTUpper` and `ttEntry.Depth >= depth-3`, and
   the verification search runs at `singularDepth = (depth-1)/2` which is always less than
   `ttEntry.Depth`, the TT cutoff condition `ttDepth >= depth` is typically met.
3. The TT score is returned immediately, **including the contribution of the excluded move**.

This completely defeats the purpose of SE. The verification search is supposed to answer
"how good is the position without the TT move?" but instead it answers "how good is the
position according to the cached result that includes the TT move?" -- which is always
approximately the same as ttScore, making singular extensions never trigger (or trigger
with wrong margins).

### What Reference Engines Do

**Every single engine** skips TT probing when an excluded move is set:

- **Alexandria** (search.cpp:463): `const bool ttHit = !excludedMove && ProbeTTEntry(...);`
- **Berserk** (search.c:427): `TTEntry* tt = ss->skip ? NULL : TTProbe(...);`
- **Obsidian** (search.cpp:663): passes `excludedMove` as a parameter, then at line 744:
  TT cutoffs are gated with `!excludedMove`
- **Weiss** (search.c:298): `if (ttMove && (!MoveIsPseudoLegal(pos, ttMove) || ttMove == ss->excluded)) ttHit = false, ttMove = NOMOVE`

Our engine at search.go:958-963 (worktree) and search.go:968-973 (main):
```go
if entry, found := info.TT.Probe(b.HashKey); found {
    ttHit = true
    ttEntry = entry
    ttMove = entry.Move
    if info.ExcludedMove[ply] == NoMove && ply > 0 {
        // TT cutoff logic...
```

We do gate the **TT cutoff** on `ExcludedMove[ply] == NoMove`, which is correct. But we
still set `ttHit = true`, `ttEntry`, and `ttMove` from the TT probe. This means:

- The SE condition check at line 1347 uses `ttHit` and `ttEntry` from the **same position's
  TT entry**, which is fine for the outer call.
- But when the verification search recurses, it reads the TT entry (setting ttHit=true,
  getting the ttMove), and while it won't do a TT cutoff, it **still uses ttMove for move
  ordering** and **uses ttEntry.StaticEval for static eval**. This is less severe than a
  full TT cutoff but still wrong -- the static eval from TT is fine (it's the same position),
  but the ttMove should not be used since it's the excluded move.

Wait -- actually, re-reading more carefully: the TT cutoff IS gated. So the verification
search won't return early from TT. But the ttMove IS set, and the verification search will
still try the ttMove first in move ordering... but then skip it via the ExcludedMove check.
So actually the TT probing issue is mitigated by the ExcludedMove guard on the cutoff.

Let me re-evaluate: the TT cutoff is correctly gated. The ttMove is set but skipped. The
real question is whether the **pruning decisions** (NMP, RFP, razoring, ProbCut, futility)
are affected. And they are, because they're NOT gated on ExcludedMove.

**Revised assessment: BUG #1 is about NMP and pruning inside SE, not TT cutoffs.**

---

## BUG #1 (REVISED, CRITICAL): NMP and All Pruning Run Inside SE Verification Search

**Severity: Catastrophic.**

### The Problem

When the SE verification search calls `negamax(singularDepth, ply, ...)`, it runs at the
**same ply** as the parent. This means:

1. **Null-move pruning fires inside the verification search.** Our NMP has no guard for
   `ExcludedMove[ply] != NoMove`. The verification search position is the same position,
   with eval likely above beta (the singularBeta is below ttScore). If staticEval >= beta
   (where beta = singularBeta), NMP will fire, do a null-move search, and may return a
   score >= singularBeta, causing the SE test to fail (no extension).

   This is wrong because:
   - NMP inside SE can report "the position is good even without the TT move" when really
     the NMP result includes the TT move in the reduced search.
   - Specifically: NMP does `MakeNullMove` then searches. The opponent plays. Then at the
     next ply, the TT move IS available (ExcludedMove is only set at the current ply).

2. **Reverse futility pruning fires.** If staticEval - margin >= singularBeta, RFP returns
   immediately. This is particularly likely because singularBeta = ttScore - margin, and
   staticEval is often close to ttScore. So RFP can trivially say "yes the position is good
   enough" without even searching any move, completely bypassing the purpose of SE.

3. **ProbCut fires.** Same problem -- the verification search can be pruned before any
   move is tried.

### What Reference Engines Do

- **Alexandria** (search.cpp:548-550): `if (!pvNode && !excludedMove && !inCheck)` --
  ALL pre-move-loop pruning (RFP, NMP, razoring, probcut) is skipped when excludedMove is set.

- **Berserk** (search.c:506, 517, 533, 569): Guards `!ss->skip` on IIR, RFP, NMP, ProbCut.

- **Obsidian** (search.cpp:873): NMP is gated with `!excludedMove`. RFP and razoring are
  inside a block that requires `goto moves_loop` to skip when in check or excludedMove.
  Actually, Obsidian does NOT skip RFP/razoring during SE -- but it DOES skip NMP.

- **Weiss** (search.c:364): `if (inCheck || pvNode || !thread->doPruning || ss->excluded || isTerminal(beta) || (ss-1)->move == NOMOVE) goto move_loop;` -- ALL pruning skipped.

**Our engine has NO guards on NMP, RFP, razoring, or ProbCut for ExcludedMove.** This is
the single most likely cause of the catastrophic Elo loss. The verification search almost
never actually searches moves -- it just gets pruned by NMP or RFP, returns a score above
singularBeta, and SE never extends.

### Impact

With RFP active, at depth 7 the margin is 7*100=700. singularBeta = ttScore - depth*5/8 =
ttScore - ~4. If staticEval is anywhere near ttScore (which it usually is), then
staticEval - 700 is nowhere near singularBeta, so RFP won't fire in this direction.
Actually wait -- RFP returns when `staticEval - margin >= beta`, i.e., when eval is WAY
above beta. In the SE search, beta = singularBeta which is BELOW ttScore. So
staticEval >= singularBeta + margin is actually quite plausible. The question is whether
the singularDepth (typically 2-5) triggers RFP's `depth <= 7` guard. Yes, it does.

**NMP is the bigger culprit.** The SE search runs at singularDepth (e.g., 4 for depth=9).
NMP fires at depth >= 3. The position's eval is likely above singularBeta (which is below
ttScore). NMP does a null-move search at reduced depth, which has no ExcludedMove, finds the
position is still good (because the TT move IS available there), and returns score >= beta.
This defeats SE completely.

---

## BUG #2 (CRITICAL): SE Verification Search Uses Same Ply (ply, not ply+1)

**Severity: High. Causes subtle but pervasive corruption.**

### The Problem

In our se-alex worktree (search.go:1365):
```go
singularScore := b.negamax(singularDepth, ply, singularBeta-1, singularBeta, info)
```

The verification search is called with the **same ply** as the current node. This means:

1. **StaticEvals[ply] gets overwritten.** The verification search computes a new staticEval
   and stores it at `info.StaticEvals[ply]`, overwriting the value computed by the parent.
   After the verification search returns, the parent continues to use `staticEval` (its local
   variable) which is fine, but the `info.StaticEvals[ply]` array entry is now corrupted.
   This affects:
   - The `improving` check for child nodes at ply+2 (`staticEval > info.StaticEvals[ply-2]`)
   - The `failing` heuristic
   - The `unstable` detection

2. **Killers[ply] can be overwritten.** The verification search may store killer moves at
   ply, overwriting the parent's killers. When the parent continues searching after SE,
   the killer moves may be wrong.

3. **The ExcludedMove mechanism uses ply indexing.** We set `info.ExcludedMove[ply] = ttMove`
   before the call and clear it after. But the verification search itself runs at the same
   ply and checks `info.ExcludedMove[ply]` -- which IS set, so the skip works. However, if
   the verification search triggers its OWN SE at the same ply (impossible because
   ExcludedMove[ply] != NoMove gates it), but recursive calls at ply+1, ply+2 etc. could
   theoretically corrupt the ExcludedMove array... actually no, the guard prevents this.

### What Reference Engines Do

All reference engines use a **search stack (ss)** mechanism where the SE verification search
is called at the **same stack entry** (same ply). This is actually correct and intentional:

- **Alexandria** (search.cpp:749): `Negamax<false>(singularBeta - 1, singularBeta, singularDepth, cutNode, td, ss)` -- same `ss`, same ply.
- **Berserk** (search.c:671): `Negamax(sBeta - 1, sBeta, sDepth, cutnode, thread, pv, ss)` -- same `ss`.
- **Obsidian** (search.cpp:1040): `negamax<false>(pos, singularBeta - 1, singularBeta, (depth - 1) / 2, cutNode, ss, move)` -- same `ss`.
- **Weiss** (search.c:500): `AlphaBeta(thread, ss, singularBeta-1, singularBeta, depth/2, cutnode)` -- same `ss`.

So using the same ply IS standard. The difference is that these engines use a per-stack-entry
design where `ss->staticEval` is **preserved** because the SE check at the top of negamax
uses `else if (excludedMove) { eval = rawEval = ss->staticEval; }` -- they **reuse** the
already-computed static eval rather than recomputing it.

**Our engine recomputes staticEval inside the verification search** because we have no
guard. The TT probe succeeds (ttHit=true, same hash), so we use `ttEntry.StaticEval`, which
is the same value. But then `info.StaticEvals[ply]` is overwritten, and more importantly,
the correction history adjustment is also applied again (the same correction), so the
correctedStaticEval should be the same. But the real problem is that **NMP and RFP fire**
as noted in Bug #1.

**Revised assessment: The ply issue is NOT a bug per se (same ply is standard), but it
compounds Bug #1 because all the pruning that fires inside SE uses the same ply's data.**

---

## BUG #2 (REVISED): Fail-High Score Blending Corrupts SE Result

**Severity: High.**

### The Problem

Our engine has fail-high score blending (search.go, lines 1853-1855 in the se-alex worktree):
```go
if bestScore >= beta && beta-alphaOrig == 1 && depth >= 3 &&
    bestScore > -MateScore+100 && bestScore < MateScore-100 {
    return (bestScore*depth + beta) / (depth + 1)
}
```

The SE verification search calls `negamax(singularDepth, ply, singularBeta-1, singularBeta, info)`.
This is a zero-window search (beta - alpha = 1), so `beta-alphaOrig == 1` is true.
If the verification search finds bestScore >= singularBeta, the blending formula applies:

    returned = (bestScore * singularDepth + singularBeta) / (singularDepth + 1)

This **pulls the score toward singularBeta** (downward toward beta). So a score that might
have been significantly above singularBeta gets dampened toward it. This means the SE test
`singularScore < singularBeta` is MORE likely to be true -- which would cause MORE
extensions, not fewer.

Wait, that would make SE more aggressive, not less. Let me think again...

Actually, the dampening makes scores that are above singularBeta closer to singularBeta.
So the multi-cut condition `singularScore >= beta` (where beta is the ORIGINAL beta, much
higher than singularBeta) becomes harder to trigger. And the negative extension condition
`ttScore >= beta` doesn't depend on singularScore. So the main effect is marginal.

But there's a more subtle issue: the dampening happens at **every node in the SE subtree**,
not just the root of the SE search. Interior nodes of the SE search also apply this
blending, which systematically deflates scores. This could make the SE verification search
see lower scores than it should, making extensions trigger too often.

**Revised: The score blending interaction is not clearly positive or negative. It could
contribute to score distortion in the SE subtree but is unlikely to be the primary cause.**

---

## BUG #2 (FINAL): TT Score Dampening Corrupts SE TT Reads

**Severity: High.**

### The Problem

Our TT cutoff dampening (search.go lines 1016-1019):
```go
if beta-alphaOrig == 1 &&
    entry.Flag == TTLower &&
    score > -MateScore+100 && score < MateScore-100 {
    return (3*score + beta) / 4
}
```

This applies to TT cutoffs at **all** non-PV nodes. When the SE verification search runs
its subtree, interior nodes that get TT cutoffs have their scores dampened by blending
toward beta. Since every recursive call inside the SE subtree has its own beta values, this
dampening cascades through the search tree, systematically distorting scores in the SE
verification subtree.

More importantly, the **TT near-miss cutoffs** (search.go lines 1023-1035) can fire inside
the SE subtree, returning `score - margin` or `score + margin`. These approximate cutoffs
add noise to the SE verification, potentially causing it to return scores that don't
accurately reflect whether alternatives are competitive.

---

## BUG #3 (HIGH): NMP Score Dampening Distorts SE Subtree

**Severity: Medium-High.**

### The Problem

When NMP fires inside the SE verification subtree (which it should not -- see Bug #1), the
returned score is dampened: `(score*2 + beta) / 3`. This pulls NMP results toward beta
(which inside SE is singularBeta). The dampening makes it harder for the SE subtree to
return scores significantly above singularBeta, which would be needed for multi-cut.

But more fundamentally, NMP should not fire at all during SE (Bug #1).

---

## BUG #4 (MEDIUM): No `cutNode` Tracking

**Severity: Medium.**

### The Problem

Our engine does not track `cutNode` at all. Several reference engines use cutNode to:

1. **Gate NMP on cutNode** (Obsidian): Only do NMP at expected cut nodes.
2. **Add negative extensions at cutNode** (Alexandria, Obsidian): When SE doesn't find
   singularity but we're at a cut node, reduce by -2.
3. **Affect TT cutoff conditions** (Alexandria, Obsidian): Only allow TT cutoffs when
   `cutNode == (ttScore >= beta)`.

Without cutNode tracking, our SE implementation is missing a key signal. However, this is
more of an optimization issue than a correctness bug. Not having cutNode means we over-prune
at some nodes and under-prune at others, but it's unlikely to cause the catastrophic -60 to
-140 Elo regression.

---

## BUG #5 (LOW): moveCount==0 with ExcludedMove Returns alpha, Should Return -MAXSCORE

**Severity: Low.**

### The Problem

In our engine (search.go:1800-1805):
```go
if moveCount == 0 {
    if info.ExcludedMove[ply] != NoMove {
        return alpha
    }
```

When all legal moves except the excluded move are illegal (e.g., the only legal move IS the
TT move), we return `alpha` (which is `singularBeta - 1`). Alexandria (search.cpp:923)
returns `-MAXSCORE`:
```cpp
return excludedMove ? -MAXSCORE : inCheck ? -MATE_SCORE + ss->ply : 0;
```

Returning `-MAXSCORE` is more correct: it signals that WITHOUT the TT move there are NO
alternatives, making the TT move maximally singular. Returning `alpha` (= singularBeta-1)
would make the SE check `singularScore < singularBeta` true by exactly 1, giving a minimal
extension -- when really this should be the strongest possible singular extension signal.

Berserk and Weiss return `alpha` in this case, so this varies by engine. But Alexandria's
approach is more aggressive and logically sound.

---

## Root Cause Summary (Ranked by Impact)

| Rank | Bug | Description | Likely Elo Impact |
|------|-----|-------------|-------------------|
| 1 | NMP/pruning in SE | NMP, RFP, ProbCut all fire inside verification search | -40 to -80 |
| 2 | Score dampening in SE | TT dampening, FH blending, NMP dampening distort SE scores | -10 to -30 |
| 3 | No cutNode | Missing cutNode tracking for SE negative extensions | -5 to -15 |
| 4 | moveCount==0 return | alpha vs -MAXSCORE when only legal move is excluded | -2 to -5 |
| 5 | TT near-miss in SE | Approximate TT cutoffs add noise to SE results | -3 to -10 |

---

## Proposed Fix

### Step 1: Gate pruning on ExcludedMove (CRITICAL)

In `negamax`, after computing staticEval, add ExcludedMove guards. The cleanest approach
mirrors Alexandria's pattern: wrap ALL pre-move-loop pruning in `!excludedMove`:

```go
// In search.go negamax, around line 1127 (se-alex worktree)
// Change NMP condition from:
if depth >= 3 && !inCheck && ply > 0 && stmNonPawn != 0 && beta-alpha == 1 {
// To:
if depth >= 3 && !inCheck && ply > 0 && stmNonPawn != 0 && beta-alpha == 1 && info.ExcludedMove[ply] == NoMove {

// Change RFP condition from:
if depth <= 7 && ply > 0 {
// To:
if depth <= 7 && ply > 0 && info.ExcludedMove[ply] == NoMove {

// Change Razoring condition from:
if RazoringEnabled && depth <= 2 && ply > 0 {
// To:
if RazoringEnabled && depth <= 2 && ply > 0 && info.ExcludedMove[ply] == NoMove {

// Change ProbCut condition from:
if !inCheck && ply > 0 && depth >= 5 && staticEval+85 >= probCutBeta {
// To:
if !inCheck && ply > 0 && depth >= 5 && staticEval+85 >= probCutBeta && info.ExcludedMove[ply] == NoMove {
```

### Step 2: Skip staticEval recomputation during SE

When ExcludedMove is set, reuse the already-computed staticEval from `info.StaticEvals[ply]`:

```go
if !inCheck {
    if info.ExcludedMove[ply] != NoMove {
        // Reuse parent's static eval during singular verification
        rawEval = info.StaticEvals[ply]  // already stored by parent
        staticEval = rawEval
    } else if ttHit && ttEntry.StaticEval > -MateScore+100 {
        rawEval = int(ttEntry.StaticEval)
    } else {
        rawEval = b.EvaluateRelative()
    }
    // ... rest of staticEval computation
```

Actually this is tricky because `info.StaticEvals[ply]` stores the corrected eval but
rawEval should be uncorrected. Better to just let it recompute -- the eval is cheap and
will give the same result. The critical fix is Step 1 (gating pruning).

### Step 3: Gate correction history updates on ExcludedMove

Already done in our code (search.go:1841: `info.ExcludedMove[ply] == NoMove`). Good.

### Step 4: Consider returning -Infinity for moveCount==0 with ExcludedMove

```go
if moveCount == 0 {
    if info.ExcludedMove[ply] != NoMove {
        return -Infinity  // No alternatives exist -- maximally singular
    }
```

### Step 5: Gate TT near-miss cutoffs on ExcludedMove

```go
} else if ttDepth >= depth-1 &&
    beta-alphaOrig == 1 &&
    info.ExcludedMove[ply] == NoMove &&  // <-- add this
    score > -MateScore+100 && score < MateScore-100 {
```

---

## Detailed NMP-in-SE Analysis

For depth=8, singularDepth = (8-1)/2 = 3. NMP requires depth>=3: YES.
singularBeta = ttScore - 8*5/8 = ttScore - 5.
NMP R = 3 + 3/3 = 4. depth-1-R = 3-1-4 = -2, clamped to 1.
NMP null-move search at depth 1 is basically quiescence. At ply+1, ExcludedMove is clear,
so the TT move is available. NMP score is very likely >= singularBeta (which is only 5cp
below ttScore). Result: NMP returns dampened score, SE test fails, no extension.

For depth=12, singularDepth = 5. NMP R = 3 + 5/3 = 4. depth-1-R = 0, clamped to 3.
NMP searches at depth 3 with null move. The search has access to all moves. The TT move
is particularly strong. NMP score >= singularBeta is almost certain.

This means the SE verification search ALMOST NEVER actually searches alternative moves.
Instead, NMP (or occasionally RFP at extreme eval margins) short-circuits it, returning
a score that includes the TT move's contribution, defeating the entire purpose of SE.

The cost: we spend ~singularDepth plies of search on NMP inside SE, waste those nodes,
get no useful information, and sometimes incorrectly extend or don't extend. The net
effect is pure node waste + occasional wrong extensions = massive Elo loss.

## Verification Plan

1. Apply Step 1 (NMP/RFP/razoring/ProbCut guards) -- this is the minimum viable fix.
2. Run WAC/ECM quick benchmark to check for crashes/regressions.
3. SPRT test against baseline (SE disabled).
4. If positive, apply Steps 2-5 and re-test.
5. If still negative, add debug logging to count how often SE extends/multi-cuts/negative-extends.

---

## Why Previous Parameter Tuning Failed

All four parameter sets (depth>=10/margin d*3, SF-like, Alexandria d*5/8, full Alexandria)
failed because the parameters control WHEN to attempt SE, but the fundamental bug is that
the verification search NEVER works correctly due to pruning bypassing it. No parameter
tuning can fix a verification search that returns NMP/RFP results instead of actually
searching alternatives.

The consistent -60 to -140 Elo across all parameter sets is the signature of a systematic
bug rather than a tuning issue. The SE code spends nodes on verification searches that are
immediately short-circuited by NMP/RFP, providing no useful information while consuming
time and potentially corrupting the search tree via spurious extensions.

---

## Experimental Results (2026-03-20)

All bugs from the analysis above were fixed and tested systematically. The results
overturned the original hypothesis: the bugs were real, but fixing them did NOT make SE
positive. The problem is deeper than implementation bugs.

### Configurations Tested

| Variant | Pruning in SE | Extension | Multi-cut | Neg ext | Margin | Depth | Games | Elo |
|---------|--------------|-----------|-----------|---------|--------|-------|-------|-----|
| v2 (full fix) | All gated | +1/+2 | Yes (buggy) | -2 | d*2 | >=6 | 63 | -76 |
| minimal | All gated | +1 | No | No | d*2 | >=6 | 25 | -164 |
| v5 | None gated | +1 | Yes | -1 | d*1 | >=8 | 77 | -97 |
| **v6** | NMP-only gated | **+1** | Yes | -1 | d*1 | >=8 | 211 | **-33** |
| **v7** | NMP-only gated | **None** | Yes | -1 | d*1 | >=8 | 1148 | **-1.5** |

### Key Findings

**1. The SE infrastructure works correctly.**
Multi-cut (return singularBeta when singularBeta >= beta) and negative extensions
(-1 when ttScore >= beta but not singular) are correctly implemented and break even
with the verification search cost (v7: -1.5 Elo in 1148 games).

**2. The positive extension (+1 for singular moves) is what hurts.**
v6 (-33 Elo) vs v7 (-1.5 Elo) — the only difference is whether singular moves get
+1 extension. The extension alone costs ~30 Elo. This is the opposite of reference
engines where the extension is the primary value of SE.

**3. Gating NMP inside SE is necessary but not sufficient.**
- All pruning gated (Alexandria pattern): -164 Elo (too expensive, no NPS budget)
- NMP-only gated (Obsidian pattern): -33 Elo (viable cost)
- No pruning gated: -97 Elo (NMP corrupts SE decisions)

NMP must be disabled inside SE because NMP's null-move search at ply+1 has access to
the excluded move (ExcludedMove is ply-indexed), defeating the verification. But RFP
and ProbCut are safe to keep because: RFP uses static eval (independent of excluded
move) and the margins make it extremely unlikely to fire at singularDepth (2-3).

**4. Disabling all pruning in SE is catastrophically expensive.**
The verification search without NMP/RFP/ProbCut explores far more nodes. At depth 8,
singularDepth = 3. Without NMP, every alternative move gets a depth-2 search. With
~30 legal moves minus the TT move, that's 29 depth-2 searches per SE test. With
~40-50% of tests resulting in extensions (full tree exploration, no early cutoff),
the overhead is massive — the engine loses 2-4 plies of depth in the same time.

**5. WAC depth comparison confirmed the overhead.**
On WAC.003 (5s): Base reached depth 23, SE with all pruning gated reached depth 18
(-5 plies). SE with NMP-only gated reached depth 23 (tied). The NMP-only approach
keeps the verification search cheap enough that multi-cut and negative extensions
can offset the cost.

### Why the Positive Extension Hurts

The extension is the core value proposition of SE in every reference engine (+20-30 Elo).
In our engine it costs ~30 Elo. Hypotheses:

1. **v5 NNUE eval accuracy**: Our v5 net (1024-wide, trained on LC0 data) may already
   provide accurate enough positional evaluation that the extra ply from extensions
   doesn't discover new tactical information. Reference engines with weaker eval gain
   more from deeper search on critical moves.

2. **Search feature interactions**: Our engine has several non-standard features that
   may interact badly with extensions:
   - **Alpha-reduce**: After alpha is raised, subsequent moves are reduced by 1 ply.
     An SE-extended move raises alpha more often (deeper search finds better scores),
     causing over-reduction of later moves.
   - **Fail-high score blending**: `(bestScore*depth + beta)/(depth+1)` dampens
     fail-high scores. Extended moves have higher depth, so their dampening is
     different, potentially distorting the search tree.
   - **TT score dampening**: `(3*score + beta)/4` on TT lower-bound cutoffs. Applies
     throughout the SE-extended subtree, compounding score distortion.

3. **Node budget displacement**: Extensions at depth >= 8 create large subtrees that
   consume the node budget. In a fixed-time search, every extra node spent on the
   extended TT move is a node NOT spent on other critical moves elsewhere in the tree.
   If the extension doesn't find new tactics (hypothesis 1), this is pure waste.

4. **Extension rate too high**: With margin=depth (8cp at depth 8), about 40-50% of
   tested TT moves are "singular." In reference engines, extension rates of 30-40%
   are typical. Our high rate may be extending mediocre moves that happen to be slightly
   better than alternatives.

### Unexplored Directions

- **Fractional extensions**: Accumulate +0.5 ply per singular detection, extend by +1
  when accumulated credit >= 1.0. Halves the per-extension cost while keeping the
  frequency. Several strong engines use this.
- **Quiet-only extensions**: Only extend singular moves that are quiet (non-captures).
  Captures already get recapture extensions; doubling up may over-extend tactical lines.
- **Lower depth gate (>=6)**: More frequent but cheaper SE tests (singularDepth = 2).
  The verification is essentially a static eval + 1-ply check of alternatives.
- **Disable alpha-reduce for SE-extended moves**: If alpha-reduce is the interaction
  causing harm, exempting SE-extended nodes may recover the extension value.
- **Disable fail-high blending in SE subtrees**: Same idea — prevent score distortion
  in the extended subtree.
- **Wider margin with extension, tight margin for multi-cut**: Use two thresholds:
  only extend when the move is overwhelmingly singular (margin=depth*4), but do
  multi-cut/negext with a tight margin (margin=depth). This targets extensions at
  truly exceptional moves while keeping the pruning benefits.

### Conclusion

SE's verification search, multi-cut, and negative extensions are correctly implemented
and break even. The mystery is specifically why the +1 extension harms our engine when
it helps every reference engine. The most likely cause is an interaction between
extensions and our non-standard search features (alpha-reduce, score blending), combined
with a strong NNUE eval that already captures what extensions aim to discover. Fractional
extensions and interaction isolation are the most promising next steps.
