# Contributing to NeuralBASIC

Thanks for looking. This is a teaching tool, so the bar for a change is not just "does it work"
but "does it teach the truth". That makes contributing here a little unusual, and this document is
mostly about the unusual parts.

## Getting set up

```bash
npm install
npm run dev
```

Then open <http://localhost:3000>.

```bash
npm test          # engine, tutor graders, persistence, curriculum, lesson figures
npm run typecheck
npm run lint
npm run build     # static export into out/
```

CI runs all four on every pull request. There is no separate formatting step — ESLint covers it.

## The one rule that will surprise you: every number in the lessons is tested

The chapters quote measured figures — "two hidden units solve XOR about 15% of the time", "50 conv
parameters beat 2,642 dense ones". Prose about a live system goes stale silently, so
`src/curriculum/lesson.test.ts` parses the numeric claims out of each lesson example's `expect`
text and checks them against distributions measured over repeated training runs.

**Editing "82%" to "95%" in a lesson fails the suite.** That is deliberate. If you change the
engine, the optimiser, a dataset, or an initialisation, figures in the prose may move — and the
test will tell you which ones.

If you are adding or changing a claim:

1. Measure it over **many runs**, not one. Training is stochastic here; a single run tells you
   almost nothing.
2. Write **ranges with margin** (roughly 20 percentage points of headroom), or put a precise mean
   in the prose and assert an aggregate. Quoting the min or max you happened to observe over N runs
   as if it were a bound is how this suite gets flaky — it has happened repeatedly.
3. Assert on **aggregates across runs**, not per-run outcomes.

Two real errors this caught while the chapters were being written, both of which read as completely
plausible:

- A draft claimed a high learning rate makes the loss curve "turn violent" on `or`. Measured, `or`
  at lr=20 and even lr=50 is perfectly smooth and converges to loss ≈ 0.
- A draft quoted ReLU and tanh as tied at 88% on XOR, from an eight-run sample. At sixty runs the
  real figures are sigmoid 97%, tanh 93%, ReLU 78% — every number wrong, and the ordering wrong too.

Small samples of a random process are how confident false beliefs get made. Chapter 2 now says so.

## Pedagogical rules that changes must respect

These are load-bearing, not style preferences.

1. **The Socratic tutor never pastes a complete working solution first.** `enforceSocratic` catches
   replies that look like one, and `src/tutor/socratic.test.ts` holds the line.
2. **The loop is predict → experiment → observe → explain.** A change that lets a learner skip
   straight to the answer removes the point.
3. **A silent default is worse than a crash.** The DSL parser used to accept anything and quietly
   fall back to `dense 2 -> 1` on `xor`, so a learner who typo'd `activaton=relu` got a plausible
   result from a network they never wrote. It now throws `DSLParseError` with a 1-based line number
   and names the valid alternatives. Do not reintroduce forgiving fallbacks — under
   predict → experiment → observe, a wrong-but-plausible result corrupts the observation.
4. **A challenge must not be answerable only by saying something false.** One early challenge marked
   "oscillates or diverges" as the correct answer and required those words to pass, so the app
   rejected the accurate answer. If a gate and reality disagree, reality wins.
5. **Explanations are graded across distinct concepts**, so keyword stuffing cannot unlock a
   chapter. `curriculum.completeness.test.ts` proves that for all five chapters.

## If your change breaks a lesson figure

Do not edit the prose to match the new number until you understand *why* it moved. A figure shifting
is often the first sign that a change altered the pedagogy, not just the implementation. Three of
the five chapters were at some point blocked by the app being unable to demonstrate their own
subject, and each time the fix was engine or data work rather than rewording:

- Chapter 3 needed a train/test split that did not exist, and then `noisy_moons`, because the clean
  2-D sets are dense and smooth enough that memorising them also generalises — they refuse to
  overfit.
- Chapter 4's convolution *lost* to dense until pooling existed, because without a
  translation-invariant readout there is nothing for weight sharing to exploit.
- Chapter 5's attention map was literally a 1×1 cell containing 1.0, because a single-token input
  makes softmax over one position return 1. It needed `negation`, a genuinely multi-token set.

## Datasets are frozen on purpose

Generators use a deterministic hash (`detUnit`) rather than `Math.random()`, so two processes see
identical data. `src/engine/datasets.test.ts` pins each set's sample count, class balance, a
checksum and its first three samples. If you change a generator, that test will fail — which is the
point. Update the fingerprint deliberately, and re-run the lesson suite several times to see which
figures moved.

## Where things live

```
src/
  engine/           # DSL parser, datasets, layers, training, export
  curriculum/       # 5 chapters, lessons, challenges, gate evaluation
  tutor/            # Socratic policy, mock replies, offline graders
  store/            # Zustand immediate-mode store + animated training loop
  components/       # IDE shell, editor, lab, network, metrics, tutor, curriculum
  lib/              # persistence, utils
  app/              # Next.js App Router entry
```

Gate evaluation lives in `src/curriculum/gates.ts` and is shared by the store and the tests, so
progression cannot drift from what the tests assert. Keep it that way.

## Pull requests

- Keep the suite green: `npm test && npm run typecheck && npm run lint && npm run build`.
- Say what you observed, not just what you changed. For anything touching training, the visualiser,
  or the curriculum, include the numbers you measured and over how many runs.
- One concern per PR. A pedagogy change and a refactor in the same diff are hard to review honestly.
- New engine behaviour needs a test that would fail without it.

## Reporting a lesson figure that doesn't reproduce

This is a real and useful bug class, and it has its own issue template. If a lesson says one thing
and the app shows another, that is a genuine defect even if every test passes — please include the
exact DSL, what the lesson claims, what you saw, and roughly how many times you ran it.
