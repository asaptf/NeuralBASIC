## What this changes

## Why

## Checks

- [ ] `npm test`
- [ ] `npm run typecheck`
- [ ] `npm run lint`
- [ ] `npm run build`

## If this touches training, the visualiser, or the curriculum

Numbers matter more than descriptions here. Please fill this in rather than deleting it.

- What did you measure, and **over how many runs**?
- Did any lesson figure move? Which, and by how much?
- If you changed a dataset generator, did you update its fingerprint in
  `src/engine/datasets.test.ts` deliberately — and re-run the lesson suite several times?

A single run tells you almost nothing about a stochastic process. If you are quoting one, say so.

## If this changes a lesson claim

- [ ] The new number is measured over many runs, not observed once
- [ ] The claim is a range with margin, or a mean in prose plus an aggregate assertion
- [ ] I understand *why* the figure moved, and it isn't a sign the pedagogy changed
