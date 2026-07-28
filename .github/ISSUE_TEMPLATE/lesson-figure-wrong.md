---
name: A lesson figure doesn't reproduce
about: The app shows something different from what a lesson claims
title: "Lesson figure: "
labels: ["pedagogy", "figures"]
---

## Where

Chapter and section heading, or the example's label:

## What the lesson claims

Quote it:

## What you actually saw

Numbers, and the exact DSL you ran:

```text

```

## How many times did you run it?

Training here is stochastic, so a single run can land well outside the typical range. Please say how
many times you ran it and roughly what spread you saw — one surprising run is useful information,
but ten runs that all disagree with the lesson is a much stronger report.

Runs:

## Anything else

Browser, whether you had a held-out split (`val=`) on the train line, and anything else you changed.

---

*Why this template exists: every numeric claim in the lessons is checked against measured
distributions by `src/curriculum/lesson.test.ts`. So a figure that doesn't reproduce for you is a
real defect even when the whole suite is green — it means the test's margins are hiding something, or
the claim is true on average but misleading in practice. Both are worth fixing.*
