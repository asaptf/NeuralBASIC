---
name: Bug report
about: Something in the app behaves wrongly
title: ""
labels: ["bug"]
---

## What happened

## What you expected

## Steps to reproduce

1.
2.
3.

## The DSL you were running

```text

```

## Environment

- Browser and version:
- Running locally (`npm run dev`) or the hosted build:

## Notes

If the DSL editor is blank or unstyled, check your network — Monaco is fetched from a CDN on first
load, so a blocked CDN breaks the editor while the rest of the app keeps working. That is a known
limitation rather than a new bug, but do report it if you think something else is going on.

If a *lesson figure* is what disagrees with the app, please use the "A lesson figure doesn't
reproduce" template instead — it asks for the things needed to judge that specific class of problem.
