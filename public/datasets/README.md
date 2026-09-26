# public/datasets

This folder is reserved for static datasets served to the browser. **It holds no
data today, and no game fetches from it.**

All fourteen games build their data in code: procedurally, and from a fixed seed
wherever randomness is involved (see `seededRandom` in `src/lib/utils.ts`). That
keeps rounds reproducible, and there is nothing to download. All ML runs
client-side (CLAUDE.md), so a dataset added here would be fetched by the game
and processed in the browser. There is no server-side preprocessing step.

Real datasets are a spec §4 unlockable ("real datasets (Titanic, MNIST, Iris)")
that has not been built. If one is added, these rules apply:

- Keep every file small enough to fetch on a slow connection (target < 500 KB).
- Datasets must be redistributable. Record the source and licence here.
- Synthetic/procedural data stays in code rather than being committed as a file.
- If a game fetches one, its playthrough should cover the fetch, and the file is
  same-origin, so the CSP needs no change.

This file is publicly served at `/datasets/README.md`, like everything under
`public/`.
