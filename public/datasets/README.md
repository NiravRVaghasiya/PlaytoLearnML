# public/datasets

Static datasets served to the browser. Per CLAUDE.md all ML runs client-side, so
these files are fetched by the game and processed with TensorFlow.js — there is
no server-side preprocessing step.

Planned contents (added when the game that needs them lands):

| File | Used by | Notes |
|---|---|---|
| `iris.json` | Sort-It Arcade, Data Detox | 150 rows, 4 features, 3 classes |
| `titanic.csv` | Data Detox, Feature Forge | Unlockable dataset (spec §4) |
| `mnist-subset.json` | Convolution Kitchen | Small subset only — keep the payload light |

Rules:
- Keep every file small enough to fetch on a slow connection (target < 500 KB).
- Datasets must be redistributable. Record the source and licence here.
- Synthetic/procedural data is generated in code (see `seededRandom` in
  `src/lib/utils.ts`) rather than committed as a file.
