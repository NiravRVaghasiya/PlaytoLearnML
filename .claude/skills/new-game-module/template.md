# Game module skeleton

Copy this structure into `src/games/<slug>/`. Replace `<Slug>` / `<slug>`.

## store.ts
```ts
import { create } from 'zustand';

// Mirror the spec's Data Model exactly.
interface <Slug>State {
  // ...spec fields
  metric: number;              // the live metric (accuracy/loss/inertia)
  failureMode: string | null;  // named ML failure when lost
  // actions:
  step: () => void;
  reset: () => void;
}

export const use<Slug>Store = create<<Slug>State>((set, get) => ({
  metric: 0,
  failureMode: null,
  step: () => {/* apply player action → recompute via ml.ts → set metric */},
  reset: () => set({ metric: 0, failureMode: null }),
}));
```

## ml.ts
```ts
// Real, client-side ML. Pure + testable. No server calls.
export function computeMetric(state: /*...*/): number {
  // e.g. accuracy, loss, inertia — computed for real
}
```

## VisualLane.tsx / CodeLane.tsx
```tsx
// Both read/write the SAME store. Visual = drag/click/slider.
// Code = editable snippet via useCodeLane, driving identical state.
```

## index.tsx
```tsx
import { GameShell } from '@/engine/GameShell';
export default function <Slug>() {
  return (
    <GameShell
      title="<Game Name>"
      metric={/* from store */}
      whyCards={/* from why-cards.ts */}
      mathDrawer={{ equation: '...', code: '...' }}
      visual={<VisualLane />}
      code={<CodeLane />}
    />
  );
}
```

## <slug>.test.ts
```ts
import { describe, it, expect } from 'vitest';
import { computeMetric } from './ml';
describe('<slug> ml', () => {
  it('computes the metric correctly', () => { /* ... */ });
  it('flags the named failure mode', () => { /* ... */ });
});
```
