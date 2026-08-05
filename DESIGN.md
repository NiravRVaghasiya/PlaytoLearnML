# DESIGN.md — GameML Design System

> Visual + interaction language for GameML. Agents: apply these tokens and rules to every screen and game. Consistency is what makes 14 games feel like one product.

## 1. Brand personality

**Playful lab.** Part arcade, part science bench. Bright, tactile, encouraging — never intimidating. The vibe: "a game that happens to be a real ML lab," not "a textbook with buttons."

- Tone: encouraging, curious, plain-language. Celebrate failure as learning.
- Motion: snappy and physical (things have weight, spring, momentum).
- Never: corporate-sterile, jargon-heavy, or punishing.

## 2. Color tokens

Colorblind-safe base (Okabe–Ito derived). Define as CSS variables + Tailwind theme extend.

```css
:root {
  /* Surfaces */
  --bg:            #0E1116;   /* app background (dark-first) */
  --surface:       #171B22;   /* cards, panels */
  --surface-2:     #212734;   /* raised elements */
  --border:        #2C3444;

  /* Text */
  --text:          #E6EAF2;
  --text-muted:    #9AA4B2;

  /* Brand */
  --primary:       #4FC3F7;   /* electric blue — primary actions */
  --primary-ink:   #06263A;

  /* Semantic (ML meaning — use consistently across ALL games) */
  --correct:       #2ECC71;   /* right / accuracy up / converged */
  --wrong:         #E74C3C;   /* misclassified / divergence / loss spike */
  --warn:          #F1C40F;   /* overfit warning / caution */
  --class-a:       #0072B2;   /* dataset class A (Okabe-Ito blue) */
  --class-b:       #E69F00;   /* dataset class B (Okabe-Ito orange) */
  --class-c:       #009E73;   /* dataset class C (green) */
  --class-d:       #CC79A7;   /* dataset class D (pink) */

  /* Progression */
  --xp:            #FFD54F;   /* XP / gold */
  --star:          #FFC107;
}
```

**Semantic color rule:** class A/B/C/D colors are FIXED across every game. Blue is always class A, orange always class B. A learner should never have to relearn what a color means when switching games. Never encode meaning in color alone — always pair with shape/label/icon (colorblind safety).

## 3. Typography

```css
--font-display: "Space Grotesk", system-ui, sans-serif;  /* headings, game titles */
--font-body:    "Inter", system-ui, sans-serif;          /* UI, body */
--font-mono:    "JetBrains Mono", ui-monospace, monospace;/* code lane, metrics, equations */
```

- Metrics/numbers always in mono (they change live — mono prevents layout jump).
- Scale (rem): 0.75 / 0.875 / 1 / 1.25 / 1.5 / 2 / 3.
- Line-height: 1.5 body, 1.2 display.

## 4. Spacing & layout

- 4px base grid. Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64.
- Radius: `--r-sm: 6px`, `--r-md: 12px`, `--r-lg: 20px`, `--r-full: 999px`.
- Game canvas gets max real estate; controls dock to a right rail (desktop) or bottom sheet (mobile).
- Shadow: soft, low-opacity, layered. No hard drop shadows.

## 5. Core layout: the GameShell

Every game renders inside `<GameShell>`. Fixed anatomy so players build muscle memory:

```
┌─────────────────────────────────────────────────────────┐
│  ← Back    Game Title            [Visual ⇄ Code] [ƒ Math] │  top bar
├──────────────────────────────────────────┬──────────────┤
│                                            │  CONTROLS    │
│              GAME CANVAS                    │  (sliders,   │
│           (visual or code lane)             │   dials,     │
│                                            │   palette)   │
│                                            ├──────────────┤
│                                            │ METRIC       │  ← always visible
│                                            │ READOUT      │
├────────────────────────────────────────────┴──────────────┤
│  💡 Why did that happen?  (feedback card, updates live)     │
├─────────────────────────────────────────────────────────┤
│  XP ▓▓▓▓░░  ⭐⭐☆   [Retry] [Next →]                        │  progress bar
└─────────────────────────────────────────────────────────┘
```

- **Metric readout is never hidden.** It's the star of the show — the thing that moves when you act.
- **Lane toggle** (top-right): `Visual ⇄ Code`. State persists across the toggle.
- **ƒ Math** button: expands a drawer with the equation + the real code driving the sim.

## 6. Component primitives (shared, in `src/components/`)

| Component | Role | Notes |
|---|---|---|
| `<MetricReadout>` | Big live number (accuracy/loss/inertia) | Mono font; animates value + color (green up / red down as semantically appropriate) |
| `<WhyCard>` | 2-line explanation tied to last action | Fades in; plain language; links to Concept Library |
| `<LaneToggle>` | Visual ⇄ Code switch | Segmented control |
| `<MathDrawer>` | Reveal-the-math panel | KaTeX equation + syntax-highlighted code |
| `<XPBar>` | XP + level | Gold fill, spring animation on gain |
| `<StarRating>` | 1–3 mastery stars | Empty → fill with pop |
| `<Slider>` / `<Dial>` | Continuous controls | Show live value; large hit target (44px min) |
| `<DatasetChip>` | Selectable dataset | Locked state for premium datasets |

## 7. Motion

- Durations: micro 120ms, standard 240ms, entrance 360ms.
- Easing: `cubic-bezier(0.22, 1, 0.36, 1)` (springy ease-out) for entrances; linear for continuous sim.
- The metric readout: value counts up/down (never snaps); color pulses on significant change.
- Reduced-motion: respect `prefers-reduced-motion` — disable decorative motion, keep functional feedback.

## 8. The "aha" moment design

The single most important UX principle. When a player acts and the metric moves, that's the learning moment. Amplify it:

- Metric change is **immediate** (<1s), **visible** (large, centered-ish), and **explained** (WhyCard).
- On a big improvement: brief celebratory micro-animation (sparkle on the metric, not the whole screen).
- On a failure: the metric goes red, the WhyCard names the failure mode, and the retry is one click away. Never a dead-end "Game Over" screen.

## 9. Accessibility (hard requirements)

- WCAG AA contrast minimum (AAA for body text where possible).
- Never color-only: pair with icon, shape, label, or pattern.
- Full keyboard nav; visible focus rings (`--primary` outline, 2px).
- ARIA live region announces metric changes for screen readers ("Accuracy increased to 84 percent").
- Min touch target 44×44px.
- Lighthouse a11y ≥ 95 per game (in Definition of Done).

## 10. Responsive

- Desktop-first for canvas games; graceful mobile via bottom-sheet controls.
- 3D games (Skier, Dimension Diver) require pointer/touch drag — provide button-based fallback controls for accessibility.
- Breakpoints: sm 640 / md 768 / lg 1024 / xl 1280.

## 11. Iconography & illustration

- Line icons, 2px stroke, rounded caps (Lucide set).
- Each game has one mascot/motif (Skier, Blacksmith, Chef, Detective) — used sparingly on the game card and empty states, not cluttering gameplay.

---

*Apply consistently. When a new pattern is needed, add it here first, then use it — don't invent one-off styles per game.*
