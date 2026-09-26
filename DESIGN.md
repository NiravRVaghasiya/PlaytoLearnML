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

- Metrics/numbers always in mono (they change live — mono prevents layout jump). `globals.css` sets `output`, `code`, `pre`, `kbd` and `samp` to mono globally.
- Scale (rem): 0.75 / 0.875 / 1 / 1.25 / 1.5 / 2 / 3.
  - Labels inside an SVG are sized in viewBox units (`text-[8px]` in a scaled `<svg>`), so they sit outside this scale by nature.
  - *Known drift:* some small HTML labels in the later games use arbitrary `text-[10px]`/`text-[11px]` (Feature Forge, Convolution Kitchen, Confusion Matrix Chef, among others). New work should use `text-xs` (0.75rem) as the floor.
- Line-height: 1.5 body, 1.2 display.

## 4. Spacing & layout

- 4px base grid. Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64.
- Radius: `--r-sm: 6px`, `--r-md: 12px`, `--r-lg: 20px`, `--r-full: 999px`.
- Game canvas gets max real estate. From `lg` (1024px) up, controls dock to a 320px right rail. Below `lg` everything stacks in one column (see §5). There is no bottom sheet.
- Shadow: soft, low-opacity, layered (`--shadow-soft`, `--shadow-raised`). No hard drop shadows.

## 5. Core layout: the GameShell

Every game renders inside `<GameShell>`. Fixed anatomy so players build muscle memory:

Desktop (`lg` and up):

```
┌─────────────────────────────────────────────────────────┐
│  ← Back    Game Title            [Visual ⇄ Code] [Σ Math] │  top bar (sticky)
├──────────────────────────────────────────┬──────────────┤
│                                            │  CONTROLS    │
│              GAME CANVAS                    │  (sliders,   │
│           (visual or code lane)             │   dials,     │
│                                            │   palette)   │
│                                            ├──────────────┤
│                                            │ METRIC       │  ← always visible
│                                            │ READOUT      │
│                                            │ (+ secondary)│
├────────────────────────────────────────────┴──────────────┤
│  NAMED FAILURE strip  (only on a loss)          [Retry]   │
├─────────────────────────────────────────────────────────┤
│  💡 Why did that happen?  (feedback card, updates live)     │
├─────────────────────────────────────────────────────────┤
│  Lv 3 ▓▓▓▓░░ +120 XP  ⭐⭐☆ (i)  badge · "not saved" hint   │  footer
│                                        [Retry] [Next →]   │
└─────────────────────────────────────────────────────────┘
```

Phones and tablets (below `lg`), one column, top to bottom: the top bar (it wraps
and **scrolls away**), then the primary metric (**pinned** to the top of the
viewport), the secondary metrics (scroll), the canvas, the controls, the failure
strip, the WhyCard, and the footer.

- **Metric readout is never hidden.** It's the star of the show — the thing that moves when you act.
  - On desktop it sits in the rail and the header is sticky.
  - Below `lg`, **only the primary metric pins**. It stays at the top while the
    canvas and controls scroll under it, and scrolls off only past the controls,
    at the failure strip, WhyCard and footer. The header isn't sticky there: it
    wraps to two or three rows, and a sticky header that tall covered the
    metric it was meant to protect. The primary card alone is about 125px.
  - Secondary metrics (a 2-column grid below `lg`) scroll with the page.
  - Only on viewports 320px tall or less (400% zoom: 320×256, 480×270) does
    nothing pin. Landscape phones (about 340–412px tall) keep the primary card
    pinned.
  - WCAG 2.4.11 (focus not obscured): pages with a shell set
    `scroll-padding-top` to match what is pinned, so keyboard focus never lands
    under it. That is 9rem below `lg`, 1rem at 320px tall or less, and 5.5rem
    at `lg` (globals.css).
- **Lane toggle** (top bar): `Visual ⇄ Code`. State persists across the toggle. Only the active lane is mounted.
- **Math** button (Σ icon): opens a modal drawer with the equation and the real code driving the sim. While it is open, the header, content and footer are `inert`.
- **Named failure** is an inline strip under the grid (name, numbers, one-click Retry), never a modal. A separate, persistent screen-reader alert region announces it. It is read out when a failure arrives or its name changes (not when only its numbers move), and again after a Retry. It empties 3s after the failure clears.
- **Landmarks:** the canvas is `<main id="game-canvas">` (the skip link's target once a game has mounted). The metric is a "Live metric" region, secondaries are "More metrics", and the rail is a "Controls" region.
- **A crashing lane** is caught by `LaneErrorBoundary`. The shell stays up and the lane shows "The visual lane stopped working", with a retry button and a pointer to the other lane.

## 6. Component primitives (shared, in `src/components/`)

| Component | Role | Notes |
|---|---|---|
| `<MetricReadout>` | Big live number (accuracy/loss/inertia) | Mono font. The value tweens; its colour pulses green or red by `goodDirection`, and a ▲/▼/– glyph carries the direction too. `state` (`neutral` / `good` / `bad` / `warn`) pins the colour and overrides the pulse, sparkle included (only `good` keeps it). Use `neutral` for direction-less counts like k or iterations. A non-finite value renders "—" and is spoken as "not measured yet". |
| `<WhyCard>` | 2-line explanation tied to last action | Fades in on each new `key`. Tone is carried by an icon and a screen-reader prefix ("Failure:", "Good news:") as well as colour. The optional Concept Library link is a 44px-tall standalone link. |
| `<LaneToggle>` | Visual ⇄ Code switch | Segmented control built as a real radiogroup: roving tabindex, and arrow keys move focus and selection together. The code option can be disabled with a reason. |
| `<MathDrawer>` | Reveal-the-math panel | Modal dialog: focus trap, Escape to close, focus returns to the trigger, keydown stops at the dialog. KaTeX loads on first open, not with the game. It renders HTML plus **MathML** (screen readers read the maths), wraps each equation in `gathered` so `\\` line breaks work, and shows the LaTeX source if KaTeX can't render it (or can't be downloaded; the next open retries). The code block has full-contrast line numbers. |
| `<XPBar>` | XP + level | Gold fill, animated on gain. A real `progressbar` ("Level 3, 140 of 300 XP"); the "+N XP" flash is announced politely and clears after 2s. |
| `<StarRating>` | 1–3 mastery stars | Empty stars in `--text-muted` (about 7:1), not `--border`. A new star pops once. A 44px "What earns each star" (i) button opens the criteria list, each marked earned or not yet in words, so touch and keyboard users aren't stuck with a hover tooltip. |
| `<Slider>` | Continuous control | Native `<input type="range">`, 44px tall; `scale="log"` for anything spanning orders of magnitude. The visible value is `aria-hidden`; the input's `aria-valuetext` is the one spoken channel. |
| `<Dial>` | Rotary control | A 64px `role="slider"` knob with 44px **− / +** buttons either side. Each tap moves one `keyStep` and announces the new value, which is the single-pointer alternative to dragging. Also takes arrows, Page Up/Down, Home/End, and vertical drag. |
| `<DatasetChip>` | Selectable option chip | Pressed state carried by a check icon; optional locked state with a reason. No game currently uses the locked state. |
| `<Button>` | Actions: Retry, Next, Step, Train, Math | Variants `primary` / `secondary` / `ghost` / `danger`; 44px min target; never colour-only (always has a label) |
| `<CodeEditor>` | The code lane's editable snippet | Mono, labelled textarea over a decorative highlight layer, so the code stays readable without colour. Tab is NOT trapped: keyboard users must be able to leave the field. The run error is a `role="alert"` box: red on the border, tint and icon, while the message itself is `--text` (red text on its own tint measured 4.1:1). |

### Outside the shell

| Pattern | Where | Notes |
|---|---|---|
| Roster card | `src/app/page.tsx` | Title, concept, a difficulty chip, a primary-tinted **"Start here"** chip on the first game, "Live metric: …", and earned stars, which appear after hydration only. |
| Error / 404 panel | `ErrorPanel` (route and global errors), `not-found.tsx`, `GameMount`'s crash panel | Each one renders a `<main>` with an `h1`: what happened in plain words, then Try again (for a code bug) or Reload (for a chunk that failed to download), then a way home. The game crash panel names the game and moves focus to its heading. Never Next's unbranded fallback. |
| App icon | `src/app/icon.svg` (plus `favicon.ico`, `apple-icon.png` drawn from it) | A `--primary` decision boundary with one `--class-b` and one `--class-a` point either side, on `--bg`. If the mark changes, regenerate the two rasters. |

## 7. Motion

- Durations: micro 120ms, standard 240ms, entrance 360ms.
- Easing: `cubic-bezier(0.22, 1, 0.36, 1)` (springy ease-out) for entrances; linear for continuous sim.
- The metric readout: value counts up/down (never snaps); color pulses on change (900ms), and an improvement adds the `metric-sparkle` glow. A `state` of `neutral`, `bad` or `warn` (a named failure sets `bad`) suppresses the sparkle along with the pulse colour.
- The shared "this one is wrong" cue, `.flash-wrong` (used by Sort-It Arcade's misclassified points), is a red **glow** (`drop-shadow`). It pulses **three times** when the element appears, then holds still. It never dips opacity, so the glyph keeps full contrast. It used to blink forever, and nothing that lasts a whole round may animate indefinitely (WCAG 2.2.2).
- Other named animations: `why-card-in` (fade and rise), `star-pop`, `xp-gain-flash` (1.6s). All run once.
- Reduced-motion: respect `prefers-reduced-motion`. The global rule cuts every animation and transition to near zero and turns `.flash-wrong` off outright; functional feedback (colour, value, text) still lands.

## 8. The "aha" moment design

The single most important UX principle. When a player acts and the metric moves, that's the learning moment. Amplify it:

- Metric change is **immediate** (<1s), **visible** (large, centered-ish), and **explained** (WhyCard).
- On a big improvement: brief celebratory micro-animation (sparkle on the metric, not the whole screen).
- On a failure: the metric goes red (`GameShell` does this automatically when `failure` is set), the named-failure strip and the WhyCard name the failure mode, and the retry is one click away. Never a dead-end "Game Over" screen.

## 9. Accessibility (hard requirements)

- WCAG AA contrast minimum (AAA for body text where possible). Red-on-red-tint text fails AA, so error messages put the red on the border, tint and icon, and the words in `--text`.
  - The same goes for data tints. Convolution Kitchen's kernel-weight cells are capped at 14–46% of `--class-a` / `--class-b` over `--surface-2` (`convolution-kitchen/tint.ts`, guarded by `tint.test.tsx` with the real tokens), so the `--text` numbers on them stay at 4.5:1 or better. Only `--text` may sit on such a tint: `--text-muted` on the strongest orange is 2.4:1. Dark text doesn't rescue a stronger tint, because between about 51% and 65% orange neither `--text` nor `--bg` reaches 4.5:1.
- Never color-only: pair with icon, shape, label, or pattern.
- Full keyboard nav; visible focus rings (`--primary` outline, 2px, `:focus-visible` only). A skip link ("Skip to content") on every page targets `#main-content`; on a game page, once the game has mounted, it focuses the canvas.
- **What gets announced, and how** (one channel each, so nothing is read twice):
  - The primary metric: a debounced polite region, 600ms after the value settles ("Accuracy increased to 84 percent"). Secondary metrics don't announce.
  - A new WhyCard: its tone prefix and **headline only**, never the body, politely, 700ms after the card changes. This comes from a persistent region in `GameShell`, because a region remounted with its content already inside often isn't read. It is skipped when it would repeat the named failure.
  - The named failure: a persistent `role="alert"` region, announced when a failure arrives, when its name changes, or when it recurs after a Retry. A detail whose numbers move under the same name is not re-read.
  - Controls: the control's own `aria-valuetext`. Visible `<output>` readouts are `aria-hidden`, because `<output>` is an implicit live region. The Dial's − / + taps are announced by a small polite region of their own.
  - Equations: KaTeX's MathML.
- **Targets.** Shared primitives, and any standalone control a game adds, are at least **44×44px**: use `<Button>`, or Tailwind's `min-h-11` / `size-11` (44px) on anything hand-rolled. The hard floor is WCAG 2.2 SC 2.5.8's **24×24px**, with its spacing and inline-text exceptions. A few game controls sit between the two (Feature Forge's remove-feature button is 32px), and links inside running text are exempt. `bun run verify:mobile` reports anything under 24px as a warning for a human to judge; it doesn't fail on it.
  - *Known exception:* Sort-It Arcade's wiggle has 25 handles, and on a phone-width field each one's full-height tap column is only about 11–12px wide, below the 24px floor. The handles' slider semantics (arrow keys, a screen reader's adjust gesture) are the precise path there. This is a design follow-up, not a pass.
- **Dragging is never the only pointer path** (WCAG 2.2 SC 2.5.7): the Dial has − / + buttons, and Sort-It Arcade's field moves the nearest handle on a single tap. On touch, a tap off a handle applies on release, and only if the finger moved 10px or less, so a scrolling swipe edits nothing.
- **Focus not obscured** (WCAG 2.4.11): see the `scroll-padding-top` rule in §5.
- Lighthouse a11y ≥ 95 per route in both the desktop (1350×940) and the mobile (360×740 @2×) pass (in Definition of Done). Lighthouse sees first load only. `bun run verify:play` checks the open Math drawer's behaviour on every game (KaTeX rendered, focus trapped, Escape, focus returned); its contrast, code-lane errors and named failures still need a manual check.

## 10. Responsive

- Desktop-first for canvas games. Mobile is graceful by stacking (§5), not by a bottom sheet. Every route must pass `bun run verify:mobile` at 360×740: no sideways scroll, no layout viewport forced wider than the screen, and the metric on screen. A wide figure (Backprop Blitz's graph, for example) scrolls sideways inside its own box instead of widening the page.
- 3D views never depend on drag. **Gradient Descent Skier's** terrain has a fixed camera, deliberately: every control is a form widget in the rail, and a 3D / Contour toggle offers an SVG contour map of the same surface. **Dimension Diver** rotates its cloud with yaw / pitch / roll sliders, and the SVG 2D shadow carries every judgement. Both fall back without WebGL: the Skier to the contour map, the Diver to a notice beside the working 2D view.
- Breakpoints: sm 640 / md 768 / lg 1024 / xl 1280. `lg` is where the right rail and the sticky header begin.

## 11. Iconography & illustration

- Line icons, 2px stroke, rounded caps (Lucide set).
- Each game has one mascot/motif (Skier, Blacksmith, Chef, Detective) — used sparingly on the game card and empty states, not cluttering gameplay. *Not built yet:* roster cards currently carry text only, and there is no illustration anywhere.
- The app icon is the one piece of brand art (§6, "Outside the shell").

---

*Apply consistently. When a new pattern is needed, add it here first, then use it — don't invent one-off styles per game.*
