/**
 * Thesis defense deck, adapted from Presentation/Chess_MoE_v001.pdf.
 * Palette and type follow the academic equity-research template:
 * cream paper, navy ink, brick accent, Georgia + Calibri.
 * KITSUNE numbers are Table 5.16 of the thesis (weaker W64/H128 base).
 */
const pptxgen = require("pptxgenjs");

const C = {
  bg: "F4EFE6",
  ink: "0E1B2C",
  inkSoft: "23324A",
  muted: "6B6F76",
  rule: "C9C2B6",
  accent: "C0392B",
  accentSoft: "F3E4E1",
  good: "E5EFE8",
  goodInk: "1E4D34",
  panel: "EAE3D5",
  white: "FCFAF6",
  navy: "0E1B2C",
};

const F = { serif: "Georgia", sans: "Calibri" };
const W = 13.333;
const H = 7.5;

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: W, height: H });
pres.layout = "WIDE";
pres.title = "Unsupervised State Bucketing for Mixture-of-Experts in Resource-Constrained Chess Engines";
pres.author = "Omar Cusma Fait";
pres.subject = "MSc thesis defense, University of Trieste";

function shadow() {
  return { type: "outer", color: "1A140C", blur: 10, offset: 2, angle: 135, opacity: 0.1 };
}

function content(slide) {
  slide.background = { color: C.bg };
}

function kicker(slide, text) {
  slide.addText(text.toUpperCase(), {
    x: 0.55, y: 0.28, w: 10.2, h: 0.28,
    fontFace: F.sans, fontSize: 12, color: C.accent,
    charSpacing: 1.6, margin: 0,
  });
}

function title(slide, text, opts = {}) {
  slide.addText(text, {
    x: 0.55, y: opts.y || 0.56, w: opts.w || 12.2, h: opts.h || 0.72,
    fontFace: F.serif, fontSize: opts.size || 30, color: C.ink,
    margin: 0, bold: false,
  });
}

function footer(slide, n, note) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.55, y: 6.98, w: 12.23, h: 0.01,
    fill: { color: C.rule }, line: { color: C.rule, width: 0 },
  });
  slide.addText(note, {
    x: 0.55, y: 7.05, w: 10.4, h: 0.32,
    fontFace: F.sans, fontSize: 11, color: C.muted, margin: 0,
  });
  slide.addText(String(n) + "  /  15", {
    x: 11.15, y: 7.05, w: 1.63, h: 0.32,
    fontFace: F.sans, fontSize: 11, color: C.muted, align: "right", margin: 0,
  });
}

function checker(slide, x, y, size, a, b) {
  const s = size / 2;
  const grid = [[a, b], [b, a]];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      slide.addShape(pres.shapes.RECTANGLE, {
        x: x + c * s, y: y + r * s, w: s, h: s,
        fill: { color: grid[r][c] },
        line: { color: grid[r][c], width: 0 },
      });
    }
  }
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: fill || C.white },
    line: { color: "E3D9C8", width: 1 },
    rectRadius: 0.08,
    shadow: shadow(),
  });
}

function cell(text, opts = {}) {
  return {
    text: String(text),
    options: {
      fill: { color: opts.fill || C.white },
      color: opts.color || C.ink,
      bold: !!opts.bold,
      align: opts.align || "center",
      valign: "middle",
      fontFace: F.sans,
      fontSize: opts.size || 13,
      margin: 5,
    },
  };
}

// ── 1. Title ────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  checker(s, 0.55, 0.48, 0.42, "F4EFE6", "C0392B");
  s.addText("MASTER'S THESIS DEFENSE   ·   UNIVERSITY OF TRIESTE", {
    x: 1.15, y: 0.52, w: 10.5, h: 0.34,
    fontFace: F.sans, fontSize: 13, color: "E7D3A1",
    charSpacing: 1.4, margin: 0, valign: "middle",
  });
  s.addText("Unsupervised State Bucketing\nfor Mixture-of-Experts in\nResource-Constrained Chess Engines", {
    x: 0.55, y: 1.7, w: 12.0, h: 2.55,
    fontFace: F.serif, fontSize: 36, color: "F4EFE6", margin: 0,
  });
  s.addText("Can a tiny chess engine choose the right expert?", {
    x: 0.55, y: 4.45, w: 11.5, h: 0.45,
    fontFace: F.serif, fontSize: 20, italic: true, color: "E7D3A1", margin: 0,
  });
  s.addText("Omar Cusma Fait", {
    x: 0.55, y: 5.55, w: 8, h: 0.38,
    fontFace: F.sans, fontSize: 18, color: "F4EFE6", margin: 0,
  });
  s.addText("MSc Data Science and Artificial Intelligence\nSupervisor: Alessio Ansuini    ·    Co-supervisor: Marco Zennaro, ICTP", {
    x: 0.55, y: 6.0, w: 11.5, h: 0.7,
    fontFace: F.sans, fontSize: 15, color: "C9C2B6", margin: 0,
  });
}

// ── 2. Constraint ───────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "The deployment constraint");
  title(s, "Specialization must preserve search speed");
  const specs = [
    ["120 MHz", "CPU frequency"],
    ["192 KB", "Internal RAM"],
    ["512 KB", "Internal flash"],
  ];
  specs.forEach((pair, i) => {
    const x = 0.55 + i * 4.15;
    card(s, x, 1.6, 3.95, 1.7);
    s.addText(pair[0], {
      x, y: 1.78, w: 3.95, h: 0.75,
      fontFace: F.serif, fontSize: 32, color: C.ink, align: "center", margin: 0,
    });
    s.addText(pair[1], {
      x: x + 0.2, y: 2.58, w: 3.55, h: 0.4,
      fontFace: F.sans, fontSize: 16, color: C.muted, align: "center", margin: 0,
    });
  });
  card(s, 0.55, 3.55, 7.55, 3.05, C.white);
  s.addText("Alpha–beta search calls the evaluator over and over. A slower network can cost a ply of search, which often matters more than a small gain in the value itself.", {
    x: 0.8, y: 3.8, w: 7.05, h: 1.5,
    fontFace: F.sans, fontSize: 18, color: C.inkSoft, margin: 0,
  });
  s.addText("Target: Wio Terminal, ARM Cortex-M4.", {
    x: 0.8, y: 5.45, w: 7.05, h: 0.7,
    fontFace: F.sans, fontSize: 15, color: C.muted, margin: 0,
  });
  card(s, 8.3, 3.55, 4.48, 3.05, C.navy);
  s.addText("The question", {
    x: 8.55, y: 3.78, w: 4.05, h: 0.32,
    fontFace: F.sans, fontSize: 13, color: "E7D3A1", charSpacing: 1.2, margin: 0,
  });
  s.addText("Can specialization improve the prediction inside this budget?", {
    x: 8.55, y: 4.25, w: 4.0, h: 1.8,
    fontFace: F.serif, fontSize: 22, color: "F4EFE6", margin: 0,
  });
  footer(s, 2, "Hardware specifications, not measured deployment.  Thesis §4.8.");
}

// ── 3. NNUE ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Efficiently updatable neural network");
  title(s, "NNUE updates only the changed features");
  const steps = [
    ["1", "A move touches few inputs", "Of the 844 board features, only the squares that change are active."],
    ["2", "Edit the accumulator", "Subtract the removed features and add the new ones. The rest stays."],
    ["3", "Reuse it on the next node", "The L1 accumulator persists across moves, so evaluation stays cheap."],
  ];
  steps.forEach((st, i) => {
    const x = 0.55 + i * 4.15;
    card(s, x, 1.7, 3.95, 3.55);
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.28, y: 1.95, w: 0.52, h: 0.52,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
    });
    s.addText(st[0], {
      x: x + 0.28, y: 1.95, w: 0.52, h: 0.52,
      fontFace: F.sans, fontSize: 16, color: "F4EFE6", align: "center", valign: "middle", margin: 0,
    });
    s.addText(st[1], {
      x: x + 0.28, y: 2.7, w: 3.4, h: 0.9,
      fontFace: F.serif, fontSize: 20, color: C.ink, margin: 0,
    });
    s.addText(st[2], {
      x: x + 0.28, y: 3.7, w: 3.4, h: 1.2,
      fontFace: F.sans, fontSize: 15, color: C.inkSoft, margin: 0,
    });
  });
  footer(s, 3, "L1 is the shared first-layer representation throughout.  Thesis Fig. 1.1.");
}

// ── 4. Hard MoE ─────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Hard mixture of experts");
  title(s, "One accumulator, one selected expert");
  const boxes = [
    ["Board", "844 features"],
    ["Shared L1", "Frozen"],
    ["Dispatcher", "Picks one head"],
    ["Expert head", "WDL output"],
  ];
  boxes.forEach((b, i) => {
    const x = 0.55 + i * 3.2;
    card(s, x, 1.85, 2.55, 1.55, i === 3 ? C.navy : C.white);
    s.addText(b[0], {
      x, y: 2.05, w: 2.55, h: 0.55,
      fontFace: F.serif, fontSize: 18, color: i === 3 ? "F4EFE6" : C.ink, align: "center", margin: 0,
    });
    s.addText(b[1], {
      x: x + 0.1, y: 2.65, w: 2.35, h: 0.4,
      fontFace: F.sans, fontSize: 14, color: i === 3 ? "E7D3A1" : C.muted, align: "center", margin: 0,
    });
    if (i < 3) {
      s.addShape(pres.shapes.RIGHT_ARROW, {
        x: x + 2.58, y: 2.42, w: 0.52, h: 0.28,
        fill: { color: C.accent }, line: { color: C.accent, width: 0 },
      });
    }
  });
  const experts = ["Expert 1", "Expert 2", "Expert K"];
  experts.forEach((name, i) => {
    card(s, 0.55 + i * 2.7, 3.8, 2.5, 0.85, i === 1 ? C.accentSoft : C.panel);
    s.addText(name, {
      x: 0.55 + i * 2.7, y: 3.8, w: 2.5, h: 0.85,
      fontFace: F.sans, fontSize: 16, color: C.ink, align: "center", valign: "middle", margin: 0,
    });
  });
  s.addText("Fine-tuning stores several small heads. At inference, L1 activations select one of them. The others stay in memory and do not run.", {
    x: 0.55, y: 4.9, w: 12.2, h: 0.85,
    fontFace: F.sans, fontSize: 16, color: C.inkSoft, margin: 0,
  });
  footer(s, 4, "Hard routing, thesis §§3.3–3.4. Extra experts still cost memory.");
}

// ── 5. Signals ──────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Alternative specialization signals");
  title(s, "Bucketing defines each expert’s task");
  const header = ["Bucket signal", "Groups positions by", "Available online?"].map((t) =>
    cell(t, { fill: C.navy, color: "F4EFE6", bold: true, align: "left", size: 13 })
  );
  const rows = [
    ["Piece count", "A hand-designed game-phase proxy", "Yes — a fixed rule"],
    ["L1 activations", "Similar hidden representations", "Yes — already computed"],
    ["Sample gradients", "Similar updates of the head", "No — needs the teacher WDL"],
    ["Jointly learned gate", "An end-to-end routing objective", "Yes — after training"],
  ];
  const body = rows.map((r, i) =>
    r.map((t, j) => cell(t, {
      fill: i === 2 ? C.accentSoft : (i % 2 ? C.panel : C.white),
      align: "left",
      bold: j === 0,
    }))
  );
  s.addTable([header, ...body], {
    x: 0.55, y: 1.55, w: 12.23, h: 3.55,
    colW: [3.15, 5.15, 3.93],
    border: [
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
    ],
    fontFace: F.sans,
    valign: "middle",
  });
  s.addText("L1 describes the position. Gradients describe how the head should change.", {
    x: 0.55, y: 5.35, w: 12.2, h: 0.45,
    fontFace: F.serif, fontSize: 18, italic: true, color: C.ink, margin: 0,
  });
  footer(s, 5, "Thesis §3.3.  Per-sample head gradient uses the teacher target, which is unknown at play time.");
}

// ── 6. Protocol ─────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Teacher distillation and held-out comparison");
  title(s, "Training and evaluation protocol");
  const stats = [
    ["150M", "Training positions, mainly from Lichess."],
    ["1.5M", "Independent test positions."],
    ["50,000", "Held-out subset for the primary MoE comparisons."],
    ["174,723", "Parameters in the reference NNUE."],
  ];
  stats.forEach((st, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.55 + col * 6.35;
    const y = 1.6 + row * 2.15;
    card(s, x, y, 6.1, 1.95);
    s.addText(st[0], {
      x: x + 0.3, y: y + 0.25, w: 5.5, h: 0.7,
      fontFace: F.serif, fontSize: 32, color: C.ink, margin: 0,
    });
    s.addText(st[1], {
      x: x + 0.3, y: y + 1.05, w: 5.5, h: 0.55,
      fontFace: F.sans, fontSize: 16, color: C.inkSoft, margin: 0,
    });
  });
  footer(s, 6, "Lc0 supplies win / draw / loss targets. Lower cross-entropy is closer to the teacher. CE is not Elo.  §§4.1–4.2, 5.1, 5.5.");
}

// ── 7. Clusters ─────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Offline structure");
  title(s, "Gradients form more separated clusters");
  s.addText("Silhouette at K = 8, each score measured in its own representation.", {
    x: 0.55, y: 1.4, w: 12, h: 0.35,
    fontFace: F.sans, fontSize: 15, color: C.muted, margin: 0,
  });
  card(s, 0.55, 1.95, 5.7, 3.35, C.navy);
  s.addText("0.191", {
    x: 0.8, y: 2.25, w: 5.2, h: 1.15,
    fontFace: F.serif, fontSize: 60, color: "F4EFE6", margin: 0,
  });
  s.addText("Sample gradients", {
    x: 0.8, y: 3.55, w: 5.2, h: 0.45,
    fontFace: F.sans, fontSize: 20, color: "E7D3A1", margin: 0,
  });
  s.addText("The most separated of the three signals.", {
    x: 0.8, y: 4.15, w: 5.1, h: 0.7,
    fontFace: F.sans, fontSize: 16, color: "C9C2B6", margin: 0,
  });
  card(s, 6.5, 1.95, 6.28, 3.35);
  s.addText("0.003", {
    x: 6.75, y: 2.25, w: 5.8, h: 1.15,
    fontFace: F.serif, fontSize: 60, color: C.ink, margin: 0,
  });
  s.addText("L1 activations", {
    x: 6.75, y: 3.55, w: 5.8, h: 0.45,
    fontFace: F.sans, fontSize: 20, color: C.accent, margin: 0,
  });
  s.addText("Almost no silhouette. Easy to draw, little separation.", {
    x: 6.75, y: 4.15, w: 5.7, h: 0.7,
    fontFace: F.sans, fontSize: 16, color: C.inkSoft, margin: 0,
  });
  footer(s, 7, "Separation is not the same as a better expert or a usable router.  Table 5.2; gradient UMAP in Fig. 5.3.");
}

// ── 8. Recovery ─────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Same L1 input, same MLP dispatcher, eight buckets");
  title(s, "Gradient buckets are hard to recover");
  const bars = [
    ["L1 buckets", 0.98, "98.0%", C.navy],
    ["Gradient buckets", 0.525, "52.5%", C.accent],
  ];
  bars.forEach((b, i) => {
    const y = 1.85 + i * 1.55;
    s.addText(b[0], {
      x: 0.55, y: y, w: 3.1, h: 0.4,
      fontFace: F.sans, fontSize: 16, color: C.ink, margin: 0,
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 3.7, y: y + 0.05, w: 7.3, h: 0.42,
      fill: { color: C.panel }, line: { color: C.panel, width: 0 }, rectRadius: 0.06,
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 3.7, y: y + 0.05, w: Math.max(0.2, 7.3 * b[1]), h: 0.42,
      fill: { color: b[3] }, line: { color: b[3], width: 0 }, rectRadius: 0.06,
    });
    s.addText(b[2], {
      x: 11.15, y: y, w: 1.6, h: 0.5,
      fontFace: F.serif, fontSize: 20, color: C.ink, margin: 0,
    });
  });
  card(s, 0.55, 5.0, 12.23, 1.7, C.white);
  s.addText("45.5 percentage-point gap", {
    x: 0.8, y: 5.18, w: 6, h: 0.4,
    fontFace: F.serif, fontSize: 20, color: C.accent, margin: 0,
  });
  s.addText("A training signal that depends on the true value is difficult to infer from L1 alone.", {
    x: 0.8, y: 5.65, w: 11.6, h: 0.7,
    fontFace: F.sans, fontSize: 16, color: C.inkSoft, margin: 0,
  });
  footer(s, 8, "Tables 5.6–5.7. MLP width 64, K = 8, 1M-position cache, 90/10 split. Clustering was fit before the split.");
}

// ── 9. Strong base ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Primary experiments, test cross-entropy");
  title(s, "Strong-base MoE gains are marginal");
  const header = ["Model", "K", "CE"].map((t) =>
    cell(t, { fill: C.navy, color: "F4EFE6", bold: true, align: t === "Model" ? "left" : "center" })
  );
  const data = [
    ["Reference NNUE", "—", "0.6303", false],
    ["Piece-count MoE", "8", "0.6303", false],
    ["L1-clustered hard MoE", "16", "0.6300", true],
    ["Gradient-clustered hard MoE", "8", "0.6300", true],
    ["Soft-gated MoE, top-2", "2", "0.6304", false],
  ];
  const body = data.map((r) => [
    cell(r[0], { align: "left", bold: true, fill: r[3] ? C.good : C.white }),
    cell(r[1], { fill: r[3] ? C.good : C.white }),
    cell(r[2], { bold: true, fill: r[3] ? C.good : C.white }),
  ]);
  s.addTable([header, ...body], {
    x: 0.55, y: 1.55, w: 8.15, h: 4.15,
    colW: [5.15, 1.2, 1.8],
    border: [
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
    ],
    valign: "middle",
  });
  card(s, 8.95, 1.55, 3.83, 4.15, C.navy);
  s.addText("−0.0003", {
    x: 9.15, y: 1.9, w: 3.45, h: 0.8,
    fontFace: F.serif, fontSize: 32, color: "F4EFE6", align: "center", margin: 0,
  });
  s.addText("Best observed change in CE.", {
    x: 9.2, y: 2.8, w: 3.35, h: 0.8,
    fontFace: F.sans, fontSize: 16, color: "E7D3A1", align: "center", margin: 0,
  });
  s.addText("No decisive improvement over the reference NNUE.", {
    x: 9.2, y: 3.8, w: 3.35, h: 1.4,
    fontFace: F.sans, fontSize: 16, color: "F4EFE6", align: "center", margin: 0,
  });
  footer(s, 9, "Tables 5.10–5.11, 50,000 held-out positions. Lower CE is better. These figures are not an Elo gain.");
}

// ── 10. Oracle ──────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Same gradient-trained experts, different selection rules");
  title(s, "The oracle exposes assignment headroom");
  const cols = [
    ["0.6301", "Actual dispatcher", "What can run online.", C.white, C.ink, C.muted],
    ["0.6280", "Gradient-centroid proxy", "Still needs the teacher target.", C.panel, C.ink, C.muted],
    ["0.6005", "Minimum-loss oracle", "Each position goes to its best expert.", C.navy, "F4EFE6", "E7D3A1"],
  ];
  cols.forEach((c, i) => {
    const x = 0.55 + i * 4.15;
    card(s, x, 1.6, 3.95, 3.7, c[3]);
    s.addText("K = 16", {
      x: x + 0.25, y: 1.8, w: 3.45, h: 0.3,
      fontFace: F.sans, fontSize: 13, color: c[5], margin: 0,
    });
    s.addText(c[0], {
      x: x + 0.2, y: 2.2, w: 3.55, h: 0.95,
      fontFace: F.serif, fontSize: 40, color: c[4], margin: 0,
    });
    s.addText(c[1], {
      x: x + 0.25, y: 3.3, w: 3.45, h: 0.7,
      fontFace: F.serif, fontSize: 18, color: c[4], margin: 0,
    });
    s.addText(c[2], {
      x: x + 0.25, y: 4.15, w: 3.45, h: 0.8,
      fontFace: F.sans, fontSize: 15, color: c[5], margin: 0,
    });
  });
  footer(s, 10, "Table 5.13. The single-head base is CE 0.6303. Centroid proxy and oracle both need the unavailable teacher target.");
}

// ── 11. Weaker base ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Follow-up on a model that still has room to improve");
  title(s, "Longer fine-tuning helps a weaker base");
  card(s, 0.55, 1.65, 5.9, 3.55);
  s.addText("BASE  ·  W64 / H128", {
    x: 0.85, y: 1.9, w: 5.3, h: 0.35,
    fontFace: F.sans, fontSize: 14, color: C.muted, charSpacing: 1.1, margin: 0,
  });
  s.addText("0.6537", {
    x: 0.85, y: 2.4, w: 5.3, h: 1.15,
    fontFace: F.serif, fontSize: 54, color: C.ink, margin: 0,
  });
  s.addText("Test CE of the single evaluation head.", {
    x: 0.85, y: 3.75, w: 5.3, h: 0.9,
    fontFace: F.sans, fontSize: 18, color: C.inkSoft, margin: 0,
  });
  card(s, 6.85, 1.65, 5.93, 3.55, C.navy);
  s.addText("FOUR L1 EXPERTS", {
    x: 7.15, y: 1.9, w: 5.4, h: 0.35,
    fontFace: F.sans, fontSize: 14, color: "E7D3A1", charSpacing: 1.1, margin: 0,
  });
  s.addText("0.6499", {
    x: 7.15, y: 2.4, w: 5.4, h: 1.15,
    fontFace: F.serif, fontSize: 54, color: "F4EFE6", margin: 0,
  });
  s.addText("Linear dispatcher, longer fine-tuning.\nA reduction of 0.0038.", {
    x: 7.15, y: 3.7, w: 5.3, h: 1.05,
    fontFace: F.sans, fontSize: 18, color: "F4EFE6", margin: 0,
  });
  footer(s, 11, "This is the weaker base, not the strong reference.  §5.5.4, K = 4. The gain depends on the model and the training budget.");
}

// ── 12. KITSUNE method ──────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Proposed two-stage method");
  title(s, "KITSUNE stitches micro-buckets offline");
  const steps = [
    ["1", "Partition L1", "Cut the activation space into many small, convex micro-buckets."],
    ["2", "Stitch offline", "Group those buckets by their mean head gradient. This step never runs during play."],
    ["3", "Train the experts", "Disconnected L1 regions can share one expert, then each head is fine-tuned."],
  ];
  steps.forEach((st, i) => {
    const y = 1.55 + i * 1.65;
    s.addShape(pres.shapes.OVAL, {
      x: 0.55, y: y + 0.28, w: 0.62, h: 0.62,
      fill: { color: i === 1 ? C.accent : C.navy },
      line: { color: i === 1 ? C.accent : C.navy, width: 0 },
    });
    s.addText(st[0], {
      x: 0.55, y: y + 0.28, w: 0.62, h: 0.62,
      fontFace: F.sans, fontSize: 18, color: "F4EFE6", align: "center", valign: "middle", margin: 0,
    });
    card(s, 1.45, y, 11.33, 1.48);
    s.addText(st[1], {
      x: 1.7, y: y + 0.16, w: 10.8, h: 0.42,
      fontFace: F.serif, fontSize: 20, color: C.ink, margin: 0,
    });
    s.addText(st[2], {
      x: 1.7, y: y + 0.64, w: 10.8, h: 0.6,
      fontFace: F.sans, fontSize: 16, color: C.inkSoft, margin: 0,
    });
  });
  footer(s, 12, "Thesis §3.3.5. Schematic, not a plot of measured geometry. Gradients shape the experts; they are not an online feature.");
}

// ── 13. Inference path ──────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "KITSUNE deployment path");
  title(s, "Online inference needs no gradients");
  const path = [
    ["Shared L1", "Already updated\nincrementally"],
    ["Linear router", "Micro-bucket m"],
    ["Static lookup", "Expert T[m]"],
    ["One head", "WDL output"],
  ];
  path.forEach((p, i) => {
    const x = 0.55 + i * 3.2;
    card(s, x, 1.6, 2.7, 2.15, i === 3 ? C.navy : C.white);
    s.addText(p[0], {
      x: x + 0.15, y: 1.8, w: 2.4, h: 0.7,
      fontFace: F.serif, fontSize: 18, color: i === 3 ? "F4EFE6" : C.ink, align: "center", margin: 0,
    });
    s.addText(p[1], {
      x: x + 0.15, y: 2.55, w: 2.4, h: 0.85,
      fontFace: F.sans, fontSize: 14, color: i === 3 ? "E7D3A1" : C.muted, align: "center", margin: 0,
    });
    if (i < 3) {
      s.addShape(pres.shapes.RIGHT_ARROW, {
        x: x + 2.72, y: 2.45, w: 0.4, h: 0.24,
        fill: { color: C.accent }, line: { color: C.accent, width: 0 },
      });
    }
  });
  card(s, 0.55, 4.05, 12.23, 2.55, C.white);
  s.addText("The online cost is a linear map plus a table lookup. Convex micro-buckets are what make that map enough.", {
    x: 0.85, y: 4.3, w: 11.6, h: 0.9,
    fontFace: F.sans, fontSize: 18, color: C.ink, margin: 0,
  });
  s.addText("The routed mixture on the weaker base is measured. It does not require a gradient at play time.", {
    x: 0.85, y: 5.25, w: 11.6, h: 0.9,
    fontFace: F.serif, fontSize: 18, italic: true, color: C.ink, margin: 0,
  });
  footer(s, 13, "§3.3.5. Router cost scales with L1 width and the micro-bucket count. The lookup itself is a fixed table.");
}

// ── 14. KITSUNE results ─────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  content(s);
  kicker(s, "Weaker base W64 / H128  ·  base CE 0.6537  ·  best single-stage L1 0.6499");
  title(s, "KITSUNE matches that L1 result", { h: 0.58, size: 28 });

  const facts = [
    [C.good, C.goodInk, "0.65043", "Routed mixture", "K = 16, two micro-buckets per expert. 0.0033 below the base, and 0.0005 higher than the best L1 run."],
    [C.accentSoft, C.accent, "0.60234", "Best expert, own bucket", "Factor 4. A 7.85% reduction against the base CE. Factor 2 is already at 0.60571."],
    [C.panel, C.ink, "Factor 8", "Stops helping", "At K = 16 the mixture rises to 0.65138 and the best expert to 0.60865."],
  ];
  facts.forEach((f, i) => {
    const y = 1.4 + i * 1.75;
    card(s, 0.5, y, 5.55, 1.62, f[0]);
    s.addText(f[2], {
      x: 0.7, y: y + 0.12, w: 5.15, h: 0.48,
      fontFace: F.serif, fontSize: 24, color: f[1], margin: 0,
    });
    s.addText(f[3], {
      x: 0.7, y: y + 0.58, w: 5.15, h: 0.28,
      fontFace: F.sans, fontSize: 13, color: C.ink, bold: true, margin: 0,
    });
    s.addText(f[4], {
      x: 0.7, y: y + 0.88, w: 5.15, h: 0.62,
      fontFace: F.sans, fontSize: 12, color: C.inkSoft, margin: 0,
    });
  });

  const header = ["K", "B′/K", "MoE CE", "Best expert", "Gain"].map((t) =>
    cell(t, { fill: C.navy, color: "F4EFE6", bold: true, size: 11 })
  );
  const raw = [
    ["4", "4", "0.65118", "0.63726", "2.51%"],
    ["4", "8", "0.65137", "0.64050", "2.01%"],
    ["8", "4", "0.65102", "0.61748", "5.54%"],
    ["8", "8", "0.65093", "0.61275", "6.26%"],
    ["16", "2", "0.65043", "0.60571", "7.34%"],
    ["16", "4", "0.65061", "0.60234", "7.85%"],
    ["16", "8", "0.65138", "0.60865", "6.89%"],
  ];
  const body = raw.map((r, i) => {
    const fill = i === 4 ? "E5EFE8" : i === 5 ? "F3E4E1" : i === 6 ? C.panel : C.white;
    return r.map((t, j) => cell(t, { fill, bold: j === 2 || j === 3, size: 12 }));
  });
  s.addTable([header, ...body], {
    x: 6.25, y: 1.4, w: 6.55, h: 5.15,
    colW: [0.85, 1.05, 1.5, 1.75, 1.4],
    border: [
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
      { pt: 0, color: C.bg },
    ],
    valign: "middle",
  });
  footer(s, 14, "Table 5.16. Expert gain uses the full-base CE, not a same-bucket comparison. Strong-base KITSUNE, Elo, and nodes per second are still open.");
}

// ── 15. Close ───────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  checker(s, 12.35, 0.4, 0.42, "F4EFE6", "C0392B");
  s.addText("MAIN CONCLUSION", {
    x: 0.6, y: 0.45, w: 10, h: 0.3,
    fontFace: F.sans, fontSize: 13, color: "E7D3A1", charSpacing: 1.6, margin: 0,
  });
  s.addText("Useful experts need\na cheap, reliable assignment.", {
    x: 0.6, y: 1.05, w: 12, h: 1.55,
    fontFace: F.serif, fontSize: 34, color: "F4EFE6", margin: 0,
  });
  const lines = [
    ["Online", "KITSUNE routes with L1 and a lookup. Gradients are used only while the buckets are stitched."],
    ["Mixture", "On the weaker base, routed CE 0.65043 matches the best single-stage L1 result, 0.6499."],
    ["Experts", "The best head, on its own bucket, reaches 0.60234. Past four micro-buckets per expert, both numbers worsen."],
  ];
  lines.forEach((ln, i) => {
    const y = 2.9 + i * 0.95;
    s.addText(ln[0], {
      x: 0.6, y: y, w: 1.7, h: 0.75,
      fontFace: F.sans, fontSize: 14, color: "E7D3A1", margin: 0, valign: "top",
    });
    s.addText(ln[1], {
      x: 2.4, y: y, w: 10.1, h: 0.8,
      fontFace: F.sans, fontSize: 16, color: "F4EFE6", margin: 0,
    });
  });
  s.addText("Thank you.    Questions?", {
    x: 0.6, y: 6.55, w: 8, h: 0.45,
    fontFace: F.serif, fontSize: 22, italic: true, color: "F4EFE6", margin: 0,
  });
  s.addText("Omar Cusma Fait   ·   University of Trieste", {
    x: 7.3, y: 6.65, w: 5.4, h: 0.35,
    fontFace: F.sans, fontSize: 13, color: "C9C2B6", align: "right", margin: 0,
  });
}

pres.writeFile({ fileName: "Presentation/Chess_MoE_defense.pptx" })
  .then(() => console.log("wrote Presentation/Chess_MoE_defense.pptx"))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
