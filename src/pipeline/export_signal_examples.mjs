import fs from "node:fs";
import path from "node:path";

const inputPath = path.resolve("data/fuel_label_dataset/all_labeled_points.csv");
const outputDir = path.resolve("artifacts/slide_assets/signal_examples");
const contextPoints = 14;

function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { cell += char; i += 1; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) { row.push(cell); cell = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i += 1;
      row.push(cell); cell = "";
      if (row.length > 1) rows.push(row);
      row = [];
    } else cell += char;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  const [headers, ...values] = rows;
  return values.map((valuesRow) => Object.fromEntries(headers.map((header, index) => [header, valuesRow[index] ?? ""])));
}

const n = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};
const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[char]));

function selectExample(groups, predicate, score, descending = true) {
  let chosen = null;
  for (const points of groups.values()) {
    for (let index = 3; index < points.length - 3; index += 1) {
      const point = points[index];
      if (!predicate(point)) continue;
      const candidate = { point, points, index, score: score(point, points, index) };
      if (!chosen || (descending ? candidate.score > chosen.score : candidate.score < chosen.score)) chosen = candidate;
    }
  }
  if (!chosen) throw new Error("Khong tim thay mau phu hop de xuat anh.");
  return chosen;
}

function localValues(points, index, radius = 7) {
  const start = Math.max(0, index - radius);
  const end = Math.min(points.length, index + radius + 1);
  return points.slice(start, end).map((point) => n(point.FuelLevel));
}

function localLabelCount(points, index, label, radius = 7) {
  const start = Math.max(0, index - radius);
  const end = Math.min(points.length, index + radius + 1);
  return points.slice(start, end).filter((point) => point.Label === label).length;
}

function writeSvg(example, fileName, title, caption, accent) {
  const width = 1400, height = 720, left = 120, right = 70, top = 112, bottom = 108;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const start = Math.max(0, example.index - contextPoints);
  const end = Math.min(example.points.length - 1, example.index + contextPoints);
  const window = example.points.slice(start, end + 1);
  const values = window.map((point) => n(point.FuelLevel));
  let min = Math.min(...values), max = Math.max(...values);
  const pad = Math.max((max - min) * 0.16, 0.5);
  min -= pad; max += pad;
  if (max === min) max += 1;
  const coords = values.map((value, index) => ({
    x: left + (plotWidth * index) / Math.max(values.length - 1, 1),
    y: top + (plotHeight * (max - value)) / (max - min),
  }));
  const event = coords[example.index - start];
  const grid = Array.from({ length: 6 }, (_, index) => {
    const y = top + (plotHeight * index) / 5;
    const label = (max - ((max - min) * index) / 5).toFixed(1);
    return `<line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" stroke="#dfe7ef" stroke-width="1"/><text x="38" y="${y + 6}" class="axis">${label}</text>`;
  }).join("\n");
  const polyline = coords.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
  const dots = coords.map((point) => `<circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="3.1" fill="#e22c2c"/>`).join("");
  const detail = `Xe: ${example.point.VehicleID} | ${example.point.FuelTime} | Fuel: ${n(example.point.FuelLevel).toFixed(1)} L`;
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    .title { font: 700 29px Arial, sans-serif; fill: #1c2b41; }
    .caption { font: 16px Arial, sans-serif; fill: #60708a; }
    .axis { font: 14px Arial, sans-serif; fill: #60708a; }
    .detail { font: 14px Arial, sans-serif; fill: #60708a; }
    .callout { font: 16px Arial, sans-serif; fill: #1c2b41; }
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="52" y="48" class="title">${esc(title)}</text>
  <text x="54" y="80" class="caption">${esc(caption)}</text>
  ${grid}
  <polyline fill="none" stroke="#e22c2c" stroke-width="3" points="${polyline}"/>
  ${dots}
  <circle cx="${event.x}" cy="${event.y}" r="10" fill="${accent}" stroke="#414141" stroke-width="2"/>
  <text x="${Math.min(event.x + 17, width - 170)}" y="${Math.max(event.y - 17, 100)}" class="callout">Diem mau</text>
  <text x="${left}" y="${height - 28}" class="detail">${esc(detail)}</text>
  <text x="52" y="${height - 58}" class="axis">Muc nhien lieu (L)</text>
</svg>`;
  const outputPath = path.join(outputDir, fileName);
  fs.writeFileSync(outputPath, svg, "utf8");
  return { File: outputPath, Title: title, VehicleID: example.point.VehicleID, FuelTime: example.point.FuelTime, FuelLevel: n(example.point.FuelLevel).toFixed(2), Label: example.point.Label };
}

fs.mkdirSync(outputDir, { recursive: true });
const rows = parseCsv(fs.readFileSync(inputPath, "utf8"));
const groups = new Map();
for (const row of rows) {
  const key = `${row.Source}|${row.VehicleID}`;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(row);
}
for (const points of groups.values()) points.sort((a, b) => String(a.FuelTime).localeCompare(String(b.FuelTime)));

const examples = [
  [selectExample(groups, (p) => p.Label === "REFUEL" && n(p.DeltaFuel) > 0, (p) => n(p.DeltaFuel)), "01_refuel.svg", "REFUEL - Tăng nhiên liệu thật", "Tăng lớn duy trì sau sự kiện, cần bám nhanh.", "#159f77"],
  [selectExample(groups, (p) => p.Label === "DRAIN" && n(p.DeltaFuel) < 0, (p) => n(p.DeltaFuel), false), "02_drain.svg", "DRAIN - Giảm nhiên liệu bất thường", "Giảm lớn cần được phân biệt với nhiễu ngắn hạn.", "#d53535"],
  [selectExample(groups, (p) => p.Label === "CONSUMPTION" && n(p.DeltaFuel) < 0 && n(p.RollingStd12) < 2, (p) => n(p.RollingStd12), false), "03_consumption.svg", "CONSUMPTION - Giảm nhiên liệu theo xu hướng", "Nhiên liệu giảm từ từ và duy trì cùng chiều theo thời gian.", "#2972b5"],
  [selectExample(
    groups,
    (p) => p.Label === "STABLE_JITTER" && n(p.RollingStd12) >= 0.08 && n(p.RollingStd12) <= 0.9 && Math.abs(n(p.DeltaFuel)) <= 0.8,
    (p, points, index) => {
      const values = localValues(points, index);
      const range = Math.max(...values) - Math.min(...values);
      return localLabelCount(points, index, "STABLE_JITTER") * 10 - Math.abs(range - 1.0);
    }
  ), "04_stable_jitter.svg", "STABLE_JITTER - Dao động nhỏ quanh mức ổn định", "Dao động nhỏ nhưng liên tục quanh một mức, cần được giữ phẳng bằng deadband.", "#2d9c44"],
  [selectExample(
    groups,
    (p) => p.Label === "SLOSHING_NOISE" && n(p.RollingStd12) >= 1.0,
    (p, points, index) => {
      const values = localValues(points, index);
      const range = Math.max(...values) - Math.min(...values);
      return localLabelCount(points, index, "SLOSHING_NOISE") * 12 + range + n(p.RollingStd12);
    }
  ), "05_sloshing_noise.svg", "SLOSHING_NOISE - Dao động mạnh do rung lắc/sóng sánh", "Một cụm dao động liên tục không nên được bộ lọc bám theo từng điểm raw.", "#ee8318"],
  [selectExample(groups, (p) => p.Label === "SLOSHING_NOISE" && (n(p.PeakReversalFlag) === 1 || n(p.ValleyReversalFlag) === 1), (p) => n(p.AbsDeltaFuel)), "06_dropout_spike.svg", "DROPOUT/SPIKE - Nhảy/tụt đột ngột rồi quay lại", "Sự kiện ngắn hạn cần được từ chối thay vì cập nhật mức nhiên liệu.", "#6c4fa9"],
];

const manifest = examples.map(([example, fileName, title, caption, accent]) => writeSvg(example, fileName, title, caption, accent));
fs.writeFileSync(path.join(outputDir, "signal_examples_manifest.json"), JSON.stringify(manifest, null, 2));
console.log(`Da xuat ${manifest.length} anh SVG vao: ${outputDir}`);
for (const item of manifest) console.log(`${path.basename(item.File)} | ${item.VehicleID} | ${item.Label} | ${item.FuelTime}`);
