// Number formatting: English text with commas for thousands. A number never separates from its unit or its
// sign at a line end (no-break space), and negative numbers use a true minus sign.
const nf = (d: number) => new Intl.NumberFormat("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
const NB = "\u00a0";
/** Format with d decimals; "−" for negatives and never a negative zero. */
const num = (x: number, d: number) => {
  const s = nf(d).format(Math.abs(x));
  return x < 0 && Number(s.replace(/,/g, "")) !== 0 ? `−${s}` : s;
};

export const int = (x: number | null | undefined) => (x == null ? "–" : num(Math.round(x), 0));
export const dec = (x: number | null | undefined, d = 1) => (x == null ? "–" : num(x, d));
export const pct = (x: number | null | undefined, d = 1) => (x == null ? "–" : `${num(x, d)}${NB}%`);
export const chf = (x: number | null | undefined, d = 0) => (x == null ? "–" : `CHF${NB}${num(x, d)}`);
export const kwh = (x: number | null | undefined) => (x == null ? "–" : `${num(Math.round(x), 0)}${NB}kWh`);
export const signedChf = (x: number, d = 0) => `${x < 0 ? "−" : "+"}CHF${NB}${nf(d).format(Math.abs(x))}`;
/** A signed amount without the currency, for table columns whose header names it. */
export const signed = (x: number, d = 0) => `${x < 0 ? "−" : "+"}${nf(d).format(Math.abs(x))}`;

/** Energy in joules, with as many decimals as its size needs. */
export const joule = (x: number) => `${num(x, x < 10 ? 2 : x < 100 ? 1 : 0)}${NB}J`;
/** A rate of arithmetic operations in GFLOP (a billion FLOPs) a second. */
export const gflop = (x: number) => `${num(x / 1e9, x < 1e9 ? 3 : x < 1e10 ? 2 : 1)}${NB}GFLOP`;
/** A count of arithmetic operations in words: 1.94 billion, 45 million. */
export const ops = (x: number) => (x >= 1e12 ? `${num(x / 1e12, 2)} trillion` : x >= 1e9 ? `${num(x / 1e9, x < 1e10 ? 2 : 1)} billion`
  : x >= 1e6 ? `${num(x / 1e6, x < 1e7 ? 1 : 0)} million` : num(Math.round(x), 0));

/** In a sentence from the data files, tie each number to the unit that follows it. */
export const tidy = (s: string) => s.replace(/(\d) (%|kWh|kW|W|K|CHF|°C)(?![A-Za-z])/g, `$1${NB}$2`);

/** Clean axis ticks between 0 and max (about n ticks). */
export function ticks(max: number, n = 4, min = 0): number[] {
  const span = max - min || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
export const niceMax = (max: number, n = 4) => {
  const t = ticks(max, n);
  const last = t[t.length - 1];
  return last >= max ? last : last + (t[1] - t[0]);
};
