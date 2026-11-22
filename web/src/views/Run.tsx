// Run jobs: each of the four job types answers on the spot. The page talks to the local service
// (python -m dh.serve), which runs the job's own model or arithmetic on this computer. Without the service
// the page shows saved answers of the same functions.
import {
  useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type KeyboardEvent, type PointerEvent, type ReactNode,
  type SyntheticEvent,
} from "react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  HERE, licenceName, similarToUpload, useApiHealth, useCaptionStatus, useEmbedSearch, useJobsStatus, useLlmStatus,
  useMeter, usePhotoSample, usePhotoSimilar, useSummary, useTryDemo, writeAnswer,
  type EmbedHit, type EmbedResult, type JobsStatus, type LlmDone, type PhotoHit, type PhotoSimilar, type TryDemo, type Workload,
} from "../data";
import { dec, gflop, int, joule, ops } from "../format";
import { Badge, Effort, LoadError, Loading, PageHead } from "../components/ui";
import { DescribePanel } from "./Describe";
import ASKS from "../llm_examples.json";
import "./jobs.css";
import "./run.css";

export const RUN_JOBS = [
  { key: "embed", label: "Text embeddings" },
  { key: "image", label: "Photo indexing" },
  { key: "render", label: "Tile rendering" },
  { key: "caption", label: "Photo descriptions" },
  { key: "llm", label: "Text generation" },
] as const;
export type RunJob = (typeof RUN_JOBS)[number]["key"];
export const isRunJob = (v: unknown): v is RunJob => RUN_JOBS.some((j) => j.key === v);

const NB = " ";
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"];
const FORMATS = "JPEG, PNG, WebP, GIF, BMP or TIFF, up to 15 MB";
const ms = (s: number) => (s < 0.0995 ? `${dec(s * 1000, s < 0.00995 ? 1 : 0)}${NB}ms` : `${dec(s, 2)}${NB}s`);
const one = (unit: string) => unit.replace(/s$/, "");

/** Width of an element that may appear after the first render (the hook in ui.tsx measures once, on mount). */
function useMeasuredWidth(fallback: number) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, w] as const;
}

function useDebounced<T>(value: T, delay: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), delay);
    return () => window.clearTimeout(t);
  }, [value, delay]);
  return v;
}

/** What was measured for this job on the laptop, and whether the service is working offline. */
type Net = { guard_installed: boolean; blocked_attempts: number };
const netText = (n: Net) => (!n.guard_installed ? "not blocked"
  : `blocked; ${n.blocked_attempts === 0 ? "no attempt" : n.blocked_attempts === 1 ? "1 attempt refused" : `${int(n.blocked_attempts)} attempts refused`} so far`);

function Facts({ job, net, children }: { job?: Workload; net?: Net; children?: ReactNode }) {
  return (
    <dl className="run-facts">
      {children}
      {job ? <div><dt>Measured for this job</dt><dd>{job.per_s >= 1 ? `${dec(job.per_s, 1)} ${job.unit} a second` : `${dec(1 / job.per_s, 1)} seconds per ${one(job.unit)}`}{job.j_per_unit != null ? <>, {dec(job.j_per_unit, 3)}{NB}J per {one(job.unit)}</> : null}</dd></div> : null}
      {job?.quality ? <div><dt>Measurement note</dt><dd>{job.quality}</dd></div> : null}
      {net ? <div><dt>Outside network</dt><dd>{netText(net)}</dd></div> : null}
    </dl>
  );
}

function Offline({ what }: { what: string }) {
  return (
    <p className="callout">
      <strong>This job runs on the team laptop.</strong> Its local service is not reachable from this copy of the
      dashboard, so {what}
    </p>
  );
}

// ------------------------------------------------------------------ text embeddings
const EXAMPLES = ["what does electricity cost", "how big is the water tank", "how long may a job wait", "price of a used server"];

function showValue(v: unknown, unit: string): string {
  if (typeof v === "number") {
    const d = Math.min(5, (String(v).split(".")[1] ?? "").length);
    return `${dec(v, d)}${unit ? NB + unit : ""}`;
  }
  if (typeof v === "string") return unit ? `${v}${NB}${unit}` : v;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) {
    if (v.length > 6) return `${v.length} values`;
    return `${v.map((x) => (typeof x === "number" ? showValue(x, "") : String(x))).join(", ")}${unit ? NB + unit : ""}`;
  }
  if (v && typeof v === "object") return "several values";
  return "";
}

function EmbedResults({ r, stale }: { r: EmbedResult; stale: boolean }) {
  return (
    <ol className="run-hits" data-stale={stale} aria-label={`Entries closest in meaning to: ${r.query}`}>
      {r.results.map((h: EmbedHit) => (
        <li key={h.id}>
          <span className="run-score" aria-hidden="true">
            <i style={{ width: `${Math.min(100, Math.max(2, h.score * 100))}%` }} />
          </span>
          <span className="run-hit-main">
            <span className="run-hit-label">{h.label}</span>
            <span className="run-hit-meta">{h.group}{showValue(h.value, h.unit) ? <>: <b>{showValue(h.value, h.unit)}</b></> : null}</span>
          </span>
          <span className="run-hit-side"><Badge status={h.status} /><span className="num">closeness {dec(h.score, 2)}</span></span>
        </li>
      ))}
    </ol>
  );
}

function EmbedPanel({ live, status, job, demo }: { live: boolean; status?: JobsStatus; job?: Workload; demo?: TryDemo }) {
  const [text, setText] = useState(EXAMPLES[0]);
  const q = useDebounced(text, 160);
  const res = useEmbedSearch(q, live);
  const saved = demo ? Object.keys(demo.text) : [];
  const [pick, setPick] = useState<string | null>(null);
  const ready = status?.embed.loaded;
  const corpus = res.data?.corpus ?? status?.embed.corpus;
  const tooShort = text.trim().length < 2;
  return (
    <section className="stage run-stage" aria-labelledby="run-embed-h">
      <div className="run-head">
        <div>
          <h2 id="run-embed-h" className="stage-title" tabIndex={-1}>{live ? "Ask the assumptions table in your own words" : "The assumptions table, searched by meaning"}</h2>
          <p className="run-what">
            The text-embedding job turns sentences into vectors of numbers. Here its model has done that for every
            entry of the assumptions table.{" "}
            {live ? "Your sentence becomes a vector too, and the entries closest in meaning come first, even when they share no word with it."
              : "Each sentence below became a vector too, and the entries closest in meaning come first, even when they share no word with it."}
          </p>
        </div>
        <Facts job={job} net={live ? status?.offline : undefined}>
          {live ? <div><dt>Text model</dt><dd>{status?.embed.error ? `could not be loaded: ${status.embed.error}` : ready ? `ready on ${HERE}` : `loading on ${HERE}`}</dd></div> : null}
        </Facts>
      </div>
      {live ? (
        <>
          <label className="field run-ask">
            Your sentence, in English
            <input className="input" type="search" value={text} onChange={(e) => setText(e.target.value)} placeholder="what does electricity cost" autoComplete="off" spellCheck={false} />
          </label>
          <div className="run-examples" role="group" aria-label="Example sentences">
            {EXAMPLES.map((x) => <button key={x} type="button" className="btn" aria-pressed={text === x} onClick={() => setText(x)}>{x}</button>)}
          </div>
          {res.isError ? <p className="callout" role="alert"><strong>The job did not answer.</strong> Try again in a moment.</p> : null}
          <p className="sr-only" role="status">{!tooShort && res.data?.results.length && !res.isPlaceholderData && q === text ? `Closest entry: ${res.data.results[0].label}` : ""}</p>
          {tooShort ? <p className="run-readout">Type a few words, or pick an example.</p> : res.data && res.data.results.length ? (
            <>
              <p className="run-readout">
                Your sentence became a vector of {corpus?.dims ?? 384} numbers in <b>{ms(res.data.seconds ?? 0)}</b> and was
                compared with {corpus?.items ?? res.data.results.length} entries.
                {corpus ? <> The table itself was turned into vectors in {ms(corpus.seconds)}, {dec(corpus.per_second, 0)} entries a second.</> : null}
              </p>
              {res.data.work ? <p className="run-readout run-effort"><Effort w={res.data.work} /></p> : null}
              <p className="run-credit">Closeness runs from 0 (nothing in common) to 1 (the same meaning); the bars are drawn on that scale.</p>
              <EmbedResults r={res.data} stale={res.isPlaceholderData || q !== text} />
            </>
          ) : <p className="run-readout">{ready ? "Computing…" : `Loading the text model on ${HERE} (a few seconds, once)…`}</p>}
        </>
      ) : demo && saved.length ? (
        <>
          <div className="run-examples" role="group" aria-label="Saved sentences">
            {saved.map((x) => <button key={x} type="button" className="btn" aria-pressed={(pick ?? saved[0]) === x} onClick={() => setPick(x)}>{x}</button>)}
          </div>
          {demo.text[pick ?? saved[0]].work ? <p className="run-readout run-effort">Recorded on the team laptop. <Effort w={demo.text[pick ?? saved[0]].work} /></p> : null}
          <EmbedResults r={demo.text[pick ?? saved[0]]} stale={false} />
        </>
      ) : null}
    </section>
  );
}

// ------------------------------------------------------------------ photo indexing
/** The credit of an archive photo: date, creator, archive, the licence and the record. */
function PhotoCredit({ hit }: { hit: PhotoHit }) {
  return (
    <span className="run-photo-credit">
      {[hit.date, hit.creator, hit.institution].filter(Boolean).join(", ")},{" "}
      <a href={hit.licence_url} target="_blank" rel="noopener noreferrer">{licenceName(hit.licence)}</a>,{" "}
      <a href={hit.object_url} target="_blank" rel="noopener noreferrer">view on Memobase</a>
    </span>
  );
}

/** A thumbnail that is not on this computer (they are downloaded separately) leaves its frame empty. */
const hideMissing = (e: SyntheticEvent<HTMLImageElement>) => { e.currentTarget.style.visibility = "hidden"; };

function PhotoCard({ hit, onPick, live }: { hit: PhotoHit; onPick?: (h: PhotoHit) => void; live: boolean }) {
  return (
    <li className="jobs-hit run-photo">
      {live ? (
        <button type="button" className="run-photo-btn" onClick={() => onPick?.(hit)} aria-label={`Find photos like: ${hit.title}`} title={hit.credit}>
          <img src={"/" + hit.thumb} alt="" onError={hideMissing} />
        </button>
      ) : <img src={"/data/" + hit.thumb} alt="" title={hit.credit} onError={hideMissing} />}
      <h3>{hit.title}</h3>
      <span>{[hit.date, hit.creator].filter(Boolean).join(", ")}</span>
      <span>{hit.institution}</span>
      {hit.score != null ? <span className="num">closeness {dec(hit.score, 2)}</span> : null}
      <span>
        <a href={hit.licence_url} target="_blank" rel="noopener noreferrer">{licenceName(hit.licence)}</a>,{" "}
        <a href={hit.object_url} target="_blank" rel="noopener noreferrer">view on Memobase</a>
      </span>
    </li>
  );
}

function PhotoPanel({ live, status, job, demo }: { live: boolean; status?: JobsStatus; job?: Workload; demo?: TryDemo }) {
  const [seed, setSeed] = useState(3);
  const [query, setQuery] = useState<PhotoHit | null>(null);
  const [mine, setMine] = useState<{ url: string; name: string; result?: PhotoSimilar; error?: string } | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement | null>(null);
  const sample = usePhotoSample(seed, live && !query && !mine);
  const similar = usePhotoSimilar(query?.signature ?? "", live && !!query);
  const [savedAt, setSavedAt] = useState(0);
  const photos = status?.photo.photos ?? similar.data?.photos ?? sample.data?.photos ?? demo?.photo_count;

  const mineUrl = mine?.url;
  useEffect(() => () => { if (mineUrl) URL.revokeObjectURL(mineUrl); }, [mineUrl]);
  const [refused, setRefused] = useState(false);

  const upload = useCallback((files: FileList | File[]) => {
    const f = Array.from(files).find((x) => IMAGE_TYPES.includes(x.type));
    setRefused(!f && files.length > 0);
    if (!f) return;
    const url = URL.createObjectURL(f);
    setQuery(null);
    setMine({ url, name: f.name });
    similarToUpload(f).then(
      (result) => setMine((m) => (m && m.url === url ? { ...m, result } : m)),
      (e: Error) => setMine((m) => (m && m.url === url ? { ...m, error: e.message } : m)),
    );
  }, []);
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDrag(false); if (live) upload(e.dataTransfer.files); };
  const restart = () => { setQuery(null); setMine(null); setRefused(false); setSeed((s) => s + 1); };

  // The answer that belongs to what is asked right now. Until it arrives the list shown before stays, dimmed;
  // it also stays when the request failed, and then nothing is announced as still on its way.
  const fresh = mine ? mine.result?.results : query ? (similar.isPlaceholderData ? undefined : similar.data?.results) : sample.data?.results;
  const failed = mine ? !!mine.error : query ? similar.isError : sample.isError;
  const last = useRef<PhotoHit[]>([]);
  if (fresh) last.current = fresh;
  const hits = fresh ?? last.current;
  const stale = !fresh;
  const pending = stale && !failed;
  const shownCount = status?.photo.photos_shown ?? similar.data?.photos_shown ?? sample.data?.photos_shown ?? demo?.photos_shown;
  return (
    <section className="stage run-stage" aria-labelledby="run-image-h" data-drag={drag}
      onDragOver={(e) => { if (live) { e.preventDefault(); setDrag(true); } }} onDragLeave={() => setDrag(false)} onDrop={onDrop}>
      <div className="run-head">
        <div>
          <h2 id="run-image-h" className="stage-title" tabIndex={-1}>{live ? "Find archive photos that look alike" : "Archive photos that look alike"}</h2>
          <p className="run-what">
            The photo-indexing job turned {photos ? int(photos) : "the"} openly licensed or public-domain photos from
            Swiss archives into vectors.{" "}
            {shownCount ? `The ${int(shownCount)} whose archive record sets no extra conditions of use are shown here. ` : ""}
            {live ? `Pick any photo to see the ones closest to it, or give it a photo of your own: the job's model turns it into a vector on ${HERE} and compares it with them.`
              : "For each photo named below, these are the photos whose vectors are closest to its own."}
          </p>
        </div>
        <Facts job={job} net={live ? status?.offline : undefined}>
          {live ? <div><dt>Photo model</dt><dd>{status?.photo.error ? `could not be loaded: ${status.photo.error}` : status?.photo.loaded ? `ready on ${HERE}` : `loading on ${HERE}`}</dd></div> : null}
        </Facts>
      </div>
      {live ? (
        <>
          <div className="run-photo-bar">
            {mine ? (
              <div className="run-query">
                <img src={mine.url} alt="Your photo" />
                <div>
                  <span className="run-query-k">Photos most like your photo</span>
                  <span className="run-readout" aria-live="polite">
                    {mine.error ? `Could not read this file: ${mine.error}`
                      : mine.result ? <>Turned into a vector in <b>{ms(mine.result.encode_seconds ?? 0)}</b>, compared with {int(mine.result.photos_shown)} photos in {ms(mine.result.rank_seconds)}. <Effort w={mine.result.work} /></>
                      : status?.photo.loaded ? "Computing…" : `Loading the photo model on ${HERE} (a few seconds, once)…`}
                  </span>
                </div>
              </div>
            ) : query ? (
              <div className="run-query">
                <img src={"/" + query.thumb} alt="" onError={hideMissing} />
                <div>
                  <span className="run-query-k">Photos most like</span>
                  <span className="run-query-t">{query.title}</span>
                  <PhotoCredit hit={query} />
                  <span className="run-readout" aria-live="polite">
                    {similar.isError ? "The job did not answer."
                      : similar.data && !similar.isPlaceholderData ? <>Compared with the {int(similar.data.photos_shown - 1)} other photos in <b>{ms(similar.data.rank_seconds)}</b>.</>
                      : "Comparing…"}
                  </span>
                </div>
              </div>
            ) : <p className="run-readout">A handful of the archive photos. Pick one to start.</p>}
            <div className="btn-row">
              <button type="button" className="btn btn-primary" onClick={() => input.current?.click()}>Use a photo of your own</button>
              <button type="button" className="btn" onClick={restart}>Show other photos</button>
              <input ref={input} type="file" accept={IMAGE_TYPES.join(",")} hidden aria-label="Photo to compare"
                onChange={(e) => { if (e.target.files) upload(e.target.files); e.target.value = ""; }} />
            </div>
          </div>
          {refused ? <p className="callout" role="alert"><strong>This file cannot be used.</strong> Choose a photo as {FORMATS}.</p> : null}
          {sample.isError || similar.isError ? <p className="callout" role="alert"><strong>The job did not answer.</strong> Try again in a moment.</p> : null}
          <ul className="jobs-grid run-photos" data-stale={stale} aria-busy={pending} aria-label={pending ? "Photos shown before; the new answer is being computed" : failed ? "Photos shown before; the new answer failed" : query ? `Photos most like ${query.title}` : mine ? "Photos most like your photo" : "Archive photos to start from"}>
            {hits.map((h) => <PhotoCard key={h.signature} hit={h} live onPick={(p) => { setMine(null); setRefused(false); setQuery(p); }} />)}
          </ul>
          <p className="run-credit">Photos: Swiss archives via Memobase / Memoriav, each shown whole under the licence or public-domain mark named beside it; only records without extra conditions of use appear. Your own photo may be {FORMATS}. A general model ranks the photos; it can be wrong.</p>
        </>
      ) : demo?.photos.length ? (
        <>
          <div className="run-examples" role="group" aria-label="Saved comparisons">
            {demo.photos.map((p, i) => <button key={p.query.signature} type="button" className="btn" aria-pressed={savedAt === i} onClick={() => setSavedAt(i)}>{p.query.title}</button>)}
          </div>
          <div className="run-query">
            <img src={"/data/" + demo.photos[savedAt].query.thumb} alt="" onError={hideMissing} />
            <div>
              <span className="run-query-k">Photos most like</span>
              <span className="run-query-t">{demo.photos[savedAt].query.title}</span>
              <PhotoCredit hit={demo.photos[savedAt].query} />
            </div>
          </div>
          <p className="run-readout">The photos whose vectors from the job are closest to this photo's{shownCount ? `, among the ${int(shownCount - 1)} other photos without extra conditions of use` : ""}.</p>
          <ul className="jobs-grid run-photos">
            {demo.photos[savedAt].results.map((h) => <PhotoCard key={h.signature} hit={h} live={false} />)}
          </ul>
          <p className="run-credit">Photos: Swiss archives via Memobase / Memoriav, each shown whole under the licence or public-domain mark named beside it. A general model ranks them; it can be wrong.</p>
        </>
      ) : null}
    </section>
  );
}

// ------------------------------------------------------------------ tile rendering
type View = { cx: number; cy: number; span: number };
type Shot = {
  url: string; view: View; iters: number; seconds: number; w: number; h: number; tiles: number; preview: boolean;
  joules: number | null; watts: number | null; flops: number | null;
};
const HOME: View = { cx: -0.6, cy: 0, span: 3.2 };
const PLACES: { name: string; view: View }[] = [
  { name: "The whole set", view: HOME },
  { name: "Seahorse valley", view: { cx: -0.745, cy: 0.11, span: 0.02 } },
  { name: "A spiral at the edge", view: { cx: -0.7435, cy: 0.1314, span: 0.0012 } },
];
const same = (a: View, b: View) => a.cx === b.cx && a.cy === b.cy && a.span === b.span;
const zoomText = (z: number) => `${z < 10 ? dec(z, 1) : int(z)}${NB}×`;
const TILE_NOTE = `One tile of the job is 320 × 40${NB}px with up to 200 steps per pixel; the work in a picture is counted in such tiles (pixels times steps).`;
const clampView = (v: View): View => ({
  cx: Math.min(2.5, Math.max(-2.5, v.cx)), cy: Math.min(2.5, Math.max(-2.5, v.cy)), span: Math.min(5, Math.max(2e-11, v.span)),
});

function RenderPanel({ live, job, demo, status }: { live: boolean; job?: Workload; demo?: TryDemo; status?: JobsStatus }) {
  const [wrap, width] = useMeasuredWidth(720);
  const W = Math.min(720, Math.max(240, Math.round(width)));
  const H = Math.round(W * (W < 480 ? 0.75 : 0.625));
  const [view, setView] = useState<View>(HOME);
  const [iters, setIters] = useState(200);
  const [shot, setShot] = useState<Shot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState(0);
  const want = useRef({ view, iters, W, H });
  want.current = { view, iters, W, H };
  const running = useRef(false);
  const next = useRef<"preview" | "full" | null>(null);
  const shown = useRef<string | null>(null);
  const box = useRef<HTMLDivElement | null>(null);
  const dragFrom = useRef<{ x: number; y: number; view: View; moved: boolean } | null>(null);
  const armed = useRef(false);                                   // a click on the picture lets the wheel zoom it
  const alive = useRef(true);
  const inflight = useRef<AbortController | null>(null);

  // One request at a time. While the view keeps changing only quick previews are asked for; the full picture
  // follows as soon as the view rests.
  const run = useCallback(async (kind: "preview" | "full") => {
    if (running.current) { next.current = kind === "full" || next.current === "full" ? "full" : "preview"; return; }
    running.current = true;
    const { view: v, iters: it, W: w, H: h } = want.current;
    const pw = kind === "preview" ? Math.round(w / 3) : w, ph = kind === "preview" ? Math.round(h / 3) : h;
    const pit = kind === "preview" ? Math.min(it, 150) : it;
    const ctl = new AbortController();
    inflight.current = ctl;
    const timer = window.setTimeout(() => ctl.abort(), 10000);    // a request that never answers must not block the next
    try {
      const res = await fetch(`/api/render?cx=${v.cx}&cy=${v.cy}&span=${v.span}&w=${pw}&h=${ph}&iters=${pit}`, { signal: ctl.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      if (!alive.current) return;
      const url = URL.createObjectURL(blob);
      const head = (n: string) => Number(res.headers.get(n) ?? 0);
      const maybe = (n: string) => { const s = res.headers.get(n); return s ? Number(s) : null; };    // absent from an older service
      setShot({ url, view: v, iters: pit, seconds: head("X-Render-Seconds"), w: head("X-Render-Width"), h: head("X-Render-Height"),
                tiles: head("X-Render-Tiles"), preview: kind === "preview",
                joules: maybe("X-Render-Joules"), watts: maybe("X-Render-Watts"), flops: maybe("X-Render-Flops") });
      if (shown.current) URL.revokeObjectURL(shown.current);
      shown.current = url;
      setError(null);
    } catch (e) {
      if (alive.current) setError((e as Error).message);
    } finally {
      window.clearTimeout(timer);
      running.current = false;
      const k = next.current;
      next.current = null;
      if (k && alive.current) void run(k);
    }
  }, []);

  useEffect(() => {
    if (!live) return;
    void run("preview");
    const t = window.setTimeout(() => void run("full"), 220);
    return () => window.clearTimeout(t);
  }, [live, view, iters, W, H, run]);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      inflight.current?.abort();
      if (shown.current) URL.revokeObjectURL(shown.current);
      shown.current = null;
    };
  }, []);

  const zoomAt = useCallback((factor: number, px = 0.5, py = 0.5) => {
    setView((v) => {
      const k = H / W;
      const X = v.cx + (px - 0.5) * v.span, Y = v.cy + (py - 0.5) * v.span * k;
      const span = Math.min(5, Math.max(2e-11, v.span * factor));
      return clampView({ cx: X - (px - 0.5) * span, cy: Y - (py - 0.5) * span * k, span });
    });
  }, [W, H]);
  const pan = (dx: number, dy: number) => setView((v) => clampView({ ...v, cx: v.cx + dx * v.span, cy: v.cy + dy * v.span }));

  // After a click on the picture, until the pointer leaves it, or with Ctrl held, the wheel zooms it; otherwise
  // the wheel scrolls the page as usual. Zooming needs a non-passive listener.
  useEffect(() => {
    const el = box.current;
    if (!el || !live) return;
    const onWheel = (e: globalThis.WheelEvent) => {
      if (!armed.current && !e.ctrlKey) return;
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(e.deltaY < 0 ? 0.8 : 1.25, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [live, zoomAt]);

  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    armed.current = true;
    dragFrom.current = { x: e.clientX, y: e.clientY, view, moved: false };
    if (e.pointerType !== "touch") e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = dragFrom.current;
    if (!d || e.pointerType === "touch") return;                 // a finger scrolls the page; it zooms with a tap
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (Math.hypot(dx, dy) > 3) d.moved = true;
    if (!d.moved) return;                                        // a plain click asks for no new picture
    setView(clampView({ ...d.view, cx: d.view.cx - (dx * d.view.span) / W, cy: d.view.cy - (dy * d.view.span) / W }));
  };
  const onUp = (e: PointerEvent<HTMLDivElement>) => {
    const d = dragFrom.current;
    dragFrom.current = null;
    if (!d) return;
    if (e.pointerType === "touch" && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 8) {
      const r = e.currentTarget.getBoundingClientRect();
      zoomAt(0.5, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    }
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = 0.15;
    if (e.key === "ArrowLeft") pan(-step, 0);
    else if (e.key === "ArrowRight") pan(step, 0);
    else if (e.key === "ArrowUp") pan(0, -step);
    else if (e.key === "ArrowDown") pan(0, step);
    else if (e.key === "+" || e.key === "=") zoomAt(0.6);
    else if (e.key === "-" || e.key === "_") zoomAt(1 / 0.6);
    else if (e.key === "0") setView(HOME);
    else return;
    e.preventDefault();
  };

  const style = useMemo(() => {
    if (!shot) return undefined;
    const s = shot.view.span / view.span;
    const tx = ((shot.view.cx - view.cx) / view.span) * W, ty = ((shot.view.cy - view.cy) / view.span) * W;
    return { width: W, height: H, transform: `translate(${tx}px, ${ty}px) scale(${s})` };
  }, [shot, view, W, H]);
  const settled = shot && !shot.preview && same(shot.view, view) && shot.iters === iters && shot.w === W && shot.h === H;
  const zoom = HOME.span / view.span;
  const saved = demo?.tiles ?? [];

  return (
    <section className="stage run-stage" aria-labelledby="run-render-h">
      <div className="run-head">
        <div>
          <h2 id="run-render-h" className="stage-title" tabIndex={-1}>{live ? "Render the picture yourself" : "Pictures rendered by the tile arithmetic"}</h2>
          <p className="run-what">
            The tile job draws strips of this picture, the Mandelbrot set, with plain arithmetic: for every pixel the
            same small sum is repeated up to a few hundred times.{" "}
            {live ? `Here the same arithmetic renders any view you choose, in strips side by side, on ${HERE}.`
              : "These three views were rendered with the same arithmetic on the team laptop."}
          </p>
        </div>
        <Facts job={job} net={live ? status?.offline : undefined} />
      </div>
      {live ? (
        <div className="run-render">
          <div ref={wrap} className="run-view-wrap">
            <div
              ref={box} className="run-view" style={{ width: W, height: H }} tabIndex={0} role="application" aria-roledescription="picture viewer"
              aria-label="The rendered picture. Drag to move it. The plus and minus keys zoom, the arrow keys move it, and 0 resets it. After a click the mouse wheel zooms too."
              onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={() => { dragFrom.current = null; }} onPointerLeave={() => { armed.current = false; }}
              onDoubleClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); zoomAt(0.5, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height); }}
              onKeyDown={onKey}
            >
              {shot ? <img src={shot.url} alt="" draggable={false} style={style} data-preview={shot.preview} /> : null}
            </div>
          </div>
          <div className="run-side">
            <p className="sr-only" role="status">{settled && shot ? `Full detail, zoom ${zoom < 10 ? dec(zoom, 1) : int(zoom)} times, rendered in ${ms(shot.seconds)}` : ""}</p>
            <dl className="readout run-render-facts">
              <div><dt>Rendered in</dt><dd>{shot ? ms(shot.seconds) : "–"}</dd></div>
              <div><dt>Picture</dt><dd>{shot ? `${int(shot.w)} × ${int(shot.h)}${NB}px` : "–"}</dd></div>
              <div><dt>Steps per pixel</dt><dd>up to {shot ? int(shot.iters) : int(iters)}</dd></div>
              <div><dt>Work in the picture</dt><dd>{shot ? `${dec(shot.tiles, shot.tiles < 10 ? 1 : 0)}${NB}tiles of the job` : "–"}</dd></div>
              <div><dt>Zoom</dt><dd>{zoomText(zoom)}</dd></div>
              <div><dt>State</dt><dd>{error ? "no answer" : settled ? "full detail" : "preview, sharpening"}</dd></div>
              {shot?.joules != null ? <div><dt>Processor energy</dt><dd>{joule(shot.joules)}{shot.watts != null ? <span className="run-sub"> at {dec(shot.watts, 1)}{NB}W</span> : null}</dd></div> : null}
              {shot?.flops != null ? <div><dt>Operations</dt><dd>{ops(shot.flops)} FLOPs{shot.seconds > 0 ? <span className="run-sub"> at {gflop(shot.flops / shot.seconds)} a second</span> : null}</dd></div> : null}
            </dl>
            <div className="btn-row run-zoom">
              <button type="button" className="btn btn-primary" onClick={() => zoomAt(0.5)}>Zoom in</button>
              <button type="button" className="btn" onClick={() => zoomAt(2)}>Zoom out</button>
              <button type="button" className="btn" onClick={() => setView(HOME)} disabled={same(view, HOME)}>Reset</button>
            </div>
            <div className="run-pad" role="group" aria-label="Move the picture">
              <button type="button" className="btn" onClick={() => pan(-0.2, 0)}>Left</button>
              <button type="button" className="btn" onClick={() => pan(0, -0.2)}>Up</button>
              <button type="button" className="btn" onClick={() => pan(0, 0.2)}>Down</button>
              <button type="button" className="btn" onClick={() => pan(0.2, 0)}>Right</button>
            </div>
            <label className="field run-detail">
              Detail: up to {iters} steps per pixel
              <input type="range" min={100} max={400} step={50} value={iters} onChange={(e) => setIters(Number(e.target.value))} />
            </label>
            <div className="run-examples" role="group" aria-label="Places to look at">
              {PLACES.map((p) => <button key={p.name} type="button" className="btn" aria-pressed={same(view, p.view)} onClick={() => setView(p.view)}>{p.name}</button>)}
            </div>
            <p className="run-credit">
              With a mouse, drag the picture to move it and double-click to zoom in. After a click the wheel zooms
              too, until the pointer leaves the picture; with Ctrl held it always zooms. On a touch screen, tap the
              picture to zoom in and move it with the Left, Up, Down and Right buttons. {TILE_NOTE} Views that hold
              more of the set take longer, because its points run every step. The job was measured with eight
              processes side by side; this view uses four threads, so it draws fewer tiles a second.
            </p>
            {error ? <p className="callout" role="alert"><strong>The job did not answer.</strong> Try again in a moment.</p> : null}
          </div>
        </div>
      ) : saved.length ? (
        <div className="run-render">
          <div className="run-view-wrap">
            <img className="run-saved" src={"/" + saved[savedAt].file} alt={`${saved[savedAt].name}, rendered by the tile arithmetic`} />
          </div>
          <div className="run-side">
            <div className="run-examples" role="group" aria-label="Saved pictures">
              {saved.map((t, i) => <button key={t.name} type="button" className="btn" aria-pressed={savedAt === i} onClick={() => setSavedAt(i)}>{t.name}</button>)}
            </div>
            <dl className="readout run-render-facts">
              <div><dt>Rendered in</dt><dd>{ms(saved[savedAt].seconds)}</dd></div>
              <div><dt>Picture</dt><dd>{int(saved[savedAt].width)} × {int(saved[savedAt].height)}{NB}px</dd></div>
              <div><dt>Steps per pixel</dt><dd>up to {int(saved[savedAt].iters)}</dd></div>
              <div><dt>Work in the picture</dt><dd>{dec(saved[savedAt].tiles, 0)}{NB}tiles of the job</dd></div>
              <div><dt>Zoom</dt><dd>{zoomText(HOME.span / saved[savedAt].span)}</dd></div>
              {saved[savedAt].joules != null ? <div><dt>Processor energy</dt><dd>{joule(saved[savedAt].joules!)}</dd></div> : null}
              {saved[savedAt].flops != null ? <div><dt>Operations</dt><dd>{ops(saved[savedAt].flops!)} FLOPs</dd></div> : null}
            </dl>
            <p className="run-credit">
              {TILE_NOTE} The job was measured with eight processes side by side; these pictures were rendered on four
              threads.
            </p>
          </div>
        </div>
      ) : null}
    </section>
  );
}

// ------------------------------------------------------------------ text generation (small language model)
/** Seconds since `since`, while an answer is being written. */
function Running({ since }: { since: number }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(t);
  }, []);
  return <>{dec(Math.max(0, now - since) / 1000, 1)}{NB}s</>;
}
const wordsIn = (s: string) => (s.trim() ? s.trim().split(/\s+/).length : 0);
/** The model sometimes marks words as bold in Markdown although it is told not to; the page shows plain text. */
const plain = (s: string) => s.replace(/\*\*/g, "").replace(/[ \t]+\n/g, "\n");
const figures = (d: LlmDone) => (
  <>
    {d.first_token_s != null ? <>First token after {ms(d.first_token_s)}. </> : null}
    {d.tokens > 1 && d.tokens_per_s != null
      ? <>Then {int(d.tokens - 1)} more in {dec(d.writing_s, 1)}{NB}s: <b>{dec(d.tokens_per_s, 1)} tokens a second</b>.</>
      : <>{int(d.tokens)} {d.tokens === 1 ? "token" : "tokens"} written.</>}
    {d.hit_limit ? " The answer was cut at the length limit." : ""}
  </>
);

function LlmPanel({ live, demo }: { live: boolean; demo?: TryDemo }) {
  const status = useLlmStatus(live);
  const st = status.data?.llm;
  const older = live && !!status.data && !status.data.llm;        // a service started before this panel existed
  const [text, setText] = useState(ASKS[0].prompt);
  const [asked, setAsked] = useState("");
  const [answer, setAnswer] = useState("");
  const [phase, setPhase] = useState<"idle" | "writing" | "done" | "stopped" | "error">("idle");
  const [done, setDone] = useState<LlmDone | null>(null);
  const [error, setError] = useState("");
  const [since, setSince] = useState(0);
  const [savedAt, setSavedAt] = useState(0);
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => () => ctl.current?.abort(), []);

  // One answer at a time: a new request ends the one before it, here and on the laptop.
  const ask = useCallback((prompt: string) => {
    const p = prompt.trim();
    if (!p) return;
    ctl.current?.abort();
    const c = new AbortController();
    ctl.current = c;
    setAsked(p); setAnswer(""); setDone(null); setError(""); setPhase("writing"); setSince(performance.now());
    writeAnswer(p, c.signal, (piece) => { if (ctl.current === c) setAnswer((a) => a + piece); }).then(
      (d) => { if (ctl.current === c) { setDone(d); setPhase(d.stopped ? "stopped" : "done"); } },
      (e: Error) => { if (ctl.current === c) { setError(e.message); setPhase("error"); } },
    );
  }, []);
  const box = useRef<HTMLTextAreaElement | null>(null);
  // Stop switches itself off, so the focus goes back to the request (a disabled button would drop it)
  const stop = () => { ctl.current?.abort(); ctl.current = null; setPhase("stopped"); box.current?.focus(); };
  const saved = demo?.llm?.answers ?? [];
  const facts = live ? st : demo?.llm;
  const model = facts?.model ?? "a small language model";

  return (
    <section className="stage run-stage" aria-labelledby="run-llm-h">
      <div className="run-head">
        <div>
          <h2 id="run-llm-h" className="stage-title" tabIndex={-1}>{live ? `Ask a small language model on ${HERE}` : "Answers written by a small language model"}</h2>
          <p className="run-what">
            Writing text with a language model is computing work too, and it turns into heat like the other jobs.
            This panel runs a small open one{facts?.parameters ? `, ${model} with ${facts.parameters} parameters,` : ""} on{" "}
            {live ? HERE : "the team laptop"}: {live ? "you write a request and the answer appears while it is being written." : "these answers were written there and saved."}
          </p>
        </div>
        <Facts net={live ? status.data?.offline : undefined}>
          {live ? <div><dt>Language model</dt><dd>{older ? "not in the running local service: restart the service" : st?.error ? `could not be loaded: ${st.error}` : st?.loaded ? `ready on ${HERE}` : `loading on ${HERE}`}</dd></div> : null}
          {live && st?.measured_tokens_per_s ? <div><dt>Speed when it loaded</dt><dd>{dec(st.measured_tokens_per_s, 1)} tokens a second</dd></div> : null}
          <div><dt>In the heat schedule</dt><dd>not yet: no batch job, no energy measurement</dd></div>
        </Facts>
      </div>
      {live ? (
        <>
          <form className="run-llm-form" onSubmit={(e) => { e.preventDefault(); ask(text); }}>
            <label className="field run-ask run-ask-llm">
              Your request, in English
              <textarea ref={box} className="input" rows={3} maxLength={2000} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); ask(text); } }} />
            </label>
            <div className="btn-row">
              <button type="submit" className="btn btn-primary" disabled={!text.trim()}>Write the answer</button>
              <button type="button" className="btn" onClick={stop} disabled={phase !== "writing"}>Stop</button>
            </div>
          </form>
          <div className="run-examples run-examples-even" role="group" aria-label="Example requests">
            {ASKS.map((x) => <button key={x.label} type="button" className="btn" aria-pressed={asked === x.prompt} onClick={() => { setText(x.prompt); ask(x.prompt); }}>{x.label}</button>)}
          </div>
          <div className="run-answer" aria-busy={phase === "writing"}>
            {asked ? <p className="run-answer-q">{asked}</p> : null}
            {phase === "idle" ? <p className="run-answer-hint">The answer appears here while {HERE} writes it. Press Enter or pick an example.</p>
              : phase === "error" ? <p className="run-answer-hint" role="alert">{older ? "The running local service does not have the language model yet. Stop it and start it again, then ask once more." : `The model did not answer: ${error}.`}</p>
              : <p className="run-answer-a">{plain(answer)}{phase === "writing" ? <span className="run-caret" aria-hidden="true" /> : null}</p>}
          </div>
          <p className="run-readout">
            {phase === "writing" ? (answer ? <>Writing: {int(wordsIn(answer))} words so far, <Running since={since} />.</>
                : st?.loaded ? <>Reading the request, <Running since={since} />.</> : `Loading the language model on ${HERE} (once), then writing…`)
              : done ? <>{phase === "stopped" ? "Stopped. " : ""}{figures(done)} <Effort w={done.work} /></>
              : phase === "stopped" ? `Stopped after ${int(wordsIn(answer))} words.`
              : "Enter sends the request; Shift and Enter start a new line."}
          </p>
          <p className="sr-only" role="status">{phase === "done" || (phase === "stopped" && answer) ? `Answer ${phase === "done" ? "finished" : "stopped"}: ${plain(answer)}` : ""}</p>
          <p className="run-credit">
            {facts?.model ? <>The model is {facts.model}, an open model under the {facts.licence} licence, in one fixed version, on {facts.threads} processor threads; an answer is at most {facts.max_tokens} tokens long. </> : null}
            A token is a word or a piece of a word. The same request gets the same answer. A model this small often
            writes things that sound right and are wrong.
          </p>
        </>
      ) : saved.length ? (
        <>
          <div className="run-examples run-examples-even" role="group" aria-label="Saved requests">
            {saved.map((a, i) => <button key={a.label} type="button" className="btn" aria-pressed={savedAt === i} onClick={() => setSavedAt(i)}>{a.label}</button>)}
          </div>
          <div className="run-answer">
            <p className="run-answer-q">{saved[savedAt].prompt}</p>
            <p className="run-answer-a">{plain(saved[savedAt].text)}</p>
          </div>
          <p className="run-readout">Written on the team laptop. {figures(saved[savedAt])} <Effort w={saved[savedAt].work} /></p>
          <p className="run-credit">
            {facts?.model ? <>The model is {facts.model}, an open model under the {facts.licence} licence, in one fixed version, on {facts.threads} processor threads. </> : null}
            A token is a word or a piece of a word. A model this small often writes things that sound right and are wrong.
          </p>
        </>
      ) : null}
    </section>
  );
}

// ------------------------------------------------------------------ photo descriptions (panel in Describe.tsx)
function CaptionFacts({ job, live }: { job?: Workload; live: boolean }) {
  const { data: st } = useCaptionStatus();
  return (
    <Facts job={job} net={live ? st?.offline : undefined}>
      {live ? <div><dt>Description model</dt><dd>{st?.error ? `could not be loaded: ${st.error}` : st?.loaded ? `ready on ${HERE}` : `loading on ${HERE}`}</dd></div> : null}
    </Facts>
  );
}

// ------------------------------------------------------------------ the processor's power, live
/** The laptop's own energy meter: the power of the processor now and over the last minute. Shown only when the
 *  running service can read the meter. */
function MeterStrip() {
  const d = useMeter(true).data;
  if (!d?.available || d.watts == null || d.series.length < 2) return null;
  const W = 260, H = 48, top = Math.max(12, ...d.series) * 1.08, n = d.series.length;
  const pts = d.series.map((v, i) => `${(W - (n - 1 - i) * (W / 119)).toFixed(1)},${(H - 3 - (v / top) * (H - 6)).toFixed(1)}`).join(" ");
  return (
    <div className="run-meter" role="group" aria-label="Processor power of the laptop">
      <div className="run-meter-now">
        <span>Processor power now</span>
        <b>{dec(d.watts, 1)}{NB}W</b>
      </div>
      <svg className="run-meter-line" viewBox={`0 0 ${W} ${H}`} width={W} height={H} aria-hidden="true">
        <line x1="0" y1={H - 3} x2={W} y2={H - 3} className="run-meter-base" />
        <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      <p className="run-meter-note">
        Read from the laptop's own energy meter every half second; the line is the last minute
        {d.low != null && d.high != null ? <>, between {dec(d.low, 1)} and {dec(d.high, 1)}{NB}W</> : null}. It covers the
        processor only, with every program that is running.
        {d.machine?.cores ? <> This computer's processor has {d.machine.cores} cores{d.machine.logical && d.machine.logical !== d.machine.cores ? <> ({d.machine.logical} threads)</> : null}; the jobs compute on {d.machine.threads} threads.</> : null}{" "}
        Under each answer: the energy the processor took while it was computed, and the arithmetic operations it
        needed, in FLOPs (one addition or multiplication of two decimal numbers each; a GFLOP is a billion of them).
        The FLOPs are calculated, not read from the hardware. All of the energy ends up as heat.
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ the page
export function RunView() {
  const search = useSearch({ strict: false }) as { job?: RunJob };
  const job: RunJob = isRunJob(search.job) ? search.job : "embed";
  const navigate = useNavigate();
  const health = useApiHealth();
  const live = health.isSuccess;
  const offline = health.isError;
  const status = useJobsStatus(live && job !== "caption" && job !== "llm");
  const s = useSummary();
  const demo = useTryDemo(offline);
  const w = (key: string) => s.data?.workloads.jobs.find((j) => j.key === key);
  // after a switch the new panel's heading takes the focus, so a screen reader says what appeared
  const before = useRef(job);
  useEffect(() => {
    if (before.current === job) return;
    before.current = job;
    document.getElementById(`run-${job}-h`)?.focus();
  }, [job]);

  return (
    <>
      <PageHead title={offline ? "The jobs, with saved answers" : `Run the jobs on ${HERE}`}>
        Four job types make the heat. {offline
          ? "On the team laptop each one answers on the spot; this copy shows answers saved from the same models and arithmetic."
          : `Here each one answers on the spot, computed on ${HERE} with the job's own model or arithmetic while outside network access is blocked.`}{" "}
        In the heat schedule the same four job types run in batches that the scheduler pauses and resumes; the answers
        here are single requests outside that schedule. A fifth panel, text generation, shows a further kind of work
        that is not in the schedule yet.
      </PageHead>

      <div className="seg run-jobs" role="group" aria-label="Job to run">
        {RUN_JOBS.map((j) => (
          <button key={j.key} type="button" aria-pressed={job === j.key} onClick={() => void navigate({ to: "/run", search: { job: j.key } })}>{j.label}</button>
        ))}
      </div>
      {live ? <MeterStrip /> : null}

      {health.isLoading ? <Loading what="the local service" /> : null}
      {offline ? <Offline what={job === "caption" ? "no new photo can be described here; the sentences below are from its recorded run." : "the answers below are saved ones."} /> : null}
      {offline && demo.isError ? <LoadError what="tryjobs_demo.json" error={demo.error} /> : null}

      {job === "embed" ? <EmbedPanel live={live} status={status.data} job={w("embed")} demo={demo.data} /> : null}
      {job === "image" ? <PhotoPanel live={live} status={status.data} job={w("image")} demo={demo.data} /> : null}
      {job === "render" ? <RenderPanel live={live} status={status.data} job={w("render")} demo={demo.data} /> : null}
      {job === "caption" ? <DescribePanel facts={<CaptionFacts job={w("caption")} live={live} />} /> : null}
      {job === "llm" && !health.isLoading ? <LlmPanel live={live} demo={demo.data} /> : null}
    </>
  );
}
