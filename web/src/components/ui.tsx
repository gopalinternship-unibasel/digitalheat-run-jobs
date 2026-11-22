import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Work } from "../data";
import { dec, gflop, joule, ops } from "../format";

/** Width of an element, kept current with a ResizeObserver (charts draw at real pixel size). */
export function useWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function PageHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {children ? <p className="lede">{children}</p> : null}
    </header>
  );
}

/** A page section: its heading on the ground, its content on one surface. `tone` picks the surface: a raised
 *  work frame (default), the dark instrument stage for the picture that leads a page, or none ("bare"). */
export function Section({ title, intro, children, id, tone = "frame" }: { title: string; intro?: ReactNode; children: ReactNode; id?: string; tone?: "frame" | "stage" | "bare" }) {
  return (
    <section className="section" id={id} aria-labelledby={id ? `${id}-h` : undefined}>
      <div className="section-head">
        <h2 id={id ? `${id}-h` : undefined}>{title}</h2>
        {intro ? <p>{intro}</p> : null}
      </div>
      <div className={tone === "bare" ? "section-body" : `section-body ${tone}`}>{children}</div>
    </section>
  );
}

/** One key figure. `before` is a unit set in front of the value (a currency), `unit` one set after it. */
export function Figure({ label, value, unit, before, detail, children, className }: { label: string; value: string; unit?: string; before?: string; detail?: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <div className={className ? `figure ${className}` : "figure"} role="listitem">
      <div className="figure-label">{label}</div>
      <div className="figure-value">
        {before ? <span className="figure-unit figure-before">{before}</span> : null}
        {value}
        {unit ? <span className="figure-unit">{unit}</span> : null}
      </div>
      {detail ? <div className="figure-detail">{detail}</div> : null}
      {children ? <div className="figure-visual">{children}</div> : null}
    </div>
  );
}

// A filled square means settled (given, sourced, measured, yes); an outlined square means the team's own
// estimate or something still open. No colour: colour is reserved for heat.
const KIND: Record<string, string> = {
  case: "solid", sourced: "solid", measured: "solid", computed: "solid", yes: "solid",
  assumption: "open", preliminary: "open", paused: "open", no: "bad",
};
const cap = (s: string) => (s === "case" ? "Given" : s.charAt(0).toUpperCase() + s.slice(1));
export function Badge({ status, children }: { status: string; children?: ReactNode }) {
  return <span className="badge" data-kind={KIND[status] ?? "solid"} data-status={status}>{children ?? cap(status)}</span>;
}

/** A caveat or a side remark. */
export function Note({ children }: { children: ReactNode }) {
  return <p className="note">{children}</p>;
}
/** The sentence a chart or table is there to support. */
export function Takeaway({ children }: { children: ReactNode }) {
  return <p className="takeaway">{children}</p>;
}

export type LegendItem = { label: string; color: string; shape?: "box" | "line" | "dash" };
export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="legend">
      {items.map((it) => (
        <li key={it.label}>
          {it.shape === "dash"
            ? <span className="ln-dash" style={{ borderTopColor: it.color }} aria-hidden="true" />
            : <span className={it.shape === "line" ? "ln" : "sw"} style={{ background: it.color }} aria-hidden="true" />}
          {it.label}
        </li>
      ))}
    </ul>
  );
}

/** A chart with a table-view twin (every value reachable without hovering). */
export function ChartOrTable({ chart, table, label }: { chart: ReactNode; table: ReactNode; label: string }) {
  const [asTable, setAsTable] = useState(false);
  return (
    <div>
      {asTable ? table : chart}
      <button type="button" className="view-toggle" onClick={() => setAsTable((v) => !v)}>
        {asTable ? `Show ${label} as a chart` : `Show ${label} as a table`}
      </button>
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return <p className="small">Loading {what}…</p>;
}
export function LoadError({ what, error }: { what: string; error: unknown }) {
  if (error) console.warn(`Could not load ${what}:`, error);
  return (
    <p className="callout" role="alert">
      <strong>This part could not be loaded.</strong> Reload the page. If it stays empty, this copy of the dashboard is
      missing its data files ({what}).
    </p>
  );
}

/** What one answer cost on the laptop, as a sentence: the processor's energy while it was computed (from the
 *  laptop's own meter) and the arithmetic operations it needed (calculated). Nothing when neither is known. */
export function Effort({ w }: { w?: Work | null }) {
  if (!w || (w.joules == null && w.flops == null)) return null;
  return (
    <>
      {w.joules != null ? <>Processor energy: <b>{joule(w.joules)}</b>{w.watts != null ? <> ({dec(w.watts, 1)}{"\u00a0"}W on average)</> : null}. </> : null}
      {w.flops != null ? <>Operations: about <b>{ops(w.flops)} FLOPs</b>{w.seconds > 0 ? <>, {gflop(w.flops / w.seconds)} a second</> : null}.</> : null}
    </>
  );
}
