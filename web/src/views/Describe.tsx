import { useCallback, useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "../components/DataTable";
import {
  describePhoto, HERE, licenceName, useApiHealth, useCaptionStatus, useDescribeDemo, type CaptionResult, type SavedDescription,
} from "../data";
import { int } from "../format";
import { Effort } from "../components/ui";
import "./describe.css";

type Item = {
  id: number; name: string; url: string; file: File;
  state: "queued" | "working" | "done" | "error"; started?: number; result?: CaptionResult; error?: string;
};

const NB = " ";
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"];

const SAVED_COLS: ColumnDef<SavedDescription, any>[] = [
  { header: "What the job wrote", accessorKey: "description", enableSorting: false, cell: (c) => <strong className="ds-saved">{c.getValue<string>()}</strong> },
  { header: "The archive's title", accessorKey: "title", enableSorting: false, meta: { className: "prose" },
    cell: (c) => <>{c.getValue<string>()}<span className="ds-credit">{[c.row.original.date, c.row.original.creator, c.row.original.institution, licenceName(c.row.original.licence)].filter(Boolean).join(", ")}</span></> },
  { header: "Photo", accessorKey: "object_url", enableSorting: false,
    cell: (c) => <a href={c.getValue<string>()} target="_blank" rel="noopener noreferrer">View on Memobase</a> },
];

/** Seconds since `since`, updated ten times a second while the description is being computed. */
function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const t = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(t);
  }, []);
  return <span className="num">{((now - since) / 1000).toFixed(1)}{NB}s</span>;
}

/** The photo-description job as one panel of the Run jobs page. `facts` is the list at the top right (model
 *  state, measured rate, network), built by the page like the lists of the other three panels. */
export function DescribePanel({ facts }: { facts?: ReactNode }) {
  const [items, setItems] = useState<Item[]>([]);
  const [drag, setDrag] = useState(false);
  const [refused, setRefused] = useState(0);
  const [allSaved, setAllSaved] = useState(false);
  const [said, setSaid] = useState("");
  const input = useRef<HTMLInputElement | null>(null);
  const nextId = useRef(1);
  const urls = useRef<string[]>([]);
  const { data: st } = useCaptionStatus();
  const offline = useApiHealth().isError;
  const saved = useDescribeDemo(offline);
  const off = st?.offline;
  const flagsOk = off ? Object.values(off.library_offline_flags).every((v) => v !== false) : false;
  const cannot = st?.error ?? null;

  useEffect(() => () => urls.current.forEach((u) => URL.revokeObjectURL(u)), []);

  const add = useCallback((files: FileList | File[]) => {
    const all = Array.from(files);
    const list = all.filter((f) => IMAGE_TYPES.includes(f.type));
    setRefused(all.length - list.length);
    if (!list.length) return;
    const fresh = list.map((f) => {
      const url = URL.createObjectURL(f);
      urls.current.push(url);
      return { id: nextId.current++, name: f.name, url, file: f, state: "queued" as const };
    });
    setItems((cur) => [...fresh.reverse(), ...cur]);
  }, []);

  // One photo at a time, oldest first, once the model is loaded. If the model cannot be loaded, the waiting
  // photos are told so instead of waiting for good.
  useEffect(() => {
    if (cannot) {
      if (items.some((i) => i.state === "queued")) {
        setItems((cur) => cur.map((i) => (i.state === "queued" ? { ...i, state: "error", error: cannot } : i)));
        setSaid(`The photos could not be described: ${cannot}`);
      }
      return;
    }
    if (!st?.loaded || items.some((i) => i.state === "working")) return;
    const next = [...items].reverse().find((i) => i.state === "queued");
    if (!next) return;
    const started = performance.now();
    setItems((cur) => cur.map((i) => (i.id === next.id ? { ...i, state: "working", started } : i)));
    describePhoto(next.file).then(
      (result) => {
        setItems((cur) => cur.map((i) => (i.id === next.id ? { ...i, state: "done", result } : i)));
        setSaid(`Photo ${next.id} described: ${result.description}`);
      },
      (e: Error) => {
        setItems((cur) => cur.map((i) => (i.id === next.id ? { ...i, state: "error", error: e.message } : i)));
        setSaid(`Photo ${next.id} could not be described: ${e.message}`);
      },
    );
  }, [items, st?.loaded, cannot]);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    add(e.dataTransfer.files);
  };

  return (
    <section className="stage run-stage ds-stage" aria-labelledby="run-caption-h">
      <div className="run-head">
        <div>
          <h2 id="run-caption-h" className="stage-title" tabIndex={-1}>{offline ? "A job that describes photos" : `Describe a photo on ${HERE}`}</h2>
          <p className="run-what">
            {offline
              ? "The photo-description job writes one English sentence about a photo. It runs Florence-2-base, an open image-to-text model stored on the team laptop; outside network access is blocked while the model runs."
              : `The photo-description job writes one English sentence about any photo you give it. It runs Florence-2-base, an open image-to-text model stored on ${HERE}. The photo goes only to the local service on ${HERE}; outside network access is blocked while the model runs.`}
          </p>
        </div>
        {facts}
      </div>

      {offline && saved.data?.length ? (
        <div className="ds-saved-block">
          <h3>What the job wrote in its recorded run</h3>
          <p className="run-readout">
            {saved.data.length} archive photos, one sentence each, written on the team laptop with outside network
            access blocked. The archive's own title stands beside each sentence; the photos themselves are on Memobase.
          </p>
          <DataTable data={allSaved ? saved.data : saved.data.slice(0, 8)} columns={SAVED_COLS} caption="Sentences written by the photo-description job, beside the archive titles" />
          {saved.data.length > 8 ? (
            <button type="button" className="view-toggle" aria-expanded={allSaved} onClick={() => setAllSaved((v) => !v)}>
              {allSaved ? "Show the first 8 only" : `Show all ${saved.data.length} sentences`}
            </button>
          ) : null}
        </div>
      ) : null}

      <div
        className="ds-drop" hidden={offline} data-drag={drag ? "true" : "false"}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <button type="button" className="btn btn-primary ds-choose" onClick={() => input.current?.click()}>
          Choose a photo
        </button>
        <span className="ds-drop-hint">You can also drop image files here. JPEG, PNG, WebP, GIF, BMP or TIFF, up to 15 MB each.</span>
        <input
          ref={input} type="file" accept={IMAGE_TYPES.join(",")} multiple hidden aria-label="Photo to describe"
          onChange={(e) => { if (e.target.files) add(e.target.files); e.target.value = ""; }}
        />
      </div>
      {refused ? (
        <p className="callout" role="alert">
          <strong>{refused === 1 ? "One file cannot be used." : `${refused} files cannot be used.`}</strong> Choose photos as JPEG, PNG, WebP, GIF, BMP or TIFF.
        </p>
      ) : null}
      <p className="sr-only" role="status">{said}</p>

      <ol className="ds-list">
        {items.map((it) => (
          <li key={it.id} className="ds-item" data-state={it.state}>
            <img src={it.url} alt={it.result ? it.result.description : `Photo ${it.id}`} className="ds-img" />
            <div className="ds-text">
              <div className="ds-label">Photo {it.id}</div>
              {it.state === "done" && it.result ? (
                <>
                  <p className="ds-desc">{it.result.description}</p>
                  <p className="ds-meta">
                    Described on {HERE} in <span className="num">{it.result.seconds.toFixed(1)}{NB}s</span>, from an image
                    of {int(it.result.width)}{NB}×{NB}{int(it.result.height)}{NB}px;{" "}
                    {(it.result.blocked_during ?? 0) === 0
                      ? "no outside connection was attempted while it was described"
                      : it.result.blocked_during === 1 ? "1 outside connection attempt was blocked while it was described"
                      : `${it.result.blocked_during} outside connection attempts were blocked while it was described`}.
                    {it.result.work ? <> <Effort w={it.result.work} /></> : null}
                  </p>
                  {it.result.archive ? (
                    <p className="ds-archive">
                      <span className="ds-archive-k">Archive title</span> {it.result.archive.title}
                      <span className="ds-credit">
                        {[it.result.archive.date, it.result.archive.creator, it.result.archive.institution].filter(Boolean).join(", ")},{" "}
                        {licenceName(it.result.archive.licence)},{" "}
                        <a href={it.result.archive.object_url} target="_blank" rel="noopener noreferrer">view on Memobase</a>
                      </span>
                    </p>
                  ) : null}
                </>
              ) : it.state === "error" ? (
                <p className="ds-error">Could not describe this file: {it.error}</p>
              ) : it.state === "working" && it.started ? (
                <>
                  <p className="ds-desc ds-pending">Describing on {HERE}… <Elapsed since={it.started} /></p>
                  <span className="ds-progress" aria-hidden="true"><span /></span>
                </>
              ) : (
                <p className="ds-desc ds-pending">{st?.loaded ? "Queued: waiting for the photos below to finish" : "Waiting for the model to load"}</p>
              )}
            </div>
          </li>
        ))}
      </ol>
      {!items.length && !offline ? <p className="run-readout ds-empty">Each photo you choose appears here with one sentence that describes it and the time it took, newest first.</p> : null}
      {!offline ? (
        <p className="run-credit">
          {off ? <>The model is Florence-2-base in one fixed version, loaded with the offline mode of its libraries {flagsOk ? "on" : "off"}, on {off.torch_threads} processor threads. </> : null}
          A general model writes the sentence; it can be wrong.
        </p>
      ) : null}
    </section>
  );
}
