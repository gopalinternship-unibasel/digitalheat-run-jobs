// Data layer. Every number comes from files written by `python -m dh.export` (web/public/data) or from
// the local Python API (`python -m dh.serve`, proxied under /api). Nothing is computed here except
// formatting and simple selections.
import { keepPreviousData, useQuery } from "@tanstack/react-query";

export type Agg = { mean: number; min: number; max: number };
export type KeyName =
  | "coverage_pct" | "dump_kwh" | "utilisation_pct" | "missed_pct" | "breakeven" | "backup_kwh" | "used_kwh"
  | "server_hours" | "electricity_kwh" | "electricity_cost" | "heat_value" | "income" | "income_t1"
  | "income_t2" | "margin_before_hardware" | "t1_hours" | "filler_hours" | "sold_hours" | "jobs_offered"
  | "jobs_missed" | "work_cancelled_pct" | "dump_pct_of_recovered" | "unrecovered_kwh" | "tank_loss_kwh"
  | "direct_kwh" | "from_tank_kwh" | "to_tank_kwh";

export interface SweepRow {
  n: number; coverage_pct: Agg; dump_kwh: Agg; utilisation_pct: Agg; missed_pct: Agg; work_cancelled_pct: Agg;
  breakeven: Agg; backup_kwh: Agg; margin_before_hardware: Agg;
}
export interface SupplyCase {
  key: string; label: string; coverage_pct: Agg; utilisation_pct: Agg; dump_kwh: Agg; missed_pct: Agg; breakeven: Agg;
  sold_hours: Agg; income: Agg; margin_before_hardware: Agg; backup_kwh: Agg;
}
export interface JobRate {
  per_day: number; reference: boolean; jobs_offered: Agg; t1_hours: Agg; missed_pct: Agg; work_cancelled_pct: Agg;
  coverage_pct: Agg; income_t1: Agg; margin_before_hardware: Agg;
}
export interface LadderRow {
  key: string; label: string; coverage_pct: Agg; dump_kwh: Agg; utilisation_pct: Agg; missed_pct: Agg;
  breakeven: Agg; work_cancelled_pct: Agg;
}
export interface MonthRow {
  month: string; days: number; space_kwh: number; dhw_kwh: number; demand_kwh: number; direct_kwh: number;
  from_tank_kwh: number; backup_kwh: number; dump_kwh: number; coverage_pct: number; server_hours: number;
  t1_hours: number; filler_hours: number; utilisation_pct: number; need_sh_per_day: number;
  run_sh_per_day: number; jobs_offered: number; jobs_missed: number;
}
export interface HardwareOption {
  key: string; label: string; machines_per_kw: number; capex_per_machine_chf: number; capex_per_slot_chf: number;
  life_years: number; crf: number; annual_per_slot_chf: number; annual_per_house_chf: number;
  hardware_per_sold_hour: number; breakeven_case: number; breakeven_full: number; price_t1: number; price_t2: number;
  price_mean: number; income_chf: number; margin_before_hardware_chf: number; margin_after_hardware_chf: number; note: string;
}
export interface Economics {
  price_el: number; cop: number; recovery: number; heat_price: number; price_t1: number; price_t2: number;
  electricity_cost: number; heat_value: number; income_t1: number; income_t2: number; income: number;
  margin_before_hardware: number; electricity_kwh: number; used_kwh: number; sold_hours: number;
  run_hours: number; t1_hours: number; filler_hours: number; breakeven: number; breakeven_floor: number;
  breakeven_floor_cop34: number; cop_field: number; breakeven_incl_vat: number; price_el_incl_vat: number;
  breakeven_2025_price: number; price_el_2025: number; filler_share_of_hours_pct: number;
  filler_share_of_income_pct: number; filler_price_floor: number; refurb_work_factor: number; capital_rate: number;
  vat_extra_cost_chf: number; threads_per_kw_new: number;
  refurb_per_thread: { price_t2: number; margin_before_hardware_chf: number; margin_after_hardware_chf: number };
  heat_credit_max_per_hour: number; hardware: HardwareOption[];
  grid: { price_el: number[]; price_compute: number[]; margin: number[][];
          breakeven_line: { price_el: number; breakeven: number }[] };
  marginal_server: { k: number; extra_hours: number; total_hours: number }[];
  hardware_by_n: HardwareByN[];
  formula: Record<string, string>;
}
export interface HardwareByN {
  n: number; sold_hours: number; coverage_pct: number; breakeven_case: number;
  breakeven_full_new: number; result_new_chf: number; breakeven_full_refurb: number; result_refurb_chf: number;
  capex_max_per_kw_chf: number; price_mean_new: number; price_mean_refurb: number;
}
export interface Sensitivity {
  key: string; label: string; note: string; change: string; coverage_pct: number; dump_kwh: number;
  utilisation_pct: number; missed_pct: number; breakeven: number;
}
export interface Workload {
  key: string; label: string; what: string; unit: string; runs: number; units_timed: number; per_s: number;
  j_per_unit: number | null; j_per_unit_min: number | null; j_per_unit_max: number | null; watts: number | null;
  other_load_min: number | null; other_load_max: number | null; quiet: boolean; quality: string;
  gflop_per_unit: number | null; eta_pct: number | null; eta_pct_min: number | null; eta_pct_max: number | null;
  scaling: "operations" | "parity" | "none"; per_kwh_parity: number | null; per_kwh_server: number | null;
  per_kwh_server_half: number | null; per_server_machine_hour: number | null;
}
export interface Workloads {
  clean_file: boolean; meter: string; state: string;
  matmul: { gflop_per_s: number; watts: number; gflop_per_joule: number };
  server: { label: string; cores: number; clock_ghz: number; flops_per_cycle: number; peak_tflop_per_s: number;
            wall_w: number; gflop_per_joule: number };
  eta_transfer: number; jobs: Workload[];
}
export interface LiveHour {
  hour: number; n_on: number; real_job: "run" | "pause"; reason: string; demand_kwh: number; soc_kwh: number;
  dump_kwh: number; backup_kwh: number; chunks_in_tick: number; worker_cpu_s: number;
}
export interface LiveRun {
  kind: string; unit: string; chunk_units: number; torch_threads: number; date: string;
  n_servers: number; policy: string; seed: number; tick_s: number; chunks: number;
  start_of_day: { soc_kwh: number; contracted_jobs_queued: number };
  hours_running: number; hours_paused: number; switches: { hour: number; to: string; ack_s: number | null }[];
  ack_s_mean: number | null; ack_s_max: number | null; worker_cpu_s_per_paused_tick_max: number | null;
  chunks_done_during_day: number; reference_wall_s: number | null; digest_reference: string;
  digest_live: string; outputs_identical: boolean; hours: LiveHour[]; file: string;
  pauses: number; units_during_day: number; had_own_reference: boolean;
  pause_ack_s_mean: number | null; pause_ack_s_max: number | null; resume_ack_s_mean: number | null; resume_ack_s_max: number | null;
}
export interface HeatFlows {
  demand: number; direct: number; from_tank: number; to_tank: number; backup: number; dump: number;
  electricity: number; recovered: number; unrecovered: number; tank_loss: number;
}
export interface Summary {
  command: string;
  provenance: { weather_csv_sha256: string; assumptions_sha256: string; seeds: number[];
                conservation_residuals: Record<string, number> };
  reference: { n: number; policy: string; forecast: string; risk_z: number; jobs_per_day: number;
               window_h: number; filler: string; price_el: number; server_kw: number; recovery: number;
               tank_kwh: number; server_starts_per_year: number; deadline_may_dump: boolean };
  house: { hours: number; heating_days: number; coldest_c: number; coldest_at: string; space_kwh: number;
           dhw_kwh: number; demand_kwh: number; peak_kwh: number; peak_at: string; need_kwh_day_mean: number;
           need_kwh_day_min: number; need_kwh_day_max: number };
  key: Record<KeyName, Agg>;
  sentence: string;
  sweep: SweepRow[];
  ladder: LadderRow[];
  monthly: MonthRow[];
  supply: SupplyCase[];
  job_rates: JobRate[];
  economics: Economics;
  sensitivities: Sensitivity[];
  workloads: Workloads;
  live: LiveRun[];
  heat_flows: HeatFlows;
}
export interface Ledger {
  n: number; seed: number; policy: string; hours: number; start: string; clock: string;
  temp: number[]; demand: number[]; direct: number[]; from_tank: number[]; backup: number[]; dump: number[];
  soc: number[]; n_on: number[]; run: number[]; to_tank: number[]; made: number[];
}
export interface Assumption {
  id: string; group: string; label: string; value: unknown; unit: string; status: string; source: string;
  url?: string; date?: string; note?: string;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export const useSummary = () =>
  useQuery({ queryKey: ["summary"], queryFn: () => getJson<Summary>("/data/summary.json"), staleTime: Infinity });
export const useLedger = () =>
  useQuery({ queryKey: ["ledger"], queryFn: () => getJson<Ledger>("/data/ledger_n8.json"), staleTime: Infinity });
export const useAssumptions = () =>
  useQuery({ queryKey: ["assumptions"], queryFn: () => getJson<Assumption[]>("/data/assumptions.json"), staleTime: Infinity });

// ---- local API (python -m dh.serve) ---------------------------------------------------------------
export interface LivePhoto { signature: string; title: string; date: string; institution: string; licence: string; object_url: string; creator: string }
export interface LiveStatus {
  state: "idle" | "starting" | "reference" | "replaying" | "finishing" | "done" | "failed";
  date?: string; tick_s?: number; hour?: number; hours?: LiveHour[]; n_servers?: number;
  worker?: { pid: number | null; state: string; chunks_done: number; chunks: number; cpu_s_last_tick: number;
             last_chunk: number | null; photos: LivePhoto[] };
  flows?: { direct: number; from_tank: number; to_tank: number; backup: number; dump: number; el: number;
            demand: number; soc: number };
  result?: { outputs_identical: boolean; digest_live: string; digest_reference: string; switches: number;
             ack_s_mean: number | null; ack_s_max: number | null; file?: string };
  message?: string; started_at?: string; kind?: string;
}
export const useApiHealth = () =>
  useQuery({ queryKey: ["api-health"], queryFn: () => getJson<{ ok: boolean }>("/api/health"), retry: false,
             refetchInterval: 5000, staleTime: 0 });
export const useLiveStatus = (enabled: boolean) =>
  useQuery({ queryKey: ["live-status"], queryFn: () => getJson<LiveStatus>("/api/live/status"), enabled,
             refetchInterval: 400, retry: false, staleTime: 0 });

// ---- photo descriptions (POST /api/caption, the caption job's own function, computed offline) -------
export interface CaptionEvidence {
  model: string; revision: string; torch_threads: number; guard_installed: boolean;
  library_offline_flags: { huggingface_hub: boolean | null; transformers: boolean | null }; blocked_attempts: number;
}
export interface CaptionStatus { loaded: boolean; loading: boolean; error: string | null; load_s: number | null; offline?: CaptionEvidence }
export interface CaptionResult {
  description: string; seconds: number; width: number; height: number; offline: CaptionEvidence; blocked_during?: number;
  work?: Work;
  archive?: { title: string; date: string; institution: string; licence: string; object_url: string; creator: string };
}
export const useCaptionStatus = () =>
  useQuery({
    queryKey: ["caption-status"], queryFn: () => getJson<CaptionStatus>("/api/caption/status?warm=1"), retry: false,
    staleTime: 0, refetchInterval: (q) => (q.state.data?.loaded || q.state.data?.error ? 15000 : 1000),
  });
export async function describePhoto(file: File): Promise<CaptionResult> {
  const res = await fetch(`/api/caption?name=${encodeURIComponent(file.name)}`, {
    method: "POST", headers: { "Content-Type": file.type || "application/octet-stream" }, body: file,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
  return body as CaptionResult;
}

export interface SearchHit {
  score: number; signature: string; record_id: string; title: string; date: string; creator: string;
  institution: string; licence: string; licence_url: string; object_url: string; extra_conditions: string;
  thumb: string; credit: string;
}
export const useSearch = (q: string, enabled: boolean) =>
  useQuery({ queryKey: ["search", q], enabled: enabled && q.trim().length > 1, retry: false, staleTime: Infinity,
             queryFn: () => getJson<{ query: string; results: SearchHit[] }>(`/api/search?k=12&q=${encodeURIComponent(q)}`) });
export interface SavedDescription {
  creator?: string;
  signature: string; description: string; title: string; date: string; institution: string; licence: string;
  licence_url: string; object_url: string;
}
export const useDescribeDemo = (enabled: boolean) =>
  useQuery({ queryKey: ["describe-demo"], enabled, staleTime: Infinity,
             queryFn: () => getJson<SavedDescription[]>("/data/describe_demo.json") });
// ---- Run jobs page: the three other job types, live (dh/tryjobs.py) -------------------------------
/** What one answer cost on the laptop: time, processor energy from its own meter, arithmetic operations (calculated). */
export interface Work { seconds: number; joules: number | null; watts: number | null; flops: number | null }
/** The processor's power in the last minute, one value every `step_s` seconds (dh/meter.py). */
export interface Meter {
  available: boolean; error: string | null; watts: number | null; low?: number; high?: number; series: number[]; step_s: number;
  machine?: { cores: number | null; logical: number | null; threads: number };
}
export const useMeter = (enabled: boolean) =>
  useQuery({ queryKey: ["meter"], enabled, retry: false, staleTime: 0, queryFn: () => getJson<Meter>("/api/meter"),
             refetchInterval: (q) => (q.state.error ? false : 1000) });
/** Where the live jobs run, as the reader of this page would say it: the service is local to the laptop. */
export const HERE = typeof location !== "undefined" && ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)
  ? "this computer" : "the team laptop";
const LICENCE_NAMES: Record<string, string> = { "PDM 1.0": "public domain (PDM 1.0)" };
/** A licence as the page names it; the Public Domain Mark is a mark, not a licence. */
export const licenceName = (l: string) => LICENCE_NAMES[l] ?? l;
type ModelState = { loaded: boolean; loading: boolean; error: string | null; load_s: number | null };
export interface EmbedCorpus { items: number; seconds: number; per_second: number; dims: number }
export interface JobsStatus {
  embed: ModelState & { corpus?: EmbedCorpus };
  photo: ModelState & { photos?: number; photos_shown?: number };
  render: { loaded: boolean; max_width: number; max_height: number; max_iters: number };
  llm?: ModelState & Partial<LlmFacts>;
  offline: { guard_installed: boolean; blocked_attempts: number };
}
export interface LlmFacts {
  model: string; parameters: string; licence: string; threads: number; max_tokens: number; measured_tokens_per_s: number;
}
/** State of the language model; asking for it starts loading the model on the laptop. */
export const useLlmStatus = (enabled: boolean) =>
  useQuery({
    queryKey: ["llm-status"], enabled, retry: false, staleTime: 0,
    queryFn: () => getJson<JobsStatus>("/api/jobs/status?warm=llm"),
    refetchInterval: (q) => (q.state.data?.llm?.loaded || q.state.data?.llm?.error ? 15000 : 1000),
  });
export interface LlmDone {
  tokens: number; prompt_tokens: number; first_token_s: number | null; writing_s: number; tokens_per_s: number | null;
  seconds: number; stopped: boolean; hit_limit: boolean; model?: string; work?: Work;
}
export interface SavedAnswer extends LlmDone { label: string; prompt: string; text: string }
/** Ask the language model; `onText` gets each piece of the answer as soon as the laptop has written it. */
export async function writeAnswer(prompt: string, signal: AbortSignal, onText: (piece: string) => void): Promise<LlmDone> {
  const res = await fetch("/api/llm/generate", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }), signal,
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done: LlmDone | null = null;
  const take = (line: string) => {
    if (!line) return;
    let item: { text?: string; done?: boolean; error?: string };
    try { item = JSON.parse(line); } catch { throw new Error("the service sent an answer the page cannot read"); }
    if (item.error) throw new Error(item.error);
    if (typeof item.text === "string") onText(item.text);
    if (item.done) done = item as unknown as LlmDone;
  };
  for (;;) {
    const { value, done: end } = await reader.read();
    buffer += end ? decoder.decode() : decoder.decode(value, { stream: true });
    let nl = buffer.indexOf("\n");
    while (nl >= 0) {
      take(buffer.slice(0, nl).trim());
      buffer = buffer.slice(nl + 1);
      nl = buffer.indexOf("\n");
    }
    if (end) break;
  }
  take(buffer.trim());                                   // a last line without a line break
  if (!done) throw new Error("the answer ended early");
  return done;
}
export const useJobsStatus = (enabled: boolean) =>
  useQuery({
    queryKey: ["jobs-status"], enabled, retry: false, staleTime: 0,
    queryFn: () => getJson<JobsStatus>("/api/jobs/status?warm=embed,photo"),
    refetchInterval: (q) => {
      const d = q.state.data;
      return !d || ((d.embed.loaded || d.embed.error) && (d.photo.loaded || d.photo.error)) ? (d ? 15000 : 1000) : 1000;
    },
  });
export interface EmbedHit {
  id: string; label: string; group: string; value: unknown; unit: string; status: string; note?: string; score: number;
}
export interface EmbedResult {
  query: string; seconds?: number; results: EmbedHit[]; corpus?: EmbedCorpus; model?: string; tokens?: number; work?: Work;
}
export const useEmbedSearch = (q: string, enabled: boolean) =>
  useQuery({
    queryKey: ["embed-search", q.trim()], enabled: enabled && q.trim().length > 1, retry: false, staleTime: Infinity,
    placeholderData: keepPreviousData,
    queryFn: () => getJson<EmbedResult>(`/api/embed/search?k=8&q=${encodeURIComponent(q.trim())}`),
  });
export type PhotoHit = Omit<SearchHit, "score"> & { score?: number };
export interface PhotoSimilar {
  query?: PhotoHit; encode_seconds?: number; rank_seconds: number; photos: number; photos_shown: number;
  results: PhotoHit[]; width?: number; height?: number; work?: Work;
}
export const usePhotoSample = (seed: number, enabled: boolean) =>
  useQuery({ queryKey: ["photo-sample", seed], enabled, retry: false, staleTime: Infinity,
             queryFn: () => getJson<{ photos: number; photos_shown: number; results: PhotoHit[] }>(`/api/photo/sample?n=12&seed=${seed}`) });
export const usePhotoSimilar = (signature: string, enabled: boolean) =>
  useQuery({ queryKey: ["photo-similar", signature], enabled: enabled && signature.length > 0, retry: false,
             staleTime: Infinity, placeholderData: keepPreviousData,
             queryFn: () => getJson<PhotoSimilar>(`/api/photo/similar?k=12&signature=${encodeURIComponent(signature)}`) });
export async function similarToUpload(file: File): Promise<PhotoSimilar> {
  const res = await fetch("/api/photo/similar?k=12", {
    method: "POST", headers: { "Content-Type": file.type || "application/octet-stream" }, body: file,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
  return body as PhotoSimilar;
}
export interface TryDemo {
  text: Record<string, EmbedResult>;
  photos: { query: PhotoHit; results: PhotoHit[] }[];
  photo_count?: number; photos_shown?: number;
  llm?: Partial<LlmFacts> & { answers: SavedAnswer[] };
  tiles: { name: string; file: string; seconds: number; width: number; height: number; iters: number; pixels: number;
           tiles: number; cx: number; cy: number; span: number; joules?: number | null; watts?: number | null;
           flops?: number | null }[];
}
export const useTryDemo = (enabled: boolean) =>
  useQuery({ queryKey: ["try-demo"], enabled, staleTime: Infinity, queryFn: () => getJson<TryDemo>("/data/tryjobs_demo.json") });
export const useSearchDemo = () =>
  useQuery({ queryKey: ["search-demo"], staleTime: Infinity,
             queryFn: () => getJson<{ results: Record<string, SearchHit[]> }>("/data/search_demo.json") });
