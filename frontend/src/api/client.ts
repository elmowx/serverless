import type {
  DatasetsAvailable,
  FitFlowResponse,
  PreviewResponse,
  Report,
  RunConfig,
  RunCreateResponse,
  RunStatusResponse,
  Source,
} from "../types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`GET ${path}: ${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function getDatasetsAvailable(): Promise<DatasetsAvailable> {
  return get<DatasetsAvailable>("/datasets/available");
}

export async function getPreview(params: {
  source: Source;
  intensity: number;
  duration_minutes: number;
  n_functions: number;
  seed: number;
  limit?: number;
}): Promise<PreviewResponse> {
  const qs = new URLSearchParams({
    source: params.source,
    intensity: String(params.intensity),
    duration_minutes: String(params.duration_minutes),
    n_functions: String(params.n_functions),
    seed: String(params.seed),
    limit: String(params.limit ?? 200),
  });
  return get<PreviewResponse>(`/datasets/preview?${qs.toString()}`);
}

export async function createRun(args: {
  solution: File;
  config: RunConfig;
  traceCsv?: File | null;
}): Promise<RunCreateResponse> {
  const fd = new FormData();
  fd.append("solution", args.solution);
  fd.append("config", JSON.stringify(args.config));
  if (args.traceCsv) fd.append("trace_csv", args.traceCsv);
  const r = await fetch(`${BASE}/runs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`POST /runs: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function getRun(runId: string): Promise<RunStatusResponse> {
  return get<RunStatusResponse>(`/runs/${runId}`);
}

export async function getReport(runId: string): Promise<Report> {
  return get<Report>(`/runs/${runId}/report`);
}

export function runEventsUrl(runId: string): string {
  return `${BASE}/runs/${runId}/events`;
}

export async function fitFlow(csv: File): Promise<FitFlowResponse> {
  const fd = new FormData();
  fd.append("trace_csv", csv);
  const r = await fetch(`${BASE}/datasets/fit-flow`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`POST /datasets/fit-flow: ${r.status} ${await r.text()}`);
  return r.json();
}
