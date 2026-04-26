export type Source = "poisson" | "flow" | "upload";

export type ContainerState = "FREE" | "WARMING_UP" | "BUSY" | "IDLE";

export type RunStatus =
  | "queued"
  | "running"
  | "done"
  | "user_error"
  | "crashed"
  | "timeout";

export interface DatasetsAvailable {
  poisson: boolean;
  flow: boolean;
  upload: boolean;
  fit_flow: boolean;
}

export interface FitFlowResponse {
  dataset_id: string;
  cached: boolean;
  passed: boolean;
  ks_p_count: number;
  ks_p_exec: number;
  n_aggregated_rows: number;
}

export interface PreviewArrival {
  timestamp_ms: number;
  function_id: number;
  execution_time_ms: number;
}

export interface PreviewResponse {
  n_total: number;
  preview: PreviewArrival[];
}

export interface RunConfig {
  source: Source;
  intensity: number;
  duration_minutes: number;
  n_functions: number;
  seed: number;
  budget: number;
  w_latency: number;
  w_cost: number;
  max_containers_cap: number;
  max_wait_ms: number;
  dataset_id?: string | null;
}

export interface RunCreateResponse {
  run_id: string;
  status: RunStatus;
  n_arrivals: number;
}

export interface RunStatusResponse {
  id: string;
  status: RunStatus;
  created_at: number;
  finished_at: number | null;
  exit_code: number | null;
  error: string | null;
  config: Record<string, unknown>;
}

export interface ContainerSummary {
  container_id: number;
  busy_frac: number;
  idle_frac: number;
  free_frac: number;
  warming_frac: number;
  cold_starts: number;
  warm_hits: number;
}

export interface TrialEvent {
  trial: number;
  x: number[];
  y: number;
  best_y: number;
  elapsed_s: number;
  step_s?: number;
  n_containers?: number;
  p99_latency_ms?: number;
  cold_start_rate?: number;
  warm_hit_rate?: number;
  idle_seconds?: number;
  p_loss?: number;
  latency_term?: number;
  cost_term?: number;
  container_summary?: ContainerSummary[];
}

export interface BestMetrics {
  p99_latency_ms: number;
  avg_latency_ms?: number;
  cold_start_rate: number;
  p_loss: number;
  idle_seconds: number;
  warm_hits?: number;
  cold_starts?: number;
  latency_term?: number;
  cost_term?: number;
  container_summary?: ContainerSummary[];
}

export interface BaselinePolicy {
  keep_alive_s: number;
  max_containers: number;
  prewarm_threshold: number;
}

export interface BaselineResult {
  name: string;
  policy: BaselinePolicy;
  y: number;
  metrics: BestMetrics;
}

export interface ConvergencePoint {
  trial: number;
  y: number;
  best_so_far: number;
  n_containers?: number;
  cold_start_rate?: number;
}

export interface ParetoPoint {
  x: number[];
  p99_latency_ms: number;
  idle_seconds: number;
}

export interface TimelineSegment {
  state: "free" | "warming_up" | "busy" | "idle" | string;
  t0_ms: number;
  t1_ms: number;
}

export interface ContainerTrack {
  container_id: number;
  segments: TimelineSegment[];
}

export interface ContainerTimeline {
  total_ms: number;
  n_containers: number;
  shown_containers: number;
  tracks: ContainerTrack[];
}

export interface Report {
  best_x: number[];
  best_y: number;
  best_metrics: BestMetrics;
  n_trials: number;
  elapsed_s: number;
  baselines: BaselineResult[];
  convergence: ConvergencePoint[];
  pareto_points: ParetoPoint[];
  container_timeline?: ContainerTimeline | null;
  config: { w_latency: number; w_cost: number; seed: number; max_wait_ms?: number };
  normalization?: {
    l_max: number;
    c_max: number;
    w_latency: number;
    w_cost: number;
  };
}
