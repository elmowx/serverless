import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { createRun, fitFlow, getDatasetsAvailable, getPreview } from "../api/client";
import type { FitFlowResponse, RunConfig } from "../types";
import HelpIcon from "../components/HelpIcon";

type SourceTab = "poisson" | "flow" | "upload" | "fit_flow";

export default function Upload() {
  const navigate = useNavigate();
  const { data: avail } = useQuery({
    queryKey: ["datasets/available"],
    queryFn: getDatasetsAvailable,
  });

  const [solution, setSolution] = useState<File | null>(null);
  const [source, setSource] = useState<SourceTab>("poisson");
  const [intensity, setIntensity] = useState(0.5);
  const [durationMinutes, setDurationMinutes] = useState(60);
  const [nFunctions, setNFunctions] = useState(10);
  const [seed, setSeed] = useState(0);
  const [budget, setBudget] = useState(30);
  const [wLatency, setWLatency] = useState(0.5);
  const [maxContainersCap, setMaxContainersCap] = useState(30);
  const [traceCsv, setTraceCsv] = useState<File | null>(null);
  const [fitCsv, setFitCsv] = useState<File | null>(null);
  const [fitResult, setFitResult] = useState<FitFlowResponse | null>(null);
  const [fitting, setFitting] = useState(false);
  const [fitError, setFitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wCost = useMemo(() => +(1 - wLatency).toFixed(2), [wLatency]);

  const canSubmit =
    !!solution &&
    (source !== "upload" || !!traceCsv) &&
    (source !== "fit_flow" || (fitResult?.passed === true)) &&
    !submitting;

  async function onFitFlow(file: File | null) {
    setFitCsv(file);
    setFitResult(null);
    setFitError(null);
    if (!file) return;
    setFitting(true);
    try {
      const res = await fitFlow(file);
      setFitResult(res);
      if (!res.passed) {
        setFitError("KS-gate failed — try a longer trace or different data.");
      }
    } catch (e) {
      setFitError(e instanceof Error ? e.message : String(e));
    } finally {
      setFitting(false);
    }
  }

  const preview = useQuery({
    enabled: source === "poisson" || source === "flow",
    queryKey: ["preview", source, intensity, durationMinutes, nFunctions, seed],
    queryFn: () =>
      getPreview({
        source: source as "poisson" | "flow",
        intensity,
        duration_minutes: durationMinutes,
        n_functions: nFunctions,
        seed,
        limit: 200,
      }),
  });

  async function onSubmit() {
    if (!solution) return;
    setSubmitting(true);
    setError(null);
    const effectiveSource = source === "fit_flow" ? "flow" : source;
    const cfg: RunConfig = {
      source: effectiveSource,
      intensity,
      duration_minutes: durationMinutes,
      n_functions: nFunctions,
      seed,
      budget,
      w_latency: wLatency,
      w_cost: wCost,
      max_containers_cap: maxContainersCap,
      dataset_id: source === "fit_flow" ? fitResult?.dataset_id ?? null : null,
    };
    try {
      const res = await createRun({ solution, config: cfg, traceCsv });
      navigate(`/runs/${res.run_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <h1 className="font-serif-warm text-xl">Configure run</h1>

      <Section
        title="1. Solution"
        subtitle="The only required upload. A single Python file exposing a top-level optimize() function."
      >
        <FileDrop
          accept=".py"
          file={solution}
          onFile={setSolution}
          hint="Drop or select a .py file defining optimize(objective, budget, bounds). ≤256 KB. Runs in a secure sandbox."
        />
      </Section>

      <Section
        title="2. Data source"
        subtitle="Where the simulated traffic comes from. This changes what your optimizer is tested against."
      >
        <div className="flex gap-2 flex-wrap">
          <Tab active={source === "poisson"} onClick={() => setSource("poisson")}>
            Synthetic (Poisson)
          </Tab>
          <Tab
            active={source === "flow"}
            disabled={!avail?.flow}
            onClick={() => setSource("flow")}
          >
            Historical (Flow){!avail?.flow && " — unavailable"}
          </Tab>
          <Tab active={source === "upload"} onClick={() => setSource("upload")}>
            Your data (raw)
          </Tab>
          <Tab
            active={source === "fit_flow"}
            disabled={!avail?.fit_flow}
            onClick={() => setSource("fit_flow")}
          >
            Train flow on my logs
          </Tab>
        </div>

        <SourceBlurb source={source} />

        {(source === "poisson" || source === "flow" || source === "fit_flow") && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <Slider
              label="Intensity"
              min={0}
              max={1}
              step={0.05}
              value={intensity}
              onChange={setIntensity}
              hint="Slider maps linearly to mean arrivals per second in [1, 50]. 0.5 = ~25 RPS, a realistic mid-load production profile. Low intensity emphasises cold-start cost; high intensity emphasises concurrency and rejections."
              help={
                <>
                  For Flow and fit_flow, this scales the traffic volume while preserving the realistic distribution of your data.
                </>
              }
            />
            <NumberInput
              label="Duration (min)"
              min={1}
              max={1440}
              step={1}
              value={durationMinutes}
              onChange={setDurationMinutes}
              hint="Length of the simulated trace. Longer traces give lower-variance objective values but make each trial slower. 60 min ≈ a few seconds of simulation per evaluation at intensity 0.5."
            />
            <NumberInput
              label="# Functions"
              min={1}
              max={200}
              step={1}
              value={nFunctions}
              onChange={setNFunctions}
              hint="Number of distinct functions in the trace. Popularity follows a realistic skew (a few functions get most of the traffic). More functions → more cold-starts."
            />
            <NumberInput
              label="Seed"
              min={0}
              max={9999}
              step={1}
              value={seed}
              onChange={setSeed}
              hint="RNG seed for trace generation. Keep it fixed if you want every evaluation to run on the exact same traffic."
            />
            {(source === "poisson" || source === "flow") && (
              <div className="col-span-full text-xs text-muted">
                {preview.isFetching
                  ? "sampling preview…"
                  : preview.data
                  ? `preview: ${preview.data.n_total} arrivals total, first ${preview.data.preview.length} shown`
                  : preview.error
                  ? `preview failed: ${(preview.error as Error).message}`
                  : ""}
              </div>
            )}
          </div>
        )}

        {source === "upload" && (
          <div className="mt-4">
            <FileDrop
              accept=".csv"
              file={traceCsv}
              onFile={setTraceCsv}
              hint="CSV columns (required): timestamp_ms, function_id, execution_time_ms. Order doesn't matter — we'll sort. Up to ~10 MB / a few million rows."
            />
          </div>
        )}

        {source === "fit_flow" && (
          <div className="mt-4 flex flex-col gap-3">
            <FileDrop
              accept=".csv"
              file={fitCsv}
              onFile={onFitFlow}
              hint="Drop a per-arrival CSV. We'll aggregate by minute and train a conditional RealNVP on your data. ~30s for a day of logs."
            />
            {fitting && (
              <div className="text-xs text-muted font-serif-warm">
                training flow… KS-gate on hold-out marginals.
              </div>
            )}
            {fitResult && (
              <div
                className={`border rounded p-3 text-xs font-mono ${
                  fitResult.passed
                    ? "border-ink/20 bg-warm/40"
                    : "border-red-300 bg-red-50"
                }`}
              >
                <div className="font-serif-warm text-[10px] uppercase mb-1">
                  {fitResult.cached ? "Using cached weights" : "Training complete"}
                </div>
                <div>dataset_id: {fitResult.dataset_id}</div>
                <div>
                  KS p-values: count={fitResult.ks_p_count.toFixed(3)}, exec=
                  {fitResult.ks_p_exec.toFixed(3)} (pass if &gt; 0.05)
                </div>
                <div>aggregated rows: {fitResult.n_aggregated_rows}</div>
                <div className="mt-1">
                  status:{" "}
                  <strong>{fitResult.passed ? "PASSED" : "FAILED"}</strong>
                </div>
              </div>
            )}
            {fitError && (
              <div className="text-xs text-red-700 break-words">
                {fitError}
              </div>
            )}
          </div>
        )}
      </Section>

      <Section
        title="3. Objective"
        subtitle="Three knobs that shape the final score your optimizer sees."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <NumberInput
            label="Budget (trials)"
            min={1}
            max={500}
            step={1}
            value={budget}
            onChange={setBudget}
            hint="Maximum number of times your optimizer can test a policy. 30-40 is usually enough for a good result."
          />
          <NumberInput
            label="Max containers cap"
            min={1}
            max={30}
            step={1}
            value={maxContainersCap}
            onChange={setMaxContainersCap}
              hint="Upper bound your optimizer is allowed to pick for max_containers. Default is 30. Useful if you want to force the search into a narrower regime."
              help={
                <>
                  The score is still normalized against the full 1..30 range, so objective values stay comparable across runs with different caps.
                </>
              }
          />
          <Slider
            label={`w_latency = ${wLatency.toFixed(2)} / w_cost = ${wCost.toFixed(2)}`}
            min={0}
            max={1}
            step={0.05}
            value={wLatency}
            onChange={setWLatency}
              hint="Trade-off between speed and cost. 1.0 = pure speed (brutal on cold starts), 0.0 = pure cost (keep-alive forever penalised). 0.5 is the default balance."
              help={
                <>
                  Both latency and cost are normalized to a 0-1 scale, so this slider directly controls how much you care about one vs. the other.
                </>
              }
          />
        </div>
      </Section>

      <div className="flex items-center gap-4">
        <button
          disabled={!canSubmit}
          onClick={onSubmit}
          className="font-serif-warm text-xs uppercase px-6 py-4 bg-ink text-paper disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent"
        >
          {submitting ? "Submitting…" : "Start benchmark"}
        </button>
        {error && (
          <span className="text-sm text-red-700 max-w-md break-words">
            {error}
          </span>
        )}
      </div>
    </div>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="font-serif-warm text-sm uppercase">{title}</h2>
      {subtitle && (
        <p className="font-serif-warm text-[14px] text-ink/70 leading-relaxed -mt-1 mb-1 max-w-2xl">
          {subtitle}
        </p>
      )}
      <div className="flex flex-col">{children}</div>
    </section>
  );
}

const SOURCE_BLURBS: Record<SourceTab, string> = {
  poisson:
    "A synthetic mathematical model. Generates traffic with daily peaks and valleys. Reliable and always available.",
  flow:
    "A machine learning model trained on real Azure data. Generates highly realistic traffic patterns based on historical serverless workloads.",
  upload:
    "Directly replay your own production traffic. Upload a CSV with exact timestamps and execution times.",
  fit_flow:
    "Upload your own logs, and we will train a custom machine learning model in ~30 seconds to generate traffic based on your data.",
};

function SourceBlurb({ source }: { source: SourceTab }) {
  return (
    <p className="font-serif-warm text-[13.5px] text-ink/70 leading-relaxed mt-2 max-w-2xl">
      {SOURCE_BLURBS[source]}
    </p>
  );
}

function Tab({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={[
        "font-serif-warm text-[10px] uppercase tracking-wider px-4 py-2 border",
        active ? "bg-ink text-paper border-ink" : "border-ink/20 hover:bg-warm",
        disabled ? "opacity-40 cursor-not-allowed" : "",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
  hint,
  help,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  hint?: string;
  help?: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="font-serif-warm uppercase text-[10px] flex items-center">
        {label}
        {help && <HelpIcon>{help}</HelpIcon>}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="accent-ink"
      />
      {hint && (
        <span className="font-serif-warm text-[12.5px] leading-snug text-muted">
          {hint}
        </span>
      )}
    </label>
  );
}

function NumberInput({
  label,
  min,
  max,
  step,
  value,
  onChange,
  hint,
  help,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  hint?: string;
  help?: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="font-serif-warm uppercase text-[10px] flex items-center">
        {label}
        {help && <HelpIcon>{help}</HelpIcon>}
      </span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10) || 0)}
        className="border border-ink/20 px-2 py-1 bg-paper rounded"
      />
      {hint && (
        <span className="font-serif-warm text-[12.5px] leading-snug text-muted">
          {hint}
        </span>
      )}
    </label>
  );
}

function FileDrop({
  accept,
  file,
  onFile,
  hint,
}: {
  accept: string;
  file: File | null;
  onFile: (f: File | null) => void;
  hint: string;
}) {
  return (
    <label className="border border-dashed border-ink/30 rounded p-6 bg-warm/30 cursor-pointer hover:bg-warm/60 flex flex-col items-start gap-2">
      <input
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      <span className="font-serif-warm text-[10px] uppercase">
        {file ? file.name : "Choose file"}
      </span>
      <span className="text-xs text-muted">{hint}</span>
      {file && (
        <span className="text-xs text-ink/70">
          {(file.size / 1024).toFixed(1)} KB
        </span>
      )}
    </label>
  );
}
