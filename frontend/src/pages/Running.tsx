import { useEffect, useMemo, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getRun } from "../api/client";
import { useRunStream } from "../hooks/useRunStream";
import ContainerStateCounters from "../components/ContainerStateCounters";
import ConvergenceChart from "../components/ConvergenceChart";
import HelpIcon from "../components/HelpIcon";
import type { ConvergencePoint } from "../types";

export default function Running() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { trials, done, connected, error } = useRunStream(runId);

  const { data: run } = useQuery({
    enabled: !!runId,
    queryKey: ["run", runId],
    queryFn: () => getRun(runId!),
    refetchInterval: 1500,
  });

  const budget = useMemo(() => {
    const cfg = run?.config as { budget?: number } | undefined;
    return cfg?.budget ?? 30;
  }, [run]);

  useEffect(() => {
    if (done?.status === "done" && runId) {
      const t = setTimeout(() => navigate(`/runs/${runId}/report`), 400);
      return () => clearTimeout(t);
    }
  }, [done, navigate, runId]);

  const nTrials = trials.length;
  const latest = trials[trials.length - 1];
  const bestSoFar = trials.length
    ? Math.min(...trials.map((t) => t.best_y))
    : null;

  const chartData: ConvergencePoint[] = trials.map((t) => ({
    trial: t.trial,
    y: t.y,
    best_so_far: t.best_y,
    n_containers: t.n_containers,
    cold_start_rate: t.cold_start_rate,
  }));

  // UI status hint: the backend /runs/:id endpoint says "running" until the
  // sandbox subprocess exits, even after the last trial was emitted. Show a
  // friendlier "wrapping up" when we've seen budget trials but the stream has
  // not yet sent a `done` event.
  const statusHint =
    done?.status ??
    (nTrials >= budget ? "wrapping up" : run?.status ?? "…");

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-baseline justify-between">
        <h1 className="font-serif-warm text-xl">Running</h1>
        <span className="font-serif-warm text-[10px] text-muted">
          run: {runId}
        </span>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1">
          <div className="h-2 bg-warm rounded overflow-hidden">
            <div
              className="h-full bg-ink transition-all"
              style={{
                width: `${Math.min(100, (nTrials / budget) * 100)}%`,
              }}
            />
          </div>
          <div className="mt-1 text-xs text-muted font-serif-warm">
            {nTrials}/{budget} trials · status: {statusHint}
            {connected ? "" : " · reconnecting"}
          </div>
        </div>
        {bestSoFar !== null && (
          <div className="font-serif-warm text-sm">
            best f(x) = {bestSoFar.toFixed(4)}
          </div>
        )}
      </div>

      {error && (
        <div className="text-xs text-red-700">stream error: {error}</div>
      )}

      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <LiveMetric
          label="trial f(x)"
          value={latest ? latest.y.toFixed(4) : "—"}
          tooltip="The final score your optimizer is trying to minimize. Combines latency and cost."
        />
        <LiveMetric
          label="p99 latency"
          value={
            latest?.p99_latency_ms != null
              ? `${latest.p99_latency_ms.toFixed(0)} ms`
              : "—"
          }
          tooltip="99% of requests were faster than this time."
        />
        <LiveMetric
          label="cold start rate"
          value={
            latest?.cold_start_rate != null
              ? `${(latest.cold_start_rate * 100).toFixed(1)} %`
              : "—"
          }
          tooltip="Percentage of requests that had to wait for a container to spin up."
        />
        <LiveMetric
          label="idle"
          value={
            latest?.idle_seconds != null
              ? `${latest.idle_seconds.toFixed(0)} cont·s`
              : "—"
          }
          tooltip="Total time containers spent waiting (the cost part of your score)."
        />
        <LiveMetric
          label="warm hit rate"
          value={
            latest?.warm_hit_rate != null
              ? `${(latest.warm_hit_rate * 100).toFixed(1)} %`
              : "—"
          }
          tooltip="Percentage of requests served instantly by an already-warm container."
        />
        <LiveMetric
          label="p_loss"
          value={
            latest?.p_loss != null
              ? `${(latest.p_loss * 100).toFixed(2)} %`
              : "—"
          }
          tooltip="Percentage of requests rejected because all allowed containers were busy."
        />
        <LiveMetric
          label="max containers"
          value={latest?.n_containers != null ? String(latest.n_containers) : "—"}
          tooltip="The max_containers limit your optimizer picked for this trial."
        />
        <LiveMetric
          label="step time"
          value={
            latest?.step_s != null ? `${latest.step_s.toFixed(2)} s` : "—"
          }
          tooltip="How long it took the backend to simulate this trial."
        />
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase text-muted">
          Container states
        </h2>
        <p className="text-sm text-ink/70 font-serif-warm -mt-1 mb-1">
          Time-weighted count of containers in each FSM state during the
          latest trial — one cat per state, numbers update per trial.
        </p>
        <ContainerStateCounters summary={latest?.container_summary} />
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">Convergence</h2>
        <ConvergenceChart data={chartData} budget={budget} height={280} />
      </section>

      {done && done.status !== "done" && (
        <div className="border border-red-300 bg-red-50 rounded p-4 text-sm">
          <div className="font-serif-warm text-xs uppercase mb-1">
            Run ended: {done.status}
          </div>
          {done.error && <pre className="whitespace-pre-wrap">{done.error}</pre>}
          
          {done.status === "timeout" && (
            <div className="mt-3 bg-red-100/50 p-3 rounded border border-red-200 text-red-800">
              <strong className="block mb-1 font-serif-warm text-[13px]">Time Limit Exceeded</strong>
              <p className="mb-2 text-red-800/90">Your optimizer took too long and was killed by the sandbox (120s wall-clock or 60s CPU limit). Here is what you can do:</p>
              <ul className="list-disc pl-5 space-y-1 text-red-800/90">
                <li><strong>Check for infinite loops:</strong> Ensure your code doesn't have a <code>while True:</code> loop without a break condition.</li>
                <li><strong>Reduce the Budget:</strong> If your optimizer is computationally heavy (like Grid Search), try lowering the budget on the previous page.</li>
                <li><strong>Reduce Trace Duration:</strong> Simulating very long traces (e.g., 24 hours) takes more time per evaluation. Try reducing the duration to 60 minutes.</li>
                <li><strong>Optimize your code:</strong> Avoid heavy operations (like training large ML models) inside the <code>optimize</code> function.</li>
              </ul>
            </div>
          )}

          <Link to="/new" className="underline mt-3 inline-block">
            Start over
          </Link>
        </div>
      )}
    </div>
  );
}

function LiveMetric({
  label,
  value,
  tooltip,
}: {
  label: string;
  value: string;
  tooltip?: ReactNode;
}) {
  return (
    <div className="border border-ink/15 rounded p-3 bg-paper">
      <div className="font-serif-warm text-[9px] uppercase text-muted flex items-center">
        {label}
        {tooltip && <HelpIcon>{tooltip}</HelpIcon>}
      </div>
      <div className="font-serif-warm text-base mt-1">{value}</div>
    </div>
  );
}
