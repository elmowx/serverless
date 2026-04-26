import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { getReport } from "../api/client";
import ConvergenceChart from "../components/ConvergenceChart";
import ContainerTimeline from "../components/ContainerTimeline";
import ObjectiveBreakdown from "../components/ObjectiveBreakdown";
import type { BaselineResult, ParetoPoint } from "../types";

const BASELINE_DESCRIPTIONS: Record<string, string> = {
  aggressive:
    "Lots of containers, short keep-alive — optimizes for low latency, burns a lot of idle capacity.",
  balanced:
    "Middle-of-the-road preset. A reasonable first guess with no tuning.",
  generous:
    "High keep-alive, many containers. Anchors the cost_term normalization (C_max is derived from this).",
  prewarm_heavy:
    "Aggressive pre-warming threshold so fewer cold starts, at the price of idle container·seconds.",
  minimal:
    "Few containers, short keep-alive. Anchors the latency_term normalization (L_max is derived from this).",
};

export default function Report() {
  const { runId } = useParams<{ runId: string }>();
  const {
    data: report,
    isLoading,
    error,
  } = useQuery({
    enabled: !!runId,
    queryKey: ["report", runId],
    queryFn: () => getReport(runId!),
  });

  if (isLoading) return <div className="font-serif-warm text-sm">Loading…</div>;
  if (error)
    return (
      <div className="text-sm text-red-700">
        Failed to load report: {(error as Error).message}
        <div className="mt-2">
          <Link to="/new" className="underline">
            Start over
          </Link>
        </div>
      </div>
    );
  if (!report) return null;

  const m = report.best_metrics;
  const convergence = report.convergence;
  const best = report.best_x;
  const wLat = report.config.w_latency;
  const wCost = report.config.w_cost;
  const norm = report.normalization;

  const paretoData = report.pareto_points.map((p: ParetoPoint) => ({
    p99: p.p99_latency_ms,
    idle: p.idle_seconds,
  }));
  const baselineScatter = report.baselines.map((b: BaselineResult) => ({
    p99: b.metrics.p99_latency_ms,
    idle: b.metrics.idle_seconds,
    name: b.name,
  }));
  const bestScatter = [
    { p99: m.p99_latency_ms, idle: m.idle_seconds, name: "your best" },
  ];

  const wonF = report.baselines.filter((b) => report.best_y < b.y).length;
  const wonP99 = report.baselines.filter(
    (b) => m.p99_latency_ms < b.metrics.p99_latency_ms,
  ).length;
  const wonIdle = report.baselines.filter(
    (b) => m.idle_seconds < b.metrics.idle_seconds,
  ).length;
  const nBaselines = report.baselines.length;

  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-baseline justify-between">
        <h1 className="font-serif-warm text-xl">Report</h1>
        <span className="font-serif-warm text-[10px] text-muted">
          run: {runId}
        </span>
      </div>

      <HowToRead wLat={wLat} wCost={wCost} />

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">Headline</h2>
        <div className="border border-ink/15 rounded bg-paper p-4 flex flex-wrap gap-6">
          <Verdict label="f(x) vs baselines" won={wonF} total={nBaselines} />
          <Verdict label="p99 latency vs baselines" won={wonP99} total={nBaselines} />
          <Verdict label="idle cost vs baselines" won={wonIdle} total={nBaselines} />
          <div className="flex flex-col">
            <span className="font-serif-warm text-[9px] uppercase text-muted">
              best f(x)
            </span>
            <span className="font-serif-warm text-lg text-accent">
              {report.best_y.toFixed(4)}
            </span>
            <span className="text-[10px] text-muted">
              over {report.n_trials} trials · {report.elapsed_s.toFixed(1)}s
            </span>
          </div>
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">Top-line metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Metric
            label="p99 latency"
            value={`${m.p99_latency_ms.toFixed(0)} ms`}
            hint="99th-percentile end-to-end latency. 1 in 100 requests is slower than this. Lower = better user experience."
          />
          <Metric
            label="cold start rate"
            value={`${(m.cold_start_rate * 100).toFixed(1)} %`}
            hint="Fraction of requests that had to wait for a container to spin up. Lower = smoother response."
          />
          <Metric
            label="p_loss (rejected)"
            value={`${(m.p_loss * 100).toFixed(2)} %`}
            hint="Fraction rejected because all k containers were busy (Erlang-B blocking). Should stay near 0."
          />
          <Metric
            label="idle cost"
            value={`${m.idle_seconds.toFixed(0)} cont·s`}
            hint="Container-seconds spent IDLE or warming with no request. This is the money you pay for empty capacity."
          />
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">Your best policy</h2>
        <p className="text-xs text-muted">
          The three knobs the optimizer tunes. Hover each field for what it
          does. For reference, the two anchor baselines are shown beside
          it — they are how L_max and C_max get their values.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <PolicyCard
            title="Your best"
            keepAlive={best[0]}
            maxContainers={best[1]}
            prewarm={best[2]}
            y={report.best_y}
            highlight
          />
          {report.baselines
            .filter((b) => b.name === "minimal" || b.name === "generous")
            .map((b) => (
              <PolicyCard
                key={b.name}
                title={b.name}
                subtitle={BASELINE_DESCRIPTIONS[b.name]}
                keepAlive={b.policy.keep_alive_s}
                maxContainers={b.policy.max_containers}
                prewarm={b.policy.prewarm_threshold}
                y={b.y}
              />
            ))}
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">Convergence</h2>
        <p className="text-xs text-muted">
          Grey dots are each trial's f(x); the orange step line is the
          best-so-far (plateau = the optimizer has stopped improving). The
          dashed grey line is the per-trial cold-start rate on the left
          axis, the dashed brown line is n_containers on the right axis —
          watch them move while f(x) is still dropping to see which knob
          the optimizer is turning.
        </p>
        <ConvergenceChart
          data={convergence}
          budget={report.n_trials}
          height={280}
        />
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">
          Objective decomposition
        </h2>
        <p className="text-xs text-muted">
          f(x) = w<sub>lat</sub> · <span className="text-accent">latency_term</span>{" "}
          + w<sub>cost</sub> ·{" "}
          <span style={{ color: "#3a8a8c" }}>cost_term</span>. Both terms are
          normalized by per-trace anchor policies, so each is ≈[0, 1] and
          they can be compared directly on one bar. Lower f(x) is better.{" "}
          <strong>
            {wCost > wLat
              ? "You weighted cost higher — narrower green means the run met that goal."
              : wLat > wCost
                ? "You weighted latency higher — narrower orange means the run met that goal."
                : "50/50 weighting, so a tall orange or tall green segment directly shows where the remaining objective value is coming from."}
          </strong>
        </p>
        <ObjectiveBreakdown
          bestY={report.best_y}
          bestMetrics={m}
          baselines={report.baselines}
          wLatency={wLat}
          wCost={wCost}
          lMax={norm?.l_max}
          cMax={norm?.c_max}
        />
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">Impact vs baselines</h2>
        <p className="text-xs text-muted">
          Δ = your_best − baseline, per dimension.{" "}
          <span className="text-emerald-700">Green</span> means your optimizer
          won that dimension (lower is always better).{" "}
          <span className="text-red-700">Red</span> means the baseline was
          better on that metric. A baseline can win one dimension and lose
          another — f(x) aggregates them with your chosen weights.
        </p>
        <div className="overflow-x-auto">
          <table className="text-sm border-collapse w-full">
            <thead className="font-serif-warm text-[10px] uppercase">
              <tr className="border-b border-ink/20">
                <th className="text-left py-2 px-2">vs baseline</th>
                <th className="text-right py-2 px-2" title="Objective value delta">
                  Δ f(x)
                </th>
                <th className="text-right py-2 px-2" title="99th-percentile latency delta (ms)">
                  Δ p99 (ms)
                </th>
                <th className="text-right py-2 px-2" title="Container-seconds delta — negative means you burned less capacity">
                  Δ idle (cont·s)
                </th>
                <th className="text-right py-2 px-2" title="Normalized cost component delta">
                  Δ cost_term
                </th>
                <th className="text-right py-2 px-2" title="Normalized latency component delta">
                  Δ lat_term
                </th>
              </tr>
            </thead>
            <tbody>
              {report.baselines.map((b) => {
                const dF = report.best_y - b.y;
                const dP99 = m.p99_latency_ms - b.metrics.p99_latency_ms;
                const dIdle = m.idle_seconds - b.metrics.idle_seconds;
                const bLat =
                  b.metrics.latency_term ??
                  (norm?.l_max ? b.metrics.p99_latency_ms / norm.l_max : 0);
                const bCost =
                  b.metrics.cost_term ??
                  (norm?.c_max ? b.metrics.idle_seconds / norm.c_max : 0);
                const mLat =
                  m.latency_term ??
                  (norm?.l_max ? m.p99_latency_ms / norm.l_max : 0);
                const mCost =
                  m.cost_term ??
                  (norm?.c_max ? m.idle_seconds / norm.c_max : 0);
                const dCost = mCost - bCost;
                const dLat = mLat - bLat;
                return (
                  <tr key={b.name} className="border-b border-ink/10">
                    <td className="py-2 px-2">
                      <span className="font-serif-warm text-xs uppercase">
                        {b.name}
                      </span>
                      <div className="text-[10px] text-muted">
                        {BASELINE_DESCRIPTIONS[b.name]}
                      </div>
                    </td>
                    <td className={cellColor(dF)}>{fmtSigned(dF, 4)}</td>
                    <td className={cellColor(dP99)}>{fmtSigned(dP99, 0)}</td>
                    <td className={cellColor(dIdle)}>{fmtSigned(dIdle, 0)}</td>
                    <td className={cellColor(dCost)}>{fmtSigned(dCost, 4)}</td>
                    <td className={cellColor(dLat)}>{fmtSigned(dLat, 4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">
          Pareto: p99 latency × idle cost
        </h2>
        <p className="text-xs text-muted">
          Each point is a policy the optimizer tried. The{" "}
          <span style={{ color: "#ff8a5b" }}>orange</span> points form the
          Pareto frontier — no other tried policy is simultaneously lower on
          both p99 latency and idle cost. Black triangles are the 5 fixed
          baselines, so you can eyeball where your search pushed the
          frontier beyond the presets.
        </p>
        <div className="h-[320px] border border-ink/10 rounded bg-paper">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 16, right: 24, bottom: 40, left: 64 }}>
              <CartesianGrid stroke="#e6dfce" strokeDasharray="2 2" />
              <XAxis
                type="number"
                dataKey="p99"
                name="p99 latency"
                unit=" ms"
                stroke="#8a8278"
                fontSize={10}
                label={{
                  value: "p99 latency (ms) — lower is better",
                  position: "insideBottom",
                  offset: -8,
                  style: { fontSize: 10, fill: "#8a8278" },
                }}
              />
              <YAxis
                type="number"
                dataKey="idle"
                name="idle"
                unit=" s"
                stroke="#8a8278"
                fontSize={10}
                label={{
                  value: "idle cost (cont·s) — lower is better",
                  angle: -90,
                  position: "left",
                  offset: 10,
                  style: {
                    fontSize: 10,
                    fill: "#8a8278",
                    textAnchor: "middle",
                  },
                }}
              />
              <ZAxis range={[80, 80]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Legend verticalAlign="top" height={24} iconSize={10} />
              <Scatter
                name="Pareto frontier"
                data={paretoData}
                fill="#ff8a5b"
                legendType="circle"
              />
              <Scatter
                name="your best"
                data={bestScatter}
                fill="#ff8a5b"
                shape="star"
                legendType="star"
              />
              <Scatter
                name="baselines"
                data={baselineScatter}
                fill="#1f1b16"
                shape="triangle"
                legendType="triangle"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </section>

      {report.container_timeline &&
        report.container_timeline.tracks.length > 0 && (
          <section className="flex flex-col gap-2">
            <h2 className="font-serif-warm text-sm uppercase">
              Container state timeline (best policy)
            </h2>
            <p className="text-xs text-muted">
              Real simulated timeline of every container under your best
              policy. Each row is one container; each segment is a stretch
              spent in a single FSM state.{" "}
              <span style={{ color: "#ff8a5b" }}>BUSY</span> stretches are
              where requests are being served (and latency accrues),{" "}
              <span style={{ color: "#3a8a8c" }}>IDLE</span> stretches are
              where idle_seconds accrue (the keep-alive cost),{" "}
              <span style={{ color: "#c59a4a" }}>WARMING_UP</span> are cold
              starts in flight. Hover a segment for its exact interval.
            </p>
            <ContainerTimeline timeline={report.container_timeline} />
          </section>
        )}

      <section className="flex flex-col gap-2">
        <h2 className="font-serif-warm text-sm uppercase">Full comparison table</h2>
        <div className="overflow-x-auto">
          <table className="text-sm border-collapse w-full">
            <thead className="font-serif-warm text-[10px] uppercase">
              <tr className="border-b border-ink/20">
                <th className="text-left py-2 px-2">Policy</th>
                <th className="text-right py-2 px-2">f(x)</th>
                <th className="text-right py-2 px-2">p99 ms</th>
                <th className="text-right py-2 px-2">cold %</th>
                <th className="text-right py-2 px-2">p_loss %</th>
                <th className="text-right py-2 px-2">idle s</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-ink/10 bg-warm/50">
                <td className="py-2 px-2 font-serif-warm text-[10px]">YOUR BEST</td>
                <td className="text-right py-2 px-2 font-bold">
                  {report.best_y.toFixed(4)}
                </td>
                <td className="text-right py-2 px-2">
                  {m.p99_latency_ms.toFixed(0)}
                </td>
                <td className="text-right py-2 px-2">
                  {(m.cold_start_rate * 100).toFixed(1)}
                </td>
                <td className="text-right py-2 px-2">
                  {(m.p_loss * 100).toFixed(2)}
                </td>
                <td className="text-right py-2 px-2">
                  {m.idle_seconds.toFixed(0)}
                </td>
              </tr>
              {report.baselines.map((b) => (
                <tr key={b.name} className="border-b border-ink/10">
                  <td className="py-2 px-2">
                    <span>{b.name}</span>
                    <div className="text-[10px] text-muted">
                      {BASELINE_DESCRIPTIONS[b.name]}
                    </div>
                  </td>
                  <td className="text-right py-2 px-2">{b.y.toFixed(4)}</td>
                  <td className="text-right py-2 px-2">
                    {b.metrics.p99_latency_ms.toFixed(0)}
                  </td>
                  <td className="text-right py-2 px-2">
                    {(b.metrics.cold_start_rate * 100).toFixed(1)}
                  </td>
                  <td className="text-right py-2 px-2">
                    {(b.metrics.p_loss * 100).toFixed(2)}
                  </td>
                  <td className="text-right py-2 px-2">
                    {b.metrics.idle_seconds.toFixed(0)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {norm && (
        <section className="text-[10px] text-muted border-t border-ink/10 pt-3">
          <span className="font-serif-warm uppercase">Run config · </span>
          w_latency = {norm.w_latency.toFixed(2)} · w_cost ={" "}
          {norm.w_cost.toFixed(2)} · seed = {report.config.seed} · max_wait_ms ={" "}
          {report.config.max_wait_ms ?? 0}
          <br />
          <span className="font-serif-warm uppercase">Normalization anchors · </span>
          L_max = {norm.l_max.toFixed(0)} ms (from the{" "}
          <span className="font-serif-warm">minimal</span> baseline) · C_max ={" "}
          {norm.c_max.toFixed(0)} cont·s (from the{" "}
          <span className="font-serif-warm">generous</span> baseline)
        </section>
      )}

      <div className="flex gap-3">
        <Link
          to="/new"
          className="font-serif-warm text-xs uppercase px-4 py-3 border border-ink/30 hover:bg-warm"
        >
          Run another
        </Link>
        <a
          href={`/api/runs/${runId}/report.pdf`}
          target="_blank"
          rel="noreferrer"
          className="font-serif-warm text-xs uppercase px-4 py-3 bg-ink text-paper hover:bg-accent"
        >
          Download PDF
        </a>
      </div>
    </div>
  );
}

function HowToRead({ wLat, wCost }: { wLat: number; wCost: number }) {
  return (
    <details className="border border-ink/15 rounded bg-warm/40 p-3 text-sm">
      <summary className="cursor-pointer font-serif-warm text-xs uppercase">
        How to read this report
      </summary>
      <div className="mt-3 flex flex-col gap-2 text-xs leading-relaxed">
        <p>
          <strong>Your optimizer</strong> searched over three knobs of a
          serverless scheduler: <code>keep_alive_s</code> (how long to keep an
          idle container alive), <code>max_containers</code> (pool size), and{" "}
          <code>prewarm_threshold</code> (when to proactively warm a spare).
          It called the simulator <em>budget</em>-many times and picked the
          policy that minimized f(x).
        </p>
        <p>
          <strong>f(x)</strong> = w<sub>lat</sub>·(p99/L_max) + w<sub>cost</sub>
          ·(idle/C_max). You picked{" "}
          <span className="font-serif-warm">w_lat={wLat.toFixed(2)}</span> and{" "}
          <span className="font-serif-warm">w_cost={wCost.toFixed(2)}</span>, so
          f(x) is literally a weighted average of two unitless numbers in
          roughly [0, 1]. A value below{" "}
          <span className="font-serif-warm">~0.5</span> is solidly beating the
          anchor baselines; a value above 1 means at least one dimension is
          worse than the worst anchor.
        </p>
        <p>
          <strong>Baselines</strong> are 5 hand-picked policies that bracket
          the search space. <em>minimal</em> and <em>generous</em> double as
          the normalization anchors (they define L_max and C_max). If your
          best is below{" "}
          <em>balanced</em>, the optimizer found something non-trivial.
        </p>
      </div>
    </details>
  );
}

function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div
      className="border border-ink/15 rounded p-3 bg-paper"
      title={hint}
    >
      <div className="font-serif-warm text-[9px] uppercase text-muted">{label}</div>
      <div className="font-serif-warm text-base mt-1">{value}</div>
      <div className="mt-1 text-[10px] text-muted leading-snug">{hint}</div>
    </div>
  );
}

function Verdict({
  label,
  won,
  total,
}: {
  label: string;
  won: number;
  total: number;
}) {
  const color =
    won === total
      ? "text-emerald-700"
      : won === 0
        ? "text-red-700"
        : "text-ink";
  return (
    <div className="flex flex-col">
      <span className="font-serif-warm text-[9px] uppercase text-muted">
        {label}
      </span>
      <span className={`font-serif-warm text-lg ${color}`}>
        {won}/{total}
      </span>
      <span className="text-[10px] text-muted">
        {won === total
          ? "swept every baseline"
          : won === 0
            ? "baselines won here"
            : `beat ${won} of ${total}`}
      </span>
    </div>
  );
}

function PolicyCard({
  title,
  subtitle,
  keepAlive,
  maxContainers,
  prewarm,
  y,
  highlight,
}: {
  title: string;
  subtitle?: string;
  keepAlive: number;
  maxContainers: number;
  prewarm: number;
  y: number;
  highlight?: boolean;
}) {
  return (
    <div
      className={[
        "border rounded p-4 flex flex-col gap-1",
        highlight ? "border-accent bg-warm/70" : "border-ink/15 bg-paper",
      ].join(" ")}
    >
      <div className="font-serif-warm text-xs uppercase">{title}</div>
      {subtitle && (
        <div className="text-[10px] text-muted leading-snug mb-1">
          {subtitle}
        </div>
      )}
      <Row
        k="keep_alive_s"
        v={keepAlive.toFixed(1)}
        hint="How long a container stays alive after its last request before being recycled"
      />
      <Row
        k="max_containers"
        v={Math.round(maxContainers).toString()}
        hint="Pool size k — requests arriving when all k are busy get rejected (p_loss)"
      />
      <Row
        k="prewarm_thresh"
        v={prewarm.toFixed(2)}
        hint="When busy ratio > threshold, proactively spin up a spare container (1.0 = disabled)"
      />
      <Row k="f(x)" v={y.toFixed(4)} hint="Combined objective — lower is better" />
    </div>
  );
}

function Row({ k, v, hint }: { k: string; v: string; hint?: string }) {
  return (
    <div className="flex justify-between text-xs" title={hint}>
      <span className="text-muted">{k}</span>
      <span>{v}</span>
    </div>
  );
}

function fmtSigned(v: number, digits: number): string {
  if (!Number.isFinite(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}`;
}

function cellColor(v: number): string {
  const base = "text-right py-2 px-2 tabular-nums";
  if (!Number.isFinite(v) || Math.abs(v) < 1e-9) return `${base} text-muted`;
  return v < 0
    ? `${base} text-emerald-700`
    : `${base} text-red-700`;
}
