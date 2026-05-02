import { useMemo } from "react";
import type { ContainerTimeline } from "../types";


type Props = {
  timeline: ContainerTimeline;
};

const STATE_COLORS: Record<string, string> = {
  busy: "#ff8a5b",
  idle: "#3a8a8c",
  warming_up: "#c59a4a",
  free: "#e6dfce",
};

const STATE_LABELS: Record<string, string> = {
  busy: "BUSY",
  idle: "IDLE",
  warming_up: "WARMING_UP",
  free: "FREE",
};

export default function ContainerTimeline({ timeline }: Props) {
  const tracks = timeline.tracks;
  const totalMs = Math.max(1, timeline.total_ms);
  const ticks = useMemo(() => buildTicks(totalMs), [totalMs]);

  return (
    <div className="border border-ink/10 rounded bg-paper p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3 font-serif-warm text-[10px] uppercase text-muted">
        {Object.entries(STATE_LABELS).map(([k, label]) => (
          <span key={k} className="flex items-center gap-1">
            <span
              className="inline-block w-3 h-3 rounded-[1px] border border-ink/10"
              style={{ background: STATE_COLORS[k] }}
            />
            {label}
          </span>
        ))}
        <span className="ml-auto">
          {timeline.shown_containers} of {timeline.n_containers} containers ·
          total {formatDuration(totalMs)}
        </span>
      </div>

      <div className="flex">
        <div className="flex flex-col gap-1 pr-2 border-r border-ink/10 text-[10px] font-serif-warm text-muted tabular-nums shrink-0">
          {tracks.map((t) => (
            <div
              key={t.container_id}
              className="flex items-center justify-end"
              style={{ height: 18 }}
            >
              c{t.container_id}
            </div>
          ))}
          <div className="h-5" />
        </div>

        <div className="flex-1 relative">
          <div className="flex flex-col gap-1">
            {tracks.map((t) => (
              <div
                key={t.container_id}
                className="relative bg-warm/40 rounded-[1px] overflow-hidden"
                style={{ height: 18 }}
              >
                {t.segments.map((seg, i) => {
                  const leftPct = (seg.t0_ms / totalMs) * 100;
                  const widthPct =
                    ((seg.t1_ms - seg.t0_ms) / totalMs) * 100;
                  return (
                    <div
                      key={i}
                      className="absolute inset-y-0"
                      style={{
                        left: `${leftPct}%`,
                        width: `${Math.max(0.05, widthPct)}%`,
                        background:
                          STATE_COLORS[seg.state] ?? STATE_COLORS.free,
                      }}
                      title={`${STATE_LABELS[seg.state] ?? seg.state} · ${formatDuration(seg.t0_ms)} → ${formatDuration(seg.t1_ms)} (${formatDuration(seg.t1_ms - seg.t0_ms)})`}
                    />
                  );
                })}
              </div>
            ))}
          </div>

          <div
            className="absolute inset-0 pointer-events-none"
            aria-hidden="true"
          >
            {ticks.values.slice(1, -1).map((v) => (
              <div
                key={v}
                className="absolute top-0 bottom-0 border-l border-ink/10 border-dashed"
                style={{ left: `${(v / totalMs) * 100}%` }}
              />
            ))}
          </div>

          <div className="relative h-5 mt-1 text-[10px] font-serif-warm text-muted tabular-nums">
            {ticks.values.map((v) => (
              <span
                key={v}
                className="absolute -translate-x-1/2"
                style={{ left: `${(v / totalMs) * 100}%` }}
              >
                {formatTick(v, ticks.unit)}
              </span>
            ))}
          </div>
          <div className="text-center text-[10px] font-serif-warm text-muted mt-1">
            simulated time ({ticks.unit === "s" ? "seconds" : "minutes"})
          </div>
        </div>
      </div>
    </div>
  );
}

function buildTicks(totalMs: number): {
  unit: "s" | "min";
  values: number[];
} {
  const unit: "s" | "min" = totalMs <= 120_000 ? "s" : "min";
  const span = unit === "s" ? 1000 : 60_000;
  const ideal = 6;
  const rawUnits = totalMs / ideal / span;
  const niceSteps = [1, 2, 5, 10, 15, 30, 60];
  const chosen =
    niceSteps.find((s) => s >= rawUnits) ?? niceSteps[niceSteps.length - 1];
  const step = chosen * span;
  const out: number[] = [];
  for (let v = 0; v <= totalMs + 1e-6; v += step) {
    out.push(v);
  }
  if (out[out.length - 1] < totalMs) out.push(totalMs);
  return { unit, values: out };
}

function formatTick(ms: number, unit: "s" | "min"): string {
  if (unit === "s") return `${(ms / 1000).toFixed(0)}s`;
  return `${(ms / 60_000).toFixed(0)}m`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}
