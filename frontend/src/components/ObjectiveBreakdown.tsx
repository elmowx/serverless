import type { BaselineResult, BestMetrics } from "../types";


type Props = {
  bestY: number;
  bestMetrics: BestMetrics;
  baselines: BaselineResult[];
  wLatency: number;
  wCost: number;
  lMax?: number;
  cMax?: number;
};

type Row = {
  name: string;
  y: number;
  latencyTerm: number;
  costTerm: number;
  highlight: boolean;
};

function resolveTerms(
  m: BestMetrics,
  lMax: number | undefined,
  cMax: number | undefined,
): { latencyTerm: number; costTerm: number } {
  if (m.latency_term != null && m.cost_term != null) {
    return { latencyTerm: m.latency_term, costTerm: m.cost_term };
  }
  const lat =
    m.latency_term ??
    (lMax && lMax > 0 ? m.p99_latency_ms / lMax : 0);
  const cost =
    m.cost_term ??
    (cMax && cMax > 0 ? m.idle_seconds / cMax : 0);
  return { latencyTerm: lat, costTerm: cost };
}

export default function ObjectiveBreakdown({
  bestY,
  bestMetrics,
  baselines,
  wLatency,
  wCost,
  lMax,
  cMax,
}: Props) {
  const bestTerms = resolveTerms(bestMetrics, lMax, cMax);
  const rows: Row[] = [
    {
      name: "YOUR BEST",
      y: bestY,
      latencyTerm: bestTerms.latencyTerm,
      costTerm: bestTerms.costTerm,
      highlight: true,
    },
    ...baselines.map((b) => {
      const terms = resolveTerms(b.metrics, lMax, cMax);
      return {
        name: b.name,
        y: b.y,
        latencyTerm: terms.latencyTerm,
        costTerm: terms.costTerm,
        highlight: false,
      };
    }),
  ];

  const maxWeighted = Math.max(
    ...rows.map((r) => wLatency * r.latencyTerm + wCost * r.costTerm),
    1e-6,
  );

  return (
    <div className="border border-ink/15 rounded bg-paper p-3">
      <div className="flex items-center gap-3 font-serif-warm text-[10px] uppercase text-muted mb-2">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 bg-accent" />
          w_lat·latency_term ({wLatency.toFixed(2)}·)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 bg-[#3a8a8c]" />
          w_cost·cost_term ({wCost.toFixed(2)}·)
        </span>
        <span className="ml-auto">= f(x)</span>
      </div>
      <div className="flex flex-col gap-1.5">
        {rows.map((r) => {
          const latPart = wLatency * r.latencyTerm;
          const costPart = wCost * r.costTerm;
          const total = latPart + costPart;
          const totalPct = (total / maxWeighted) * 100;
          const latPct = total > 0 ? (latPart / total) * totalPct : 0;
          const costPct = total > 0 ? (costPart / total) * totalPct : 0;
          return (
            <div
              key={r.name}
              className={[
                "flex items-center gap-2 text-xs",
                r.highlight ? "font-bold" : "",
              ].join(" ")}
            >
              <div className="w-28 shrink-0 font-serif-warm text-[10px] uppercase text-ink">
                {r.name}
              </div>
              <div className="flex-1 h-5 relative bg-warm/40 rounded overflow-hidden border border-ink/10">
                <div
                  className="absolute inset-y-0 left-0 bg-accent"
                  style={{ width: `${latPct}%` }}
                  title={`latency_term=${r.latencyTerm.toFixed(3)} → ${latPart.toFixed(3)}`}
                />
                <div
                  className="absolute inset-y-0 bg-[#3a8a8c]"
                  style={{ left: `${latPct}%`, width: `${costPct}%` }}
                  title={`cost_term=${r.costTerm.toFixed(3)} → ${costPart.toFixed(3)}`}
                />
              </div>
              <div className="w-20 text-right tabular-nums">
                {r.y.toFixed(4)}
              </div>
              <div className="w-28 text-right tabular-nums text-muted text-[10px]">
                lat {latPart.toFixed(3)} · cost {costPart.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
