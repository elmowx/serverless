import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ConvergencePoint } from "../types";

function fixedXTicks(budget: number): number[] {
  if (budget <= 1) return [1];
  const step = Math.max(1, Math.ceil(budget / 5));
  const ticks = new Set<number>([1]);
  for (let t = step; t < budget; t += step) ticks.add(t);
  ticks.add(budget);
  return Array.from(ticks).sort((a, b) => a - b);
}

function niceYDomain(points: ConvergencePoint[]): [number, number] {
  if (points.length === 0) return [0, 1];
  const maxY = Math.max(...points.map((p) => Math.max(p.y, p.best_so_far)));
  if (!Number.isFinite(maxY) || maxY <= 0) return [0, 1];
  const padded = maxY * 1.1;
  const ceil = Math.ceil(padded * 2) / 2; // round up to next 0.5
  return [0, Math.max(ceil, 0.5)];
}

function yTicks([lo, hi]: [number, number]): number[] {
  const step = 0.5;
  const ticks: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(Number(v.toFixed(2)));
  return ticks;
}

interface Props {
  data: ConvergencePoint[];
  budget: number;
  height?: number;
  showTooltipExtras?: boolean;
}

export default function ConvergenceChart({
  data,
  budget,
  height = 280,
  showTooltipExtras = true,
}: Props) {
  const xTicks = fixedXTicks(Math.max(budget, 1));
  const yDomain = niceYDomain(data);
  const yTickArr = yTicks(yDomain);
  // n_containers lives on a separate scale (≤ 30); round up to the next
  // multiple of 5 so the right axis has clean ticks.
  const maxContainers = data.reduce(
    (m, p) => (typeof p.n_containers === "number" ? Math.max(m, p.n_containers) : m),
    0,
  );
  const rightMax = Math.max(5, Math.ceil(Math.max(maxContainers, 1) / 5) * 5);
  return (
    <div
      className="border border-ink/10 rounded bg-paper"
      style={{ height, width: "100%" }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 24, right: 24, bottom: 32, left: 40 }}
        >
          <CartesianGrid stroke="#e6dfce" strokeDasharray="2 2" />
          <XAxis
            type="number"
            dataKey="trial"
            domain={[1, Math.max(budget, 1)]}
            ticks={xTicks}
            stroke="#8a8278"
            fontSize={11}
            label={{ value: "trial", position: "insideBottom", dy: 14, fontSize: 11 }}
          />
          <YAxis
            yAxisId="left"
            type="number"
            domain={yDomain}
            ticks={yTickArr}
            stroke="#8a8278"
            fontSize={11}
            label={{
              value: "f(x) · cold_rate",
              angle: -90,
              position: "insideLeft",
              dx: -4,
              fontSize: 11,
            }}
          />
          {showTooltipExtras && (
            <YAxis
              yAxisId="right"
              orientation="right"
              type="number"
              domain={[0, rightMax]}
              allowDecimals={false}
              stroke="#b08968"
              fontSize={11}
              label={{
                value: "n_containers",
                angle: 90,
                position: "insideRight",
                dx: 4,
                fontSize: 11,
              }}
            />
          )}
          <Tooltip
            contentStyle={{ fontSize: 11 }}
            labelFormatter={(v) => `trial ${v}`}
            formatter={(value, name) => {
              if (value == null) return ["—", String(name)];
              const n = typeof value === "number" ? value : Number(value);
              const key = String(name);
              if (key === "f(x) per trial" || key === "best so far") {
                return [Number.isFinite(n) ? n.toFixed(4) : String(value), key];
              }
              if (key === "cold_rate") {
                return [
                  Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : String(value),
                  key,
                ];
              }
              if (key === "n_containers") {
                return [
                  Number.isFinite(n) ? String(Math.round(n)) : String(value),
                  key,
                ];
              }
              return [String(value), key];
            }}
          />
          <Legend
            verticalAlign="top"
            height={24}
            wrapperStyle={{ fontSize: 11 }}
          />
          <Scatter
            yAxisId="left"
            name="f(x) per trial"
            dataKey="y"
            fill="#8a8278"
            shape="circle"
            fillOpacity={0.55}
            isAnimationActive={false}
          />
          <Line
            yAxisId="left"
            type="stepAfter"
            name="best so far"
            dataKey="best_so_far"
            stroke="#ff8a5b"
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />
          {showTooltipExtras && (
            <>
              <Line
                yAxisId="left"
                type="monotone"
                name="cold_rate"
                dataKey="cold_start_rate"
                stroke="#8a8278"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                name="n_containers"
                dataKey="n_containers"
                stroke="#b08968"
                strokeWidth={1.5}
                strokeDasharray="2 3"
                dot={false}
                isAnimationActive={false}
              />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
