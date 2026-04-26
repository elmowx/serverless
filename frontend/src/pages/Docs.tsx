import { type ReactNode } from "react";

const SIGNATURE = `def optimize(
    objective,                     # callable: list[float] -> float
    budget: int,                   # hard cap on objective() calls
    bounds: list[tuple[float, float]],   # per-dim search bounds
) -> list[float]:
    ...`;

export default function Docs() {
  return (
    <div className="flex flex-col gap-10 max-w-3xl">
      <header className="flex flex-col gap-2">
        <p className="font-serif-warm text-[10px] uppercase tracking-[0.2em] text-muted">
          documentation
        </p>
        <h1 className="font-serif-warm text-4xl leading-tight text-ink">
          Documentation
        </h1>
        <p className="font-serif-warm text-[16px] text-ink/80 leading-relaxed">
          Reference for the simulator, the objective, the search space, and the optimizer contract.
        </p>
      </header>

      <Section num="1" title="G/G/k/k simulator">
        <p>
          A discrete-event simulator with a four-state container FSM: FREE, WARMING_UP, BUSY, IDLE.
          Cold-start latency is the sum of four phases — <code>env_init</code>, <code>code_loading</code>,
          <code> runtime_start</code>, <code>function_init</code> — sampled from the SIR-Lab 2025 trace.
        </p>
        <p>
          On arrival, the router prefers a prewarmed IDLE container, then a FREE slot (cold start),
          then a recycled IDLE from another function (cold start). Requests targeting a function with
          a WARMING_UP container join a per-function pending queue and resume on <code>WARM_UP_DONE</code>.
        </p>
      </Section>

      <Section num="2" title="Waiting queue and max_wait_ms">
        <p>
          When the per-function pending path is unavailable and no slot is free, the request enters a
          global FIFO waiting queue. Each request carries an SLA: if the queue head has waited longer
          than <code>max_wait_ms</code>, it is rejected. The default is 30000 ms.
        </p>
        <p>
          Wait time is part of <code>latency_ms = (now - request.timestamp_ms)</code> and contributes
          directly to <code>CVaR_0.99</code> in the loss. Rejections accumulate in <code>p_loss</code>.
          Setting <code>max_wait_ms = 0</code> reverts to a strict G/G/k/k loss queue.
        </p>
      </Section>

      <Section num="3" title="Objective">
        <p>The optimizer minimizes a normalized scalar in [0, 1]:</p>
        <CodeBlock code={`f(x) = w_lat · min(CVaR_0.99(x) / L_max, 1.0) + w_cost · min(idle_seconds(x) / C_max, 1.0)`} />
        <p>
          <code>CVaR_0.99</code> is the mean of the worst-1% latency tail; it is more stable than p99
          across seeds. <code>L_max</code> and <code>C_max</code> are calibrated on the trace by
          simulating two reference policies (<code>minimal</code>, <code>generous</code>) before the
          optimizer runs, so <code>f(x)</code> is comparable across workloads.
        </p>
        <p>
          The <code>min(·, 1.0)</code> clip prevents catastrophic policies (e.g. one container under
          1000 RPS) from dominating surrogate models in Bayesian or evolutionary optimizers.
        </p>
      </Section>

      <Section num="4" title="Workload generation">
        <p><strong>Synthetic Poisson (NHPP).</strong> Non-homogeneous Poisson with diurnal envelope:</p>
        <CodeBlock code={`λ(t) = λ_base · (1 + 0.6 · sin(2π · t / 24h - π/2))`} />
        <p>
          The 0.6 amplitude gives a 4:1 peak-to-trough ratio. Function popularity follows Zipf(1.2),
          calibrated to Azure 2019.
        </p>
        <p><strong>Conditional flow.</strong> A RealNVP trained on Azure 2019 day 1 (262,490 rows, 200 functions).
          Target is <code>(log1p(count), log1p(avg_exec_time_ms))</code>, condition is
          <code> (rank_norm, sin(hour), cos(hour))</code>. Discrete log timestamps are smoothed by
          uniform dequantization U(-0.49, 0.49). Intensity scaling uses Bernoulli thinning (≤1) or
          bootstrap with ±5 s temporal jitter (&gt;1) to preserve the learned joint distribution.</p>
      </Section>

      <Section num="5" title="Search space">
        <CodeBlock code={`bounds = [(1.0, 1800.0), (1.0, max_containers_cap), (0.1, 1.0)]`} />
        <ul className="list-disc pl-6 space-y-1">
          <li><code>keep_alive_s</code> ∈ [1, 1800]. 30 minutes covers the long tail of warm-pool benefits.</li>
          <li><code>max_containers</code> ∈ [1, max_containers_cap]. Default cap is 30; user-tunable per run.</li>
          <li><code>prewarm_threshold</code> ∈ [0.1, 1.0]. If <code>busy / max_containers</code> exceeds this, a new container is spawned proactively. 1.0 disables prewarming.</li>
        </ul>
        <p>
          <code>max_wait_ms</code> is a fixed hyperparameter of the run, not a coordinate of the search space.
        </p>
      </Section>

      <Section num="6" title="Baseline policies">
        <p>Five fixed policies are simulated alongside the optimizer for comparison.</p>
        <table className="w-full text-sm border-collapse mt-2">
          <thead>
            <tr className="border-b border-ink/15 text-left">
              <Th>Name</Th>
              <Th>keep_alive</Th>
              <Th>max_cont.</Th>
              <Th>prewarm</Th>
            </tr>
          </thead>
          <tbody className="font-serif-warm">
            <tr className="border-b border-ink/10">
              <td className="py-2 pr-3"><code>minimal</code></td>
              <td className="py-2 pr-3">60.0s</td>
              <td className="py-2 pr-3">3</td>
              <td className="py-2">1.0</td>
            </tr>
            <tr className="border-b border-ink/10">
              <td className="py-2 pr-3"><code>generous</code></td>
              <td className="py-2 pr-3">1800.0s</td>
              <td className="py-2 pr-3">10</td>
              <td className="py-2">1.0</td>
            </tr>
            <tr className="border-b border-ink/10">
              <td className="py-2 pr-3"><code>aggressive</code></td>
              <td className="py-2 pr-3">30.0s</td>
              <td className="py-2 pr-3">20</td>
              <td className="py-2">1.0</td>
            </tr>
            <tr className="border-b border-ink/10">
              <td className="py-2 pr-3"><code>balanced</code></td>
              <td className="py-2 pr-3">300.0s</td>
              <td className="py-2 pr-3">10</td>
              <td className="py-2">1.0</td>
            </tr>
            <tr className="border-b border-ink/10">
              <td className="py-2 pr-3"><code>prewarm_heavy</code></td>
              <td className="py-2 pr-3">600.0s</td>
              <td className="py-2 pr-3">10</td>
              <td className="py-2">0.6</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-3">
          <code>minimal</code> and <code>generous</code> anchor the normalization: the former
          starves the system to find <code>L_max</code>, the latter overprovisions it to find <code>C_max</code>.
        </p>
      </Section>

      <Section num="7" title="Optimizer contract and sandbox">
        <CodeBlock code={SIGNATURE} />
        <ul className="list-disc pl-6 space-y-1 mt-3">
          <li>Wall-clock timeout: 120 s. CPU limit: 60 s (<code>RLIMIT_CPU</code>).</li>
          <li>Memory: 512 MB cap (<code>RLIMIT_AS</code>/<code>RLIMIT_DATA</code> on Linux; macOS no-ops, enforced via Docker).</li>
          <li>FDs: <code>RLIMIT_NOFILE = 64</code>. Processes: <code>RLIMIT_NPROC = 32</code>.</li>
          <li>Heavy ML libraries (e.g. torch) are blocked to keep submitted optimizers lightweight.</li>
        </ul>
      </Section>

      <Section num="8" title="Report metrics">
        <ul className="list-disc pl-6 space-y-2">
          <li><strong>p99 latency.</strong> 99th percentile of per-request latency (queue wait + cold/warm start + execution).</li>
          <li><strong>CVaR_0.99.</strong> Mean of the worst-1% latency tail.</li>
          <li><strong>cold-start rate.</strong> Fraction of requests routed to a FREE or recycled IDLE container.</li>
          <li><strong>warm-hit rate.</strong> Fraction of requests served from a prewarmed IDLE.</li>
          <li><strong>p_loss.</strong> Fraction of requests rejected (queue timeout or no capacity).</li>
          <li><strong>idle_seconds.</strong> Total container-seconds spent in IDLE state; the cost-side input to <code>f(x)</code>.</li>
        </ul>
      </Section>
    </div>
  );
}

function Section({
  num,
  title,
  children,
}: {
  num: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-baseline gap-3">
        <span className="font-serif-warm text-[10px] text-muted">§{num}</span>
        <h2 className="font-serif-warm text-[26px] leading-tight text-ink">
          {title}
        </h2>
      </div>
      <div className="font-serif-warm text-[15px] text-ink/85 leading-relaxed space-y-3">
        {children}
      </div>
    </section>
  );
}

function Th({ children }: { children: ReactNode }) {
  return (
    <th className="font-serif-warm text-[9px] uppercase text-muted py-2 pr-3 align-top">
      {children}
    </th>
  );
}

function CodeBlock({ code }: { code: string }) {
  return (
    <pre className="bg-warm/50 border border-ink/10 rounded-lg p-4 overflow-x-auto text-[12.5px] leading-relaxed">
      <code>{code}</code>
    </pre>
  );
}
