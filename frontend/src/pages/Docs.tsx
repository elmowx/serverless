import { useState, type ReactNode } from "react";

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
          Serverless Black-Box: Deep Dive
        </h1>
        <p className="font-serif-warm text-[16px] text-ink/80 leading-relaxed">
          This document details the rigorous mathematical foundations, internal logic, and exact mechanics of the Serverless Black-Box benchmark. It includes detailed explanations of <em>why</em> specific formulas, constants, and architectural decisions were chosen.
        </p>
      </header>

      <Section
        num="1"
        title="The G/G/k/k Simulator & FSM"
        intro="The mathematical and logical foundation of the serverless environment."
      >
        <p>
          The core of the benchmark is a discrete-event simulator modeling a <strong>G/G/k/k loss queue</strong> with a four-state container Finite State Machine (FSM).
        </p>
        <ul className="list-disc pl-6 space-y-2">
          <li><strong>FREE:</strong> An unallocated concurrency slot.</li>
          <li>
            <strong>WARMING_UP:</strong> The container is initializing. Cold start latency is strictly the sum of four phases: <code>env_init</code>, <code>code_loading</code>, <code>runtime_start</code>, and <code>function_init</code>.
          </li>
          <li><strong>BUSY:</strong> The container is actively executing a request.</li>
          <li><strong>IDLE:</strong> The container is warm, kept-alive, and waiting for a hit.</li>
        </ul>
        
        <Rationale title="Why these 4 cold-start phases?">
          We use the exact phases and empirical distributions from Huawei's SIR-Lab 2025 dataset (805,745 production events). Most academic simulators treat cold starts as a single random variable. Breaking it into 4 phases allows us to accurately model <em>partial</em> cold starts (e.g., a prewarmed container skips <code>env_init</code> but still pays for <code>code_loading</code> and <code>function_init</code>).
        </Rationale>

        <p className="mt-4"><strong>Routing Logic & Pending Queue:</strong></p>
        <p>
          When a request arrives, the router attempts to assign it to a container using a strict preference hierarchy:
        </p>
        <ol className="list-decimal pl-6 space-y-1 mb-2">
          <li><strong>Prewarmed IDLE:</strong> Containers in the IDLE state with no specific <code>function_id</code> bound yet.</li>
          <li><strong>FREE:</strong> Triggers a full cold start.</li>
          <li><strong>Recycled IDLE:</strong> Reclaims a stale IDLE container from another function (triggers a full cold start).</li>
        </ol>
        
        <Rationale title="Why a pending queue for WARMING_UP?">
          If a target function already has a container in the <code>WARMING_UP</code> state, new requests join a <strong>per-function pending queue</strong> rather than being immediately rejected or triggering redundant cold starts. Early versions of this simulator used "strict-loss" (rejecting the request), but this was overly pessimistic. The pending queue accurately mirrors AWS Lambda's provisioned concurrency behavior, where requests wait for a currently-warming container to emit a <code>WARM_UP_DONE</code> event.
        </Rationale>
      </Section>

      <Section
        num="2"
        title="The Black-Box Objective"
        intro="The exact mathematical formula your optimizer is minimizing."
      >
        <p>
          The simulator collapses the entire trace execution into a single normalized scalar <code>f(x) ∈ [0, 1]</code>:
        </p>
        <CodeBlock code={`f(x) = w_lat · min(CVaR_0.99(x) / L_max, 1.0) + w_cost · min(idle_seconds(x) / C_max, 1.0)`} />
        
        <Rationale title="Why CVaR_0.99 instead of p99?">
          <code>CVaR_0.99</code> (Conditional Value at Risk) is the exact mean latency of the top 1% worst requests (<code>E[L | L ≥ p99]</code>). The 99th percentile (p99) is a single-point statistic that is highly volatile across different RNG seeds. CVaR averages the entire tail, significantly reducing variance and creating a smoother objective landscape for derivative-free optimizers (like TPE or Gaussian Processes).
        </Rationale>

        <Rationale title="Why L_max and C_max normalization?">
          Without normalization, <code>f(x)</code> values would be incomparable between a 5-minute trace and a 24-hour trace. Before your optimizer runs, the system evaluates two extreme reference policies (<code>minimal</code> and <code>generous</code>) on your specific trace to find the theoretical maximum latency (<code>L_max</code>) and maximum cost (<code>C_max</code>). This dynamically maps both latency and cost to a strict <code>[0, 1]</code> scale, regardless of the workload size.
        </Rationale>

        <Rationale title="Why strict min(..., 1.0) clipping?">
          If an optimizer explores a catastrophically bad policy (e.g., allowing only 1 container for 1000 RPS), the latency explodes to infinity. Without clipping, this massive penalty dominates the surrogate models in Bayesian optimizers, ruining their ability to explore effectively. Clipping ensures the objective never exceeds 1.0.
        </Rationale>
      </Section>

      <Section
        num="3"
        title="Workload Generation Mathematics"
        intro="How the synthetic and historical traces are mathematically generated."
      >
        <p><strong>1. Synthetic Poisson (NHPP)</strong></p>
        <CodeBlock code={`λ(t) = λ_base · (1 + 0.6 · sin(2π · t / 24h - π/2))`} />
        
        <Rationale title="Why amplitude 0.6 and Zipf(1.2)?">
          The 0.6 amplitude creates a realistic peak-to-trough ratio (approx 4:1) that matches typical daily human-driven traffic patterns observed in cloud providers. Function popularity follows a Zipf distribution (<code>s = 1.2</code>) because serverless invocations are heavily skewed (a few functions get 90% of traffic); 1.2 is the exact empirical parameter observed in the Azure 2019 dataset.
        </Rationale>

        <p className="mt-4"><strong>2. Historical Flow (Conditional RealNVP)</strong></p>
        <p>
          A normalizing flow trained on Azure 2019 day 1 data (262,490 rows, 200 functions).
        </p>
        
        <Rationale title="Why uniform dequantization U(-0.49, 0.49)?">
          The target space is <code>(log1p(count), log1p(avg_exec_time_ms))</code>. Azure logs round execution times to the nearest millisecond (or tens of ms). This discreteness causes infinite density spikes in continuous normalizing flows, failing the KS-gate test. Adding uniform noise <code>U(-0.49, 0.49)</code> before the log transform smooths the distribution perfectly.
        </Rationale>

        <Rationale title="Why scalar rank_norm instead of one-hot encoding?">
          The condition space is <code>(function_rank_norm, sin(hour), cos(hour))</code>. Encoding 200 functions would require a 200-dimensional one-hot vector, which is too sparse for a lightweight flow model and leads to poor generalization. Rank normalization collapses this to a single continuous dimension, effectively leveraging the Zipfian structure of the data.
        </Rationale>

        <Rationale title="Why Bernoulli thinning / Bootstrap for intensity?">
          To scale intensity, naive linear multiplication (e.g., <code>count * 2</code>) destroys the learned joint distribution <code>p(N, t̄ | rank, hour)</code>. Instead, we use per-arrival Bernoulli thinning (for scale ≤ 1) or bootstrap resampling with ±5s temporal jitter (for scale &gt; 1). This rigorously preserves the exact marginal distributions per-arrival.
        </Rationale>
      </Section>

      <Section
        num="4"
        title="The Search Space (Policy Variables)"
        intro="The dimensions of the vector 'x' your optimizer must output."
      >
        <CodeBlock code={`bounds = [(1.0, 1800.0), (1.0, max_containers_cap), (0.1, 1.0)]`} />
        
        <Rationale title="Why these specific bounds?">
          <ul className="list-disc pl-4 space-y-1 mt-2">
            <li><strong>1800.0s (30 mins) max keep-alive:</strong> Cloud providers typically reap idle containers after 10-20 minutes. 30 minutes is a practical upper bound that captures the long tail of benefits without allowing the optimizer to blow up costs infinitely.</li>
            <li><strong>max_containers_cap (default 30):</strong> Represents a realistic concurrency limit for a single function in a shared tenant environment to prevent starvation of other tenants.</li>
            <li><strong>prewarm_threshold [0.1, 1.0]:</strong> If the ratio of <code>BUSY / max_containers</code> exceeds this threshold, the system proactively spawns a new container. A threshold of 1.0 effectively disables pre-warming.</li>
          </ul>
        </Rationale>
      </Section>

      <Section
        num="5"
        title="Baseline Strategies"
        intro="The 5 hand-crafted policies used for normalization and benchmarking."
      >
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
        
        <Rationale title="Why these specific baselines?">
          <ul className="list-disc pl-4 space-y-1 mt-2">
            <li><strong>minimal & generous:</strong> These act as the absolute boundary anchors. <code>minimal</code> intentionally starves the system to find the worst-case latency (<code>L_max</code>). <code>generous</code> intentionally wastes resources to find the worst-case cost (<code>C_max</code>).</li>
            <li><strong>aggressive (30s):</strong> Tests the hypothesis that aggressively killing containers saves money without hurting p99 too much under high load.</li>
            <li><strong>balanced (300s):</strong> 5 minutes is a standard industry default for keep-alive windows.</li>
            <li><strong>prewarm_heavy (0.6):</strong> Tests if predictive scaling (warming up at 60% capacity) offsets the cost of early initialization.</li>
          </ul>
        </Rationale>
      </Section>

      <Section
        num="6"
        title="Optimizer Contract & Sandbox Limits"
        intro="Execution constraints for your solution.py."
      >
        <CodeBlock code={SIGNATURE} />
        <ul className="list-disc pl-6 space-y-1 mt-4">
          <li><strong>Execution limits:</strong> 120s wall-clock timeout. 60s hard CPU limit (<code>RLIMIT_CPU</code>).</li>
          <li><strong>Memory limits:</strong> 512 MB cap (<code>RLIMIT_AS</code> / <code>RLIMIT_DATA</code> on Linux).</li>
          <li><strong>File descriptors & Processes:</strong> <code>RLIMIT_NOFILE = 64</code>, <code>RLIMIT_NPROC = 32</code>.</li>
        </ul>
        
        <Rationale title="Why these sandbox limits?">
          The 60s CPU limit ensures that infinite loops (e.g., <code>while True: pass</code>) are killed reliably. The 512 MB memory limit prevents memory bombs from crashing the host. <em>Note:</em> On macOS, <code>RLIMIT_AS</code> and <code>RLIMIT_DATA</code> silently no-op due to kernel quirks; the Docker deployment is required to enforce them strictly. Heavy libraries like <code>torch</code> are blocked to enforce the "lightweight optimizer" constraint.
        </Rationale>
      </Section>

      <Section
        num="7"
        title="Report Metrics"
        intro="Mathematical definitions of the output metrics."
      >
        <ul className="list-disc pl-6 space-y-2">
          <li><strong>p99 latency:</strong> The 99th percentile of the empirical latency distribution <code>P(L ≤ l) = 0.99</code>.</li>
          <li><strong>CVaR_0.99:</strong> <code>E[L | L ≥ p99]</code>. The expected value of latency given that it exceeds the 99th percentile.</li>
          <li><strong>Cold Start Rate:</strong> <code>N_cold / N_total</code>. The fraction of requests that were routed to a <code>FREE</code> or recycled container, triggering a cold start.</li>
          <li><strong>Warm Hit Rate:</strong> <code>N_warm / N_total</code>. The fraction of requests routed to an <code>IDLE</code> container.</li>
          <li><strong>p_loss:</strong> <code>N_rejected / N_total</code>. The fraction of requests dropped because all <code>max_containers</code> were <code>BUSY</code> or <code>WARMING_UP</code>.</li>
        </ul>
      </Section>
    </div>
  );
}

function Section({
  num,
  title,
  intro,
  children,
}: {
  num: string;
  title: string;
  intro?: string;
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
      {intro && (
        <p className="font-serif-warm text-[15px] text-ink/70 leading-relaxed -mt-1">
          {intro}
        </p>
      )}
      <div className="font-serif-warm text-[15px] text-ink/85 leading-relaxed space-y-3">
        {children}
      </div>
    </section>
  );
}

function Rationale({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mt-4 mb-2 bg-warm/30 border-l-2 border-ink/30 pl-4 py-3 pr-4 rounded-r-md">
      <strong className="block font-serif-warm text-[14px] text-ink mb-1">{title}</strong>
      <div className="font-serif-warm text-[13.5px] text-ink/80 leading-relaxed">
        {children}
      </div>
    </div>
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
