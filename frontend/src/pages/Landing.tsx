import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="flex flex-col gap-12">
      {/* Hero */}
      <section className="pt-6 pb-12 md:pt-8 md:pb-16 border-b border-ink/15">
        <div className="flex flex-col gap-6 max-w-3xl">
          <p className="font-serif-warm text-[13px] uppercase tracking-[0.2em] text-muted">
            black-box benchmark · serverless scheduling
          </p>
          <h1 className="font-serif-warm text-4xl md:text-5xl leading-[1.05] text-ink">
            Optimize scheduling policies
            <br />
            under a calibrated simulator.
          </h1>
          <p className="max-w-2xl text-[17px] leading-relaxed text-ink/80 font-serif-warm">
            Upload a Python{" "}
            <code className="font-mono text-[14px] bg-warm/60 px-1.5 py-0.5">
              optimize()
            </code>{" "}
            function, pick a workload source, and run your policy against a
            G/G/k/k simulator with cold-start phases calibrated on 805 745 real
            Huawei events.
          </p>
          <div className="flex flex-wrap gap-3 pt-1">
            <Link
              to="/new"
              className="font-serif-warm text-[15px] px-5 py-2.5 bg-ink text-paper hover:bg-ink/85 transition"
            >
              Run a benchmark
            </Link>
            <Link
              to="/docs"
              className="font-serif-warm text-[15px] px-5 py-2.5 border border-ink/30 hover:bg-warm/40 transition"
            >
              How it works
            </Link>
          </div>
        </div>
      </section>

      {/* Three feature cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card
          title="Black-box objective"
          body="Your optimizer sees f(x)→float, nothing more. We evaluate it inside a discrete-event simulator with four-state FSM containers."
        />
        <Card
          title="Three data sources"
          body="Synthetic non-homogeneous Poisson with diurnal bursts, a conditional RealNVP flow trained on Azure 2019, or your own CSV."
        />
        <Card
          title="Compared to baselines"
          body="Every run is benchmarked against five named policies and a live Pareto front of the points you've already visited."
        />
      </section>
    </div>
  );
}

function Card({ title, body }: { title: string; body: string }) {
  return (
    <div className="border border-ink/15 p-6">
      <h3 className="font-serif-warm text-[19px] text-ink mb-2 leading-snug">
        {title}
      </h3>
      <p className="font-serif-warm text-[15px] leading-relaxed text-ink/75">
        {body}
      </p>
    </div>
  );
}
