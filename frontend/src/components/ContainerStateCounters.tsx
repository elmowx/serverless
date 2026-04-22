import { motion, AnimatePresence } from "framer-motion";
import CatSprite, { type CatColor } from "./CatSprite";
import { STATE_ANIM } from "./catAnimations";
import type { ContainerState, ContainerSummary } from "../types";

/**
 * Live counter per FSM state. Each card shows one pixel cat (posed for
 * that state) and a big number: the time-weighted count of containers
 * in that state during the most recent trial. Replaces the old
 * PixelCatWorkers display (one sprite per container), which didn't scale
 * beyond ~10 containers and didn't communicate the state distribution.
 */

type StateMeta = {
  state: ContainerState;
  label: string;
  description: string;
  color: CatColor;
};

const STATES: StateMeta[] = [
  {
    state: "FREE",
    label: "Free",
    description: "slots not allocated to any function",
    color: 1,
  },
  {
    state: "WARMING_UP",
    label: "Warming up",
    description: "spinning up — env, code, runtime, init",
    color: 2,
  },
  {
    state: "BUSY",
    label: "Busy",
    description: "executing a request right now",
    color: 3,
  },
  {
    state: "IDLE",
    label: "Idle",
    description: "warm, kept-alive, waiting for a hit",
    color: 2,
  },
];

interface Counts {
  FREE: number;
  WARMING_UP: number;
  BUSY: number;
  IDLE: number;
}

function computeCounts(summary: ContainerSummary[] | undefined): Counts {
  const out: Counts = { FREE: 0, WARMING_UP: 0, BUSY: 0, IDLE: 0 };
  if (!summary) return out;
  for (const cs of summary) {
    out.FREE += Math.max(0, cs.free_frac);
    out.WARMING_UP += Math.max(0, cs.warming_frac);
    out.BUSY += Math.max(0, cs.busy_frac);
    out.IDLE += Math.max(0, cs.idle_frac);
  }
  return out;
}

function formatCount(v: number): string {
  if (v <= 0) return "0";
  if (v < 0.5) return "·";
  return Math.round(v).toString();
}

export default function ContainerStateCounters({
  summary,
}: {
  summary: ContainerSummary[] | undefined;
}) {
  const counts = computeCounts(summary);
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {STATES.map((meta) => (
        <StateCard
          key={meta.state}
          meta={meta}
          value={counts[meta.state]}
          empty={!summary || summary.length === 0}
        />
      ))}
    </div>
  );
}

function StateCard({
  meta,
  value,
  empty,
}: {
  meta: StateMeta;
  value: number;
  empty: boolean;
}) {
  const anim = STATE_ANIM[meta.state];
  const display = empty ? "—" : formatCount(value);
  return (
    <div className="border border-ink/15 rounded-xl bg-paper/90 p-3 flex items-center gap-3">
      <div className="shrink-0 bg-warm/40 rounded-lg w-16 h-16 flex items-center justify-center overflow-hidden">
        <CatSprite
          color={meta.color}
          row={anim.row}
          frames={anim.frames}
          fps={anim.fps}
          scale={2}
        />
      </div>
      <div className="flex flex-col min-w-0">
        <div className="font-serif-warm text-[9px] uppercase tracking-wider text-muted">
          {meta.label}
        </div>
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.div
            key={display}
            className="font-serif-warm text-2xl leading-none mt-1"
            initial={{ y: -8, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 8, opacity: 0 }}
            transition={{ type: "spring", stiffness: 360, damping: 24 }}
          >
            {display}
          </motion.div>
        </AnimatePresence>
        <div
          className="text-[11px] text-ink/60 mt-1 truncate"
          title={meta.description}
        >
          {meta.description}
        </div>
      </div>
    </div>
  );
}
