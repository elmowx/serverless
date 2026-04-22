import { useEffect, useRef, useState } from "react";
import { runEventsUrl } from "../api/client";
import type { RunStatus, TrialEvent } from "../types";

interface DoneEvent {
  status: RunStatus;
  error: string | null;
  exit_code: number | null;
}

export interface StreamState {
  trials: TrialEvent[];
  done: DoneEvent | null;
  connected: boolean;
  error: string | null;
}

export function useRunStream(runId: string | undefined): StreamState {
  const [trials, setTrials] = useState<TrialEvent[]>([]);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!runId) return;
    const es = new EventSource(runEventsUrl(runId));
    esRef.current = es;
    es.onopen = () => setConnected(true);
    es.onerror = () => {
      setConnected(false);
    };
    es.addEventListener("trial", (e) => {
      try {
        const ev = JSON.parse((e as MessageEvent).data) as TrialEvent;
        setTrials((prev) => [...prev, ev]);
      } catch (err) {
        setError(String(err));
      }
    });
    es.addEventListener("done", (e) => {
      try {
        const payload = JSON.parse((e as MessageEvent).data) as DoneEvent;
        setDone(payload);
      } catch (err) {
        setError(String(err));
      } finally {
        es.close();
        esRef.current = null;
      }
    });
    return () => {
      es.close();
      esRef.current = null;
    };
  }, [runId]);

  return { trials, done, connected, error };
}
