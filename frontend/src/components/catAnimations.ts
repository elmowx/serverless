import type { ContainerState } from "../types";

export interface CatAnim {
  row: number;
  frames: number;
  fps: number;
}

export const STATE_ANIM: Record<ContainerState, CatAnim> = {
  FREE: { row: 28, frames: 2, fps: 1 },
  WARMING_UP: { row: 66, frames: 8, fps: 6 },
  BUSY: { row: 44, frames: 8, fps: 10 },
  IDLE: { row: 64, frames: 8, fps: 4 },
};

export const IDLE_ANIM: CatAnim = { row: 0, frames: 6, fps: 3 };
