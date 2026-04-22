import type { ContainerState } from "../types";

/**
 * Each pose is a 16×32 sprite (2 vertical cells on the 16×16 sheet grid).
 * `row` is the TOP cell of the pose; the renderer draws 2 rows tall.
 *
 * Calibrated visually against cat1.png (352×1696):
 *   - row 0  (6 fr): sitting REST              — IDLE_ANIM default
 *   - row 28 (2 fr): sphinx SLEEP (lying)      — FREE
 *   - row 44 (8 fr): WALK-RIGHT side walker    — BUSY
 *   - row 64 (8 fr): WASH sit / grooming       — IDLE
 *   - row 66 (8 fr): YAWN stand / stretch      — WARMING_UP
 */
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
