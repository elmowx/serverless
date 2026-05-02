import { useId } from "react";

export type CatColor = 1 | 2 | 3;

export interface CatSpriteProps {
  color: CatColor;
  row: number;
  frames: number;
  fps?: number;
  scale?: number;
  flip?: boolean;
  className?: string;
}

const CELL = 16;
const SPRITE = 32;
const COLS = 22;
const ROWS = 106;

export default function CatSprite({
  color,
  row,
  frames,
  fps = 6,
  scale = 3,
  flip = false,
  className = "",
}: CatSpriteProps) {
  const id = useId().replace(/:/g, "_");
  const width = SPRITE * scale;
  const height = SPRITE * scale;
  const sheetW = COLS * CELL * scale;
  const sheetH = ROWS * CELL * scale;
  const animName = `cat-${id}-r${row}-f${frames}`;
  const duration = frames / fps;
  const endX = -frames * SPRITE * scale;
  const yOffset = -row * CELL * scale;

  return (
    <>
      <style>{`
        @keyframes ${animName} {
          from { background-position: 0px ${yOffset}px; }
          to   { background-position: ${endX}px ${yOffset}px; }
        }
      `}</style>
      <div
        className={`cat-sprite ${className}`}
        style={{
          width,
          height,
          backgroundImage: `url(/sprites/cats/cat${color}.png)`,
          backgroundSize: `${sheetW}px ${sheetH}px`,
          backgroundPosition: `0px ${yOffset}px`,
          backgroundRepeat: "no-repeat",
          imageRendering: "pixelated",
          transform: flip ? "scaleX(-1)" : undefined,
          animation: `${animName} ${duration}s steps(${frames}) infinite`,
        }}
      />
    </>
  );
}
