import { useState, type ReactNode } from "react";

/**
 * A small "(?)" pill with a hover/focus tooltip. No portal gymnastics,
 * no libraries — just a positioned div. Good for inline glossary help
 * next to form labels and metric pills.
 */
export default function HelpIcon({
  children,
  side = "top",
}: {
  children: ReactNode;
  side?: "top" | "bottom";
}) {
  const [open, setOpen] = useState(false);
  const pos =
    side === "top"
      ? "bottom-full mb-2 left-1/2 -translate-x-1/2"
      : "top-full mt-2 left-1/2 -translate-x-1/2";
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label="More info"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          e.preventDefault();
          setOpen((v) => !v);
        }}
        className="inline-flex items-center justify-center w-4 h-4 rounded-full border border-ink/30 text-[9px] font-serif-warm text-ink/70 hover:bg-warm hover:text-ink transition ml-1 align-middle"
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className={`absolute z-50 ${pos} w-64 text-left font-serif-warm text-[12px] leading-snug bg-ink text-paper rounded-lg px-3 py-2 shadow-lg pointer-events-none`}
        >
          {children}
        </span>
      )}
    </span>
  );
}
