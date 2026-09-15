import { useState, type ReactNode } from "react";

export function Disclosure({
  title,
  meta,
  children,
  defaultOpen = false,
}: {
  title: string;
  meta?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-border">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-baseline justify-between gap-6 py-5 text-left transition-colors duration-200 hover:text-foreground"
      >
        <span className="text-[17px] tracking-tight">{title}</span>
        <span className="flex shrink-0 items-baseline gap-4">
          {meta && <span className="label-tech hidden sm:inline">{meta}</span>}
          <span className="label-tech w-3 text-right">{open ? "–" : "+"}</span>
        </span>
      </button>
      {open && (
        <div className="max-w-3xl space-y-4 pb-7 text-sm leading-relaxed text-muted-foreground">
          {children}
        </div>
      )}
    </div>
  );
}

export function Section({
  code,
  title,
  children,
}: {
  code: string;
  title?: string;
  children: ReactNode;
}) {
  return (
    <section className="rule-t grid gap-8 py-14 md:grid-cols-[190px_minmax(0,1fr)] md:gap-16 md:py-20">
      <div>
        <div className="label-tech md:sticky md:top-24">{code}</div>
        {title && (
          <h2 className="mt-3 text-2xl tracking-tight md:sticky md:top-32 md:text-[26px]">
            {title}
          </h2>
        )}
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}
