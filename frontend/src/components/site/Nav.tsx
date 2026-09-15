import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

const LINKS = [
  { to: "/benchmark", label: "Benchmark" },
  { to: "/connectome", label: "Connectome" },
  { to: "/methodology", label: "Methodology" },
] as const;

export function Nav() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/88 backdrop-blur-sm">
      <div className="page-x mx-auto flex h-14 max-w-[1600px] items-center justify-between">
        <div className="flex items-center gap-8">
          <Link to="/" className="group flex items-baseline gap-2" onClick={() => setOpen(false)}>
            <span className="text-[15px] font-medium tracking-[0.18em]">ABB</span>
            <span className="label-tech hidden sm:inline">Animal Brain Benchmark</span>
          </Link>
        </div>

        <nav className="hidden items-center gap-7 md:flex">
          {LINKS.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className="label-tech transition-colors duration-200 hover:text-foreground"
              activeProps={{ className: "!text-foreground" }}
            >
              {l.label}
            </Link>
          ))}
          <span className="h-4 w-px bg-border" />
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="label-tech transition-colors duration-200 hover:text-foreground"
          >
            GitHub
          </a>
        </nav>

        <button
          type="button"
          aria-label="Toggle navigation"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="label-tech border border-border px-3 py-1.5 transition-colors duration-200 hover:border-border-strong hover:text-foreground md:hidden"
        >
          {open ? "Close" : "Menu"}
        </button>
      </div>

      {open && (
        <div className="page-x border-t border-border pb-6 pt-4 md:hidden">
          <div className="flex flex-col gap-4">
            {LINKS.map((l) => (
              <Link
                key={l.to}
                to={l.to}
                onClick={() => setOpen(false)}
                className="text-lg tracking-tight text-muted-foreground"
                activeProps={{ className: "!text-foreground" }}
              >
                {l.label}
              </Link>
            ))}
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="text-lg tracking-tight text-muted-foreground"
            >
              GitHub
            </a>
          </div>
        </div>
      )}
    </header>
  );
}

export function Footer() {
  return (
    <footer className="rule-t mt-28">
      <div className="page-x mx-auto grid max-w-[1600px] gap-10 py-14 md:grid-cols-[1.4fr_1fr_1fr]">
        <div>
          <div className="text-[15px] font-medium tracking-[0.18em]">ABB</div>
          <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted-foreground">
            Animal Brain Benchmark evaluates connectome-topology-constrained neural architectures
            derived from the MaleCNS v1.0 <em>Drosophila</em> connectome against conventional
            artificial neural architectures.
          </p>
        </div>
        <div>
          <div className="label-tech">Sections</div>
          <div className="mt-4 flex flex-col gap-2 text-sm text-muted-foreground">
            <Link to="/benchmark" className="hover:text-foreground">
              Benchmark
            </Link>
            <Link to="/connectome" className="hover:text-foreground">
              Connectome explorer
            </Link>
            <Link to="/methodology" className="hover:text-foreground">
              Methodology
            </Link>
          </div>
        </div>
        <div>
          <div className="label-tech">Scope</div>
          <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
            All models are computational. ABB does not recreate a living animal, a complete nervous
            system, or consciousness.
          </p>
        </div>
      </div>
    </footer>
  );
}
