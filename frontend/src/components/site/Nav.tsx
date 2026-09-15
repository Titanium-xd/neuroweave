import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

function GithubIcon() {
  return (
    <svg
      height={16} width={16} viewBox="0 0 16 16" aria-hidden="true"
      fill="currentColor" className="inline-block align-[-2px]"
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

const LINKS = [
  { to: "/arena",         label: "Live Arena" },
  { to: "/how-it-works",  label: "How It Works" },
  { to: "/benchmark",     label: "Benchmark" },
  { to: "/connectome",    label: "Connectome" },
  { to: "/methodology",   label: "Methodology" },
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
            <span className="text-[15px] font-medium tracking-[0.18em]">NeuroWeave</span>
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
            href="https://github.com/Titanium-xd/neuroweave"
            target="_blank"
            rel="noreferrer"
            className="label-tech flex items-center gap-1.5 transition-colors duration-200 hover:text-foreground"
          >
            <GithubIcon />
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
              href="https://github.com/Titanium-xd/neuroweave"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 text-lg tracking-tight text-muted-foreground"
            >
              <GithubIcon />
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
      <div className="page-x mx-auto grid max-w-[1600px] gap-10 py-14 md:grid-cols-[1.4fr_1fr_1fr_1fr]">
        <div>
          <div className="text-[15px] font-medium tracking-[0.18em]">NeuroWeave</div>
          <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted-foreground">
            NeuroWeave evaluates connectome-topology-constrained neural architectures
            derived from the MaleCNS v1.0 <em>Drosophila</em> connectome against conventional
            artificial neural architectures.
          </p>
        </div>
        <div>
          <div className="label-tech">Sections</div>
          <div className="mt-4 flex flex-col gap-2 text-sm text-muted-foreground">
            <Link to="/arena" className="hover:text-foreground">
              Live Arena
            </Link>
            <Link to="/how-it-works" className="hover:text-foreground">
              How It Works
            </Link>
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
            All models are computational. NeuroWeave does not recreate a living animal,
            a complete nervous system, or consciousness.
          </p>
        </div>
        <div>
          <div className="label-tech">Case Study by</div>
          <div className="mt-4 space-y-1">
            <div className="text-sm font-medium">Parva Trivedi</div>
            <div className="mt-3 flex flex-col gap-2 text-sm text-muted-foreground">
              <a
                href="https://github.com/Titanium-xd"
                target="_blank" rel="noreferrer"
                className="flex items-center gap-1.5 hover:text-foreground"
              >
                <GithubIcon />
                github.com/Titanium-xd
              </a>
              <a
                href="https://discord.com/users/titanium_xd"
                target="_blank" rel="noreferrer"
                className="hover:text-foreground"
              >
                Discord — titanium_xd
              </a>
              <a
                href="https://linkedin.com/in/parvatrivedi"
                target="_blank" rel="noreferrer"
                className="hover:text-foreground"
              >
                LinkedIn — Parva Trivedi
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
