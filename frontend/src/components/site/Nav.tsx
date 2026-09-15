import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

function GithubIcon() {
  return (
    <svg height={16} width={16} viewBox="0 0 16 16" aria-label="GitHub" fill="currentColor"
      className="inline-block align-[-2px] shrink-0">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function DiscordIcon() {
  return (
    <svg height={16} width={16} viewBox="0 0 24 24" aria-label="Discord" fill="currentColor"
      className="inline-block align-[-2px] shrink-0">
      <path d="M20.317 4.492c-1.53-.69-3.17-1.2-4.885-1.49a.075.075 0 0 0-.079.036c-.21.369-.444.85-.608 1.23a18.566 18.566 0 0 0-5.487 0 12.36 12.36 0 0 0-.617-1.23A.077.077 0 0 0 8.562 3c-1.714.29-3.354.8-4.885 1.491a.07.07 0 0 0-.032.027C.533 9.093-.32 13.555.099 17.961a.08.08 0 0 0 .031.055 20.03 20.03 0 0 0 5.993 2.98.078.078 0 0 0 .084-.026c.462-.62.874-1.275 1.226-1.963.021-.04.001-.088-.041-.104a13.201 13.201 0 0 1-1.872-.878.075.075 0 0 1-.008-.125c.126-.093.252-.19.372-.287a.075.075 0 0 1 .078-.01c3.927 1.764 8.18 1.764 12.061 0a.075.075 0 0 1 .079.009c.12.098.245.195.372.288a.075.075 0 0 1-.006.125c-.598.344-1.22.635-1.873.877a.075.075 0 0 0-.041.105c.36.687.772 1.341 1.225 1.962a.077.077 0 0 0 .084.028 19.963 19.963 0 0 0 6.002-2.981.076.076 0 0 0 .032-.054c.5-5.094-.838-9.52-3.549-13.442a.06.06 0 0 0-.031-.028zM8.02 15.278c-1.182 0-2.157-1.069-2.157-2.38 0-1.312.956-2.38 2.157-2.38 1.21 0 2.176 1.077 2.157 2.38 0 1.312-.956 2.38-2.157 2.38zm7.975 0c-1.183 0-2.157-1.069-2.157-2.38 0-1.312.955-2.38 2.157-2.38 1.21 0 2.176 1.077 2.157 2.38 0 1.312-.946 2.38-2.157 2.38z" />
    </svg>
  );
}

function LinkedInIcon() {
  return (
    <svg height={16} width={16} viewBox="0 0 24 24" aria-label="LinkedIn" fill="currentColor"
      className="inline-block align-[-2px] shrink-0">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
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
          <Link to="/" className="group flex items-center gap-0" onClick={() => setOpen(false)}>
            <span
              className="text-[17px] font-semibold tracking-[-0.01em]"
              style={{ fontFamily: "var(--font-sans)" }}
            >Neuro</span><span
              className="text-[18px] font-normal italic tracking-[-0.01em]"
              style={{ fontFamily: "var(--font-serif)" }}
            >Weave</span>
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
            <div className="mt-3 flex items-center gap-4 text-muted-foreground">
              <a
                href="https://github.com/Titanium-xd"
                target="_blank" rel="noreferrer"
                title="GitHub — Titanium-xd"
                className="transition-colors hover:text-foreground"
              >
                <GithubIcon />
              </a>
              <a
                href="https://discord.com/users/titanium.dc"
                target="_blank" rel="noreferrer"
                title="Discord — titanium.dc"
                className="transition-colors hover:text-foreground"
              >
                <DiscordIcon />
              </a>
              <a
                href="https://www.linkedin.com/in/parva-trivedi/"
                target="_blank" rel="noreferrer"
                title="LinkedIn — Parva Trivedi"
                className="transition-colors hover:text-foreground"
              >
                <LinkedInIcon />
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
