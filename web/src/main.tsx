// The "Run jobs" part of the team's dashboard as a page of its own: one route, /run.
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Link, Outlet, RouterProvider, createRootRoute, createRoute, createRouter, redirect } from "@tanstack/react-router";
import "./styles.css";
import { RunView, isRunJob, type RunJob } from "./views/Run";

type Theme = "system" | "light" | "dark";
function readTheme(): Theme {
  try {
    const t = localStorage.getItem("dh-theme");
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}
function ThemeSwitch() {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") delete root.dataset.theme;
    else root.dataset.theme = theme;
    try {
      if (theme === "system") localStorage.removeItem("dh-theme");
      else localStorage.setItem("dh-theme", theme);
    } catch {
      /* storage blocked: the choice lasts for this visit */
    }
  }, [theme]);
  return (
    <div className="theme-switch" role="group" aria-label="Colour theme">
      {(["light", "system", "dark"] as Theme[]).map((t) => (
        <button key={t} type="button" aria-pressed={theme === t} onClick={() => setTheme(t)}>
          {t === "system" ? "Auto" : t === "light" ? "Light" : "Dark"}
        </button>
      ))}
    </div>
  );
}

function Glyph() {
  return (
    <svg className="mark-glyph" viewBox="0 0 26 26" aria-hidden="true">
      <rect x="0" y="0" width="26" height="26" rx="7" fill="var(--ink)" />
      <rect x="5" y="6" width="7.5" height="14" rx="1.6" fill="var(--on-ink)" />
      <circle cx="8.75" cy="9.6" r="1.25" fill="var(--ink)" />
      <circle cx="8.75" cy="13" r="1.25" fill="var(--ink)" />
      <circle cx="8.75" cy="16.4" r="1.25" fill="var(--ink)" />
      <path d="M12.5 13H16" stroke="var(--heat-now)" strokeWidth="2.6" />
      <rect x="16" y="7.5" width="5" height="12.5" rx="2.5" fill="var(--heat-tank)" />
    </svg>
  );
}

function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="top" data-open={open ? "true" : "false"}>
      <div className="top-inner">
        <Link to="/run" className="mark" aria-label="Heat from computing, run the jobs">
          <Glyph />
          <span className="mark-name">Heat from computing</span>
        </Link>
        <button type="button" className="menu-btn" aria-expanded={open} aria-controls="site-nav" onClick={() => setOpen((v) => !v)}>
          {open ? "Close" : "Menu"}
        </button>
        <nav className="nav" id="site-nav" aria-label="Sections">
          <Link to="/run" onClick={() => setOpen(false)}>Run jobs</Link>
        </nav>
        <ThemeSwitch />
      </div>
    </header>
  );
}

const rootRoute = createRootRoute({
  component: () => (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <Header />
      <main id="main" tabIndex={-1}>
        <Outlet />
      </main>
      <footer className="foot">
        <div className="foot-inner">
          <p>Team Cordon Vert, sprintd Basel GreenTech. Case: DigitalHeat, “Useful load, useful heat”. This is the “Run jobs” part of the team's dashboard.</p>
        </div>
      </footer>
    </>
  ),
  notFoundComponent: () => <p>This page does not exist. <Link to="/run">Go to Run jobs</Link>.</p>,
});

const routes = [
  createRoute({ getParentRoute: () => rootRoute, path: "/", beforeLoad: () => { throw redirect({ to: "/run" }); } }),
  createRoute({
    getParentRoute: () => rootRoute, path: "/run", component: RunView,
    validateSearch: (s: Record<string, unknown>): { job?: RunJob } => (isRunJob(s.job) ? { job: s.job } : {}),
  }),
];

const router = createRouter({ routeTree: rootRoute.addChildren(routes), defaultPreload: "intent", scrollRestoration: true });
declare module "@tanstack/react-router" {
  interface Register { router: typeof router }
}

const queryClient = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false } } });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
