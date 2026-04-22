import { Link, Route, Routes } from "react-router-dom";
import Landing from "./pages/Landing";
import Upload from "./pages/Upload";
import Running from "./pages/Running";
import Report from "./pages/Report";
import Docs from "./pages/Docs";

function NavLink({ to, label }: { to: string; label: string }) {
  return (
    <Link
      to={to}
      className="font-serif-warm text-[14px] text-ink/80 hover:text-ink transition"
    >
      {label}
    </Link>
  );
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-ink/15">
        <div className="max-w-5xl mx-auto flex items-center justify-between px-6 py-4">
          <Link to="/" className="font-serif-warm text-[17px] text-ink">
            serverless.bb
          </Link>
          <nav className="flex gap-6">
            <NavLink to="/" label="Home" />
            <NavLink to="/new" label="Run" />
            <NavLink to="/docs" label="Docs" />
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/new" element={<Upload />} />
          <Route path="/runs/:runId" element={<Running />} />
          <Route path="/runs/:runId/report" element={<Report />} />
          <Route path="/docs" element={<Docs />} />
        </Routes>
      </main>
      <footer className="border-t border-ink/15 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-4 font-serif-warm text-[13px] text-muted">
          <a
            href="https://t.me/elmowx"
            target="_blank"
            rel="noreferrer"
            className="hover:text-ink transition"
          >
            @elmowx
          </a>
        </div>
      </footer>
    </div>
  );
}
