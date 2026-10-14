"""Esiti tecnici e traccia durevole; un successo tecnico non approva il PDF.

Un solo writer per volta: il padre attende il sottoprocesso, poi rilegge il
manifest. Ogni invocazione ha una directory propria, anche nelle riprese.
Il resume per lotto previsto da P7 sta in `comics/traduzione/ai_review.py`
(manifesto dei lotti); qui c'e' la scrittura atomica che lo rende affidabile.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


EXIT_CODES = {"complete": 0, "failed": 1, "partial": 2, "waiting_credit": 3}
ACTIVE_REPORT = ContextVar("comics_report", default=None)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def testo_console(testo):
    """Il testo reso scrivibile sulla console di questo sistema.

    Quello che stampiamo arriva dal modello o dall'OCR e puo' contenere
    qualsiasi cosa - una stella, un trattino lungo, un accento. La console di
    Windows e' cp1252: senza questa rete un carattere che non sa scrivere fa
    cadere la fase. Un resoconto non deve mai poter fermare il lavoro che
    racconta.
    """
    codifica = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(testo).encode(codifica, "replace").decode(codifica, "replace")


def scrivi_atomico(path, testo):
    """Scrive tutto o niente: un crash a meta' non lascia un file troncato.

    Il temporaneo sta nella stessa cartella perche' `os.replace` e' atomico
    solo dentro lo stesso filesystem, e i dati vengono forzati su disco PRIMA
    dello scambio: senza `fsync` il rename puo' arrivare al disco prima del
    contenuto, e dopo un crash resta un file valido ma vuoto. E' la differenza
    fra riprendere da un lotto salvato e rifarlo.
    """
    path = Path(path)
    temporaneo = path.with_name(path.name + ".tmp")
    with temporaneo.open("w", encoding="utf-8", newline="") as handle:
        handle.write(testo)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporaneo, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def issue(code, reason, pages=(), ids=()):
    return {"code": code, "reason": reason, "pages": list(pages), "ids": list(ids)}


@dataclass
class PhaseResult:
    name: str
    requested: int
    completed: int = 0
    unit: str = "batches"
    issues: list = field(default_factory=list)
    pages: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    repairs: list = field(default_factory=list)

    @property
    def failed(self):
        return self.requested - self.completed

    @property
    def status(self):
        if self.completed == self.requested and not self.issues:
            return "complete"
        return "partial" if self.completed else "failed"

    @property
    def complete(self):
        return self.status == "complete"

    def to_dict(self):
        return {**asdict(self), "failed": self.failed, "status": self.status}


def code_provenance(root):
    root = Path(root)
    # Include untracked effective source/tests, not generated session files.
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in {".py", ".toml"}
        and not {".git", ".venv", "lavoro", "tmp", "__pycache__"}.intersection(
            p.relative_to(root).parts
        )
    )
    paths += sorted(root.glob("requirements*.txt"))
    digest, files = hashlib.sha256(), {}
    for path in sorted(set(paths)):
        name = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + content + b"\0")
        files[name] = hashlib.sha256(content).hexdigest()
    def git(*args):
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=10,
            )
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    versions = {"python": platform.python_version()}
    for package in ("PyMuPDF", "Pillow", "rapidocr-onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "effective_sha256": digest.hexdigest(), "files": files, "versions": versions,
    }


class RunReport:
    def __init__(self, path, data):
        self.path, self.data = Path(path), data

    @classmethod
    def create(cls, directory, config, root):
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        path = Path(directory) / "runs" / run_id / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=False)
        report = cls(path, {
            "schema": 1, "run_id": run_id, "started_at": utc_now(),
            "status": "running", "quality": "draft_unverified",
            "coverage": {"denominator": None, "verified": False},
            "config": config, "provenance": code_provenance(root),
            "phases": {}, "issues": [], "artifacts": {},
        })
        report.save()
        report.event("run_started")
        return report

    @classmethod
    def load(cls, path):
        return cls(path, json.loads(Path(path).read_text(encoding="utf-8")))

    def reload(self):
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self):
        self.data["updated_at"] = utc_now()
        scrivi_atomico(self.path, json.dumps(
            self.data, ensure_ascii=False, indent=2, default=str))

    def event(self, kind, **details):
        entry = {"time": utc_now(), "event": kind, **details}
        with self.path.with_name("events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def start_phase(self, name, requested, unit="batches"):
        self.data["phases"][name] = {
            "requested": requested, "completed": 0, "failed": 0, "unit": unit,
            "status": "running", "started_at": utc_now(), "issues": [],
        }
        self.save()
        self.event("phase_started", phase=name, requested=requested, unit=unit)

    def finish_phase(self, result, final=True):
        old = self.data["phases"].get(result.name, {})
        self.data["phases"][result.name] = {
            **old, **result.to_dict(),
        }
        if final:
            self.data["phases"][result.name]["finished_at"] = utc_now()
        else:
            self.data["phases"][result.name].update(status="running", failed=len(result.issues))
        self.save()
        self.event("phase_finished" if final else "phase_progress",
                   **self.data["phases"][result.name])

    def artifact(self, name, path, **details):
        self.data["artifacts"][name] = {
            "path": str(Path(path).resolve()), "sha256": sha256(path),
            "bytes": Path(path).stat().st_size, **details,
        }
        self.save()

    def wait_credit(self, waiting):
        self.data["status"] = "waiting_credit" if waiting else "running"
        self.save()

    def finish(self, error=None):
        if error:
            self.data["issues"].append(issue("execution_failed", str(error)))
        unfinished = [
            p for p in self.data["phases"].values() if p["status"] == "running"
        ]
        for phase in unfinished:
            phase.update(status="failed", failed=phase["requested"] - phase["completed"])
        phases = list(self.data["phases"].values())
        failed = any(p["status"] == "failed" for p in phases)
        if error or unfinished or (failed and "output_pdf" not in self.data["artifacts"]):
            status = "failed"
        elif self.data["issues"] or any(p["status"] != "complete" for p in phases):
            status = "partial"
        else:
            status = "complete"
        self.data.update(status=status, finished_at=utc_now(), exit_code=EXIT_CODES[status])
        self.save()
        self.event("run_finished", status=status)
        return EXIT_CODES[status]


def event(kind, **details):
    report = ACTIVE_REPORT.get()
    if report:
        report.event(kind, **details)


def start_phase(name, requested, unit="batches"):
    report = ACTIVE_REPORT.get()
    if report:
        report.start_phase(name, requested, unit)


def finish_phase(result, final=True):
    report = ACTIVE_REPORT.get()
    if report:
        report.finish_phase(result, final=final)


def plan_index(plan, expected_pages=None):
    """Niente collassi silenziosi nei dict per id duplicati o pagine assenti."""
    if not isinstance(plan, dict) or not isinstance(plan.get("pagine"), list):
        raise ValueError("manca la lista pagine")
    pages, ids = {}, set()
    for page in plan["pagine"]:
        if not isinstance(page, dict) or type(page.get("numero")) is not int:
            raise ValueError("numero pagina non valido")
        number = page["numero"]
        if number < 1 or number in pages or page.get("indice", number - 1) != number - 1:
            raise ValueError("pagina duplicata o indice incoerente: %s" % number)
        if not isinstance(page.get("blocchi"), list):
            raise ValueError("lista blocchi mancante: p%d" % number)
        blocks = {}
        for block in page["blocchi"]:
            bid = block.get("id") if isinstance(block, dict) else None
            if not isinstance(bid, str) or not bid or bid in ids:
                raise ValueError("id mancante o duplicato: %s" % bid)
            ids.add(bid)
            blocks[bid] = block
        pages[number] = blocks
    if expected_pages is not None and set(pages) != set(expected_pages):
        raise ValueError("pagine attese %s, ricevute %s" % (sorted(expected_pages), sorted(pages)))
    return pages


def record_coverage(plan, render_only=False):
    report = ACTIVE_REPORT.get()
    if not report:
        return
    skipped = []
    for page in plan["pagine"]:
        for block in page["blocchi"]:
            if not block.get("salta"):
                continue
            render = block.get("_render") or {}
            reason = next((key for key in (
                "unreviewed", "source_conflict", "page_protected", "compound", "contenuto_in",
                "scartato_come_disegno",
            ) if render.get(key)), "ai_skip_unclassified")
            skipped.append(issue(reason, "blocco preservato, non certificato decorativo",
                                 [page["numero"]], [block["id"]]))
    observed = [r for p in plan["pagine"] for r in p.get("sorgente", {}).get("righe", [])]
    unresolved = [r["id"] for r in observed if r["stato"] == "esclusa" and r.get("motivo") != "duplicate"]
    report.data["coverage"].update(
        plan_blocks=sum(len(p["blocchi"]) for p in plan["pagine"]), skipped=skipped,
        observed_lines=len(observed), unresolved_source_lines=unresolved,
    )
    if skipped and not render_only:
        report.data["issues"].append(issue(
            "coverage_unresolved", "%d blocchi saltati da classificare" % len(skipped),
            sorted({p for s in skipped for p in s["pages"]}),
            [bid for s in skipped for bid in s["ids"]],
        ))
    report.save()


def cli_entry(main):
    """Registra anche un'eccezione, senza trasformarla in un falso successo."""
    try:
        return main()
    except BaseException as exc:
        report = ACTIVE_REPORT.get()
        if report and (not isinstance(exc, SystemExit) or exc.code not in (None, 0)):
            report.reload()
            report.finish(error=exc)
        raise
    finally:
        ACTIVE_REPORT.set(None)


def timed_call_usage(proc):
    """Token dichiarati dalla CLI, non costi o token unici; nessun testo privato."""
    totals = {}
    for line in (proc.stdout or "").splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        if value.get("type") in ("turn.completed", "result"):
            usage = value.get("usage") or {}
            for key, amount in usage.items():
                if isinstance(amount, int) and not isinstance(amount, bool):
                    totals[key] = totals.get(key, 0) + amount
    return totals


def clock():
    return time.monotonic()
