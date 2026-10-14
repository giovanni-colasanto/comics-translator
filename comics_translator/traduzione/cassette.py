"""Cassette delle risposte del modello: una run reale le registra, i test le riproducono.

Meccanismo fail-closed: in modalita' replay una chiamata senza cassette e' un
errore duro, mai una chiamata reale di ripiego — consumare crediti per sbaglio
e' esattamente cio' che il meccanismo deve rendere impossibile. La chiave di
ogni cassette copre tutto cio' che il modello ha davvero visto (modello, prompt,
pixel delle immagini): il prompt da solo nomina i file ma non li contiene.

Il registro copre traduzione e QA visuale perche' entrambe passano da
`ai_review.run_agent`; non copre cattura ne' OCR, che restano locali. Il corpus
vive in `corpus/<sessione>/`, fuori dalla cartella volatile `lavoro/`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

SCHEMA = 1

# Stato del processo: una sola modalita' per esecuzione, attivata dall'entrypoint.
_stato = {"dir": None, "registra": False, "scritti": 0, "riprodotti": 0}


def attiva(cartella: Path, registra: bool) -> None:
    """Attiva le cassette. In replay il manifest deve gia' esistere."""
    cartella = Path(cartella)
    if registra:
        (cartella / "cassette").mkdir(parents=True, exist_ok=True)
    else:
        manifest = cartella / "manifest.json"
        if not manifest.is_file():
            sys.exit(
                "Cassette non trovate in %s (manifest assente). Rigenera il corpus "
                "con --registra-cassette oppure togli --cassette." % cartella
            )
    _stato.update(dir=cartella, registra=registra, scritti=0, riprodotti=0)


def attivo() -> bool:
    return _stato["dir"] is not None


def in_registrazione() -> bool:
    return attivo() and _stato["registra"]


def chiave(ai: str, model: str, prompt_sha256: str, immagini_sha: list[str]) -> str:
    """Chiave stabile: provider, modello, prompt e contenuti delle immagini."""
    payload = json.dumps(
        {"ai": ai, "model": model, "prompt_sha256": prompt_sha256, "images": immagini_sha},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percorso(k: str) -> Path:
    return _stato["dir"] / "cassette" / ("%s.json" % k)


def _immagini_sha(images) -> list[str]:
    return [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images]


def riproduci(ai, model, prompt_sha256, images, output_path: Path) -> dict:
    """Carica la cassette registrata. Mancante: errore duro, nessuna chiamata reale."""
    k = chiave(ai, model, prompt_sha256, _immagini_sha(images))
    path = _percorso(k)
    if not path.is_file():
        sys.exit(
            "Cassette assente per %s (chiave %s, prompt %s). In replay non si "
            "chiama mai il modello: rigenera il corpus o controlla --review-model "
            "e la versione di PyMuPDF." % (Path(output_path).name, k[:16], prompt_sha256[:16])
        )
    voce = json.loads(path.read_text("utf-8"))
    contenuto = voce.get("output")
    if contenuto is not None:
        grezzo = contenuto.encode("utf-8")
        if hashlib.sha256(grezzo).hexdigest() != voce.get("output_sha256"):
            sys.exit("Cassette %s corrotta: output_sha256 non corrisponde." % path.name)
        Path(output_path).write_bytes(grezzo)
    _stato["riprodotti"] += 1
    return voce


def registra(ai, model, prompt_sha256, images, output_path: Path, proc,
             usage=None, elapsed_s: float = 0.0) -> None:
    """Salva l'esito di una chiamata reale. Nessun comportamento extra per la run."""
    output = None
    output_sha = None
    if output_path is not None and Path(output_path).is_file():
        grezzo = Path(output_path).read_bytes()
        output = grezzo.decode("utf-8")
        output_sha = hashlib.sha256(grezzo).hexdigest()
    stdout = (getattr(proc, "stdout", "") or "").strip().splitlines()
    voce = {
        "schema": SCHEMA, "ai": ai, "model": model, "prompt_sha256": prompt_sha256,
        "images": _immagini_sha(images), "output_file": Path(output_path).name,
        "output_sha256": output_sha, "output": output,
        "returncode": getattr(proc, "returncode", None),
        "stdout_tail": stdout[-1] if stdout else "",
        "usage": usage or {}, "elapsed_s": round(elapsed_s, 3),
    }
    k = chiave(ai, model, prompt_sha256, voce["images"])
    _percorso(k).write_text(json.dumps(voce, ensure_ascii=False, indent=1), "utf-8")
    _stato["scritti"] += 1


def registra_timeout(ai, model, prompt_sha256, images) -> None:
    """Un timeout reale si riproduce come timeout: lotto invalido, stesso percorso."""
    stdout = ""
    voce = {
        "schema": SCHEMA, "ai": ai, "model": model, "prompt_sha256": prompt_sha256,
        "images": _immagini_sha(images), "output_file": None,
        "output_sha256": None, "output": None, "returncode": None,
        "stdout_tail": stdout, "usage": {}, "elapsed_s": 0.0, "timeout": True,
    }
    k = chiave(ai, model, prompt_sha256, voce["images"])
    _percorso(k).write_text(json.dumps(voce, ensure_ascii=False, indent=1), "utf-8")
    _stato["scritti"] += 1


def pixmap_attesi(pdf: Path, dpi: int = 100) -> dict:
    """Hash dei pixmap pagina per pagina: il confronto portante del replay e2e."""
    import pymupdf

    attesi = {}
    with pymupdf.open(str(pdf)) as doc:
        for numero, pagina in enumerate(doc, start=1):
            attesi[str(numero)] = hashlib.sha256(
                pagina.get_pixmap(dpi=dpi).samples
            ).hexdigest()
    return attesi


def aggiungi_attesi(cartella: Path, pdf: Path, dpi: int = 100) -> None:
    """Registra nel manifest i pixmap attesi del PDF prodotto dalla run reale."""
    cartella = Path(cartella)
    manifest = json.loads((cartella / "manifest.json").read_text("utf-8"))
    manifest.setdefault("attesi_dpi", dpi)
    manifest["attesi"] = pixmap_attesi(pdf, dpi)
    manifest["atteso_sha256"] = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()
    (cartella / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8"
    )


def _versione_pymupdf():
    try:
        import pymupdf

        return getattr(pymupdf, "__version__", None) or str(pymupdf.version[0])
    except Exception:
        return None


def chiudi(unito_sha256: str | None = None, piano_sha256: str | None = None,
           sessione: dict | None = None) -> dict | None:
    """Scrive il manifest del corpus e restituisce il riepilogo per gli eventi."""
    if not attivo():
        return None
    indice = {}
    for percorso in sorted((_stato["dir"] / "cassette").glob("*.json")):
        voce = json.loads(percorso.read_text("utf-8"))
        indice[percorso.stem] = {
            "ai": voce.get("ai"),
            "output_sha256": voce.get("output_sha256"),
            "timeout": bool(voce.get("timeout")),
        }
    manifest = {
        "schema": SCHEMA,
        "creato": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pymupdf": _versione_pymupdf(),
        "unito_sha256": unito_sha256,
        "piano_sha256": piano_sha256,
        "sessione": sessione or {},
        "cassette": indice,
    }
    (_stato["dir"] / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8"
    )
    return {"modalita": "registrazione" if _stato["registra"] else "replay",
            "cassette_scritte": _stato["scritti"],
            "cassette_riprodotte": _stato["riprodotti"],
            "cassette_totali": len(indice)}
