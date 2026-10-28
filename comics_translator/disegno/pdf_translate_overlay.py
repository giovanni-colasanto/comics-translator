#!/usr/bin/env python3
"""Traduce un PDF sostituendo il testo *in posizione*, mantenendo il layout.

Due rami, scelti pagina per pagina:
  - se la pagina ha testo, PyMuPDF ne estrae i blocchi con coordinate, font,
    corpo e colore, le redaction cancellano l'originale lasciando intatti
    immagini e grafica, e la traduzione viene riscritta nello stesso rettangolo;
  - se la pagina e' un'immagine (scansione, fumetto, screenshot), il testo e'
    fatto di pixel: se ne occupa image_overlay.py, con OCR e copertura.

In mezzo c'e' il piano (plan.py): lo stato intermedio serializzato su disco.
Puo' essere fermato li' (--plan-out), riscritto da un revisore che guarda le
pagine (--ai-review) o a mano, e poi eseguito (--plan-in). Chi rivede cambia un
documento, non il codice, e non rifa' ne' l'OCR ne' la traduzione.

Uso tipico:
    python pdf_translate_overlay.py input.pdf --from en --to it
    python pdf_translate_overlay.py input.pdf --from en --to it --ai-review
    python pdf_translate_overlay.py input.pdf --from en --to it --plan-out piano.json
    python pdf_translate_overlay.py input.pdf --plan-in piano.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from comics_translator import RADICE, pipeline_status as status
from comics_translator.traduzione import ai_review
from comics_translator.traduzione import cassette
from comics_translator.sorgente import source_inventory as inventory_data
from comics_translator.sorgente import text_localization as localization
from comics_translator.verifica import segnalazioni
from comics_translator.verifica import visual_qa
from comics_translator.disegno import image_overlay
from comics_translator.disegno import sblocco_automatico
from comics_translator.disegno.image_overlay import ImageJob, draw_group, plan_page
from comics_translator.disegno.plan import build_plan, load_plan, rebuild_page, save_plan

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    try:
        import fitz
    except ImportError:
        sys.exit("Manca PyMuPDF. Installa le dipendenze: pip install -r requirements.txt")


# --------------------------------------------------------------------------- #
# Font
# --------------------------------------------------------------------------- #

FONT_DIRS = [
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
]

# famiglia -> (regular, bold, italic, bold-italic)
FAMILIES = {
    "arial": ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
    "calibri": ("calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"),
    "times": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
    "segoeui": ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
    "comic": ("comic.ttf", "comicbd.ttf", "comici.ttf", "comicz.ttf"),
    "dejavusans": (
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique.ttf",
        "DejaVuSans-BoldOblique.ttf",
    ),
    # per cinese/giapponese/coreano
    "msgothic": ("msgothic.ttc", "msgothic.ttc", "msgothic.ttc", "msgothic.ttc"),
    "msyh": ("msyh.ttc", "msyhbd.ttc", "msyh.ttc", "msyhbd.ttc"),
}


def _find_font_file(name: str) -> Path | None:
    p = Path(name)
    if p.is_file():
        return p
    for d in FONT_DIRS:
        if not d.is_dir():
            continue
        cand = d / name
        if cand.is_file():
            return cand
        # ricerca case-insensitive, anche nelle sottocartelle (tipico su Linux)
        for found in d.rglob("*"):
            if found.is_file() and found.name.lower() == name.lower():
                return found
    return None


@dataclass
class FontSet:
    """I quattro tagli di una famiglia, gia' caricati come fitz.Font."""

    regular: "fitz.Font"
    bold: "fitz.Font"
    italic: "fitz.Font"
    bold_italic: "fitz.Font"

    def pick(self, bold: bool, italic: bool) -> "fitz.Font":
        if bold and italic:
            return self.bold_italic
        if bold:
            return self.bold
        if italic:
            return self.italic
        return self.regular


def load_fontset(spec: str) -> FontSet:
    """`spec` e' una famiglia nota (arial, calibri, ...) o il path di un .ttf."""
    key = spec.lower().replace(" ", "")
    if key in FAMILIES:
        names = FAMILIES[key]
    else:
        if not Path(spec).is_file():
            sys.exit(
                "Font '%s' non trovato. Usa una famiglia nota (%s) "
                "oppure il path di un file .ttf/.otf." % (spec, ", ".join(sorted(FAMILIES)))
            )
        names = (spec, spec, spec, spec)

    base = _find_font_file(names[0])
    if base is None:
        sys.exit("Nessun file font trovato per '%s'. Passa un path esplicito con --font." % spec)

    paths = [(_find_font_file(n) or base) for n in names]
    return FontSet(*[fitz.Font(fontfile=str(p)) for p in paths])


# --------------------------------------------------------------------------- #
# Estrazione blocchi
# --------------------------------------------------------------------------- #

FLAG_ITALIC = 1 << 1
FLAG_BOLD = 1 << 4


@dataclass
class Block:
    page: int
    rect: "fitz.Rect"  # area del testo originale: e' quella che viene cancellata
    text: str
    size: float
    color: tuple
    bold: bool
    italic: bool
    centered: bool
    translated: str = ""
    box: "fitz.Rect | None" = None  # area in cui scrivere, eventualmente allargata
    source_region_id: str | None = None

    @property
    def target(self) -> "fitz.Rect":
        return self.box or self.rect


def _dominant(values: list) -> object:
    """Il valore piu' frequente, pesato sulla lunghezza del testo."""
    tally: dict = {}
    for value, weight in values:
        tally[value] = tally.get(value, 0) + weight
    return max(tally.items(), key=lambda kv: kv[1])[0]


def extract_blocks(page: "fitz.Page", pno: int, min_chars: int, *, inventory=None, localize=False) -> tuple[list[Block], list]:
    """Restituisce (blocchi di testo, rettangoli delle immagini da non invadere)."""
    blocks: list[Block] = []
    images: list = []
    data = page.get_text("dict")
    if inventory is not None:
        inventory["acquisizioni"].append({"tipo": "pdf_text", "min_chars": min_chars})

    for blk in data.get("blocks", []):
        if blk.get("type") != 0:  # 0 = testo, 1 = immagine
            if "bbox" in blk:
                images.append(fitz.Rect(blk["bbox"]))
            continue

        source_ids = []
        if inventory is not None:
            for line in blk.get("lines", []):
                raw = "".join(span.get("text", "") for span in line.get("spans", []))
                source_ids.append(inventory_data.add_line(
                    inventory, pno, raw=raw, normalized=" ".join(raw.split()),
                    box=line.get("bbox", blk["bbox"]), kind="pdf_text",
                    direction=line.get("dir", (1, 0)),
                ))

        native_aliases = localization.native_line_duplicates(blk.get("lines", [])) if localize else {}
        if inventory is not None:
            source_rows = {r["id"]: r for r in inventory["righe"]}
            for duplicate, canonical in native_aliases.items():
                source_rows[source_ids[duplicate]].update(
                    stato="esclusa", motivo="duplicate", duplicato_di=source_ids[canonical],
                )

        def record(reason=None):
            if inventory is None:
                return None
            raw = " ".join("".join(s.get("text", "") for s in ln.get("spans", []))
                           for ln in blk.get("lines", []))
            return inventory_data.add_region(
                inventory, pno, source_ids, list(blk["bbox"]), raw,
                kind="blocco_pdf", reason=reason,
            )

        lines_text: list[str] = []
        sizes, colors, bolds, italics = [], [], [], []
        skip = False

        for line_index, line in enumerate(blk.get("lines", [])):
            if line_index in native_aliases:
                continue
            # testo ruotato/verticale: lo lasciamo com'e', riscriverlo sarebbe peggio
            if abs(line.get("dir", (1, 0))[1]) > 0.01:
                skip = True
                break
            parts = []
            for span in line.get("spans", []):
                t = span.get("text", "")
                if not t:
                    continue
                parts.append(t)
                w = max(len(t), 1)
                sizes.append((round(span.get("size", 10.0), 1), w))
                colors.append((span.get("color", 0), w))
                flags = span.get("flags", 0)
                bolds.append((bool(flags & FLAG_BOLD), w))
                italics.append((bool(flags & FLAG_ITALIC), w))
            if parts:
                lines_text.append("".join(parts).strip())

        if skip or not lines_text:
            record("rotated_text" if skip else "empty_text")
            continue

        # ricompone il paragrafo, ricucendo la sillabazione di fine riga
        text = ""
        for i, ln in enumerate(lines_text):
            if i == 0:
                text = ln
            elif text.endswith("-"):
                text = text[:-1] + ln
            else:
                text += " " + ln
        text = " ".join(text.split())

        if len(text) < min_chars or not any(c.isalpha() for c in text):
            record("too_short" if len(text) < min_chars else "no_letters")
            continue

        rect = fitz.Rect(blk["bbox"])
        if rect.is_empty or rect.width < 4 or rect.height < 4:
            record("small_geometry")
            continue

        # euristica: blocco corto e centrato nella pagina -> lo ricentriamo
        pw = page.rect.width
        left_gap, right_gap = rect.x0, pw - rect.x1
        centered = (
            len(lines_text) <= 2
            and left_gap > 0.12 * pw
            and abs(left_gap - right_gap) < 0.06 * pw
        )

        blocks.append(
            Block(
                page=pno,
                rect=rect,
                text=text,
                size=float(_dominant(sizes)) if sizes else 10.0,
                color=fitz.sRGB_to_pdf(int(_dominant(colors))) if colors else (0, 0, 0),
                bold=bool(_dominant(bolds)) if bolds else False,
                italic=bool(_dominant(italics)) if italics else False,
                centered=centered,
                source_region_id=record(),
            )
        )

    return blocks, images


def expand_boxes(page: "fitz.Page", blocks: list[Block], images: list, pad: float = 2.0) -> None:
    """Allarga l'area di scrittura verso destra e verso il basso, negli spazi vuoti.

    Serve perche' il bbox di partenza e' stretto attorno al testo originale: una
    traduzione piu' lunga sarebbe costretta a rimpicciolire anche quando accanto
    c'e' pagina bianca. I limiti sono gli altri blocchi, le immagini e la gabbia
    del testo (nessuna espansione oltre i margini gia' usati dalla pagina).
    """
    if not blocks:
        return

    text_right = max(b.rect.x1 for b in blocks)
    text_left = min(b.rect.x0 for b in blocks)
    text_bottom = max(b.rect.y1 for b in blocks)
    obstacles = [b.rect for b in blocks] + list(images)

    for b in blocks:
        left, right, bottom = text_left, text_right, text_bottom

        for o in obstacles:
            if o is b.rect:
                continue
            # ostacoli che stanno sulla stessa fascia orizzontale
            if o.y1 > b.rect.y0 + 0.5 and o.y0 < b.rect.y1 - 0.5:
                if o.x1 <= b.rect.x0:
                    left = max(left, o.x1 + pad)
                elif o.x0 >= b.rect.x1:
                    right = min(right, o.x0 - pad)
            # ostacoli sotto, sulla stessa colonna
            if o.x1 > b.rect.x0 + 0.5 and o.x0 < b.rect.x1 - 0.5 and o.y0 >= b.rect.y1:
                bottom = min(bottom, o.y0 - pad)

        right = max(right, b.rect.x1)
        bottom = max(bottom, b.rect.y1)

        if b.centered:
            # il testo centrato resta centrato: si allarga in modo simmetrico
            cx = (b.rect.x0 + b.rect.x1) / 2
            half = max(min(cx - min(left, b.rect.x0), right - cx), b.rect.width / 2)
            b.box = fitz.Rect(cx - half, b.rect.y0, cx + half, bottom)
        else:
            b.box = fitz.Rect(b.rect.x0, b.rect.y0, right, bottom)


# --------------------------------------------------------------------------- #
# Traduzione
# --------------------------------------------------------------------------- #

MAX_CHUNK = 4500  # limite pratico dell'endpoint Google gratuito

ENGINES = ("auto", "argos", "deepl", "google", "libre", "none")

# L'endpoint gratuito di Google, quando e' sotto blocco, non solleva eccezioni:
# restituisce il testo della propria pagina di errore. Va intercettato, altrimenti
# finisce dritto dentro il PDF.
BROKEN_MARKERS = (
    "that’s an error",
    "that's an error",
    "error 500 (server error)",
    "that’s all we know",
    "that's all we know",
    "<!doctype html",
    "<html",
)


class TranslationFailed(RuntimeError):
    pass


def looks_broken(result: str) -> bool:
    if not result or not result.strip():
        return True
    low = result.lower()
    return any(m in low for m in BROKEN_MARKERS)


def _split_long(text: str, limit: int = MAX_CHUNK) -> list[str]:
    if len(text) <= limit:
        return [text]
    marked = text.replace("! ", "!\x00").replace("? ", "?\x00").replace(". ", ".\x00")
    chunks: list[str] = []
    cur = ""
    for sentence in marked.split("\x00"):
        if cur and len(cur) + len(sentence) + 1 > limit:
            chunks.append(cur)
            cur = sentence
        else:
            cur = ("%s %s" % (cur, sentence)).strip()
    if cur:
        chunks.append(cur)
    return chunks


def resolve_engine(requested: str, source: str, target: str) -> str:
    """'auto' sceglie il primo motore realmente utilizzabile, dal migliore in giu'."""
    if requested != "auto":
        return requested
    if os.environ.get("DEEPL_API_KEY"):
        return "deepl"
    try:
        import argostranslate.translate  # noqa: F401

        if source != "auto":
            return "argos"
    except ImportError:
        pass
    return "google"


class Translator:
    def __init__(self, engine: str, source: str, target: str, cache_path: Path | None):
        self.engine = engine
        self.source = source
        self.target = target
        self.cache_path = cache_path
        self.failures = 0
        self.cache: dict[str, str] = {}
        if cache_path and cache_path.is_file():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self._backend = self._make_backend() if engine != "none" else None

    # -- backend ----------------------------------------------------------- #

    def _make_backend(self):
        if self.engine == "argos":
            return self._make_argos()
        try:
            from deep_translator import DeeplTranslator, GoogleTranslator, LibreTranslator
        except ImportError:
            sys.exit("Manca deep-translator. Installa: pip install -r requirements.txt")

        if self.engine == "google":
            return GoogleTranslator(source=self.source, target=self.target)
        if self.engine == "deepl":
            key = os.environ.get("DEEPL_API_KEY")
            if not key:
                sys.exit("Engine deepl selezionato ma DEEPL_API_KEY non e' impostata.")
            return DeeplTranslator(api_key=key, source=self.source, target=self.target)
        if self.engine == "libre":
            base = os.environ.get("LIBRETRANSLATE_URL", "https://libretranslate.com/")
            return LibreTranslator(
                source=self.source,
                target=self.target,
                base_url=base,
                api_key=os.environ.get("LIBRETRANSLATE_API_KEY"),
            )
        sys.exit("Engine sconosciuto: %s" % self.engine)

    def _make_argos(self):
        """NMT offline: nessuna chiave, nessun rate limit, modello scaricato una volta."""
        try:
            import argostranslate.package as ap
            import argostranslate.translate as at
        except ImportError:
            sys.exit("Engine argos selezionato ma argostranslate non e' installato "
                     "(pip install argostranslate).")

        if self.source == "auto":
            sys.exit("argos non rileva la lingua da solo: indica la lingua di partenza "
                     "con --from (es. --from en).")

        installed = {(lang.code, t.to_lang.code)
                     for lang in at.get_installed_languages()
                     for t in lang.translations_from}
        if (self.source, self.target) not in installed:
            print("Scarico il modello argos %s->%s (una tantum)..." % (self.source, self.target))
            try:
                ap.update_package_index()
                pkg = next(
                    p for p in ap.get_available_packages()
                    if p.from_code == self.source and p.to_code == self.target
                )
                ap.install_from_path(pkg.download())
            except StopIteration:
                sys.exit("Nessun modello argos per la coppia %s->%s. "
                         "Prova a passare dall'inglese o usa --engine deepl."
                         % (self.source, self.target))
            except Exception as exc:
                sys.exit("Download del modello argos fallito: %s" % exc)

        src, tgt = self.source, self.target
        return type("ArgosBackend", (), {"translate": staticmethod(
            lambda text: at.translate(text, src, tgt))})()

    # -- traduzione -------------------------------------------------------- #

    def _key(self, text: str) -> str:
        return "%s|%s|%s|%s" % (self.engine, self.source, self.target, text)

    def _call(self, text: str) -> str:
        out = []
        for chunk in _split_long(text):
            last_err = None
            for attempt in range(4):
                try:
                    res = self._backend.translate(chunk)
                    if not isinstance(res, str) or looks_broken(res):
                        raise TranslationFailed("risposta non valida dal motore")
                    out.append(res)
                    break
                except Exception as exc:  # rete instabile, rate limit, endpoint bloccato
                    last_err = exc
                    time.sleep(1.5 * (attempt + 1))
            else:
                self.failures += 1
                print("  ! blocco non tradotto (%s): tengo l'originale" % last_err, file=sys.stderr)
                out.append(chunk)
        return " ".join(out)

    def translate_all(self, blocks: list[Block], workers: int) -> None:
        if self.engine == "none":
            for b in blocks:
                b.translated = b.text
            return

        todo = []
        for b in blocks:
            hit = self.cache.get(self._key(b.text))
            if hit is not None:
                b.translated = hit
            else:
                todo.append(b)

        if not todo:
            print("Tutti i %d blocchi presi dalla cache." % len(blocks))
            return

        print("Traduco %d blocchi con '%s' (%d da cache)..."
              % (len(todo), self.engine, len(blocks) - len(todo)))
        if workers <= 1:
            results = [self._call(b.text) for b in todo]
        else:
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda b: self._call(b.text), todo))

        for b, res in zip(todo, results):
            b.translated = res or b.text
            if self.failures == 0 or res != b.text:
                self.cache[self._key(b.text)] = b.translated
        self.save_cache()

        if self.failures:
            print("  ! %d blocchi su %d non tradotti." % (self.failures, len(todo)), file=sys.stderr)
        if self.failures == len(todo):
            sys.exit(
                "Il motore '%s' non ha tradotto nulla: non genero un PDF identico all'originale.\n"
                "L'endpoint gratuito di Google viene bloccato spesso. Alternative:\n"
                "  --engine argos --from <lingua>   (offline, nessuna chiave)\n"
                "  --engine deepl                   (con DEEPL_API_KEY impostata)" % self.engine
            )

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8"
            )
        except Exception as exc:
            print("  ! cache non salvata: %s" % exc, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Disegno
# --------------------------------------------------------------------------- #

LINE_H = 1.16


def wrap(font: "fitz.Font", text: str, width: float, size: float) -> list[str]:
    """Manda a capo il testo dentro `width` misurando con le metriche reali."""
    lines: list[str] = []
    for word in text.split():
        if not lines:
            lines.append(word)
            continue
        cand = "%s %s" % (lines[-1], word)
        if font.text_length(cand, size) <= width:
            lines[-1] = cand
        else:
            lines.append(word)

    # parole singole piu' larghe della colonna: spezzate a forza
    out: list[str] = []
    for ln in lines:
        while font.text_length(ln, size) > width and len(ln) > 1:
            cut = len(ln)
            while cut > 1 and font.text_length(ln[:cut], size) > width:
                cut -= 1
            out.append(ln[:cut])
            ln = ln[cut:]
        out.append(ln)
    return out


def fit(font, text, rect, start_size, min_size, grow):
    """Il corpo piu' grande (fino a start_size * grow) per cui il testo entra.

    Deve entrare in altezza *e* la parola piu' lunga deve stare nella larghezza:
    senza il secondo controllo una parola troppo larga verrebbe spezzata a meta'
    invece di far rimpicciolire il testo.
    """
    words = text.split() or [text]
    size = start_size * grow
    while size >= min_size:
        if max(font.text_length(w, size) for w in words) <= rect.width:
            lines = wrap(font, text, rect.width, size)
            if len(lines) * size * LINE_H <= rect.height + 0.6:
                return size, lines
        size -= 0.25
    return min_size, wrap(font, text, rect.width, min_size)


def draw_block(page: "fitz.Page", blk: Block, fonts: FontSet, min_size: float, grow: float) -> bool:
    font = fonts.pick(blk.bold, blk.italic)
    rect = blk.target
    size, lines = fit(font, blk.translated, rect, blk.size, min_size, grow)

    writer = fitz.TextWriter(page.rect)
    asc = min(max(font.ascender, 0.7), 1.0)
    written = False
    for i, line in enumerate(lines):
        y = rect.y0 + size * asc + i * size * LINE_H
        if y > rect.y1 + size:  # tracimato: meglio troncare che sporcare cio' che sta sotto
            break
        x = rect.x0
        if blk.centered:
            x += max(0.0, (rect.width - font.text_length(line, size)) / 2)
        writer.append(fitz.Point(x, y), line, font=font, fontsize=size)
        written = True

    if not written:
        return False
    writer.write_text(page, color=blk.color)
    return True


def erase_originals(page: "fitz.Page", blocks: list[Block], fill_white: bool) -> None:
    fill = (1, 1, 1) if fill_white else None
    for blk in blocks:
        page.add_redact_annot(blk.rect, fill=fill)
    try:
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )
    except (TypeError, AttributeError):  # PyMuPDF piu' vecchio
        try:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        except TypeError:
            page.apply_redactions()


# --------------------------------------------------------------------------- #
# OCR opzionale
# --------------------------------------------------------------------------- #

def run_ocr(src: Path, lang: str) -> Path:
    exe = shutil.which("ocrmypdf")
    if not exe:
        sys.exit(
            "--ocr richiede ocrmypdf (pip install ocrmypdf) piu' Tesseract e Ghostscript "
            "installati nel sistema."
        )
    out = Path(tempfile.gettempdir()) / ("%s.ocr.pdf" % src.stem)
    print("OCR in corso (%s)..." % lang)
    proc = subprocess.run(
        [exe, "--skip-text", "-l", lang, str(src), str(out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit("ocrmypdf ha fallito:\n%s" % proc.stderr.strip())
    return out


# --------------------------------------------------------------------------- #
# Output bilingue
# --------------------------------------------------------------------------- #

def build_dual(original: Path, translated: Path, out: Path) -> None:
    """PDF affiancato: a sinistra l'originale, a destra la traduzione."""
    src = fitz.open(original)
    dst = fitz.open(translated)
    dual = fitz.open()
    for i in range(min(src.page_count, dst.page_count)):
        r = src[i].rect
        page = dual.new_page(width=r.width * 2, height=r.height)
        page.show_pdf_page(fitz.Rect(0, 0, r.width, r.height), src, i)
        page.show_pdf_page(fitz.Rect(r.width, 0, r.width * 2, r.height), dst, i)
    dual.save(out, garbage=3, deflate=True)
    dual.close()
    src.close()
    dst.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_pages(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a) - 1, int(b)))
        else:
            wanted.add(int(part) - 1)
    return sorted(p for p in wanted if 0 <= p < total)


def plan_to_objects(plan: dict) -> tuple:
    """Dal piano agli oggetti da disegnare, pagina per pagina."""
    inventory_data.validate_sources(plan)
    per_page: dict[int, list] = {}
    image_jobs: dict[int, list] = {}
    for page in plan.get("pagine", []):
        pno = int(page.get("indice", page.get("numero", 1) - 1))
        blocks, jobs = rebuild_page(page.get("blocchi", []), Block, ImageJob, page=page)
        if blocks:
            per_page[pno] = blocks
        if jobs:
            image_jobs[pno] = jobs
    return per_page, image_jobs


def _render_document(doc, pages, per_page, image_jobs, fonts, image_fonts, args, out_path):
    """Disegna su un documento aperto; apertura e chiusura restano al chiamante."""
    print("Riscrivo le pagine...")
    for pno in pages:
        page = doc[pno]

        # prima si puliscono tutti i balloon, poi si scrive: al contrario il
        # riempimento di un gruppo cancella il testo del gruppo disegnato prima,
        # ed e' cosi' che sparivano le ultime parole dei balloon accavallati
        illeggibili = []
        for fase in ("riempi", "scrivi"):
            for job in image_jobs.get(pno, []):
                text = job.translated.upper() if job.upper else job.translated
                disegnato = draw_group(
                    page=page,
                    box_px=job.box_px,
                    text=text,
                    scale=job.scale,
                    bg=job.bg,
                    light_text=job.light_text,
                    base_size=job.base_size,
                    fonts=image_fonts,
                    min_size=args.min_size,
                    fit_fn=fit,
                    fill_poly=job.fill_poly,
                    riempi=fase == "riempi",
                    scrivi=fase == "scrivi",
                    centered=job.centered,
                    lobi=_lobi_da_scrivere(job),
                )
                if fase == "riempi" and not disegnato:
                    illeggibili.append(job)
        if illeggibili:
            # Non e' un dettaglio estetico: quelle battute restano in originale
            # e la copertura della pagina non e' quella che il piano prometteva.
            print("  p%d: %d blocchi lasciati in originale, il testo non entrava"
                  % (pno + 1, len(illeggibili)))
            status.event("render_block_illegible", page=pno + 1,
                         blocks=len(illeggibili),
                         regions=[getattr(j, "source_region_id", None) for j in illeggibili])

        blocks = per_page.get(pno) or []
        if not blocks:
            continue
        erase_originals(page, blocks, args.fill_white)
        for blk in blocks:
            draw_block(page, blk, fonts, args.min_size, args.grow)

    if hasattr(doc, "subset_fonts"):
        try:
            doc.subset_fonts()
        except Exception:
            pass
    doc.save(out_path, garbage=4, deflate=True)
    print("OK -> %s" % out_path)


def _segna_illeggibili(plan, image_jobs, image_fonts, args) -> int:
    """Marca `salta` i blocchi il cui testo non entra; ritorna quanti sono."""
    blocchi = {
        block.get("_source_region_id") or block.get("sorgente", {}).get("regione"): block
        for page in plan.get("pagine", [])
        for block in page.get("blocchi", [])
        if isinstance(block, dict)
    }
    segnati = 0
    for jobs in image_jobs.values():
        for job in jobs:
            block = blocchi.get(getattr(job, "source_region_id", None))
            if block is None or block.get("salta"):
                continue
            text = job.translated.upper() if job.upper else job.translated
            if images_fit(job, text, image_fonts, args):
                # Un blocco corretto a mano e riprovato (salta: false) puo'
                # portarsi dietro il testo_illeggibile della passata precedente:
                # se ora entra, quello stantio non deve piu' far dichiarare
                # perso un lavoro che sulla tavola c'e' davvero.
                block.get("_render", {}).pop("testo_illeggibile", None)
                continue
            block["salta"] = True
            block.setdefault("_render", {})["testo_illeggibile"] = True
            segnati += 1
            status.event("render_block_illegible", block=block["id"],
                         text=(block.get("testo") or "")[:80])
    if segnati:
        print("  %d blocchi restano in originale: la traduzione non entra nel riquadro"
              % segnati)
    return segnati


def _lobi_da_scrivere(job):
    """I lobi del job con il testo nella forma del lettering (maiuscolo se serve)."""
    return [dict(lobo, testo=lobo["testo"].upper() if job.upper else lobo["testo"])
            for lobo in getattr(job, "lobi", None) or []]


def images_fit(job, text, image_fonts, args) -> bool:
    """Come `draw_group`: per lobi se il balloon ne ha, altrimenti nel riquadro stretto."""
    font = image_fonts.pick(bold=True, italic=False)
    return image_overlay.entra_il_job(
        font, text, job.box_px, job.scale, job.fill_poly, job.base_size, args.min_size, fit,
        lobi=_lobi_da_scrivere(job))


def render_plan(source_pdf, plan, pages, fonts, args, out_path) -> list[int]:
    """Apre una copia pulita della sorgente e applica un piano completo."""
    status.plan_index(plan, [p + 1 for p in pages])
    expected_hash = plan.get("sorgente_sha256")
    if expected_hash and status.sha256(source_pdf) != expected_hash:
        raise ValueError("il PDF sorgente non corrisponde all'hash del piano")
    status.start_phase("render", len(pages), "pages")
    # Chi salta e chi no puo' essere cambiato dopo il piano (revisione, QA):
    # la protezione va ricalcolata qui, dove si decide cosa viene disegnato.
    localization.enforce_localization(plan)
    per_page, image_jobs = plan_to_objects(plan)
    selected = list(pages)
    image_fonts = load_fontset(args.image_font) if image_jobs else None
    # Un blocco il cui testo non entra nel riquadro resta in originale: allora
    # il suo inchiostro va difeso come quello di ogni altro blocco non
    # riscritto, o il vicino glielo copre a meta'. Per questo si rimisura
    # prima di disegnare, si aggiorna il piano e si ricalcolano le protezioni.
    if image_fonts and _segna_illeggibili(plan, image_jobs, image_fonts, args):
        localization.enforce_localization(plan)
        per_page, image_jobs = plan_to_objects(plan)
    # Stessa ragione: il `enforce_localization` qui sopra puo' proteggere un
    # blocco che nel piano era attivo e tradotto (una correzione QA che cambia
    # i `salta` vicini, un riempimento ridisegnato a mano, un --plan-in di un
    # piano ritoccato). Finora la dichiarazione stava solo sul percorso di
    # traduzione: chi rendeva un piano gia' fatto perdeva l'italiano in
    # silenzio. Su imperial-guardians p26 ("AT YOUR BIDDING. / HOW DO YOU WANT
    # 'EM?") il balloon tradotto e' sparito cosi', con esito "complete".
    segnala_traduzioni_soppresse(plan)
    # Dichiarato qui e non nel `main`: il rifiuto e' un esito del rendering, e
    # ogni percorso che disegna (prima passata, ri-render dopo il QA, --plan-in)
    # deve riportarlo. Nel piano il `salta` resta, quindi le passate successive
    # non rifanno la misura e senza questa riga chiuderebbero in silenzio.
    segnala_traduzioni_non_disegnate(plan)
    doc = fitz.open(source_pdf)
    try:
        _render_document(
            doc, selected, per_page, image_jobs, fonts, image_fonts, args, out_path,
        )
    finally:
        doc.close()
    with fitz.open(source_pdf) as source, fitz.open(out_path) as output:
        if source.page_count != output.page_count or any(
            source[pno].rect != output[pno].rect for pno in range(source.page_count)
        ):
            raise ValueError("numero o dimensioni delle pagine alterati nel rendering")
        report = status.ACTIVE_REPORT.get()
        if report:
            report.artifact("output_pdf", out_path, pages=output.page_count)
    status.finish_phase(status.PhaseResult("render", len(pages), len(pages),
                                           unit="pages", pages=[p + 1 for p in pages]))
    return selected


def esegui_sblocco_automatico(plan, source_pdf=None, banda=True, dividi=True):
    """Disegna i blocchi protetti che il modello ha tradotto e l'area permette.

    Vedi `comics_translator.disegno.sblocco_automatico`: e' lo `sblocca` dei `.finale`
    fatto con la natura detta dal modello e le righe vicine verificate. Anche i
    rifiuti vanno nel registro, con il motivo: sono il punto da cui guardare.

    Con la sorgente a disposizione si fanno altri due gesti dei `.finale`: i
    rifiutati per area vanno su una banda del colore del fondo, e i blocchi che
    tengono due balloon separati si dividono in un blocco per balloon.
    """
    from comics_translator.disegno import dividi_balloon
    from comics_translator.disegno.operazioni import Officina

    decisioni = sblocco_automatico.sblocca_tradotti(plan)
    officina = Officina(plan, source_pdf)
    try:
        if banda:
            sblocco_automatico.sblocca_con_banda(plan, officina, decisioni)
        divisi = dividi_balloon.dividi_piano(officina) if dividi and source_pdf else []
        if officina.toccati:
            officina.sigilla(sorted({officina.pagina_di[t["id"]]["numero"]
                                     for t in officina.toccati
                                     if t["id"] in officina.pagina_di}))
            localization.enforce_localization(plan)
    finally:
        officina.chiudi()
    sbloccati = [d for d in decisioni if d["esito"] == "sbloccato"]
    bande = [d for d in decisioni if d["esito"] == "banda"]
    rifiutati = [d for d in decisioni if d["esito"] == "resta_saltato"]
    status.event("sblocco_automatico",
                 sbloccati=[d["id"] for d in sbloccati],
                 bande=[d["id"] for d in bande],
                 divisi=[{"id": d["id"], "balloon": d["balloon"]} for d in divisi],
                 rifiutati=[{"id": d["id"], "motivo": d.get("motivo"),
                             "motivo_banda": d.get("motivo_banda")} for d in rifiutati])
    if decisioni or divisi:
        print("  sblocco automatico: %d blocchi protetti ora disegnati, %d su banda, "
              "%d divisi per balloon, %d restano saltati per l'area"
              % (len(sbloccati), len(bande), len(divisi), len(rifiutati)))
    return decisioni


def segnala_traduzioni_soppresse(plan):
    """Dichiara i blocchi tradotti che la protezione tiene invisibili.

    Non li corregge (allentare la protezione rischia di coprire arte reale):
    li porta nel manifest e a schermo, cosi' smettono di passare in silenzio.
    Era il guasto che su ASM #7 lasciava didascalie tradotte con l'inglese
    ancora visibile, invisibile a OCR e audit raster. Ogni voce e' recuperabile
    con uno `sblocca` motivato in una correzione QA.
    """
    soppresse = localization.traduzioni_soppresse(plan)
    if not soppresse:
        return soppresse
    pagine = sorted({s["pagina"] for s in soppresse})
    voce = status.issue(
        "traduzione_soppressa_da_protezione",
        "blocchi tradotti ma tenuti in originale dalla protezione automatica; "
        "sbloccabili con una correzione 'sblocca'",
        pagine, [s["id"] for s in soppresse])
    report = status.ACTIVE_REPORT.get()
    issues = report.data.setdefault("issues", []) if report is not None else []
    # Traduzione e rendering la dichiarano entrambi: nella stessa corsa e' lo
    # stesso guasto, non due. Ripeterlo gonfia il manifest, conta due volte la
    # stessa perdita e nel registro della prova su superman-1 stampava due volte
    # le stesse 25 battute.
    if voce in issues:
        return soppresse
    status.event("traduzioni_soppresse",
                 blocchi=[s["id"] for s in soppresse], pagine=pagine)
    if report is not None:
        issues.append(voce)
        report.save()
    contraddetti = [s for s in soppresse if s.get("motivo") != "protezione"]
    print("  ! %d blocchi TRADOTTI restano invisibili (pagine %s): l'italiano c'e', "
          "va sbloccato" % (len(soppresse), ", ".join(str(n) for n in pagine)))
    if contraddetti:
        # Qui non e' la maschera a nascondere la traduzione: e' il modello che si
        # contraddice (ha chiuso una battuta che aveva tradotto, o ha tradotto
        # cio' che aveva chiamato grafica). Va detto a parte, perche' si guarda
        # in un altro posto: la risposta del modello, non la geometria.
        print("    di cui %d per contraddizione del modello: %s"
              % (len(contraddetti),
                 ", ".join("%s (%s)" % (s["id"], s["motivo"]) for s in contraddetti[:6])))
    return soppresse


def segnala_testo_scartato_come_disegno(plan):
    """Dichiara i gruppi OCR esclusi che hanno la forma del testo (Gap C).

    L'esclusione `background_noise` agisce PRIMA della traduzione: quello che
    scarta non arriva al modello, non arriva al QA e non manca nemmeno
    all'audit, che misura la copertura sulla sorgente sigillata. Su Imperial ha
    portato via interi balloon di dialogo senza che una riga lo dicesse.

    Non si alza la soglia da soli: il rumore del dialogo su fondo carico e
    quello di un'onomatopea disegnata si accavallano, e allentare l'esclusione
    significa coprire di sfondo l'arte della tavola accanto. Si dichiara, e
    l'operatore decide guardando la pagina. Per rimetterne in gioco uno si
    rilancia l'analisi con `--max-bg-noise` piu' alto.
    """
    sospetti = segnalazioni.testo_scartato_come_disegno(plan)
    if not sospetti:
        return sospetti
    pagine = sorted({s["pagina"] for s in sospetti if s["pagina"] is not None})
    status.event("testo_scartato_come_disegno",
                 regioni=[s["id"] for s in sospetti], pagine=pagine)
    report = status.ACTIVE_REPORT.get()
    if report is not None:
        report.data.setdefault("issues", []).append(status.issue(
            "testo_scartato_come_disegno",
            "gruppi OCR esclusi come 'testo sul disegno' che hanno la forma di un "
            "blocco di testo; mai passati alla traduzione, rivedibili con "
            "--max-bg-noise piu' alto",
            pagine, [s["id"] for s in sospetti]))
        report.save()
    print("  ! %d gruppi scartati come disegno sembrano testo (pagine %s): "
          "non sono mai arrivati alla traduzione"
          % (len(sospetti), ", ".join(str(n) for n in pagine)))
    for sospetto in sospetti:
        # testo OCR grezzo: puo' contenere qualunque carattere, e la console no
        print(status.testo_console(
            "      p%s %s: %s"
            % (sospetto["pagina"], sospetto["forma"], sospetto["testo"])))
    return sospetti


def segnala_dialogo_protetto(plan):
    """Dichiara il dialogo che la protezione ha zittito prima della traduzione.

    E' il piu' grande dei quattro guasti dichiarati e nessuno degli altri lo
    vedeva: il blocco arriva al modello gia' con `salta: true`, il modello
    ricopia l'originale come il prompt gli dice, e l'inglese finisce nel PDF.
    Il Gap A cerca una traduzione nascosta che qui non e' mai stata chiesta, il
    Gap B esclude i blocchi saltati per costruzione, il Gap C guarda gli scarti
    dell'OCR e questi sono acquisiti.

    Su carnage-3 erano 25 blocchi, di cui 19 dialogo vero, e il danno si vedeva
    a meta' balloon: «DECINE / PEOPLE DIED IN THAT POLICE STATION TODAY».
    Recuperarli costa uno `sblocca` motivato PIU' la traduzione scritta a mano,
    perche' quella non esiste da nessuna parte.
    """
    protetti = segnalazioni.dialogo_protetto_non_tradotto(plan)
    if not protetti:
        return protetti
    pagine = sorted({p["pagina"] for p in protetti if p["pagina"] is not None})
    status.event("dialogo_protetto_non_tradotto",
                 blocchi=[p["id"] for p in protetti], pagine=pagine)
    report = status.ACTIVE_REPORT.get()
    if report is not None:
        report.data.setdefault("issues", []).append(status.issue(
            "dialogo_protetto_non_tradotto",
            "blocchi di dialogo protetti prima della traduzione: il modello non li "
            "ha mai tradotti e restano in inglese; servono uno 'sblocca' motivato e "
            "la traduzione", pagine, [p["id"] for p in protetti]))
        report.save()
    print("  ! %d blocchi di dialogo sono rimasti in inglese perche' protetti "
          "prima della traduzione (pagine %s)"
          % (len(protetti), ", ".join(str(n) for n in pagine)))
    return protetti


def segnala_traduzioni_non_disegnate(plan):
    """Dichiara le traduzioni che il renderer ha rifiutato (Gap E).

    `_segna_illeggibili` riporta a `salta: true` il blocco il cui italiano non
    entra nel riquadro. E' la scelta giusta - meglio l'originale che due battute
    sovrapposte - ma finora finiva li': il `salta` restava nel piano, la passata
    successiva non ci riprovava, e la run chiudeva senza che nulla dicesse che
    una traduzione pagata non era arrivata sulla tavola.

    Nessun altro segnale copre il caso, `inglese_residuo` compreso: quello
    misura le righe scoperte dentro un'area riscritta, e un blocco tornato a
    `salta` non e' piu' un'area riscritta. Su carnage-5 erano 53 blocchi contro
    le 4 righe che il gate dichiarava.

    Non si corregge da soli: il riquadro giusto dipende da quanta tavola si e'
    disposti a coprire, e quello lo decide chi guarda la pagina.
    """
    persi = segnalazioni.traduzione_non_disegnata(plan)
    if not persi:
        return persi
    pagine = sorted({p["pagina"] for p in persi if p["pagina"] is not None})
    status.event("traduzione_non_disegnata",
                 blocchi=[p["id"] for p in persi], pagine=pagine)
    report = status.ACTIVE_REPORT.get()
    if report is not None:
        report.data.setdefault("issues", []).append(status.issue(
            "traduzione_non_disegnata",
            "blocchi tradotti che il renderer ha rifiutato perche' il testo non "
            "entra nel riquadro: l'italiano c'e' nel piano, sulla tavola resta "
            "l'inglese; serve allargare il riquadro o accorciare la battuta",
            pagine, [p["id"] for p in persi]))
        report.save()
    print("  ! %d traduzioni non sono state disegnate: il testo non entra nel "
          "riquadro (pagine %s)"
          % (len(persi), ", ".join(str(n) for n in pagine)))
    return persi


def segnala_traduzioni_copiate(plan):
    """Dichiara le frasi che il modello ha restituito ancora in inglese (Gap B).

    Una parola o due che ripetono l'originale sono una scelta legittima
    (onomatopee, nomi, insegne); una frase intera no: e' il modello che sotto
    degrado ricopia la sorgente, e nel PDF finisce inglese. Non e' correggibile
    in automatico - ritradurre da soli il blocco vorrebbe dire inventare senza
    la tavola davanti - quindi si dichiara e si lascia al QA o all'operatore.
    """
    copie = segnalazioni.traduzione_uguale_all_originale(plan)
    if not copie:
        return copie
    pagine = sorted({c["pagina"] for c in copie if c["pagina"] is not None})
    status.event("traduzione_uguale_all_originale",
                 blocchi=[c["id"] for c in copie], pagine=pagine)
    report = status.ACTIVE_REPORT.get()
    if report is not None:
        report.data.setdefault("issues", []).append(status.issue(
            "traduzione_uguale_all_originale",
            "frasi restituite identiche all'originale: traduzione mancata, non skip "
            "editoriale", pagine, [c["id"] for c in copie]))
        report.save()
    print("  ! %d frasi restano identiche all'inglese (pagine %s): traduzione mancata"
          % (len(copie), ", ".join(str(n) for n in pagine)))
    return copie


def risolvi_qa_provider(args):
    """Provider e modello del QA visuale, distinti da quelli della traduzione.

    Verifica incrociata: far rileggere il lavoro di un modello a un altro trova
    errori che un modello sul proprio lavoro non vede (misurato su ASM #7:
    gpt-5.4 traduce "SOULWALL" come "Sock-Wale" e il QA con lo stesso gpt-5.4
    non lo pesca). Le regole, in ordine:

    - `--qa-model` esplicito vince sempre, con qualunque provider;
    - `--qa-ai` senza `--qa-model`: il provider QA col SUO modello di default,
      non quello della traduzione (che apparterrebbe a un altro provider);
    - senza `--qa-ai`: identico a prima del cambiamento - il QA usa provider e
      modello della traduzione, cosi' chi non chiede la verifica incrociata non
      vede alcuna differenza.
    """
    # getattr con default: chi costruisce `args` senza i campi QA (test o
    # chiamate piu' vecchie) ottiene il comportamento pre-feature.
    qa_ai_arg = getattr(args, "qa_ai", None)
    qa_model_arg = getattr(args, "qa_model", None)
    qa_ai = qa_ai_arg or args.review_ai
    if qa_model_arg is not None:
        qa_model = qa_model_arg
    elif qa_ai_arg is not None:
        qa_model = None
    else:
        qa_model = args.review_model
    return qa_ai, qa_model


def run_visual_qa(source_pdf, out_path, plan, pages, fonts, args):
    """Controlla il primo rendering e lo rigenera una volta se serve."""
    if not args.visual_qa:
        return plan
    review_dir = Path(args.review_dir) if args.review_dir else out_path.with_suffix(".revisione")
    qa_ai, qa_model = risolvi_qa_provider(args)
    if qa_ai != args.review_ai:
        print("QA visuale a verifica incrociata: traduzione con %s, QA con %s"
              % (args.review_ai, qa_ai))
    result = visual_qa.review_rendered(
        source_pdf=source_pdf,
        rendered_pdf=out_path,
        plan=plan,
        pages=pages,
        workdir=review_dir / "qa-visuale",
        ai=qa_ai,
        model=qa_model,
        timeout=args.review_timeout,
        batch=args.review_batch,
        # Stesso flag della traduzione: chi riprende una run morta la riprende
        # tutta, e il QA e' l'altra meta' della spesa. `getattr` perche' chi
        # costruisce `args` a mano (test, chiamate piu' vecchie) non ce l'ha.
        riprendi=getattr(args, "riprendi_lotti", False),
    )
    corrected, changes = result.plan, result.changes
    status.event("qa_corrections", changes=changes, validated_batches=result.validated_batches,
                 total_batches=result.total_batches)
    if not result.complete:
        print(
            "QA visuale INCOMPLETO: %d/%d lotti validati; serve revisione manuale."
            % (result.validated_batches, result.total_batches), file=sys.stderr,
        )
    if not changes:
        if result.complete:
            print("QA visuale completato: nessuna correzione proposta.")
        else:
            print("QA visuale: nessuna correzione validata; PDF non verificato.")
        return plan

    corrected_path = review_dir / "piano.controllato.json"
    save_plan(corrected_path, corrected)
    print("QA visuale: %d interventi -> %s" % (changes, corrected_path))
    raw_path = out_path.with_suffix(".pre-qa.pdf")
    shutil.copy2(out_path, raw_path)
    print("Output prima del QA conservato -> %s" % raw_path)
    render_plan(source_pdf, corrected, pages, fonts, args, out_path)
    return corrected


def esegui_riparazione(source_pdf, out_path, plan, pages, fonts, args, workdir):
    """Rilegge il PDF finito e ripara l'inglese rimasto (vedi verifica/riparazione).

    Un guasto della riparazione non butta via la traduzione: il PDF resta
    quello di prima, e il guasto si dichiara.
    """
    from comics_translator.verifica import riparazione

    def rendi(piano):
        render_plan(source_pdf, piano, pages, fonts, args, out_path)

    ai = args.riparazione_ai or args.review_ai
    model = args.riparazione_model or (args.review_model if ai == args.review_ai else None)
    lavoro = Path(workdir) / "riparazione"
    # La riparazione lavora sul piano in place: se cade a meta' si torna a
    # questo, che corrisponde al PDF di prima.
    copia_prima = copy.deepcopy(plan)
    try:
        plan, _ = riparazione.ripara(
            source_pdf, out_path, plan, pages, rendi, lavoro, ai=ai, model=model,
            timeout=args.review_timeout, giri=max(1, args.riparazione_giri),
            batch=args.review_batch)
    except Exception as exc:  # la traduzione fatta vale piu' della riparazione
        status.event("riparazione_fallita", errore=str(exc))
        print("  ! riparazione non eseguita: %s" % exc, file=sys.stderr)
        pre = out_path.with_name(out_path.stem + ".pre-riparazione.pdf")
        if pre.is_file():
            out_path.write_bytes(pre.read_bytes())
        return copia_prima
    save_plan(Path(workdir) / "piano.riparato.json", plan)
    return plan


def build_requested_outputs(source_pdf, out_path, args) -> None:
    """Genera gli output derivati soltanto dopo l'ultimo rendering."""
    if not args.dual:
        return
    dual_path = out_path.with_suffix(".dual.pdf")
    if dual_path.exists():
        dual_path.unlink()
    build_dual(source_pdf, out_path, dual_path)
    print("OK -> %s" % dual_path)


def main():
    ap = argparse.ArgumentParser(
        description="Traduce un PDF sovrascrivendo il testo in posizione (layout preservato).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="PDF di partenza")
    ap.add_argument("-o", "--output", type=Path, help="PDF tradotto (default: <input>.<lingua>.pdf)")
    ap.add_argument("--from", dest="source", default="auto", help="lingua di partenza ('auto' per rilevarla)")
    ap.add_argument("--to", dest="target", default="it", help="lingua di arrivo")
    ap.add_argument("--engine", choices=list(ENGINES), default="auto",
                    help="motore di traduzione; 'auto' sceglie deepl > argos > google, "
                         "'none' ricopia il testo originale (prova di layout, offline)")
    ap.add_argument("--pages", help="pagine da tradurre, es. '1-5,8' (default: tutte)")
    ap.add_argument("--mode", choices=["auto", "text", "image"], default="auto",
                    help="'text' usa il testo del PDF, 'image' fa OCR sui pixel, "
                         "'auto' sceglie pagina per pagina")
    ap.add_argument("--font", default="arial", help="famiglia o path del font per il testo tradotto")
    ap.add_argument("--image-font", default="comic",
                    help="font per le pagine trattate a immagine (i fumetti vogliono 'comic')")
    ap.add_argument("--ocr-dpi", type=int, default=400, help="risoluzione di rendering per l'OCR")
    ap.add_argument("--ocr-conf", type=float, default=0.5, help="confidenza minima dell'OCR")
    ap.add_argument("--no-localization", action="store_true",
                    help="disattiva P2 (solo confronto offline con il percorso precedente)")
    ap.add_argument("--no-ocr-recovery", action="store_true",
                    help="non esegue i ritagli OCR mirati; conserva mappa e deduplicazione P2")
    ap.add_argument("--max-bg-noise", type=float, default=15.0,
                    help="quanto puo' essere disomogeneo lo sfondo per poterlo coprire; "
                         "sopra questa soglia il testo si considera parte del disegno e "
                         "resta com'e' (un balloon sta sotto 5, un'onomatopea sopra 40)")
    ap.add_argument("--min-size", type=float, default=4.5, help="corpo minimo prima di troncare")
    ap.add_argument("--grow", type=float, default=1.0,
                    help="moltiplicatore del corpo di partenza (es. 1.05 per testo piu' grande)")
    ap.add_argument("--min-chars", type=int, default=2, help="ignora i blocchi piu' corti di cosi'")
    ap.add_argument("--no-expand", action="store_true",
                    help="non allargare i riquadri negli spazi bianchi (piu' fedele, piu' rimpicciolimenti)")
    ap.add_argument("--workers", type=int, default=4, help="richieste di traduzione in parallelo")
    ap.add_argument("--fill-white", action="store_true",
                    help="copre di bianco l'area originale (utile se restano aloni)")
    ap.add_argument("--dual", action="store_true", help="genera anche il PDF bilingue affiancato")
    ap.add_argument("--ocr", action="store_true", help="passa prima da ocrmypdf (PDF scansionati)")
    ap.add_argument("--ocr-lang", default="eng", help="lingua Tesseract per l'OCR, es. 'ita' o 'eng+ita'")
    ap.add_argument("--no-cache", action="store_true", help="non leggere/scrivere la cache di traduzione")

    rev = ap.add_argument_group("piano e revisione")
    rev.add_argument("--plan-out", type=Path, metavar="FILE",
                     help="scrive il piano di lavoro e si ferma, senza produrre il PDF")
    rev.add_argument("--plan-in", type=Path, metavar="FILE",
                     help="parte da un piano gia' pronto: niente OCR, niente traduzione")
    rev.add_argument("--ai-review", action="store_true",
                     help="fa rivedere il piano all'AI scelta, che guarda le pagine")
    rev.add_argument("--review-ai", default=ai_review.AI_DEFAULT,
                     choices=list(ai_review.AI_SUPPORTATE),
                     help="provider della revisione: codex (default) o claude")
    rev.add_argument("--review-model", default=None,
                     help="modello (default: gpt-5.4 per Codex, haiku per Claude)")
    rev.add_argument("--review-dir", type=Path, metavar="DIR",
                     help="cartella di lavoro della revisione (default: <output>.revisione)")
    rev.add_argument("--review-timeout", type=int, default=1800,
                     help="secondi massimi per ogni lotto della revisione")
    rev.add_argument("--review-batch", type=int, default=2,
                     help="quante pagine per chiamata al modello")
    rev.add_argument("--riprendi-lotti", dest="riprendi_lotti", action="store_true",
                     help="riprende i lotti gia' validati salvati nella cartella di "
                          "revisione - traduzione e QA visuale - invece di richiamare "
                          "il modello; vale per i lotti con lo stesso piano e prompt, "
                          "anche cambiando provider o modello")
    rev.add_argument("--ritenta-lotti", dest="ritenta_lotti", type=int,
                     default=ai_review.RITENTA_LOTTI, metavar="N",
                     help="nuovi tentativi per un lotto di traduzione non valido per "
                          "intero (timeout, CLI fallita, blocchi mancanti); 0 li disattiva")
    rev.add_argument("--no-sblocco-auto", dest="sblocco_auto", action="store_false",
                     help="non disegna i blocchi protetti che il modello ha tradotto e "
                          "classificato come dialogo/didascalia, anche se l'area e' sicura")
    rev.set_defaults(sblocco_auto=True)
    rev.add_argument("--no-riparazione", dest="riparazione", action="store_false",
                     help="non esegue il ciclo di riparazione dell'inglese rimasto sul PDF "
                          "finito (default: attivo con --ai-review)")
    rev.set_defaults(riparazione=None)
    rev.add_argument("--riparazione-giri", dest="riparazione_giri", type=int, default=2,
                     metavar="N", help="giri massimi di riparazione (rilettura e ridisegno)")
    rev.add_argument("--riparazione-ai", dest="riparazione_ai", default=None,
                     choices=list(ai_review.AI_SUPPORTATE),
                     help="provider della riparazione (default: quello della traduzione)")
    rev.add_argument("--riparazione-model", dest="riparazione_model", default=None,
                     help="modello della riparazione (default: quello della traduzione)")
    rev.add_argument("--ripara", action="store_true",
                     help="con --plan-in: rifa' sblocco, bande, divisione per balloon e "
                          "riparazione su un piano gia' tradotto, senza ritradurre")
    rev.add_argument("--qa-ai", dest="qa_ai", default=None,
                     choices=list(ai_review.AI_SUPPORTATE),
                     help="provider DIVERSO per il QA visuale (default: lo stesso della "
                          "traduzione). Un modello che rilegge il lavoro di un altro "
                          "trova errori che rileggendo il proprio non vedrebbe.")
    rev.add_argument("--qa-model", dest="qa_model", default=None,
                     help="modello del QA visuale (default: quello del provider QA scelto, "
                          "o il modello di traduzione se --qa-ai non e' passato)")
    rev.add_argument("--registra-cassette", dest="registra_cassette", type=Path,
                     metavar="DIR",
                     help="registra le risposte del modello in cassette riutilizzabili")
    rev.add_argument("--cassette", type=Path, metavar="DIR",
                     help="replay delle risposte registrate: nessuna chiamata reale "
                          "al modello, cassette mancante = errore duro")
    rev.add_argument("--visual-qa", dest="visual_qa", action="store_true",
                     help="dopo il rendering confronta automaticamente originale e traduzione")
    rev.add_argument("--no-visual-qa", dest="visual_qa", action="store_false",
                     help="disattiva il controllo visuale automatico post-render")
    rev.set_defaults(visual_qa=None)
    ap.add_argument("--no-audit", dest="audit", action="store_false",
                    help="non misurare il PDF prodotto con l'audit raster offline")
    ap.set_defaults(audit=True)
    ap.add_argument("--audit-ocr", dest="audit_ocr", action="store_true",
                    help="nell'audit rilegge anche il PDF finito con l'OCR e cerca righe "
                         "in inglese (default: attivo con --ai-review)")
    ap.add_argument("--no-audit-ocr", dest="audit_ocr", action="store_false")
    ap.set_defaults(audit_ocr=None)
    ap.add_argument("--run-manifest", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.visual_qa is None:
        # Con la revisione AI il controllo del risultato e' parte dello stesso flusso.
        args.visual_qa = args.ai_review
    if args.audit_ocr is None:
        args.audit_ocr = args.ai_review
    if args.riparazione is None:
        args.riparazione = args.ai_review or args.ripara

    if not args.input.is_file():
        sys.exit("File non trovato: %s" % args.input)

    # Se il QA gira con un provider diverso dalla traduzione, controllane login
    # e CLI ORA: fallire dopo 40 minuti di traduzione per un QA non autenticato
    # sarebbe uno spreco. Solo su chiamate reali: in replay a cassette nessun
    # provider viene contattato. Il default (QA = provider di traduzione) e'
    # gia' validato dalla fase di traduzione, quindi non lo si ricontrolla.
    if (args.ai_review and args.visual_qa and args.qa_ai
            and args.qa_ai != args.review_ai and not args.cassette):
        ai_review.prepare_provider(args.qa_ai, args.qa_model)

    source_pdf = run_ocr(args.input, args.ocr_lang) if args.ocr else args.input
    out_path = args.output or args.input.with_suffix(".%s.pdf" % args.target)
    if out_path.parent and str(out_path.parent):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    report = (status.RunReport.load(args.run_manifest) if args.run_manifest else
              status.RunReport.create(out_path.with_suffix(".run"), vars(args), RADICE))
    status.ACTIVE_REPORT.set(report)
    report.data["translation_config"] = vars(args).copy()
    if args.cassette and args.registra_cassette:
        sys.exit("--cassette e --registra-cassette sono mutuamente esclusivi.")
    if args.cassette:
        cassette.attiva(args.cassette, registra=False)
        status.event("cassette", mode="replay", dir=str(args.cassette))
    elif args.registra_cassette:
        cassette.attiva(args.registra_cassette, registra=True)
        status.event("cassette", mode="registrazione", dir=str(args.registra_cassette))
    print("Registro esiti: %s" % report.path)
    fonts = load_fontset(args.font)
    doc = fitz.open(source_pdf)
    pages = parse_pages(args.pages, doc.page_count)
    report.data["requested_pages"] = [p + 1 for p in pages]
    report.artifact("source_pdf", source_pdf, pages=doc.page_count)
    if not pages:
        doc.close()
        raise ValueError("nessuna pagina selezionata")

    if args.plan_in:
        plan = load_plan(args.plan_in)
        print("Piano caricato da %s: %d pagine." % (args.plan_in, len(pages)))
        doc.close()
        if args.ripara:
            localization.enforce_localization(plan)
            if args.sblocco_auto:
                esegui_sblocco_automatico(plan, source_pdf)
        pages = render_plan(source_pdf, plan, pages, fonts, args, out_path)
        plan = run_visual_qa(source_pdf, out_path, plan, pages, fonts, args)
        if args.ripara and args.riparazione:
            workdir = Path(args.review_dir) if args.review_dir else out_path.with_suffix(".revisione")
            plan = esegui_riparazione(source_pdf, out_path, plan, pages, fonts, args, workdir)
        build_requested_outputs(source_pdf, out_path, args)
        if args.ripara:
            esegui_audit_struttura(source_pdf, out_path, plan, args, report)
        status.record_coverage(plan, render_only=True)
        return finish_translation(report)

    print("Analizzo %d pagine di %s..." % (len(pages), args.input.name))
    all_blocks: list = []
    per_page: dict[int, list[Block]] = {}
    image_jobs: dict[int, list] = {}
    sources: dict[int, dict] = {}
    analysis = status.PhaseResult("analysis", len(pages), unit="pages")
    status.start_phase("analysis", len(pages), "pages")

    for pno in pages:
        blocks: list[Block] = []
        sources[pno] = inventory_data.new_inventory()
        sources[pno]["localizzazione_attiva"] = not args.no_localization
        if args.mode != "image":
            blocks, images = extract_blocks(doc[pno], pno, args.min_chars, inventory=sources[pno],
                                             localize=not args.no_localization)
            if not args.no_localization:
                blocks = localization.deduplicate_native(blocks, sources[pno])
            if not args.no_expand:
                expand_boxes(doc[pno], blocks, images)

        # pagina senza testo estraibile: e' un'immagine, si passa all'OCR
        if args.mode == "image" or (args.mode == "auto" and not blocks):
            jobs, nlines, skipped = plan_page(
                doc[pno], pno, args.ocr_dpi, args.ocr_conf, not args.no_expand,
                args.max_bg_noise, inventory=sources[pno],
                localize=not args.no_localization, recover=not args.no_ocr_recovery,
            )
            print("  p%d: immagine, OCR -> %d righe in %d gruppi%s"
                  % (pno + 1, nlines, len(jobs),
                     " (%d scartati: testo sul disegno)" % skipped if skipped else ""))
            image_jobs[pno] = jobs
            all_blocks.extend(jobs)
            excluded_lines = sum(line["stato"] == "esclusa" and line.get("motivo") != "duplicate"
                                 for line in sources[pno]["righe"])
            if skipped or excluded_lines or not jobs:
                analysis.issues.append(status.issue(
                    "ocr_groups_excluded" if skipped else "source_lines_excluded" if excluded_lines else "no_detected_text",
                    "%d gruppi e %d righe esclusi; dettagli nel piano v2, copertura non certificata"
                    % (skipped, excluded_lines),
                    [pno + 1],
                ))
            analysis.completed += 1
            analysis.pages.append(pno + 1)
            status.event("ocr_page", page=pno + 1, lines=nlines, groups=len(jobs), excluded=skipped,
                         observed_lines=len(sources[pno]["righe"]), excluded_lines=excluded_lines)
            continue

        per_page[pno] = blocks
        all_blocks.extend(blocks)
        analysis.completed += 1
        analysis.pages.append(pno + 1)
        excluded_lines = sum(line["stato"] == "esclusa" and line.get("motivo") != "duplicate"
                                 for line in sources[pno]["righe"])
        if excluded_lines:
            analysis.issues.append(status.issue("source_lines_excluded",
                                               "%d righe native escluse; dettagli nel piano v2" % excluded_lines,
                                               [pno + 1]))
        if not blocks:
            analysis.issues.append(status.issue("no_detected_text", "nessun blocco", [pno + 1]))
    status.finish_phase(analysis)

    chars = sum(len(b.text) for b in all_blocks)
    print("Trovati %d blocchi, %d caratteri." % (len(all_blocks), chars))

    engine = resolve_engine(args.engine, args.source, args.target)
    cache_path = None if args.no_cache else out_path.with_suffix(".cache.json")
    translator = Translator(engine, args.source, args.target, cache_path)
    # argos gira in locale: il parallelismo non aiuta e il modello non e' thread-safe
    workers = 1 if engine in ("argos", "none") else max(1, args.workers)
    translator.translate_all(all_blocks, workers)

    # --- il piano: lo stato intermedio che la revisione puo' riscrivere -------
    plan = build_plan(doc, pages, per_page, image_jobs, {
        "sorgente": args.input.name,
        "lingua_da": args.source,
        "lingua_a": args.target,
        "motore": engine,
        "sorgente_sha256": report.data["artifacts"]["source_pdf"]["sha256"],
    }, sources=sources)

    # Quello che l'analisi ha scartato prima della traduzione: si dichiara qui,
    # perche' da qui in poi non lo vede piu' nessuno (ne' il modello, ne' il QA,
    # ne' l'audit, che misura la copertura sulla sorgente sigillata).
    segnala_testo_scartato_come_disegno(plan)

    # P3 e' una fase esplicita: una struttura incompleta non puo' produrre
    # uno stato globale complete, anche quando il renderer conserva la pagina.
    from comics_translator.sorgente.page_structure import structure_issues
    p3 = status.PhaseResult("structure", len(pages), unit="pages")
    p3.completed = len(pages)
    p3.pages = [p + 1 for p in pages]
    p3.issues.extend(structure_issues(plan))
    status.start_phase("structure", len(pages), "pages")
    status.finish_phase(p3)

    piano_sha256 = None
    if args.ai_review:
        workdir = Path(args.review_dir) if args.review_dir else out_path.with_suffix(".revisione")
        reviewed = ai_review.review(
            plan, doc, pages, workdir, args.source, args.target,
            ai=args.review_ai, model=args.review_model, timeout=args.review_timeout,
            batch=args.review_batch, riprendi=args.riprendi_lotti,
            ritenta=args.ritenta_lotti,
        )
        plan, changed = ai_review.apply_review(plan, reviewed.plan)
        piano_sigillato = workdir / "piano.json"
        piano_sha256 = (hashlib.sha256(piano_sigillato.read_bytes()).hexdigest()
                        if piano_sigillato.is_file() else None)
        if not reviewed.outcome.complete:
            print("Traduzione %s: %d/%d lotti; output NON approvato."
                  % (reviewed.outcome.status, reviewed.outcome.completed, reviewed.outcome.requested))
            # Un lotto non tradotto non deve cancellare/riscrivere l'originale.
            if engine == "none":
                valid_ids = set(reviewed.outcome.ids)
                for page in plan["pagine"]:
                    for block in page["blocchi"]:
                        if block["id"] not in valid_ids:
                            block["salta"] = True
                            block["_render"]["unreviewed"] = True
        print("  %d campi rivisti" % changed)
        localization.enforce_localization(plan)
        if args.sblocco_auto:
            esegui_sblocco_automatico(plan, source_pdf)
        segnala_traduzioni_soppresse(plan)
        segnala_traduzioni_copiate(plan)
        segnala_dialogo_protetto(plan)
        applicato = workdir / "piano.applicato.json"
        save_plan(applicato, plan)
        print("  piano fuso -> %s  (riapplicabile con --plan-in)" % applicato)
    else:
        outcome = status.PhaseResult("translation", len(pages), len(pages), unit="pages")
        if engine == "none":
            outcome.issues.append(status.issue("no_translation_engine", "prova di layout, non traduzione"))
        status.finish_phase(outcome)

    if args.plan_out:
        save_plan(args.plan_out, plan)
        doc.close()
        print("Piano scritto -> %s" % args.plan_out)
        print("Rivedilo e poi applicalo con:  --plan-in %s" % args.plan_out)
        status.record_coverage(plan)
        _chiudi_cassette(report, piano_sha256, args)
        return finish_translation(report)

    doc.close()
    pages = render_plan(source_pdf, plan, pages, fonts, args, out_path)
    plan = run_visual_qa(source_pdf, out_path, plan, pages, fonts, args)
    if args.ai_review and args.riparazione:
        plan = esegui_riparazione(source_pdf, out_path, plan, pages, fonts, args, workdir)
    build_requested_outputs(source_pdf, out_path, args)
    esegui_audit_struttura(source_pdf, out_path, plan, args, report)
    if not args.visual_qa:
        report.data["issues"].append(status.issue("qa_disabled", "PDF non controllato visualmente"))
    status.record_coverage(plan)
    _chiudi_cassette(report, piano_sha256, args)
    return finish_translation(report)


# Difetti veri del rendering: il candidato ha toccato o lasciato qualcosa che
# non doveva. `righe_non_coperte` NON e' qui: e' il resoconto delle esclusioni
# volute (crediti, insegne, lettering disegnato), e alzarlo non e' un guasto.
DIFETTI_AUDIT = (
    ("pixel_fuori_area", "audit_pixel_fuori_area",
     "il candidato ha cambiato pixel fuori dalle aree dichiarate dal piano"),
    ("lobi_vuoti", "audit_lobi_vuoti",
     "lobi riempiti dal renderer e rimasti senza inchiostro di testo"),
    ("voci_non_disegnate", "audit_voci_non_disegnate",
     "voci del piano che nel PDF non hanno lasciato traccia"),
)


def esegui_audit_struttura(source_pdf, out_path, plan, args, report):
    """Misura il PDF appena prodotto, invece di limitarsi a produrlo.

    `structure_audit` esisteva dal 30 agosto ma restava un comando da lanciare
    a mano: il verdetto su una run c'era solo se qualcuno si ricordava di
    chiederlo. Qui gira da solo alla fine del rendering, sugli stessi file gia'
    scritti - nessuna cattura, nessuna chiamata al modello, nessun credito -
    e lascia il rapporto accanto al PDF.

    La cartella e' sempre nuova (l'audit rifiuta di sovrascrivere le prove di
    un giro precedente), quindi porta l'orario nel nome.
    """
    from comics_translator.verifica import structure_audit

    if not getattr(args, "audit", True):
        return None
    if not out_path.is_file():
        return None

    quando = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out_path.with_name(out_path.stem + ".audit") / quando
    pagine = len(plan.get("pagine", []))
    status.start_phase("structure_audit", pagine, "pages")
    esito = status.PhaseResult("structure_audit", pagine, unit="pages")
    try:
        report_audit = structure_audit.audit(source_pdf, plan, out_path, out_dir)
    except (ValueError, OSError) as exc:
        # Un audit che non gira non e' un audit pulito: va detto e conta.
        esito.issues.append(status.issue("audit_non_eseguito", str(exc)))
        status.finish_phase(esito)
        print("  ! audit raster non eseguito: %s" % exc)
        return None

    pagine_con_difetti = set()
    for chiave, codice, motivo in DIFETTI_AUDIT:
        colpite = [p["pagina"] for p in report_audit["pagine"] if p.get(chiave)]
        if colpite:
            pagine_con_difetti.update(colpite)
            esito.issues.append(status.issue(codice, motivo, colpite))
    residui = [p["pagina"] for p in report_audit["pagine"]
               if any(r.get("leggibile") for r in p.get("righe_scoperte") or [])]
    if residui:
        pagine_con_difetti.update(residui)
        esito.issues.append(status.issue(
            "audit_righe_scoperte",
            "testo originale leggibile rimasto scoperto dentro un'area "
            "riscritta: nessuna maschera lo cancella", residui))
    # Priorita' n.1: nessun testo reale lasciato in inglese. Fra le righe
    # scoperte, quelle che sono parole vere (non suoni ne' spazzatura OCR) sono
    # il segnale azionabile - l'operatore le recupera una per una.
    inglese = report_audit.get("inglese_residuo") or []
    if inglese:
        pagine_inglese = sorted({v["pagina"] for v in inglese})
        pagine_con_difetti.update(pagine_inglese)
        esito.issues.append(status.issue(
            "inglese_residuo",
            "testo vero rimasto in inglese sulla pagina: %s"
            % "; ".join("p%s %r" % (v["pagina"], v["testo"]) for v in inglese),
            pagine_inglese))
    if getattr(args, "audit_ocr", False):
        pagine_ocr = verifica_inglese_resa(out_path, plan, out_dir)
        if pagine_ocr:
            pagine_con_difetti.update(pagine_ocr)
            esito.issues.append(status.issue(
                "inglese_ocr_render",
                "righe in inglese rilette dal PDF finito, fuori dai blocchi lasciati "
                "in originale di proposito (vedi inglese_render.json)", pagine_ocr))
    orfane = [p["pagina"] for p in report_audit["pagine"]
              if (p.get("copertura") or {}).get("righe_orfane")]
    if orfane:
        pagine_con_difetti.update(orfane)
        esito.issues.append(status.issue(
            "audit_righe_orfane",
            "righe della sorgente che nessun blocco rivendica: originale non "
            "cancellato da nessuna maschera", orfane))
    doppie = [p["pagina"] for p in report_audit["pagine"]
              if (p.get("copertura") or {}).get("righe_duplicate")]
    if doppie:
        pagine_con_difetti.update(doppie)
        esito.issues.append(status.issue(
            "audit_righe_duplicate", "righe della sorgente attribuite a due voci", doppie))
    non_misurabili = [p["pagina"] for p in report_audit["pagine"]
                      if not (p.get("copertura") or {}).get("misurabile")]
    if non_misurabili:
        pagine_con_difetti.update(non_misurabili)
        esito.issues.append(status.issue(
            "audit_pagina_non_misurabile",
            "pagina senza osservazioni di sorgente: non misurabile, non pulita",
            non_misurabili))

    esito.completed = pagine - len(pagine_con_difetti)
    esito.pages = sorted(pagine_con_difetti)
    status.finish_phase(esito)
    percorso = out_dir / "audit.json"
    report.artifact("audit_report", percorso, pagine_bloccanti=report_audit["pagine_bloccanti"])
    if pagine_con_difetti:
        print("  ! audit raster: difetti su %d pagine (%s) -> %s"
              % (len(pagine_con_difetti),
                 ", ".join(str(n) for n in sorted(pagine_con_difetti)), percorso))
    else:
        print("  audit raster: nessun difetto di rendering -> %s" % percorso)
    escluse = sorted({p["pagina"] for p in report_audit["pagine"]
                      if (p.get("copertura") or {}).get("righe_escluse_per_scelta")})
    if escluse:
        print("    righe lasciate in originale da blocchi saltati, su pagine: %s"
              % ", ".join(str(n) for n in escluse))
    return report_audit


def verifica_inglese_resa(out_path, plan, out_dir):
    """Rilegge il PDF finito con un OCR nuovo e riporta le pagine con inglese.

    E' la misura indipendente dal piano: vede anche i blocchi saltati per
    intero, i gruppi mai diventati blocco e le traduzioni rifiutate, che
    `inglese_residuo` per costruzione non puo' vedere. Un guasto dell'OCR non
    ferma la run: si dichiara, perche' un controllo non eseguito non e' pulito.
    """
    from comics_translator.verifica import inglese_render

    try:
        rapporto = inglese_render.verifica_pdf(out_path, piano=plan)
    except Exception as exc:  # OCR assente o rotto: dichiararlo, non fallire
        status.event("inglese_ocr_non_eseguito", errore=str(exc))
        print("  ! controllo OCR dell'inglese non eseguito: %s" % exc)
        return []
    percorso = Path(out_dir) / "inglese_render.json"
    percorso.write_text(json.dumps(rapporto, ensure_ascii=False, indent=2), encoding="utf-8")
    report = status.ACTIVE_REPORT.get()
    if report:
        report.artifact("inglese_render", percorso,
                        righe_non_dichiarate=rapporto["righe_non_dichiarate"])
    if rapporto["pagine_con_inglese"]:
        print("  ! OCR del PDF finito: %d righe in inglese su pagine %s"
              % (rapporto["righe_non_dichiarate"],
                 ", ".join(str(n) for n in rapporto["pagine_con_inglese"])))
    return rapporto["pagine_con_inglese"]


def _chiudi_cassette(report, piano_sha256, args):
    """Chiude le cassette e registra l'esito negli eventi del manifest padre."""
    riepilogo = cassette.chiudi(
        unito_sha256=report.data.get("artifacts", {}).get("source_pdf", {}).get("sha256"),
        piano_sha256=piano_sha256,
        sessione={"ai": args.review_ai, "model": args.review_model,
                  "batch": args.review_batch, "timeout": args.review_timeout,
                  "pagine": args.pages, "sorgente": args.input.name},
    )
    if riepilogo:
        status.event("cassette_chiusa", **riepilogo)


def finish_translation(report):
    # Il padre termina la propria fase dopo il ritorno del sottoprocesso.
    if report.data.get("translation_config", {}).get("run_manifest"):
        phases = [p for key, p in report.data["phases"].items() if key != "pipeline_translation"]
        code = (1 if any(p["status"] == "failed" for p in phases)
                and "output_pdf" not in report.data["artifacts"] else
                2 if report.data["issues"] or any(p["status"] != "complete" for p in phases) else 0)
        report.save()
    else:
        code = report.finish()
    print("PDF/risultato: bozza non certificata; esito tecnico %s. Registro: %s"
          % ({0: "complete", 1: "failed", 2: "partial"}[code], report.path))
    return code


if __name__ == "__main__":
    sys.exit(status.cli_entry(main))
