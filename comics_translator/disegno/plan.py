"""Il piano di lavoro: il documento che sta tra l'analisi e il disegno.

La pipeline e' spezzata in due meta' che comunicano solo attraverso questo file:

    analizza  ->  piano.json  ->  disegna

Nel mezzo il piano puo' essere riscritto da chiunque — un modello che guarda le
pagine, o tu con un editor. Chi lo riscrive non tocca il codice e non rifa' ne'
l'OCR ne' la traduzione: cambia il documento, e l'algoritmo esegue.

Il formato v2 separa tre responsabilita':
  - `sorgente`: osservazioni originali con hash, righe e regioni persistenti;
  - blocchi: testo e decisioni; solo EDITABLE e' modificabile dal revisore;
  - `_render`: geometrie e stile necessari a ridisegnare, non modificabili
    dalle correzioni QA. I piani v1 restano leggibili senza migrazione implicita.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys

from comics_translator import pipeline_status
from comics_translator.sorgente import source_inventory as inventory_data

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


VERSION = 2
COMPOUND_PROFILE_MIN_CHARS = 120
COMPOUND_AREA_RATIO = 0.08
PAGE_WIDE_COMPOUND_MIN_WIDTH_RATIO = 0.75
MERGE_MIN_FILL_OVERLAP = 0.80
CONTAINED_BLOCK_MIN_RATIO = 0.70


def block_id(page_no: int, text: str) -> str:
    """Alias legacy v1: da solo NON distingue ripetizioni nella stessa pagina."""
    return "p%d-%s" % (page_no + 1, hashlib.md5(text.encode("utf-8")).hexdigest()[:6])


# --------------------------------------------------------------------------- #
# Serializzazione
# --------------------------------------------------------------------------- #

def _text_block_entry(blk, page_no: int, page_rect) -> dict:
    rect = blk.target
    return {
        "id": block_id(page_no, blk.text),
        "_source_region_id": getattr(blk, "source_region_id", None),
        "pos": "x=%d%% y=%d%%" % (
            round(100 * rect.x0 / page_rect.width),
            round(100 * rect.y0 / page_rect.height),
        ),
        "originale": blk.text,
        "automatica": blk.translated,
        "testo": blk.translated,
        "salta": False,
        "unisci_con": None,
        "corpo": None,
        "allinea": "centro" if blk.centered else "sinistra",
        "_render": {
            "tipo": "testo",
            "rect": list(blk.rect),
            "box": list(blk.target),
            "size": blk.size,
            "color": list(blk.color),
            "bold": blk.bold,
            "italic": blk.italic,
        },
    }


def _image_block_entry(job, page_no: int, page_rect) -> dict:
    x0, y0, x1, y1 = [v * job.scale for v in job.box_px]
    page_area = float(page_rect.width) * float(page_rect.height)
    area_ratio = (abs(x1 - x0) * abs(y1 - y0) / page_area) if page_area else 0.0
    # Due aree di riempimento distinte sono sempre due spazi diversi: nei piani
    # raccolti finora nessun blocco a piu' profili si e' reso bene, nemmeno
    # quelli corti, quindi qui la lunghezza non conta. Serve invece per il
    # profilo unico molto largo, dove il testo breve e' spesso legittimo.
    candidato = bool(getattr(job, "scartato_come_disegno", False))
    compound = not candidato and (len(job.fill_poly) > 1 or (
        len(job.text) >= COMPOUND_PROFILE_MIN_CHARS
        and area_ratio >= COMPOUND_AREA_RATIO
    ))
    entry = {
        "id": block_id(page_no, job.text),
        "_source_region_id": getattr(job, "source_region_id", None),
        "pos": "x=%d%% y=%d%%" % (
            round(100 * x0 / page_rect.width),
            round(100 * y0 / page_rect.height),
        ),
        "originale": job.text,
        "automatica": job.translated,
        "testo": job.translated,
        # Due grandi aree bianche nello stesso gruppo indicano di solito che
        # il clustering ha unito titoli, colonne o balloon distinti. Anche un
        # solo profilo che copre gran parte della pagina segnala lo stesso
        # errore: un blocco editoriale con dentro credits o colonne vicine.
        # Il piano non sa ancora dividerli: conservarli e' piu' sicuro che
        # cancellarli.
        "salta": compound or candidato,
        "unisci_con": None,
        "corpo": None,
        "allinea": "centro" if getattr(job, "centered", True) else "sinistra",
        "_render": {
            "tipo": "immagine",
            "box_px": list(job.box_px),
            "fill_poly": [[list(p) for p in profilo] for profilo in job.fill_poly],
            "scale": job.scale,
            "bg": list(job.bg),
            "light_text": job.light_text,
            "upper": job.upper,
            "base_size": job.base_size,
            "box_pt": [x0, y0, x1, y1],
            "compound": compound,
            "area_ratio": round(area_ratio, 4),
        },
    }
    if candidato:
        # Gap C: un gruppo che l'analisi ha scartato come disegno ma che ha la
        # forma del testo. Diventa un blocco SALTATO, cosi' arriva al modello
        # (che lo traduce e dice se e' dialogo o grafica) senza che nulla venga
        # coperto: disegnarlo o no lo decide lo sblocco automatico, dopo.
        entry["_render"]["scartato_come_disegno"] = True
    return entry


def _protect_page_if_wide_compound(entries: list, page_width: float) -> bool:
    """Conserva una pagina se un gruppo ne attraversa quasi tutta la larghezza.

    In questi casi il gruppo protetto contiene di solito testata, colonne e
    frammenti di testo vicini. Tradurre soltanto gli altri blocchi produce una
    pagina ibrida, con frasi spezzate e struttura grafica incoerente.
    """
    if page_width <= 0:
        return False
    wide_compound = any(
        (render := (entry.get("_render") or {})).get("tipo") == "immagine"
        and render.get("compound")
        and len(render.get("box_pt") or []) == 4
        and abs(render["box_pt"][2] - render["box_pt"][0]) / page_width
        >= PAGE_WIDE_COMPOUND_MIN_WIDTH_RATIO
        for entry in entries
    )
    if not wide_compound:
        return False
    for entry in entries:
        render = entry.get("_render") or {}
        if render.get("tipo") != "immagine":
            continue
        entry["salta"] = True
        render["page_protected"] = True
    return True


def _protect_contained_blocks(entries: list) -> int:
    """Conserva il frammento OCR che sta dentro il rettangolo di un altro blocco.

    Succede quando l'OCR estrae due volte lo stesso testo: una come parte del
    gruppo grande e una come frammento a se'. I due blocchi vengono poi scritti
    uno sopra l'altro e diventano illeggibili. Il frammento non aggiunge nulla
    - le sue parole sono gia' nel gruppo che lo contiene - quindi si lascia
    l'originale sotto e si scrive solo il blocco grande.
    """
    boxes = []
    for entry in entries:
        render = entry.get("_render") or {}
        if render.get("scartato_come_disegno"):
            # Un candidato del Gap C copre spesso un'onomatopea grande: se
            # contasse come contenitore farebbe saltare il balloon vero che ci
            # sta dentro, e da contenuto non ha nulla da proteggere.
            continue
        box = render.get("box_px") if render.get("tipo") == "immagine" else None
        if box and len(box) == 4:
            boxes.append((entry, [float(v) for v in box]))

    protetti = 0
    for entry, box in boxes:
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        for altro, altro_box in boxes:
            if altro is entry:
                continue
            area_altro = max(1.0, (altro_box[2] - altro_box[0]) * (altro_box[3] - altro_box[1]))
            if area > area_altro:
                continue
            ix = max(0.0, min(box[2], altro_box[2]) - max(box[0], altro_box[0]))
            iy = max(0.0, min(box[3], altro_box[3]) - max(box[1], altro_box[1]))
            if (ix * iy) / area < CONTAINED_BLOCK_MIN_RATIO:
                continue
            entry["salta"] = True
            entry["_render"]["contenuto_in"] = altro["id"]
            protetti += 1
            break
    return protetti


def build_plan(doc, pages, per_page: dict, image_jobs: dict, meta: dict, *, sources=None) -> dict:
    plan = {
        **meta,
        "versione": VERSION,
        "pagine": [],
    }
    for pno in pages:
        page_rect = doc[pno].rect
        entries = []
        for job in image_jobs.get(pno, []):
            entries.append(_image_block_entry(job, pno, page_rect))
        for blk in per_page.get(pno, []):
            entries.append(_text_block_entry(blk, pno, page_rect))
        page = {
            "numero": pno + 1,
            "indice": pno,
            "modo": "immagine" if pno in image_jobs else "testo",
            "larghezza_pt": round(page_rect.width, 1),
            "altezza_pt": round(page_rect.height, 1),
            "blocchi": entries,
        }
        inventory_data.attach_sources(page, (sources or {}).get(pno))
        _protect_page_if_wide_compound(entries, float(page_rect.width))
        _protect_contained_blocks(entries)
        if page["sorgente"].get("localizzazione_attiva"):
            from comics_translator.sorgente.text_localization import finalize_page
            finalize_page(page)
        from comics_translator.sorgente.page_structure import activate_validated_units, build_structure
        page["struttura"] = build_structure(page)
        activate_validated_units(page)
        plan["pagine"].append(page)
    inventory_data.validate_sources(plan)
    return plan


def save_plan(path, plan: dict, *, projection=False) -> None:
    # Scrittura atomica: il piano e' lo stato da cui si riprende, e un piano
    # troncato da un crash a meta' scrittura vale meno di un piano assente,
    # perche' sembra leggibile.
    if not projection:
        inventory_data.validate_sources(plan)
    pipeline_status.scrivi_atomico(
        path, json.dumps(plan, ensure_ascii=False, indent=1, allow_nan=False))


def load_plan(path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            plan = json.load(fh)
    except FileNotFoundError:
        sys.exit("Piano non trovato: %s" % path)
    except json.JSONDecodeError as exc:
        sys.exit("Piano illeggibile (%s): %s" % (path, exc))
    if not isinstance(plan, dict) or "pagine" not in plan:
        sys.exit("Piano malformato: manca l'elenco 'pagine' in %s" % path)
    inventory_data.validate_sources(plan)
    return plan


def migrate_v1(plan: dict) -> dict:
    """Migrazione esplicita, non distruttiva; mantiene id e istruzioni di disegno."""
    pipeline_status.plan_index(plan)  # id legacy ambigui: errore, mai rinumerazione tacita
    if inventory_data.schema_version(plan) == 2:
        inventory_data.validate_sources(plan)
        return copy.deepcopy(plan)
    result = copy.deepcopy(plan)
    result["versione"] = VERSION
    result["migrazione"] = {"da": 1, "righe_originali_disponibili": False}
    for page in result["pagine"]:
        inventory_data.attach_sources(page, preserve_ids=True)
    inventory_data.validate_sources(result)
    return result


def migrate_plan_file(source, destination):
    """Scrive solo una nuova copia; nessuna sovrascrittura, nemmeno accidentale."""
    from pathlib import Path
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("la migrazione richiede un file di destinazione diverso")
    result = migrate_v1(load_plan(source))
    payload = json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Migrazione esplicita del piano v1 -> v2")
    parser.add_argument("--migrate-v2", required=True, metavar="PIANO")
    parser.add_argument("--out", required=True, metavar="COPIA")
    args = parser.parse_args(argv)
    try:
        migrate_plan_file(args.migrate_v2, args.out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Piano v2 salvato in %s; sorgente invariata." % args.out)


# --------------------------------------------------------------------------- #
# Fusione delle revisioni
# --------------------------------------------------------------------------- #

EDITABLE = ("testo", "salta", "unisci_con", "corpo", "allinea")


def _render_box(entry: dict):
    render = entry.get("_render") or {}
    box = render.get("box_px") if render.get("tipo") == "immagine" else render.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [float(v) for v in box]
    except (TypeError, ValueError):
        return None


def _fill_bbox(block: dict) -> list | None:
    """Rettangolo che racchiude i profili di riempimento del blocco."""
    profili = (block.get("_render") or {}).get("fill_poly") or []
    punti = [p for profilo in profili for p in profilo if len(p) >= 2]
    if not punti:
        return None
    xs = [float(p[0]) for p in punti]
    ys = [float(p[1]) for p in punti]
    return [min(xs), min(ys), max(xs), max(ys)]


def _same_balloon(source: dict, target: dict) -> bool:
    """Due frammenti dello stesso balloon condividono quasi tutto il profilo.

    E' la differenza fra il caso buono, cioe' l'OCR che spezza una battuta in
    due righe dentro un'unica nuvoletta, e il caso cattivo, cioe' due nuvolette
    diverse che si toccano: fondere le seconde riscrive il testo nell'unione dei
    due rettangoli, sborda dai bordi e lascia vuota la nuvoletta di sotto.
    """
    a, b = _fill_bbox(source), _fill_bbox(target)
    if a is None or b is None:
        return True  # senza profili non si puo' giudicare: decide il resto
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aree = [max(1.0, (a[2] - a[0]) * (a[3] - a[1])),
            max(1.0, (b[2] - b[0]) * (b[3] - b[1]))]
    return (ix * iy) / min(aree) >= MERGE_MIN_FILL_OVERLAP


def _merge_geometry_ok(source: dict, target: dict) -> bool:
    """Una fusione AI deve assomigliare a due frammenti dello stesso balloon."""
    a, b = _render_box(source), _render_box(target)
    if a is None or b is None:
        return False
    ar, br = source.get("_render") or {}, target.get("_render") or {}
    if ar.get("tipo") != br.get("tipo"):
        return False

    aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
    bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    if intersection / min(aw * ah, bw * bh) > 0.55:
        return False

    gap_x = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    if gap_x > 0 and gap_y > 0:
        return False
    nearby = max(20.0, 0.75 * min(max(aw, ah), max(bw, bh)))
    if max(gap_x, gap_y) > nearby:
        return False
    original_chars = len(str(source.get("originale", ""))) + len(str(target.get("originale", "")))
    return original_chars <= 320


def validate_merges(plan: dict) -> list:
    """Elenca fusioni impossibili o pericolose senza modificare il piano."""
    by_id, page_by_id = {}, {}
    for page_index, page in enumerate(plan.get("pagine", [])):
        for block in page.get("blocchi", []):
            if block.get("id"):
                by_id[block["id"]] = block
                page_by_id[block["id"]] = page_index

    problems, targets = [], set()
    for source_id, source in by_id.items():
        target_id = source.get("unisci_con")
        if not target_id:
            continue
        target = by_id.get(target_id)
        if target_id == source_id or target is None:
            problems.append("%s -> %s: destinazione inesistente" % (source_id, target_id))
        elif page_by_id[source_id] != page_by_id[target_id]:
            problems.append("%s -> %s: pagine diverse" % (source_id, target_id))
        elif target_id in targets:
            problems.append("%s -> %s: destinazione gia' usata" % (source_id, target_id))
        elif not _merge_geometry_ok(source, target):
            problems.append("%s -> %s: blocchi non adiacenti" % (source_id, target_id))
        elif not _same_balloon(source, target):
            problems.append("%s -> %s: nuvolette diverse" % (source_id, target_id))
        else:
            from comics_translator.sorgente.text_localization import merge_conflicts
            conflicts = merge_conflicts(plan["pagine"][page_by_id[source_id]], source, target)
            if conflicts:
                problems.append("%s -> %s: fusione sopra testo protetto" % (source_id, target_id))
            else:
                targets.add(target_id)

    for start in by_id:
        seen, current = set(), start
        while current in by_id and by_id[current].get("unisci_con"):
            if current in seen:
                problems.append("%s: ciclo di fusione" % start)
                break
            seen.add(current)
            current = by_id[current]["unisci_con"]
    return sorted(set(problems))


def merge_review(plan: dict, reviewed: dict) -> tuple:
    """Riporta nel piano i soli campi modificabili della versione rivista.

    Tutto il resto — id, `_render`, struttura delle pagine — viene ignorato:
    una revisione non puo' inventare blocchi o spostare rettangoli per errore.
    """
    by_id = {}
    for page in reviewed.get("pagine", []):
        for blk in page.get("blocchi", []):
            if isinstance(blk, dict) and blk.get("id"):
                by_id[blk["id"]] = blk

    changed, unknown = 0, 0
    seen = set()
    for page in plan["pagine"]:
        for blk in page["blocchi"]:
            seen.add(blk["id"])
            revised = by_id.get(blk["id"])
            if not revised:
                continue
            for field in EDITABLE:
                if field not in revised:
                    continue
                value = revised[field]
                if field == "testo":
                    if not isinstance(value, str) or not value.strip():
                        continue
                    value = value.strip()
                render = blk.get("_render") or {}
                if (field == "salta" and value is False
                        and (render.get("compound") or render.get("page_protected")
                             or render.get("contenuto_in") or render.get("source_conflict")
                             or render.get("scartato_come_disegno"))):
                    continue
                if field == "unisci_con" and value:
                    trial = copy.deepcopy(plan)
                    trial_block = next(
                        item for p in trial["pagine"] for item in p["blocchi"]
                        if item["id"] == blk["id"]
                    )
                    trial_block[field] = value
                    if validate_merges(trial):
                        continue
                if blk.get(field) != value:
                    blk[field] = value
                    changed += 1
    unknown = len(set(by_id) - seen)
    return changed, unknown


# --------------------------------------------------------------------------- #
# Ricostruzione degli oggetti da disegnare
# --------------------------------------------------------------------------- #

def _entry_top(entry: dict) -> float:
    """Coordinata verticale del blocco, in punti, per ordinarli come si leggono."""
    render = entry.get("_render") or {}
    if render.get("tipo") == "immagine":
        return float(render["box_px"][1]) * float(render.get("scale", 1.0))
    box = render.get("box") or render.get("rect") or [0.0, 0.0]
    return float(box[1])


def _merge_pairs(entries: list) -> list:
    """Applica `unisci_con`: fonde i rettangoli e concatena i testi."""
    # Il piano e' riusato dal QA e dai rendering successivi. Le fusioni sono
    # oggetti temporanei di disegno, non modifiche al documento di lavoro.
    entries = copy.deepcopy(entries)
    by_id = {e["id"]: e for e in entries}
    absorbed = set()
    problems = validate_merges({"pagine": [{"blocchi": entries}]})
    invalid_sources = {problem.split(" ->", 1)[0].split(":", 1)[0] for problem in problems}

    for entry in entries:
        target_id = entry.get("unisci_con")
        if not target_id or target_id == entry["id"] or entry["id"] in invalid_sources:
            continue
        target = by_id.get(target_id)
        if target is None or target["id"] in absorbed or entry["id"] in absorbed:
            continue

        rk, tk = entry["_render"], target["_render"]
        if rk.get("tipo") != tk.get("tipo"):
            continue

        if rk["tipo"] == "immagine":
            a, b = rk["box_px"], tk["box_px"]
            tk["box_px"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
            # il rettangolo si fonde, i profili si sommano: coprire solo quello
            # del primo lascerebbe l'altro balloon in lingua originale
            tk["fill_poly"] = (tk.get("fill_poly") or []) + (rk.get("fill_poly") or [])
        else:
            tk["box"] = list(fitz.Rect(tk["box"]) | fitz.Rect(rk["box"]))
            tk["rect"] = list(fitz.Rect(tk["rect"]) | fitz.Rect(rk["rect"]))

        # l'ordine di lettura decide chi viene prima nel testo fuso
        first, second = sorted((entry, target), key=_entry_top)
        target["testo"] = " ".join(t for t in (first["testo"], second["testo"]) if t).strip()
        absorbed.add(entry["id"])

    return [e for e in entries if e["id"] not in absorbed]


def entry_text(entry: dict) -> str:
    return (entry.get("testo") or entry.get("automatica") or "").strip()


def drawable_entries(entries: list, *, page=None, merge=True) -> list:
    """Le voci il cui testo finisce davvero sulla pagina: adattate da P3, non
    saltate e con testo non vuoto.

    Con `merge=False` restano separate le voci che `unisci_con` fonderebbe:
    il rettangolo non e' quello disegnato, ma ogni voce conserva la propria
    regione sorgente. Serve a chi misura la copertura, non la geometria.
    """
    working = entries
    if page is not None and (page.get("struttura") or {}).get("versione") == 2:
        from comics_translator.sorgente.page_structure import adapt_render_entries
        working = adapt_render_entries(page, entries)
    live = [e for e in working if not e.get("salta")]
    if merge:
        live = _merge_pairs(live)
    return [entry for entry in live if entry_text(entry)]


# Quanta parte di una riga vicina deve cadere nel riquadro di un blocco per
# considerarla un residuo dello stesso balloon, e non arte accanto. Il riquadro
# si allarga di mezza altezza di riga: l'OCR non centra i box al pixel.
QUOTA_RIGA_DENTRO = 0.5


def riga_dentro_il_blocco(riga_box, blocco_box, quota=QUOTA_RIGA_DENTRO) -> bool:
    x0, y0, x1, y1 = [float(v) for v in riga_box]
    area = max(1e-9, (x1 - x0) * (y1 - y0))
    bx0, by0, bx1, by1 = [float(v) for v in blocco_box]
    # Il margine vale mezza riga perche' l'OCR non centra i box al pixel, ma la
    # tolleranza non puo' superare il blocco che sta misurando: su una riga alta
    # quanto la tavola gonfiava il riquadro di migliaia di pixel e faceva
    # risultare "dentro" qualunque cosa. Su imperial-guardians p2 il titolo
    # verticale "JIM SHOOTER" (UNA riga OCR alta 4200 px) e' sparito cosi',
    # coperto dal riempimento della colonna sbloccata accanto. Il banco offline
    # sui FATTO passa da 191 a 187 sblocchi automatici su 326 manuali: i quattro
    # che cadono sono quelli che stavano in piedi solo grazie al margine.
    pad = 0.5 * max(0.0, min(y1 - y0, by1 - by0))
    bx0, by0, bx1, by1 = bx0 - pad, by0 - pad, bx1 + pad, by1 + pad
    ix = max(0.0, min(x1, bx1) - max(x0, bx0))
    iy = max(0.0, min(y1, by1) - max(y0, by0))
    return ix * iy / area >= quota


def poligoni_sbloccati(render, page):
    """Le righe inglesi che restano scoperte dentro un blocco sbloccato a mano.

    `source_overlap` elenca le righe che stanno nell'area del blocco senza
    appartenergli: sono la ragione per cui la protezione era scattata. Quando
    l'operatore sblocca il blocco quelle righe restano dov'erano, e siccome la
    maschera segue il contorno del balloon (o il riquadro del blocco) la parte
    che sporge non viene coperta: l'inglese si legge accanto all'italiano.
    Misurato su carnage-3: "ME TO FIND" sopra "MI HAI ASSUNTO PER TROVARE",
    "YOU LEAVE THEM" dietro "PERCHÉ LASCIARLI".

    Sbloccare vuol dire proprio prendersi la responsabilita' di coprire quella
    zona - il motivo e' scritto nel piano. Senza sblocco non si tocca niente:
    li' la protezione sta ancora difendendo inchiostro che nessuno riscrive.

    Ma si copre solo la riga che sta per lo piu' DENTRO il riquadro del blocco.
    `source_overlap` elenca anche righe che il riquadro sfiora appena: su
    spiderman-13 era il balloon "HA HA!" sopra, su spidey-63-8 le onomatopee
    WHAM!/CRASH!/FLAME ON! con un riquadro diagonale enorme. Coperte intere,
    diventavano rettangoli bianchi al posto dell'arte; quattro `.finale` hanno
    dovuto azzerare `source_overlap` a mano per salvarle.
    """
    if not (render.get("sbloccato_a_mano") or render.get("sbloccato_auto")) or page is None:
        return []
    ids = render.get("source_overlap") or []
    if not ids:
        return []
    righe = {r["id"]: r for r in (page.get("sorgente") or {}).get("righe", [])
             if isinstance(r, dict) and r.get("id")}
    proprio = render.get("box_px")
    poligoni = []
    for rid in ids:
        box = (righe.get(rid) or {}).get("box_px")
        if not box or len(box) != 4:
            continue
        if proprio and len(proprio) == 4 and not riga_dentro_il_blocco(box, proprio):
            continue
        x0, y0, x1, y1 = box
        poligoni.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return poligoni


def profili_di_copertura(render, page):
    """I profili che il rendering deve riempire, sblocchi compresi.

    Attenzione all'effetto collaterale: `draw_group` riempie il rettangolo del
    blocco SOLO quando non ha profili. Aggiungere le righe sbloccate a un
    blocco che di profili non ne aveva gli toglierebbe il riempimento del
    balloon, quindi in quel caso il rettangolo entra fra i profili insieme a
    loro.
    """
    profili = [[list(punto) for punto in profilo]
               for profilo in render.get("fill_poly") or [] if len(profilo) >= 3]
    sbloccati = poligoni_sbloccati(render, page)
    if not sbloccati:
        return profili
    if not profili and render.get("box_px"):
        x0, y0, x1, y1 = render["box_px"]
        profili = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]
    return profili + sbloccati


def _lobi_della_voce(entry: dict, text: str, page, altri: list) -> list:
    """I lobi da scrivere per una voce immagine, o [] se resta una battuta sola.

    Due fonti: un'unita' P3 multilobo gia' fusa (ogni membro e' un lobo con la
    sua battuta) oppure un blocco solo i cui lobi si riconoscono da righe e
    contorno. Un lobo non scrive dove scrive un'altra voce: cresce fino ai
    riquadri degli altri e si ferma.
    """
    from comics_translator.disegno import lobi as lobi_mod

    render = entry.get("_render") or {}
    if page is None or render.get("tipo") != "immagine":
        return []
    membri = render.get("p3_membri") or []
    if len(membri) >= 2:
        per_id = {r["id"]: r for r in (page.get("sorgente") or {}).get("righe", [])
                  if isinstance(r, dict)}
        risultato = []
        for membro in membri:
            righe = [per_id[i]["box_px"] for i in membro.get("righe") or []
                     if i in per_id and per_id[i].get("box_px")]
            if not righe or not membro.get("testo") or not membro.get("fill_poly"):
                return []
            lobo = lobi_mod.Lobo(righe=righe, testo_en="", profili=membro["fill_poly"])
            risultato.append({"box_px": lobi_mod.riquadro_di_scrittura(lobo, altri),
                              "testo": membro["testo"], "profili": membro["fill_poly"]})
        return risultato
    return lobi_mod.prepara(render, lobi_mod.righe_del_blocco(entry, page), text, altri)


def rebuild_page(entries: list, Block, ImageJob, *, page=None) -> tuple:
    """Da voci del piano a job; P3 adatta solo unita' validate."""
    blocks, jobs = [], []

    disegnabili = drawable_entries(entries, page=page)
    riquadri = {e["id"]: (e.get("_render") or {}).get("box_px") for e in disegnabili}
    for entry in disegnabili:
        render = entry.get("_render") or {}
        text = entry_text(entry)
        forced = entry.get("corpo")

        if render.get("tipo") == "immagine":
            job = ImageJob(
                page=0,
                box_px=list(render["box_px"]),
                text=entry.get("originale", ""),
                upper=bool(render.get("upper")),
                base_size=float(forced) if forced else float(render.get("base_size", 10.0)),
                bg=tuple(render.get("bg", (255, 255, 255))),
                light_text=bool(render.get("light_text")),
                scale=float(render.get("scale", 0.18)),
                fill_poly=profili_di_copertura(render, page),
                translated=text,
                centered=entry.get("allinea") != "sinistra",
                # serve a ritrovare il blocco dal job prima di disegnare
                # nel piano salvato la chiave privata non c'e' piu': resta
                # l'id della regione dentro `sorgente`
                source_region_id=(entry.get("_source_region_id")
                                  or (entry.get("sorgente") or {}).get("regione")),
                lobi=_lobi_della_voce(
                    entry, text, page,
                    [b for i, b in riquadri.items() if i != entry["id"] and b and len(b) == 4]),
            )
            jobs.append(job)
        else:
            blk = Block(
                page=0,
                rect=fitz.Rect(render.get("rect", render.get("box"))),
                text=entry.get("originale", ""),
                size=float(forced) if forced else float(render.get("size", 10.0)),
                color=tuple(render.get("color", (0, 0, 0))),
                bold=bool(render.get("bold")),
                italic=bool(render.get("italic")),
                centered=entry.get("allinea") == "centro",
                translated=text,
                box=fitz.Rect(render["box"]) if render.get("box") else None,
            )
            blocks.append(blk)

    return blocks, jobs


if __name__ == "__main__":
    main()
