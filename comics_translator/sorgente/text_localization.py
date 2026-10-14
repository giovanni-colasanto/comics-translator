"""P2: localizzazione conservativa, distinta da traduzione e segmentazione.

Non certifica la completezza: vede solo candidati del rilevatore OCR.
I retry preservano tutte le ipotesi; una lettura discordante non sostituisce
il testo esistente. Le soglie sono esplicite e testate, non un classificatore
di balloon. Nessuna dipendenza o modello aggiuntivo.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pymupdf as fitz

from comics_translator.sorgente import source_inventory as sources


@dataclass(frozen=True)
class RecoveryLimits:
    max_crops: int = 4
    max_crop_pixels: int = 4_000_000
    max_total_pixels: int = 12_000_000
    max_dpi: int = 800
    magnification: float = 2.0
    weak_confidence: float = 0.75
    confirmed_confidence: float = 0.85


def text_key(text):
    # Non eliminare punteggiatura: "NO?" e "NO!" non sono duplicati certi.
    return " ".join(text.split()).casefold()


def bounds(line):
    return [line.x0, line.y0, line.x1, line.y1]


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    inter = overlap(a, b)
    return inter / max(area(a) + area(b) - inter, 1e-9)


def same_observation(a, b):
    return (
        bool(text_key(a.text))
        and text_key(a.text) == text_key(b.text)
        and iou(bounds(a), bounds(b)) >= 0.85
    )


def deduplicate(lines):
    """Indici canonici e alias; nessuna catena transitiva o fusione per substring."""
    canonical, aliases = [], {}
    ranked = sorted(
        range(len(lines)),
        key=lambda i: (
            bool(lines[i].exclusion_reason),
            bool(lines[i].provenance),
            -lines[i].conf,
            i,
        ),
    )
    for index in ranked:
        match = next(
            (
                other
                for other in canonical
                if same_observation(lines[index], lines[other])
            ),
            None,
        )
        if match is None:
            canonical.append(index)
        else:
            aliases[index] = match
    return sorted(canonical), aliases


def crop_spec(page_rect, line, base_dpi, limits):
    """Box in punti e budget raster prima dell'allocazione, bordi arrotondati."""
    scale = 72.0 / base_dpi
    box = fitz.Rect([v * scale for v in bounds(line)])
    pad = max(3.0, box.height * 0.7)
    clip = (box + (-pad, -pad, pad, pad)) & page_rect
    if clip.is_empty or not all(math.isfinite(v) for v in clip):
        return None
    dpi = min(limits.max_dpi, int(base_dpi * limits.magnification))
    dpi = min(
        dpi, int(72 * math.sqrt(limits.max_crop_pixels / max(clip.get_area(), 1)))
    )

    # +2 include l'arrotondamento esterno di MuPDF a entrambi i bordi.
    def pixels(d):
        return (math.ceil(clip.width * d / 72) + 2) * (
            math.ceil(clip.height * d / 72) + 2
        )

    while dpi > base_dpi and pixels(dpi) > limits.max_crop_pixels:
        dpi -= 1
    if dpi <= base_dpi:
        return None
    return clip, dpi, pixels(dpi)


def render_crop(page, clip, dpi):
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return image, (pix.x * 72.0 / dpi, pix.y * 72.0 / dpi)


def recover(page, observations, base_dpi, recognize, limits=None):
    """Retry limitati, stesso OCR; ritorna copie e rapporto, mai altera l'input.

    Si promuove solo la stessa lettura prima debole, concorde in testo e
    posizione. Alternative e nuove rilevazioni rimangono esplicitamente
    irrisolte e protette. P3/P5 potranno decidere come utilizzarle.
    """
    limits = limits or RecoveryLimits()
    output = list(observations)
    canonical, _ = deduplicate(observations)
    candidates = [
        i
        for i in canonical
        if observations[i].exclusion_reason
        or observations[i].conf < limits.weak_confidence
        or len(observations[i].text.strip()) < 2
    ]
    report = {
        "limiti": asdict(limits),
        "candidati": len(candidates),
        "tentativi": [],
        "rinviati": [],
        "pixel": 0,
        "promozioni_concordi": 0,
    }
    for index in candidates:
        line = observations[index]
        if len(report["tentativi"]) >= limits.max_crops:
            report["rinviati"].append({"indice": index, "motivo": "crop_budget"})
            continue
        spec = crop_spec(page.rect, line, base_dpi, limits)
        if spec is None:
            report["rinviati"].append({"indice": index, "motivo": "resolution_budget"})
            continue
        clip, dpi, pixels = spec
        if report["pixel"] + pixels > limits.max_total_pixels:
            report["rinviati"].append({"indice": index, "motivo": "pixel_budget"})
            continue
        attempt = {
            "indice": index,
            "box_pt": list(clip),
            "dpi": dpi,
            "pixel_budget": pixels,
        }
        report["tentativi"].append(attempt)
        report["pixel"] += pixels
        started = time.monotonic()
        try:
            image, origin = render_crop(page, clip, dpi)
            local_lines = recognize(image)
            attempt["rilevate"] = len(local_lines)
            for local in local_lines:
                factor = base_dpi / dpi
                offset = [origin[0] * base_dpi / 72, origin[1] * base_dpi / 72]
                quad = [
                    [p[0] * factor + offset[0], p[1] * factor + offset[1]]
                    for p in (
                        local.quad
                        or [
                            [local.x0, local.y0],
                            [local.x1, local.y0],
                            [local.x1, local.y1],
                            [local.x0, local.y1],
                        ]
                    )
                ]
                xs, ys = [p[0] for p in quad], [p[1] for p in quad]
                candidate = replace(
                    local, x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys), quad=quad
                )
                nearby = [
                    old
                    for old in observations
                    if iou(bounds(old), bounds(candidate)) >= 0.60
                ]
                agrees = any(same_observation(candidate, old) for old in nearby)
                # Il testo nuovo/discordante non si sceglie solo per la confidenza.
                reason = candidate.exclusion_reason
                if not agrees:
                    reason = (
                        "recovery_alternative" if nearby else "recovery_new_candidate"
                    )
                elif not reason and not any(not old.exclusion_reason for old in nearby):
                    if candidate.conf >= limits.confirmed_confidence:
                        report["promozioni_concordi"] += 1
                    else:
                        reason = "recovery_unconfirmed"
                provenance = {
                    "tipo": "ritaglio",
                    "indice_seme": index,
                    "dpi": dpi,
                    "origine_pt": list(origin),
                    "quad_locale_px": local.quad,
                    "testo_locale": local.raw_text,
                    "motivo": reason,
                }
                output.append(
                    replace(candidate, exclusion_reason=reason, provenance=provenance)
                )
        except (RuntimeError, ValueError, MemoryError) as exc:
            attempt["errore"] = str(exc)
        finally:
            attempt["secondi"] = round(time.monotonic() - started, 4)
    return output, report


def register_rows(inventory, observations, page_no, scale, normalize, localize=True):
    """Registra prima tutte le evidenze; deduplica solo la selezione per i job."""
    canonical, aliases = (
        deduplicate(observations) if localize else (list(range(len(observations))), {})
    )
    ids = {}
    for i, line in enumerate(observations):
        reason = "duplicate" if i in aliases else line.exclusion_reason
        ident = sources.add_line(
            inventory,
            page_no,
            raw=line.raw_text if line.raw_text is not None else line.text,
            normalized=normalize(line.text),
            box=bounds(line),
            kind="ocr",
            scale=scale,
            quad=line.quad,
            confidence=line.conf,
            reason=reason,
        )
        ids[id(line)] = ident
        row = inventory["righe"][-1]
        row["provenienza"] = line.provenance or {"tipo": "pagina"}
        row["motivo_ocr"] = line.exclusion_reason
    by_id = {r["id"]: r for r in inventory["righe"]}
    for index, target in aliases.items():
        by_id[ids[id(observations[index])]]["duplicato_di"] = ids[
            id(observations[target])
        ]
    # La provenienza distingue una rilettura dalla presenza di un nuovo vicino.
    # Non e' dedup testuale: le alternative restano grezze, escluse e incerte.
    originals = [i for i, line in enumerate(observations) if not line.provenance]
    for index, line in enumerate(observations):
        if not line.provenance or line.provenance.get("tipo") != "ritaglio":
            continue
        related = [
            i
            for i in originals
            if iou(bounds(line), bounds(observations[i])) >= 0.60
            or (
                overlap(bounds(line), bounds(observations[i]))
                / max(area(bounds(line)), 1e-9)
                >= 0.85
                and 0.5
                <= (line.y1 - line.y0)
                / max(observations[i].y1 - observations[i].y0, 1e-9)
                <= 1.5
            )
        ]
        by_id[ids[id(line)]]["correlata_a"] = [
            ids[id(observations[aliases.get(i, i)])] for i in related
        ]
    return [
        observations[i] for i in canonical if not observations[i].exclusion_reason
    ], ids


def native_line_duplicates(lines):
    canonical, aliases = [], {}
    for index, row in enumerate(lines):
        text = "".join(s.get("text", "") for s in row.get("spans", []))
        other = next(
            (
                j
                for j in canonical
                if text_key(text)
                == text_key(
                    "".join(s.get("text", "") for s in lines[j].get("spans", []))
                )
                and iou(row["bbox"], lines[j]["bbox"]) >= 0.85
            ),
            None,
        )
        if other is None:
            canonical.append(index)
        else:
            aliases[index] = other
    return aliases


def deduplicate_native(blocks, inventory):
    """Doppio layer PDF identico; mai cancellare battute uguali in zone diverse."""
    kept = []
    regions = {r["id"]: r for r in inventory["regioni"]}
    rows = {r["id"]: r for r in inventory["righe"]}
    for block in blocks:
        other = next(
            (
                b
                for b in kept
                if text_key(b.text) == text_key(block.text)
                and iou(list(b.rect), list(block.rect)) >= 0.85
            ),
            None,
        )
        if other is None:
            kept.append(block)
            continue
        region = regions[block.source_region_id]
        region.update(
            stato="esclusa", motivo="duplicate", duplicato_di=other.source_region_id
        )
        target = regions[other.source_region_id]
        for ident in region["righe"]:
            row = rows[ident]
            match = next(
                (
                    rows[r]
                    for r in target["righe"]
                    if text_key(rows[r]["testo_grezzo"])
                    == text_key(row["testo_grezzo"])
                    and iou(rows[r]["box_pt"], row["box_pt"]) >= 0.85
                ),
                None,
            )
            row.update(stato="esclusa", motivo="duplicate")
            if match:
                row["duplicato_di"] = match["id"]
    return kept


def polygon_hits_box(poly, box):
    """Intersezione conservativa, anche per profili concavi e rettangoli interni."""
    if len(poly) < 3 or area(box) <= 0:
        return False
    lo = [
        min(p[0] for p in poly),
        min(p[1] for p in poly),
        max(p[0] for p in poly),
        max(p[1] for p in poly),
    ]
    if overlap(lo, box) <= 0:
        return False
    import cv2  # gia' dipendenza del renderer OCR

    contour = np.asarray(poly, dtype=np.float32)
    corners = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    if any(cv2.pointPolygonTest(contour, tuple(p), False) >= 0 for p in corners):
        return True
    if any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in poly):
        return True
    # Clipping Liang-Barsky: un segmento puo' attraversare il box senza vertici interni.
    for a, b in zip(poly, poly[1:] + poly[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        low, high = 0.0, 1.0
        for p, q in (
            (-dx, a[0] - box[0]),
            (dx, box[2] - a[0]),
            (-dy, a[1] - box[1]),
            (dy, box[3] - a[1]),
        ):
            if p == 0:
                if q < 0:
                    low, high = 1.0, 0.0
                    break
            elif p < 0:
                low = max(low, q / p)
            else:
                high = min(high, q / p)
        if low <= high:
            return True
    return False


def row_polygon(row):
    box = row["box_pt"]
    return row.get("quad_pt") or [
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]],
    ]


def axis_aligned(row):
    """Solo rettangoli reali nel sottrattore storico a box; i ruotati nel guard."""
    poly = row_polygon(row)
    box = row["box_pt"]
    return all(
        any(abs(x - edge) < 0.1 for edge in (box[0], box[2]))
        and any(abs(y - edge) < 0.1 for edge in (box[1], box[3]))
        for x, y in poly
    )


# Una riga senza testo riconosciuto non lascia inchiostro da difendere. Solo
# questo caso: "no_letters" e "too_short" sono stringhe vere ma non traducibili
# (un numero civico, un orario), e coprirle toglierebbe qualcosa al lettore.
RIGHE_SENZA_INCHIOSTRO = frozenset({"empty_text"})

# Riletture incerte della stessa area: l'inchiostro c'e', ma e' quello che il
# blocco sta gia' riscrivendo, purche' cada dentro l'area che copre comunque.
RIGHE_RILETTURA = frozenset({
    "low_confidence",
    "recovery_alternative",
    "recovery_new_candidate",
    "recovery_unconfirmed",
})

# Motivi di salto che appartengono ad altre regole: questa non li tocca.
# `salta_deciso` e' il piu' recente: lo scrive apply_corrections quando una
# revisione, il QA o una correzione a mano DECIDE di saltare un blocco. Senza
# di esso il ricalcolo trovava solo il marcatore `skip_localizzazione` rimasto
# da un giro precedente, concludeva che il salto era suo e lo revocava: il
# blocco tornava attivo e il suo inglese ricompariva nel PDF finale.
ALTRE_PROTEZIONI = ("unreviewed", "compound", "page_protected", "contenuto_in",
                    "salta_deciso", "scartato_come_disegno")

# Chi ha guardato l'area e ne risponde: l'operatore (`sbloccato_a_mano`, con il
# motivo scritto) o lo sblocco automatico (`sbloccato_auto`, con la natura
# detta dal modello e le righe vicine verificate una per una).
SBLOCCHI = ("sbloccato_a_mano", "sbloccato_auto")

# Protezioni che il programma impone da solo, senza che nessuno abbia deciso.
PROTEZIONI_AUTOMATICHE = ("source_conflict", "scartato_come_disegno", "compound",
                          "contenuto_in")

# Nature che il modello assegna a cio' che NON e' una battuta: insegne, testate,
# loghi, rumori disegnati. Non si riattivano da sole quando una protezione cade:
# disegnarle vuol dire coprire l'arte con un rettangolo.
LETTERING = ("grafica", "rumore")


def righe_risolte_da_p3(render, zones):
    """Le righe che P3 ha gia' verificato per un blocco multilobo riattivato.

    `activate_validated_units` riattiva solo unita' eseguibili, cioe' senza
    conflitti sorgente esterni: le righe che toglie (`p3_conflitti_precedenti`)
    sono riletture incerte dentro i lobi, senza un blocco proprietario. Il
    ricalcolo le rileggeva come inchiostro da difendere e riproteggeva il
    blocco; la validazione del lotto vedeva un salto e lo leggeva come una
    riattivazione del modello. Su imperial-guardians il lotto 9 (pagine 17-18)
    e' caduto cosi' due volte, con la risposta del modello identica ai `salta`
    di partenza: e' il difetto "glm riattiva i source_conflict" di LEZIONI.

    Le righe di un ALTRO blocco restano fuori: se quel blocco resta in
    originale, la protezione serve ancora, anche dentro la stessa unita'.
    """
    if not render.get("p3_multilobo"):
        return set()
    return {ident for ident in render.get("p3_conflitti_precedenti") or []
            if (zones.get(ident) or {}).get("blocco") is None}

# Stessa tolleranza OCR di belongs_to: una riga non si allinea al pixel.
TOLLERANZA_AREA_PT = 3.0


def within_polygons(quad, polys, tol=TOLLERANZA_AREA_PT):
    """Vero se il profilo sta dentro una delle aree, con tolleranza in pt."""
    import cv2

    for poly in polys:
        if len(poly) < 3:
            continue
        contour = np.asarray(poly, np.float32)
        if all(
            cv2.pointPolygonTest(contour, (float(x), float(y)), True) >= -tol
            for x, y in quad
        ):
            return True
    return False


def covered_areas(render):
    """Le aree che il blocco ricopre davvero quando viene disegnato.

    Senza profili del balloon `draw_group` riempie il rettangolo: la stessa
    scelta va rifatta qui, altrimenti si difende un'area che nessuno copre.
    """
    if render.get("tipo") == "immagine":
        scale = render["scale"]
        polys = [p for p in render.get("fill_poly") or [] if len(p) >= 3]
        if polys:
            return [
                [sources.coordinates(point, scale) for point in poly]
                for poly in polys
            ]
        box = sources.coordinates(render["box_px"], scale)
        return [[
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ]]
    rect = render.get("box") or render["rect"]
    return [[
        [rect[0], rect[1]],
        [rect[2], rect[1]],
        [rect[2], rect[3]],
        [rect[0], rect[3]],
    ]]


def belongs_to(row, own, rows):
    related = set(row.get("correlata_a", []))
    if not related or not related <= own:
        return False
    # Tolleranza OCR esplicita di 3pt; non basta una sovrapposizione parziale.
    return all(
        any(
            rows[r]["box_pt"][0] - 3 <= x <= rows[r]["box_pt"][2] + 3
            and rows[r]["box_pt"][1] - 3 <= y <= rows[r]["box_pt"][3] + 3
            for r in related
        )
        for x, y in row_polygon(row)
    )


def polygons_intersect(a, b):
    """Profili concavi/ruotati: nessun allargamento al bounding box."""
    import cv2

    if not a or not b:
        return False
    ca, cb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if any(cv2.pointPolygonTest(ca, tuple(p), False) >= 0 for p in b):
        return True
    if any(cv2.pointPolygonTest(cb, tuple(p), False) >= 0 for p in a):
        return True

    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    for p, q in zip(a, a[1:] + a[:1]):
        for r, s in zip(b, b[1:] + b[:1]):
            if max(min(p[0], q[0]), min(r[0], s[0])) > min(
                max(p[0], q[0]), max(r[0], s[0])
            ):
                continue
            if max(min(p[1], q[1]), min(r[1], s[1])) > min(
                max(p[1], q[1]), max(r[1], s[1])
            ):
                continue
            if (
                side(p, q, r) * side(p, q, s) <= 0
                and side(r, s, p) * side(r, s, q) <= 0
            ):
                return True
    return False


def merge_conflicts(page, first, second):
    """Anche il rettangolo risultante da una fusione deve rispettare gli altri."""
    inventory = page.get("sorgente", {})
    if not inventory.get("localizzazione"):
        return []
    own = set(first["sorgente"]["righe"]) | set(second["sorgente"]["righe"])
    rows = {r["id"]: r for r in inventory["righe"]}
    boxes = []
    for block in (first, second):
        render = block["_render"]
        boxes.append(
            sources.coordinates(render["box_px"], render["scale"])
            if render["tipo"] == "immagine"
            else render.get("box") or render["rect"]
        )
    a, b = boxes
    writing = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    return [
        ident
        for ident, row in rows.items()
        if ident not in own
        and row.get("motivo") != "duplicate"
        and not belongs_to(row, own, rows)
        and polygon_hits_box(row_polygon(row), writing)
    ]


def finalize_page(page):
    """Mappa di proprieta' in pt, conflitti e omissioni; non e' un gold standard."""
    inventory = page["sorgente"]
    rows = {r["id"]: r for r in inventory["righe"]}
    owners = {r: b["id"] for b in page["blocchi"] for r in b["sorgente"]["righe"]}
    areas, divergences = [], []
    for ident, row in rows.items():
        if row.get("motivo") == "duplicate":
            continue
        owner = owners.get(ident)
        protected = row["stato"] == "esclusa" or owner is None
        areas.append(
            {
                "riga": ident,
                "box_pt": list(row["box_pt"]),
                "quad_pt": row_polygon(row),
                "blocco": owner,
                "stato": "protetta" if protected else "occupata",
                "motivo": row.get("motivo") if protected else None,
            }
        )
        if protected:
            divergences.append(
                {"codice": row.get("motivo") or "line_without_block", "riga": ident}
            )
    for block in page["blocchi"]:
        render = block["_render"]
        own = set(block["sorgente"]["righe"])
        if render["tipo"] == "immagine":
            scale = render["scale"]
            writing = sources.coordinates(render["box_px"], scale)
            polys = [
                [sources.coordinates(p, scale) for p in poly]
                for poly in render.get("fill_poly", [])
            ]
        else:
            writing = list(render.get("box") or render["rect"])
            rect = render["rect"]
            polys = [
                [
                    [rect[0], rect[1]],
                    [rect[2], rect[1]],
                    [rect[2], rect[3]],
                    [rect[0], rect[3]],
                ]
            ]
        overlaps = []
        for zone in areas:
            row = rows[zone["riga"]]
            if zone["riga"] in own or belongs_to(row, own, rows):
                continue
            quad = zone["quad_pt"]
            if polygon_hits_box(quad, writing) or any(
                polygons_intersect(poly, quad) for poly in polys
            ):
                overlaps.append(zone["riga"])
        # La sovrapposizione si registra sempre: e' un dato della pagina. Se
        # protegga davvero il blocco lo decide enforce_page, che sa quali
        # vicini vengono riscritti e quali no.
        render["source_overlap"] = overlaps
    inventory["localizzazione"] = {
        "versione": 1,
        "aree": areas,
        "divergenze": divergences,
        "duplicati": sum(r.get("motivo") == "duplicate" for r in rows.values()),
        "denominatore_indipendente": None,
        "verificata": False,
    }
    page["sorgente"] = sources.seal(inventory)
    enforce_page(page)


def _protegge(zone, row, polys, skipped):
    """Se coprire quella riga toglierebbe al lettore inchiostro non riscritto.

    Il rendering riempie **tutte** le aree di una pagina prima di scriverci
    sopra qualunque testo (`_render_document`, fasi "riempi" e "scrivi"): due
    blocchi che si dividono lo stesso balloon si coprono a vicenda il fondo
    senza cancellarsi le battute. Conta quindi una cosa sola: che il vicino
    venga davvero riscritto. Se resta in originale, invece, il riempimento
    glielo cancella e la protezione serve.
    """
    owner = zone.get("blocco")
    if owner is not None:
        return owner in skipped
    motivo = row.get("motivo")
    if motivo in RIGHE_SENZA_INCHIOSTRO:
        return False
    if motivo in RIGHE_RILETTURA and within_polygons(row_polygon(row), polys):
        return False
    return True


def enforce_page(page):
    """Aggiorna protezioni e `salta` in base a chi viene riscritto adesso.

    Idempotente e senza immagini: lavora sulle sovrapposizioni gia' registrate.
    Va rieseguita dopo la revisione AI e dopo le correzioni QA, perche' un
    blocco marcato `salta` dopo il piano rende protettivo un confine che prima
    non lo era.
    """
    inventory = page.get("sorgente") or {}
    localization = inventory.get("localizzazione")
    if not localization:
        return
    rows = {r["id"]: r for r in inventory.get("righe", [])}
    zones = {a["riga"]: a for a in localization.get("aree", [])}
    blocks = [b for b in page.get("blocchi", []) if isinstance(b, dict)]

    def render_of(block):
        return block.get("_render") or {}

    def overlaps_of(block):
        render = render_of(block)
        # I piani vecchi hanno solo source_conflict: era la lista completa.
        if "source_overlap" in render:
            return render["source_overlap"] or []
        return render.get("source_conflict") or []

    def per_scelta_sua(block):
        """Salta per una ragione che non e' questa protezione."""
        render = render_of(block)
        if not block.get("salta"):
            return False
        if render.get("skip_localizzazione"):
            return False
        # Le altre protezioni restano loro: un blocco fuso, contenuto in un
        # altro, su pagina protetta o mai passato dalla revisione non va
        # riaperto qui, nemmeno quando il conflitto che lo copriva decade.
        if any(render.get(m) for m in ALTRE_PROTEZIONI):
            return True
        # Piani precedenti al marcatore: il salto veniva solo dai conflitti.
        if "source_overlap" not in render and render.get("source_conflict"):
            return False
        return True

    # Chi non viene riscritto (pagina protetta, blocco contenuto, scelta della
    # revisione) tiene il suo inchiostro originale: quello va difeso.
    seed = {b["id"] for b in blocks if per_scelta_sua(b)}
    skipped = set(seed)
    aree = {b["id"]: covered_areas(render_of(b)) for b in blocks if render_of(b)}

    while True:
        conflicts = {}
        for block in blocks:
            overlaps = overlaps_of(block)
            # Sbloccato a mano: qualcuno ha guardato la tavola e ha risposto
            # della copertura. Resta scritto nel piano chi e perche'.
            if not overlaps or any(render_of(block).get(k) for k in SBLOCCHI):
                conflicts[block["id"]] = []
                continue
            polys = aree.get(block["id"], [])
            risolte = righe_risolte_da_p3(render_of(block), zones)
            conflicts[block["id"]] = [
                ident for ident in overlaps
                if ident not in risolte
                and _protegge(zones.get(ident, {}), rows.get(ident, {}), polys, skipped)
            ]
        nuovi = {bid for bid, found in conflicts.items() if found} - skipped
        if not nuovi:
            break
        skipped |= nuovi

    for block in blocks:
        render = block.get("_render")
        if render is None:
            continue
        found = conflicts.get(block["id"]) or []
        if found:
            block["salta"] = True
            render["source_conflict"] = found
            # Il marcatore dice che il salto e' nostro e non una scelta della
            # revisione: al giro dopo lo si puo' revocare se il motivo cade.
            # Chi saltava gia' per conto suo resta senza, o la sua decisione
            # verrebbe cancellata dal primo ricalcolo.
            if block["id"] not in seed:
                render["skip_localizzazione"] = True
        elif block.get("salta") and block.get("natura") in LETTERING:
            # Il motivo del salto e' caduto, ma il modello ha detto che qui non
            # c'e' una battuta: c'e' un logo, una testata, un rumore disegnato.
            # Riattivarlo lo fa DISEGNARE, e il riempimento del blocco cancella
            # l'arte che c'e' sotto. Su imperial-guardians p2 e' cosi' che il
            # titolo verticale "JIM SHOOTER", alto quanto la tavola, e' sparito
            # sotto un rettangolo bianco appena i blocchi vicini sono stati
            # riattivati a mano. Lo sblocco automatico segue la stessa regola:
            # da solo apre solo dialogo e didascalia (`sblocco_automatico`).
            render["lettering_dichiarato"] = True
            # `source_conflict` resta: se una traduzione vera c'e' davvero,
            # `traduzioni_soppresse` la dichiara invece di perderla in silenzio.
        else:
            render.pop("source_conflict", None)
            render.pop("skip_localizzazione", None)
            if block["id"] not in seed and block.get("salta"):
                block["salta"] = False


def enforce_localization(plan):
    """Riporta l'intero piano alla protezione minima ancora necessaria."""
    for page in plan.get("pagine", []):
        enforce_page(page)
    return plan


# Nature con cui il modello dice "qui c'e' testo della storia".
NATURE_TESTO = ("dialogo", "didascalia")


def _senza_forma(testo):
    """Solo lettere e cifre: due scritture della stessa cosa diventano uguali."""
    return re.sub(r"[\W_]+", "", testo or "").casefold()


def _contraddizione_del_modello(block, render):
    """Il motivo per cui il salto di questo blocco contraddice il modello stesso.

    Due forme, entrambe viste nella prova A3 (superman-1) e invisibili a ogni
    controllo, perche' `salta_deciso` e le nature di lettering escludono il
    blocco sia da `traduzioni_soppresse` sia da `inglese_render`:

    - `salto_del_modello`: natura `dialogo`, traduzione buona, e poi `salta`
      deciso dal modello stesso ("Contact! Contact! Open fire!" p21,
      "Jor... Oh my Rao..." p33). Chi guarda la tavola dice che e' una battuta:
      il salto e' un residuo del prompt, non una decisione sulla tavola;
    - `natura_contraddetta`: natura `grafica` o `rumore` - cioe' "non
      tradurre" - ma il testo E' tradotto (le tre voci della tuta a p19). Una
      delle due cose e' sbagliata, e va guardata.

    Non si sblocca da soli: il blocco resta come e', ma dichiarato.
    """
    # Confronto normalizzato: l'OCR perde spazi e maiuscole ("Aew" -> "AEW",
    # "Jimshooter" -> "Jim Shooter"), e quelle non sono traduzioni. Senza questo,
    # su tre albi risultavano contraddetti nove loghi e nomi propri.
    testo = _senza_forma(block.get("testo"))
    if not testo or testo == _senza_forma(block.get("originale")):
        return None
    natura = block.get("natura")
    if natura in NATURE_TESTO and render.get("salta_deciso"):
        return "salto_del_modello"
    if natura in LETTERING:
        return "natura_contraddetta"
    return None


def traduzioni_soppresse(plan):
    """Blocchi che il modello HA tradotto ma la protezione tiene invisibili.

    La protezione automatica (`source_conflict`) e' volutamente conservativa:
    di fronte a inchiostro incerto o non identificato preferisce lasciare
    l'originale che rischiare di coprire arte vera (un numero, un simbolo, una
    rilettura). Giusto in generale, ma quando il blocco ha gia' una traduzione
    valida quella scelta la nasconde e IN SILENZIO: l'italiano c'e', nel PDF si
    legge l'inglese. Misurato su ASM #7 p13/p14/p6, dove tre didascalie tradotte
    restavano coperte da righe OCR spazzatura sovrapposte.

    Non si tenta di sbloccarle da soli: allentare la protezione rischierebbe di
    coprire arte reale, e il programma non distingue un misread dal blocco da un
    numero disegnato sulla tavola. Le si DICHIARA, cosi' l'operatore le rivede e
    le sblocca con un motivo (come per qualunque altra protezione). Il seme
    della segnalazione e' un contratto chiaro: traduzione vera + salto imposto
    dalla protezione, non dalla revisione.
    """
    soppresse = []
    for page in plan.get("pagine", []):
        for block in page.get("blocchi", []):
            if not isinstance(block, dict) or not block.get("salta"):
                continue
            render = block.get("_render") or {}
            motivo = _contraddizione_del_modello(block, render)
            if motivo:
                soppresse.append({"pagina": page.get("numero"), "id": block["id"],
                                  "testo": (block.get("testo") or "").strip()[:60],
                                  "motivo": motivo})
                continue
            # Salto imposto da una protezione automatica: `source_conflict`, ma
            # anche un candidato del Gap C, un compound o un blocco contenuto che
            # lo sblocco automatico non ha aperto (prova A2, spidey-63-9: 28
            # battute tradotte restavano in inglese senza nessun avviso).
            # Un salto deciso dalla revisione (`salta_deciso`) o uno sblocco a
            # mano non contano: li' l'italiano non e' una vittima della maschera.
            if (not any(render.get(k) for k in PROTEZIONI_AUTOMATICHE)
                    or any(render.get(k) for k in SBLOCCHI)
                    or render.get("salta_deciso")):
                continue
            testo = (block.get("testo") or "").strip()
            originale = (block.get("originale") or "").strip()
            # Traduzione vera: c'e' un testo, diverso dall'inglese copiato.
            if testo and testo != originale:
                soppresse.append({
                    "pagina": page.get("numero"),
                    "id": block["id"],
                    "testo": testo[:60],
                    "motivo": "protezione",
                })
    return soppresse


def validate_localization(inventory, blocks):
    localization = inventory.get("localizzazione")
    if localization is None:  # v1 e primi piani v2 restano leggibili
        return
    rows = {r["id"]: r for r in inventory["righe"]}
    block_ids = {b["id"] for b in blocks}
    areas = localization.get("aree", [])
    refs = [a.get("riga") for a in areas]
    expected = {r["id"] for r in rows.values() if r.get("motivo") != "duplicate"}
    if len(refs) != len(set(refs)) or set(refs) != expected:
        raise ValueError("mappa localizzazione incompleta o duplicata")
    for zone in areas:
        if zone.get("box_pt") != rows[zone["riga"]]["box_pt"] or zone.get(
            "quad_pt"
        ) != row_polygon(rows[zone["riga"]]):
            raise ValueError("coordinate mappa localizzazione incoerenti")
        if zone.get("blocco") is not None and zone["blocco"] not in block_ids:
            raise ValueError("proprietario mappa localizzazione inesistente")
    for row in rows.values():
        if set(row.get("correlata_a", [])) - set(rows):
            raise ValueError("riferimento rilettura non valido")
        target = row.get("duplicato_di")
        if target and (
            target not in rows
            or target == row["id"]
            or rows[target].get("duplicato_di")
        ):
            raise ValueError("riferimento duplicato non valido")
    for block in blocks:
        conflicts = block.get("_render", {}).get("source_conflict", [])
        if set(conflicts) - set(rows):
            raise ValueError("conflitto con riga inesistente")
        if conflicts and not block.get("salta"):
            raise ValueError("blocco protetto da source_conflict riattivato")


def compare_reference(plan, reference):
    """Audit di un inventario UMANO di test, non prodotto dallo stesso OCR.

    Distingue localizzazione, trascrizione e presenza nel piano; non misura
    la bonta' della traduzione. Un riferimento parziale resta parziale.
    """
    if reference.get("sorgente_sha256") != plan.get("sorgente_sha256"):
        raise ValueError("riferimento e piano appartengono a sorgenti diverse")
    pages = {p["numero"]: p for p in plan["pagine"]}
    results = []
    for unit in reference["unita"]:
        page = pages.get(unit["pagina"])
        if page is None:
            results.append({"id": unit["id"], "stato": "missing_page"})
            continue
        rows = [
            r
            for r in page["sorgente"]["righe"]
            if r.get("motivo") != "duplicate"
            and overlap(r["box_pt"], unit["box_pt"]) / max(area(r["box_pt"]), 1e-9)
            >= 0.5
        ]
        rows.sort(key=lambda r: (r["box_pt"][1], r["box_pt"][0]))

        def normalize(text):
            return re.sub(r"[\W_]+", "", text).casefold()

        transcript = normalize(" ".join(r["testo_grezzo"] for r in rows))
        expected = normalize(unit["testo"])
        covered = {
            r
            for b in page["blocchi"]
            if not b.get("salta")
            for r in b["sorgente"]["righe"]
        }
        results.append(
            {
                "id": unit["id"],
                "localizzata": bool(rows),
                "trascrizione_presente": bool(expected) and expected == transcript,
                "nel_piano_attivo": bool(rows)
                and all(r["id"] in covered for r in rows),
                "righe": [r["id"] for r in rows],
            }
        )
    return {
        "ambito": reference.get("ambito", "parziale"),
        "unita": results,
        "criterio_testo": "uguaglianza_alfanumerica",
        "approvazione_traduzione": False,
    }
