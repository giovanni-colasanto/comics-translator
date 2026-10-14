"""P3.4: verifica offline del rendering, indipendente dal renderer.

Il piano dichiara quali aree ha il permesso di modificare; qui si misura sui
pixel se il PDF prodotto ha rispettato quella dichiarazione. Il denominatore
della copertura e' la sorgente sigillata, non l'elenco dei blocchi eseguiti:
alzare `salta` abbassa la copertura, non la migliora.

Nessuna correzione viene applicata e nessun PDF viene riscritto.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from comics_translator.disegno import plan as plan_module
from comics_translator.verifica.segnalazioni import inglese_residuo
from comics_translator.sorgente import source_inventory as sources


AUDIT_DPI = 150
# Antialiasing e ricompressione muovono di poco i canali anche dove nulla e'
# stato ridisegnato: sotto questa soglia una differenza non e' una modifica.
CHANNEL_TOLERANCE = 24
# Il testo puo' sbordare di poco dal riquadro: e' un difetto di impaginazione
# (P4), non una modifica fuori area. Il margine e' dichiarato, non tarato.
AUTHORIZED_MARGIN_PT = 3.0
# Quanta parte di una riga originale deve essere dentro l'area cancellata
# perche' si possa dire coperta. Meta' e' una soglia larga di proposito: una
# riga coperta a meta' e' gia' illeggibile, e cosi' l'antialiasing dei bordi e
# le righe che sporgono di poco non diventano difetti inventati.
FRAZIONE_RIGA_COPERTA = 0.5
# Quante lettere latine servono perche' una riga scoperta sia inglese che si
# legge e non spazzatura dell'OCR. Sotto questa soglia il rilievo resta scritto
# nel rapporto ma non blocca: sui sette albi gia' misurati separa i residui
# veri ("DOES", "NOTHINGCANSTOP") da segni singoli, ideogrammi e file di stelle.
LETTERE_LEGGIBILI = 3
# Sotto questa frazione di pixel diversi dal riempimento, il lobo e' vuoto.
INK_MIN_RATIO = 0.002


def _to_pixels(points, rect, dpi):
    factor = dpi / 72.0
    return np.asarray(
        [[(p[0] - rect.x0) * factor, (p[1] - rect.y0) * factor] for p in points],
        np.float32,
    )


def _rect_polygon(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def entry_shapes(entry):
    """Aree in punti PDF che questa voce puo' toccare: riempimento e scrittura.

    Il riquadro di scrittura serve anche quando esistono i profili: il testo
    viene impaginato dentro il rettangolo, non dentro il contorno del balloon.
    """
    render = entry.get("_render") or {}
    if render.get("tipo") == "immagine":
        scale = float(render.get("scale", 1.0))
        shapes = [
            [[point[0] * scale, point[1] * scale] for point in profile]
            for profile in render.get("fill_poly") or []
            if len(profile) >= 3
        ]
        shapes.append(_rect_polygon([v * scale for v in render["box_px"]]))
        return shapes
    shapes = [_rect_polygon(render.get("rect", render.get("box")))]
    if render.get("box"):
        shapes.append(_rect_polygon(render["box"]))
    return shapes


def erase_shapes(entry):
    """Solo cio' che viene davvero riempito col colore di sfondo.

    Non e' `entry_shapes`, che elenca tutto quello che la voce PUO' toccare
    (riempimento piu' riquadro di scrittura). Il renderer, quando ha i
    profili, riempie i profili e basta: il resto del riquadro resta com'era,
    e se li' c'era testo originale quel testo si legge ancora.
    """
    render = entry.get("_render") or {}
    if render.get("tipo") == "immagine":
        scale = float(render.get("scale", 1.0))
        profili = [
            [[point[0] * scale, point[1] * scale] for point in profile]
            for profile in render.get("fill_poly") or []
            if len(profile) >= 3
        ]
        return profili or [_rect_polygon([v * scale for v in render["box_px"]])]
    return [_rect_polygon(render.get("rect", render.get("box")))]


def entry_envelope(entry):
    """Il rettangolo che contiene tutto cio' che la voce tocca.

    Non il contorno esatto: proprio le tasche che il contorno si e' perso sono
    il posto dove resta l'inglese scoperto. Serve a dire "questa riga sta
    dentro il balloon che abbiamo riscritto", non "questa riga e' cancellata".
    """
    punti = [p for shape in entry_shapes(entry) for p in shape]
    xs = [p[0] for p in punti]
    ys = [p[1] for p in punti]
    return _rect_polygon([min(xs), min(ys), max(xs), max(ys)])


def authorized_shapes(page):
    entries = plan_module.drawable_entries(page.get("blocchi", []), page=page)
    return [shape for entry in entries for shape in entry_shapes(entry)]


def polygon_mask(shapes, rect, dpi, size, margin_pt=AUTHORIZED_MARGIN_PT):
    """Maschera dei pixel autorizzati, allargata del margine dichiarato."""
    mask = np.zeros(size, np.uint8)
    for shape in shapes:
        cv2.fillPoly(mask, [np.rint(_to_pixels(shape, rect, dpi)).astype(np.int32)], 255)
    radius = int(math.ceil(margin_pt * dpi / 72.0))
    if radius > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        mask = cv2.dilate(mask, kernel)
    return mask


def changed_mask(before, after, tolerance=CHANNEL_TOLERANCE):
    if before.shape != after.shape:
        raise ValueError("le due pagine hanno dimensioni diverse")
    delta = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2)
    return (delta > tolerance).astype(np.uint8) * 255


def page_array(document, pno, dpi):
    page = document[pno]
    pix = page.get_pixmap(dpi=dpi)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(image[:, :, :3]), page.rect


def lobe_ink_ratio(image, polygon, background, rect, dpi):
    """Frazione di pixel che, dentro il lobo, non sono il colore di riempimento."""
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(_to_pixels(polygon, rect, dpi)).astype(np.int32)], 255)
    # Il bordo del riempimento e' sempre diverso dallo sfondo: erodere evita di
    # scambiare l'antialiasing del contorno per testo scritto dentro il lobo.
    radius = max(1, int(round(1.5 * dpi / 72.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    mask = cv2.erode(mask, kernel)
    total = int((mask > 0).sum())
    if total == 0:
        return None
    inside = image[mask > 0].astype(np.int16)
    delta = np.abs(inside - np.asarray(background, np.int16)).max(axis=1)
    return float((delta > CHANNEL_TOLERANCE).sum()) / total


def _changed_inside(changed, shapes, rect, dpi):
    mask = polygon_mask(shapes, rect, dpi, changed.shape, margin_pt=0.0)
    return int((cv2.bitwise_and(changed, mask) > 0).sum())


def _drawing_defects(image, changed, page, rect, dpi):
    """Distingue due difetti che si somigliano ma non sono lo stesso.

    Un lobo riempito e rimasto senza inchiostro ha perso la battuta dentro un
    balloon gia' cancellato. Una voce che non ha cambiato un solo pixel non e'
    stata disegnata affatto: la pagina e' ancora quella originale, e chiamarla
    lobo vuoto attribuirebbe al rendering un danno che non ha fatto.
    """
    empty, undrawn = [], []
    for entry in plan_module.drawable_entries(page.get("blocchi", []), page=page):
        render = entry.get("_render") or {}
        if render.get("tipo") != "immagine":
            continue
        scale = float(render.get("scale", 1.0))
        background = [float(c) for c in render.get("bg", (255, 255, 255))]
        profiles = [p for p in render.get("fill_poly") or [] if len(p) >= 3]
        polygons = [
            [[point[0] * scale, point[1] * scale] for point in profile]
            for profile in profiles
        ]
        if not _changed_inside(changed, polygons or entry_shapes(entry), rect, dpi):
            undrawn.append({"blocco": entry["id"], "testo": entry_text_preview(entry)})
            continue
        for index, polygon in enumerate(polygons):
            if not _changed_inside(changed, [polygon], rect, dpi):
                continue
            ratio = lobe_ink_ratio(image, polygon, background, rect, dpi)
            if ratio is None or ratio >= INK_MIN_RATIO:
                continue
            empty.append({
                "blocco": entry["id"],
                "lobo": index,
                "unita": render.get("p3_unita"),
                "multilobo": bool(render.get("p3_multilobo")),
                "inchiostro": round(ratio, 6),
            })
    return empty, undrawn


def entry_text_preview(entry):
    return plan_module.entry_text(entry)[:80]


def _owned_rows(page):
    """A quali righe sorgente si riferisce ogni voce effettivamente disegnata.

    Le fusioni restano separate: il testo di una voce assorbita viene comunque
    disegnato, e contarla come non coperta sottostimerebbe la copertura.
    """
    regions = {r["id"]: r for r in page["sorgente"]["regioni"]}
    owners = {}
    entries = plan_module.drawable_entries(
        page.get("blocchi", []), page=page, merge=False
    )
    for entry in entries:
        render = entry.get("_render") or {}
        region_ids = list(render.get("p3_regioni") or [])
        if not region_ids:
            region_id = (entry.get("sorgente") or {}).get("regione")
            region_ids = [region_id] if region_id else []
        for region_id in region_ids:
            for row_id in regions.get(region_id, {}).get("righe", []):
                owners.setdefault(row_id, []).append(entry["id"])
    return owners


def _righe_rivendicate(page):
    """Le righe che un blocco del piano dichiara proprie, disegnato o no.

    Serve a separare due cose che `righe_non_coperte` teneva insieme:
    una riga lasciata in originale da un blocco `salta` (esclusione voluta,
    dichiarata, con un blocco che la copre) e una riga che NON appartiene a
    nessun blocco. La seconda non e' una scelta: e' inglese che nessuna
    maschera cancella e che ricompare sotto la traduzione del vicino.
    """
    regions = {r["id"]: r for r in page["sorgente"]["regioni"]}
    rivendicate = {}
    for block in page.get("blocchi", []):
        if not isinstance(block, dict) or not block.get("id"):
            continue
        render = block.get("_render") or {}
        region_ids = list(render.get("p3_regioni") or [])
        if not region_ids:
            region_id = (block.get("sorgente") or {}).get("regione")
            region_ids = [region_id] if region_id else []
        proprie = set((block.get("sorgente") or {}).get("righe") or [])
        for region_id in region_ids:
            proprie.update(regions.get(region_id, {}).get("righe", []))
        for row_id in proprie:
            rivendicate.setdefault(row_id, []).append(block["id"])
    return rivendicate


def coverage(page):
    """Copertura misurata sulla sorgente sigillata, non sui blocchi eseguiti."""
    inventory = page.get("sorgente") or {}
    if not inventory.get("righe"):
        # Senza osservazioni il denominatore non esiste: dichiararlo, non
        # trattare una pagina non misurabile come una pagina pulita.
        return {"misurabile": False, "motivo": "sorgente_senza_righe"}
    rows = inventory["righe"]
    acquired = {row["id"] for row in rows if row["stato"] == "acquisita"}
    excluded = [row for row in rows if row["stato"] != "acquisita"]
    owners = _owned_rows(page)
    covered = acquired & set(owners)
    duplicated = [
        {"riga": row_id, "blocchi": sorted(set(ids))}
        for row_id, ids in sorted(owners.items())
        if len(set(ids)) > 1
    ]
    skipped = [
        {"blocco": block["id"],
         "motivo": next(
             (key for key in ("testo_illeggibile", "skip_localizzazione",
                              "source_conflict", "unreviewed", "page_protected")
              if (block.get("_render") or {}).get(key)),
             "salta",
         )}
        for block in page.get("blocchi", []) if block.get("salta")
    ]
    # Una riga acquisita che nessun blocco rivendica non e' un'esclusione:
    # e' testo originale che nessuna maschera cancella. Sui casi reali era il
    # grawlix di Captain America #2 p2 ('#', '#%&', 'DOES') e le didascalie
    # di ASM '63 #3 p13/p18, tutti finiti in chiaro sotto la traduzione.
    rivendicate = _righe_rivendicate(page)
    orfane = sorted(row_id for row_id in acquired - set(owners)
                    if row_id not in rivendicate)
    return {
        "misurabile": True,
        "righe_acquisite": len(acquired),
        "righe_coperte": len(covered),
        "righe_non_coperte": sorted(acquired - covered),
        "righe_orfane": orfane,
        "righe_escluse_per_scelta": sorted(
            row_id for row_id in acquired - set(owners) if row_id in rivendicate),
        "righe_duplicate": duplicated,
        "righe_escluse": len(excluded),
        "blocchi_saltati": skipped,
    }


def righe_scoperte(page, rect, dpi, size):
    """Righe originali che restano scoperte dentro un'area riscritta.

    Non e' la stessa cosa di una riga lasciata in originale per scelta: quella
    appartiene a un blocco che l'ha dichiarata e la copre. Queste non
    appartengono a nessun blocco, nessuna maschera le cancella, e cadono
    dentro il riquadro dove una voce vicina scrive la traduzione: il risultato
    e' inglese in chiaro sotto o accanto all'italiano appena scritto.

    Il caso misurato e' Captain America #2 pagina 2: l'OCR aveva spezzato il
    balloon in due blocchi piu' tre righe rimaste senza padrone ('#', '#%&',
    'DOES'), e la prima resa le ha lasciate leggibili sotto la traduzione.
    Fuori dalle aree riscritte non si guarda: li' l'originale che resta e'
    grafica della tavola, non un residuo.
    """
    inventario = page.get("sorgente") or {}
    righe = [r for r in inventario.get("righe", []) if r.get("box_pt")]
    entries = plan_module.drawable_entries(page.get("blocchi", []), page=page)
    if not righe or not entries:
        return []
    rivendicate = _righe_rivendicate(page)
    cancellato = polygon_mask(
        [shape for e in entries for shape in erase_shapes(e)], rect, dpi, size)
    riscritto = polygon_mask([entry_envelope(e) for e in entries], rect, dpi, size,
                             margin_pt=0.0)
    scoperte = []
    for riga in righe:
        if riga["id"] in rivendicate:
            continue
        maschera = np.zeros(size, np.uint8)
        cv2.fillPoly(maschera, [np.rint(
            _to_pixels(_rect_polygon(riga["box_pt"]), rect, dpi)).astype(np.int32)], 255)
        totale = int((maschera > 0).sum())
        if not totale or not (cv2.bitwise_and(maschera, riscritto) > 0).any():
            continue
        coperta = int((cv2.bitwise_and(maschera, cancellato) > 0).sum()) / totale
        if coperta < FRAZIONE_RIGA_COPERTA:
            testo = (riga.get("testo_normalizzato") or "")[:40]
            scoperte.append({
                "riga": riga["id"],
                "coperta": round(coperta, 3),
                "testo": testo,
                "leggibile": len(re.findall(r"[A-Za-z]", testo)) >= LETTERE_LEGGIBILI,
            })
    return scoperte


def audit_page(page, source_image, candidate_image, rect, dpi):
    changed = changed_mask(source_image, candidate_image)
    allowed = polygon_mask(authorized_shapes(page), rect, dpi, changed.shape)
    outside = cv2.bitwise_and(changed, cv2.bitwise_not(allowed))
    empty, undrawn = _drawing_defects(candidate_image, changed, page, rect, dpi)
    result = {
        "pagina": page["numero"],
        "dpi": dpi,
        "pixel_cambiati": int((changed > 0).sum()),
        "pixel_fuori_area": int((outside > 0).sum()),
        "lobi_vuoti": empty,
        "voci_non_disegnate": undrawn,
        "righe_scoperte": righe_scoperte(page, rect, dpi, changed.shape),
        "copertura": coverage(page),
    }
    # Fra le righe scoperte leggibili, quelle che sono testo vero in inglese:
    # e' il segnale azionabile della priorita' "nessun inglese sulla pagina".
    # Sottoinsieme delle leggibili, quindi non allarga il blocco - lo spiega.
    result["inglese_residuo"] = inglese_residuo(
        result["righe_scoperte"], page["numero"])
    covered = result["copertura"]
    result["bloccante"] = bool(
        result["pixel_fuori_area"]
        or empty
        or undrawn
        or [r for r in result["righe_scoperte"] if r["leggibile"]]
        or result["inglese_residuo"]
        or not covered["misurabile"]
        or covered["righe_non_coperte"]
        or covered["righe_duplicate"]
    )
    return result, outside


def _open(path, plan):
    document = fitz.open(path)
    if document.page_count < len(plan["pagine"]):
        document.close()
        raise ValueError("%s ha meno pagine del piano" % path)
    return document


def audit(source_pdf, plan, candidate_pdf, out_dir, *, reference_pdf=None,
          dpi=AUDIT_DPI):
    """Confronta sorgente, riferimento e candidato; scrive prove in `out_dir`."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("la cartella di audit esiste gia': %s" % out_dir)
    sources.validate_sources(plan)
    plan = copy.deepcopy(plan)
    out_dir.mkdir(parents=True)

    # Una pagina alla volta: tre albi interi rasterizzati insieme non stanno
    # in memoria, e l'audit deve poter girare su una sessione lunga.
    source = _open(source_pdf, plan)
    candidate = _open(candidate_pdf, plan)
    reference = _open(reference_pdf, plan) if reference_pdf else None
    pages, blocking = [], []
    try:
        for page in plan["pagine"]:
            pno = int(page.get("indice", page["numero"] - 1))
            source_image, rect = page_array(source, pno, dpi)
            candidate_image, _ = page_array(candidate, pno, dpi)
            result, outside = audit_page(page, source_image, candidate_image, rect, dpi)
            if reference is not None:
                drift = changed_mask(page_array(reference, pno, dpi)[0], candidate_image)
                result["riferimento"] = {
                    "pixel_diversi": int((drift > 0).sum()),
                    "identica": not bool((drift > 0).any()),
                }
            if result["pixel_fuori_area"]:
                cv2.imwrite(
                    str(out_dir / ("p%02d-fuori-area.png" % page["numero"])), outside
                )
            if result["bloccante"]:
                blocking.append(page["numero"])
            pages.append(result)
    finally:
        for document in (source, candidate, reference):
            if document is not None:
                document.close()

    report = {
        "versione": 1,
        "dpi": dpi,
        "tolleranza_canale": CHANNEL_TOLERANCE,
        "margine_pt": AUTHORIZED_MARGIN_PT,
        "sorgente": str(source_pdf),
        "candidato": str(candidate_pdf),
        "riferimento": str(reference_pdf) if reference_pdf else None,
        "pagine": pages,
        "pagine_bloccanti": blocking,
        "inglese_residuo": [voce for p in pages for voce in p.get("inglese_residuo", [])],
        "esito": "bloccante" if blocking else "pulito",
    }
    (out_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="P3.4: confronto raster offline fra sorgente, riferimento e candidato",
    )
    parser.add_argument("--source", type=Path, required=True, help="PDF sorgente")
    parser.add_argument("--plan", type=Path, required=True, help="piano v2 usato per il rendering")
    parser.add_argument("--candidate", type=Path, required=True, help="PDF da verificare")
    parser.add_argument("--reference", type=Path, help="PDF del percorso precedente")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="cartella nuova per report e prove (rifiuta una esistente)")
    parser.add_argument("--dpi", type=int, default=AUDIT_DPI)
    args = parser.parse_args()

    data = json.loads(args.plan.read_text(encoding="utf-8"))
    report = audit(args.source, data, args.candidate, args.out_dir,
                   reference_pdf=args.reference, dpi=args.dpi)
    print("esito: %s; pagine bloccanti: %s"
          % (report["esito"], report["pagine_bloccanti"] or "nessuna"))
    residui = report["inglese_residuo"]
    if residui:
        print("INGLESE RESIDUO: %d righe di testo vero lasciate in inglese:" % len(residui))
        for voce in residui:
            print("   p%-3s %-24s coperta=%.2f  %r"
                  % (voce["pagina"], voce["riga"], voce.get("coperta") or 0.0, voce["testo"]))
    return 1 if report["pagine_bloccanti"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
