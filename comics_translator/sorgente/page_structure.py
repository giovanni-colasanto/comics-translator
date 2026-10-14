"""P3.2/3: struttura geometrica validata e adattatore multilobo fail-closed.

La sorgente OCR resta sigillata. I contorni sono candidati; diventano una
unita' di disegno soltanto quando proprieta', righe e relazioni sono complete.
Le pagine incomplete restano diagnosticate come tali, anche se alcune unita'
locali possono essere renderizzate in sicurezza.
"""

from __future__ import annotations

import copy
import json
import math

from comics_translator.sorgente import source_inventory as sources
from comics_translator.sorgente.text_localization import RIGHE_RILETTURA, row_polygon, within_polygons


STRUCTURE_VERSION = 2
SHARED_CONTOUR_MIN_RATIO = 0.15
STYLE_COLOR_TOLERANCE = 12.0
ROW_HULL_TOLERANCE_PT = 2.0


def rectangle(box):
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _valid_polygon(poly):
    return (
        isinstance(poly, list) and len(poly) >= 3
        and all(isinstance(p, list) and len(p) == 2 for p in poly)
        and all(type(v) in (int, float) and math.isfinite(v) for p in poly for v in p)
        and abs(sum(a[0] * b[1] - b[0] * a[1]
                    for a, b in zip(poly, poly[1:] + poly[:1]))) > 0
    )


def _intersection_ratio(a, b):
    """Area di intersezione / area del contorno convesso piu' piccolo."""
    import cv2
    import numpy as np

    ca = cv2.convexHull(np.asarray(a, np.float32))
    cb = cv2.convexHull(np.asarray(b, np.float32))
    area, _ = cv2.intersectConvexConvex(ca, cb)
    denominator = min(abs(float(cv2.contourArea(ca))), abs(float(cv2.contourArea(cb))))
    return float(area) / denominator if denominator > 1e-6 else 0.0


def _intersects(a, b):
    ax, ay = zip(*a)
    bx, by = zip(*b)
    if max(min(ax), min(bx)) > min(max(ax), max(bx)):
        return False
    if max(min(ay), min(by)) > min(max(ay), max(by)):
        return False
    return _intersection_ratio(a, b) > 1e-6


def _inside(row, poly):
    """Contenimento conservativo: anche i bordi non attraversano rientranze."""
    quad = row_polygon(row)
    if not within_polygons(quad, [poly], tol=0):
        return False
    import cv2
    import numpy as np

    contour = np.asarray(quad, np.float32)
    for a, b in zip(poly, poly[1:] + poly[:1]):
        for x, y in (a, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]):
            if cv2.pointPolygonTest(contour, (float(x), float(y)), True) > 0:
                return False
    return True


def _inside_candidate(row, candidate, candidates_per_region):
    """Proprieta' OCR: profilo esatto, o inviluppo convesso solo se univoco.

    I contorni estratti da balloon scuri seguono spesso i rientri lasciati dai
    glifi chiari. Per un'unica regione gia' legata a un blocco, l'inviluppo
    convesso recupera quelle righe senza poter scegliere fra lobi concorrenti.
    """
    poly = candidate["poligono_pt"]
    if _inside(row, poly):
        return True
    if candidate.get("blocco") is None or candidates_per_region != 1:
        return False
    import cv2
    import numpy as np

    hull = cv2.convexHull(np.asarray(poly, np.float32)).reshape(-1, 2).tolist()
    return within_polygons(row_polygon(row), [hull], tol=ROW_HULL_TOLERANCE_PT)

def _block_by_region(page):
    result = {}
    for block in page.get("blocchi", []):
        region = (block.get("sorgente") or {}).get("regione")
        if region and region not in result:
            result[region] = block
    return result


def _style(block):
    render = (block or {}).get("_render") or {}
    bg = render.get("bg")
    if render.get("tipo") != "immagine" or not (
        isinstance(bg, (list, tuple)) and len(bg) == 3
        and all(type(v) in (int, float) and math.isfinite(v) for v in bg)
    ):
        return None
    return {
        "bg": tuple(float(v) for v in bg),
        "light_text": bool(render.get("light_text")),
        "scale": float(render.get("scale", 1.0)),
    }


def _styles_compatible(a, b):
    sa, sb = _style(a), _style(b)
    if sa is None or sb is None:
        return False
    return (
        sa["light_text"] == sb["light_text"]
        and abs(sa["scale"] - sb["scale"]) <= 1e-6
        and max(abs(x - y) for x, y in zip(sa["bg"], sb["bg"]))
        <= STYLE_COLOR_TOLERANCE
    )


def _overlaps(block):
    render = (block or {}).get("_render") or {}
    return set(render.get("source_overlap") or render.get("source_conflict") or [])


def _components(ids, links):
    remaining, result = set(ids), []
    while remaining:
        start = min(remaining)
        stack, component = [start], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(links[current] - component)
        remaining -= component
        result.append(component)
    return result


def _direct_links(page, candidates):
    """Relazioni con una prova fisica, prima della chiusura transitiva."""
    links = {candidate["id"]: set() for candidate in candidates}
    evidence = {}
    rows_by_region = {r["id"]: set(r["righe"]) for r in page["sorgente"]["regioni"]}
    blocks = _block_by_region(page)
    for index, a in enumerate(candidates):
        for b in candidates[index + 1:]:
            aid, bid = a["id"], b["id"]
            intersects = _intersects(a["poligono_pt"], b["poligono_pt"])
            reasons = []
            if a["regione_sorgente"] == b["regione_sorgente"] and not intersects:
                reasons.append("contorni_distinti_stessa_regione")
            elif (
                a["regione_sorgente"] != b["regione_sorgente"]
                and intersects
                and _intersection_ratio(a["poligono_pt"], b["poligono_pt"])
                >= SHARED_CONTOUR_MIN_RATIO
                and _styles_compatible(
                    blocks.get(a["regione_sorgente"]),
                    blocks.get(b["regione_sorgente"]),
                )
            ):
                a_cross = _overlaps(blocks.get(a["regione_sorgente"])) & rows_by_region.get(
                    b["regione_sorgente"], set()
                )
                b_cross = _overlaps(blocks.get(b["regione_sorgente"])) & rows_by_region.get(
                    a["regione_sorgente"], set()
                )
                if a_cross or b_cross:
                    reasons.extend(("stile_compatibile", "conflitto_ocr_incrociato"))
            if reasons:
                links[aid].add(bid)
                links[bid].add(aid)
                evidence[frozenset((aid, bid))] = reasons
    return links, evidence


def _relationships(page, candidates):
    links, evidence = _direct_links(page, candidates)
    # Se A-B e B-C hanno prove fisiche, un'eventuale intersezione A-C e'
    # interna alla stessa forma composta e non e' un conflitto indipendente.
    for component in _components([c["id"] for c in candidates], links):
        for aid in component:
            for bid in component:
                if aid >= bid:
                    continue
                links[aid].add(bid)
                links[bid].add(aid)
                evidence.setdefault(frozenset((aid, bid)), ["componente_condivisa"])
    conflicts = {candidate["id"]: set() for candidate in candidates}
    for index, a in enumerate(candidates):
        for b in candidates[index + 1:]:
            if b["id"] in links[a["id"]]:
                continue
            if _intersects(a["poligono_pt"], b["poligono_pt"]):
                conflicts[a["id"]].add(b["id"])
                conflicts[b["id"]].add(a["id"])
    return links, conflicts, evidence


def _bbox(polygons):
    points = [point for poly in polygons for point in poly]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _uncertain_inside(row, polygons):
    return row.get("motivo") in RIGHE_RILETTURA and within_polygons(
        row_polygon(row), polygons, tol=3.0
    )


def _build_units(page, candidates, unresolved, evidence):
    candidate_by_id = {c["id"]: c for c in candidates}
    rows = {r["id"]: r for r in page["sorgente"]["righe"]}
    regions = {r["id"]: r for r in page["sorgente"]["regioni"]}
    blocks = {
        b["id"]: b for b in page.get("blocchi", [])
        if isinstance(b.get("id"), str) and b["id"]
    }
    block_by_region = _block_by_region(page)
    links = {c["id"]: set(c["collegamenti"]) for c in candidates}
    source_order = {r["id"]: r["ordine"] for r in page["sorgente"]["righe"]}
    unresolved_by_id = {r["riga"]: r for r in unresolved}
    units = []
    for component in _components(candidate_by_id, links):
        ordered = sorted(
            component,
            key=lambda ident: (
                min(p[1] for p in candidate_by_id[ident]["poligono_pt"]),
                min(p[0] for p in candidate_by_id[ident]["poligono_pt"]),
                ident,
            ),
        )
        members = [candidate_by_id[ident] for ident in ordered]
        polygons = [c["poligono_pt"] for c in members]
        block_ids = []
        for candidate in members:
            block = block_by_region.get(candidate["regione_sorgente"])
            if block and isinstance(block.get("id"), str) and block["id"] not in block_ids:
                block_ids.append(block["id"])
        assigned = sorted(
            {rid for candidate in members for rid in candidate["righe"]},
            key=lambda rid: (source_order[rid], rid),
        )
        member_source_rows = {
            rid for candidate in members
            for rid in regions[candidate["regione_sorgente"]]["righe"]
        }
        reasons = []
        if any(candidate["conflitti"] for candidate in members):
            reasons.append("sovrapposizione_ambigua")
        if any(not candidate["righe"] for candidate in members):
            reasons.append("lobo_senza_righe")
        if len(block_ids) != len(members):
            reasons.append("testo_non_segmentato_per_lobo")
        if any(rid in unresolved_by_id and rows[rid]["stato"] != "esclusa"
               for rid in member_source_rows):
            reasons.append("righe_acquisite_irrisolte")
        external, uncertain = [], []
        for block_id in block_ids:
            block = blocks[block_id]
            render = block.get("_render") or {}
            if any(render.get(flag) for flag in (
                "page_protected", "contenuto_in", "scartato_come_disegno"
            )):
                reasons.append("protezione_non_locale")
            for rid in _overlaps(block):
                if rid in member_source_rows:
                    continue
                row = rows.get(rid)
                if row and _uncertain_inside(row, polygons):
                    uncertain.append(rid)
                else:
                    external.append(rid)
        if external:
            reasons.append("conflitti_sorgente_esterni")
        reasons = sorted(set(reasons))
        clues = sorted({
            clue
            for aid in component for bid in component if aid < bid
            for clue in evidence.get(frozenset((aid, bid)), [])
        })
        units.append({
            "id": sources.unique_id(units, page["indice"], "u", ordered),
            "tipo": "multilobo" if len(members) > 1 else "singola",
            "stato": "eseguibile" if not reasons else "incompleta",
            "candidati": ordered,
            "regioni_sorgente": [c["regione_sorgente"] for c in members],
            "blocchi": block_ids,
            "righe": assigned,
            "poligoni_pt": copy.deepcopy(polygons),
            "area_scrittura_pt": _bbox(polygons),
            "indizi": clues,
            "righe_incerte": sorted(set(uncertain)),
            "conflitti_sorgente_esterni": sorted(set(external)),
            "motivi": reasons,
        })
    return units


def _diagnostics(page, candidates, unresolved, units):
    rows = {r["id"]: r for r in page["sorgente"]["righe"]}
    excluded = sum(1 for item in unresolved if rows[item["riga"]]["stato"] == "esclusa")
    acquired = len(unresolved) - excluded
    incomplete = [u["id"] for u in units if u["stato"] != "eseguibile"]
    empty = [c["id"] for c in candidates if not c["righe"]]
    certified = not unresolved and not incomplete and not empty
    return {
        "copertura_certificata": certified,
        "bloccante": not certified,
        "righe_irrisolte_acquisite": acquired,
        "righe_escluse": excluded,
        "candidati_vuoti": empty,
        "unita_incomplete": incomplete,
        "unita_eseguibili": [u["id"] for u in units if u["stato"] == "eseguibile"],
    }


def build_structure(page):
    """Genera candidati, proprieta' e unita'; non modifica la pagina."""
    source = page["sorgente"]
    if sources.seal(source)["sha256"] != source.get("sha256"):
        raise ValueError("struttura P3: hash sorgente non valido")
    blocks = _block_by_region(page)
    candidates = []
    for region in source["regioni"]:
        render = (blocks.get(region["id"]) or {}).get("_render", {})
        scale = render.get("scale", 1)
        contours = render.get("fill_poly", [])
        polygons = (
            [[[round(float(v) * scale, 6) for v in p] for p in poly] for poly in contours]
            if contours else [rectangle(region["box_pt"])]
        )
        for poly in polygons:
            if not _valid_polygon(poly):
                continue
            candidates.append({
                "id": sources.unique_id(candidates, page["indice"], "c", [region["id"], poly]),
                "regione_sorgente": region["id"],
                "blocco": (blocks.get(region["id"]) or {}).get("id"),
                "poligono_pt": poly,
                "indizi": ["contorno_esistente" if contours else "gruppo_righe_legacy"],
                "tipo_testo": "non_classificato",
                "righe": [],
                "collegamenti": [],
                "conflitti": [],
            })
    by_region = {r["id"]: set(r["righe"]) for r in source["regioni"]}
    links, conflicts, evidence = _relationships(page, candidates)
    for candidate in candidates:
        candidate["collegamenti"] = sorted(links[candidate["id"]])
        candidate["conflitti"] = sorted(conflicts[candidate["id"]])
    candidates_per_region = {
        region_id: sum(c["regione_sorgente"] == region_id for c in candidates)
        for region_id in by_region
    }
    unresolved = []
    for row in source["righe"]:
        if row.get("stato") == "esclusa":
            unresolved.append({
                "riga": row["id"], "motivo": "sorgente_esclusa",
                "motivo_sorgente": row.get("motivo"),
            })
            continue
        own = [
            c for c in candidates
            if row["id"] in by_region[c["regione_sorgente"]]
            and _inside_candidate(row, c, candidates_per_region[c["regione_sorgente"]])
        ]
        if len(own) == 1 and not own[0]["conflitti"]:
            own[0]["righe"].append(row["id"])
        else:
            unresolved.append({
                "riga": row["id"],
                "motivo": "regione_ambigua" if own else "fuori_candidati",
            })
    rows = {r["id"]: r for r in source["righe"]}
    for candidate in candidates:
        candidate["righe"].sort(
            key=lambda rid: (rows[rid]["box_pt"][1], rows[rid]["box_pt"][0], rid)
        )
    ordered = sorted(
        candidates,
        key=lambda c: (
            min(p[1] for p in c["poligono_pt"]),
            min(p[0] for p in c["poligono_pt"]), c["id"],
        ),
    )
    units = _build_units(page, candidates, unresolved, evidence)
    diagnostics = _diagnostics(page, candidates, unresolved, units)
    result = {
        "versione": STRUCTURE_VERSION,
        "sorgente_sha256": source["sha256"],
        "stato": "pronta" if diagnostics["copertura_certificata"] else "parziale",
        "candidati": candidates,
        "unita": units,
        "irrisolte": unresolved,
        "diagnostica": diagnostics,
        "lettura": {"regioni": [c["id"] for c in ordered], "confermato": False},
        "limiti": [
            "nessuna_classificazione_semantica",
            "ordine_geometrico_non_editoriale",
            "testo_multilobo_non_segmentato_se_unico_blocco",
        ],
    }
    validate_structure(page, result)
    return result


def _validate_v1(page, data):
    """Contratto del primo incremento, conservato per i piani salvati."""
    source = page["sorgente"]
    if data.get("stato") != "ipotesi_non_eseguibile":
        raise ValueError("struttura P3: promozione v1 non supportata")
    rows = {r["id"]: r for r in source["righe"]}
    regions = {r["id"]: r for r in source["regioni"]}
    ids, assigned = set(), set()
    for candidate in data["candidati"]:
        if not isinstance(candidate["id"], str) or candidate["id"] in ids:
            raise ValueError("struttura P3: id candidato duplicato/non valido")
        ids.add(candidate["id"])
        region = regions.get(candidate["regione_sorgente"])
        if region is None or not _valid_polygon(candidate["poligono_pt"]):
            raise ValueError("struttura P3: regione/poligono non valido")
        if candidate["conflitti"] and candidate["righe"]:
            raise ValueError("struttura P3: regione ambigua con proprieta'")
        for rid in candidate["righe"]:
            if rid in assigned or rid not in region["righe"]:
                raise ValueError("struttura P3: riga duplicata o estranea")
            if rows[rid]["stato"] == "esclusa" or not _inside(rows[rid], candidate["poligono_pt"]):
                raise ValueError("struttura P3: riga esclusa/fuori regione")
            assigned.add(rid)
    for item in data["irrisolte"]:
        rid = item["riga"]
        if rid in assigned or rid not in rows or not item.get("motivo"):
            raise ValueError("struttura P3: irrisolte duplicate/non valide")
        assigned.add(rid)
    if assigned != set(rows):
        raise ValueError("struttura P3: righe perse")
    order = data["lettura"]["regioni"]
    if len(order) != len(set(order)) or set(order) != ids or data["lettura"]["confermato"] is not False:
        raise ValueError("struttura P3: ordine duplicato/incompleto o confermato")
    for a in data["candidati"]:
        expected = {
            b["id"] for b in data["candidati"] if b is not a
            and _intersects(a["poligono_pt"], b["poligono_pt"])
        }
        if set(a["conflitti"]) != expected:
            raise ValueError("struttura P3: sovrapposizione non dichiarata")


def validate_structure(page, structure=None):
    """Valida il contratto e ricalcola relazioni, unita' e diagnostica."""
    data = structure if structure is not None else page.get("struttura")
    if data is None:
        return
    source = page["sorgente"]
    if data.get("sorgente_sha256") != source["sha256"]:
        raise ValueError("struttura P3: versione/hash sorgente non valido")
    if data.get("versione") == 1:
        _validate_v1(page, data)
        return
    if data.get("versione") != STRUCTURE_VERSION:
        raise ValueError("struttura P3: versione/hash sorgente non valido")
    rows = {r["id"]: r for r in source["righe"]}
    regions = {r["id"]: r for r in source["regioni"]}
    block_ids = {
        b["id"] for b in page.get("blocchi", [])
        if isinstance(b.get("id"), str) and b["id"]
    }
    candidate_counts = {
        region_id: sum(c.get("regione_sorgente") == region_id
                       for c in data.get("candidati", []))
        for region_id in regions
    }
    ids, assigned = set(), set()
    for candidate in data.get("candidati", []):
        cid = candidate.get("id")
        region = regions.get(candidate.get("regione_sorgente"))
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("struttura P3: id candidato duplicato/non valido")
        ids.add(cid)
        if region is None or not _valid_polygon(candidate.get("poligono_pt")):
            raise ValueError("struttura P3: regione/poligono non valido")
        if candidate.get("blocco") is not None and candidate["blocco"] not in block_ids:
            raise ValueError("struttura P3: blocco candidato inesistente")
        if set(candidate.get("collegamenti", [])) & set(candidate.get("conflitti", [])):
            raise ValueError("struttura P3: relazione candidata incoerente")
        if candidate.get("conflitti") and candidate.get("righe"):
            raise ValueError("struttura P3: regione ambigua con proprieta'")
        for rid in candidate.get("righe", []):
            if rid in assigned or rid not in region["righe"]:
                raise ValueError("struttura P3: riga duplicata o estranea")
            if rows[rid]["stato"] == "esclusa" or not _inside_candidate(
                rows[rid], candidate, candidate_counts[candidate["regione_sorgente"]]
            ):
                raise ValueError("struttura P3: riga esclusa/fuori regione")
            assigned.add(rid)
    unresolved_ids = set()
    for item in data.get("irrisolte", []):
        rid = item.get("riga")
        if rid in assigned or rid in unresolved_ids or rid not in rows or not item.get("motivo"):
            raise ValueError("struttura P3: irrisolte duplicate/non valide")
        unresolved_ids.add(rid)
    if assigned | unresolved_ids != set(rows):
        raise ValueError("struttura P3: righe perse")
    order = data.get("lettura", {}).get("regioni", [])
    if len(order) != len(set(order)) or set(order) != ids or data["lettura"].get("confermato") is not False:
        raise ValueError("struttura P3: ordine duplicato/incompleto o confermato")
    links, conflicts, evidence = _relationships(page, data["candidati"])
    for candidate in data["candidati"]:
        if set(candidate.get("collegamenti", [])) != links[candidate["id"]]:
            raise ValueError("struttura P3: collegamento non dimostrato")
        if set(candidate.get("conflitti", [])) != conflicts[candidate["id"]]:
            raise ValueError("struttura P3: sovrapposizione non dichiarata")
    expected_units = _build_units(page, data["candidati"], data["irrisolte"], evidence)
    if data.get("unita") != expected_units:
        raise ValueError("struttura P3: unita' non coerenti")
    expected_diagnostics = _diagnostics(page, data["candidati"], data["irrisolte"], expected_units)
    if data.get("diagnostica") != expected_diagnostics:
        raise ValueError("struttura P3: diagnostica incoerente")
    expected_state = "pronta" if expected_diagnostics["copertura_certificata"] else "parziale"
    if data.get("stato") != expected_state:
        raise ValueError("struttura P3: stato incoerente")


def activate_validated_units(page):
    """Riattiva soltanto i blocchi di unita' multilobo eseguibili."""
    validate_structure(page)
    structure = page.get("struttura") or {}
    by_id = {b["id"]: b for b in page.get("blocchi", [])}
    changes = 0
    for unit in structure.get("unita", []):
        if unit.get("tipo") != "multilobo" or unit.get("stato") != "eseguibile":
            continue
        if len(unit.get("blocchi", [])) != len(unit.get("candidati", [])):
            continue
        for block_id in unit["blocchi"]:
            block = by_id[block_id]
            render = block.setdefault("_render", {})
            previous = render.pop("source_conflict", None)
            if previous:
                render["p3_conflitti_precedenti"] = list(previous)
            render["p3_unita"] = unit["id"]
            render["p3_multilobo"] = True
            structural_skip = bool(
                previous or render.get("skip_localizzazione") or render.get("compound")
            )
            protected_elsewhere = any(render.get(flag) for flag in (
                "unreviewed", "page_protected", "contenuto_in", "scartato_come_disegno"
            ))
            if block.get("salta") and structural_skip and not protected_elsewhere:
                block["salta"] = False
                changes += 1
    return changes


def adapt_render_entries(page, entries=None):
    """Converte unita' multilobo validate in un singolo job composito."""
    validate_structure(page)
    result = copy.deepcopy(entries if entries is not None else page.get("blocchi", []))
    by_id = {entry["id"]: entry for entry in result}
    absorbed = set()
    for unit in (page.get("struttura") or {}).get("unita", []):
        if unit.get("tipo") != "multilobo" or unit.get("stato") != "eseguibile":
            continue
        member_ids = unit.get("blocchi", [])
        if len(member_ids) != len(unit.get("candidati", [])) or any(
            ident not in by_id for ident in member_ids
        ):
            continue
        members = [by_id[ident] for ident in member_ids]
        if any(member.get("salta") for member in members):
            continue
        if any((m.get("_render") or {}).get("tipo") != "immagine" for m in members):
            continue
        # Profili perfettamente identici sono gia' gestiti correttamente dal
        # rendering in due fasi (prima tutti i riempimenti, poi tutti i testi).
        # Fonderli cambierebbe quale singolo blocco viene marcato se una sola
        # battuta non entra, senza offrire alcun vantaggio geometrico.
        polygon_keys = {
            json.dumps(poly, sort_keys=True, separators=(",", ":"))
            for member in members
            for poly in member["_render"].get("fill_poly", [])
        }
        if len(polygon_keys) < 2:
            continue
        leader = members[0]
        render = leader["_render"]
        boxes = [m["_render"]["box_px"] for m in members]
        # Ogni membro e' un lobo con la sua battuta: il renderer li scrive uno
        # per uno (`comics_translator.disegno.lobi`) invece della concatenazione a cavallo
        # della strozzatura (imperial-guardians p25, prova A1).
        membri = [{
            "id": member["id"],
            "testo": str(member.get("testo") or member.get("automatica") or "").strip(),
            "righe": list((member.get("sorgente") or {}).get("righe") or []),
        } for member in members]
        render["box_px"] = [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ]
        render["fill_poly"] = []
        for member, polygon_pt, membro in zip(members, unit["poligoni_pt"], membri):
            member_render = member["_render"]
            profiles = member_render.get("fill_poly") or []
            if not profiles:
                scale = float(member_render.get("scale", 1.0))
                profiles = [[
                    [round(point[0] / scale, 6), round(point[1] / scale, 6)]
                    for point in polygon_pt
                ]]
            render["fill_poly"].extend(copy.deepcopy(profiles))
            membro["fill_poly"] = copy.deepcopy(profiles)
        render["p3_membri"] = membri
        render["compound"] = False
        render["p3_unita"] = unit["id"]
        render["p3_multilobo"] = True
        render["p3_lobi"] = len(unit["candidati"])
        render["p3_blocchi"] = list(member_ids)
        render["p3_regioni"] = list(unit["regioni_sorgente"])
        for field in ("originale", "automatica", "testo"):
            leader[field] = " ".join(
                str(member.get(field) or "").strip() for member in members
                if str(member.get(field) or "").strip()
            )
        leader["salta"] = False
        absorbed.update(member_ids[1:])
    return [entry for entry in result if entry["id"] not in absorbed]


def structure_issues(plan):
    """Problemi bloccanti da registrare come fase P3 separata."""
    issues = []
    for page in plan.get("pagine", []):
        data = page.get("struttura") or {}
        diagnostic = data.get("diagnostica") or {}
        if not diagnostic.get("bloccante"):
            continue
        ids = list(diagnostic.get("unita_incomplete") or [])
        reason = (
            "%d righe acquisite irrisolte, %d righe escluse, %d candidati vuoti, "
            "%d unita' incomplete"
            % (
                diagnostic.get("righe_irrisolte_acquisite", 0),
                diagnostic.get("righe_escluse", 0),
                len(diagnostic.get("candidati_vuoti") or []),
                len(ids),
            )
        )
        issues.append({"code": "p3_structure_incomplete", "reason": reason,
                       "pages": [page["numero"]], "ids": ids})
    return issues


def add_structure(plan):
    """Analisi esplicita su copia: i vecchi piani non vengono riscritti."""
    sources.validate_sources(plan)
    result = copy.deepcopy(plan)
    if sources.schema_version(result) != 2:
        raise ValueError("P3 richiede un piano v2 con sorgente")
    for page in result["pagine"]:
        page["struttura"] = build_structure(page)
        activate_validated_units(page)
    sources.validate_sources(result)
    return result
