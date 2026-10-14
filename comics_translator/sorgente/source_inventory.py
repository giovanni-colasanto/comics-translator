"""Piano v2: osservazioni sorgente persistenti, indipendenti dal rendering.

Le regioni sono i gruppi dell'analisi esistente, NON una segmentazione dei
lobi. Una riga esclusa rimane un'osservazione: non diventa testo tradotto.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math


def digest(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def unique_id(items, page_no, kind, identity):
    """Coordinate fisiche, non pixel/DPI; distinguere anche duplicati esatti.

    La stabilita' vale per le stesse osservazioni, non promette che una nuova
    esecuzione OCR a un altro DPI trovi identici testi e coordinate.
    """
    base = "p%d-%s%s" % (page_no + 1, kind, digest(identity)[:16])
    used = {item["id"] for item in items}
    candidate, occurrence = base, 1
    while candidate in used:
        occurrence += 1
        candidate = "%s-%d" % (base, occurrence)
    return candidate


def coordinates(box, scale=1.0):
    return [round(float(v) * scale, 6) for v in box]


def new_inventory():
    return {"disponibilita": "rilevata", "acquisizioni": [], "righe": [], "regioni": []}


def add_line(
    inventory,
    page_no,
    *,
    raw,
    normalized,
    box,
    kind,
    scale=1.0,
    quad=None,
    confidence=None,
    reason=None,
    direction=None,
):
    box_pt = coordinates(box, scale)
    identity = [kind, raw, [round(v, 3) for v in box_pt]]
    lines = inventory["righe"]
    line = {
        "id": unique_id(lines, page_no, "l", identity),
        "tipo": kind,
        "testo_grezzo": raw,
        "testo_normalizzato": normalized,
        "box_pt": box_pt,
        # Conservare i float grezzi: arrotondare i pixel prima di validarli
        # contro i punti puo' introdurre un secondo arrotondamento incoerente.
        "box_px": [float(v) for v in box] if kind == "ocr" else None,
        "quad_px": [[float(v) for v in p] for p in quad]
        if quad and kind == "ocr"
        else None,
        "quad_pt": [coordinates(p, scale) for p in quad] if quad else None,
        "scala_px_pt": scale if kind == "ocr" else None,
        "confidenza": confidence,
        "direzione": list(direction) if direction else None,
        "ordine": len(lines),
        "stato": "esclusa" if reason else "acquisita",
        "motivo": reason,
    }
    lines.append(line)
    return line["id"]


def add_region(
    inventory, page_no, line_ids, box_pt, text, *, kind, reason=None, box_px=None
):
    regions = inventory["regioni"]
    box_pt = coordinates(box_pt)
    region = {
        "id": unique_id(
            regions,
            page_no,
            "r",
            [kind, sorted(line_ids), [round(v, 3) for v in box_pt], text],
        ),
        "tipo": kind,
        "tipo_testo": "non_classificato",
        "righe": list(line_ids),
        "box_pt": box_pt,
        "box_px": box_px,
        "testo_normalizzato": text,
        "ordine": len(regions),
        "stato": "esclusa" if reason else "candidata",
        "motivo": reason,
    }
    regions.append(region)
    if reason:
        members = set(line_ids)
        for line in inventory["righe"]:
            if line["id"] in members and not line["motivo"]:
                line.update(stato="esclusa", motivo=reason)
    return region["id"]


def seal(inventory):
    result = copy.deepcopy(inventory)
    result.pop("sha256", None)
    result["sha256"] = digest(result)
    return result


def attach_sources(page, observations=None, *, preserve_ids=False):
    """Collega blocchi e osservazioni su una copia; legacy senza righe inventate."""
    inventory = (
        copy.deepcopy(observations) if observations is not None else new_inventory()
    )
    inventory.pop("sha256", None)
    if observations is None:
        inventory["disponibilita"] = "legacy_senza_righe"
    by_region = {r["id"]: r for r in inventory["regioni"]}
    new_blocks = []
    for block in page["blocchi"]:
        old_id = block["id"]
        region_id = block.pop("_source_region_id", None)
        if observations is None:
            region_id = None
        if region_id is None:
            render = block.get("_render") or {}
            box = (
                render.get("box_px")
                if render.get("tipo") == "immagine"
                else render.get("rect")
            )
            scale = render.get("scale", 1) if render.get("tipo") == "immagine" else 1
            box_pt = coordinates(box, scale) if box else [0, 0, 0, 0]
            region_id = add_region(
                inventory,
                page["numero"] - 1,
                [],
                box_pt,
                block.get("originale", ""),
                kind="legacy",
                reason="righe_sorgente_non_disponibili",
            )
            by_region[region_id] = inventory["regioni"][-1]
        if region_id not in by_region:
            raise ValueError("regione sorgente inesistente: %s" % region_id)
        block["legacy_id"] = old_id
        block["sorgente"] = {
            "regione": region_id,
            "righe": list(by_region[region_id]["righe"]),
        }
        if not preserve_ids:
            block["id"] = unique_id(new_blocks, page["numero"] - 1, "b", region_id)
        new_blocks.append(block)
    # Ordine candidato, non una comprensione della sequenza delle vignette.
    inventory["lettura"] = {
        "metodo": "geometrico_y_x",
        "confermato": False,
        "regioni": [
            r["id"]
            for r in sorted(
                inventory["regioni"],
                key=lambda r: (r["box_pt"][1], r["box_pt"][0], r["ordine"]),
            )
        ],
    }
    page["sorgente"] = seal(inventory)


def schema_version(plan):
    version = plan.get("versione", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError(
            "versione piano non supportata: %r (supportate: 1, 2)" % version
        )
    return version


def _finite_vector(value, size):
    return (
        isinstance(value, list)
        and len(value) == size
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    )


def validate_sources(plan):
    """Valida v2 completo, non le proiezioni compatte inviate all'AI."""
    if schema_version(plan) == 1:
        return
    all_ids, block_ids, page_numbers = set(), set(), set()
    if not isinstance(plan.get("pagine"), list):
        raise ValueError("pagine v2 non valide")
    for page in plan["pagine"]:
        inventory = page.get("sorgente")
        if not isinstance(inventory, dict):
            raise ValueError("piano v2 senza inventario sorgente")
        if inventory.get("sha256") != seal(inventory)["sha256"]:
            raise ValueError("inventario sorgente modificato: hash non valido")
        number = page.get("numero")
        if type(number) is not int or number < 1 or number in page_numbers:
            raise ValueError("numero pagina mancante o duplicato")
        if "indice" in page and page["indice"] != number - 1:
            raise ValueError("indice pagina incoerente")
        page_numbers.add(number)
        lines, regions = {}, {}
        for key, index in (("righe", lines), ("regioni", regions)):
            items = inventory.get(key)
            if not isinstance(items, list):
                raise ValueError("inventario senza %s" % key)
            for order, item in enumerate(items):
                ident = item.get("id")
                if not isinstance(ident, str) or not ident or ident in all_ids:
                    raise ValueError("id sorgente mancante o duplicato: %s" % ident)
                all_ids.add(ident)
                index[ident] = item
                if item.get("ordine") != order or not _finite_vector(
                    item.get("box_pt"), 4
                ):
                    raise ValueError(
                        "ordine/coordinate sorgente non validi: %s" % ident
                    )
                allowed = (
                    ("acquisita", "esclusa")
                    if key == "righe"
                    else ("candidata", "esclusa")
                )
                if item.get("stato") not in allowed:
                    raise ValueError("stato sorgente non valido: %s" % ident)
                if item.get("stato") == "esclusa" and not item.get("motivo"):
                    raise ValueError("scarto senza motivo: %s" % ident)
                if key == "righe":
                    if not all(
                        isinstance(item.get(k), str)
                        for k in ("testo_grezzo", "testo_normalizzato")
                    ):
                        raise ValueError("testo riga non valido: %s" % ident)
                    if item.get("tipo") == "ocr":
                        scale = item.get("scala_px_pt")
                        if (
                            not isinstance(scale, (int, float))
                            or isinstance(scale, bool)
                            or not math.isfinite(scale)
                            or scale <= 0
                            or not _finite_vector(item.get("box_px"), 4)
                        ):
                            raise ValueError(
                                "scala/coordinate OCR non valide: %s" % ident
                            )
                        if coordinates(item["box_px"], scale) != item["box_pt"]:
                            raise ValueError("coordinate px/pt incoerenti: %s" % ident)
                    confidence = item.get("confidenza")
                    if confidence is not None and (
                        not _finite_vector([confidence], 1) or not 0 <= confidence <= 1
                    ):
                        raise ValueError("confidenza non valida: %s" % ident)
                    quad = item.get("quad_pt")
                    if quad is not None and (
                        not isinstance(quad, list)
                        or len(quad) != 4
                        or not all(_finite_vector(p, 2) for p in quad)
                    ):
                        raise ValueError("quadrilatero non valido: %s" % ident)
                    quad_px = item.get("quad_px")
                    if quad_px is not None:
                        if (
                            item.get("tipo") != "ocr"
                            or not isinstance(quad_px, list)
                            or len(quad_px) != 4
                            or not all(_finite_vector(p, 2) for p in quad_px)
                            or [coordinates(p, item["scala_px_pt"]) for p in quad_px]
                            != quad
                        ):
                            raise ValueError(
                                "quadrilatero px/pt incoerente: %s" % ident
                            )
        reading = inventory.get("lettura") or {}
        refs = reading.get("regioni")
        if (
            reading.get("metodo") != "geometrico_y_x"
            or reading.get("confermato") is not False
            or not isinstance(refs, list)
            or not all(isinstance(r, str) for r in refs)
            or len(refs) != len(regions)
            or set(refs) != set(regions)
        ):
            raise ValueError("ordine di lettura sorgente non valido")
        assigned = set()
        for region in regions.values():
            refs = region.get("righe")
            if (
                not isinstance(refs, list)
                or not all(isinstance(r, str) for r in refs)
                or len(refs) != len(set(refs))
                or set(refs) - set(lines)
            ):
                raise ValueError("riferimenti righe non validi: %s" % region["id"])
            if assigned.intersection(refs):
                raise ValueError("riga assegnata a piu' regioni")
            assigned.update(refs)
        for ident, line in lines.items():
            if ident not in assigned and line.get("stato") != "esclusa":
                raise ValueError("riga acquisita senza regione: %s" % ident)
        used_regions = set()
        for block in page["blocchi"]:
            ident = block.get("id")
            if not isinstance(ident, str) or not ident or ident in block_ids:
                raise ValueError("id blocco mancante o duplicato: %s" % ident)
            block_ids.add(ident)
            refs = block.get("sorgente") or {}
            region = regions.get(refs.get("regione"))
            if region is None or refs.get("righe") != region["righe"]:
                raise ValueError(
                    "blocco senza riferimenti sorgente validi: %s" % block["id"]
                )
            if region["id"] in used_regions:
                raise ValueError("regione assegnata a piu' blocchi")
            used_regions.add(region["id"])
        from comics_translator.sorgente.text_localization import validate_localization
        validate_localization(inventory, page["blocchi"])
        from comics_translator.sorgente.page_structure import validate_structure
        validate_structure(page)


def resolve_correction_ids(plan, spec):
    """Alias v1 ammessi solo se univoci; nessun recupero basato su somiglianza."""
    if schema_version(plan) == 1:
        return spec
    blocks = [b for page in plan["pagine"] for b in page["blocchi"]]
    exact = {b["id"] for b in blocks}
    aliases = {}
    for block in blocks:
        if block.get("legacy_id"):
            aliases.setdefault(block["legacy_id"], []).append(block["id"])

    def resolve(ident):
        if not isinstance(ident, str):  # il validatore QA descrive il tipo errato
            return ident
        if ident in exact or ident not in aliases:
            return ident
        if len(aliases[ident]) != 1:
            raise ValueError("id legacy ambiguo: %s; usare un id v2" % ident)
        return aliases[ident][0]

    result = copy.deepcopy(spec)
    for key in ("set", "replace"):
        original = result.get(key, {})
        if not isinstance(original, dict):
            continue  # il validatore QA produce l'errore consueto
        mapped = {}
        for ident, fields in original.items():
            canonical = resolve(ident)
            if canonical in mapped:
                raise ValueError("correzioni duplicate tramite alias: %s" % canonical)
            if key == "set" and isinstance(fields, dict) and fields.get("unisci_con"):
                fields["unisci_con"] = resolve(fields["unisci_con"])
            mapped[canonical] = fields
        result[key] = mapped
    return result
