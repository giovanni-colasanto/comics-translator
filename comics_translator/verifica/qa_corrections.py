#!/usr/bin/env python3
"""Applica correzioni QA a un piano e, facoltativamente, rigenera il PDF.

Il file di correzione non duplica il piano tecnico. Registra soltanto modifiche
puntuali e sostituzioni testuali, così resta piccolo, revisionabile e riusabile.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

from comics_translator.sorgente import text_localization as localization
from comics_translator.disegno.plan import EDITABLE, load_plan, save_plan, validate_merges
from comics_translator.sorgente.source_inventory import resolve_correction_ids, validate_sources


def _blocks(plan: dict) -> dict:
    return {
        block["id"]: block
        for page in plan.get("pagine", [])
        for block in page.get("blocchi", [])
        if isinstance(block, dict) and block.get("id")
    }


def _validate_value(field: str, value) -> None:
    if field == "testo" and (not isinstance(value, str) or not value.strip()):
        raise ValueError("'testo' deve essere una stringa non vuota")
    if field == "salta" and not isinstance(value, bool):
        raise ValueError("'salta' deve essere true o false")
    if field == "unisci_con" and value is not None and not isinstance(value, str):
        raise ValueError("'unisci_con' deve essere un id o null")
    if field == "corpo" and value is not None and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(value) or value <= 0
    ):
        raise ValueError("'corpo' deve essere un numero positivo o null")
    if field == "allinea" and value not in (None, "centro", "sinistra"):
        raise ValueError("'allinea' deve essere centro, sinistra o null")


def _registra_decisione_di_salto(block: dict, salta: bool) -> None:
    """Scrive nel blocco CHI ha deciso il salto, prima di cambiarlo.

    `enforce_localization` gira subito dopo questa funzione e revoca i salti
    che ritiene suoi, riconoscendoli dal marcatore `skip_localizzazione`. Un
    marcatore rimasto da un giro precedente gli faceva revocare anche il salto
    appena deciso qui: il blocco tornava attivo e il suo testo originale
    ricompariva nel PDF, dopo essere stato messo da parte apposta.

    Chi chiama decide se una conferma vale come decisione: vedi `decisioni`
    in apply_corrections. Un cambiamento vero del campo lo e' sempre.
    """
    render = block.get("_render")
    if not isinstance(render, dict):
        return
    if salta:
        render.pop("skip_localizzazione", None)
        render["salta_deciso"] = True
    else:
        render.pop("salta_deciso", None)


def apply_corrections(plan: dict, spec: dict, decisioni: bool = True) -> tuple[dict, int]:
    """Restituisce una copia corretta del piano e il numero di interventi.

    `decisioni=False` dice che lo `spec` non e' un elenco di scelte puntuali ma
    l'intera risposta di un lotto di revisione, che rimanda indietro TUTTI i
    blocchi cosi' com'erano. Li' un `salta: true` su un blocco gia' saltato e'
    una conferma, non una decisione: registrarlo congelerebbe per sempre le
    protezioni automatiche, che invece devono decadere col conflitto che le ha
    causate. Restano decisioni i cambiamenti veri, in entrambi i modi.
    """
    validate_sources(plan)
    spec = resolve_correction_ids(plan, spec)
    result = copy.deepcopy(plan)
    by_id = _blocks(result)
    changes = 0

    sets = spec.get("set", {})
    replacements = spec.get("replace", {})
    unlocks = spec.get("sblocca", {})
    if not isinstance(sets, dict) or not isinstance(replacements, dict):
        raise ValueError("'set' e 'replace' devono essere oggetti JSON")
    if not isinstance(unlocks, dict):
        raise ValueError("'sblocca' deve essere un oggetto JSON")
    unknown = (set(sets) | set(replacements) | set(unlocks)) - set(by_id)
    if unknown:
        raise ValueError("id sconosciuti: %s" % ", ".join(sorted(unknown)))

    # Lo sblocco manuale toglie una protezione automatica su un blocco preciso,
    # e obbliga a dire perche': la copertura di quell'area e' stata guardata.
    # Non e' un permesso generale e non sopravvive a un nuovo piano.
    for block_id, motivo in unlocks.items():
        if not isinstance(motivo, str) or not motivo.strip():
            raise ValueError("sblocca.%s: serve il motivo della verifica" % block_id)
        block = by_id[block_id]
        render = block.get("_render")
        if render is None:
            raise ValueError("sblocca.%s: blocco senza stato tecnico" % block_id)
        render["sbloccato_a_mano"] = motivo.strip()
        if block.get("salta"):
            block["salta"] = False
        # Chi sblocca revoca anche la decisione di saltare presa prima.
        render.pop("salta_deciso", None)
        changes += 1

    for block_id, fields in sets.items():
        if not isinstance(fields, dict):
            raise ValueError("set.%s deve essere un oggetto" % block_id)
        invalid = set(fields) - set(EDITABLE)
        if invalid:
            raise ValueError("campi non modificabili per %s: %s" % (
                block_id, ", ".join(sorted(invalid))))
        block = by_id[block_id]
        for field, value in fields.items():
            _validate_value(field, value)
            if field == "testo":
                value = value.strip()
            cambia = block.get(field) != value
            if field == "salta" and (cambia or decisioni):
                _registra_decisione_di_salto(block, value)
            if cambia:
                block[field] = value
                changes += 1

    for block_id, rules in replacements.items():
        if not isinstance(rules, list):
            raise ValueError("replace.%s deve essere una lista" % block_id)
        block = by_id[block_id]
        text = (block.get("testo") or block.get("automatica") or "")
        for index, rule in enumerate(rules, start=1):
            if not isinstance(rule, dict) or set(rule) != {"old", "new"}:
                raise ValueError("replace.%s[%d] richiede solo old e new" % (block_id, index))
            old, new = rule["old"], rule["new"]
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ValueError("old e new devono essere stringhe, old non vuota")
            count = text.count(old)
            if count != 1:
                raise ValueError("replace.%s[%d]: old trovato %d volte" % (block_id, index, count))
            text = text.replace(old, new, 1)
            changes += 1
        block["testo"] = text

    # Una correzione che mette o toglie un `salta` cambia chi viene riscritto,
    # e con lui quali confini vanno ancora protetti.
    localization.enforce_localization(result)
    # La riattivazione richiesta a mano non deve fallire in silenzio: se la
    # protezione e' ancora necessaria l'operatore deve saperlo.
    riattivati = [
        block_id for block_id, fields in sets.items()
        if fields.get("salta") is False and _blocks(result)[block_id].get("salta")
    ]
    if riattivati:
        raise ValueError(
            "blocco protetto da source_conflict riattivato: %s"
            % ", ".join(sorted(riattivati))
        )
    validate_sources(result)
    problems = validate_merges(result)
    if problems:
        raise ValueError("fusioni non valide: %s" % "; ".join(problems))
    return result, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="piano.applicato.json di partenza")
    parser.add_argument("--corrections", required=True, type=Path, help="file JSON con set/replace")
    parser.add_argument("--out-plan", required=True, type=Path, help="piano corretto da scrivere")
    parser.add_argument("--input-pdf", type=Path, help="PDF sorgente; se presente rigenera il documento")
    parser.add_argument("--output-pdf", type=Path, help="PDF corretto da produrre")
    parser.add_argument("--from", dest="source", default="en")
    parser.add_argument("--to", dest="target", default="it")
    args = parser.parse_args()

    if bool(args.input_pdf) != bool(args.output_pdf):
        parser.error("--input-pdf e --output-pdf vanno passati insieme")
    plan = load_plan(args.plan)
    try:
        with args.corrections.open(encoding="utf-8") as handle:
            spec = json.load(handle)
        corrected, changes = apply_corrections(plan, spec)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    save_plan(args.out_plan, corrected)
    print("Piano corretto: %d interventi -> %s" % (changes, args.out_plan))

    if args.input_pdf:
        command = [
            sys.executable, "-m", "comics_translator.disegno.pdf_translate_overlay",
            str(args.input_pdf), "--plan-in", str(args.out_plan),
            "--from", args.source, "--to", args.target,
            "--output", str(args.output_pdf),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
