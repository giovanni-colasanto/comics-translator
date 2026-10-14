"""Banco di prova sugli albi gia' chiusi a mano: quanto del lavoro manuale la pipeline fa ora.

Ogni cartella `lavoro/FATTO <nome>` contiene il piano che la pipeline ha
prodotto e quello che un operatore (umano o un'altra chat) ha corretto per
consegnare il PDF: `piano.manuale.json`. Sono coppie etichettate gratis, e
misurare una modifica li' costa zero crediti.

Il replay e' offline e rifa' solo le parti che non chiedono il modello:

1. **recupero dei lotti**: le risposte salvate (`piano.rivisto.lottoNN.json`)
   si rivalidano con la validazione di adesso, recupero per blocco compreso,
   e si confronta con cio' che la run originale aveva accettato;
2. **sblocco automatico, a oracolo**: le vecchie risposte non hanno la `natura`
   e non traducevano i blocchi protetti. Si usano allora la traduzione e la
   scelta dell'operatore come se le avesse date il modello (dialogo se l'ha
   disegnato, grafica se no), e si misura quanti dei suoi sblocchi supera la
   sola verifica geometrica - e quanti ne rifiuta, con quale motivo. E' la
   parte che il programma fa da solo; la classificazione resta del modello;
3. **candidati del Gap C**: quanti blocchi creati a mano stavano su una regione
   che ora diventa candidata.

Non misura la qualita' della traduzione ne' la geometria dei riempimenti che
l'operatore ha ritoccato: quelle restano da guardare sulla tavola.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

from comics_translator import manifesto_lotti
from comics_translator.disegno import sblocco_automatico
from comics_translator.sorgente import text_localization as localization
from comics_translator.traduzione import ai_review
from comics_translator.disegno.image_overlay import candidato_scartato


def _norm(testo):
    return re.sub(r"[\W_]+", "", testo or "").casefold()


def _blocchi(plan):
    return {b["id"]: b for p in plan.get("pagine", []) for b in p.get("blocchi", [])}


def _tradotto(block):
    return (not block.get("salta") and bool(_norm(block.get("testo")))
            and _norm(block.get("testo")) != _norm(block.get("originale")))


def _carica(percorso):
    return json.loads(Path(percorso).read_text(encoding="utf-8"))


def replay_lotti(base, rev):
    """Rivalida le risposte salvate: (piano rivisto, id vecchi, id nuovi, esiti)."""
    manifesto = manifesto_lotti.leggi(rev)
    pagine = {p["numero"]: p for p in base["pagine"]}
    rivisti, vecchi, nuovi = [], set(), set()
    esiti = Counter()
    for etichetta, voce in sorted(manifesto.items()):
        lotto = [pagine[n] for n in voce.get("pagine") or [] if n in pagine]
        id_lotto = [b["id"] for p in lotto for b in p["blocchi"]]
        if voce.get("stato") == manifesto_lotti.VALIDO:
            vecchi.update(id_lotto)
        risposta = rev / ("piano.rivisto.%s.json" % etichetta)
        if not lotto or not risposta.is_file():
            esiti["senza_risposta"] += 1
            continue
        _, parziale, _, errore, _ = ai_review._valuta_lotto(lotto, risposta, "")
        if parziale is None or not parziale["pagine"]:
            esiti["perso"] += 1
            continue
        esiti["valido" if errore is None else "recuperato_per_blocco"] += 1
        rivisti.extend(parziale["pagine"])
        nuovi.update(b["id"] for p in parziale["pagine"] for b in p["blocchi"])
    return {"pagine": rivisti}, vecchi, nuovi, esiti


def misura_albo(cartella):
    cartella = Path(cartella)
    rev = next(cartella.glob("*.it.revisione"), None)
    if rev is None or not (rev / "piano.json").is_file() or not (rev / "piano.manuale.json").is_file():
        return {"albo": cartella.name, "saltato": "manca piano.json o piano.manuale.json"}
    base = _carica(rev / "piano.json")
    manuale = _carica(rev / "piano.manuale.json")
    finale_auto = next((rev / n for n in ("piano.controllato.json", "piano.applicato.json")
                        if (rev / n).is_file()), None)
    automatico = _carica(finale_auto) if finale_auto else base
    per_base, per_manuale, per_auto = _blocchi(base), _blocchi(manuale), _blocchi(automatico)

    # --- 1. recupero dei lotti -------------------------------------------------
    rivisto, vecchi, nuovi, esiti = replay_lotti(base, rev)
    recuperati = sorted(nuovi - vecchi)
    # blocchi che l'operatore ha dovuto sbloccare perche' il loro lotto era perso
    sbloccati_a_mano = {bid for bid, b in per_manuale.items()
                        if bid in per_auto and per_auto[bid].get("salta") and _tradotto(b)}
    unreviewed_a_mano = {bid for bid in sbloccati_a_mano
                         if (per_auto[bid].get("_render") or {}).get("unreviewed")}
    risultato = {
        "albo": cartella.name,
        "pagine": len(base["pagine"]),
        "blocchi": len(per_base),
        "lotti": dict(esiti),
        "blocchi_recuperati_dai_lotti": len(recuperati),
        "unreviewed_sbloccati_a_mano": len(unreviewed_a_mano),
        "unreviewed_a_mano_ora_recuperati": len(unreviewed_a_mano & set(recuperati)),
    }

    # --- 2. sblocco automatico a oracolo ---------------------------------------
    try:
        piano, _ = ai_review.apply_review(copy.deepcopy(base), rivisto)
    except (ValueError, SystemExit) as exc:
        risultato["sblocco"] = {"errore": "replay non applicabile: %s" % exc}
        piano = None
    if piano is not None:
        if str(base.get("motore", "")).lower() in ("", "none"):
            for pagina in piano["pagine"]:
                for block in pagina["blocchi"]:
                    if block["id"] not in nuovi:
                        block["salta"] = True
                        block.setdefault("_render", {})["unreviewed"] = True
        localization.enforce_localization(piano)
        prima = {bid for bid, b in _blocchi(piano).items() if not b.get("salta")}
        for bid, block in _blocchi(piano).items():
            fatto = per_manuale.get(bid)
            if fatto is None:
                continue
            if _tradotto(fatto):
                block["natura"] = "dialogo"
                block["testo"] = fatto["testo"]
                # l'oracolo e' sul contenuto, non sui guasti di lotto gia' misurati sopra
                block.setdefault("_render", {}).pop("unreviewed", None)
            else:
                block["natura"] = "grafica"
        localization.enforce_localization(piano)
        try:
            decisioni = sblocco_automatico.sblocca_tradotti(piano)
        except ValueError as exc:
            decisioni = []
            risultato["sblocco_errore"] = str(exc)
        dopo = _blocchi(piano)
        protetti_a_mano = {bid for bid in sbloccati_a_mano
                           if any((per_auto[bid].get("_render") or {}).get(k) for k in
                                  ("source_conflict", "skip_localizzazione"))}
        sbloccati = {d["id"] for d in decisioni if d["esito"] == "sbloccato"}
        rifiuti = Counter((d.get("motivo") or "?").split(":")[0]
                          for d in decisioni if d["esito"] != "sbloccato")
        disegnati_ora = {bid for bid, b in dopo.items() if not b.get("salta")}
        risultato["sblocco"] = {
            "protetti_sbloccati_a_mano": len(protetti_a_mano),
            "ora_disegnati": len(protetti_a_mano & disegnati_ora),
            "sbloccati_dal_modulo": len(sbloccati & protetti_a_mano),
            "rilasciati_dal_ricalcolo": len((protetti_a_mano & disegnati_ora) - sbloccati),
            "rifiutati": dict(rifiuti),
            "disegnati_contro_la_scelta_manuale": sorted(
                bid for bid in disegnati_ora - prima
                if bid in per_manuale and per_manuale[bid].get("salta")),
        }

    # --- 3. candidati del Gap C -------------------------------------------------
    regioni = {r["id"]: r for p in base["pagine"] for r in (p.get("sorgente") or {}).get("regioni", [])}
    nuovi_a_mano = [b for bid, b in per_manuale.items() if bid not in per_base and _tradotto(b)]
    su_regione_scartata = [b for b in nuovi_a_mano
                           if (regioni.get((b.get("sorgente") or {}).get("regione")) or {})
                           .get("motivo") == "background_noise"]
    candidati = [b for b in su_regione_scartata
                 # stessa regola dell'analisi (plan_page), non la sola forma del
                 # Gap C: le didascalie di una riga sono candidate anche loro
                 if candidato_scartato(regioni[b["sorgente"]["regione"]].get("testo_normalizzato"),
                                       len(regioni[b["sorgente"]["regione"]].get("righe") or []))]
    risultato["gap_c"] = {
        "blocchi_creati_a_mano": len(nuovi_a_mano),
        "su_regione_background_noise": len(su_regione_scartata),
        "ora_candidati": len(candidati),
    }
    return risultato


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay offline sugli albi FATTO")
    ap.add_argument("--radice", type=Path, required=True,
                    help="cartella lavoro/ che contiene le cartelle 'FATTO <nome>'")
    ap.add_argument("--json", dest="json_out", type=Path)
    args = ap.parse_args(argv)
    risultati = []
    for cartella in sorted(args.radice.glob("FATTO *")):
        print("== %s" % cartella.name, flush=True)
        esito = misura_albo(cartella)
        risultati.append(esito)
        print(json.dumps(esito, ensure_ascii=False))
    if args.json_out:
        args.json_out.write_text(json.dumps(risultati, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
