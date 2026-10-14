"""Cosa la pipeline ha buttato via o lasciato in inglese, detto ad alta voce.

Due guasti misurati sulle prove reali hanno la stessa forma: il programma
sbaglia in SILENZIO e il difetto arriva fino al PDF senza che niente lo dica.

- Gap C: l'esclusione `background_noise` scarta un gruppo OCR PRIMA della
  traduzione. Su Imperial ha buttato interi balloon di dialogo, mai passati ne'
  al traduttore ne' al QA e invisibili anche all'audit raster, che misura la
  copertura sulla sorgente sigillata: una riga mai acquisita non manca a
  nessuno.
- Gap B: il modello restituisce l'inglese identico all'originale. Non e'
  distinguibile in automatico da uno skip legittimo - onomatopee, nomi e
  insegne copiano anche loro l'originale - ma su una frase intera la copia non
  e' una scelta editoriale, e' degrado.
- Gap D: il blocco arriva al modello gia' protetto (`salta: true`), il modello
  fa quello che il prompt gli dice - ricopia l'originale - e il blocco esce in
  inglese. Nessuno degli altri tre lo vede: il Gap A cerca una traduzione
  nascosta che qui non e' mai stata chiesta, il Gap B esclude i blocchi saltati
  per costruzione, il Gap C guarda gli scarti dell'OCR e questi sono acquisiti.
  E' il piu' grande dei quattro, e il danno visivo e' peggiore di un balloon
  lasciato intero in inglese, perche' la protezione colpisce FRAMMENTI: su
  carnage-3 p15 si leggeva «DECINE / PEOPLE DIED IN THAT POLICE STATION
  TODAY», meta' italiano e meta' inglese nella stessa nuvoletta.

Come per `traduzioni_soppresse` (Gap A), qui non si CORREGGE niente: alzare la
soglia del rumore per far passare un balloon significa coprire di bianco
l'onomatopea disegnata della tavola accanto, e il programma non distingue le
due cose. Si dichiara, e la decisione resta all'operatore.

Le soglie sotto non sono inventate: vengono dalla misura sui piani gia' salvati
(`python -m comics_translator.verifica.segnalazioni`), che si puo' rifare quando il corpus
cresce.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Gap C: gruppi OCR scartati come disegno
# --------------------------------------------------------------------------- #

# Misura su 212 gruppi scartati in 17 piani salvati (Daredevil #2, Imperial
# parte 2, Captain America #2, Deadpool 13): quello che separa il dialogo perso
# dal lettering d'arte non e' il rumore dello sfondo - i valori si accavallano -
# ma la FORMA del blocco. Un balloon ha piu' righe e molti caratteri; un logo,
# un'onomatopea o un prezzo di copertina stanno in una riga corta.
RIGHE_BLOCCO_TESTO = 3
CARATTERI_BLOCCO_TESTO = 30

# Seconda forma: la didascalia breve ("LATER.", "NEW YORK.") che il blocco di
# testo non prende. Chiude la frase con un punto, non con un punto esclamativo:
# nel corpus il punto esclamativo e' il marchio delle copertine e delle
# onomatopee ("TOMBSTONE!", "SPIDEY CUTS LOOSE!"), il punto quello delle
# didascalie.
FINE_DIDASCALIA = (".", ":")
CARATTERI_DIDASCALIA = 5

# Terza forma, aggiunta dopo carnage-3: li' il punto esclamativo escluso da solo
# ha fatto perdere un balloon vero, "I DON'T LIKE THAT ONE BIT!" (2 righe, 24
# caratteri: sotto la soglia del blocco di testo). Ammesso, ma con due vincoli
# presi dai casi che devono restare fuori: piu' di una riga, perche' onomatopee
# e blurb di copertina stanno su una sola; e almeno tre parole DIVERSE, perche'
# "BANG! BANG! BANG!" di parole ne ha tre ma una sola distinta. Sui 238 scarti
# del corpus questa forma aggiunge due voci: il balloon di carnage-3 e un
# titolo di copertina - che testo e' comunque.
PAROLE_ESCLAMATIVA = 3


def _testo_regione(regione):
    return (regione.get("testo_normalizzato") or regione.get("testo") or "").strip()


def _sembra_blocco_di_testo(testo, righe):
    return righe >= RIGHE_BLOCCO_TESTO and len(testo) >= CARATTERI_BLOCCO_TESTO


def _ha_lettere(testo):
    # senza lettere e' una misura, un prezzo o una data: quelli restano disegno
    return any(ch.isalpha() for ch in testo)


def _sembra_didascalia(testo):
    if len(testo) < CARATTERI_DIDASCALIA or not testo.endswith(FINE_DIDASCALIA):
        return False
    return _ha_lettere(testo)


def _sembra_esclamativa(testo, righe):
    if righe < 2 or not testo.endswith("!"):
        return False
    return len(set(testo.split())) >= PAROLE_ESCLAMATIVA and _ha_lettere(testo)


def forma_di_testo(testo, righe):
    """La forma di testo di un gruppo scartato, o None se sembra disegno.

    E' la stessa prova usata per dichiarare il Gap C, esposta perche' l'analisi
    la usa per trasformare quei gruppi in blocchi candidati (saltati) invece di
    buttarli: il modello li vedra' e dira' se sono dialogo.
    """
    testo = (testo or "").strip()
    if _sembra_blocco_di_testo(testo, righe):
        return "blocco_di_testo"
    if _sembra_didascalia(testo):
        return "didascalia"
    if _sembra_esclamativa(testo, righe):
        return "esclamativa"
    return None


def testo_scartato_come_disegno(plan):
    """Gruppi esclusi da `background_noise` che hanno la forma del testo.

    Restituisce le voci da rivedere, con il motivo per cui sono state pescate:
    l'operatore deve poter dire "questo e' un balloon" o "questo e' il logo"
    guardando la tavola, non fidandosi di un numero. Un gruppo che e' gia'
    diventato blocco candidato non si dichiara: e' arrivato al modello.
    """
    sospetti = []
    for page in plan.get("pagine", []) or []:
        if not isinstance(page, dict):
            continue
        inventario = page.get("sorgente") or {}
        candidate = {
            (b.get("sorgente") or {}).get("regione")
            for b in page.get("blocchi", []) or [] if isinstance(b, dict)
        }
        for regione in inventario.get("regioni", []) or []:
            if not isinstance(regione, dict) or regione.get("motivo") != "background_noise":
                continue
            if regione.get("id") in candidate:
                continue
            testo = _testo_regione(regione)
            righe = len(regione.get("righe") or [])
            forma = forma_di_testo(testo, righe)
            if forma is None:
                continue
            sospetti.append({
                "pagina": page.get("numero"),
                "id": regione.get("id"),
                "forma": forma,
                "righe": righe,
                "rumore_sfondo": regione.get("rumore_sfondo"),
                "testo": testo[:80],
            })
    return sospetti


# --------------------------------------------------------------------------- #
# Gap B: traduzione uguale all'originale
# --------------------------------------------------------------------------- #

# Un'onomatopea, un nome proprio o un'insegna stanno in una o due parole e
# copiare l'originale li' e' la scelta giusta. Da tre parole in su non e' piu'
# una scelta: e' una frase rimasta in inglese.
PAROLE_FRASE = 3


def _normalizza(testo):
    return " ".join((testo or "").split()).casefold()


def traduzione_uguale_all_originale(plan):
    """Blocchi non saltati in cui il modello ha ricopiato l'inglese.

    Il confronto ignora spazi e maiuscole: rimettere in maiuscolo la stessa
    frase inglese non e' una traduzione, e cercare la copia esatta se la
    lascerebbe passare.
    """
    copie = []
    for page in plan.get("pagine", []) or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocchi", []) or []:
            if not isinstance(block, dict) or block.get("salta"):
                continue
            testo = _normalizza(block.get("testo"))
            originale = _normalizza(block.get("originale"))
            if not testo or testo != originale:
                continue
            if len(testo.split()) < PAROLE_FRASE:
                continue  # una o due parole: puo' essere un nome o un suono
            copie.append({
                "pagina": page.get("numero"),
                "id": block.get("id"),
                "parole": len(testo.split()),
                "testo": (block.get("originale") or "").strip()[:80],
            })
    return copie


# --------------------------------------------------------------------------- #
# Gap D: dialogo protetto prima ancora di essere tradotto
# --------------------------------------------------------------------------- #

# Qui il conteggio delle parole non serve: l'OCR dei fumetti perde gli spazi, e
# "Peoplediedinthat policestation" e' una frase di sei parole scritta in due.
# Quello che separa il parlato dal suono sono le VOCALI: "Ktk k-clk" e "Hnh."
# non ne hanno nessuna, mentre ogni parola vera ne ha. La `y` conta come vocale,
# altrimenti "Why?Why" - dialogo vero, misurato su carnage-3 - finirebbe fra i
# rumori.
VOCALI = set("aeiouyàèéìòùâêîôûäëïöü")
VOCALI_MINIME = 2
QUOTA_VOCALI = 0.20
# A 4 scartava "You": tre lettere vere, due vocali, source_conflict con la riga
# "DARE?!" proprio accanto - la meta' del balloon "YOU DARE?!" di doom-1-2 (9
# settembre 2026), invisibile a ogni segnale finche' non si e' tradotto il
# balloon vicino. Il filtro vero resta VOCALI_MINIME/QUOTA_VOCALI: a 3 caratteri
# un rumore da due vocali su tre lettere e' raro quanto a quattro.
CARATTERI_PARLATO = 3


def _sembra_parlato(testo):
    lettere = [c for c in testo.lower() if c.isalpha()]
    if len(testo) < CARATTERI_PARLATO or not lettere:
        return False
    vocali = sum(c in VOCALI for c in lettere)
    return vocali >= VOCALI_MINIME and vocali / len(lettere) >= QUOTA_VOCALI


def dialogo_protetto_non_tradotto(plan):
    """Blocchi che la protezione ha messo a tacere PRIMA della traduzione.

    Il contratto e' preciso: il blocco salta per `source_conflict` (la
    protezione automatica, non una scelta della revisione), nessuno l'ha
    sbloccato a mano, e il testo ripete l'originale - segno che il modello non
    ha mai tradotto, perche' `salta: true` gli diceva di lasciar stare.

    Come per gli altri, non si sblocca da soli: la protezione conservativa
    resta giusta, e solo chi guarda la tavola sa se sotto c'e' un balloon o un
    numero disegnato. Si dichiara, e si recupera con uno `sblocca` motivato
    piu' la traduzione, che qui va scritta perche' non esiste.
    """
    protetti = []
    for page in plan.get("pagine", []) or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocchi", []) or []:
            if not isinstance(block, dict) or not block.get("salta"):
                continue
            render = block.get("_render") or {}
            if not render.get("source_conflict") or render.get("sbloccato_a_mano") \
                    or render.get("sbloccato_auto"):
                continue
            testo = (block.get("testo") or "").strip()
            if _normalizza(testo) != _normalizza(block.get("originale")):
                continue     # una traduzione c'e': la dichiara il Gap A
            if not _sembra_parlato(testo):
                continue     # senza vocali e' un suono, e li' l'inglese e' giusto
            protetti.append({
                "pagina": page.get("numero"),
                "id": block.get("id"),
                "testo": testo[:80],
            })
    return protetti


# --------------------------------------------------------------------------- #
# Gap E: la traduzione c'e' ma non e' stata disegnata
# --------------------------------------------------------------------------- #


def traduzione_non_disegnata(plan):
    """Blocchi tradotti che il renderer ha rifiutato e riportato a `salta`.

    E' il difetto piu' silenzioso, ed e' quello che manda piu' inglese sulla
    tavola. `_segna_illeggibili` marca `salta: true` il blocco il cui italiano
    non entra nel riquadro; quel `salta` resta nel piano, quindi il rendering
    successivo non lo tenta nemmeno e chiude senza un evento. Il lavoro del
    modello e' li', pagato e corretto, e non arriva alla pagina.

    Nessuno degli altri segnali lo vede, `inglese_residuo` compreso: l'audit
    misura le righe scoperte *dentro un'area riscritta*, e un blocco tornato a
    `salta` non e' piu' un'area riscritta. Misurato su carnage-5 il 9 settembre
    2026: 53 blocchi persi cosi', mentre il gate ne dichiarava 4.

    Non si corregge da soli: allargare il riquadro vuol dire decidere quanta
    tavola coprire, e quello lo sa solo chi la guarda.
    """
    persi = []
    for page in plan.get("pagine", []) or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocchi", []) or []:
            if not isinstance(block, dict):
                continue
            render = block.get("_render") or {}
            if not render.get("testo_illeggibile"):
                continue
            testo = (block.get("testo") or "").strip()
            if not testo:
                continue
            persi.append({
                "pagina": page.get("numero"),
                "id": block.get("id"),
                "testo": testo[:80],
            })
    return persi


# --------------------------------------------------------------------------- #
# Inglese residuo sulla pagina resa: una riga scoperta che e' testo vero
# --------------------------------------------------------------------------- #

# L'audit raster (`structure_audit.righe_scoperte`) trova le righe originali
# rimaste leggibili dentro un'area riscritta e che nessun blocco rivendica: e'
# inglese che nessuna maschera cancella. Ma "leggibile" li' e' solo un conteggio
# di lettere, e ci finiscono anche i suoni disegnati ("KR-CU") e la spazzatura
# OCR ("mvt"). Questa prova separa il testo vero - da recuperare - dal resto.
#
# `_sembra_parlato` (Gap D) chiede due vocali perche' misura un balloon intero
# fuso dall'OCR; una riga scoperta e' spesso una sola parola corta - "WHAT?!"
# ha una vocale sola ma e' inglese vero. Serve una prova piu' larga, ancora
# ancorata alle vocali: almeno un TOKEN alfabetico di 3+ lettere con una vocale.
# Cosi' "WHAT", "SPIDER", "MORE" passano; "KR", "CU", "HM", "mvt", "Hnk", i
# grawlix e i numeri no. E' volutamente generosa: un falso positivo lo scarta
# l'operatore guardando la tavola, un falso negativo spedisce inglese in chiaro.


def _token_con_vocale(testo):
    for token in re.findall(r"[^\W\d_]+", testo or "", re.UNICODE):
        if len(token) >= 3 and any(ch.casefold() in VOCALI for ch in token):
            return token
    return None


def sembra_testo_reale(testo):
    """Se la riga contiene una parola vera: inglese da recuperare, non un suono."""
    return _token_con_vocale(testo) is not None


def inglese_residuo(righe_scoperte, numero_pagina=None):
    """Fra le righe scoperte dell'audit, quelle che sono testo vero in inglese.

    `righe_scoperte` e' la lista prodotta da `structure_audit` per una pagina:
    voci con `riga`, `testo`, `coperta`, `leggibile`. Si tengono solo le
    leggibili che hanno la forma di una parola vera. Come per gli altri Gap non
    si corregge niente: si dichiara la riga, la pagina e il testo, e la
    decisione (recuperare la rilettura, coprire, o confermare che e' arte)
    resta all'operatore.
    """
    residui = []
    for voce in righe_scoperte or []:
        if not isinstance(voce, dict) or not voce.get("leggibile"):
            continue
        testo = (voce.get("testo") or "").strip()
        if not sembra_testo_reale(testo):
            continue
        residui.append({
            "pagina": numero_pagina,
            "riga": voce.get("riga"),
            "coperta": voce.get("coperta"),
            "testo": testo[:80],
        })
    return residui


# --------------------------------------------------------------------------- #
# Misura sui piani gia' salvati
# --------------------------------------------------------------------------- #

def contiene_traduzioni(plan):
    """Se nessun blocco e' diverso dall'originale il piano non e' stato tradotto.

    Vale per il piano dell'OCR e per le proiezioni mandate al modello: li'
    `testo` ripete l'originale per costruzione, e contarle come copie
    significherebbe segnalare l'intero albo. La misura sui file gia' salvati
    deve saperlo; in pipeline il contesto e' noto al chiamante.
    """
    for page in plan.get("pagine", []) or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocchi", []) or []:
            if not isinstance(block, dict):
                continue
            testo = _normalizza(block.get("testo"))
            if testo and testo != _normalizza(block.get("originale")):
                return True
    return False


def misura(percorsi):
    """Conta scarti e copie sui piani indicati, senza toccarli.

    Serve a tarare le soglie qui sopra su prove gia' fatte invece che su una
    tavola sola: nessuna cattura, nessuna chiamata AI, nessun file riscritto.
    """
    esito = {"piani": [], "scartati": 0, "scartati_sospetti": 0, "copie": 0,
             "protetti": 0, "non_tradotti": 0}
    for percorso in percorsi:
        try:
            plan = json.loads(Path(percorso).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            esito["piani"].append({"file": str(percorso), "errore": str(exc)})
            continue
        if not isinstance(plan, dict):
            esito["piani"].append({"file": str(percorso), "errore": "non e' un piano"})
            continue
        scartati = sum(
            1
            for page in plan.get("pagine", []) or []
            if isinstance(page, dict)
            for regione in (page.get("sorgente") or {}).get("regioni", []) or []
            if isinstance(regione, dict) and regione.get("motivo") == "background_noise"
        )
        sospetti = testo_scartato_come_disegno(plan)
        protetti = dialogo_protetto_non_tradotto(plan)
        tradotto = contiene_traduzioni(plan)
        copie = traduzione_uguale_all_originale(plan) if tradotto else []
        esito["piani"].append({
            "file": str(percorso),
            "scartati": scartati,
            "sospetti": sospetti,
            "protetti": protetti,
            "tradotto": tradotto,
            "copie": copie,
        })
        esito["scartati"] += scartati
        esito["scartati_sospetti"] += len(sospetti)
        esito["protetti"] += len(protetti)
        esito["copie"] += len(copie)
        esito["non_tradotti"] += 0 if tradotto else 1
    return esito


def _stampa(esito, dettaglio):
    for piano in esito["piani"]:
        if "errore" in piano:
            print("!! %s: %s" % (piano["file"], piano["errore"]))
            continue
        if not (piano["scartati"] or piano["copie"] or piano["protetti"]):
            continue
        print("%s: %d scartati come disegno, %d con forma di testo, %s"
              % (piano["file"], piano["scartati"], len(piano["sospetti"]),
                 "%d copie dell'originale" % len(piano["copie"]) if piano["tradotto"]
                 else "copie non misurabili (piano non tradotto)"))
        if not dettaglio:
            continue
        for voce in piano["sospetti"]:
            print("   p%-3s %-15s righe=%-2d rumore=%-6s %s"
                  % (voce["pagina"], voce["forma"], voce["righe"],
                     round(voce["rumore_sfondo"] or 0, 1), voce["testo"]))
        for voce in piano["protetti"]:
            print("   p%-3s protetto e mai tradotto: %s" % (voce["pagina"], voce["testo"]))
        for voce in piano["copie"]:
            print("   p%-3s copia %d parole: %s"
                  % (voce["pagina"], voce["parole"], voce["testo"]))
    print("\nTotale: %d gruppi scartati, di cui %d con forma di testo; %d blocchi "
          "protetti e mai tradotti; %d traduzioni uguali all'originale "
          "(%d piani non tradotti, esclusi dal conto delle copie)"
          % (esito["scartati"], esito["scartati_sospetti"], esito["protetti"],
             esito["copie"], esito["non_tradotti"]))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Misura su piani gia' salvati quanto testo viene scartato come "
                    "disegno e quante traduzioni ricopiano l'originale.")
    ap.add_argument("piani", nargs="*", help="file piano*.json da misurare")
    ap.add_argument("--radice", default=None,
                    help="cerca ricorsivamente i piani sotto questa cartella")
    ap.add_argument("--dettaglio", action="store_true", help="elenca anche le voci")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="scrive la misura completa in questo file")
    args = ap.parse_args(argv)

    percorsi = [Path(p) for p in args.piani]
    if args.radice:
        percorsi += sorted(
            p for p in Path(args.radice).rglob("piano*.json") if ".git" not in p.parts
        )
    if not percorsi:
        ap.error("indicare almeno un piano oppure --radice")

    esito = misura(percorsi)
    _stampa(esito, args.dettaglio)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(esito, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Misura scritta -> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
