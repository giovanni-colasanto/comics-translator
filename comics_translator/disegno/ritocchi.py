"""Due ritocchi dei `.finale` fatti prima del rendering.

Prova v3 del 29-30 ottobre 2026 (imperial-guardians-1, 12-doom-owud-2,
captain-america-2025-7): l'OCR dava zero o quasi righe inglesi, ma il controllo
a vista trovava difetti su 17 pagine di un albo. Oltre ai lobi
(`dividi_balloon`) i gesti ricorrenti erano due.

1. Righe doppie dell'OCR. Il recupero rilegge una zona e lascia la seconda
   lettura fuori dai blocchi (`recovery_alternative`, `recovery_new_candidate`,
   `empty_text`...). Quando il blocco e' su banda o la forma del balloon e'
   stretta, la seconda lettura sporge: restano mezze lettere sopra e sotto il
   testo italiano ("fantasmi barrati", «THEM UP», «HALA--?»). Si copre la
   riga doppia; il riquadro di scrittura resta quello.

2. Forma del balloon dubbia. Il riempimento preso dalla forma puo' essere un
   rettangolo molto piu' grande delle righe (taglia il bordo del balloon) o
   inglobare le righe di altri blocchi (unisce due balloon). Li' i `.finale`
   passavano alla banda: i soli rettangoli delle righe proprie.

Le soglie vengono dai blocchi corretti a mano in quei tre albi; ogni ritocco
scrive il motivo nel piano (`sbloccato_auto`) e nel registro dell'Officina.
"""

from __future__ import annotations

from comics_translator.disegno import sblocco_automatico as sblocco
from comics_translator.sorgente import text_localization as localization

NATURE = ("dialogo", "didascalia")

# --- righe doppie
RIGHE_DOPPIE = frozenset(localization.RIGHE_RILETTURA) | frozenset(
    localization.RIGHE_SENZA_INCHIOSTRO)
# Quanta parte della riga doppia deve cadere su una riga del blocco: sotto,
# e' un'altra battuta vicina e non una rilettura.
SOVRAPPOSIZIONE_MINIMA = 0.25
# Sopra questa quota la riga doppia e' gia' coperta dal riempimento: le righe
# lasciate cosi' nei tre albi stavano tutte sopra 0,6 e a vista erano pulite;
# quelle corrette a mano stavano sotto 0,35.
QUOTA_GIA_COPERTA = 0.5

# --- forma dubbia
# Area del riempimento rispetto al rettangolo delle righe proprie.
AREA_RETTANGOLO = 1.5
AREA_ALTRUI = 2.0
# Un balloon e' un ovale: un riempimento che occupa quasi tutto il suo
# rettangolo circoscritto e' un rettangolo (le didascalie lo sono davvero, per
# questo la prova vale solo per il dialogo). Sopra 1,02 sono piu' profili
# sovrapposti (lobi), che hanno la loro logica.
RETTANGOLARE = (0.95, 1.02)
# Righe di altri blocchi coperte per oltre meta' dal riempimento.
RIGHE_ALTRUI = 2
# La banda scrive nell'unione delle righe proprie: se queste sono una piccola
# parte del riquadro di scrittura (la riparazione ha allargato il blocco su
# righe orfane) la banda lo restringerebbe a una riga e l'italiano non entra
# piu'. imperial-guardians-1 p11: 0,06, e il balloon tornava in inglese; i
# casi corretti a mano stavano fra 0,29 e 0,61.
RIGHE_NEL_RIQUADRO = 0.25


def _righe_libere(page):
    """Le righe che nessun blocco possiede e nessuna zona assegna."""
    inventario = page.get("sorgente") or {}
    possedute = {i for b in page.get("blocchi", []) if isinstance(b, dict)
                 for i in (b.get("sorgente") or {}).get("righe") or []}
    assegnate = {z["riga"] for z in (inventario.get("localizzazione") or {}).get("aree", [])
                 if z.get("blocco")}
    return [r for r in inventario.get("righe", [])
            if r["id"] not in possedute and r["id"] not in assegnate
            and r.get("motivo") in RIGHE_DOPPIE and r.get("box_px")]


def _sovrapposizione(riga, altra) -> float:
    """Frazione dell'area di `riga` che cade su `altra`."""
    larghezza = min(riga[2], altra[2]) - max(riga[0], altra[0])
    altezza = min(riga[3], altra[3]) - max(riga[1], altra[1])
    if larghezza <= 0 or altezza <= 0:
        return 0.0
    area = max(1.0, (riga[2] - riga[0]) * (riga[3] - riga[1]))
    return larghezza * altezza / area


def _tocca_altri(page, box, proprietario, disegnati):
    """Un rettangolo sopra `box` toccherebbe righe di blocchi lasciati in originale?"""
    from comics_translator.disegno.operazioni import rettangoli

    inventario = page.get("sorgente") or {}
    zone = {z["riga"]: z.get("blocco")
            for z in (inventario.get("localizzazione") or {}).get("aree", [])}
    righe_blocco = {}
    for b in page.get("blocchi", []):
        if isinstance(b, dict):
            for i in (b.get("sorgente") or {}).get("righe") or []:
                righe_blocco[i] = b["id"]
    banda = {"fill_poly": rettangoli([box])}
    for riga in inventario.get("righe", []):
        chi = righe_blocco.get(riga["id"]) or zone.get(riga["id"])
        if chi is None or chi == proprietario or chi in disegnati:
            continue
        if riga.get("box_px") and sblocco.quota_coperta(riga["box_px"], banda) \
                >= sblocco.QUOTA_RIGA_TOCCATA:
            return chi
    return None


def copri_righe_doppie(officina) -> list[dict]:
    """Copre le seconde letture dell'OCR che sporgono dai blocchi disegnati."""
    fatti = []
    for numero, page in list(officina.pagine.items()):
        libere = _righe_libere(page)
        if not libere:
            continue
        disegnati = {b["id"] for b in page.get("blocchi", [])
                     if isinstance(b, dict) and not b.get("salta")}
        prese = set()
        for block in [b for b in page.get("blocchi", []) if isinstance(b, dict)]:
            if block.get("salta") or block.get("natura") not in NATURE:
                continue
            render = block.get("_render") or {}
            if render.get("tipo") != "immagine":
                continue
            proprie = officina.box_righe(block)
            if not proprie:
                continue
            da_coprire = []
            for riga in libere:
                if riga["id"] in prese:
                    continue
                box = riga["box_px"]
                if max(_sovrapposizione(box, p) for p in proprie) < SOVRAPPOSIZIONE_MINIMA:
                    continue
                if sblocco.quota_coperta(box, render) >= QUOTA_GIA_COPERTA:
                    continue
                if _tocca_altri(page, box, block["id"], disegnati):
                    continue
                da_coprire.append(riga)
            if not da_coprire:
                continue
            prese.update(r["id"] for r in da_coprire)
            # Solo copertura: allargare il riquadro fa crescere il corpo e
            # il testo finisce sul blocco accanto (imperial-guardians-1 p18).
            officina.copri(block["id"], [r["box_px"] for r in da_coprire], "ritocchi",
                           "righe doppie dell'OCR: %s" % ", ".join(r["id"] for r in da_coprire))
            fatti.append({"pagina": numero, "id": block["id"],
                          "righe": [r["id"] for r in da_coprire]})
    return fatti


def _misure_forma(page, block, righe_proprie):
    import cv2
    import numpy as np

    render = block.get("_render") or {}
    profili = [p for p in render.get("fill_poly") or [] if len(p) >= 3]
    if not profili or not righe_proprie:
        return None
    x0 = min(b[0] for b in righe_proprie)
    y0 = min(b[1] for b in righe_proprie)
    x1 = max(b[2] for b in righe_proprie)
    y1 = max(b[3] for b in righe_proprie)
    area_righe = max(1.0, (x1 - x0 + 16) * (y1 - y0 + 16))
    area = sum(abs(cv2.contourArea(np.asarray(p, np.float32))) for p in profili)
    punti = np.concatenate([np.asarray(p, np.float32) for p in profili])
    _, _, larghezza, altezza = cv2.boundingRect(punti)
    rettangolarita = area / max(1.0, float(larghezza * altezza))
    righe = {r["id"]: r for r in (page.get("sorgente") or {}).get("righe", [])}
    altrui = 0
    for altro in page.get("blocchi", []):
        if not isinstance(altro, dict) or altro is block:
            continue
        for i in (altro.get("sorgente") or {}).get("righe") or []:
            if i in righe and righe[i].get("box_px") and \
                    sblocco.quota_coperta(righe[i]["box_px"], render) > 0.5:
                altrui += 1
    return area / area_righe, rettangolarita, altrui


def forma_dubbia(page, block, righe_proprie):
    """Il motivo per cui il riempimento del balloon non e' credibile, o None."""
    render = block.get("_render") or {}
    if block.get("salta") or block.get("natura") != "dialogo":
        return None
    if render.get("tipo") != "immagine" or render.get("banda"):
        return None
    if render.get("compound_per_lobi") or render.get("p3_multilobo"):
        return None
    misure = _misure_forma(page, block, righe_proprie)
    if misure is None:
        return None
    box = render.get("box_px")
    if box and len(box) == 4:
        x0 = min(b[0] for b in righe_proprie)
        y0 = min(b[1] for b in righe_proprie)
        x1 = max(b[2] for b in righe_proprie)
        y1 = max(b[3] for b in righe_proprie)
        riquadro = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        if (x1 - x0) * (y1 - y0) / riquadro < RIGHE_NEL_RIQUADRO:
            return None
    area, rettangolarita, altrui = misure
    if RETTANGOLARE[0] <= rettangolarita <= RETTANGOLARE[1] and area >= AREA_RETTANGOLO:
        return "riempimento squadrato (area %.1f volte le righe)" % area
    if altrui >= RIGHE_ALTRUI and area >= AREA_ALTRUI:
        return "il riempimento copre %d righe di altri blocchi" % altrui
    return None


def bande_per_forme_dubbie(officina) -> list[dict]:
    """Passa alla banda i dialoghi con un riempimento dalla forma sbagliata."""
    fatti = []
    for numero, page in list(officina.pagine.items()):
        disegnati = {b["id"] for b in page.get("blocchi", [])
                     if isinstance(b, dict) and not b.get("salta")}
        for block in [b for b in page.get("blocchi", []) if isinstance(b, dict)]:
            motivo = forma_dubbia(page, block, officina.box_righe(block))
            if not motivo:
                continue
            if sblocco._banda_tocca(page, block, disegnati):
                continue
            officina.banda(block["id"], "ritocchi", motivo)
            fatti.append({"pagina": numero, "id": block["id"], "motivo": motivo})
    return fatti
