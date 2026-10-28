"""Un blocco, due balloon: dividerlo in un blocco per balloon.

L'OCR raggruppa le righe per vicinanza, e due balloon accostati dello stesso
parlante (o di due) diventano un blocco solo. Il renderer scrive allora una
battuta sola a cavallo dei due contorni, oppure il modello salta il blocco.
Nei `.finale` e' il gesto piu' ricorrente dopo lo sblocco: "OCR ha fuso DUE
balloon separati -> il blocco resta salta e si creano 2 EXTRA per balloon"
(spiderman-18 p8-p15, woman-2 p6/p19/p22, woman-4 p12/p22, us-18 p11, us-20 p7).

`lobi.py` divide gia' un balloon a piu' lobi uniti da una strozzatura. Qui il
caso e' diverso: i balloon sono DUE, separati dal bordo. Lo si riconosce
dall'immagine senza soglie sul testo: lo sfondo del balloon attorno alle
lettere e' una sola componente connessa; se le righe di un blocco poggiano su
componenti diverse, stanno in balloon diversi. Un balloon aperto sulla
cornice della vignetta fonde le componenti: si resta a un blocco, cioe' al
comportamento di prima.

La traduzione si divide contando le frasi, come per i lobi; se i conti non
tornano non si divide (lo fara' il ciclo di riparazione, che chiede al modello).
"""

from __future__ import annotations

import numpy as np

from comics_translator.disegno import lobi
from comics_translator.disegno.operazioni import unione

NATURE = ("dialogo", "didascalia")
# Distanza di colore (massimo sui canali) entro cui un pixel e' "fondo del balloon".
TOLLERANZA_FONDO = 45
# Margine attorno a una riga in cui cercare il fondo che la circonda.
ANELLO = 6
# Quanto del ritaglio puo' occupare una componente per essere un balloon e non
# la pagina bianca attorno (un balloon aperto sulla cornice).
QUOTA_MASSIMA = 0.6
MARGINE = 220


def _normalizza(testo):
    import re
    return re.sub(r"[\W_]+", "", testo or "").casefold()


def gruppi_per_balloon(img, righe_box, fondo):
    """Le righe raggruppate per componente di fondo; None se e' una sola."""
    import cv2

    if img is None or len(righe_box) < 2:
        return None
    zona = unione(righe_box)
    x0 = max(0, int(zona[0]) - MARGINE)
    y0 = max(0, int(zona[1]) - MARGINE)
    x1 = min(img.shape[1], int(zona[2]) + MARGINE)
    y1 = min(img.shape[0], int(zona[3]) + MARGINE)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = img[y0:y1, x0:x1].astype(np.int16)
    vicino = (np.abs(crop - np.asarray(fondo[:3], np.int16)).max(axis=2)
              <= TOLLERANZA_FONDO).astype(np.uint8)
    n, etichette, stats, _ = cv2.connectedComponentsWithStats(vicino, connectivity=4)
    area_crop = float(vicino.size)
    gruppi = {}
    for indice, box in enumerate(righe_box):
        bx0, by0, bx1, by1 = [int(round(v)) for v in box]
        bx0, bx1 = bx0 - x0, bx1 - x0
        by0, by1 = by0 - y0, by1 - y0
        a = [max(0, bx0 - ANELLO), max(0, by0 - ANELLO),
             min(vicino.shape[1], bx1 + ANELLO), min(vicino.shape[0], by1 + ANELLO)]
        if a[2] <= a[0] or a[3] <= a[1]:
            return None
        fetta = etichette[a[1]:a[3], a[0]:a[2]]
        valori = fetta[fetta > 0]
        if valori.size == 0:
            return None
        conti = np.bincount(valori)
        etichetta = int(conti.argmax())
        if stats[etichetta, cv2.CC_STAT_AREA] / area_crop > QUOTA_MASSIMA:
            return None  # fondo aperto: non e' un balloon chiuso
        # Il fondo di un balloon CIRCONDA le sue righe. Sull'arte (didascalie
        # stampate sul disegno) il colore simile e' a chiazze: una chiazza che
        # non abbraccia la riga non dice niente, e allora non si divide.
        cx, cy, cw, ch = (stats[etichetta, cv2.CC_STAT_LEFT], stats[etichetta, cv2.CC_STAT_TOP],
                          stats[etichetta, cv2.CC_STAT_WIDTH], stats[etichetta, cv2.CC_STAT_HEIGHT])
        if not (cx <= bx0 + 2 and cy <= by0 + 2 and cx + cw >= bx1 - 2 and cy + ch >= by1 - 2):
            return None
        gruppi.setdefault(etichetta, []).append(indice)
    if len(gruppi) == 1:
        # Un balloon solo: puo' avere due lobi uniti da una strozzatura senza
        # salto fra le righe (spiderman-18 p8, "I--" / "IF I SAY..."), che
        # lobi.py vede solo sui compound.
        unico = next(iter(gruppi))
        righe_crop = [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0] for b in righe_box]
        gruppi = per_strozzatura(etichette == unico, righe_crop)
        if not gruppi:
            return None
        return gruppi
    # ordine di lettura: dall'alto, poi da sinistra
    return sorted(gruppi.values(),
                  key=lambda g: (min(righe_box[i][1] for i in g), min(righe_box[i][0] for i in g)))


# Una rientranza del contorno conta come strozzatura se e' profonda almeno
# questa frazione del "diametro" del balloon (radice dell'area): un ovale non
# ne ha, la coda ne fa due ma il taglio fra quelle lascia tutte le righe da una
# parte sola, e allora non divide niente.
PROFONDITA_MINIMA = 0.07
PASSO = 4  # si lavora a un quarto della risoluzione: bastano e avanzano


def _si_accavallano(a, b) -> bool:
    larghezza = min(a[2], b[2]) - max(a[0], b[0])
    altezza = min(a[3], b[3]) - max(a[1], b[1])
    return larghezza > 8 and altezza > 8  # i riquadri OCR consecutivi si sfiorano


def per_strozzatura(componente, righe_box):
    """Gruppi di righe per lobo, tagliando la forma fra due rientranze profonde.

    La forma piena del balloon (lettere comprese) ha, fra due lobi, due
    rientranze del contorno una di fronte all'altra. Si prova il taglio fra
    ogni coppia di rientranze profonde, dalla piu' profonda: vale il primo che
    lascia righe da entrambe le parti, senza righe intrecciate fra i gruppi.
    """
    import cv2

    piccola = cv2.resize(componente.astype(np.uint8), None, fx=1 / PASSO, fy=1 / PASSO,
                         interpolation=cv2.INTER_NEAREST)
    contorni, _ = cv2.findContours(piccola, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contorni:
        return None
    contorno = max(contorni, key=cv2.contourArea)
    area = cv2.contourArea(contorno)
    if area < 50 or len(contorno) < 8:
        return None
    piena = np.zeros_like(piccola)
    cv2.drawContours(piena, [contorno], -1, 1, thickness=-1)
    scafo = cv2.convexHull(contorno, returnPoints=False)
    try:
        difetti = cv2.convexityDefects(contorno, scafo)
    except cv2.error:
        return None
    if difetti is None:
        return None
    soglia = PROFONDITA_MINIMA * area ** 0.5
    rientranze = sorted(((float(d[3]) / 256.0,
                          tuple(int(v) for v in contorno[int(d[2])][0]))
                         for d in difetti.reshape(-1, 4)
                         if float(d[3]) / 256.0 >= soglia), reverse=True)[:6]
    centri = [((b[0] + b[2]) / 2 / PASSO, (b[1] + b[3]) / 2 / PASSO) for b in righe_box]
    coppie = sorted(((pa + pb, a, b) for i, (pa, a) in enumerate(rientranze)
                     for (pb, b) in rientranze[i + 1:]), reverse=True)
    for _, a, b in coppie:
        tagliata = piena.copy()
        cv2.line(tagliata, a, b, 0, thickness=2)
        n, etichette = cv2.connectedComponents(tagliata, connectivity=4)
        if n < 3:
            continue
        ys, xs = np.nonzero(etichette)
        lab = etichette[ys, xs]
        gruppi = {}
        for indice, (cx, cy) in enumerate(centri):
            d = (xs - cx) ** 2 + (ys - cy) ** 2
            gruppi.setdefault(int(lab[int(d.argmin())]), []).append(indice)
        if len(gruppi) < 2:
            continue
        ingombri = [unione([righe_box[i] for i in g]) for g in gruppi.values()]
        if any(_si_accavallano(x, y) for k, x in enumerate(ingombri) for y in ingombri[k + 1:]):
            continue  # righe intrecciate: il taglio passa in mezzo a una battuta
        return sorted(gruppi.values(), key=lambda g: (
            min(righe_box[i][1] for i in g), min(righe_box[i][0] for i in g)))
    return None


def dividi_testo(testi_originali, traduzione):
    """La traduzione in tante parti quanti i gruppi, contando le frasi."""
    attese = [len(lobi.frasi(t, ocr=True)) for t in testi_originali]
    frasi = lobi.frasi(traduzione)
    if not all(attese) or sum(attese) != len(frasi):
        return None
    parti, i = [], 0
    for quante in attese:
        parti.append(" ".join(frasi[i:i + quante]).strip())
        i += quante
    return parti if all(parti) else None


def candidato(block):
    render = block.get("_render") or {}
    if render.get("tipo") != "immagine" or block.get("natura") not in NATURE:
        return False
    if render.get("compound") or render.get("compound_per_lobi") or render.get("p3_multilobo"):
        return False  # i lobi li divide gia' lobi.py
    testo = _normalizza(block.get("testo"))
    if not testo or testo == _normalizza(block.get("originale")):
        return False
    return len((block.get("sorgente") or {}).get("righe") or []) >= 2


def dividi_piano(officina) -> list[dict]:
    """Divide i blocchi a piu' balloon; restituisce cosa ha fatto."""
    fatti = []
    for numero, page in list(officina.pagine.items()):
        img = None
        for block in [b for b in page.get("blocchi", []) if isinstance(b, dict)]:
            if not candidato(block):
                continue
            if img is None:
                img = officina.immagine(numero)
                if img is None:
                    break
            righe = officina.righe(numero)
            ids = [i for i in block["sorgente"]["righe"] if i in righe]
            boxes = [righe[i]["box_px"] for i in ids]
            fondo = tuple((block["_render"].get("bg") or (255, 255, 255))[:3])
            gruppi = gruppi_per_balloon(img, boxes, fondo)
            if not gruppi:
                continue
            testi = [" ".join(righe[ids[i]]["testo_normalizzato"] for i in g) for g in gruppi]
            parti = dividi_testo(testi, block["testo"])
            if parti is None:
                block["_render"]["balloon_separati"] = len(gruppi)
                continue
            render = block["_render"]
            block["salta"] = True
            render["salta_deciso"] = True
            render.pop("sbloccato_auto", None)
            nuovi = []
            for g, parte, originale in zip(gruppi, parti, testi):
                nuovo = officina.nuovo_blocco(
                    numero, [ids[i] for i in g], parte, "divisione",
                    "balloon %d di %d di %s" % (len(nuovi) + 1, len(gruppi), block["id"]),
                    sfondo="balloon", bg=fondo, originale=originale,
                    upper=render.get("upper", True), light_text=render.get("light_text"),
                    natura=block.get("natura"))
                nuovi.append(nuovo["id"])
            if block["id"] in officina.blocchi:
                render["diviso_in"] = nuovi
            fatti.append({"pagina": numero, "id": block["id"], "balloon": len(gruppi),
                          "nuovi": nuovi})
    return fatti
