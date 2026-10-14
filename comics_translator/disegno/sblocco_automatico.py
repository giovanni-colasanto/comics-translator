"""Sblocco automatico: il lavoro che i `.finale` facevano a mano, con le stesse prove.

Sui 14 albi FATTO (363 pagine) il 61% degli interventi manuali era lo stesso
gesto: un blocco di dialogo saltato per `source_conflict` veniva guardato sulla
tavola, tradotto e sbloccato. La protezione non sbagliava a fermarlo - una riga
OCR vicina puo' essere un'onomatopea disegnata - ma non aveva modo di sapere
che cosa fosse quella riga. Chi guardava la pagina si': lo sapeva il modello
che traduceva, che pero' non veniva interpellato.

Ora il modello traduce anche i blocchi protetti e dice la loro `natura`. Qui
si decide se disegnarli, con due prove che devono valere INSIEME:

1. la natura: `dialogo` o `didascalia`, e un testo davvero tradotto (diverso
   dall'originale). La grafica e il rumore restano come sono;
2. la geometria, riga per riga: ogni riga vicina che la maschera coprirebbe
   deve essere (a) di un blocco che viene comunque riscritto, (b) una riga
   senza inchiostro o una rilettura della stessa zona, oppure (c) un residuo
   che sta per lo piu' DENTRO il riquadro del blocco. Basta una riga di un
   vicino che resta in originale, o una riga esterna che il riquadro sfiora,
   e il blocco resta saltato.

La prova (c) e' la stessa che ora limita `poligoni_sbloccati`: si copre solo
cio' che sta dentro, cosi' "HA HA!" e WHAM! vicini non diventano bianco.

Il calcolo e' il massimo punto fisso: si parte assumendo che tutti i candidati
vengano disegnati e si tolgono quelli con una riga non sicura, finche' non
cambia piu' niente. Due battute dello stesso balloon che si coprono a vicenda
si sbloccano insieme; una battuta accanto a una grafica protetta no.

Non e' una protezione allentata: e' la stessa decisione motivata di uno
`sblocca`, con il motivo scritto nel piano (`sbloccato_auto`) e ripetibile.
"""

from __future__ import annotations

import re

from comics_translator.disegno.plan import riga_dentro_il_blocco
from comics_translator.sorgente import text_localization as localization

NATURE_DA_DISEGNARE = ("dialogo", "didascalia")

# Protezioni che non riguardano una riga vicina ma il blocco stesso: un gruppo
# che copre piu' aree, un duplicato, una pagina intera conservata, una
# traduzione che non entra, una decisione presa da qualcuno. Qui non si toccano.
PROTEZIONI_NON_GEOMETRICHE = (
    "compound", "contenuto_in", "page_protected", "testo_illeggibile",
    "unreviewed", "salta_deciso", "sbloccato_a_mano",
)

# Righe che non difendono inchiostro da lettore: nessun testo riconosciuto, o
# un'altra lettura della stessa zona che il blocco sta riscrivendo.
RIGHE_SENZA_ARTE = frozenset(localization.RIGHE_SENZA_INCHIOSTRO) | {"duplicate"}


def _normalizza(testo):
    return re.sub(r"[\W_]+", "", testo or "").casefold()


def compound_divisibile(block, page) -> bool:
    """Un `compound` che si scrive lobo per lobo non e' piu' "un testo su piu' aree".

    Il `compound` nasce perche' una battuta sola centrata su due aree finiva a
    cavallo del contorno: nessun blocco a piu' profili si era mai reso bene.
    Se righe e traduzione si dividono fra le aree (`comics_translator.disegno.lobi`) quel
    motivo cade, e resta da verificare soltanto la geometria, come per tutti.
    Su imperial-guardians (prova A1) erano 5 compound sbloccati a mano.
    """
    from comics_translator.disegno import lobi

    render = block.get("_render") or {}
    if not render.get("compound") or page is None:
        return False
    altri = [(b.get("_render") or {}).get("box_px") for b in page.get("blocchi", [])
             if isinstance(b, dict) and b is not block and not b.get("salta")]
    return bool(lobi.prepara(render, lobi.righe_del_blocco(block, page),
                             block.get("testo") or "",
                             [a for a in altri if a and len(a) == 4]))


def candidato(block, page=None) -> bool:
    """Un blocco saltato solo per la sua area, con una traduzione vera e la
    natura di testo."""
    render = block.get("_render") or {}
    if not block.get("salta") or render.get("tipo") != "immagine":
        return False
    divisibile = compound_divisibile(block, page)
    if not (render.get("source_conflict") or render.get("skip_localizzazione")
            or render.get("scartato_come_disegno") or divisibile):
        return False
    protezioni = [k for k in PROTEZIONI_NON_GEOMETRICHE if render.get(k)]
    if divisibile:
        protezioni = [k for k in protezioni if k != "compound"]
    if protezioni:
        return False
    if block.get("natura") not in NATURE_DA_DISEGNARE:
        return False
    testo = _normalizza(block.get("testo"))
    return bool(testo) and testo != _normalizza(block.get("originale"))


# Un candidato del Gap C non ha la forma del balloon: si riempie con il
# RETTANGOLO delle proprie righe, del colore campionato. Su un fondo sporco quel
# rettangolo copre il disegno. `rumore_sfondo` e' la misura che lo ha scartato
# (dentro un balloon resta sotto 5, sul disegno supera 40): oltre questa soglia
# il rettangolo non si disegna da solo. Prova A2 (spidey-63-9): sopra 38 tutti
# i candidati sbloccati avevano coperto arte o fuso balloon vicini (il titolo
# di p2, una vignetta intera di p19 in nero); sotto 24 erano didascalie e
# balloon interi. Sui FATTO 45 dei 137 gruppi disegnati a mano stanno sopra la
# soglia: li' l'operatore misurava il riquadro, il programma non lo sa fare.
RUMORE_MAX_CANDIDATO = 30.0


def _motivo_fondo_non_sicuro(block, page):
    """None se il riempimento del blocco puo' poggiare sul suo fondo, altrimenti perche' no."""
    render = block.get("_render") or {}
    if not render.get("scartato_come_disegno") or render.get("fill_poly"):
        return None
    regione = (block.get("sorgente") or {}).get("regione")
    for r in ((page or {}).get("sorgente") or {}).get("regioni", []) or []:
        if r.get("id") == regione:
            rumore = r.get("rumore_sfondo")
            if rumore is None:
                return None
            if float(rumore) > RUMORE_MAX_CANDIDATO:
                return "fondo_disegnato:%.0f" % float(rumore)
            return None
    return None


# Sotto questa frazione di riga coperta dal riempimento, la riga non viene
# toccata: il rettangolo dell'OCR sfiora il blocco, la maschera no.
QUOTA_RIGA_TOCCATA = 0.02


def _profili_coperti(render):
    profili = [p for p in (render.get("fill_poly") or []) if len(p) >= 3]
    if profili:
        return profili
    box = render.get("box_px")
    if box and len(box) == 4:
        x0, y0, x1, y1 = box
        return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]
    return []


def quota_coperta(box_riga, render) -> float:
    """Che frazione del rettangolo della riga cade nell'area che il blocco riempie."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = [int(round(v)) for v in box_riga]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    # Righe enormi (un'onomatopea di 1700 px): basta misurare a risoluzione ridotta.
    passo = max(1, int(max(w, h) / 400))
    mask = np.zeros((h // passo + 1, w // passo + 1), np.uint8)
    for profilo in _profili_coperti(render):
        locale = np.round((np.asarray(profilo, np.float64) - [x0, y0]) / passo).astype(np.int32)
        cv2.fillPoly(mask, [locale], 1)
    return float(mask.mean())


def _motivo_riga_non_sicura(page, block, ident, disegnati):
    """None se la riga `ident` si puo' coprire sbloccando `block`, altrimenti perche' no."""
    inventario = page.get("sorgente") or {}
    righe = {r["id"]: r for r in inventario.get("righe", [])}
    zone = {z["riga"]: z for z in (inventario.get("localizzazione") or {}).get("aree", [])}
    riga = righe.get(ident)
    if riga is None:
        return "riga_sconosciuta"
    proprietario = (zone.get(ident) or {}).get("blocco")
    if proprietario is not None:
        # Di un altro blocco: si copre solo se anche quello viene riscritto,
        # altrimenti il riempimento gli cancellerebbe le lettere.
        return None if proprietario in disegnati else "vicino_in_originale:%s" % proprietario
    if riga.get("motivo") in RIGHE_SENZA_ARTE:
        return None
    render = block.get("_render") or {}
    box_riga = riga.get("box_px")
    box_blocco = render.get("box_px")
    if box_riga and box_blocco and riga_dentro_il_blocco(box_riga, box_blocco):
        return None  # residuo dello stesso balloon, dentro il riquadro
    if box_riga and len(box_riga) == 4:
        quota = quota_coperta(box_riga, render)
        # Il riempimento non la tocca: la sovrapposizione era dei rettangoli,
        # non dell'inchiostro. Prova A1: "SHRUKK" a p6, "EVER" a p12, "I'M TOLD"
        # a p21 fermavano balloon tradotti senza che la maschera li sfiorasse.
        if quota < QUOTA_RIGA_TOCCATA:
            return None
        # Una rilettura incerta (confidenza bassa, candidato del recupero) che
        # sta dentro l'area riempita e' inchiostro dello stesso balloon: e' la
        # regola che `enforce_page` applica gia' (`RIGHE_RILETTURA`).
        if riga.get("motivo") in localization.RIGHE_RILETTURA and quota >= 0.5:
            return None
    return "riga_esterna:%s" % ident


def sblocca_tradotti(plan) -> list[dict]:
    """Sblocca i blocchi che superano entrambe le prove; restituisce le decisioni.

    Modifica il piano sul posto e poi ricalcola le protezioni. Ogni voce dice
    blocco, pagina, esito (`sbloccato` o `resta_saltato`) e il motivo: anche un
    rifiuto va scritto, perche' e' il punto da cui ripartire guardando la tavola.
    """
    decisioni = []
    for page in plan.get("pagine", []):
        blocchi = [b for b in page.get("blocchi", []) if isinstance(b, dict)]
        candidati = {b["id"]: b for b in blocchi if candidato(b, page)}
        if not candidati:
            continue
        gia_disegnati = {b["id"] for b in blocchi if not b.get("salta")}
        rifiuti = {bid: motivo for bid, motivo in
                   ((bid, _motivo_fondo_non_sicuro(b, page)) for bid, b in candidati.items())
                   if motivo}
        attivi = set(candidati) - set(rifiuti)
        while True:
            disegnati = gia_disegnati | attivi
            tolti = {}
            for bid in attivi:
                block = candidati[bid]
                render = block.get("_render") or {}
                for ident in render.get("source_overlap") or []:
                    motivo = _motivo_riga_non_sicura(page, block, ident, disegnati)
                    if motivo:
                        tolti[bid] = motivo
                        break
            if not tolti:
                break
            attivi -= set(tolti)
            rifiuti.update(tolti)
        for bid, block in candidati.items():
            if bid in attivi:
                render = block["_render"]
                render["sbloccato_auto"] = (
                    "natura=%s; righe vicine verificate: %d"
                    % (block.get("natura"), len(render.get("source_overlap") or [])))
                if render.pop("compound", None):
                    # resta scritto perche' era saltato: le aree si scrivono per lobi
                    render["compound_per_lobi"] = True
                block["salta"] = False
                decisioni.append({"pagina": page.get("numero"), "id": bid,
                                  "esito": "sbloccato", "natura": block.get("natura")})
            else:
                decisioni.append({"pagina": page.get("numero"), "id": bid,
                                  "esito": "resta_saltato", "motivo": rifiuti.get(bid)})
    if decisioni:
        localization.enforce_localization(plan)
        # Controprova: `sbloccato_auto` azzera i conflitti del blocco, quindi il
        # ricalcolo non dovrebbe revocare nulla. Se succede e' un difetto di
        # questo modulo, ma far morire la run dopo aver pagato tutti i lotti
        # (come faceva il ValueError) butta via la traduzione intera: si chiude
        # di nuovo il blocco, lo si dichiara revocato e la run prosegue.
        per_id = {b["id"]: b for p in plan["pagine"] for b in p.get("blocchi", [])}
        for decisione in decisioni:
            block = per_id[decisione["id"]]
            if decisione["esito"] == "sbloccato" and block.get("salta"):
                render = block.get("_render") or {}
                render.pop("sbloccato_auto", None)
                decisione["esito"] = "resta_saltato"
                decisione["motivo"] = "revocato_dal_ricalcolo"
    return decisioni
