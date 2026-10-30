"""Inglese rimasto sulla pagina resa, cercato con un OCR nuovo e non con il piano.

`inglese_residuo` (structure_audit) guarda solo le righe che l'analisi aveva
acquisito e che cadono dentro un'area riscritta. Sui 14 albi FATTO ha segnalato
27 delle 170 pagine che poi sono state corrette a mano per testo rimasto in
inglese: non vede i blocchi saltati per intero, i gruppi mai diventati blocco,
le traduzioni rifiutate dal renderer. Il suo denominatore e' l'OCR della
pipeline stessa, quindi eredita tutto quello che quell'OCR ha perso o scartato.

Qui il denominatore e' il PDF finito: lo si rilegge da capo e, riga per riga,
si guarda se e' inglese. La lingua si riconosce dalle parole funzionali (the,
you, what / il, che, non), che nel fumetto ci sono quasi sempre e che le
onomatopee non hanno: "BLAM" e "KRAKOOM" non contano ne' da una parte ne'
dall'altra. Un'insegna o un nome proprio in inglese contano poco per lo stesso
motivo.

Non corregge niente e non sostituisce l'audit raster: e' la misura che dice se
la pagina e' davvero in italiano, da confrontare con le esclusioni dichiarate
(una riga dentro un blocco saltato di proposito e' marcata `dichiarata`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from comics_translator.verifica import da_lasciare

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

DPI_OCR = 150
CONFIDENZA_MINIMA = 0.6

# Parole funzionali: frequenti, corte, quasi mai in un'onomatopea o un logo.
# Le parole che esistono anche in italiano stanno fuori da entrambe le liste,
# altrimenti una battuta italiana conta come inglese: "no", "a", "me", "come",
# "se", ma anche "in" ("LASCIAMI IN PACE"), "all" ("ALL'OMICIDIO!" spezzato
# sull'apostrofo), "i" (articolo), "so", "do", "oh", "okay", "hey". Misurato sui
# PDF di spiderman-13 e batman-2, dove erano la maggior parte dei falsi positivi.
INGLESE = frozenset("""
the you your yours and what whats that thats this these those is are was were
be been being am have has had does did dont doesnt didnt cant wont isnt arent
not with for from of to at on into out about just get got know think want
here there where when why who how which will would could should can may might
we our us they them their he him his she her it its im ive ill id my mine
any some something nothing everything anyone someone because but or if
than then too very really now never always again still only also back over
let lets gonna yeah yes
people today tonight tomorrow yesterday time thing things way day night going
doing done said say tell told make made take took see saw look need like good
right sorry please thank thanks ever every other more much many little big old
first last down off away after before maybe sure well guy world life
dead kill help wait every everyone nobody
""".split())

ITALIANO = frozenset("""
il lo la gli le un una uno di da del dello della dei degli delle al allo alla ai
agli alle nel nello nella nei negli nelle sul sulla sui con per tra fra che chi
cosa non sono sei siamo siete era erano essere ho hai ha abbiamo avete hanno
questo questa questi queste quello quella quelli quelle qui qua li cosi perche
ma pero anche ancora gia piu mai sempre tutto tutti tutta niente nulla io tu lui
lei noi voi loro mio mia tuo tua suo sua nostro vostro ci ti mi si vi ne dove
quando adesso ora dopo prima fare fatto detto va vai andiamo ecco bene grazie
""".split())

_PAROLA = re.compile(r"[^\W\d_]+", re.UNICODE)
_ACCENTI_IT = set("àèéìòù")


def _senza_accenti(parola):
    return (parola.replace("à", "a").replace("è", "e").replace("é", "e")
            .replace("ì", "i").replace("ò", "o").replace("ù", "u"))


TOKEN_FUSO = 10

# Dentro un token fuso ("WHATDOYOUTHINK") si cercano solo parole inglesi lunghe
# che non compaiono come pezzi di parole italiane: con le corte, "ABBANDONATI"
# conteneva "and" e "VIVERELAVITA" "ever", e tornavano inglesi (misurato sul PDF
# reso di spiderman-13). Servono almeno due parole cosi' nello stesso token.
_FUSE_EN = tuple(sorted("""
what that this your have know there where with from about they them their were
gonna really never always because something nothing people today going doing
think would could should dont cant youre thats
""".split(), key=len, reverse=True))
_FUSE_IT = tuple(sorted((p for p in ITALIANO if len(p) >= 4), key=len, reverse=True))
FUSE_MINIME = 2


def _conta_fuse(token, parole):
    """Quante parole funzionali stanno dentro un token senza spazi, senza sovrapporle."""
    usato = [False] * len(token)
    trovate = 0
    for parola in parole:
        inizio = token.find(parola)
        while inizio >= 0:
            fine = inizio + len(parola)
            if not any(usato[inizio:fine]):
                usato[inizio:fine] = [True] * len(parola)
                trovate += 1
            inizio = token.find(parola, inizio + 1)
    return trovate


def _parole(testo):
    parole = [p.casefold() for p in _PAROLA.findall(testo or "")]
    # L'OCR dei fumetti perde gli spazi: "WHATDOYOUWANT". Se c'e' wordninja li
    # rimette lo strumento della pipeline; altrimenti restano fusi e li conta
    # `_conta_fuse`.
    if not any(len(p) >= TOKEN_FUSO for p in parole):
        return parole
    try:
        from comics_translator.disegno.image_overlay import fix_spacing
    except Exception:  # pragma: no cover - dipendenza facoltativa
        return parole
    rifatte = []
    for p in _PAROLA.findall(testo or ""):
        if len(p) >= TOKEN_FUSO:
            rifatte.extend(x.casefold() for x in _PAROLA.findall(fix_spacing(p)))
        else:
            rifatte.append(p.casefold())
    return rifatte


def lingua_riga(testo):
    """'en', 'it' o None (non si puo' dire: suoni, nomi, numeri, una parola sola)."""
    grezze = [p.casefold() for p in _PAROLA.findall(testo or "")]
    if any(ch in _ACCENTI_IT for p in grezze for ch in p):
        return "it"
    parole = _parole(testo)
    if not parole:
        return None
    en = sum(p in INGLESE for p in parole)
    it = sum(_senza_accenti(p) in ITALIANO for p in parole)
    for fusa in (p for p in parole if len(p) >= TOKEN_FUSO):
        trovate = _conta_fuse(fusa, _FUSE_EN)
        if trovate >= FUSE_MINIME:
            en += trovate
        it += _conta_fuse(_senza_accenti(fusa), _FUSE_IT)
    if en > it and (en >= 2 or (en == 1 and len(parole) <= 3 and len(parole[0]) >= 2)):
        return "en"
    if it > en and it >= 1:
        return "it"
    return None


NATURE_LETTERING = ("grafica", "rumore")


def _norm(testo):
    return re.sub(r"[\W_]+", "", testo or "").casefold()


def saltato_di_proposito(block):
    """Un blocco lasciato in originale per scelta, non per un guasto.

    Conta la natura detta dal modello: un logo o un rumore restano in inglese
    per convenzione. Un dialogo o una didascalia saltati sono invece inglese
    rimasto sulla pagina, tradotti o no: prova A2 (spidey-63-9), 177 righe
    inglesi "dichiarate" contro 11 segnalate, quasi tutte battute in blocchi
    che la protezione o lo sblocco rifiutato tenevano saltati. Senza natura
    (piani vecchi) resta la regola di prima, ma solo se il testo e' ancora
    l'originale: una traduzione vera non disegnata non e' una scelta.
    """
    if not block.get("salta"):
        return False
    natura = block.get("natura")
    if natura:
        return natura in NATURE_LETTERING
    return _norm(block.get("testo")) in ("", _norm(block.get("originale")))


def _riquadri_saltati(pagina_piano):
    """I riquadri (pt) dei blocchi lasciati in originale di proposito."""
    riquadri = []
    for block in (pagina_piano or {}).get("blocchi", []) or []:
        if not isinstance(block, dict) or not saltato_di_proposito(block):
            continue
        render = block.get("_render") or {}
        if render.get("tipo") == "immagine" and render.get("box_px"):
            scala = float(render.get("scale", 1.0))
            riquadri.append([v * scala for v in render["box_px"]])
        elif render.get("rect"):
            riquadri.append(list(render["rect"]))
    return riquadri


# Sotto questa lunghezza (caratteri, senza spazi) un pezzo di riga combacia con
# l'originale per caso: "THE", "KAL", "NO" stanno in mezzo a qualunque cosa.
CONFRONTO_MINIMO = 7


def _testi_della_pagina(pagina_piano):
    """(inglese, italiano) della pagina, normalizzati: e' il metro di paragone.

    L'inglese della pagina lo sappiamo: e' `originale` di ogni blocco e il testo
    delle righe dell'inventario. Una riga riletta sul PDF finito che combacia con
    quello, e che NON compare nella traduzione, e' inglese rimasto: non serve
    riconoscere la lingua, basta confrontare. E' la misura che mancava - sulla
    bozza di superman-1 il lessico vedeva 7 righe su 23 pagine da correggere,
    perche' "YEARS AGO. NINE MILLION LIGHT-YEARS FROM EARTH." ha una sola parola
    funzionale.
    """
    inglese, italiano = [], []
    for block in (pagina_piano or {}).get("blocchi", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("natura") in NATURE_LETTERING:
            # Onomatopee, loghi e insegne restano in originale per convenzione:
            # il loro testo non e' un metro per dire "qui e' rimasto inglese"
            # (la "RRRMBLE" di superman-1 p11 risultava cosi' un residuo).
            italiano.append(_norm(block.get("originale")))
            continue
        inglese.append(_norm(block.get("originale")))
        italiano.append(_norm(block.get("testo")))
    for riga in ((pagina_piano or {}).get("sorgente") or {}).get("righe", []) or []:
        if isinstance(riga, dict):
            inglese.append(_norm(riga.get("testo_normalizzato") or riga.get("testo_grezzo")))
    return [t for t in inglese if len(t) >= CONFRONTO_MINIMO], [t for t in italiano if t]


def _come_l_originale(testo, inglese, italiano) -> bool:
    """La riga riletta combacia con l'originale della pagina e non con l'italiano.

    Il doppio controllo serve ai nomi propri: "KAL-EL" sta nell'originale ma
    anche nella traduzione, e non e' inglese rimasto. Conta solo cio' che la
    traduzione NON ha ripreso.
    """
    letto = _norm(testo)
    if len(letto) < CONFRONTO_MINIMO:
        return False
    if any(letto in t for t in italiano):
        return False
    return any(letto in t for t in inglese)


def righe_inglesi(img, dpi, pagina_piano=None, ocr=None, min_conf=CONFIDENZA_MINIMA):
    """Le righe OCR in inglese di un'immagine di pagina, con il riquadro in pt."""
    if ocr is None:
        from comics_translator.disegno.image_overlay import ocr_lines as ocr
    scala = 72.0 / dpi
    saltati = _riquadri_saltati(pagina_piano)
    inglese, italiano = _testi_della_pagina(pagina_piano)
    trovate = []
    for riga in ocr(img, min_conf):
        if (lingua_riga(riga.text) != "en"
                and not _come_l_originale(riga.text, inglese, italiano)):
            continue
        box = [riga.x0 * scala, riga.y0 * scala, riga.x1 * scala, riga.y1 * scala]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        dichiarata = any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in saltati)
        # Crediti, avvisi legali, marchi e onomatopee restano in inglese per
        # regola (`da_lasciare`): dichiarati, non residui da riparare.
        convenzione = da_lasciare.motivo(riga.text)
        voce = {"testo": riga.text[:120], "box_pt": [round(v, 1) for v in box],
                "confidenza": round(float(riga.conf), 3),
                "dichiarata": dichiarata or bool(convenzione)}
        if convenzione:
            voce["convenzione"] = convenzione
        trovate.append(voce)
    return trovate


def verifica_pdf(pdf, pagine=None, piano=None, dpi=DPI_OCR, ocr=None):
    """Rilegge il PDF e riporta, pagina per pagina, le righe ancora in inglese."""
    from comics_translator.disegno.image_overlay import page_to_array

    per_numero = {p.get("numero"): p for p in (piano or {}).get("pagine", [])}
    risultato = []
    with fitz.open(pdf) as doc:
        numeri = pagine or list(range(1, doc.page_count + 1))
        for numero in numeri:
            img, dpi_effettivo = page_to_array(doc[numero - 1], dpi)
            righe = righe_inglesi(img, dpi_effettivo, per_numero.get(numero), ocr=ocr)
            risultato.append({"pagina": numero, "righe": righe,
                              "non_dichiarate": sum(not r["dichiarata"] for r in righe)})
    return {
        "versione": 1,
        "pdf": str(pdf),
        "dpi": dpi,
        "pagine": risultato,
        "pagine_con_inglese": [p["pagina"] for p in risultato if p["non_dichiarate"]],
        "righe_non_dichiarate": sum(p["non_dichiarate"] for p in risultato),
    }


def _pagine(spec):
    if not spec:
        return None
    numeri = []
    for parte in spec.split(","):
        if "-" in parte:
            a, b = parte.split("-", 1)
            numeri.extend(range(int(a), int(b) + 1))
        else:
            numeri.append(int(parte))
    return numeri


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path, help="PDF tradotto da rileggere")
    ap.add_argument("--piano", type=Path, help="piano usato per il render (marca le esclusioni)")
    ap.add_argument("--pagine", help="es. 1-5,8 (default: tutte)")
    ap.add_argument("--dpi", type=int, default=DPI_OCR)
    ap.add_argument("--json", dest="json_out", type=Path, help="scrive il rapporto qui")
    args = ap.parse_args(argv)
    piano = json.loads(args.piano.read_text(encoding="utf-8")) if args.piano else None
    rapporto = verifica_pdf(args.pdf, _pagine(args.pagine), piano, args.dpi)
    for pagina in rapporto["pagine"]:
        for riga in pagina["righe"]:
            if not riga["dichiarata"]:
                # la console di Windows non sa scrivere ogni glifo che l'OCR legge
                testo = riga["testo"].encode(sys.stdout.encoding or "utf-8",
                                             "replace").decode(sys.stdout.encoding
                                                               or "utf-8", "replace")
                print("p%-3d %s" % (pagina["pagina"], testo))
    print("\n%d righe in inglese non dichiarate su %d pagine"
          % (rapporto["righe_non_dichiarate"], len(rapporto["pagine_con_inglese"])))
    if args.json_out:
        args.json_out.write_text(json.dumps(rapporto, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
