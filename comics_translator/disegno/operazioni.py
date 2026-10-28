"""Le operazioni con cui si chiude a mano un albo, dentro il motore.

Sui 38 albi `FATTO` le chat che consegnavano il PDF hanno scritto ogni volta
uno script `.finale` con le stesse sette mosse (`qa/finale_helpers.py` nel
reader): sbloccare un blocco tradotto, ridargli un riempimento, allargarlo
sulle righe orfane, coprire un fantasma, riquadrare una didascalia su una banda
piena, creare un blocco su righe mai diventate blocco, inventare una riga per
il testo che l'OCR non ha visto. Misura del 28 ottobre 2026: 1.939 interventi
su 612 pagine, circa 8.600 comandi delle chat.

Qui le stesse mosse diventano funzioni su un piano, senza stato di modulo, cosi'
che le usino lo sblocco con banda, la divisione per balloon e il ciclo di
riparazione. Ogni operazione lascia nel blocco CHI l'ha decisa e perche'
(`sbloccato_auto`), che e' la sola parola che il ricalcolo della protezione
rispetta: senza, il primo `enforce_page` rimette `salta` e l'inglese torna
(LEZIONI, "lo sblocco manuale senza marcatore torna salta").
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from comics_translator.sorgente import page_structure
from comics_translator.sorgente import source_inventory

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

BIANCO = (255.0, 255.0, 255.0)
SCALA_STANDARD = 0.18
# px di balloon attorno alle righe quando si cerca la forma: oltre, un balloon
# aperto sulla cornice della vignetta allaga la pagina (superman-1).
INTORNO = 320
# Margine attorno a una riga coperta da un rettangolo: abbastanza per le code
# delle lettere, poco per non mangiare il bordo del balloon.
PAD_RIGA = 8


def rettangoli(boxes, pad=PAD_RIGA):
    """Un profilo rettangolare per ogni riquadro, allargato di `pad`."""
    return [[[x0 - pad, y0 - pad], [x1 + pad, y0 - pad],
             [x1 + pad, y1 + pad], [x0 - pad, y1 + pad]]
            for x0, y0, x1, y1 in boxes]


def unione(boxes):
    boxes = [b for b in boxes if b and len(b) == 4]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def luminanza(colore) -> float:
    r, g, b = [float(c) for c in colore[:3]]
    return 0.299 * r + 0.587 * g + 0.114 * b


class Officina:
    """Il piano di un albo, i suoi indici e le immagini della sorgente.

    `sorgente` e' il PDF di partenza (unito.pdf) o None: senza immagini le
    operazioni che campionano colori o cercano la forma del balloon ripiegano
    sul bianco e sui rettangoli delle righe.
    """

    def __init__(self, plan, sorgente=None):
        self.plan = plan
        self.sorgente = sorgente
        self._doc = None
        self._immagini = {}
        self.toccati = []
        self.reindicizza()

    # ------------------------------------------------------------------ indici
    def reindicizza(self):
        self.blocchi, self.pagina_di, self.pagine = {}, {}, {}
        for page in self.plan.get("pagine", []):
            self.pagine[page["numero"]] = page
            for block in page.get("blocchi", []):
                if isinstance(block, dict):
                    self.blocchi[block["id"]] = block
                    self.pagina_di[block["id"]] = page

    def righe(self, numero):
        return {r["id"]: r for r in self.pagine[numero]["sorgente"]["righe"]}

    def scala(self, numero) -> float:
        """La scala pt/px della pagina, letta dai suoi blocchi (0.18 di norma,
        diversa sulle doppie: 0.2215 su imperial-guardians p2)."""
        voti = Counter(float(b["_render"]["scale"])
                       for b in self.pagine[numero].get("blocchi", [])
                       if isinstance(b, dict) and (b.get("_render") or {}).get("scale"))
        if voti:
            return voti.most_common(1)[0][0]
        righe = self.pagine[numero]["sorgente"].get("righe") or []
        for r in righe:
            if r.get("scala_px_pt"):
                return float(r["scala_px_pt"])
        return SCALA_STANDARD

    def immagine(self, numero):
        """La pagina sorgente in pixel del piano (quelli di `box_px`)."""
        if self.sorgente is None:
            return None
        scala = self.scala(numero)
        chiave = (numero, round(scala, 6))
        if chiave not in self._immagini:
            if self._doc is None:
                self._doc = fitz.open(self.sorgente)
            page = self.pagine[numero]
            pno = int(page.get("indice", numero - 1))
            pix = self._doc[pno].get_pixmap(matrix=fitz.Matrix(1 / scala, 1 / scala))
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)[:, :, :3]
            self._immagini[chiave] = np.ascontiguousarray(arr)
        return self._immagini[chiave]

    def chiudi(self):
        if self._doc is not None:
            self._doc.close()
            self._doc = None
        self._immagini.clear()

    # --------------------------------------------------------------- colori
    def colore_mediano(self, numero, box):
        """Il colore che domina un riquadro: su una didascalia stampata sull'arte
        il testo e' minoritario, e il mediano restituisce il fondo."""
        img = self.immagine(numero)
        if img is None:
            return BIANCO
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return BIANCO
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        return tuple(float(v) for v in np.median(patch, axis=0))

    def colore_della_banda(self, numero, boxes, pad=PAD_RIGA):
        """Il fondo attorno alle righe: la cornice di `pad` px fuori dal
        rettangolo di ogni riga, dove c'e' sfondo e non lettere."""
        img = self.immagine(numero)
        if img is None:
            return BIANCO
        campioni = []
        for x0, y0, x1, y1 in boxes:
            a = [int(max(0, x0 - pad)), int(max(0, y0 - pad)),
                 int(min(img.shape[1], x1 + pad)), int(min(img.shape[0], y1 + pad))]
            if a[2] <= a[0] or a[3] <= a[1]:
                continue
            fetta = img[a[1]:a[3], a[0]:a[2]].astype(np.float32)
            maschera = np.ones(fetta.shape[:2], bool)
            ix0, iy0 = int(x0) - a[0], int(y0) - a[1]
            ix1, iy1 = int(x1) - a[0], int(y1) - a[1]
            maschera[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
            if maschera.any():
                campioni.append(fetta[maschera])
        if not campioni:
            return self.colore_mediano(numero, unione(boxes))
        return tuple(float(v) for v in np.median(np.concatenate(campioni), axis=0))

    def testo_chiaro(self, numero, boxes, fondo) -> bool:
        """Il lettering originale era piu' chiaro del suo fondo?

        Si guardano i pixel delle righe che si staccano dal fondo: sono le
        lettere. Le didascalie bianche stampate sull'arte di woman-5 p9 hanno un
        fondo medio ("PRIMA." su grigio), e decidere dal solo fondo le scriveva
        in nero. Senza immagine si torna alla regola del fondo scuro.
        """
        img = self.immagine(numero)
        lum_fondo = luminanza(fondo)
        if img is not None:
            lettere = []
            for x0, y0, x1, y1 in boxes:
                fetta = img[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)].reshape(-1, 3)
                if fetta.size == 0:
                    continue
                lum = fetta.astype(np.float32) @ np.asarray([0.299, 0.587, 0.114], np.float32)
                lettere.append(lum[np.abs(lum - lum_fondo) > 60])
            if lettere:
                tutte = np.concatenate(lettere)
                if tutte.size >= 20:
                    return float(np.median(tutte)) > lum_fondo
        return lum_fondo < 110

    # --------------------------------------------------------------- misure
    def adatta_corpo(self, block, base_hint=None):
        """`base_size` tarato sul testo italiano dentro il riquadro."""
        from comics_translator.disegno import pdf_translate_overlay as pto

        render = block["_render"]
        testo = block.get("testo") or ""
        if render.get("upper", True):
            testo = testo.upper()
        if base_hint is None:
            base_hint = max(float(render.get("base_size") or 20.0), 20.0)
        scala = float(render["scale"])
        rect = fitz.Rect(*[v * scala for v in render["box_px"]])
        inner = fitz.Rect(rect.x0 + 1.5, rect.y0 + 1.5, rect.x1 - 1.5, rect.y1 - 1.5)
        if inner.is_empty:
            return None
        font = pto.load_fontset("comic").pick(True, False)
        size, righe = pto.fit(font, testo, inner, base_hint, 4.5, 1.0)
        render["base_size"] = round(size * 1.9, 3)
        return size, righe

    # ----------------------------------------------------------- operazioni
    def _marca(self, block, chi, motivo):
        render = block["_render"]
        render["sbloccato_auto"] = "%s: %s" % (chi, motivo)
        self.toccati.append({"id": block["id"], "operazione": chi, "motivo": motivo})

    def sveglia(self, block_id, chi, motivo, testo=None):
        """Disegna un blocco saltato con la sua traduzione.

        Se non ha un riempimento (era saltato dall'origine: fill_poly vuoto) gli
        si danno i rettangoli delle sue righe, o l'inglese resta come fantasma
        sotto l'italiano (superman-3, sezione G).
        """
        block = self.blocchi[block_id]
        if testo is not None:
            block["testo"] = testo
        render = block["_render"]
        block["salta"] = False
        for chiave in ("source_conflict", "skip_localizzazione", "salta_deciso",
                       "testo_illeggibile", "page_protected", "lettering_dichiarato"):
            render.pop(chiave, None)
        if render.pop("compound", None):
            render["compound_per_lobi"] = True
        if not [p for p in render.get("fill_poly") or [] if len(p) >= 3]:
            boxes = self.box_righe(block)
            if boxes:
                render["fill_poly"] = rettangoli(boxes)
                render["box_px"] = unione([render["box_px"]] + boxes)
        self._marca(block, chi, motivo)
        return block

    def box_righe(self, block):
        page = self.pagina_di[block["id"]]
        righe = self.righe(page["numero"])
        return [list(righe[i]["box_px"]) for i in (block.get("sorgente") or {}).get("righe", [])
                if i in righe and righe[i].get("box_px")]

    def banda(self, block_id, chi, motivo, testo=None):
        """Riempie con il colore del fondo SOLO i rettangoli delle righe proprie.

        E' il gesto dei `.finale` per le didascalie stampate sull'arte e per i
        balloon che la verifica dell'area rifiuta: un rettangolo per riga non
        tocca la grafica accanto come farebbe il riempimento del balloon, e
        l'inglese non resta. Si scrive nell'unione delle righe.
        """
        block = self.sveglia(block_id, chi, motivo, testo)
        page = self.pagina_di[block_id]
        boxes = self.box_righe(block)
        if not boxes:
            return block
        render = block["_render"]
        fondo = self.colore_della_banda(page["numero"], boxes)
        render["fill_poly"] = rettangoli(boxes)
        render["box_px"] = unione(boxes)
        render["bg"] = [round(v, 1) for v in fondo]
        render["light_text"] = self.testo_chiaro(page["numero"], boxes, fondo)
        render["banda"] = True
        self.adatta_corpo(block)
        return block

    def estendi_fill(self, block_id, boxes, chi, motivo, pad=PAD_RIGA, allarga_testo=True):
        """Copre righe orfane accanto a un blocco gia' tradotto e, se serve,
        allarga il riquadro di scrittura su di esse."""
        block = self.blocchi[block_id]
        render = block["_render"]
        render["fill_poly"] = (render.get("fill_poly") or []) + rettangoli(boxes, pad)
        if allarga_testo:
            render["box_px"] = unione([render["box_px"]] + [list(b) for b in boxes])
            self.adatta_corpo(block)
        self._marca(block, chi, motivo)
        return block

    def copri(self, block_id, boxes, chi, motivo, pad=PAD_RIGA):
        """Solo riempimento, ne' riquadro ne' testo: per i fantasmi inglesi
        rimasti nelle fessure fra blocchi gia' a posto (spidey-63-11 p23)."""
        return self.estendi_fill(block_id, boxes, chi, motivo, pad, allarga_testo=False)

    def nuova_riga(self, numero, seme, box_px, testo):
        """Riga sintetica per il testo che l'OCR non ha mai visto."""
        page = self.pagine[numero]
        inv = page["sorgente"]
        scala = self.scala(numero)
        rid = "p%d-l%s" % (numero, hashlib.sha256(seme.encode("utf-8")).hexdigest()[:18])
        box_px = [float(v) for v in box_px]
        box_pt = [round(v * scala, 6) for v in box_px]
        quad_px = [[box_px[0], box_px[1]], [box_px[2], box_px[1]],
                   [box_px[2], box_px[3]], [box_px[0], box_px[3]]]
        quad_pt = [[round(p[0] * scala, 6), round(p[1] * scala, 6)] for p in quad_px]
        inv["righe"].append({
            "id": rid, "tipo": "ocr", "testo_grezzo": testo, "testo_normalizzato": testo,
            "box_pt": box_pt, "box_px": box_px, "quad_px": quad_px, "quad_pt": quad_pt,
            "scala_px_pt": scala, "confidenza": 1.0, "direzione": None,
            "ordine": len(inv["righe"]), "stato": "acquisita", "motivo": None,
            "provenienza": {"tipo": "pagina"}, "motivo_ocr": None})
        loc = inv.get("localizzazione")
        if loc is not None:
            loc["aree"].append({"riga": rid, "box_pt": box_pt, "quad_pt": quad_pt,
                                "blocco": None, "stato": None, "motivo": None})
        return rid

    def _libera_righe(self, numero, ids):
        """Toglie le righe a regioni e blocchi che le possiedono: il blocco
        proprietario puo' solo essere saltato (non si ruba testo disegnato)."""
        page = self.pagine[numero]
        inv = page["sorgente"]
        ids = set(ids)
        for reg in list(inv.get("regioni", [])):
            comuni = ids & set(reg["righe"])
            if not comuni:
                continue
            resto = [r for r in reg["righe"] if r not in ids]
            reg["righe"] = resto
            if not resto:
                inv["regioni"].remove(reg)
                if reg["id"] in (inv.get("lettura") or {}).get("regioni", []):
                    inv["lettura"]["regioni"].remove(reg["id"])
        regioni = {r["id"] for r in inv.get("regioni", [])}
        for block in list(page.get("blocchi", [])):
            if not isinstance(block, dict):
                continue
            src = block.get("sorgente") or {}
            proprie = set(src.get("righe") or [])
            if proprie & ids:
                if not block.get("salta"):
                    raise ValueError("righe di un blocco disegnato: %s" % block["id"])
                src["righe"] = [r for r in src["righe"] if r not in ids]
            if src.get("regione") and src["regione"] not in regioni:
                # Svuotato del tutto: il blocco non ha piu' niente da difendere
                # ne' da scrivere, e un riferimento a una regione sparita non
                # passa la validazione (vedi claim_rows dei `.finale`).
                page["blocchi"].remove(block)
                self.blocchi.pop(block["id"], None)
                self.pagina_di.pop(block["id"], None)
                for zona in (inv.get("localizzazione") or {}).get("aree", []):
                    if zona.get("blocco") == block["id"]:
                        zona.update({"blocco": None, "stato": None, "motivo": None})
        for i, reg in enumerate(inv.get("regioni", [])):
            reg["ordine"] = i

    def nuovo_blocco(self, numero, riga_ids, testo, chi, motivo, *, sfondo="balloon",
                     bg=None, originale=None, upper=True, light_text=None, natura=None):
        """Blocco nuovo su righe (vere o sintetiche) che nessun blocco disegna.

        `sfondo="balloon"` cerca la forma del balloon attorno alle righe (con
        l'immagine annullata fuori da un intorno, superman-1); `"banda"`
        copre i soli rettangoli delle righe con il colore del fondo.
        """
        page = self.pagine[numero]
        inv = page["sorgente"]
        righe = self.righe(numero)
        riga_ids = [r for r in riga_ids if r in righe]
        if not riga_ids:
            raise ValueError("nessuna riga per il blocco nuovo")
        self._libera_righe(numero, riga_ids)
        scala = self.scala(numero)
        boxes = [list(righe[i]["box_px"]) for i in riga_ids]
        seme = "%s|%s" % (",".join(riga_ids), testo)
        rid = "p%d-r%s" % (numero, hashlib.sha256(seme.encode("utf-8")).hexdigest()[:18])
        box_px = unione(boxes)
        inv.setdefault("regioni", []).append({
            "id": rid, "tipo": "gruppo_ocr", "tipo_testo": "non_classificato",
            "righe": list(riga_ids), "box_pt": [round(v * scala, 6) for v in box_px],
            "box_px": box_px,
            "testo_normalizzato": " ".join(righe[i]["testo_normalizzato"] for i in riga_ids),
            "ordine": len(inv["regioni"]), "stato": "candidata", "motivo": None,
            "rumore_sfondo": 0.0})
        inv.setdefault("lettura", {"regioni": []}).setdefault("regioni", []).append(rid)
        img = self.immagine(numero)
        fill_poly = rettangoli(boxes)
        if sfondo == "balloon":
            fondo = tuple(bg) if bg is not None else BIANCO
            if img is not None:
                trovato = self._forma_del_balloon(img, box_px, fondo, boxes)
                if trovato is not None:
                    box_px, fill_poly = trovato
        else:
            fondo = tuple(bg) if bg is not None else self.colore_della_banda(numero, boxes)
        if light_text is None:
            light_text = self.testo_chiaro(numero, boxes, fondo)
        bid = "p%d-b%s" % (numero, rid.split("-r", 1)[1])
        block = {
            "id": bid, "pos": "x=0% y=0%",
            "originale": originale or " ".join(righe[i]["testo_grezzo"] for i in riga_ids),
            "automatica": originale or " ".join(righe[i]["testo_grezzo"] for i in riga_ids),
            "testo": testo, "salta": False, "unisci_con": None, "corpo": None,
            "allinea": "centro",
            "sorgente": {"regione": rid, "righe": list(riga_ids)},
            "_render": {
                "tipo": "immagine", "box_px": box_px, "fill_poly": fill_poly,
                "scale": scala, "bg": [round(v, 1) for v in fondo],
                "light_text": bool(light_text), "upper": upper, "base_size": 20.0,
                "compound": False, "area_ratio": 0.01, "banda": sfondo != "balloon",
            },
        }
        if natura:
            block["natura"] = natura
        page.setdefault("blocchi", []).append(block)
        self.blocchi[bid] = block
        self.pagina_di[bid] = page
        loc = inv.get("localizzazione")
        if loc is not None:
            for zona in loc["aree"]:
                if zona["riga"] in riga_ids:
                    zona.update({"blocco": bid, "stato": "occupata", "motivo": None})
        self.adatta_corpo(block)
        self._marca(block, chi, motivo)
        return block

    def _forma_del_balloon(self, img, seed_box, fondo, righe):
        from comics_translator.disegno import image_overlay

        x0 = max(0, int(seed_box[0]) - INTORNO)
        y0 = max(0, int(seed_box[1]) - INTORNO)
        x1 = min(img.shape[1], int(seed_box[2]) + INTORNO)
        y1 = min(img.shape[0], int(seed_box[3]) + INTORNO)
        chiusa = np.zeros_like(img)
        chiusa[y0:y1, x0:x1] = img[y0:y1, x0:x1]
        try:
            box_px, fill_poly = image_overlay.prepara_area(chiusa, list(seed_box), tuple(fondo),
                                                           righe=righe)
        except Exception:  # forma non trovata: restano i rettangoli delle righe
            return None
        if not fill_poly:
            return None
        return list(box_px), fill_poly

    # ------------------------------------------------------------------ fine
    def sigilla(self, numeri=None):
        """Sigilli e struttura delle pagine toccate: un piano con inventario non
        sigillato non passa la validazione del rendering."""
        for numero in numeri or list(self.pagine):
            page = self.pagine[numero]
            page["sorgente"] = source_inventory.seal(page["sorgente"])
            if page.get("struttura") is not None:
                page["struttura"] = page_structure.build_structure(page)
        return self.plan
