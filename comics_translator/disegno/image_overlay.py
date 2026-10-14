"""Ramo per i PDF fatti di immagini (scansioni, fumetti, screenshot).

Qui il testo e' pixel dentro l'immagine, non un oggetto del PDF: non c'e' niente
da estrarre e niente da redigere. La strategia e' quella di Google Lens, ma
scritta dentro un PDF vero:

  1. la pagina viene renderizzata e passata a RapidOCR, che restituisce le righe
     con le loro coordinate;
  2. le righe vicine vengono raggruppate (un balloon, un paragrafo) perche'
     tradurre riga per riga darebbe risultati pessimi;
  3. l'area del gruppo viene allargata finche' i pixel intorno restano di colore
     uniforme: dentro un balloon bianco si arriva cosi' fino al bordo nero;
  4. l'area viene coperta con il colore di sfondo campionato e il testo tradotto
     viene scritto sopra, centrato.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np

from comics_translator.sorgente import source_inventory as inventory_data
from comics_translator.sorgente import text_localization as localization
from comics_translator.verifica.segnalazioni import forma_di_testo

# px OCR di aria intorno alle righe di un candidato del Gap C: quanto basta a
# coprire le code delle lettere, non abbastanza da mangiare l'arte vicina.
MARGINE_CANDIDATO = 6

_TOKEN_LETTERE = re.compile(r"[^\W\d_]+", re.UNICODE)


def candidato_scartato(testo, righe):
    """Se un gruppo scartato come disegno va comunque mostrato al modello.

    Oltre alle forme del Gap C (`forma_di_testo`), anche le didascalie di una
    riga senza punto finale: "NINE MONTHS AGO" e "DEADREEFSALIXOUTWORLD" su
    imperial-guardians p5 non erano candidati e sono rimasti in inglese. Qui
    la soglia puo' essere larga, perche' un candidato non copre niente: resta
    saltato finche' il modello non dice che e' dialogo o didascalia. Restano
    fuori i suoni corti ("BLAM", "KRAKOOM") e le parole senza vocali.
    """
    if forma_di_testo(testo, righe):
        return True
    token = _TOKEN_LETTERE.findall(testo or "")
    lettere = sum(len(t) for t in token)
    parole = [t for t in token if len(t) >= 3 and any(c in "aeiouyAEIOUY" for c in t)]
    return lettere >= 12 and (len(parole) >= 2 or (parole and max(map(len, parole)) >= 16))

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            sys.exit(
                "Il PDF e' fatto di immagini e serve l'OCR: pip install rapidocr-onnxruntime"
            )
        # Nessuna riga gia' riconosciuta deve sparire nel filtro interno.
        _ENGINE = RapidOCR(text_score=0.0)
    return _ENGINE


@dataclass
class OcrLine:
    """Una riga riconosciuta, in coordinate pixel dell'immagine renderizzata."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    conf: float
    raw_text: str | None = None
    quad: list | None = None
    exclusion_reason: str | None = None
    provenance: dict | None = None

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


MAX_MEGAPIXEL = 48.0


def page_to_array(page: "fitz.Page", dpi: int) -> tuple:
    # tetto di sicurezza: su una pagina molto grande il dpi richiesto potrebbe
    # produrre un'immagine da parecchi GB
    wanted = (page.rect.width * dpi / 72.0) * (page.rect.height * dpi / 72.0) / 1e6
    if wanted > MAX_MEGAPIXEL:
        dpi = max(72, int(dpi * (MAX_MEGAPIXEL / wanted) ** 0.5))
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(img[:, :, :3]), dpi


def ocr_lines(img: np.ndarray, min_conf: float, *, include_excluded=False) -> list[OcrLine]:
    result, _ = _engine()(img)
    lines: list[OcrLine] = []
    for box, raw, conf in result or []:
        raw = raw or ""
        text = raw.strip()
        reason = ("empty_text" if not text else "low_confidence" if conf < min_conf
                  else "no_letters" if not any(c.isalpha() for c in text) else None)
        if reason and not include_excluded:
            continue
        quad = [[float(v) for v in point] for point in box]
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        lines.append(OcrLine(min(xs), min(ys), max(xs), max(ys), text, float(conf),
                             raw_text=raw, quad=quad, exclusion_reason=reason))
    return lines


# --------------------------------------------------------------------------- #
# Ricostruzione degli spazi
# --------------------------------------------------------------------------- #
# Il maiuscolo fitto dei fumetti fa perdere gli spazi all'OCR: "WOULDKILL".
# wordninja sa rimetterli, ma da solo distrugge apostrofi e punteggiatura
# ("ME,SHI'AR?" -> "me shi ' ar"), quindi lo si applica solo dove serve
# davvero e si accetta il risultato solo se e' fatto di parole vere.

_WORDS: dict | None = None
_TOKEN = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


def _vocab() -> dict:
    global _WORDS
    if _WORDS is None:
        try:
            import wordninja

            _WORDS = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
        except Exception:
            _WORDS = {}
    return _WORDS


def fix_spacing(text: str) -> str:
    """Rimette gli spazi dentro le parole incollate, senza toccare il resto."""
    vocab = _vocab()
    if not vocab:
        return text
    import wordninja

    out = []
    for token in _TOKEN.findall(text):
        low = token.lower()
        if not token.isalpha() or len(token) < 6 or low in vocab:
            out.append(token)
            continue
        parts = wordninja.split(low)
        # si accetta solo se ogni pezzo e' una parola vera di almeno due lettere
        if len(parts) > 1 and all(len(p) >= 2 and p in vocab for p in parts):
            rebuilt = " ".join(parts)
            out.append(rebuilt.upper() if token.isupper() else rebuilt)
        else:
            out.append(token)
    return "".join(out)


def _join_lines(texts: list[str]) -> str:
    """Unisce le righe di un gruppo, ricucendo la sillabazione dove ha senso."""
    vocab = _vocab()
    joined = ""
    for i, raw in enumerate(texts):
        cur = raw.strip()
        if i == 0:
            joined = cur
        elif joined.endswith("-"):
            # "STARK-" + "MADE": trattino vero o sillabazione? decide il dizionario
            tail = joined[:-1].split()[-1] if joined[:-1].split() else ""
            head = cur.split()[0] if cur.split() else ""
            if (tail + head).lower() in vocab:
                joined = joined[:-1] + cur
            else:
                joined = joined + " " + cur
        else:
            joined += " " + cur
    return " ".join(joined.split())


# --------------------------------------------------------------------------- #
# Raggruppamento in balloon / paragrafi
# --------------------------------------------------------------------------- #

@dataclass
class Group:
    lines: list[OcrLine] = field(default_factory=list)

    @property
    def x0(self) -> float:
        return min(line.x0 for line in self.lines)

    @property
    def y0(self) -> float:
        return min(line.y0 for line in self.lines)

    @property
    def x1(self) -> float:
        return max(line.x1 for line in self.lines)

    @property
    def y1(self) -> float:
        return max(line.y1 for line in self.lines)

    @property
    def avg_height(self) -> float:
        return sum(line.height for line in self.lines) / len(self.lines)

    @property
    def corpo_righe(self) -> float:
        """L'altezza tipica delle righe, non la media.

        Quando l'OCR fonde nel gruppo un logo o un titolo disegnato, la media
        si alza e con lei `base_size`; la soglia di leggibilita' e' relativa a
        `base_size`, quindi il testo vero viene giudicato troppo piccolo e il
        blocco resta in inglese (`render_block_illegible`). La mediana non si
        muove per una riga sola, e su un gruppo di righe omogenee coincide con
        la media: i balloon normali non cambiano corpo.
        """
        altezze = sorted(line.height for line in self.lines)
        meta = len(altezze) // 2
        if len(altezze) % 2:
            return altezze[meta]
        return (altezze[meta - 1] + altezze[meta]) / 2.0

    @property
    def text(self) -> str:
        return _join_lines([fix_spacing(line.text) for line in self.lines])

    @property
    def upper(self) -> bool:
        letters = [c for line in self.lines for c in line.text if c.isalpha()]
        if not letters:
            return False
        return sum(c.isupper() for c in letters) / len(letters) > 0.8


def _corridor_is_clear(img: np.ndarray, g: "Group", line: OcrLine, tol: float = 40.0) -> bool:
    """C'e' solo sfondo uniforme tra il gruppo e la riga candidata?

    E' il criterio che distingue due balloon vicini da due righe dello stesso
    balloon: tra balloon diversi passano il bordo nero e il disegno, e la
    striscia intermedia smette di essere uniforme.
    """
    top, bottom = int(g.y1), int(line.y0)
    if bottom - top <= 1:
        return True
    # solo la fascia dove le due si sovrappongono: allargarla all'unione farebbe
    # entrare pezzi di disegno fuori dal balloon e spezzerebbe gruppi buoni
    x0 = int(max(0, max(g.x0, line.x0)))
    x1 = int(min(img.shape[1], min(g.x1, line.x1)))
    if x1 - x0 < 2:
        x0 = int(max(0, min(g.x0, line.x0)))
        x1 = int(min(img.shape[1], max(g.x1, line.x1)))
    if x1 - x0 < 2:
        return False

    strip = img[top:bottom, x0:x1]
    if strip.size == 0:
        return True
    bg, _ = sample_background(img, (g.x0, g.y0, g.x1, g.y1), pad=2)
    diff = np.abs(strip.astype(np.float32) - np.array(bg, dtype=np.float32)).max(axis=2)
    return float((diff < tol).mean()) > 0.90


def cluster(lines: list[OcrLine], img: np.ndarray, gap_factor: float = 1.0) -> list[Group]:
    """Righe vicine, allineate e senza ostacoli in mezzo finiscono nello stesso gruppo."""
    groups: list[Group] = []
    for line in sorted(lines, key=lambda item: (item.y0, item.x0)):
        placed = False
        for g in groups:
            gap = line.y0 - g.y1
            if gap > gap_factor * max(g.avg_height, line.height):
                continue
            if gap < -0.5 * line.height:  # righe affiancate, non impilate
                continue
            overlap = min(g.x1, line.x1) - max(g.x0, line.x0)
            narrower = min(g.x1 - g.x0, line.x1 - line.x0)
            centers_close = abs(g.lines[-1].cx - line.cx) < 0.6 * max(narrower, 1)
            if not (overlap > 0.3 * narrower or centers_close):
                continue
            if not _corridor_is_clear(img, g, line):
                continue
            g.lines.append(line)
            placed = True
            break
        if not placed:
            groups.append(Group([line]))
    return groups


def background_noise(img: np.ndarray, box: tuple, light_text: bool) -> float:
    """Quanto e' 'sporco' lo sfondo su cui poggia il testo.

    Si isolano i pixel che non sono inchiostro e se ne misura la dispersione:
    dentro un balloon sono tutti dello stesso bianco e il valore resta sotto 5,
    sopra il disegno (onomatopee, insegne) supera 40. E' la distinzione che
    serve, e a differenza del conteggio dei pixel di sfondo non dipende da
    quanto testo ci sia dentro il rettangolo ne' da dove cade il bordo.
    """
    x0, y0, x1, y1 = [int(v) for v in box]
    patch = img[max(0, y0):y1, max(0, x0):x1]
    if patch.size == 0:
        return 999.0
    lum = patch.mean(axis=2)
    thr = np.median(lum)
    selected = patch[lum <= thr] if light_text else patch[lum >= thr]
    if selected.size == 0:
        return 999.0
    return float(selected.std(axis=0).mean())


# --------------------------------------------------------------------------- #
# Sfondo: campionamento ed espansione
# --------------------------------------------------------------------------- #

def sample_background(img: np.ndarray, box: tuple, pad: int = 3) -> tuple:
    """Colore di sfondo dell'area e se il testo era chiaro su scuro."""
    h, w = img.shape[:2]
    x0 = max(0, int(box[0]) - pad)
    y0 = max(0, int(box[1]) - pad)
    x1 = min(w, int(box[2]) + pad)
    y1 = min(h, int(box[3]) + pad)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return (255, 255, 255), False

    lum = patch.mean(axis=2)
    # lo sfondo e' la parte piu' estesa: se la mediana e' chiara il testo e' scuro
    dark_text = float(np.median(lum)) > 127.0
    pct = 85 if dark_text else 15
    bg = tuple(float(np.percentile(patch[:, :, c], pct)) for c in range(3))
    return bg, not dark_text


def _tappa_buchi(mask: np.ndarray) -> np.ndarray:
    """Chiude le zone non-sfondo che l'esterno non raggiunge: sono i glifi.

    Serve due volte. Alla fine, perche' le lettere devono far parte dell'area
    da cancellare, altrimenti restano scoperte e in lingua originale. E prima
    di sigillare, perche' su una maschera perforata dalle lettere l'erosione
    sbriciola il balloon invece di staccare il varco.
    """
    import cv2

    inv = (mask == 0).astype(np.uint8)
    padded = np.pad(inv, 1, constant_values=1)
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 2)
    fuori = padded[1:-1, 1:-1] == 2
    return ((mask > 0) | ((inv == 1) & ~fuori)).astype(np.uint8)


def _raggio_sigillo(box: list, righe=()) -> int:
    """Quanto stringere per staccare un varco senza smontare il balloon.

    Il varco per cui la maschera cola e' piu' sottile di una riga di testo: e'
    il punto in cui il contorno del balloon sfiora il bordo della vignetta, o
    l'antialiasing che li salda. Una riga di testo, invece, sta comodamente
    dentro qualunque balloon. Un quarto dell'altezza di riga stacca il primo
    senza intaccare il secondo.
    """
    altezze = [float(r[3]) - float(r[1]) for r in righe if float(r[3]) > float(r[1])]
    riferimento = float(np.median(altezze)) if altezze else float(box[3]) - float(box[1])
    return int(max(2, min(12, round(riferimento * 0.25))))


def _senza_colature(near: np.ndarray, etichette_sotto, raggio: int):
    """La componente del balloon, senza cio' che raggiunge solo per un varco.

    Si stringe la maschera di sfondo finche' i collegamenti sottili si
    spezzano, si sceglie la componente che regge ancora il testo e poi la si
    riallarga della stessa quantita' dentro la maschera di partenza: il bordo
    vero del balloon torna dov'era, il margine fra le vignette no, perche' per
    riattraversarlo servirebbe piu' del raggio con cui si e' stretto.

    Restituisce None quando il sigillo non e' applicabile - balloon piu' sottile
    del raggio, o testo che dopo la stretta non poggia piu' su nulla: in quei
    casi decide il chiamante, che torna al comportamento senza sigillo.
    """
    import cv2

    if raggio < 2:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raggio + 1,) * 2)
    stretta = cv2.erode(near, kernel)
    if not stretta.any():
        return None
    count, labels = cv2.connectedComponents(stretta, connectivity=4)
    if count < 2:
        return None
    etichette = etichette_sotto(labels, stretta)
    if not etichette:
        return None
    seme = np.isin(labels, list(etichette)).astype(np.uint8)
    # riallargare e intersecare: si recupera il bordo del balloon, non il varco
    return (cv2.dilate(seme, kernel) & near).astype(bool)


# Quanto puo' essere grande l'interno di un balloon rispetto al riquadro delle
# righe che contiene. Misurato sui profili accettati di tre albi (superman-1,
# spidey-63-8, batman-2): mediana 1,5-2,6, novantesimo percentile 2,7-4,1. Oltre
# questo rapporto la macchia non e' piu' un balloon: ha passato il contorno ed
# e' arrivata alla cornice della vignetta o al margine della tavola.
COLATURA_SU_RIGHE = 6.0

# Con quanti sigilli via via piu' larghi si prova a staccare la colatura. Il
# primo e' il raggio storico (un quarto di riga, al massimo 12 px): su Marvel
# bastava, perche' il varco era l'antialiasing fra balloon e bordo. Su Absolute
# Superman il balloon e' APERTO sulla cornice bianca della vignetta e il varco e'
# largo quanto il bordo: senza i raggi piu' grandi la maschera prendeva la
# vignetta intera, il profilo veniva scartato e al suo posto si copriva il
# rettangolo cresciuto - i rettangoli bianchi sull'arte di p5, p16, p17, p18.
SIGILLI = (1.0, 1.8, 2.6, 3.6)


def _colata(comp, righe, offset) -> bool:
    """True se la maschera e' troppo grande per essere il balloon delle righe."""
    boxes = [r for r in righe if float(r[2]) > float(r[0]) and float(r[3]) > float(r[1])]
    if not boxes or comp is None:
        return False
    ox, oy = offset
    x0 = min(float(r[0]) for r in boxes) - ox
    y0 = min(float(r[1]) for r in boxes) - oy
    x1 = max(float(r[2]) for r in boxes) - ox
    y1 = max(float(r[3]) for r in boxes) - oy
    area_righe = max(1.0, (x1 - x0) * (y1 - y0))
    return float(np.count_nonzero(comp)) > COLATURA_SU_RIGHE * area_righe


def _sigillo_progressivo(pieno, etichette_sotto, box, righe, offset):
    """La componente del balloon, sigillata quanto serve; None se non si isola.

    Si prova col raggio storico e, se la macchia resta troppo grande per essere
    un balloon (`_colata`), con sigilli piu' larghi. Si tiene il primo che non
    cola; se nessuno riesce si restituisce comunque il migliore (il piu' piccolo
    ottenuto), perche' una maschera stretta e' sempre meglio di una che ha
    inghiottito la vignetta.
    """
    base = _raggio_sigillo(box, righe)
    migliore = None
    for fattore in SIGILLI:
        tentativo = _senza_colature(pieno, etichette_sotto, int(round(base * fattore)))
        if tentativo is None:
            continue
        if not _colata(tentativo, righe, offset):
            return tentativo
        if migliore is None or np.count_nonzero(tentativo) < np.count_nonzero(migliore):
            migliore = tentativo
    return migliore


def _seme_di_fondo(roi: np.ndarray, cx: int, cy: int, bg: tuple,
                   tol: float = 40.0, raggio: int = 30):
    """Un pixel di FONDO del balloon vicino a (cx, cy), per il flood-fill.

    Il seme va scelto sul colore del balloon, cioe' su `bg`, non sulla
    luminosita'. Cercare "un pixel chiaro" funziona solo finche' il balloon e'
    bianco: su una didascalia a fondo rosso con lettering bianco il pixel piu'
    chiaro dell'intorno **e' il testo**, il riempimento parte da dentro una
    lettera e si ferma sul suo contorno. La maschera esce grande quanto un
    glifo, `shrink_to_mask` restituisce un riquadro di pochi pixel e il blocco
    viene poi scartato come illeggibile: e' il difetto misurato su carnage-5,
    dove 41 dei 53 blocchi persi erano `light_text`.

    Il centro di una riga cade quasi sempre sull'inchiostro, quindi si cerca il
    pixel di fondo piu' vicino nell'intorno. Ritorna (x, y) o None.
    """
    h, w = roi.shape[:2]
    cx = int(min(max(cx, 0), w - 1))
    cy = int(min(max(cy, 0), h - 1))
    x0, y0 = max(0, cx - raggio), max(0, cy - raggio)
    x1, y1 = min(w, cx + raggio + 1), min(h, cy + raggio + 1)
    finestra = roi[y0:y1, x0:x1]
    if finestra.size == 0:
        return None
    vicino = np.abs(finestra.astype(np.float32)
                    - np.array(bg, dtype=np.float32)).max(axis=2) < tol
    if not vicino.any():
        # `bg` campionato male (o balloon fuori tolleranza): si ripiega sul
        # pixel chiaro, che resta giusto sui balloon bianchi
        vicino = finestra.min(axis=2) > 140
    if not vicino.any():
        return None
    ys, xs = np.nonzero(vicino)
    # il piu' vicino al centro della finestra
    dist = (xs - (cx - x0)) ** 2 + (ys - (cy - y0)) ** 2
    i = int(np.argmin(dist))
    return (int(xs[i] + x0), int(ys[i] + y0))


def _near_balloon(roi: np.ndarray, offset, semi, bg: tuple, tol: float) -> np.ndarray:
    """Pixel di sfondo del balloon che contiene il testo.

    Preferisce un flood-fill che segue il gradiente (ogni pixel confrontato col
    vicino gia' incluso): cosi' abbraccia i balloon crema/gialli dal centro
    quasi bianco ai bordi giallo saturo, fermandosi sul contorno nero e sullo
    sfondo. Il solo `bg` con soglia fissa perdeva i bordi saturi (differenza sul
    canale blu > tol) e l'inglese sbucava ai lati - il difetto dei balloon di
    Deadpool. Il seme si cerca sul colore del balloon (`_seme_di_fondo`), non
    sulla luminosita': su una didascalia scura con lettering chiaro il pixel
    piu' chiaro e' il testo, e il riempimento resterebbe dentro una lettera.
    Se nessuna riga trova un seme si torna alla soglia globale attorno al `bg`.
    """
    try:
        import cv2
    except ImportError:
        return (np.abs(roi.astype(np.float32) - np.array(bg, dtype=np.float32)
                       ).max(axis=2) < tol).astype(np.uint8)
    soglia = (np.abs(roi.astype(np.float32) - np.array(bg, dtype=np.float32)
                     ).max(axis=2) < tol).astype(np.uint8)
    ox, oy = offset
    h, w = roi.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)
    contigua = np.ascontiguousarray(roi)
    delta = int(round(tol / 2.0))
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    for r in semi:
        seme = _seme_di_fondo(contigua, int((r[0] + r[2]) / 2) - ox,
                              int((r[1] + r[3]) / 2) - oy, bg, tol)
        if seme is None:
            continue
        cv2.floodFill(contigua.copy(), fill, seme, 0,
                      (delta, delta, delta), (delta, delta, delta), flags)
    # UNIONE, non alternativa: il flood-fill aggiunge il gradiente che la soglia
    # perde (i bordi gialli dei balloon crema), la soglia garantisce che non si
    # finisca mai sotto il comportamento storico se il riempimento si ferma
    # presto. Tenere solo il flood-fill era una regressione: su imperial-3
    # collassava 25 riquadri su 81 - fino a 2x34 px - e quei blocchi venivano
    # poi scartati come illeggibili, lasciando l'inglese sulla tavola.
    return (soglia | (fill[1:-1, 1:-1] > 0).astype(np.uint8)).astype(np.uint8)


def _balloon_mask(img: np.ndarray, box: list, bg: tuple, righe=(),
                  margin: int = 160, tol: float = 40.0):
    """Maschera dell'interno dei balloon che contengono il testo del gruppo.

    Si prendono le componenti connesse di colore di sfondo su cui poggiano le
    righe dell'OCR e se ne tappano i buchi, cosi' i glifi tornano a far parte
    dell'area. Serve perche' l'OCR ogni tanto non rileva una riga di una lettera
    sola: senza tappare i buchi l'espansione si fermerebbe contro quel
    carattere, che resterebbe scoperto e in lingua originale in mezzo alla
    traduzione.

    Le componenti sono piu' d'una quando il raggruppamento ha messo insieme due
    balloon vicini. Coprirne uno solo e' il caso peggiore che ci sia: la
    traduzione dei due viene scritta a cavallo, e sotto resta in chiaro la
    battuta originale dell'altro.
    """
    try:
        import cv2
    except ImportError:
        return None, (0, 0)

    h, w = img.shape[:2]
    x0 = max(0, int(box[0]) - margin)
    y0 = max(0, int(box[1]) - margin)
    x1 = min(w, int(box[2]) + margin)
    y1 = min(h, int(box[3]) + margin)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None, (x0, y0)

    bx0, by0 = max(0, int(box[0]) - x0), max(0, int(box[1]) - y0)
    bx1, by1 = int(box[2]) - x0, int(box[3]) - y0

    # Flood-fill dal testo che segue il gradiente dell'interno del balloon: cosi'
    # i balloon crema/gialli entrano tutti in maschera, non solo la parte chiara.
    near = _near_balloon(roi, (x0, y0), [box] + list(righe), bg, tol)

    def etichette_sotto(labels, valida):
        """Le componenti su cui poggiano il riquadro e le righe del gruppo."""
        dentro = labels[by0:by1, bx0:bx1][valida[by0:by1, bx0:bx1] > 0]
        if dentro.size == 0:
            return set()
        trovate = {int(np.bincount(dentro).argmax())}
        # una componente per ogni riga del gruppo: se le righe stanno in due
        # balloon diversi servono tutte e due le macchie di sfondo
        for r in righe:
            rx0, ry0 = max(0, int(r[0]) - x0), max(0, int(r[1]) - y0)
            rx1, ry1 = int(r[2]) - x0, int(r[3]) - y0
            sotto = labels[ry0:ry1, rx0:rx1][valida[ry0:ry1, rx0:rx1] > 0]
            if sotto.size:
                trovate.add(int(np.bincount(sotto).argmax()))
        trovate.discard(0)
        return trovate

    pieno = _tappa_buchi(near)
    comp = _sigillo_progressivo(pieno, etichette_sotto, box, righe, (x0, y0))
    if comp is None:
        count, labels = cv2.connectedComponents(pieno, connectivity=4)
        if count < 2:
            return None, (x0, y0)
        etichette = etichette_sotto(labels, pieno)
        if not etichette:
            return None, (x0, y0)
        comp = np.isin(labels, list(etichette))
    return _tappa_buchi(comp.astype(np.uint8)).astype(bool), (x0, y0)


def _muri(limits, box: list, w: int, h: int) -> tuple:
    """I quattro limiti oltre i quali il riquadro non deve crescere.

    Se un limite cade dentro il riquadro di partenza viene ignorato: l'ingombro
    dell'OCR e' intoccabile, e' solo la crescita che deve cedere il passo.
    """
    lx0, ly0, lx1, ly1 = (0.0, 0.0, float(w), float(h)) if limits is None else limits
    return (min(lx0, box[0]), min(ly0, box[1]), max(lx1, box[2]), max(ly1, box[3]))


def grow_in_mask(mask: np.ndarray, box: list, step: int = 4, max_grow: int = 400,
                 limits=None) -> list:
    """Allarga il rettangolo finche' resta dentro la maschera del balloon.

    `limits` ferma la crescita prima dei blocchi vicini: la maschera da sola non
    basta, perche' due balloon che si toccano formano una macchia sola e il
    testo di uno finirebbe a coprire quello dell'altro.
    """
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    mx0, my0, mx1, my1 = _muri(limits, [x0, y0, x1, y1], w, h)

    def ok(strip: np.ndarray) -> bool:
        return strip.size > 0 and float(strip.mean()) > 0.98

    for _ in range(max_grow // step):
        moved = False
        if x0 - step >= mx0 and ok(mask[y0:y1, x0 - step:x0]):
            x0 -= step
            moved = True
        if x1 + step <= mx1 and ok(mask[y0:y1, x1:x1 + step]):
            x1 += step
            moved = True
        if y1 + step <= my1 and ok(mask[y1:y1 + step, x0:x1]):
            y1 += step
            moved = True
        if y0 - step >= my0 and ok(mask[y0 - step:y0, x0:x1]):
            y0 -= step
            moved = True
        if not moved:
            break
    return [x0, y0, x1, y1]


def shrink_to_mask(mask: np.ndarray, box: list, step: int = 4,
                   max_shrink: int = 240, soglia: float = 0.92) -> list:
    """Riporta il rettangolo dentro il balloon, togliendo le strisce di fuori.

    Il rettangolo dell'OCR e' cieco alla forma: due balloon in diagonale hanno
    ingombri che si accavallano anche quando i balloon non si toccano, ed e' da
    li' che nasce il testo scritto addosso al vicino. I glifi non rischiano
    niente: nella maschera i loro buchi sono gia' stati tappati, quindi una
    striscia di testo risulta piena e non viene mai tolta.
    """
    x0, y0, x1, y1 = [int(v) for v in box]

    def fuori(strip: np.ndarray) -> bool:
        return strip.size == 0 or float(strip.mean()) < soglia

    minimo = 6 * step
    for _ in range(max_shrink // step):
        mosso = False
        if x1 - x0 > minimo and fuori(mask[y0:y1, x0:x0 + step]):
            x0 += step
            mosso = True
        if x1 - x0 > minimo and fuori(mask[y0:y1, x1 - step:x1]):
            x1 -= step
            mosso = True
        if y1 - y0 > minimo and fuori(mask[y0:y0 + step, x0:x1]):
            y0 += step
            mosso = True
        if y1 - y0 > minimo and fuori(mask[y1 - step:y1, x0:x1]):
            y1 -= step
            mosso = True
        if not mosso:
            break
    return [x0, y0, x1, y1]


def contorno_maschera(mask: np.ndarray, box: list, vietati=(), righe=(),
                      pad: int = 3, semplifica: float = 1.5, limiti=None) -> list:
    """Il profilo del balloon come poligono, in coordinate della maschera.

    Un poligono solo invece di tante fasce: riempire a fasce funziona sulla
    carta ma non nel PDF, perche' ogni bordo viene sfumato e a ogni giunzione
    resta una cucitura da cui il testo originale si rilegge in trasparenza.

    Dalla maschera si tolgono prima le righe degli altri gruppi: se il bordo del
    balloon ha una fessura la macchia di sfondo dilaga nella vignetta, e senza
    questo taglio il riempimento andrebbe a coprire la battuta del vicino.

    Togliere le sole righe altrui pero' non basta, perche' fra due balloon
    resta il margine chiaro della tavola: entrambe le maschere se lo prendono,
    si sovrappongono, e il riempimento cancella il bordo fra le vignette
    fondendo i due balloon in una macchia sola. `limiti` sono gli stessi muri
    che gia' fermano la crescita del riquadro di scrittura: applicarli anche
    qui e' cio' che tiene separate cancellazione e impaginazione. Il riquadro e
    le righe proprie restano sempre dentro: un muro non puo' scoprire
    l'originale che stiamo per coprire.
    """
    try:
        import cv2
    except ImportError:
        return []

    piena = mask.astype(np.uint8).copy()
    h, w = piena.shape[:2]
    for v in vietati:
        x0, y0 = max(0, int(v[0]) - pad), max(0, int(v[1]) - pad)
        x1, y1 = min(w, int(v[2]) + pad), min(h, int(v[3]) + pad)
        if x1 > x0 and y1 > y0:
            piena[y0:y1, x0:x1] = 0

    if limiti is not None:
        intoccabili = [box] + [list(r) for r in righe]
        mx0 = min([limiti[0]] + [b[0] for b in intoccabili]) - pad
        my0 = min([limiti[1]] + [b[1] for b in intoccabili]) - pad
        mx1 = max([limiti[2]] + [b[2] for b in intoccabili]) + pad
        my1 = max([limiti[3]] + [b[3] for b in intoccabili]) + pad
        fuori = np.ones((h, w), np.uint8)
        fx0, fy0 = max(0, int(mx0)), max(0, int(my0))
        fx1, fy1 = min(w, int(mx1)), min(h, int(my1))
        if fx1 > fx0 and fy1 > fy0:
            fuori[fy0:fy1, fx0:fx1] = 0
            piena[fuori > 0] = 0

    contorni, _ = cv2.findContours(piena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if not contorni:
        return []

    # quelli che reggono il testo, non il piu' grande: se la maschera e'
    # dilagata altrove il pezzo grosso e' proprio quello da non riempire
    ancore = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)]
    ancore += [((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in righe]
    scelti = [c for c in contorni
              if any(cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0
                     for x, y in ancore)]
    if not scelti:
        scelti = [max(contorni, key=cv2.contourArea)]

    return [[[int(p[0][0]), int(p[0][1])] for p in cv2.approxPolyDP(c, semplifica, True)]
            for c in scelti]


def _bbox(profilo: list) -> list:
    xs = [q[0] for q in profilo]
    ys = [q[1] for q in profilo]
    return [min(xs), min(ys), max(xs), max(ys)]


def _principale(profili: list, righe: list) -> list:
    """Il profilo che contiene piu' righe del gruppo: li' va scritto il testo."""
    def quante(profilo):
        x0, y0, x1, y1 = _bbox(profilo)
        dentro = sum(1 for r in righe
                     if x0 <= (r[0] + r[2]) / 2 <= x1 and y0 <= (r[1] + r[3]) / 2 <= y1)
        return (dentro, (x1 - x0) * (y1 - y0))

    return _bbox(max(profili, key=quante))


def _dentro_bbox(box: list, bbox: list) -> list:
    return [max(box[0], bbox[0]), max(box[1], bbox[1]),
            min(box[2], bbox[2]), min(box[3], bbox[3])]


def _con_le_righe_proprie(box: list, righe: list, confine=None) -> list:
    """Il riquadro di scrittura non puo' essere piu' stretto del testo che sostituisce.

    Le righe proprie del blocco sono esattamente l'inchiostro che il blocco
    cancella: scriverci dentro non puo' danneggiare nessun vicino, mentre
    togliere quello spazio produce il difetto misurato su Captain America #2
    pagina 4, dove la didascalia aveva un riquadro largo 254 px contro i 502
    del testo originale, la traduzione non entrava e il blocco tornava in
    inglese. `confine` e' il limite oltre il quale non si scrive comunque: il
    profilo del balloon che viene riempito, o i muri dei vicini.
    """
    if not righe:
        return box
    proprio = [min(r[0] for r in righe), min(r[1] for r in righe),
               max(r[2] for r in righe), max(r[3] for r in righe)]
    if confine is not None:
        proprio = _dentro_bbox(proprio, confine)
    if proprio[2] <= proprio[0] or proprio[3] <= proprio[1]:
        return box
    return [min(box[0], proprio[0]), min(box[1], proprio[1]),
            max(box[2], proprio[2]), max(box[3], proprio[3])]


# Aria attorno alle righe nella fascia di riserva: copre le code delle lettere
# senza mangiare l'arte accanto. Stesso margine dei candidati del Gap C.
BANDA_ARIA = 6

# Quanto lontano dalle proprie righe puo' arrivare la macchia di sfondo, in
# altezze di riga. Un balloon vero sporge meno di una riga oltre il suo
# lettering; oltre quell'intorno la macchia ha passato il contorno ed e' finita
# sulla cornice della vignetta o sull'arte. Il valore tiene insieme le due cose
# che contano: non cancellare i bordi del balloon (0,8 righe bastano a chiudere
# l'inchiostro) e lasciare all'italiano, che e' piu' lungo, un po' di spazio in
# piu' del lettering originale. E' lo stesso vincolo che nel `.finale` di
# superman-1 ha rimesso a posto p16-p18.
INTORNO_RIGHE = 0.8
INTORNO_MINIMO = 20


def _intorno_delle_righe(mask, righe):
    """La maschera ritagliata a un intorno delle righe del blocco.

    Senza questo taglio il riempimento seguiva il bianco del balloon fin dentro
    la cornice della vignetta e la toppa usciva rettangolare sopra i bordi e
    sopra l'arte (Absolute Superman p4, p5, p16-p18). Quello che si perde e'
    solo il fondo del balloon lontano dal testo, dove non c'e' niente da
    cancellare.
    """
    boxes = [r for r in righe or () if r and r[2] > r[0] and r[3] > r[1]]
    if not boxes or mask is None:
        return mask
    altezze = [r[3] - r[1] for r in boxes]
    margine = max(INTORNO_MINIMO, INTORNO_RIGHE * float(np.median(altezze)))
    h, w = mask.shape[:2]
    x0 = max(0, int(min(r[0] for r in boxes) - margine))
    y0 = max(0, int(min(r[1] for r in boxes) - margine))
    x1 = min(w, int(np.ceil(max(r[2] for r in boxes) + margine)))
    y1 = min(h, int(np.ceil(max(r[3] for r in boxes) + margine)))
    if x1 <= x0 or y1 <= y0:
        return mask
    ritagliata = np.zeros_like(mask)
    ritagliata[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return ritagliata


def banda_delle_righe(righe, aria: int = BANDA_ARIA) -> list:
    """Il riempimento minimo di un blocco: una fascia per riga, l'inchiostro e basta.

    Serve quando la forma del balloon non si isola. Prima, senza profilo, chi
    disegna copriva il RETTANGOLO CRESCIUTO del blocco: su Absolute Superman
    erano rettangoli bianchi sopra i bordi delle vignette e sopra l'arte (p4,
    p5), e su una didascalia scura in mezzo al disegno la toppa si vedeva tutta.
    La fascia delle righe copre esattamente l'inglese che si sta sostituendo:
    puo' restare visibile un pezzo di fondo attorno alle lettere, ma non si
    cancella niente che non fosse testo.
    """
    profili = []
    for r in righe or ():
        x0, y0, x1, y1 = [float(v) for v in r]
        if x1 <= x0 or y1 <= y0:
            continue
        profili.append([[x0 - aria, y0 - aria], [x1 + aria, y0 - aria],
                        [x1 + aria, y1 + aria], [x0 - aria, y1 + aria]])
    return profili


def _profili_sicuri(profili: list, box: list, max_area_ratio: float = 5.0) -> list:
    """Scarta una macchia di sfondo propagata oltre il testo locale."""
    area_box = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    safe = []
    for profilo in profili:
        if len(profilo) < 3:
            continue
        bounds = _bbox(profilo)
        area = max(0.0, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
        if area <= area_box * max_area_ratio:
            safe.append(profilo)
    return safe


def prepara_area(img: np.ndarray, box: list, bg: tuple, limits=None,
                 vietati=(), righe=()) -> tuple:
    """Dove scrivere e cosa coprire: (rettangolo del testo, profilo del balloon).

    Senza maschera si torna al comportamento a strisce di prima, col profilo
    vuoto: chi disegna coprira' tutto il rettangolo, come si e' sempre fatto.
    """
    mask, (ox, oy) = _balloon_mask(img, box, bg, righe)
    if mask is None:
        cresciuto = grow_in_background(img, box, bg, limits=limits)
        return (_con_le_righe_proprie(cresciuto, list(righe), limits),
                banda_delle_righe(righe))

    def locale(r):
        return None if r is None else [r[0] - ox, r[1] - oy, r[2] - ox, r[3] - oy]

    # La macchia di sfondo si ferma a un intorno delle righe: oltre non c'e' piu'
    # il balloon, c'e' la cornice della vignetta (i balloon di Absolute Superman
    # sono aperti su di essa) o l'arte accanto. Vale per il riempimento E per
    # l'impaginazione, perche' si scrive solo dove si cancella.
    mask = _intorno_delle_righe(mask, [locale(r) for r in righe])
    dentro = shrink_to_mask(mask, locale(box))
    grown = grow_in_mask(mask, dentro, limits=locale(limits))
    profili = contorno_maschera(mask, dentro, [locale(v) for v in vietati],
                                [locale(r) for r in righe], limiti=locale(limits))
    profili = _profili_sicuri(profili, locale(box))
    if not profili:
        # Nessuna forma affidabile (macchia scartata perche' propagata oltre il
        # testo): si copre l'inchiostro e non il rettangolo, che qui sarebbe una
        # toppa sull'arte. Il riquadro di scrittura resta quello cresciuto.
        grown = _con_le_righe_proprie(grown, [locale(r) for r in righe], locale(limits))
        return ([grown[0] + ox, grown[1] + oy, grown[2] + ox, grown[3] + oy],
                banda_delle_righe(righe))

    # se l'OCR ha messo due balloon nello stesso gruppo il rettangolo li
    # abbraccia entrambi, e il testo verrebbe centrato sull'unione: cioe' in
    # mezzo, a cavallo del bordo e sopra la battuta del balloon di fianco. Si
    # scrive in quello che regge piu' righe; gli altri vengono solo ripuliti.
    proprie = [locale(r) for r in righe]
    if profili:
        principale = _principale(profili, proprie)
        grown = _dentro_bbox(grown, principale)
        # Dentro il profilo che viene riempito il testo proprio e' gia'
        # cancellato: quello spazio spetta alla traduzione.
        grown = _con_le_righe_proprie(grown, proprie, principale)
    else:
        grown = _con_le_righe_proprie(grown, proprie, locale(limits))
    # una lista di profili, non uno solo: un gruppo puo' stare su due balloon,
    # e due blocchi fusi con `unisci_con` ne portano uno per parte
    return ([grown[0] + ox, grown[1] + oy, grown[2] + ox, grown[3] + oy],
            [[[p[0] + ox, p[1] + oy] for p in profilo] for profilo in profili])


def expand_box(img: np.ndarray, box: list, bg: tuple, limits=None) -> list:
    """Area di scrittura definitiva: dentro il balloon se c'e', altrimenti a strisce."""
    mask, (ox, oy) = _balloon_mask(img, box, bg)
    if mask is None:
        return grow_in_background(img, box, bg, limits=limits)
    local = [box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy]
    local_limits = None if limits is None else [
        limits[0] - ox, limits[1] - oy, limits[2] - ox, limits[3] - oy]
    grown = grow_in_mask(mask, local, limits=local_limits)
    return [grown[0] + ox, grown[1] + oy, grown[2] + ox, grown[3] + oy]


def grow_in_background(
    img: np.ndarray, box: list, bg: tuple, tol: float = 26.0, step: int = 4,
    max_grow: int = 90, limits=None
) -> list:
    """Allarga il rettangolo finche' la striscia che aggiunge e' ancora sfondo.

    Dentro un balloon si ferma sul bordo nero; su una scansione si ferma sul
    testo o sulla grafica vicina. Cosi' la traduzione ha tutto lo spazio bianco
    realmente disponibile invece del solo ingombro dell'originale.
    """
    h, w = img.shape[:2]
    bg_arr = np.array(bg, dtype=np.float32)

    def is_background(strip: np.ndarray) -> bool:
        if strip.size == 0:
            return False
        diff = np.abs(strip.astype(np.float32) - bg_arr).max(axis=2)
        return float((diff < tol).mean()) > 0.97

    x0, y0, x1, y1 = [int(v) for v in box]
    mx0, my0, mx1, my1 = _muri(limits, [x0, y0, x1, y1], w, h)
    for _ in range(max_grow // step):
        moved = False
        if x0 - step >= mx0 and is_background(img[y0:y1, x0 - step:x0]):
            x0 -= step
            moved = True
        if x1 + step <= mx1 and is_background(img[y0:y1, x1:x1 + step]):
            x1 += step
            moved = True
        if y1 + step <= my1 and is_background(img[y1:y1 + step, x0:x1]):
            y1 += step
            moved = True
        if y0 - step >= my0 and is_background(img[y0 - step:y0, x0:x1]):
            y0 -= step
            moved = True
        if not moved:
            break
    return [x0, y0, x1, y1]


# --------------------------------------------------------------------------- #
# Vicinato: nessun gruppo puo' scrivere addosso a un altro
# --------------------------------------------------------------------------- #

MARGINE = 6  # px OCR di aria fra due riquadri che si sono incontrati


def limiti_crescita(box: list, altri: list, shape: tuple, pad: int = MARGINE) -> list:
    """Fin dove puo' crescere `box` senza avvicinarsi troppo agli `altri`.

    Guarda solo gli ostacoli che stanno davvero sulla traiettoria: quelli sulla
    stessa fascia orizzontale fermano la crescita laterale, quelli sulla stessa
    colonna quella verticale. Gli obliqui li prende dopo `separa_sovrapposti`.
    """
    h, w = shape[:2]
    lx0, ly0, lx1, ly1 = 0.0, 0.0, float(w), float(h)
    for o in altri:
        if o[3] > box[1] and o[1] < box[3]:  # stessa fascia orizzontale
            if o[2] <= box[0]:
                lx0 = max(lx0, o[2] + pad)
            elif o[0] >= box[2]:
                lx1 = min(lx1, o[0] - pad)
        if o[2] > box[0] and o[0] < box[2]:  # stessa colonna
            if o[3] <= box[1]:
                ly0 = max(ly0, o[3] + pad)
            elif o[1] >= box[3]:
                ly1 = min(ly1, o[1] - pad)
    return [lx0, ly0, lx1, ly1]


def _cede(a: list, oa: list, b: list, ob: list, lo: int, hi: int, quanto: float) -> None:
    """Toglie `quanto` di sovrapposizione su un asse, a spese della sola crescita.

    Chi ha guadagnato spazio lo restituisce; dentro il riquadro originale non si
    entra mai, perche' quello e' testo davvero letto dall'OCR.
    """
    if a[lo] > b[lo]:  # `a` e' sempre quello che viene prima sull'asse
        a, oa, b, ob = b, ob, a, oa
    puo_a = max(0.0, a[hi] - oa[hi])
    puo_b = max(0.0, ob[lo] - b[lo])
    if puo_a + puo_b <= 0:
        return  # si sovrappongono gia' gli originali: non e' roba nostra
    quota_a = min(puo_a, quanto * puo_a / (puo_a + puo_b))
    a[hi] -= quota_a
    resto = quanto - quota_a
    if resto > 0:
        b[lo] += min(puo_b, resto)


def separa_sovrapposti(boxes: list, originali: list, pad: int = MARGINE,
                       passate: int = 4) -> int:
    """Disfa le sovrapposizioni rimaste, un po' per uno fra i due riquadri."""
    separati = 0
    for _ in range(passate):
        toccato = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                dx = min(a[2], b[2]) - max(a[0], b[0])
                dy = min(a[3], b[3]) - max(a[1], b[1])
                if dx <= 0 or dy <= 0:
                    continue
                # si cede sull'asse dove costa meno: l'altro resta com'e'
                if dx <= dy:
                    _cede(a, originali[i], b, originali[j], 0, 2, dx + pad)
                else:
                    _cede(a, originali[i], b, originali[j], 1, 3, dy + pad)
                toccato = True
                separati += 1
        if not toccato:
            break
    return separati


# --------------------------------------------------------------------------- #
# Scrittura sulla pagina
# --------------------------------------------------------------------------- #

CORPO_ASSOLUTO = 2.5  # sotto questo corpo non si scende: sarebbe illeggibile

# Il corpo di partenza viene dall'altezza delle lettere lette sulla tavola. Se
# per far stare la traduzione bisogna scendere sotto la sua meta', il riquadro
# non e' quello di quel testo: succede con le didascalie spezzate dall'OCR, dove
# il pezzo riconosciuto e' una frazione della striscia vera. Misurato sui piani
# raccolti: il grosso dei balloon sta fra 0.7 e 0.9, sotto 0.5 ci finiscono solo
# i riquadri sbagliati.
FRAZIONE_CORPO_MINIMA = 0.5


def _entra(font, testo: str, lines: list, size: float, rect) -> bool:
    """Il testo sta davvero dentro il rettangolo, in altezza e in larghezza.

    Si controlla anche la parola piu' lunga dell'originale, non solo le righe
    gia' mandate a capo: al corpo minimo `fit` spezza in due le parole che non
    ci stanno, e una riga che finisce con "DISASTROS" non e' un testo che entra.
    """
    if len(lines) * size * 1.16 > rect.height + 0.6:
        return False
    if any(font.text_length(line, size) > rect.width + 0.5 for line in lines):
        return False
    parole = testo.split() or [testo]
    return max(font.text_length(w, size) for w in parole) <= rect.width + 0.5


def _misura(font, text, inner, base_size, min_size, fit_fn):
    """Corpo e righe con cui il testo entra, oppure None se non entra mai.

    Prima si prova al corpo minimo dichiarato, poi al minimo assoluto: sotto
    non si scende, perche' una riga di due punti non e' una traduzione.
    """
    minimo = base_size * FRAZIONE_CORPO_MINIMA
    size, lines = fit_fn(font, text, inner, base_size, min_size, 1.0)
    if _entra(font, text, lines, size, inner):
        return (size, lines) if size >= minimo else None
    # `fit_fn` si ferma a `min_size` anche quando il testo non ci sta lo stesso,
    # e quello che avanza finiva addosso al balloon accanto. Un corpo sotto il
    # minimo si legge male; due battute sovrapposte non si leggono affatto.
    size, lines = fit_fn(font, text, inner, base_size,
                         max(CORPO_ASSOLUTO, min_size * 0.55), 1.0)
    if not _entra(font, text, lines, size, inner) or size < minimo:
        return None
    return size, lines


# Quanto si puo' stringere il riquadro di scrittura per farlo stare dentro il
# balloon, per lato. Sotto, il corpo scende troppo: meglio il riquadro pieno.
RIDUZIONE_MINIMA = 0.6


def area_di_scrittura(box_px, scale, fill_poly=None, inset=1.5):
    """Il rettangolo (pt) in cui scrivere: quello delle righe, stretto dentro il balloon.

    Il riquadro del blocco e' il rettangolo delle righe inglesi. L'italiano e'
    piu' lungo, lo riempie fino agli angoli, e sui balloon ovali gli angoli
    cadono fuori: le lettere finivano sopra il contorno nero (prova A1 su
    imperial-guardians, p11 e p15, quasi ogni balloon). Qui il rettangolo si
    stringe attorno al suo centro finche' angoli e punti medi dei lati stanno
    dentro i profili di riempimento, fino a `RIDUZIONE_MINIMA` per lato.

    Senza profili, o con il centro fuori dal balloon (profilo concavo), resta il
    rettangolo di sempre: niente di peggio di prima.
    """
    rect = fitz.Rect(*[v * scale for v in box_px])
    inner = fitz.Rect(rect.x0 + inset, rect.y0 + inset, rect.x1 - inset, rect.y1 - inset)
    profili = [np.asarray(p, np.float32) for p in (fill_poly or []) if len(p) >= 3]
    if inner.is_empty or not profili or not scale:
        return inner
    import cv2

    def dentro(x_pt, y_pt):
        punto = (float(x_pt / scale), float(y_pt / scale))
        return any(cv2.pointPolygonTest(c, punto, False) >= 0 for c in profili)

    cx, cy = (inner.x0 + inner.x1) / 2, (inner.y0 + inner.y1) / 2
    if not dentro(cx, cy):
        return inner
    mezza_w, mezza_h = inner.width / 2, inner.height / 2
    fattore = 1.0
    while True:
        x0, x1 = cx - mezza_w * fattore, cx + mezza_w * fattore
        y0, y1 = cy - mezza_h * fattore, cy + mezza_h * fattore
        punti = ((x0, y0), (x1, y0), (x0, y1), (x1, y1),
                 (cx, y0), (cx, y1), (x0, cy), (x1, cy))
        if all(dentro(x, y) for x, y in punti) or fattore <= RIDUZIONE_MINIMA:
            return fitz.Rect(x0, y0, x1, y1)
        fattore = max(RIDUZIONE_MINIMA, fattore * 0.96)


def misura_nel_balloon(font, text, box_px, scale, fill_poly, base_size, min_size, fit_fn,
                       inset=1.5):
    """(rettangolo, misura) per scrivere: prima dentro il balloon, poi il riquadro pieno.

    Se nel rettangolo stretto il testo non entra nemmeno al corpo minimo, si
    torna al riquadro delle righe: una battuta che sfiora il contorno si legge,
    una battuta rifiutata resta in inglese.
    """
    pieno = area_di_scrittura(box_px, scale, None, inset)
    stretto = area_di_scrittura(box_px, scale, fill_poly, inset)
    for inner in ((stretto, pieno) if stretto != pieno else (pieno,)):
        if inner.is_empty:
            continue
        misura = _misura(font, text, inner, base_size, min_size, fit_fn)
        if misura is not None:
            return inner, misura
    return pieno, None


def misura_lobi(font, lobi, scale, base_size, min_size, fit_fn, inset=1.5):
    """[(rettangolo, (corpo, righe))] per ogni lobo, con un corpo comune; None se uno non entra.

    Il lettering di un balloon ha un corpo solo: un lobo piccolo scritto piu'
    grande del lobo accanto si legge come un'altra voce. Si misura ogni lobo, si
    prende il corpo piu' piccolo e si rimanda a capo con quello.
    """
    misure = []
    for lobo in lobi or []:
        testo = lobo.get("testo") or ""
        if not testo.strip():
            return None
        inner, misura = misura_nel_balloon(font, testo, lobo["box_px"], scale,
                                           lobo.get("profili"), base_size, min_size,
                                           fit_fn, inset)
        if misura is None:
            return None
        misure.append((inner, testo, misura))
    if not misure:
        return None
    comune = min(m[2][0] for m in misure)
    risultato = []
    for inner, testo, misura in misure:
        corpo, righe = fit_fn(font, testo, inner, comune, comune, 1.0)
        if corpo <= comune + 1e-6 and _entra(font, testo, righe, corpo, inner):
            risultato.append((inner, (corpo, righe)))
        else:
            risultato.append((inner, misura))
    return risultato


def entra_il_job(font, text, box_px, scale, fill_poly, base_size, min_size, fit_fn,
                 lobi=None, inset=1.5) -> bool:
    """Come lo disegna `draw_group`: per lobi se si puo', altrimenti nel riquadro."""
    if lobi and misura_lobi(font, lobi, scale, base_size, min_size, fit_fn, inset):
        return True
    return misura_nel_balloon(font, text, box_px, scale, fill_poly, base_size,
                              min_size, fit_fn, inset)[1] is not None


def _scrivi_righe(writer, font, inner, size, lines, centered):
    line_h = size * 1.16
    total = len(lines) * line_h
    # blocco centrato verticalmente: nei balloon e' quello che ci si aspetta
    top = inner.y0 + max(0.0, (inner.height - total) / 2)
    asc = min(max(font.ascender, 0.7), 1.0)
    wrote = False
    for i, line in enumerate(lines):
        # nessun taglio: il corpo e' gia' stato ridotto fin dove serviva, e se
        # anche cosi' l'ultima riga sborda dal balloon si scrive lo stesso. Una
        # riga che esce di poco si legge; una battuta troncata no.
        y = top + size * asc + i * line_h
        x = inner.x0
        if centered:
            x += max(0.0, (inner.width - font.text_length(line, size)) / 2)
        writer.append(fitz.Point(x, y), line, font=font, fontsize=size)
        wrote = True
    return wrote


def draw_group(
    page: "fitz.Page",
    box_px: list,
    text: str,
    scale: float,
    bg: tuple,
    light_text: bool,
    base_size: float,
    fonts,
    min_size: float,
    fit_fn,
    inset: float = 1.5,
    fill_poly: list = None,
    riempi: bool = True,
    scrivi: bool = True,
    centered: bool = True,
    lobi: list = None,
) -> None:
    """Copre l'area con il colore di sfondo e ci scrive sopra la traduzione.

    Con `fill_poly` si copre la forma del balloon invece del rettangolo che la
    contiene: un rettangolo sui balloon in diagonale sconfina su quello accanto
    e ne cancella mezza battuta.
    """
    rect = fitz.Rect(*[v * scale for v in box_px])
    fill = tuple(c / 255.0 for c in bg)
    font = fonts.pick(bold=True, italic=False)
    # Balloon a piu' lobi: una battuta per lobo. Se anche un solo lobo non
    # entra si torna al riquadro unico, che almeno non perde parole.
    per_lobo = misura_lobi(font, lobi, scale, base_size, min_size, fit_fn, inset) if lobi else None
    inner, misura = misura_nel_balloon(font, text, box_px, scale, fill_poly, base_size,
                                       min_size, fit_fn, inset)
    if per_lobo is not None:
        misura = misura or per_lobo[0][1]
    if misura is None:
        # Non ci sta nemmeno al corpo minimo assoluto: qui il riquadro
        # riconosciuto e' molto piu' piccolo del testo, e quello che si
        # otterrebbe e' una riga illeggibile sopra un'area coperta a meta'.
        # L'originale, che almeno si legge, vale piu' di una traduzione persa.
        return False
    size, lines = misura
    if riempi:
        profili = [p for p in (fill_poly or []) if len(p) >= 3]
        if profili:
            for profilo in profili:
                forma = page.new_shape()
                forma.draw_polyline([fitz.Point(p[0] * scale, p[1] * scale)
                                     for p in profilo])
                forma.finish(color=None, fill=fill, closePath=True)
                forma.commit(overlay=True)
        else:
            page.draw_rect(rect, color=None, fill=fill, overlay=True)
    if not scrivi:
        return True

    writer = fitz.TextWriter(page.rect)
    if per_lobo is not None:
        wrote = False
        for lobo_inner, (lobo_size, lobo_lines) in per_lobo:
            wrote = _scrivi_righe(writer, font, lobo_inner, lobo_size, lobo_lines, True) or wrote
    else:
        wrote = _scrivi_righe(writer, font, inner, size, lines, centered)
    if wrote:
        writer.write_text(page, color=(1, 1, 1) if light_text else (0, 0, 0))
    return True


# --------------------------------------------------------------------------- #
# Orchestrazione di una pagina
# --------------------------------------------------------------------------- #

@dataclass
class ImageJob:
    """Un gruppo pronto da tradurre e ridisegnare."""

    page: int
    box_px: list
    text: str
    upper: bool
    base_size: float
    bg: tuple
    light_text: bool
    scale: float  # pixel dell'immagine OCR -> punti PDF
    fill_poly: list = field(default_factory=list)  # i profili dei balloon da coprire
    translated: str = ""
    centered: bool = True  # compatibile con i job/piani precedenti senza allinea
    # Riferimento di servizio (ritrovare il blocco dal job): non fa parte di
    # come il job viene disegnato, quindi resta fuori dal confronto.
    source_region_id: str | None = field(default=None, compare=False)
    # Gruppo scartato come disegno ma con la forma del testo (Gap C): diventa un
    # blocco saltato che arriva comunque al modello.
    scartato_come_disegno: bool = field(default=False, compare=False)
    # Balloon a piu' lobi (`comics_translator.disegno.lobi`): [{box_px, testo, profili}].
    # Vuoto = una battuta sola nel riquadro del blocco, come sempre.
    lobi: list = field(default_factory=list, compare=False)


def _to_sentence_case(text: str) -> str:
    """I fumetti sono in maiuscolo, i modelli di traduzione no.

    Passare "TELL THEM TO STAND DOWN" a un modello NMT peggiora sensibilmente la
    resa: viene tradotto in minuscolo normale e rimesso in maiuscolo al momento
    di disegnarlo.
    """
    lowered = text.lower()
    out = []
    capitalize = True
    for ch in lowered:
        if capitalize and ch.isalpha():
            out.append(ch.upper())
            capitalize = False
        else:
            out.append(ch)
        if ch in ".!?":
            capitalize = True
    return "".join(out)


def plan_page(
    page: "fitz.Page",
    pno: int,
    dpi: int,
    min_conf: float,
    expand: bool,
    max_bg_noise: float = 15.0,
    *,
    inventory: dict | None = None,
    localize: bool = True,
    recover: bool = True,
    candidati_scartati: bool = True,
) -> tuple:
    """OCR + aree; opzionalmente conserva anche le osservazioni escluse.

    Con `candidati_scartati` i gruppi esclusi come disegno che hanno la forma
    del testo tornano come job marcati `scartato_come_disegno`: il piano li
    tiene saltati, ma arrivano al modello invece di sparire (Gap C).
    """
    requested_dpi = dpi
    img, dpi = page_to_array(page, dpi)  # il dpi puo' essere stato ridotto
    scale = 72.0 / dpi
    observations = ocr_lines(img, min_conf, include_excluded=True)
    inventory = inventory if inventory is not None else inventory_data.new_inventory()
    inventory["localizzazione_attiva"] = localize
    if localize and recover:
        observations, inventory["recupero"] = localization.recover(
            page, observations, dpi,
            lambda crop: ocr_lines(crop, min_conf, include_excluded=True),
        )
    inventory["acquisizioni"].append({
        "tipo": "ocr", "dpi_richiesto": requested_dpi, "dpi_effettivo": dpi,
        "dimensioni_px": [int(img.shape[1]), int(img.shape[0])], "scala_px_pt": scale,
    })
    lines, line_ids = localization.register_rows(
        inventory, observations, pno, scale, fix_spacing, localize=localize,
    )
    region_ids = {}

    # prima si tengono i gruppi buoni, poi si allargano: l'espansione deve
    # sapere dove stanno gli altri, altrimenti due balloon vicini si invadono
    tenuti = []
    candidati = []
    skipped = 0
    for g in cluster(lines, img):
        text = g.text
        box = [g.x0, g.y0, g.x1, g.y1]
        reason = "too_short" if len(text) < 2 else None
        bg, light = sample_background(img, box)
        noise = background_noise(img, box, light) if reason is None else None
        if noise is not None and noise > max_bg_noise:
            reason = "background_noise"
            skipped += 1  # contatore legacy; i nuovi scarti sono nell'inventario
        region_ids[id(g)] = inventory_data.add_region(
            inventory, pno, [line_ids[id(line)] for line in g.lines],
            inventory_data.coordinates(box, scale), text, kind="gruppo_ocr",
            reason=reason, box_px=list(box),
        )
        inventory["regioni"][-1]["rumore_sfondo"] = noise
        if reason == "background_noise" and candidati_scartati and candidato_scartato(
                g.text, len(g.lines)):
            candidati.append((g, box, bg, light))
        if reason:
            continue
        tenuti.append((g, box, bg, light))

    originali = [list(box) for _, box, _, _ in tenuti]
    boxes = [list(box) for box in originali]
    profili = [[] for _ in tenuti]
    righe = [
        [[line.x0, line.y0, line.x1, line.y1] for line in g.lines]
        for g, _, _, _ in tenuti
    ]
    if expand:
        for i, (_, _, bg, _) in enumerate(tenuti):
            altri = [o for j, o in enumerate(originali) if j != i]
            righe_altrui = [r for j, rs in enumerate(righe) if j != i for r in rs]
            if localize:
                own = {line_ids[id(line)] for line in tenuti[i][0].lines}
                source_rows = {r["id"]: r for r in inventory["righe"]}
                righe_altrui = [
                    r["box_px"] for r in inventory["righe"]
                    if r["id"] not in own and r.get("motivo") != "duplicate" and r.get("box_px")
                    and not localization.belongs_to(r, own, source_rows)
                    and localization.axis_aligned(r)
                ]
            boxes[i], profili[i] = prepara_area(
                img, originali[i], bg,
                limiti_crescita(originali[i], altri, img.shape),
                righe_altrui, righe[i])
        separa_sovrapposti(boxes, originali)

    jobs = [
        ImageJob(
            page=pno,
            box_px=boxes[i],
            text=_to_sentence_case(g.text) if g.upper else g.text,
            upper=g.upper,
            base_size=g.corpo_righe * 0.78 * scale,
            bg=bg,
            light_text=light,
            scale=scale,
            fill_poly=profili[i],
            source_region_id=region_ids[id(g)],
        )
        for i, (g, _, bg, light) in enumerate(tenuti)
    ]
    # I candidati del Gap C non si allargano e non spingono i vicini: restano
    # sul rettangolo delle proprie righe, perche' intorno c'e' spesso arte.
    altezza, larghezza = img.shape[:2]
    for g, box, bg, light in candidati:
        riquadro = [max(0, box[0] - MARGINE_CANDIDATO), max(0, box[1] - MARGINE_CANDIDATO),
                    min(larghezza, box[2] + MARGINE_CANDIDATO),
                    min(altezza, box[3] + MARGINE_CANDIDATO)]
        jobs.append(ImageJob(
            page=pno,
            box_px=riquadro,
            text=_to_sentence_case(g.text) if g.upper else g.text,
            upper=g.upper,
            base_size=g.corpo_righe * 0.78 * scale,
            bg=bg,
            light_text=light,
            scale=scale,
            fill_poly=[],
            source_region_id=region_ids[id(g)],
            scartato_come_disegno=True,
        ))
    return jobs, len(lines), skipped
