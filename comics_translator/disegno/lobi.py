"""Balloon a piu' lobi: una battuta per lobo, non una battuta a cavallo della strozzatura.

Prova A1 su imperial-guardians (16 settembre 2026): 9 balloon su 29 pagine
avevano due o tre lobi. Il renderer conosce un solo modello, "un blocco = un
riquadro = un testo", e scriveva la traduzione intera centrata sul blocco:
sopra il contorno, a cavallo della strozzatura, con i lobi piccoli lasciati
bianchi e vuoti ("SAME." e "SKRULLS." a p8, "SURE." a p27). Il `.finale` li ha
riscritti a mano uno per uno: 14 lobi e una ventina di interventi.

Tutto quello che serve per farlo da soli c'e' gia' nel piano:

- **quali righe stanno in quale lobo**: se il blocco ha piu' profili di
  riempimento (i `compound`), ogni riga appartiene al profilo che la contiene;
  se il profilo e' uno solo, i lobi sono gruppi di righe separati da un salto
  verticale, ma il salto conta solo se il contorno si STRINGE fra i due gruppi:
  un balloon ovale con due frasi e un capoverso ha lo stesso salto e nessuna
  strozzatura, e non si divide;
- **quale pezzo di traduzione va in quale lobo**: si contano le frasi. Ogni
  lobo inglese ne ha un certo numero, la traduzione ne ha altrettante in
  totale, e l'ordine di lettura e' lo stesso. Sui 9 casi della prova i conti
  tornano quasi sempre (a p7 l'OCR perde il punto di "CAPTAIN MARVEL." e il
  lobo resta unito al successivo). Quando non tornano non si indovina: si resta
  alla divisione piu' grossa o al disegno di prima, che non perde parole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

import numpy as np

# Fine di frase: punti, esclamativi, interrogativi, puntini (anche il carattere
# unico) e i trattini lunghi con cui i fumetti interrompono una battuta ("IS--").
# Contano solo se seguiti da uno spazio o dalla fine: "C.B" e "HAND-TO-HAND" no.
FINE_FRASE = re.compile(r"(?:\.{2,}|…|[.!?]+|--+|—)[\"'»”’)]*(?=\s|$)")
_LETTERA = re.compile(r"\w")

# Un gruppo di righe comincia un lobo nuovo se il salto dal precedente supera
# questa frazione dell'altezza mediana di riga. Dentro un lobo le righe OCR si
# accavallano (salto negativo); fra due lobi c'e' il contorno: +52..+94 px su
# righe alte 85..110 px nei casi della prova. Il salto da solo non basta mai:
# decide la strozzatura.
SALTO_FRA_LOBI = 0.35
# Il contorno si stringe se la sua larghezza nel salto e' sotto questa frazione
# della larghezza dei due lobi misurata all'altezza delle loro righe.
STROZZATURA = 0.85
# Quanta aria lasciare fra il testo e il contorno, in frazione di riga.
ARIA_CONTORNO = 0.12


# L'OCR perde spesso lo spazio dopo il punto: "KL'RT.ITIS", "ALLIANCE.THE".
# Sulle righe OCR un punto attaccato a una lettera chiude la frase se la parola
# prima ha almeno due lettere: "S.H.I.E.L.D." e "C.B." restano interi.
_PUNTO_FUSO = re.compile(r"(?<=[^\W\d_]{2})([.!?]+)(?=[^\W\d_])")


def frasi(testo: str, ocr: bool = False) -> list[str]:
    """Le frasi di una battuta, in ordine. Un resto senza punto finale e' una frase.

    `ocr=True` per il testo letto sulla tavola, dove lo spazio dopo il punto
    manca spesso; la traduzione del modello ha gli spazi al posto giusto.
    """
    testo = " ".join((testo or "").split())
    if ocr:
        testo = _PUNTO_FUSO.sub(lambda m: m.group(1) + " ", testo)
    parti, inizio = [], 0
    for m in FINE_FRASE.finditer(testo):
        pezzo = testo[inizio:m.end()].strip()
        # "...E POI" comincia con dei puntini: senza lettere non e' una frase,
        # restano attaccati a quella che segue.
        if not _LETTERA.search(pezzo):
            continue
        parti.append(pezzo)
        inizio = m.end()
    resto = testo[inizio:].strip()
    if _LETTERA.search(resto):
        parti.append(resto)
    return parti


@dataclass
class Lobo:
    """Un lobo del balloon: le sue righe inglesi e dove si puo' scrivere."""

    righe: list  # box_px delle righe proprie
    testo_en: str
    profili: list = field(default_factory=list)  # poligoni px che lo contengono
    banda: tuple | None = None  # (y0, y1) px, quando il profilo e' condiviso
    # Solo sul primo lobo di un compound: la divisione piu' fine, con le
    # strozzature DENTRO i profili (p16: "DID YOU SAY THRONEWORLD?" e' un
    # lobo piccolo attaccato al grande). Si usa se le frasi si allineano.
    sottolobi: list = field(default_factory=list)

    @property
    def box(self) -> list:
        return [min(r[0] for r in self.righe), min(r[1] for r in self.righe),
                max(r[2] for r in self.righe), max(r[3] for r in self.righe)]


def _dentro(poligono, box) -> int:
    """Quanti dei 9 punti interni della riga cadono nel poligono."""
    import cv2

    contorno = np.asarray(poligono, np.float32)
    x0, y0, x1, y1 = box
    punti = [(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)
             for fx in (0.2, 0.5, 0.8) for fy in (0.25, 0.5, 0.75)]
    return sum(cv2.pointPolygonTest(contorno, (float(x), float(y)), False) >= 0
               for x, y in punti)


class _Maschera:
    """Il riempimento rasterizzato in coordinate locali, per misure di forma."""

    def __init__(self, profili, margine=4):
        import cv2

        punti = np.concatenate([np.asarray(p, np.float64) for p in profili])
        self.ox = int(np.floor(punti[:, 0].min())) - margine
        self.oy = int(np.floor(punti[:, 1].min())) - margine
        w = int(np.ceil(punti[:, 0].max())) - self.ox + margine + 1
        h = int(np.ceil(punti[:, 1].max())) - self.oy + margine + 1
        self.mask = np.zeros((h, w), np.uint8)
        for p in profili:
            locale = np.round(np.asarray(p, np.float64) - [self.ox, self.oy]).astype(np.int32)
            cv2.fillPoly(self.mask, [locale], 1)

    def larghezza(self, y, x) -> int:
        """Lunghezza della corsa piena che contiene (x, y); 0 se il punto e' fuori."""
        yi, xi = int(round(y)) - self.oy, int(round(x)) - self.ox
        h, w = self.mask.shape
        if not (0 <= yi < h and 0 <= xi < w) or not self.mask[yi, xi]:
            return 0
        riga = self.mask[yi]
        a = xi
        while a > 0 and riga[a - 1]:
            a -= 1
        b = xi
        while b < w - 1 and riga[b + 1]:
            b += 1
        return b - a + 1


def _gruppi_per_salto(righe, altezza):
    """Righe ordinate per y divise dove il salto supera `SALTO_FRA_LOBI` righe."""
    gruppi = [[righe[0]]]
    fondo = righe[0][0][3]
    for riga in righe[1:]:
        if riga[0][1] - fondo > SALTO_FRA_LOBI * altezza:
            gruppi.append([riga])
        else:
            gruppi[-1].append(riga)
        fondo = max(fondo, riga[0][3])
    return gruppi


def _box(gruppo):
    return [min(r[0][0] for r in gruppo), min(r[0][1] for r in gruppo),
            max(r[0][2] for r in gruppo), max(r[0][3] for r in gruppo)]


def _larghezza_massima(maschera, box) -> int:
    x = (box[0] + box[2]) / 2
    return max(maschera.larghezza(y, x) for y in np.arange(box[1], box[3] + 1, 4))


def _strozzatura(maschera, sopra, sotto):
    """La y del punto piu' stretto fra i due gruppi, o None se il contorno non si stringe.

    Si misura il minimo su tutto il salto, non a meta': a p8 la strozzatura sotto
    "SAME." sta a un terzo del salto, e a meta' il contorno e' gia' largo.
    Ogni lobo si misura nel suo punto piu' largo: un lobo piccolo in cima e' stretto
    al centro delle sue righe quanto la strozzatura stessa.
    """
    a, b = _box(sopra), _box(sotto)
    x = ((a[0] + a[2]) / 2 + (b[0] + b[2]) / 2) / 2
    ys = np.arange(min(a[3], b[1]), max(a[3], b[1]) + 1, 4)
    if len(ys) == 0:
        return None
    larghezze = [maschera.larghezza(y, x) for y in ys]
    i = int(np.argmin(larghezze))
    lobi = [_larghezza_massima(maschera, a), _larghezza_massima(maschera, b)]
    if min(lobi) <= 0 or larghezze[i] >= STROZZATURA * min(lobi):
        return None
    return float(ys[i])


def lobi_del_blocco(render: dict, righe: list) -> list[Lobo] | None:
    """I lobi di un blocco immagine, o None se il blocco ne ha uno solo.

    `righe` e' la lista (box_px, testo) delle righe proprie del blocco.
    """
    profili = [p for p in (render.get("fill_poly") or []) if len(p) >= 3]
    righe = sorted(((list(map(float, b)), t) for b, t in righe if b and len(b) == 4),
                   key=lambda r: (r[0][1], r[0][0]))
    if len(righe) < 2 or not profili:
        return None

    if len(profili) >= 2:
        # compound: ogni riga al profilo che la contiene di piu'
        per_profilo = {}
        for riga in righe:
            punteggi = [_dentro(p, riga[0]) for p in profili]
            migliore = max(range(len(profili)), key=lambda i: punteggi[i])
            if punteggi[migliore] == 0:
                return None  # una riga fuori da tutti: la forma non e' quella dei lobi
            per_profilo.setdefault(migliore, []).append(riga)
        if len(per_profilo) < 2:
            return None
        lobi = [Lobo(righe=[r[0] for r in gruppo],
                     testo_en=" ".join(r[1] for r in gruppo),
                     profili=[profili[i]])
                for i, gruppo in per_profilo.items()]
        lobi.sort(key=lambda lobo: (lobo.box[1], lobo.box[0]))
        fini = []
        for i, gruppo in per_profilo.items():
            fini.extend(_per_strozzatura([profili[i]], gruppo) or
                        [Lobo(righe=[r[0] for r in gruppo],
                              testo_en=" ".join(r[1] for r in gruppo),
                              profili=[profili[i]])])
        fini.sort(key=lambda lobo: (lobo.box[1], lobo.box[0]))
        if len(fini) > len(lobi):
            lobi[0].sottolobi = fini
        return lobi

    return _per_strozzatura(profili, righe)


def _per_strozzatura(profili, righe) -> list[Lobo] | None:
    """Lobi di un profilo solo: gruppi di righe separati da una strozzatura del contorno."""
    if len(righe) < 2:
        return None
    altezza = median(r[0][3] - r[0][1] for r in righe)
    gruppi = _gruppi_per_salto(righe, altezza)
    if len(gruppi) < 2:
        return None
    maschera = _Maschera(profili)
    uniti, strozzature = [gruppi[0]], []
    for gruppo in gruppi[1:]:
        y = _strozzatura(maschera, uniti[-1], gruppo)
        if y is not None:
            uniti.append(gruppo)
            strozzature.append(y)
        else:
            uniti[-1] = uniti[-1] + gruppo
    if len(uniti) < 2:
        return None
    ys = [p[1] for p in profili[0]]
    confini = [min(ys)] + strozzature + [max(ys)]
    return [Lobo(righe=[r[0] for r in gruppo],
                 testo_en=" ".join(r[1] for r in gruppo),
                 profili=profili,
                 banda=(confini[i], confini[i + 1]))
            for i, gruppo in enumerate(uniti)]


def dividi_traduzione(lobi: list[Lobo], traduzione: str) -> list[str] | None:
    """Il pezzo di traduzione di ogni lobo, o None se le frasi non si allineano."""
    conteggi = [len(frasi(lobo.testo_en, ocr=True)) for lobo in lobi]
    italiane = frasi(traduzione)
    if any(c == 0 for c in conteggi) or sum(conteggi) != len(italiane):
        return None
    pezzi, i = [], 0
    for c in conteggi:
        pezzi.append(" ".join(italiane[i:i + c]))
        i += c
    return pezzi


def riquadro_di_scrittura(lobo: Lobo, vietati=()) -> list:
    """Il rettangolo px in cui scrivere il lobo: dalle sue righe, allargato dentro la forma.

    `vietati` sono i riquadri degli altri blocchi che vengono scritti: il lobo
    cresce fino a loro e non oltre (a p16 il lobo "TOO OBVIOUS?" confina con il
    balloon "I THINK TAL PRIME...", e il profilo OCR li copre entrambi).
    """
    import cv2

    from comics_translator.disegno.image_overlay import grow_in_mask, shrink_to_mask

    maschera = _Maschera(lobo.profili)
    altezza = median(r[3] - r[1] for r in lobo.righe)
    aria = max(3, int(round(ARIA_CONTORNO * altezza)))
    mask = cv2.erode(maschera.mask, np.ones((2 * aria + 1, 2 * aria + 1), np.uint8))
    h, w = mask.shape
    if lobo.banda is not None:
        y0 = int(np.ceil(lobo.banda[0])) - maschera.oy + aria
        y1 = int(np.floor(lobo.banda[1])) - maschera.oy - aria
        mask[:max(0, y0)] = 0
        mask[max(0, y1):] = 0
    for v in vietati or ():
        vx0 = max(0, int(v[0]) - maschera.ox - aria)
        vy0 = max(0, int(v[1]) - maschera.oy - aria)
        vx1 = min(w, int(np.ceil(v[2])) - maschera.ox + aria)
        vy1 = min(h, int(np.ceil(v[3])) - maschera.oy + aria)
        if vx1 > vx0 and vy1 > vy0:
            mask[vy0:vy1, vx0:vx1] = 0
    x0, y0, x1, y1 = lobo.box
    locale = [max(0, int(x0) - maschera.ox), max(0, int(y0) - maschera.oy),
              min(w, int(np.ceil(x1)) - maschera.ox), min(h, int(np.ceil(y1)) - maschera.oy)]
    if locale[2] - locale[0] < 8 or locale[3] - locale[1] < 8:
        return lobo.box
    ristretto = shrink_to_mask(mask, locale, step=2)
    cresciuto = grow_in_mask(mask, ristretto, step=4, max_grow=int(max(w, h)))
    return [cresciuto[0] + maschera.ox, cresciuto[1] + maschera.oy,
            cresciuto[2] + maschera.ox, cresciuto[3] + maschera.oy]


def _dentro_un_vietato(box, vietati) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(v[0] <= cx <= v[2] and v[1] <= cy <= v[3] for v in vietati or ())


def prepara(render: dict, righe: list, traduzione: str, vietati=()) -> list[dict]:
    """I lobi pronti da disegnare: [{box_px, testo, profili}], o [] se non si divide.

    Se le righe di un lobo stanno dentro il riquadro di un altro blocco, quel
    lobo e' del balloon accanto (l'OCR ha messo "NO" di "NO ONE MUST KNOW" nel
    blocco sopra, imperial-guardians p25): non si divide.
    """
    lobi = lobi_del_blocco(render, righe)
    if not lobi:
        return []
    if any(_dentro_un_vietato(lobo.box, vietati) for lobo in lobi):
        return []
    pezzi = None
    fini = lobi[0].sottolobi
    if fini:
        pezzi = dividi_traduzione(fini, traduzione)
        if pezzi is not None:
            lobi = fini
    if pezzi is None:
        pezzi = dividi_traduzione(lobi, traduzione)
    if pezzi is None:
        return []
    return [{"box_px": riquadro_di_scrittura(lobo, vietati), "testo": pezzo,
             "profili": lobo.profili}
            for lobo, pezzo in zip(lobi, pezzi)]


def righe_del_blocco(entry: dict, page: dict) -> list:
    """(box_px, testo) delle righe proprie di un blocco, dall'inventario sigillato."""
    inventario = (page or {}).get("sorgente") or {}
    per_id = {r["id"]: r for r in inventario.get("righe", []) if isinstance(r, dict)}
    risultato = []
    for ident in (entry.get("sorgente") or {}).get("righe") or []:
        riga = per_id.get(ident)
        if riga and riga.get("box_px"):
            risultato.append((riga["box_px"], riga.get("testo_normalizzato") or ""))
    return risultato
