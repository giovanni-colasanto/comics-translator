"""I balloon crema/gialli hanno un gradiente che il solo `bg` non copre.

Deadpool usa balloon color crema/giallo con il centro quasi bianco e i bordi
giallo saturo. Il `bg` memorizzato e' un campione singolo: se cade sulla parte
chiara, la tolleranza (40) centrata li' lascia fuori il giallo dei bordi
(differenza > 40 sul canale blu). La maschera del balloon si ferma attorno al
testo e l'inglese originale sbuca ai lati.

Misurato su deadpool-2 l'8 settembre 2026: ~45 balloon crema rotti su 10 pagine
(es. p15 "SEI UN PROBLEMA INTERESSANTE, THOR..." giallo b~180 con bg b~223).
La correzione centra la tolleranza sul colore VERO del balloon (mediana dei
pixel chiari nell'area del testo), non sul `bg`.
"""

import unittest

import numpy as np

from comics_translator.disegno.image_overlay import _balloon_mask


def balloon_giallo_gradiente() -> tuple:
    """Ellisse giallo su vignetta scura: centro chiaro, bordi giallo saturo.

    Il canale blu scende da 235 al centro a 185 al bordo: 185 dista 50 dal `bg`
    chiaro qui sotto, cioe' oltre la tolleranza. Con la vecchia maschera l'anello
    esterno resta scoperto.
    """
    h, w = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cx, cy, rx, ry = 200, 200, 150, 100
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    dentro = d <= 1.0
    blu = np.clip(235 - (d * 50), 185, 235)  # 235 al centro -> 185 al bordo
    img[..., 0] = np.where(dentro, 253, 0)
    img[..., 1] = np.where(dentro, 251, 0)
    img[..., 2] = np.where(dentro, blu.astype(np.uint8), 0)
    # una riga di testo nera al centro, come la troverebbe l'OCR
    img[190:215, 120:280] = 0
    box = [120, 190, 280, 215]
    righe = [[120, 190, 280, 215]]
    # bg campionato sulla parte chiara: e' proprio il caso che rompe la maschera
    bg = (253, 251, 233)
    return img, box, bg, righe


class BalloonCremaGradienteTest(unittest.TestCase):
    def test_la_maschera_copre_anche_i_bordi_gialli_saturi(self):
        img, box, bg, righe = balloon_giallo_gradiente()
        mask, (ox, oy) = _balloon_mask(img, box, bg, righe=righe)
        self.assertIsNotNone(mask, "il balloon giallo va rilevato")

        # l'ellisse vera, in coordinate della maschera
        yy, xx = np.ogrid[:400, :400]
        ellisse = ((xx - 200) / 150.0) ** 2 + ((yy - 200) / 100.0) ** 2 <= 1.0
        piena = np.zeros((400, 400), dtype=bool)
        piena[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
        coperta = float((piena & ellisse).sum()) / float(ellisse.sum())
        self.assertGreater(coperta, 0.9,
                           "il giallo saturo del bordo deve entrare nella maschera")

    def test_il_bordo_saturo_da_solo_non_resta_scoperto(self):
        """L'anello esterno (b~185) e' proprio quello che prima sbucava."""
        img, box, bg, righe = balloon_giallo_gradiente()
        mask, (ox, oy) = _balloon_mask(img, box, bg, righe=righe)
        yy, xx = np.ogrid[:400, :400]
        d = ((xx - 200) / 150.0) ** 2 + ((yy - 200) / 100.0) ** 2
        anello = (d > 0.7) & (d <= 0.98)   # solo il bordo giallo, senza il testo
        piena = np.zeros((400, 400), dtype=bool)
        piena[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
        coperta = float((piena & anello).sum()) / float(anello.sum())
        self.assertGreater(coperta, 0.9,
                           "l'anello giallo esterno resta fuori maschera come col bug")


if __name__ == "__main__":
    unittest.main()
