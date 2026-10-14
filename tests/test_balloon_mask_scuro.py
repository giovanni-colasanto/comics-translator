"""Su un balloon scuro il pixel piu' chiaro e' il TESTO, non il fondo.

Carnage parla in didascalie a fondo rosso pieno con lettering bianco. Cercare
il seme del flood-fill come "un pixel chiaro vicino alla riga" ci fa cadere
dentro una lettera: il riempimento si ferma sul contorno del glifo, la maschera
esce grande quanto una `O` e `shrink_to_mask` restituisce un riquadro di pochi
pixel. Il blocco viene poi scartato come illeggibile e sulla tavola resta
l'inglese.

Misurato su carnage-5 il 9 settembre 2026: 53 blocchi tradotti e mai disegnati,
tutti con `area_ratio` 0.0, 41 dei quali `light_text: True`. Riquadro mediano
31x37 px contro 548x215 px dei blocchi disegnati bene.

Il seme va quindi cercato sul colore del balloon (`bg`), che e' l'unica cosa che
distingue il fondo dal testo in tutti e due i versi.
"""

import unittest

import numpy as np

from comics_translator.disegno.image_overlay import _balloon_mask, _near_balloon, _seme_di_fondo


def didascalia_rossa() -> tuple:
    """Riquadro rosso pieno con lettering bianco su vignetta chiara.

    E' la forma delle didascalie di Carnage: fondo scuro saturo, testo chiaro,
    contorno piu' scuro del fondo.
    """
    h, w = 400, 400
    img = np.full((h, w, 3), 200, dtype=np.uint8)   # vignetta chiara attorno
    y0, y1, x0, x1 = 150, 260, 80, 320
    img[y0:y1, x0:x1] = (134, 45, 48)               # fondo della didascalia
    img[y0:y0 + 4, x0:x1] = (40, 12, 14)            # contorno
    img[y1 - 4:y1, x0:x1] = (40, 12, 14)
    img[y0:y1, x0:x0 + 4] = (40, 12, 14)
    img[y0:y1, x1 - 4:x1] = (40, 12, 14)
    # due righe di lettering BIANCO: sono i pixel piu' chiari del riquadro
    img[175:195, 110:290] = 255
    img[210:230, 110:290] = 255
    box = [110, 175, 290, 230]
    righe = [[110, 175, 290, 195], [110, 210, 290, 230]]
    bg = (134, 45, 48)
    return img, box, bg, righe


class SemeSuBalloonScuroTest(unittest.TestCase):
    def test_il_seme_non_cade_sul_lettering_chiaro(self):
        img, box, bg, _ = didascalia_rossa()
        # centro della prima riga: cade in pieno sul bianco del testo
        seme = _seme_di_fondo(img, 200, 185, bg)
        self.assertIsNotNone(seme, "un fondo rosso deve dare un seme")
        x, y = seme
        colore = img[y, x]
        self.assertLess(int(colore.min()), 140,
                        "il seme e' finito sul lettering bianco invece che sul fondo")

    def test_su_balloon_bianco_il_seme_resta_valido(self):
        """La correzione non deve rompere il caso normale."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[40:160, 40:160] = 255
        img[90:110, 60:140] = 0          # testo nero
        seme = _seme_di_fondo(img, 100, 100, (255, 255, 255))
        self.assertIsNotNone(seme)
        x, y = seme
        self.assertEqual(int(img[y, x].min()), 255)


class MascheraSuBalloonScuroTest(unittest.TestCase):
    def test_la_maschera_copre_il_riquadro_e_non_un_glifo(self):
        img, box, bg, righe = didascalia_rossa()
        mask, (ox, oy) = _balloon_mask(img, box, bg, righe=righe)
        self.assertIsNotNone(mask, "il riquadro rosso va rilevato")

        piena = np.zeros((400, 400), dtype=bool)
        piena[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
        # il fondo vero, senza contorno e senza lettering
        yy, xx = np.ogrid[:400, :400]
        fondo = ((yy >= 154) & (yy < 256) & (xx >= 84) & (xx < 316))
        fondo = fondo & ~((yy >= 175) & (yy < 195) & (xx >= 110) & (xx < 290))
        fondo = fondo & ~((yy >= 210) & (yy < 230) & (xx >= 110) & (xx < 290))
        coperta = float((piena & fondo).sum()) / float(fondo.sum())
        self.assertGreater(coperta, 0.9,
                           "la maschera copre un glifo invece del fondo della didascalia")

    def test_non_esce_dal_riquadro_sulla_vignetta(self):
        img, box, bg, righe = didascalia_rossa()
        mask, (ox, oy) = _balloon_mask(img, box, bg, righe=righe)
        piena = np.zeros((400, 400), dtype=bool)
        piena[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
        yy, xx = np.ogrid[:400, :400]
        dentro = (yy >= 150) & (yy < 260) & (xx >= 80) & (xx < 320)
        fuori = int((piena & ~dentro).sum())
        self.assertLess(fuori, 200,
                        "il riempimento e' uscito sulla vignetta attorno")


class UnioneMaiPeggioDellaSogliaTest(unittest.TestCase):
    """Il flood-fill puo' solo AGGIUNGERE alla soglia, mai togliere.

    E' l'invariante che tiene: il riempimento serve a recuperare il gradiente
    che la soglia perde, ma se si ferma presto (fondo rumoroso, seme sfortunato)
    la soglia deve restare. Tenendo il solo flood-fill, su imperial-3 il 9
    settembre 2026 collassavano 25 riquadri su 81 - fino a 2x34 px - contro i 4
    del comportamento storico.
    """

    def _soglia(self, roi, bg, tol):
        return (np.abs(roi.astype(np.float32) - np.array(bg, dtype=np.float32)
                       ).max(axis=2) < tol).astype(np.uint8)

    def test_contiene_sempre_la_soglia(self):
        casi = []
        img, box, bg, righe = didascalia_rossa()
        casi.append((img, box, bg, righe))
        # balloon chiaro su fondo rumoroso: il riempimento fatica, la soglia no
        rng = np.random.default_rng(7)
        img2 = rng.integers(0, 90, (300, 300, 3), dtype=np.uint8)
        img2[80:220, 60:240] = 250
        img2[140:160, 90:210] = 0
        casi.append((img2, [90, 140, 210, 160], (250, 250, 250),
                     [[90, 140, 210, 160]]))

        for img, box, bg, righe in casi:
            tol = 40.0
            x0, y0 = max(0, box[0] - 160), max(0, box[1] - 160)
            x1 = min(img.shape[1], box[2] + 160)
            y1 = min(img.shape[0], box[3] + 160)
            roi = img[y0:y1, x0:x1]
            unione = _near_balloon(roi, (x0, y0), [box] + list(righe), bg, tol)
            soglia = self._soglia(roi, bg, tol)
            mancanti = int((soglia & ~unione.astype(bool)).sum())
            self.assertEqual(mancanti, 0,
                             "il riempimento ha tolto pixel che la soglia teneva")


if __name__ == "__main__":
    unittest.main()
