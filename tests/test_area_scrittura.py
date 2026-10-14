"""Il testo si scrive dentro il balloon, non negli angoli del rettangolo delle righe.

Prova A1 su imperial-guardians (15 settembre 2026): su quasi ogni balloon ovale
di p11 e p15 le righe italiane passavano sopra il contorno nero, perche' il
riquadro di scrittura era il rettangolo delle righe inglesi e l'italiano, piu'
lungo, lo riempiva fino agli angoli.
"""

import math
import unittest

import pymupdf as fitz

from comics_translator.disegno import image_overlay as io
from comics_translator.disegno.pdf_translate_overlay import fit, load_fontset


def ovale(cx, cy, rx, ry, n=48):
    return [[cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)]
            for i in range(n)]


class AreaDiScritturaTest(unittest.TestCase):
    def test_su_un_ovale_gli_angoli_rientrano_nel_balloon(self):
        box = [0, 0, 200, 120]
        profilo = ovale(100, 60, 115, 70)
        area = io.area_di_scrittura(box, 1.0, [profilo], inset=0)
        self.assertLess(area.width, 200)
        import cv2
        import numpy as np
        contorno = np.asarray(profilo, np.float32)
        for x, y in ((area.x0, area.y0), (area.x1, area.y0), (area.x0, area.y1), (area.x1, area.y1)):
            self.assertGreaterEqual(cv2.pointPolygonTest(contorno, (x, y), False), 0)
        self.assertAlmostEqual((area.x0 + area.x1) / 2, 100, delta=0.01)

    def test_una_didascalia_rettangolare_non_si_stringe(self):
        box = [10, 10, 110, 60]
        profilo = [[4, 4], [116, 4], [116, 66], [4, 66]]
        area = io.area_di_scrittura(box, 1.0, [profilo], inset=0)
        self.assertEqual(area, fitz.Rect(10, 10, 110, 60))

    def test_senza_profili_resta_il_riquadro(self):
        self.assertEqual(io.area_di_scrittura([0, 0, 50, 20], 2.0, None, inset=0),
                         fitz.Rect(0, 0, 100, 40))

    def test_con_il_centro_fuori_dal_profilo_resta_il_riquadro(self):
        box = [0, 0, 100, 100]
        profilo_a_c = [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]]
        self.assertEqual(io.area_di_scrittura(box, 1.0, [profilo_a_c], inset=0),
                         fitz.Rect(0, 0, 100, 100))

    def test_non_si_stringe_oltre_la_riduzione_minima(self):
        area = io.area_di_scrittura([0, 0, 100, 100], 1.0, [ovale(50, 50, 10, 10)], inset=0)
        self.assertAlmostEqual(area.width, 100 * io.RIDUZIONE_MINIMA, delta=0.01)


class MisuraNelBalloonTest(unittest.TestCase):
    def setUp(self):
        self.font = load_fontset("arial").pick(True, False)

    def test_un_testo_corto_entra_nell_area_stretta(self):
        box = [0, 0, 200, 120]
        profilo = ovale(100, 60, 115, 70)
        area, misura = io.misura_nel_balloon(self.font, "CIAO", box, 1.0, [profilo], 12, 4.5, fit)
        self.assertIsNotNone(misura)
        self.assertLess(area.width, 197)

    def test_se_nell_area_stretta_non_entra_si_torna_al_riquadro(self):
        """Mai rifiutare una traduzione che prima entrava: sfiora il contorno, ma c'e'."""
        box = [0, 0, 80, 30]
        profilo = ovale(40, 15, 41, 16)
        testo = "UNA BATTUTA ABBASTANZA LUNGA DA RIEMPIRE"
        pieno = io.area_di_scrittura(box, 1.0, None)
        if io._misura(self.font, testo, pieno, 9, 4.5, fit) is None:
            self.skipTest("il testo di prova non entra nemmeno nel riquadro pieno")
        area, misura = io.misura_nel_balloon(self.font, testo, box, 1.0, [profilo], 9, 4.5, fit)
        self.assertIsNotNone(misura)


class CandidatoScartatoTest(unittest.TestCase):
    def test_didascalie_di_una_riga_senza_punto(self):
        for testo in ("NINE MONTHS AGO", "DEADREEFSALIXOUTWORLD", "Eight months ago..."):
            self.assertTrue(io.candidato_scartato(testo, 1), testo)

    def test_suoni_corti_restano_fuori(self):
        for testo in ("BLAM", "KRAKOOM!", "BLAMBLAMBLAM", "HNNGH", "#$%&", "1978"):
            self.assertFalse(io.candidato_scartato(testo, 1), testo)


if __name__ == "__main__":
    unittest.main()
