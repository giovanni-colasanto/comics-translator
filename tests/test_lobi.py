"""Balloon a piu' lobi: righe per lobo, traduzione per frasi, riquadro dentro la forma.

Casi della prova A1 su imperial-guardians (16 settembre 2026): "SAME. /
INVESTIGATING A POTENTIAL THREAT... / SKRULLS." a p8, "SURE. / I MAIM TO
PLEASE." a p27, i compound di p7 e p16. Il renderer scriveva la battuta intera a
cavallo della strozzatura; il `.finale` riscriveva i lobi a mano.
"""

import math
import unittest

import cv2
import numpy as np

from comics_translator.disegno import lobi as L


def ovale(cx, cy, rx, ry, n=64):
    return [[cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)]
            for i in range(n)]


def unione(*poligoni, lato=1200):
    """Il contorno esterno dell'unione di piu' poligoni, come lo darebbe l'OCR."""
    mask = np.zeros((lato, lato), np.uint8)
    for p in poligoni:
        cv2.fillPoly(mask, [np.round(np.asarray(p)).astype(np.int32)], 1)
    contorni, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[float(x), float(y)] for x, y in max(contorni, key=cv2.contourArea)[:, 0]]


def riga(cx, y, larghezza, testo, altezza=40):
    return ([cx - larghezza / 2, y, cx + larghezza / 2, y + altezza], testo)


class FrasiTest(unittest.TestCase):
    def test_punti_interrogativi_puntini_e_trattini(self):
        self.assertEqual(L.frasi("Hai detto Mondo Trono? Se qualcuno e'-- troppo ovvio? E poi..."),
                         ["Hai detto Mondo Trono?", "Se qualcuno e'--", "troppo ovvio?", "E poi..."])

    def test_il_resto_senza_punto_e_una_frase(self):
        self.assertEqual(L.frasi("SURE. I MAIM TO PLEASE,"), ["SURE.", "I MAIM TO PLEASE,"])

    def test_puntini_iniziali_restano_con_la_frase(self):
        self.assertEqual(L.frasi("...E poi anche l'unione, prima che nasca."),
                         ["...E poi anche l'unione, prima che nasca."])

    def test_ocr_senza_spazio_dopo_il_punto(self):
        self.assertEqual(len(L.frasi("MYLORD KL'RT.ITIS MERELY A THING.", ocr=True)), 2)
        # sulla traduzione no: li' lo spazio c'e' sempre, e "Kl'rt.It" non esiste
        self.assertEqual(len(L.frasi("MYLORD KL'RT.ITIS MERELY A THING.")), 1)

    def test_sigle_non_spezzano(self):
        self.assertEqual(L.frasi("S.H.I.E.L.D. IS HERE.", ocr=True), ["S.H.I.E.L.D.", "IS HERE."])
        self.assertEqual(len(L.frasi("CALL S.H.I.E.L.D.NOW", ocr=True)), 1)


class LobiDelBloccoTest(unittest.TestCase):
    def test_due_ovali_uniti_da_una_strozzatura_sono_due_lobi(self):
        profilo = unione(ovale(500, 300, 260, 120), ovale(500, 520, 240, 130))
        righe = [riga(500, 240, 300, "YOUR WISECRACK GAME"), riga(500, 280, 320, "IS FEEBLE, DARKHAWK."),
                 riga(500, 520, 300, "AND YOUR HAND-TO-HAND"), riga(500, 560, 200, "IS SLOPPY.")]
        lobi = L.lobi_del_blocco({"fill_poly": [profilo]}, righe)
        self.assertIsNotNone(lobi)
        self.assertEqual([len(lobo.righe) for lobo in lobi], [2, 2])
        self.assertLess(lobi[0].banda[1], 520)
        self.assertGreater(lobi[0].banda[1], 320)

    def test_un_ovale_con_un_capoverso_resta_un_lobo(self):
        """Il salto c'e', la strozzatura no: e' un balloon solo con due frasi."""
        profilo = ovale(500, 400, 320, 260)
        righe = [riga(500, 250, 300, "CAPTAIN"), riga(500, 290, 300, "MARVEL."),
                 riga(500, 400, 300, "WHAT ARE YOU"), riga(500, 440, 300, "DOING HERE?")]
        self.assertIsNone(L.lobi_del_blocco({"fill_poly": [profilo]}, righe))

    def test_compound_ogni_riga_al_suo_profilo(self):
        alto, basso = ovale(400, 200, 200, 90), ovale(700, 500, 220, 120)
        righe = [riga(700, 480, 250, "CAPTAIN MARVEL. WHAT?"), riga(400, 190, 250, "YEAH, RUN...")]
        lobi = L.lobi_del_blocco({"fill_poly": [basso, alto]}, righe)
        self.assertEqual([lobo.testo_en for lobo in lobi], ["YEAH, RUN...", "CAPTAIN MARVEL. WHAT?"])
        self.assertEqual(lobi[0].profili, [alto])

    def test_una_riga_fuori_da_tutti_i_profili_non_si_indovina(self):
        righe = [riga(400, 190, 200, "A."), riga(1000, 1000, 100, "B.")]
        self.assertIsNone(L.lobi_del_blocco(
            {"fill_poly": [ovale(400, 200, 200, 90), ovale(700, 500, 220, 120)]}, righe))


class DividiTraduzioneTest(unittest.TestCase):
    def lobi(self, *testi):
        return [L.Lobo(righe=[[0, 0, 10, 10]], testo_en=t) for t in testi]

    def test_frasi_allineate_ai_lobi(self):
        self.assertEqual(
            L.dividi_traduzione(self.lobi("SAME.", "INVESTIGATING. I'LL SAY IT.", "SKRULLS."),
                                "Idem. A indagare. Lo dico io. Skrull."),
            ["Idem.", "A indagare. Lo dico io.", "Skrull."])

    def test_se_i_conti_non_tornano_non_si_divide(self):
        self.assertIsNone(L.dividi_traduzione(self.lobi("CAPTAIN MARVEL.", "WHAT?"),
                                              "Captain Marvel, che ci fai qui?"))


class PreparaTest(unittest.TestCase):
    def setUp(self):
        self.profilo = unione(ovale(500, 300, 260, 120), ovale(500, 520, 240, 130))
        self.righe = [riga(500, 260, 300, "SURE."), riga(500, 540, 300, "I MAIM TO PLEASE.")]

    def test_un_riquadro_per_lobo_dentro_la_sua_banda(self):
        pronti = L.prepara({"fill_poly": [self.profilo]}, self.righe, "Certo. Felice di mutilare.")
        self.assertEqual([p["testo"] for p in pronti], ["Certo.", "Felice di mutilare."])
        alto, basso = pronti[0]["box_px"], pronti[1]["box_px"]
        self.assertLessEqual(alto[3], basso[1])
        contorno = np.asarray(self.profilo, np.float32)
        for box in (alto, basso):
            for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
                self.assertGreaterEqual(cv2.pointPolygonTest(contorno, (float(x), float(y)), False), 0)
        # il riquadro cresce oltre le righe inglesi: l'italiano e' piu' lungo
        self.assertGreater(basso[2] - basso[0], 300)

    def test_lobo_con_le_righe_dentro_un_altro_blocco_non_si_divide(self):
        """p25: "NO" di "NO ONE MUST KNOW" messo dall'OCR nel balloon sopra."""
        vietato = [300, 500, 700, 600]
        self.assertEqual(
            L.prepara({"fill_poly": [self.profilo]}, self.righe, "Certo. Felice.", [vietato]), [])

    def test_il_lobo_non_cresce_dentro_un_altro_blocco(self):
        vietato = [640, 380, 900, 700]
        pronti = L.prepara({"fill_poly": [self.profilo]}, self.righe, "Certo. Felice.", [vietato])
        self.assertLess(pronti[1]["box_px"][2], vietato[0])


class MisuraLobiTest(unittest.TestCase):
    def test_un_corpo_comune_e_un_rettangolo_per_lobo(self):
        from comics_translator.disegno import image_overlay as io
        from comics_translator.disegno.pdf_translate_overlay import fit, load_fontset

        font = load_fontset("arial").pick(True, False)
        profilo = unione(ovale(500, 300, 260, 120), ovale(500, 520, 240, 130))
        pronti = L.prepara({"fill_poly": [profilo]},
                           [riga(500, 260, 300, "SURE."), riga(500, 540, 300, "I MAIM TO PLEASE.")],
                           "Certo. Felice di mutilare.")
        misure = io.misura_lobi(font, pronti, 0.18, 12, 4.5, fit)
        self.assertEqual(len(misure), 2)
        self.assertEqual(misure[0][1][0], misure[1][1][0])
        self.assertLessEqual(misure[0][0].y1, misure[1][0].y0)
        self.assertTrue(io.entra_il_job(font, "CERTO. FELICE DI MUTILARE.", [240, 180, 760, 690],
                                        0.18, [profilo], 12, 4.5, fit, lobi=pronti))


if __name__ == "__main__":
    unittest.main()
