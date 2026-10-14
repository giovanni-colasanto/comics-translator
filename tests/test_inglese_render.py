"""Inglese sulla pagina resa: classificazione della lingua e righe dichiarate.

Il gate `inglese_residuo` aveva segnalato 27 delle 170 pagine corrette a mano
sui FATTO, perche' guarda solo righe gia' acquisite dall'OCR della pipeline.
Questo controllo rilegge il PDF finito: qui si fissa la parte deterministica,
senza OCR reale (l'OCR si passa come funzione).
"""

import unittest

import numpy as np

from comics_translator.disegno.image_overlay import OcrLine
from comics_translator.verifica import inglese_render as ir


class LinguaRigaTest(unittest.TestCase):
    def test_battute_inglesi(self):
        for testo in ("WHAT DO YOU WANT?", "I DON'T KNOW WHERE HE IS.", "YOU DARE?!",
                      "PEOPLE DIED IN THAT POLICE STATION TODAY"):
            self.assertEqual(ir.lingua_riga(testo), "en", testo)

    def test_battute_italiane(self):
        for testo in ("COSA VUOI?", "NON SO DOVE SIA.", "È TUTTO ANDATO STORTO",
                      "PERCHE' NON VUOI TORNARE A CASA?"):
            self.assertEqual(ir.lingua_riga(testo), "it", testo)

    def test_parole_comuni_alle_due_lingue_non_fanno_inglese(self):
        """I falsi positivi misurati sui PDF resi di spiderman-13 e batman-2."""
        for testo in ("ALL'OMICIDIO!", "LASCIAMI IN", "VOLO,IN PERFETTA", "IN OGNI",
                      "Nell'istante in", "PROPRIO IN", "OH!ASPETTA,", "I RAGAZZI",
                      "ABBANDONATI", "DIVENTACARNE", "VIVERELAVITA", "INTELLIGENZAE",
                      "COMINCIANDOA", "AMISUREDIPROLUNGAMENTO", "SATELLITARE AZIENDALE"):
            self.assertNotEqual(ir.lingua_riga(testo), "en", testo)

    def test_suoni_nomi_e_numeri_non_si_giudicano(self):
        for testo in ("BLAM", "KRAKOOM!", "SPIDER-MAN", "#$%&", "1963", "", "HNNGH"):
            self.assertIsNone(ir.lingua_riga(testo), testo)

    def test_l_ocr_senza_spazi_si_riconosce_lo_stesso(self):
        self.assertEqual(ir.lingua_riga("WHATDOYOUTHINKYOUAREDOING"), "en")


class RigheIngletsiTest(unittest.TestCase):
    def ocr(self, *righe):
        def leggi(_img, _conf):
            return [OcrLine(x0, y0, x1, y1, testo, 0.95) for testo, (x0, y0, x1, y1) in righe]
        return leggi

    def test_le_righe_dentro_un_blocco_saltato_sono_dichiarate(self):
        pagina = {"numero": 1, "blocchi": [
            {"id": "b", "salta": True, "_render": {"tipo": "immagine", "scale": 0.5,
                                                   "box_px": [0, 0, 200, 100]}},
        ]}
        img = np.zeros((10, 10, 3), np.uint8)
        righe = ir.righe_inglesi(img, 144, pagina, ocr=self.ocr(
            ("THIS IS NOT FOR YOU", (10, 10, 150, 40)),     # dentro il blocco saltato
            ("WHERE ARE YOU GOING?", (300, 300, 500, 340)),  # fuori: inglese vero
            ("DOVE STAI ANDANDO?", (300, 400, 500, 440)),    # italiano
        ))
        self.assertEqual([(r["testo"], r["dichiarata"]) for r in righe],
                         [("THIS IS NOT FOR YOU", True), ("WHERE ARE YOU GOING?", False)])
        self.assertEqual(righe[1]["box_pt"], [150.0, 150.0, 250.0, 170.0])

    def test_una_battuta_saltata_non_e_dichiarata(self):
        """Prova A2 (spidey-63-9): i dialoghi saltati nascondevano 177 righe inglesi."""
        def blocco(bid, natura, testo, x0):
            return {"id": bid, "salta": True, "natura": natura, "originale": "HEY YOU",
                    "testo": testo,
                    "_render": {"tipo": "immagine", "scale": 1.0, "box_px": [x0, 0, x0 + 100, 50]}}
        pagina = {"numero": 1, "blocchi": [
            blocco("logo", "grafica", "HEY YOU", 0),
            blocco("dialogo", "dialogo", "Ehi, tu", 200),
            blocco("vecchio", None, "HEY YOU", 400),
            blocco("tradotto", None, "Ehi, tu", 600),
        ]}
        img = np.zeros((10, 10, 3), np.uint8)
        righe = ir.righe_inglesi(img, 72, pagina, ocr=self.ocr(
            ("WHAT ARE YOU DOING", (10, 10, 90, 40)),
            ("WHAT ARE YOU DOING", (210, 10, 290, 40)),
            ("WHAT ARE YOU DOING", (410, 10, 490, 40)),
            ("WHAT ARE YOU DOING", (610, 10, 690, 40)),
        ))
        self.assertEqual([r["dichiarata"] for r in righe], [True, False, True, False])


class ComeLOriginaleTest(unittest.TestCase):
    """L'inglese si riconosce anche confrontandolo con l'originale dell'albo.

    Il lessico delle parole funzionali non vede le didascalie brevi: "YEARS AGO.
    NINE MILLION LIGHT-YEARS FROM EARTH." (superman-1 p3) ha un solo indizio
    ("from") e la regola ne chiede due, cosi' la bozza risultava pulita su 8
    pagine che invece erano da correggere. Ma l'inglese della pagina e' scritto
    nel piano: se la riga riletta combacia con `originale` e non con la
    traduzione, e' inglese rimasto - senza indovinare la lingua.
    """

    def ocr(self, *righe):
        def leggi(_img, _conf):
            return [OcrLine(x0, y0, x1, y1, testo, 0.95) for testo, (x0, y0, x1, y1) in righe]
        return leggi

    def pagina(self):
        return {
            "numero": 3,
            "blocchi": [{
                "id": "p3-b1", "salta": False, "natura": "didascalia",
                "originale": "YEARS AGO. NINE MILLION LIGHT-YEARS FROM EARTH.",
                "testo": "Anni fa. Nove milioni di anni luce dalla Terra.",
                "_render": {"tipo": "immagine", "scale": 1.0, "box_px": [0, 0, 10, 10]},
            }, {
                "id": "p3-b2", "salta": False, "natura": "dialogo",
                "originale": "KAL-EL, NO.", "testo": "Kal-El, no.",
                "_render": {"tipo": "immagine", "scale": 1.0, "box_px": [0, 0, 10, 10]},
            }],
            "sorgente": {"righe": [
                {"id": "r1", "testo_normalizzato": "THE PREVIOUS NIGHT."},
            ]},
        }

    def righe(self, *ocr):
        img = np.zeros((10, 10, 3), np.uint8)
        return ir.righe_inglesi(img, 72, self.pagina(), ocr=self.ocr(*ocr))

    def test_una_didascalia_breve_non_sfugge_piu(self):
        righe = self.righe(("YEARS AGO. NINE MILLION", (500, 500, 700, 540)))
        self.assertEqual([r["testo"] for r in righe], ["YEARS AGO. NINE MILLION"])

    def test_anche_una_riga_dell_inventario_vale_da_paragone(self):
        righe = self.righe(("THE PREVIOUS NIGHT.", (500, 500, 700, 540)))
        self.assertEqual(len(righe), 1)

    def test_un_nome_ripreso_dalla_traduzione_non_e_inglese_rimasto(self):
        """"KAL-EL" sta nell'originale e nella traduzione: non e' un residuo."""
        self.assertEqual(self.righe(("KAL-EL, NO.", (500, 500, 700, 540))), [])

    def test_la_traduzione_disegnata_non_si_segnala(self):
        self.assertEqual(self.righe(("Anni fa. Nove milioni", (500, 500, 700, 540))), [])

    def test_un_pezzo_corto_non_basta_al_confronto(self):
        """Poche lettere combaciano per caso: il confronto chiede una riga vera.

        ("THE" da solo resta comunque segnalato dal lessico: e' una parola
        funzionale inglese. Qui si fissa che il CONFRONTO non ci si appoggia.)
        """
        inglese, italiano = ir._testi_della_pagina(self.pagina())
        self.assertFalse(ir._come_l_originale("THE", inglese, italiano))
        self.assertFalse(ir._come_l_originale("NO.", inglese, italiano))
        self.assertTrue(ir._come_l_originale("NINE MILLION", inglese, italiano))


if __name__ == "__main__":
    unittest.main()
