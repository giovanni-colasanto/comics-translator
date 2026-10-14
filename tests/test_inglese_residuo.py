"""Il gate che dichiara l'inglese vero rimasto sulla pagina resa.

`structure_audit.righe_scoperte` trova le righe originali leggibili che nessun
blocco rivendica dentro un'area riscritta: e' inglese non cancellato. Ma "leggibile"
li' e' solo un conteggio di lettere, e ci finiscono anche i suoni disegnati e la
spazzatura OCR. Questo gate separa il testo vero - da recuperare - dal resto.

Casi reali (deadpool-2 finale, 8 settembre 2026): fra le tre righe scoperte
leggibili - 'WHAT?!', 'mvt', 'KR-CU' - solo 'WHAT?!' e' inglese vero.
"""

import unittest

from comics_translator.verifica.segnalazioni import inglese_residuo, sembra_testo_reale


class SembraTestoRealeTest(unittest.TestCase):
    def test_parole_vere_anche_con_una_sola_vocale(self):
        for testo in ("WHAT?!", "SPIDER-MAN", "NO MORE,HUH?", "WHO'S NEXT?",
                      "Peoplediedinthat", "BY ASGARD--"):
            self.assertTrue(sembra_testo_reale(testo), testo)

    def test_suoni_grawlix_e_frammenti_non_passano(self):
        for testo in ("mvt", "KR-CU", "HM", "Hnk", "HRRRRR", "###", "#%&",
                      "12.5", "", "a", "KR"):
            self.assertFalse(sembra_testo_reale(testo), testo)


class IngleseResiduoTest(unittest.TestCase):
    def test_tiene_solo_le_leggibili_che_sono_testo_vero(self):
        scoperte = [
            {"riga": "p6-r1", "testo": "WHAT?!", "coperta": 0.27, "leggibile": True},
            {"riga": "p7-r2", "testo": "mvt", "coperta": 0.0, "leggibile": True},
            {"riga": "p9-r3", "testo": "KR-CU", "coperta": 0.05, "leggibile": True},
        ]
        residui = inglese_residuo(scoperte, numero_pagina=6)
        self.assertEqual(len(residui), 1)
        self.assertEqual(residui[0]["riga"], "p6-r1")
        self.assertEqual(residui[0]["testo"], "WHAT?!")
        self.assertEqual(residui[0]["pagina"], 6)

    def test_ignora_le_righe_non_leggibili(self):
        scoperte = [{"riga": "x", "testo": "WHAT?!", "coperta": 0.2, "leggibile": False}]
        self.assertEqual(inglese_residuo(scoperte), [])

    def test_lista_vuota_o_none(self):
        self.assertEqual(inglese_residuo([]), [])
        self.assertEqual(inglese_residuo(None), [])


if __name__ == "__main__":
    unittest.main()
