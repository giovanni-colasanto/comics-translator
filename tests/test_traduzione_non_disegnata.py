"""Il Gap E: la traduzione c'e', il renderer l'ha rifiutata, nessuno lo dice.

`_segna_illeggibili` riporta a `salta: true` il blocco il cui italiano non entra
nel riquadro, e scrive `_render.testo_illeggibile`. Quel `salta` resta nel
piano: la passata di rendering successiva non ci riprova e chiude senza
emettere nulla. Il lavoro del modello e' sul disco, pagato e corretto, e sulla
tavola resta l'inglese.

Misurato su carnage-5 il 9 settembre 2026: 53 blocchi persi cosi', mentre il
gate `inglese_residuo` ne dichiarava 4 - perche' l'audit misura le righe
scoperte DENTRO un'area riscritta, e un blocco tornato a `salta` non e' piu'
un'area riscritta.
"""

import unittest

from comics_translator.verifica.segnalazioni import traduzione_non_disegnata


def piano(blocchi):
    return {"pagine": [{"numero": 7, "blocchi": blocchi}]}


class TraduzioneNonDisegnataTest(unittest.TestCase):
    def test_dichiara_il_blocco_rifiutato_dal_renderer(self):
        persi = traduzione_non_disegnata(piano([
            {"id": "p7-b1", "testo": "Non e' un problema.", "salta": True,
             "_render": {"testo_illeggibile": True}},
        ]))
        self.assertEqual(len(persi), 1)
        self.assertEqual(persi[0]["id"], "p7-b1")
        self.assertEqual(persi[0]["pagina"], 7)
        self.assertEqual(persi[0]["testo"], "Non e' un problema.")

    def test_ignora_i_blocchi_saltati_per_scelta(self):
        """Onomatopee e insegne saltano di proposito: non sono una perdita."""
        persi = traduzione_non_disegnata(piano([
            {"id": "p7-b2", "testo": "KRAKOOM", "salta": True,
             "_render": {"salta_deciso": True}},
            {"id": "p7-b3", "testo": "Ciao.", "salta": False, "_render": {}},
        ]))
        self.assertEqual(persi, [])

    def test_ignora_il_blocco_senza_traduzione(self):
        """Senza testo non c'e' lavoro perso da dichiarare."""
        persi = traduzione_non_disegnata(piano([
            {"id": "p7-b4", "testo": "  ", "salta": True,
             "_render": {"testo_illeggibile": True}},
        ]))
        self.assertEqual(persi, [])

    def test_piano_vuoto_o_malformato(self):
        self.assertEqual(traduzione_non_disegnata({}), [])
        self.assertEqual(traduzione_non_disegnata({"pagine": [None]}), [])
        self.assertEqual(traduzione_non_disegnata({"pagine": [{"blocchi": [None]}]}), [])

    def test_sopravvive_al_secondo_rendering(self):
        """E' il caso che rendeva il difetto invisibile.

        Alla seconda passata `_segna_illeggibili` salta i blocchi gia' `salta`
        e non emette piu' nessun evento: la dichiarazione deve venire dal piano,
        non dagli eventi di quella passata.
        """
        controllato = piano([
            {"id": "p7-b5", "testo": "Morira' congelata.", "salta": True,
             "_render": {"testo_illeggibile": True}},
        ])
        self.assertEqual(len(traduzione_non_disegnata(controllato)), 1)


if __name__ == "__main__":
    unittest.main()
