"""Sblocco automatico: la riga vicina conta solo se il riempimento la tocca davvero.

Prova A1 su imperial-guardians: quattro balloon tradotti restavano in inglese
per una "riga esterna" che la maschera del blocco non sfiorava nemmeno -
l'onomatopea "SHRUKK" a p6, "EVER" a p12, "I'M TOLD" a p21, una "C" illeggibile
a p11. La sovrapposizione era dei rettangoli OCR, non dell'inchiostro.
"""

import unittest

from comics_translator.disegno import sblocco_automatico as sblocco


def pagina(riga_box, motivo="background_noise"):
    return {
        "numero": 6,
        "sorgente": {
            "righe": [{"id": "r1", "box_px": riga_box, "motivo": motivo}],
            "localizzazione": {"aree": [{"riga": "r1", "blocco": None}]},
        },
    }


BLOCCO = {
    "id": "b1",
    "_render": {
        "box_px": [900, 4700, 1600, 5500],
        # un balloon a rombo: il rettangolo del blocco sfora, la forma no
        "fill_poly": [[[1250, 4750], [1600, 5100], [1250, 5450], [900, 5100]]],
    },
}


class RigaFuoriDalRiempimentoTest(unittest.TestCase):
    def test_onomatopea_che_sfiora_solo_il_rettangolo_non_blocca(self):
        """"SHRUKK" di p6: 19 px di rettangolo in comune, zero di maschera."""
        box = [1399, 3785, 3081, 4765]
        self.assertLess(sblocco.quota_coperta(box, BLOCCO["_render"]), sblocco.QUOTA_RIGA_TOCCATA)
        self.assertIsNone(sblocco._motivo_riga_non_sicura(pagina(box), BLOCCO, "r1", set()))

    def test_arte_dentro_il_riempimento_blocca_ancora(self):
        box = [1450, 4900, 2300, 5300]  # per lo piu fuori dal riquadro, ma ne copre un pezzo
        self.assertTrue(sblocco._motivo_riga_non_sicura(pagina(box), BLOCCO, "r1", set())
                        .startswith("riga_esterna"))

    def test_rilettura_incerta_dentro_il_balloon_e_dello_stesso_balloon(self):
        """p26: un glifo di recupero OCR a confidenza bassa dentro il lobo."""
        box = [1150, 5000, 1350, 5200]
        self.assertIsNone(sblocco._motivo_riga_non_sicura(
            pagina(box, motivo="recovery_new_candidate"), BLOCCO, "r1", set()))

    def test_senza_profili_conta_il_rettangolo(self):
        render = {"box_px": [0, 0, 100, 100]}
        self.assertAlmostEqual(sblocco.quota_coperta([0, 0, 50, 100], render), 1.0, places=1)
        self.assertEqual(sblocco.quota_coperta([200, 200, 300, 300], render), 0.0)


if __name__ == "__main__":
    unittest.main()
