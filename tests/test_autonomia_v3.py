"""Autonomia v3: le mosse dei `.finale` dentro il motore.

Analisi del 28 ottobre 2026 su 38 albi FATTO: 1.939 interventi manuali dopo il
QA. Sugli ultimi 20 il 46% era testo gia' tradotto e mai disegnato; il resto
balloon fusi, righe orfane, testo mai visto dall'OCR. Ogni mossa ha qui il suo
caso positivo e il suo caso negativo.
"""

import unittest

import cv2
import numpy as np

from comics_translator.disegno import dividi_balloon
from comics_translator.disegno import sblocco_automatico as sblocco
from comics_translator.disegno.operazioni import Officina, rettangoli
from comics_translator.sorgente import source_inventory as sources
from comics_translator.sorgente import text_localization as loc
from comics_translator.verifica import da_lasciare, riparazione

from test_sblocco_automatico import analyzed, blocchi, line


class DaLasciareTest(unittest.TestCase):
    def test_crediti_legale_marchi_onomatopee(self):
        for testo in ("WRITTEN BY AL EWING", "ART BY PACO MEDINA", "COLORS",
                      "VC's JOE SABINO", "© 2026 MARVEL", "All rights reserved",
                      "MARVEL", "KRKK", "AAAAGH", "THWP"):
            self.assertIsNotNone(da_lasciare.motivo(testo), testo)

    def test_le_battute_non_si_lasciano(self):
        for testo in ("THE ART OF WAR IS A LIE", "COVER ME!", "WHAT ARE YOU DOING?",
                      "PREVIOUSLY...", "TO BE CONTINUED", "LIVE", "NEXT:"):
            self.assertIsNone(da_lasciare.motivo(testo), testo)


def _pagina_con_balloon(forme):
    """Fondo grigio, balloon bianchi bordati di nero."""
    img = np.full((600, 800, 3), 140, np.uint8)
    for forma in forme:
        maschera = np.zeros(img.shape[:2], np.uint8)
        for tipo, *p in forma:
            if tipo == "ellisse":
                cv2.ellipse(maschera, p[0], p[1], 0, 0, 360, 255, -1)
        contorni, _ = cv2.findContours(maschera, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contorni, -1, (255, 255, 255), -1)
        cv2.drawContours(img, contorni, -1, (0, 0, 0), 4)
    return img


def _scrivi(img, boxes):
    for x0, y0, x1, y1 in boxes:
        for x in range(x0 + 2, x1 - 4, 12):
            cv2.rectangle(img, (x, y0 + 3), (x + 6, y1 - 3), (0, 0, 0), -1)


class DivisionePerBalloonTest(unittest.TestCase):
    def test_due_balloon_separati_si_dividono(self):
        img = _pagina_con_balloon([[("ellisse", (200, 200), (150, 80))],
                                   [("ellisse", (580, 380), (150, 80))]])
        righe = [[120, 185, 280, 215], [500, 365, 660, 395]]
        _scrivi(img, righe)
        gruppi = dividi_balloon.gruppi_per_balloon(img, righe, (255, 255, 255))
        self.assertEqual(gruppi, [[0], [1]])

    def test_due_lobi_con_strozzatura_si_dividono(self):
        img = _pagina_con_balloon([[("ellisse", (300, 200), (200, 70)),
                                    ("ellisse", (460, 330), (110, 90))]])
        righe = [[180, 175, 420, 200], [190, 205, 410, 230], [400, 315, 520, 345]]
        _scrivi(img, righe)
        gruppi = dividi_balloon.gruppi_per_balloon(img, righe, (255, 255, 255))
        self.assertEqual(gruppi, [[0, 1], [2]])

    def test_un_ovale_con_due_righe_resta_intero(self):
        img = _pagina_con_balloon([[("ellisse", (400, 300), (220, 120))]])
        righe = [[260, 250, 540, 285], [270, 310, 530, 345]]
        _scrivi(img, righe)
        self.assertIsNone(dividi_balloon.gruppi_per_balloon(img, righe, (255, 255, 255)))

    def test_testo_sul_disegno_non_si_divide(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        righe = [[100, 100, 300, 130], [100, 300, 300, 330]]
        self.assertIsNone(dividi_balloon.gruppi_per_balloon(img, righe, (255, 255, 255)))

    def test_il_testo_si_divide_contando_le_frasi(self):
        self.assertEqual(
            dividi_balloon.dividi_testo(["I'VE BEEN HERE. I--", "IF I SAY IT, IT'S READY."],
                                        "Sono qui. Io-- Se lo dico, e' pronto."),
            ["Sono qui. Io--", "Se lo dico, e' pronto."])
        self.assertIsNone(dividi_balloon.dividi_testo(["ONE. TWO.", "THREE."], "Uno e due e tre."))


class OperazioniTest(unittest.TestCase):
    def test_banda_disegna_solo_le_righe_proprie(self):
        built = analyzed([line(), line("0400", (22, 22, 78, 30), reason="no_letters")])
        blocco = blocchi(built)[0]
        self.assertTrue(blocco["salta"])
        officina = Officina(built)
        officina.banda(blocco["id"], "prova", "test")
        self.assertFalse(blocco["salta"])
        self.assertTrue(blocco["_render"]["banda"])
        self.assertEqual(len(blocco["_render"]["fill_poly"]), 1)
        self.assertIn("prova", blocco["_render"]["sbloccato_auto"])
        loc.enforce_localization(built)
        self.assertFalse(blocco["salta"])
        sources.validate_sources(built)

    def test_nuovo_blocco_su_riga_sintetica_e_valido(self):
        built = analyzed([line()])
        officina = Officina(built)
        riga = officina.nuova_riga(1, "seme", [200, 400, 300, 430], "NEXT:")
        nuovo = officina.nuovo_blocco(1, [riga], "NEL PROSSIMO NUMERO:", "prova", "test",
                                      sfondo="banda")
        officina.sigilla([1])
        loc.enforce_localization(built)
        sources.validate_sources(built)
        self.assertFalse(nuovo["salta"])
        self.assertEqual(nuovo["sorgente"]["righe"], [riga])


class SbloccoConBandaTest(unittest.TestCase):
    def test_rifiutato_per_area_va_su_banda_se_la_banda_non_tocca_nessuno(self):
        rows = [line("HELLO", (20, 20, 80, 32)), line("BLAM", (20, 100, 80, 140),
                                                     reason="no_letters")]
        from unittest.mock import patch
        grande = [[[10, 10], [120, 10], [120, 150], [10, 150]]]
        with patch("comics_translator.disegno.image_overlay.prepara_area",
                   side_effect=[([15, 15, 110, 70], grande)]):
            built = analyzed(rows, expand=True)
        blocco = blocchi(built)[0]
        blocco["natura"] = "dialogo"
        decisioni = sblocco.sblocca_tradotti(built)
        self.assertEqual(decisioni[0]["esito"], "resta_saltato")
        sblocco.sblocca_con_banda(built, Officina(built), decisioni)
        self.assertEqual(decisioni[0]["esito"], "banda")
        self.assertFalse(blocco["salta"])
        self.assertEqual(blocco["_render"]["fill_poly"], rettangoli([blocco_righe(built)]))

    def test_la_grafica_rifiutata_non_va_su_banda(self):
        decisioni = [{"id": "x", "esito": "resta_saltato", "motivo": "fondo_disegnato:40"}]
        built = analyzed([line()])
        decisioni[0]["id"] = blocchi(built)[0]["id"]
        blocchi(built)[0]["natura"] = "grafica"
        blocchi(built)[0]["salta"] = True
        # la banda si chiede solo per i blocchi passati da `candidato`: qui un
        # blocco grafico non ci arriva mai, ma anche forzandolo non ha righe
        # da difendere e resta una scelta del chiamante; il filtro vero e'
        # `candidato`, che lo esclude.
        self.assertFalse(sblocco.candidato(blocchi(built)[0], built["pagine"][0]))


def blocco_righe(built):
    page = built["pagine"][0]
    righe = {r["id"]: r for r in page["sorgente"]["righe"]}
    return righe[page["blocchi"][0]["sorgente"]["righe"][0]]["box_px"]


class RiparazioneTest(unittest.TestCase):
    def _officina(self):
        built = analyzed([line("HELLO", (100, 100, 200, 120))])
        blocco = blocchi(built)[0]
        blocco["salta"] = False
        blocco["_render"]["box_px"] = [100, 100, 200, 120]
        return built, Officina(built), blocco

    def test_fantasma_dentro_il_riquadro_si_copre(self):
        _, officina, blocco = self._officina()
        op, b = riparazione.attribuisci(officina, 1, [120, 105, 160, 115])
        self.assertEqual((op, b["id"]), ("copri", blocco["id"]))

    def test_riga_orfana_sotto_si_attacca_al_blocco(self):
        _, officina, blocco = self._officina()
        op, b = riparazione.attribuisci(officina, 1, [110, 125, 190, 140])
        self.assertEqual((op, b["id"]), ("estendi", blocco["id"]))

    def test_riga_lontana_va_al_modello(self):
        _, officina, _ = self._officina()
        self.assertEqual(riparazione.attribuisci(officina, 1, [100, 500, 200, 520]),
                         (None, None))

    def test_le_operazioni_sbagliate_cadono_da_sole(self):
        built, officina, blocco = self._officina()
        dati = [{"pagina": 1, "residui": [{"id": "p1-r1", "testo": "GO"}],
                 "blocchi": [{"id": blocco["id"]}]}]
        spec = {"operazioni": [
            {"op": "testo", "blocco": blocco["id"], "testo": "CIAO A TUTTI"},
            {"op": "testo", "blocco": "inventato", "testo": "X"},
            {"op": "cancella_tutto"},
            {"op": "lascia", "residuo": "p1-r1", "motivo": "logo"},
        ]}
        fatte, scartate = riparazione.applica_operazioni(officina, spec, dati, {1: 72})
        self.assertEqual(len(fatte), 2)
        self.assertEqual(len(scartate), 2)
        self.assertEqual(blocco["testo"], "CIAO A TUTTI")

    def test_un_nuovo_ripetuto_si_scarta_e_la_pagina_resta_valida(self):
        built, officina, _ = self._officina()
        dati = [riparazione._dati_pagina(officina, 1, [], 72)]
        op = {"op": "nuovo", "pagina": 1, "box": [300, 400, 380, 420], "testo": "BORSA"}
        fatte, scartate = riparazione.applica_operazioni(
            officina, {"operazioni": [op, dict(op)]}, dati, {1: 72})
        self.assertEqual((len(fatte), len(scartate)), (1, 1))
        officina.sigilla([1])
        loc.enforce_localization(built)
        sources.validate_sources(built)

    def test_traduci_residuo_usa_il_riquadro_dell_immagine(self):
        built, officina, _ = self._officina()
        scala = officina.scala(1)
        dpi = 144
        dati = [riparazione._dati_pagina(
            officina, 1, [{"testo": "GO AWAY", "box_pt": [v * scala for v in (300, 400, 380, 420)]}],
            dpi)]
        spec = {"operazioni": [{"op": "traduci_residuo", "residuo": "p1-r1",
                                "testo": "VATTENE", "sfondo": "banda"}]}
        fatte, scartate = riparazione.applica_operazioni(officina, spec, dati, {1: dpi})
        self.assertEqual(scartate, [])
        self.assertEqual(len(fatte), 1)
        riga = built["pagine"][0]["sorgente"]["righe"][-1]
        # andata e ritorno px del piano -> px dell'immagine -> px del piano
        for v, atteso in zip(riga["box_px"], (300, 400, 380, 420)):
            self.assertAlmostEqual(v, atteso, delta=2)


if __name__ == "__main__":
    unittest.main()
