"""Sblocco automatico e blocchi candidati del Gap C: casi positivi e negativi.

Lo sblocco sostituisce il gesto piu' frequente dei `.finale` (327 blocchi
`source_conflict` sbloccati a mano sui 14 albi FATTO), ma solo quando valgono
insieme la natura detta dal modello e la verifica dell'area riga per riga.
I negativi contano quanto i positivi: una grafica vicina non si copre mai.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pymupdf as fitz

from comics_translator.disegno import image_overlay as images
from comics_translator.disegno import plan
from comics_translator.disegno import sblocco_automatico as sblocco
from comics_translator.sorgente import page_structure
from comics_translator.sorgente import source_inventory as sources
from comics_translator.sorgente import text_localization as loc
from comics_translator.traduzione import ai_review
from comics_translator.verifica import segnalazioni


def line(text="HELLO", box=(20, 20, 80, 32), conf=0.98, reason=None):
    x0, y0, x1, y1 = box
    return images.OcrLine(x0, y0, x1, y1, text, conf, raw_text=text,
                          quad=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                          exclusion_reason=reason, provenance=None)


def analyzed(rows, expand=False, rumore=None):
    data = sources.new_inventory()
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 300))
    patches = [
        patch("comics_translator.disegno.image_overlay.ocr_lines", return_value=rows),
        patch("comics_translator.disegno.image_overlay.page_to_array",
              return_value=(np.full((600, 400, 3), 255, np.uint8), 144)),
    ]
    if rumore is not None:
        patches.append(patch("comics_translator.disegno.image_overlay.background_noise",
                             return_value=rumore))
    for p in patches:
        p.start()
    try:
        jobs, _, _ = images.plan_page(page, 0, 144, 0.5, expand, inventory=data,
                                      recover=False)
    finally:
        for p in patches:
            p.stop()
    for job in jobs:
        job.translated = "CIAO"
    return plan.build_plan([page], [0], {}, {0: jobs}, {}, sources={0: data})


def blocchi(built):
    return built["pagine"][0]["blocchi"]


class SbloccoPositivoTest(unittest.TestCase):
    def _residuo_dentro(self):
        # una rilettura senza lettere dentro il balloon: protegge, ma e' dentro
        return analyzed([line(), line("0400", (22, 22, 78, 30), reason="no_letters")])

    def test_dialogo_tradotto_con_residuo_dentro_viene_sbloccato(self):
        built = self._residuo_dentro()
        blocco = blocchi(built)[0]
        self.assertTrue(blocco["salta"])
        self.assertTrue(blocco["_render"]["source_conflict"])
        blocco["natura"] = "dialogo"

        decisioni = sblocco.sblocca_tradotti(built)

        self.assertEqual([d["esito"] for d in decisioni], ["sbloccato"])
        self.assertFalse(blocco["salta"])
        self.assertIn("natura=dialogo", blocco["_render"]["sbloccato_auto"])
        self.assertNotIn("source_conflict", blocco["_render"])
        sources.validate_sources(built)
        stabile = copy.deepcopy(built)
        loc.enforce_localization(built)
        self.assertEqual(built, stabile, "il ricalcolo non revoca lo sblocco")
        # il residuo dentro il riquadro viene coperto
        poligoni = plan.poligoni_sbloccati(blocco["_render"], built["pagine"][0])
        self.assertEqual(len(poligoni), 1)
        self.assertEqual(segnalazioni.dialogo_protetto_non_tradotto(built), [])

    def test_fixpoint_sblocca_insieme_una_catena_di_blocchi_di_testo(self):
        """C candidato del Gap C protegge B, B protegge A: tutti e tre testo."""
        doppio = [[10, 10], [120, 10], [120, 130], [10, 130]]
        rows = [line("UNO"), line("DUE", (20, 100, 80, 112)), line("TRE", (20, 180, 80, 192))]
        areas = [([15, 15, 110, 70], [doppio]),
                 ([15, 75, 110, 150], [[[10, 90], [120, 90], [120, 210], [10, 210]]]),
                 ([15, 170, 110, 210], [[[10, 170], [120, 170], [120, 210], [10, 210]]])]
        with patch("comics_translator.disegno.image_overlay.prepara_area", side_effect=areas):
            built = analyzed(rows, expand=True)
        a, b, c = blocchi(built)
        c["salta"] = True
        # Il marcatore nasce all'analisi: la struttura P3 lo tiene nei suoi
        # motivi, quindi va ricostruita come se C fosse arrivato cosi' dall'OCR.
        c["_render"]["scartato_come_disegno"] = True
        built["pagine"][0]["struttura"] = page_structure.build_structure(built["pagine"][0])
        loc.enforce_localization(built)
        self.assertTrue(a["salta"] and b["salta"])
        for blocco in (a, b, c):
            blocco["natura"] = "dialogo"

        decisioni = sblocco.sblocca_tradotti(built)

        self.assertEqual(sorted(d["esito"] for d in decisioni), ["sbloccato"] * 3)
        self.assertFalse(any(x["salta"] for x in (a, b, c)))
        sources.validate_sources(built)


    def test_uno_sblocco_revocato_non_fa_morire_la_run(self):
        """Il ValueError arrivava dopo tutti i lotti pagati: ora si dichiara."""
        built = self._residuo_dentro()
        blocco = blocchi(built)[0]
        blocco["natura"] = "dialogo"

        def ricalcolo_che_revoca(piano):
            for p in piano["pagine"]:
                for b in p["blocchi"]:
                    b["salta"] = True

        with patch.object(sblocco.localization, "enforce_localization",
                          side_effect=ricalcolo_che_revoca):
            (decisione,) = sblocco.sblocca_tradotti(built)
        self.assertEqual(decisione["esito"], "resta_saltato")
        self.assertEqual(decisione["motivo"], "revocato_dal_ricalcolo")
        self.assertTrue(blocco["salta"])
        self.assertNotIn("sbloccato_auto", blocco["_render"])


class SbloccoNegativoTest(unittest.TestCase):
    _residuo_dentro = SbloccoPositivoTest._residuo_dentro

    def test_la_grafica_non_si_sblocca(self):
        built = self._residuo_dentro()
        blocchi(built)[0]["natura"] = "grafica"
        self.assertEqual(sblocco.sblocca_tradotti(built), [])
        self.assertTrue(blocchi(built)[0]["salta"])

    def test_senza_natura_non_si_sblocca(self):
        built = self._residuo_dentro()
        self.assertEqual(sblocco.sblocca_tradotti(built), [])
        self.assertTrue(blocchi(built)[0]["salta"])

    def test_senza_traduzione_vera_non_si_sblocca(self):
        built = self._residuo_dentro()
        blocco = blocchi(built)[0]
        blocco["natura"] = "dialogo"
        blocco["testo"] = blocco["originale"]
        self.assertEqual(sblocco.sblocca_tradotti(built), [])
        self.assertTrue(blocco["salta"])

    def test_una_decisione_presa_su_grafica_non_si_scavalca(self):
        built = self._residuo_dentro()
        blocco = blocchi(built)[0]
        blocco["natura"] = "grafica"
        blocco["_render"]["salta_deciso"] = True
        self.assertEqual(sblocco.sblocca_tradotti(built), [])

    def test_il_salto_del_modello_su_un_dialogo_tradotto_si_revoca(self):
        """182 dei 424 sblocchi manuali degli ultimi 20 FATTO: dialogo, tradotto,
        saltato dal modello per il vecchio prompt ("piu' lobi -> salta")."""
        built = self._residuo_dentro()
        blocco = blocchi(built)[0]
        blocco["natura"] = "dialogo"
        blocco["_render"]["salta_deciso"] = True

        decisioni = sblocco.sblocca_tradotti(built)

        self.assertEqual([d["esito"] for d in decisioni], ["sbloccato"])
        self.assertFalse(blocco["salta"])
        self.assertNotIn("salta_deciso", blocco["_render"])
        self.assertIn("salto del modello revocato", blocco["_render"]["sbloccato_auto"])
        sources.validate_sources(built)
        loc.enforce_localization(built)
        self.assertFalse(blocco["salta"])

    def test_il_salto_del_modello_senza_traduzione_resta(self):
        built = self._residuo_dentro()
        blocco = blocchi(built)[0]
        blocco["natura"] = "dialogo"
        blocco["testo"] = blocco["originale"]
        blocco["_render"]["salta_deciso"] = True
        self.assertEqual(sblocco.sblocca_tradotti(built), [])

    def test_una_riga_esterna_che_il_riempimento_sfiora_blocca_lo_sblocco(self):
        """HA HA! sopra, WHAM! accanto: la maschera la tocca, il riquadro no."""
        rows = [line("HELLO", (20, 20, 80, 32)), line("BLAM", (20, 100, 80, 140),
                                                     reason="no_letters")]
        grande = [[[10, 10], [120, 10], [120, 150], [10, 150]]]
        with patch("comics_translator.disegno.image_overlay.prepara_area",
                   side_effect=[([15, 15, 110, 70], grande)]):
            built = analyzed(rows, expand=True)
        blocco = blocchi(built)[0]
        self.assertTrue(blocco["salta"])
        blocco["natura"] = "dialogo"

        decisioni = sblocco.sblocca_tradotti(built)

        self.assertEqual(decisioni[0]["esito"], "resta_saltato")
        self.assertTrue(decisioni[0]["motivo"].startswith("riga_esterna"))
        self.assertTrue(blocco["salta"])

    def test_un_vicino_che_resta_in_originale_ferma_tutta_la_catena(self):
        doppio = [[10, 10], [120, 10], [120, 130], [10, 130]]
        rows = [line("UNO"), line("DUE", (20, 100, 80, 112)), line("TRE", (20, 180, 80, 192))]
        areas = [([15, 15, 110, 70], [doppio]),
                 ([15, 75, 110, 150], [[[10, 90], [120, 90], [120, 210], [10, 210]]]),
                 ([15, 170, 110, 210], [[[10, 170], [120, 170], [120, 210], [10, 210]]])]
        with patch("comics_translator.disegno.image_overlay.prepara_area", side_effect=areas):
            built = analyzed(rows, expand=True)
        a, b, c = blocchi(built)
        c["salta"] = True
        c["_render"]["salta_deciso"] = True   # onomatopea decisa dal modello
        loc.enforce_localization(built)
        a["natura"] = b["natura"] = "dialogo"

        decisioni = {d["id"]: d for d in sblocco.sblocca_tradotti(built)}

        self.assertEqual(decisioni[a["id"]]["esito"], "resta_saltato")
        self.assertEqual(decisioni[b["id"]]["esito"], "resta_saltato")
        self.assertTrue(decisioni[b["id"]]["motivo"].startswith("vicino_in_originale"))
        self.assertTrue(a["salta"] and b["salta"] and c["salta"])


class PoligoniSbloccatiTest(unittest.TestCase):
    def test_si_copre_solo_la_riga_che_sta_dentro_il_riquadro(self):
        pagina = {"sorgente": {"righe": [
            {"id": "dentro", "box_px": [20, 20, 90, 40]},
            {"id": "fuori", "box_px": [300, 300, 500, 360]},
        ]}}
        render = {"sbloccato_a_mano": "visto", "box_px": [0, 0, 100, 100],
                  "source_overlap": ["dentro", "fuori"]}
        poligoni = plan.poligoni_sbloccati(render, pagina)
        self.assertEqual(poligoni, [[[20, 20], [90, 20], [90, 40], [20, 40]]])

    def test_senza_sblocco_non_si_copre_niente(self):
        pagina = {"sorgente": {"righe": [{"id": "dentro", "box_px": [20, 20, 90, 40]}]}}
        render = {"box_px": [0, 0, 100, 100], "source_overlap": ["dentro"]}
        self.assertEqual(plan.poligoni_sbloccati(render, pagina), [])

    def test_una_riga_alta_quanto_la_tavola_non_si_copre(self):
        """Il titolo verticale di imperial-guardians p2.

        "JIM SHOOTER" e' UNA riga OCR alta 4200 px. Il margine di mezza riga
        gonfiava il riquadro del blocco sbloccato accanto di altrettanto, la
        riga risultava "dentro" e il riempimento la cancellava: sulla tavola
        restava un rettangolo bianco al posto del titolo.
        """
        pagina = {"sorgente": {"righe": [{"id": "titolo", "box_px": [99, 753, 1344, 4982]}]}}
        render = {"sbloccato_a_mano": "colonna sinistra", "box_px": [1193, 4855, 2462, 5760],
                  "source_overlap": ["titolo"]}
        self.assertEqual(plan.poligoni_sbloccati(render, pagina), [])

    def test_un_residuo_grande_quanto_basta_si_copre_ancora(self):
        """La regola nuova non chiude la porta al caso per cui esiste."""
        pagina = {"sorgente": {"righe": [{"id": "resto", "box_px": [80, 10, 140, 34]}]}}
        render = {"sbloccato_a_mano": "visto", "box_px": [0, 0, 100, 100],
                  "source_overlap": ["resto"]}
        self.assertEqual(plan.poligoni_sbloccati(render, pagina),
                         [[[80, 10], [140, 10], [140, 34], [80, 34]]])



class CandidatiScartatiTest(unittest.TestCase):
    def test_una_didascalia_scartata_come_disegno_diventa_un_blocco_saltato(self):
        built = analyzed([line("THE NEXT DAY.", (20, 20, 160, 32))], rumore=20.0)
        (blocco,) = blocchi(built)
        self.assertTrue(blocco["salta"])
        self.assertTrue(blocco["_render"]["scartato_come_disegno"])
        regione = {r["id"]: r for r in built["pagine"][0]["sorgente"]["regioni"]}[
            blocco["sorgente"]["regione"]]
        self.assertEqual(regione["motivo"], "background_noise")
        sources.validate_sources(built)
        self.assertEqual(ai_review.motivo_protezione(blocco), "scartato_come_disegno")
        self.assertEqual(segnalazioni.testo_scartato_come_disegno(built), [],
                         "arrivato al modello: non e' piu' un Gap C da dichiarare")
        self.assertEqual(plan.drawable_entries(blocchi(built), page=built["pagine"][0]), [])

        blocco["natura"] = "didascalia"
        blocco["testo"] = "IL GIORNO DOPO."
        self.assertEqual(sblocco.sblocca_tradotti(built)[0]["esito"], "sbloccato")
        disegnate = plan.drawable_entries(blocchi(built), page=built["pagine"][0])
        self.assertEqual([e["id"] for e in disegnate], [blocco["id"]])
        sources.validate_sources(built)

    def test_un_candidato_su_fondo_disegnato_resta_saltato_e_dichiarato(self):
        """Prova A2 (spidey-63-9): il rettangolo del candidato copriva il disegno.

        Il titolo di p2 (rumore 66) e una vignetta di p19 (rumore 61) sparivano
        sotto un rettangolo pieno. La traduzione non si perde in silenzio.
        """
        built = analyzed([line("THE NEXT DAY.", (20, 20, 160, 32))], rumore=45.0)
        (blocco,) = blocchi(built)
        blocco["natura"] = "didascalia"
        blocco["testo"] = "IL GIORNO DOPO."
        (decisione,) = sblocco.sblocca_tradotti(built)
        self.assertEqual(decisione["esito"], "resta_saltato")
        self.assertTrue(decisione["motivo"].startswith("fondo_disegnato"))
        self.assertTrue(blocco["salta"])
        self.assertEqual([x["id"] for x in loc.traduzioni_soppresse(built)], [blocco["id"]])

    def test_un_onomatopea_scartata_resta_fuori_dal_piano(self):
        built = analyzed([line("BLAM", (20, 20, 80, 60))], rumore=40.0)
        self.assertEqual(blocchi(built), [])

    def test_i_candidati_si_possono_disattivare(self):
        data = sources.new_inventory()
        pagina = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 300))
        with (patch("comics_translator.disegno.image_overlay.ocr_lines",
                    return_value=[line("THE NEXT DAY.", (20, 20, 160, 32))]),
              patch("comics_translator.disegno.image_overlay.page_to_array",
                    return_value=(np.full((600, 400, 3), 255, np.uint8), 144)),
              patch("comics_translator.disegno.image_overlay.background_noise", return_value=40.0)):
            jobs, _, skipped = images.plan_page(pagina, 0, 144, 0.5, False, inventory=data,
                                                recover=False, candidati_scartati=False)
        self.assertEqual((jobs, skipped), ([], 1))


if __name__ == "__main__":
    unittest.main()
