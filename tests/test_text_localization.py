"""P2: casi positivi/negativi sintetici, senza rete o OCR reale nella suite."""

import copy
import json
import tempfile
from pathlib import Path
import unittest

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pymupdf as fitz

from comics_translator.disegno import image_overlay as images
from comics_translator.disegno import pdf_translate_overlay as pipeline
from comics_translator.disegno import plan
from comics_translator import pipeline_status as status
from comics_translator.sorgente import source_inventory as sources
from comics_translator.sorgente import text_localization as loc
from comics_translator.verifica.qa_corrections import apply_corrections


def line(text="HELLO", box=(20, 20, 80, 32), conf=0.98, reason=None, provenance=None):
    x0, y0, x1, y1 = box
    return images.OcrLine(
        x0,
        y0,
        x1,
        y1,
        text,
        conf,
        raw_text=text,
        quad=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        exclusion_reason=reason,
        provenance=provenance,
    )


def analyzed(rows, expand=False):
    data = sources.new_inventory()
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 300))
    with (
        patch("comics_translator.disegno.image_overlay.ocr_lines", return_value=rows),
        patch(
            "comics_translator.disegno.image_overlay.page_to_array",
            return_value=(np.full((600, 400, 3), 255, np.uint8), 144),
        ),
    ):
        jobs, _, _ = images.plan_page(
            page, 0, 144, 0.5, expand, inventory=data, recover=False
        )
    for job in jobs:
        job.translated = "CIAO"
    built = plan.build_plan([page], [0], {}, {0: jobs}, {}, sources={0: data})
    return built


class DeduplicationTest(unittest.TestCase):
    def test_same_text_same_place_has_one_owner_and_two_observations(self):
        built = analyzed([line(), line()])
        page = built["pagine"][0]
        self.assertEqual(len(page["blocchi"]), 1)
        self.assertEqual(len(page["sorgente"]["righe"]), 2)
        self.assertEqual(page["sorgente"]["localizzazione"]["duplicati"], 1)
        self.assertFalse(page["blocchi"][0]["salta"])
        sources.validate_sources(built)

    def test_repeated_words_in_different_balloons_are_not_removed(self):
        built = analyzed([line("NO!"), line("NO!", (20, 100, 80, 112))])
        self.assertEqual(len(built["pagine"][0]["blocchi"]), 2)

    def test_punctuation_and_substrings_are_not_deduplicated(self):
        for other in ("HELLO?", "HELL", "HELLO WORLD"):
            canonical, aliases = loc.deduplicate([line(), line(other)])
            self.assertEqual(canonical, [0, 1])
            self.assertEqual(aliases, {})

    def test_partial_overlap_is_not_enough(self):
        _, aliases = loc.deduplicate([line(), line(box=(45, 20, 105, 32))])
        self.assertEqual(aliases, {})

    def test_good_original_wins_over_higher_confidence_retry(self):
        canonical, aliases = loc.deduplicate(
            [line(conf=0.9), line(conf=0.99, provenance={"tipo": "ritaglio"})]
        )
        self.assertEqual(canonical, [0])
        self.assertEqual(aliases, {1: 0})

    def test_weak_original_can_be_superseded_without_losing_it(self):
        built = analyzed(
            [
                line(conf=0.3, reason="low_confidence"),
                line(conf=0.95, provenance={"tipo": "ritaglio"}),
            ]
        )
        page = built["pagine"][0]
        rows = page["sorgente"]["righe"]
        self.assertEqual(rows[0]["duplicato_di"], rows[1]["id"])
        self.assertEqual(rows[0]["motivo_ocr"], "low_confidence")
        self.assertEqual(page["blocchi"][0]["sorgente"]["righe"], [rows[1]["id"]])

    def test_native_duplicate_layer_is_removed_but_observations_remain(self):
        with fitz.open() as doc:
            page = doc.new_page(width=200, height=300)
            page.insert_text((20, 30), "HELLO")
            page.insert_text((20, 30), "HELLO")
            data = sources.new_inventory()
            blocks, _ = pipeline.extract_blocks(
                page, 0, 2, inventory=data, localize=True
            )
            kept = loc.deduplicate_native(blocks, data)
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0].text, "HELLO")
            self.assertEqual(len(data["righe"]), 2)
            self.assertEqual(data["righe"][1]["motivo"], "duplicate")


class NativeDedupStatusTest(unittest.TestCase):
    def test_duplicate_layer_is_not_a_missing_text_issue(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, plan_path = (
                root / "source.pdf",
                root / "out.pdf",
                root / "plan.json",
            )
            with fitz.open() as doc:
                page = doc.new_page(width=200, height=300)
                page.insert_text((20, 30), "HELLO")
                page.insert_text((20, 30), "HELLO")
                doc.save(source)
            argv = [
                "pdf_translate_overlay.py",
                str(source),
                "--mode",
                "text",
                "--engine",
                "none",
                "--no-visual-qa",
                "--plan-out",
                str(plan_path),
                "-o",
                str(output),
            ]
            with patch("sys.argv", argv):
                status.cli_entry(pipeline.main)
            saved = plan.load_plan(plan_path)
            self.assertEqual(
                saved["pagine"][0]["sorgente"]["localizzazione"]["duplicati"], 1
            )
            manifest = json.loads(
                next(root.glob("out.run/runs/*/manifest.json")).read_text("utf-8")
            )
            self.assertEqual(manifest["phases"]["analysis"]["status"], "complete")


class RecoveryTest(unittest.TestCase):
    def exercise(self, initial, recovered, limits=None, origin=(0, 0)):
        page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 300))
        before = copy.deepcopy(initial)
        with patch(
            "comics_translator.sorgente.text_localization.render_crop",
            return_value=(np.zeros((80, 200, 3), np.uint8), origin),
        ) as renderer:
            result, report = loc.recover(
                page, initial, 144, lambda image: recovered, limits
            )
        self.assertEqual(initial, before)
        return result, report, renderer

    def test_concordant_retry_promotes_weak_line_with_provenance(self):
        result, report, _ = self.exercise(
            [line(conf=0.3, reason="low_confidence")],
            [line(box=(40, 40, 160, 64), conf=0.95)],
        )
        self.assertIsNone(result[-1].exclusion_reason)
        self.assertEqual(report["promozioni_concordi"], 1)
        self.assertEqual(result[-1].provenance["dpi"], 288)
        self.assertEqual(loc.bounds(result[-1]), [20, 20, 80, 32])

    def test_different_high_confidence_reading_is_not_promoted(self):
        result, report, _ = self.exercise(
            [line(conf=0.3, reason="low_confidence")],
            [line("WORLD", (40, 40, 160, 64), 0.99)],
        )
        self.assertEqual(result[-1].exclusion_reason, "recovery_alternative")
        self.assertEqual(report["promozioni_concordi"], 0)

    def test_medium_confidence_retry_is_still_unconfirmed(self):
        result, _, _ = self.exercise(
            [line(conf=0.3, reason="low_confidence")],
            [line(box=(40, 40, 160, 64), conf=0.6)],
        )
        self.assertEqual(result[-1].exclusion_reason, "recovery_unconfirmed")

    def test_new_crop_detection_is_visible_but_not_automatically_rewritten(self):
        result, _, _ = self.exercise(
            [line(conf=0.3, reason="low_confidence")],
            [line("NEW", (40, 120, 160, 144), 0.99)],
        )
        self.assertEqual(result[-1].exclusion_reason, "recovery_new_candidate")

    def test_crop_origin_maps_back_to_page_coordinates(self):
        result, _, _ = self.exercise(
            [line(conf=0.3, reason="low_confidence")],
            [line(box=(32, 28, 152, 52), conf=0.95)],
            origin=(2, 3),
        )
        self.assertEqual(loc.bounds(result[-1]), [20, 20, 80, 32])
        self.assertEqual(result[-1].provenance["origine_pt"], [2, 3])

    def test_crop_count_and_total_pixel_budgets_prevent_allocations(self):
        for limits, reason in (
            (loc.RecoveryLimits(max_crops=0), "crop_budget"),
            (loc.RecoveryLimits(max_total_pixels=1), "pixel_budget"),
            (loc.RecoveryLimits(max_dpi=144), "resolution_budget"),
        ):
            _, report, renderer = self.exercise(
                [line(conf=0.3, reason="low_confidence")], [], limits
            )
            renderer.assert_not_called()
            self.assertEqual(report["rinviati"][0]["motivo"], reason)

    def test_per_crop_budget_is_checked_before_render(self):
        seed = line(box=(0, 0, 400, 600), conf=0.2, reason="low_confidence")
        limits = loc.RecoveryLimits(max_crop_pixels=500_000)
        spec = loc.crop_spec(fitz.Rect(0, 0, 200, 300), seed, 144, limits)
        self.assertIsNotNone(spec)
        self.assertLessEqual(spec[2], limits.max_crop_pixels)

    def test_crop_failure_is_recorded_and_original_remains(self):
        initial = [line(conf=0.3, reason="low_confidence")]
        with patch(
            "comics_translator.sorgente.text_localization.render_crop", side_effect=RuntimeError("fixture error")
        ):
            result, report = loc.recover(
                SimpleNamespace(rect=fitz.Rect(0, 0, 200, 300)),
                initial,
                144,
                lambda image: [],
            )
        self.assertEqual(result, initial)
        self.assertEqual(report["tentativi"][0]["errore"], "fixture error")


class OccupancyTest(unittest.TestCase):
    def test_numbers_symbols_and_empty_detections_are_protected(self):
        built = analyzed(
            [
                line("0400", reason="no_letters"),
                line("#$%", (20, 80, 80, 92), reason="no_letters"),
                line("", (20, 140, 80, 152), reason="empty_text"),
            ]
        )
        page = built["pagine"][0]
        self.assertEqual(page["blocchi"], [])
        self.assertEqual(len(page["sorgente"]["localizzazione"]["aree"]), 3)
        self.assertTrue(
            all(
                a["stato"] == "protetta"
                for a in page["sorgente"]["localizzazione"]["aree"]
            )
        )
        self.assertFalse(page["sorgente"]["localizzazione"]["verificata"])

    def test_mask_cannot_erase_excluded_neighbor_even_if_contour_drops_hole(self):
        rows = [line(), line("0400", (85, 20, 105, 32), reason="no_letters")]
        poly = [[15, 15], [110, 15], [110, 40], [15, 40]]
        with patch(
            "comics_translator.disegno.image_overlay.prepara_area", return_value=([15, 15, 110, 40], [poly])
        ) as prepare:
            built = analyzed(rows, expand=True)
        block = built["pagine"][0]["blocchi"][0]
        self.assertTrue(block["salta"])
        self.assertEqual(len(block["_render"]["source_conflict"]), 1)
        self.assertIn([85, 20, 105, 32], prepare.call_args.args[4])
        with self.assertRaisesRegex(ValueError, "source_conflict"):
            apply_corrections(built, {"set": {block["id"]: {"salta": False}}})

    def test_crop_alternative_is_not_a_foreign_neighbor(self):
        rows = [
            line(),
            line(
                "HE110", reason="recovery_alternative", provenance={"tipo": "ritaglio"}
            ),
        ]
        built = analyzed(rows)
        page = built["pagine"][0]
        self.assertFalse(page["blocchi"][0]["salta"])
        self.assertEqual(len(page["sorgente"]["localizzazione"]["divergenze"]), 1)
        self.assertEqual(
            page["sorgente"]["righe"][1]["correlata_a"],
            [page["sorgente"]["righe"][0]["id"]],
        )
        sources.validate_sources(built)

    def test_crop_alternative_extending_outside_owner_still_protected(self):
        built = analyzed(
            [
                line(),
                line(
                    "HE110",
                    box=(20, 20, 110, 32),
                    reason="recovery_alternative",
                    provenance={"tipo": "ritaglio"},
                ),
            ]
        )
        self.assertTrue(built["pagine"][0]["blocchi"][0]["salta"])

    def test_rotated_quad_not_bounding_corner_is_protected(self):
        diamond = line("SFX", (70, 25, 130, 85), reason="low_confidence")
        diamond.quad = [[100, 25], [130, 55], [100, 85], [70, 55]]
        built = analyzed([line(), diamond])
        self.assertFalse(built["pagine"][0]["blocchi"][0]["salta"])
        diamond.quad = [[75, 25], [130, 55], [100, 85], [70, 32]]
        built = analyzed([line(), diamond])
        self.assertTrue(built["pagine"][0]["blocchi"][0]["salta"])

    def test_polygons_cross_without_contained_vertices(self):
        self.assertTrue(
            loc.polygons_intersect(
                [[0, 4], [10, 4], [10, 6], [0, 6]], [[4, 0], [6, 0], [6, 10], [4, 10]]
            )
        )
        self.assertFalse(
            loc.polygons_intersect(
                [[0, 0], [2, 0], [2, 2], [0, 2]], [[4, 0], [6, 0], [6, 2], [4, 2]]
            )
        )

    def test_merge_cannot_span_an_untranscribed_neighbor(self):
        built = analyzed(
            [
                line("ONE", (20, 20, 50, 32)),
                line("TWO", (200, 20, 230, 32)),
                line("0400", (110, 20, 135, 32), reason="no_letters"),
            ]
        )
        page = built["pagine"][0]
        self.assertEqual(len(page["blocchi"]), 2)
        first, second = page["blocchi"]
        self.assertFalse(first["salta"])
        self.assertFalse(second["salta"])
        first["unisci_con"] = second["id"]
        with (
            patch("comics_translator.disegno.plan._same_balloon", return_value=True),
            patch("comics_translator.disegno.plan._merge_geometry_ok", return_value=True),
        ):
            self.assertTrue(
                any(
                    "testo protetto" in problem
                    for problem in plan.validate_merges(built)
                )
            )
        self.assertEqual(len(loc.merge_conflicts(page, first, second)), 1)

    def test_clear_merge_is_allowed_by_source_map(self):
        built = analyzed(
            [line("ONE", (20, 20, 50, 32)), line("TWO", (200, 20, 230, 32))]
        )
        page = built["pagine"][0]
        self.assertEqual(loc.merge_conflicts(page, *page["blocchi"]), [])

    def test_safe_disjoint_neighbor_does_not_disable_translation(self):
        built = analyzed(
            [line(), line("0400", (100, 100, 130, 112), reason="no_letters")]
        )
        self.assertFalse(built["pagine"][0]["blocchi"][0]["salta"])

    def test_concave_polygon_uses_actual_shape_not_just_bounding_box(self):
        poly = [[0, 0], [10, 0], [10, 3], [3, 3], [3, 10], [0, 10]]
        self.assertFalse(loc.polygon_hits_box(poly, [5, 5, 8, 8]))
        self.assertTrue(loc.polygon_hits_box(poly, [1, 5, 2, 8]))
        self.assertTrue(loc.polygon_hits_box(poly, [2, -1, 4, 1]))

    def test_empty_map_is_rejected_even_after_rehash(self):
        built = analyzed([line()])
        page = built["pagine"][0]
        page["sorgente"]["localizzazione"]["aree"] = []
        page["sorgente"] = sources.seal(page["sorgente"])
        with self.assertRaisesRegex(ValueError, "localizzazione"):
            sources.validate_sources(built)

    def test_roundtrip_and_render_do_not_mutate_source_map(self):
        built = analyzed([line()])
        before = copy.deepcopy(built)
        pipeline.plan_to_objects(built)
        pipeline.plan_to_objects(built)
        self.assertEqual(built, before)


class ReferenceTest(unittest.TestCase):
    def test_independent_reference_can_report_missing_text(self):
        built = analyzed([line()])
        built["sorgente_sha256"] = "test"
        reference = {
            "sorgente_sha256": "test",
            "unita": [
                {"id": "a", "pagina": 1, "box_pt": [10, 10, 40, 16], "testo": "HELLO"},
                {
                    "id": "b",
                    "pagina": 1,
                    "box_pt": [10, 70, 40, 76],
                    "testo": "MISSING",
                },
            ],
        }
        report = loc.compare_reference(built, reference)
        self.assertTrue(report["unita"][0]["trascrizione_presente"])
        self.assertFalse(report["unita"][1]["localizzata"])
        self.assertFalse(report["approvazione_traduzione"])
        self.assertEqual(report["ambito"], "parziale")

    def test_reference_is_bound_to_original_pdf(self):
        with self.assertRaisesRegex(ValueError, "sorgenti diverse"):
            loc.compare_reference({"sorgente_sha256": "a"}, {"sorgente_sha256": "b"})


class TraduzioniSoppresseTest(unittest.TestCase):
    """Rileva i blocchi tradotti che la protezione tiene invisibili.

    La protezione conservativa e' giusta (meglio l'originale che coprire arte),
    ma quando nasconde una traduzione VALIDA lo fa in silenzio: su ASM #7 tre
    didascalie tradotte restavano con l'inglese visibile. Non si allenta la
    protezione: la si dichiara.
    """

    def _blocco(self, testo, originale, salta, **render):
        return {"id": "b1", "originale": originale, "testo": testo,
                "salta": salta, "_render": render}

    def _plan(self, blocco):
        return {"pagine": [{"numero": 3, "blocchi": [blocco]}]}

    def test_traduzione_vera_soppressa_da_source_conflict_e_segnalata(self):
        b = self._blocco("Dove sono in città?", "WHERE AM I", True,
                         source_conflict=["r1"], skip_localizzazione=True)
        got = loc.traduzioni_soppresse(self._plan(b))
        self.assertEqual([x["id"] for x in got], ["b1"])
        self.assertEqual(got[0]["pagina"], 3)

    def test_anche_candidati_compound_e_contenuti_sono_protezioni_automatiche(self):
        """Prova A2 (spidey-63-9): battute tradotte ma saltate senza avviso."""
        for marcatore in ("scartato_come_disegno", "compound", "contenuto_in"):
            with self.subTest(marcatore=marcatore):
                b = self._blocco("Dove sono?", "WHERE AM I", True, **{marcatore: True})
                self.assertEqual([x["id"] for x in loc.traduzioni_soppresse(self._plan(b))],
                                 ["b1"])

    def test_un_salto_deciso_dalla_revisione_non_e_soppressione(self):
        b = self._blocco("Dove sono?", "WHERE AM I", True, compound=True, salta_deciso=True)
        self.assertEqual(loc.traduzioni_soppresse(self._plan(b)), [])

    def test_il_modello_che_chiude_una_battuta_che_ha_tradotto_si_dichiara(self):
        """Prova A3 (superman-1): "Contact! Contact! Open fire!" a p21.

        Natura `dialogo`, traduzione buona, e poi `salta` deciso dal modello
        stesso. Nessun controllo lo vedeva: `salta_deciso` lo toglieva da qui e
        la natura di testo lo toglieva dalle righe "dichiarate" di
        `inglese_render`. Resta saltato, ma va detto.
        """
        b = self._blocco("Contatto! Aprite il fuoco!", "CONTACT! OPEN FIRE!", True,
                         salta_deciso=True)
        b["natura"] = "dialogo"
        got = loc.traduzioni_soppresse(self._plan(b))
        self.assertEqual([(x["id"], x["motivo"]) for x in got], [("b1", "salto_del_modello")])

    def test_una_grafica_tradotta_e_una_contraddizione(self):
        """Le voci della tuta a p19: classificate grafica, ma tradotte."""
        b = self._blocco("Ricarica solare sospesa.", "SOLAR CHARGING SUSPENDED.", True)
        b["natura"] = "grafica"
        got = loc.traduzioni_soppresse(self._plan(b))
        self.assertEqual([(x["id"], x["motivo"]) for x in got],
                         [("b1", "natura_contraddetta")])

    def test_un_logo_lasciato_in_originale_non_e_una_contraddizione(self):
        b = self._blocco("DAILY BUGLE", "DAILY BUGLE", True)
        b["natura"] = "grafica"
        self.assertEqual(loc.traduzioni_soppresse(self._plan(b)), [])

    def test_senza_traduzione_vera_il_salto_del_modello_resta_una_scelta(self):
        b = self._blocco("HEY, YOU!", "HEY, YOU!", True, salta_deciso=True)
        b["natura"] = "dialogo"
        self.assertEqual(loc.traduzioni_soppresse(self._plan(b)), [])

    def test_uno_skip_della_revisione_non_e_una_soppressione(self):
        # il modello ha lasciato l'inglese (testo == originale): non e' una
        # traduzione nascosta dalla protezione
        b = self._blocco("WHERE AM I", "WHERE AM I", True,
                         source_conflict=["r1"], skip_localizzazione=True)
        self.assertEqual(loc.traduzioni_soppresse(self._plan(b)), [])

    def test_un_blocco_disegnato_non_e_soppresso(self):
        b = self._blocco("Dove sono?", "WHERE AM I", False, source_overlap=["r1"])
        self.assertEqual(loc.traduzioni_soppresse(self._plan(b)), [])

    def test_un_salto_senza_protezione_non_e_soppressione(self):
        # salta per scelta della revisione, non per source_conflict
        b = self._blocco("Dove sono?", "WHERE AM I", True, salta_deciso=True)
        self.assertEqual(loc.traduzioni_soppresse(self._plan(b)), [])

    def test_uno_sblocco_a_mano_non_e_piu_soppresso(self):
        b = self._blocco("Dove sono?", "WHERE AM I", True,
                         source_conflict=["r1"], sbloccato_a_mano="area vista")
        self.assertEqual(loc.traduzioni_soppresse(self._plan(b)), [])


class SharedBalloonTest(unittest.TestCase):
    """Due blocchi nello stesso balloon non sono vicini estranei.

    Il rendering riempie tutte le aree prima di scrivere qualunque testo, e
    finche' entrambi vengono riscritti nessuno dei due cancella l'altro.
    """

    doppio = [[10, 10], [120, 10], [120, 130], [10, 130]]

    def _pagina_condivisa(self):
        # due battute nello stesso balloon: rettangoli separati in verticale,
        # ma lo stesso profilo da riempire, che contiene le righe di entrambi
        rows = [line("UNO"), line("DUE", (20, 100, 80, 112))]
        aree = [
            ([15, 15, 110, 70], [self.doppio]),
            ([15, 75, 110, 125], [self.doppio]),
        ]
        with patch("comics_translator.disegno.image_overlay.prepara_area", side_effect=aree):
            return analyzed(rows, expand=True)

    def test_blocchi_dello_stesso_balloon_restano_tradotti(self):
        built = self._pagina_condivisa()
        blocchi = built["pagine"][0]["blocchi"]
        self.assertEqual(len(blocchi), 2)
        for block in blocchi:
            self.assertFalse(block["salta"])
            self.assertNotIn("source_conflict", block["_render"])
            # la sovrapposizione resta registrata: e' un dato della pagina
            self.assertTrue(block["_render"]["source_overlap"])
        sources.validate_sources(built)

    def test_vicino_lasciato_in_originale_torna_a_proteggere(self):
        built = self._pagina_condivisa()
        primo, secondo = built["pagine"][0]["blocchi"]
        secondo["salta"] = True  # come farebbe la revisione o una correzione
        loc.enforce_localization(built)
        self.assertTrue(primo["salta"])
        self.assertEqual(len(primo["_render"]["source_conflict"]), 1)
        self.assertTrue(secondo["salta"])

    def test_riattivare_a_mano_un_blocco_ancora_protetto_e_un_errore(self):
        built = self._pagina_condivisa()
        primo, secondo = built["pagine"][0]["blocchi"]
        secondo["salta"] = True
        loc.enforce_localization(built)
        with self.assertRaisesRegex(ValueError, "source_conflict"):
            apply_corrections(built, {"set": {primo["id"]: {"salta": False}}})

    def test_il_salto_deciso_sopravvive_al_ricalcolo(self):
        """Il ritorno dell'inglese misurato su ASM2: il marcatore vecchio
        faceva credere al ricalcolo che il salto fosse suo, e lo revocava."""
        built = self._pagina_condivisa()
        primo, secondo = built["pagine"][0]["blocchi"]
        secondo["salta"] = True
        loc.enforce_localization(built)
        self.assertTrue(primo["salta"])
        self.assertTrue(primo["_render"]["skip_localizzazione"])

        # Ora primo viene messo da parte per scelta, non per il conflitto.
        corretto, _ = apply_corrections(built, {"set": {primo["id"]: {"salta": True}}})
        deciso = corretto["pagine"][0]["blocchi"][0]
        self.assertTrue(deciso["salta"])
        self.assertNotIn("skip_localizzazione", deciso["_render"])
        self.assertTrue(deciso["_render"]["salta_deciso"])

        # Il conflitto decade: il salto automatico se ne andrebbe, questo no.
        corretto["pagine"][0]["blocchi"][1]["salta"] = False
        loc.enforce_localization(corretto)
        self.assertTrue(deciso["salta"])

    def test_il_salto_automatico_decade_ancora_col_suo_conflitto(self):
        built = self._pagina_condivisa()
        primo, secondo = built["pagine"][0]["blocchi"]
        secondo["salta"] = True
        loc.enforce_localization(built)
        self.assertTrue(primo["salta"])
        secondo["salta"] = False
        loc.enforce_localization(built)
        self.assertFalse(primo["salta"])
        self.assertNotIn("salta_deciso", primo["_render"])

    def test_il_lotto_di_revisione_non_congela_i_salti_automatici(self):
        built = self._pagina_condivisa()
        primo, secondo = built["pagine"][0]["blocchi"]
        secondo["salta"] = True
        loc.enforce_localization(built)

        # La revisione rimanda indietro tutti i blocchi, primo compreso.
        corretto, _ = apply_corrections(
            built, {"set": {primo["id"]: {"salta": True}}}, decisioni=False)
        confermato = corretto["pagine"][0]["blocchi"][0]
        self.assertNotIn("salta_deciso", confermato["_render"])
        corretto["pagine"][0]["blocchi"][1]["salta"] = False
        loc.enforce_localization(corretto)
        self.assertFalse(confermato["salta"])

    def test_lo_sblocco_a_mano_revoca_la_decisione_di_saltare(self):
        built = self._pagina_condivisa()
        primo, secondo = built["pagine"][0]["blocchi"]
        corretto, _ = apply_corrections(built, {"set": {primo["id"]: {"salta": True}}})
        self.assertTrue(corretto["pagine"][0]["blocchi"][0]["_render"]["salta_deciso"])
        finito, _ = apply_corrections(
            corretto, {"sblocca": {primo["id"]: "area guardata a 200 dpi"}})
        sbloccato = finito["pagine"][0]["blocchi"][0]
        self.assertFalse(sbloccato["salta"])
        self.assertNotIn("salta_deciso", sbloccato["_render"])

    def test_enforce_non_cambia_nulla_alla_seconda_passata(self):
        built = self._pagina_condivisa()
        loc.enforce_localization(built)
        prima = copy.deepcopy(built)
        loc.enforce_localization(built)
        self.assertEqual(prima, built)


    def test_illegible_neighbor_is_protected_before_any_fill(self):
        built = self._pagina_condivisa()
        page = built["pagine"][0]
        original_source = copy.deepcopy(page["sorgente"])
        first, second = page["blocchi"]
        second["testo"] = "QUESTA DIDASCALIA NON ENTRA NEL FRAMMENTO RICONOSCIUTO " * 12
        _, jobs = pipeline.plan_to_objects(built)
        marked = pipeline._segna_illeggibili(
            built, jobs, pipeline.load_fontset("arial"), SimpleNamespace(min_size=4.5))
        self.assertEqual(marked, 1)
        self.assertTrue(second["_render"]["testo_illeggibile"])
        loc.enforce_localization(built)
        self.assertTrue(first["salta"])
        self.assertTrue(first["_render"]["source_conflict"])
        _, remaining = pipeline.plan_to_objects(built)
        self.assertFalse(any(remaining.values()))
        self.assertEqual(page["sorgente"], original_source)
        sources.validate_sources(built)

    def test_illegible_flag_is_cleared_once_the_fixed_text_fits(self):
        """Un blocco corretto a mano e riprovato non resta perso per sempre.

        `_segna_illeggibili` marca `testo_illeggibile` quando il testo non
        entra, ma un operatore puo' riportare il blocco a `salta: false` con
        un testo piu' corto che ora entra: se il flag stantio resta,
        `traduzione_non_disegnata` lo dichiara perso in eterno anche se sulla
        tavola c'e' davvero.
        """
        built = self._pagina_condivisa()
        page = built["pagine"][0]
        first, second = page["blocchi"]
        first["_render"]["testo_illeggibile"] = True  # residuo di un tentativo precedente
        _, jobs = pipeline.plan_to_objects(built)
        marked = pipeline._segna_illeggibili(
            built, jobs, pipeline.load_fontset("arial"), SimpleNamespace(min_size=4.5))
        self.assertEqual(marked, 0)
        self.assertNotIn("testo_illeggibile", first["_render"])
        self.assertFalse(first["salta"])

    def test_protection_propagates_transitively_in_any_block_order(self):
        # A copre B; B copre C. C rimane originale: anche A deve fermarsi,
        # ma solo dopo che la protezione di B e' stata propagata.
        rows = [line("UNO"), line("DUE", (20, 100, 80, 112)),
                line("TRE", (20, 180, 80, 192))]
        areas = [([15, 15, 110, 70], [self.doppio]),
                 ([15, 75, 110, 150], [[[10, 90], [120, 90], [120, 210], [10, 210]]]),
                 ([15, 170, 110, 210], [[[10, 170], [120, 170], [120, 210], [10, 210]]])]
        with patch("comics_translator.disegno.image_overlay.prepara_area", side_effect=areas):
            source = analyzed(rows, expand=True)
        self.assertTrue(all(not b["salta"] for b in source["pagine"][0]["blocchi"]))
        skipped_id = source["pagine"][0]["blocchi"][2]["id"]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                built = copy.deepcopy(source)
                page = built["pagine"][0]
                page["blocchi"] = list(reversed(page["blocchi"])) if reverse else page["blocchi"]
                next(b for b in page["blocchi"] if b["id"] == skipped_id)["salta"] = True
                loc.enforce_localization(built)
                self.assertTrue(all(b["salta"] for b in page["blocchi"]))
                stable = copy.deepcopy(built)
                loc.enforce_localization(built)
                self.assertEqual(built, stable)
                self.assertEqual(page["sorgente"], source["pagine"][0]["sorgente"])


class WeakReadingTest(unittest.TestCase):
    def test_lettura_incerta_dentro_il_proprio_balloon_non_protegge(self):
        # la stessa area, riletta male: e' l'inchiostro che il blocco riscrive
        rows = [line(), line("HE1I0", (22, 22, 78, 30), reason="low_confidence")]
        built = analyzed(rows)
        block = built["pagine"][0]["blocchi"][0]
        self.assertFalse(block["salta"])
        self.assertNotIn("source_conflict", block["_render"])

    def test_lettura_incerta_fuori_dall_area_protegge_ancora(self):
        rows = [line(), line("ALTRO", (20, 60, 80, 72), reason="low_confidence")]
        with patch(
            "comics_translator.disegno.image_overlay.prepara_area",
            return_value=([15, 15, 110, 80], [[[15, 15], [110, 15], [110, 80], [15, 80]]]),
        ):
            built = analyzed(rows, expand=True)
        block = built["pagine"][0]["blocchi"][0]
        # la riga cade dentro l'area coperta: nessuna protezione
        self.assertFalse(block["salta"])
        # spostandola fuori dal profilo, invece, la protezione torna
        block["_render"]["fill_poly"] = [[[15, 15], [110, 15], [110, 40], [15, 40]]]
        loc.enforce_localization(built)
        self.assertTrue(block["salta"])

    def test_numero_senza_lettere_resta_protetto_anche_dentro_l_area(self):
        rows = [line(), line("0400", (22, 22, 78, 30), reason="no_letters")]
        built = analyzed(rows)
        block = built["pagine"][0]["blocchi"][0]
        self.assertTrue(block["salta"])
        self.assertEqual(len(block["_render"]["source_conflict"]), 1)


class OldPlanMigrationTest(unittest.TestCase):
    def test_un_piano_vecchio_viene_riaperto_da_enforce(self):
        built = analyzed([line(), line("HE1I0", (22, 22, 78, 30), reason="low_confidence")])
        block = built["pagine"][0]["blocchi"][0]
        # piano nel formato precedente: solo source_conflict, blocco saltato
        block["_render"]["source_conflict"] = block["_render"].pop("source_overlap")
        block["salta"] = True
        loc.enforce_localization(built)
        self.assertFalse(block["salta"])

    def test_un_blocco_saltato_per_scelta_sua_non_viene_riaperto(self):
        built = analyzed([line(), line("HE1I0", (22, 22, 78, 30), reason="low_confidence")])
        block = built["pagine"][0]["blocchi"][0]
        block["salta"] = True  # scelta della revisione, non protezione
        loc.enforce_localization(built)
        self.assertTrue(block["salta"])


class FillBeforeTextTest(unittest.TestCase):
    """Il presupposto su cui poggia la protezione ridotta.

    Se il rendering tornasse a riempire e scrivere blocco per blocco, due
    battute nello stesso balloon si cancellerebbero di nuovo a vicenda.
    """

    def test_tutti_i_riempimenti_precedono_ogni_scrittura(self):
        def job(text):
            return SimpleNamespace(
                translated=text, upper=False, box_px=[0, 0, 20, 20], scale=1.0,
                bg=(255, 255, 255), light_text=False, base_size=8.0,
                fill_poly=[], centered=True,
            )

        doc = fitz.open()
        doc.new_page(width=100, height=100)
        args = SimpleNamespace(min_size=4.0, fill_white=True, grow=1.0)
        fasi = []
        with (
            patch("comics_translator.disegno.pdf_translate_overlay.draw_group",
                  side_effect=lambda **kw: fasi.append(
                      ("riempi" if kw["riempi"] else "scrivi", kw["text"]))),
            tempfile.TemporaryDirectory() as tmp,
        ):
            pipeline._render_document(
                doc, [0], {}, {0: [job("UNO"), job("DUE")]}, None, None,
                args, str(Path(tmp) / "out.pdf"),
            )
        doc.close()
        self.assertEqual([f for f, _ in fasi],
                         ["riempi", "riempi", "scrivi", "scrivi"])


class ManualUnlockTest(unittest.TestCase):
    """Lo sblocco manuale: una deroga per un blocco solo, con il motivo scritto."""

    def _pagina_protetta(self):
        rows = [line(), line("0400", (22, 22, 78, 30), reason="no_letters")]
        return analyzed(rows)

    def test_sblocca_toglie_la_protezione_e_lascia_scritto_il_motivo(self):
        built = self._pagina_protetta()
        block = built["pagine"][0]["blocchi"][0]
        self.assertTrue(block["salta"])
        corretto, changes = apply_corrections(built, {
            "sblocca": {block["id"]: "verificato sul ritaglio: sotto c'e' solo il balloon"},
            "set": {block["id"]: {"testo": "CIAO"}},
        })
        dopo = corretto["pagine"][0]["blocchi"][0]
        self.assertFalse(dopo["salta"])
        self.assertNotIn("source_conflict", dopo["_render"])
        self.assertIn("ritaglio", dopo["_render"]["sbloccato_a_mano"])
        self.assertEqual(dopo["testo"], "CIAO")
        # il piano di partenza non viene toccato
        self.assertTrue(built["pagine"][0]["blocchi"][0]["salta"])

    def test_sblocca_senza_motivo_e_rifiutato(self):
        built = self._pagina_protetta()
        block = built["pagine"][0]["blocchi"][0]
        with self.assertRaisesRegex(ValueError, "motivo"):
            apply_corrections(built, {"sblocca": {block["id"]: "  "}})

    def test_sblocca_su_id_inesistente_e_rifiutato(self):
        built = self._pagina_protetta()
        with self.assertRaisesRegex(ValueError, "id sconosciuti"):
            apply_corrections(built, {"sblocca": {"p9-inesistente": "perche' si"}})

    def test_lo_sblocco_non_si_estende_agli_altri_blocchi(self):
        rows = [
            line("UNO"),
            line("0400", (22, 22, 78, 30), reason="no_letters"),
            line("DUE", (20, 100, 80, 112)),
            line("#$%", (22, 102, 78, 110), reason="no_letters"),
        ]
        built = analyzed(rows)
        primo, secondo = built["pagine"][0]["blocchi"]
        corretto, _ = apply_corrections(built, {"sblocca": {primo["id"]: "verificato"}})
        blocchi = {b["id"]: b for p in corretto["pagine"] for b in p["blocchi"]}
        self.assertFalse(blocchi[primo["id"]]["salta"])
        self.assertTrue(blocchi[secondo["id"]]["salta"])


class LetteringDichiaratoTest(unittest.TestCase):
    """Un logo non si riattiva da solo quando la protezione che lo copriva cade.

    Il ricalcolo revoca una protezione appena il motivo decade e rimette il
    blocco a `salta: false`: da li' il renderer lo DISEGNA, e il riempimento
    copre l'arte. Su imperial-guardians p2 il titolo verticale "JIM SHOOTER",
    alto quanto la tavola, e' sparito sotto un rettangolo bianco non appena i
    blocchi vicini sono stati riattivati a mano (16 settembre 2026).
    """

    def _pagina(self, natura, testo="Jim Shooter"):
        return {
            "numero": 2,
            "blocchi": [{
                "id": "p2-b1", "originale": "Jim shoote er", "testo": testo,
                "salta": True, "natura": natura,
                "_render": {"tipo": "testo", "rect": [0, 0, 10, 100],
                            "source_overlap": ["p2-r1"],
                            "source_conflict": ["p2-r1"], "skip_localizzazione": True},
            }],
            "sorgente": {
                # la riga e' vuota: il motivo della protezione e' caduto
                "righe": [{"id": "p2-r1", "box_pt": [0, 0, 5, 5], "motivo": "empty_text"}],
                "localizzazione": {"aree": [{"riga": "p2-r1", "blocco": None}]},
            },
        }

    def test_un_blocco_grafica_resta_in_originale(self):
        page = self._pagina("grafica")
        loc.enforce_page(page)
        blocco = page["blocchi"][0]
        self.assertTrue(blocco["salta"])
        self.assertTrue(blocco["_render"]["lettering_dichiarato"])

    def test_un_rumore_disegnato_resta_in_originale(self):
        page = self._pagina("rumore", testo="KRAKOOM")
        loc.enforce_page(page)
        self.assertTrue(page["blocchi"][0]["salta"])

    def test_una_battuta_torna_visibile_come_prima(self):
        page = self._pagina("dialogo", testo="Dove sono?")
        loc.enforce_page(page)
        blocco = page["blocchi"][0]
        self.assertFalse(blocco["salta"])
        self.assertNotIn("source_conflict", blocco["_render"])

    def test_senza_natura_il_comportamento_non_cambia(self):
        """I piani precedenti al campo `natura` non devono cambiare esito."""
        page = self._pagina(None, testo="Dove sono?")
        loc.enforce_page(page)
        self.assertFalse(page["blocchi"][0]["salta"])

    def test_una_traduzione_vera_tenuta_ferma_resta_dichiarata(self):
        """Il salto non e' silenzioso: chi ha tradotto davvero viene elencato."""
        page = self._pagina("grafica", testo="Insegna: PERICOLO")
        loc.enforce_page(page)
        soppresse = loc.traduzioni_soppresse({"pagine": [page]})
        self.assertEqual([s["id"] for s in soppresse], ["p2-b1"])
