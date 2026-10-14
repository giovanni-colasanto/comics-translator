"""P1: dati reali nei formati, geometrie sintetiche, nessuna chiamata AI."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pymupdf as fitz

from comics_translator.traduzione import ai_review
from comics_translator.disegno import image_overlay as images
from comics_translator.disegno import pdf_translate_overlay as pipeline
from comics_translator.disegno import plan
from comics_translator.sorgente import source_inventory as inventory
from comics_translator.verifica import visual_qa
from comics_translator.verifica.qa_corrections import apply_corrections


def result_row(text, y, confidence=0.98, factor=1):
    quad = [
        [v * factor for v in p] for p in [[20, y], [80, y], [80, y + 12], [20, y + 12]]
    ]
    return [quad, text, confidence]


def analyzed(rows, dpi=144, effective_dpi=None):
    data = inventory.new_inventory()
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 300))
    with (
        patch("comics_translator.disegno.image_overlay._engine", return_value=lambda img: (rows, None)),
        patch(
            "comics_translator.disegno.image_overlay.page_to_array",
            return_value=(np.full((800, 400, 3), 255, np.uint8), effective_dpi or dpi),
        ),
    ):
        # P1 fixtures verify the pre-P2 selection contract explicitly.
        jobs, _, _ = images.plan_page(page, 0, dpi, 0.5, False, inventory=data, localize=False)
    for job in jobs:
        job.translated = "NO!"
    built = plan.build_plan([page], [0], {}, {0: jobs}, {}, sources={0: data})
    return built, data, jobs


class InventoryTest(unittest.TestCase):
    def test_every_ocr_observation_survives_including_exclusions(self):
        rows = [
            result_row("  HELLO  ", 20),
            result_row("0400", 60),
            result_row("#$%&", 100),
            result_row("LOW", 140, 0.1),
            result_row(" ", 180),
            result_row("I", 220),
        ]
        built, _, jobs = analyzed(rows)
        source = built["pagine"][0]["sorgente"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(source["righe"]), 6)
        self.assertEqual(
            [line["testo_grezzo"] for line in source["righe"]], [r[1] for r in rows]
        )
        self.assertEqual(source["righe"][0]["testo_normalizzato"], "HELLO")
        self.assertEqual(
            [line["motivo"] for line in source["righe"]],
            [
                None,
                "no_letters",
                "no_letters",
                "low_confidence",
                "empty_text",
                "too_short",
            ],
        )
        self.assertEqual(source["righe"][0]["quad_px"], rows[0][0])
        self.assertEqual(source["righe"][0]["confidenza"], 0.98)
        inventory.validate_sources(built)

    def test_background_exclusion_keeps_text_and_geometry(self):
        with patch("comics_translator.disegno.image_overlay.background_noise", return_value=30):
            built, _, jobs = analyzed([result_row("MISSED DIALOGUE", 20)])
        # Il gruppo scartato con forma di testo non sparisce piu': diventa un
        # candidato saltato che arriva al modello (Gap C). Non copre niente.
        self.assertEqual([job.scartato_come_disegno for job in jobs], [True])
        (blocco,) = built["pagine"][0]["blocchi"]
        self.assertTrue(blocco["salta"])
        self.assertTrue(blocco["_render"]["scartato_come_disegno"])
        source = built["pagine"][0]["sorgente"]
        self.assertEqual(source["righe"][0]["motivo"], "background_noise")
        self.assertEqual(source["regioni"][0]["rumore_sfondo"], 30)
        self.assertEqual(source["regioni"][0]["righe"], [source["righe"][0]["id"]])

    def test_physical_coordinates_and_ids_do_not_depend_on_pixel_scale(self):
        first, _, _ = analyzed([result_row("NO!", 20)], dpi=144)
        second, _, _ = analyzed([result_row("NO!", 20, factor=2)], dpi=288)
        a, b = (p["pagine"][0] for p in (first, second))
        self.assertEqual(a["blocchi"][0]["id"], b["blocchi"][0]["id"])
        self.assertEqual(
            a["sorgente"]["righe"][0]["box_pt"], b["sorgente"]["righe"][0]["box_pt"]
        )
        self.assertNotEqual(
            a["sorgente"]["righe"][0]["box_px"], b["sorgente"]["righe"][0]["box_px"]
        )

    def test_effective_not_requested_dpi_drives_coordinates(self):
        built, _, _ = analyzed([result_row("NO!", 20)], dpi=400, effective_dpi=144)
        source = built["pagine"][0]["sorgente"]
        self.assertEqual(source["acquisizioni"][0]["dpi_richiesto"], 400)
        self.assertEqual(source["acquisizioni"][0]["dpi_effettivo"], 144)
        self.assertEqual(source["righe"][0]["box_pt"], [10, 10, 40, 16])

    def test_repeated_and_identical_detections_have_unique_ids(self):
        for positions in ((20, 80), (20, 20)):
            built, _, _ = analyzed([result_row("NO!", y) for y in positions])
            blocks = built["pagine"][0]["blocchi"]
            self.assertEqual(len(blocks), 2)
            self.assertNotEqual(blocks[0]["id"], blocks[1]["id"])
            self.assertEqual(blocks[0]["legacy_id"], blocks[1]["legacy_id"])
            inventory.validate_sources(built)

    def test_native_short_numeric_and_rotated_lines_are_preserved(self):
        with fitz.open() as doc:
            page = doc.new_page(width=200, height=300)
            page.insert_text((20, 30), "HELLO")
            page.insert_text((20, 70), "0400")
            page.insert_text((20, 110), "I")
            page.insert_text((150, 200), "ROTATED", rotate=90)
            data = inventory.new_inventory()
            blocks, _ = pipeline.extract_blocks(page, 0, 2, inventory=data)
            built = plan.build_plan(doc, [0], {0: blocks}, {}, {}, sources={0: data})
        self.assertEqual(len(blocks), 1)
        lines = built["pagine"][0]["sorgente"]["righe"]
        self.assertEqual(len(lines), 4)
        self.assertEqual(
            [r["motivo"] for r in lines],
            [None, "no_letters", "too_short", "rotated_text"],
        )
        self.assertTrue(all(r["quad_px"] is None for r in lines))
        inventory.validate_sources(built)

    def test_roundtrip_corrections_and_rendering_leave_source_unchanged(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        expected = copy.deepcopy(built)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan.save_plan(path, built)
            loaded = plan.load_plan(path)
        self.assertEqual(loaded, expected)
        bid = built["pagine"][0]["blocchi"][0]["id"]
        corrected, _ = apply_corrections(built, {"set": {bid: {"testo": "CIAO"}}})
        self.assertEqual(
            corrected["pagine"][0]["sorgente"], expected["pagine"][0]["sorgente"]
        )
        self.assertEqual(
            pipeline.plan_to_objects(corrected), pipeline.plan_to_objects(corrected)
        )
        self.assertEqual(built, expected)

    def test_source_change_and_broken_reference_are_rejected(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        broken = copy.deepcopy(built)
        broken["pagine"][0]["sorgente"]["righe"][0]["testo_grezzo"] = "invented"
        with self.assertRaisesRegex(ValueError, "hash"):
            inventory.validate_sources(broken)
        built["pagine"][0]["blocchi"][0]["sorgente"]["righe"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "riferimenti"):
            inventory.validate_sources(built)

    def test_inconsistent_coordinates_are_rejected_even_with_updated_hash(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        page = built["pagine"][0]
        page["sorgente"]["righe"][0]["box_pt"][0] += 1
        page["sorgente"] = inventory.seal(page["sorgente"])
        with self.assertRaisesRegex(ValueError, "px/pt"):
            inventory.validate_sources(built)

    def test_ai_projections_do_not_include_inventory_or_legacy_aliases(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        for pages in (
            ai_review._senza_render(built["pagine"]),
            visual_qa.selected_plan_pages(built, [0]),
        ):
            text = json.dumps(pages)
            self.assertNotIn('"sorgente"', text)
            self.assertNotIn('"legacy_id"', text)
            self.assertNotIn('"_render"', text)
            self.assertNotIn('"struttura"', text)
        inventory.validate_sources(built)


class LegacyTest(unittest.TestCase):
    def legacy(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        built["versione"] = 1
        for page in built["pagine"]:
            page.pop("sorgente")
            page.pop("struttura", None)  # P3 non esisteva nello schema v1.
            for block in page["blocchi"]:
                block.pop("sorgente")
                block["id"] = block.pop("legacy_id")
        return built

    def test_migration_is_explicit_idempotent_and_preserves_rendering(self):
        legacy = self.legacy()
        expected = copy.deepcopy(legacy)
        upgraded = plan.migrate_v1(legacy)
        self.assertEqual(legacy, expected)
        self.assertEqual(upgraded["pagine"][0]["sorgente"]["righe"], [])
        self.assertEqual(
            upgraded["pagine"][0]["sorgente"]["disponibilita"], "legacy_senza_righe"
        )
        self.assertEqual(plan.migrate_v1(upgraded), upgraded)
        self.assertEqual(
            pipeline.plan_to_objects(legacy), pipeline.plan_to_objects(upgraded)
        )

    def test_load_v1_does_not_migrate_or_rewrite_file(self):
        legacy = self.legacy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan.save_plan(path, legacy)
            before = path.read_bytes()
            self.assertEqual(plan.load_plan(path), legacy)
            self.assertEqual(path.read_bytes(), before)

    def test_duplicate_legacy_id_refuses_ambiguous_migration(self):
        legacy = self.legacy()
        legacy["pagine"][0]["blocchi"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicato"):
            plan.migrate_v1(legacy)

    def test_unknown_schema_and_incomplete_v2_are_rejected(self):
        for version in (0, 3, "2", True):
            with (
                self.subTest(version=version),
                self.assertRaisesRegex(ValueError, "versione"),
            ):
                inventory.validate_sources({"versione": version, "pagine": []})
        with self.assertRaisesRegex(ValueError, "inventario"):
            inventory.validate_sources({"versione": 2, "pagine": [{"blocchi": []}]})

    def test_old_correction_id_resolves_only_when_unique(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        alias = built["pagine"][0]["blocchi"][0]["legacy_id"]
        corrected, changes = apply_corrections(
            built, {"set": {alias: {"testo": "CIAO"}}}
        )
        self.assertEqual(changes, 1)
        self.assertEqual(corrected["pagine"][0]["blocchi"][0]["testo"], "CIAO")
        repeated, _, _ = analyzed([result_row("HELLO", 20), result_row("HELLO", 80)])
        with self.assertRaisesRegex(ValueError, "ambiguo"):
            apply_corrections(repeated, {"set": {alias: {"testo": "CIAO"}}})

    def test_alias_and_current_id_cannot_apply_twice(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        block = built["pagine"][0]["blocchi"][0]
        spec = {
            "set": {block["id"]: {"testo": "A"}, block["legacy_id"]: {"testo": "B"}}
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            apply_corrections(built, spec)


class ContractTest(unittest.TestCase):
    def test_fractional_ocr_pixels_survive_without_double_rounding(self):
        row = result_row("HELLO", 20)
        row[0] = [[p[0] + 0.123456789, p[1] + 0.987654321] for p in row[0]]
        built, _, _ = analyzed([row], dpi=400)
        line = built["pagine"][0]["sorgente"]["righe"][0]
        self.assertEqual(line["quad_px"], row[0])
        self.assertEqual(line["box_px"][0], row[0][0][0])
        inventory.validate_sources(built)

    def test_acquisition_and_candidate_reading_orders_are_distinct(self):
        built, _, _ = analyzed([result_row("BOTTOM", 100), result_row("TOP", 20)])
        source = built["pagine"][0]["sorgente"]
        self.assertEqual(
            [r["testo_grezzo"] for r in source["righe"]], ["BOTTOM", "TOP"]
        )
        self.assertEqual([r["ordine"] for r in source["righe"]], [0, 1])
        by_id = {r["id"]: r for r in source["regioni"]}
        ordered = [by_id[ident]["box_pt"][1] for ident in source["lettura"]["regioni"]]
        self.assertEqual(ordered, sorted(ordered))
        self.assertFalse(source["lettura"]["confermato"])

    def test_duplicate_canonical_block_id_is_rejected(self):
        built, _, _ = analyzed([result_row("HELLO", 20), result_row("BYE", 100)])
        blocks = built["pagine"][0]["blocchi"]
        blocks[1]["id"] = blocks[0]["id"]
        with self.assertRaisesRegex(ValueError, "id blocco.*duplicato"):
            inventory.validate_sources(built)

    def test_corrupt_order_and_quadrilateral_are_rejected(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        for kind in ("order", "quad"):
            broken = copy.deepcopy(built)
            page = broken["pagine"][0]
            if kind == "order":
                page["sorgente"]["lettura"]["regioni"] = []
            else:
                page["sorgente"]["righe"][0]["quad_px"][0][0] += 10
            page["sorgente"] = inventory.seal(page["sorgente"])
            with self.assertRaisesRegex(ValueError, "lettura|quadrilatero"):
                inventory.validate_sources(broken)

    def test_acquired_row_cannot_disappear_from_all_regions(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        page = built["pagine"][0]
        page["sorgente"]["regioni"][0]["righe"] = []
        page["sorgente"] = inventory.seal(page["sorgente"])
        with self.assertRaisesRegex(ValueError, "senza regione"):
            inventory.validate_sources(built)

    def test_render_boundary_rejects_mutated_source(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        built["pagine"][0]["sorgente"]["righe"][0]["testo_grezzo"] = "CHANGED"
        with self.assertRaisesRegex(ValueError, "hash"):
            pipeline.plan_to_objects(built)

    def test_bad_merge_type_uses_clear_qa_error(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        ident = built["pagine"][0]["blocchi"][0]["id"]
        with self.assertRaisesRegex(ValueError, "unisci_con"):
            apply_corrections(built, {"set": {ident: {"unisci_con": ["invalid"]}}})

    def test_migration_cli_preserves_original_and_refuses_existing_destination(self):
        legacy = LegacyTest().legacy()
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp) / "v1.json", Path(tmp) / "v2.json"
            plan.save_plan(source, legacy)
            original_bytes = source.read_bytes()
            plan.main(["--migrate-v2", str(source), "--out", str(destination)])
            self.assertEqual(plan.load_plan(destination)["versione"], 2)
            self.assertEqual(source.read_bytes(), original_bytes)
            result_bytes = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                plan.migrate_plan_file(source, destination)
            with self.assertRaisesRegex(ValueError, "diverso"):
                plan.migrate_plan_file(source, source)
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual(destination.read_bytes(), result_bytes)

    def test_render_rejects_a_different_source_pdf_before_drawing(self):
        built, _, _ = analyzed([result_row("HELLO", 20)])
        built["sorgente_sha256"] = "a" * 64
        with (
            patch("comics_translator.disegno.pdf_translate_overlay.status.sha256", return_value="b" * 64),
            patch("comics_translator.disegno.pdf_translate_overlay._render_document") as draw,
        ):
            with self.assertRaisesRegex(ValueError, "PDF sorgente"):
                pipeline.render_plan("other.pdf", built, [0], None, None, "unused.pdf")
            draw.assert_not_called()

    def test_empty_page_is_represented_without_invented_rows(self):
        page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 300))
        built = plan.build_plan(
            [page], [0], {}, {}, {}, sources={0: inventory.new_inventory()}
        )
        self.assertEqual(built["pagine"][0]["sorgente"]["righe"], [])
        self.assertEqual(built["pagine"][0]["sorgente"]["lettura"]["regioni"], [])
        inventory.validate_sources(built)
