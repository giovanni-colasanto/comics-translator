"""P3: proprieta' geometriche distinte dalle categorie semantiche gold."""

import copy
import json
from pathlib import Path
import unittest

from comics_translator.sorgente import page_structure as structure
from comics_translator.sorgente import source_inventory as inventory

CASES = json.loads((Path(__file__).parent / "fixtures/p3_structure_cases.json").read_text("utf-8"))["cases"]


def page_for(case, scale=1):
    source = inventory.new_inventory()
    for i, box in enumerate(case["rows"]):
        text = case.get("texts", ["LINE %s" % j for j in range(len(case["rows"]))])[i]
        inventory.add_line(source, 0, raw=text, normalized=text, box=box, kind="test")
    blocks = []
    for members, contours in zip(case["regions"], case["contours"]):
        boxes = [case["rows"][i] for i in members]
        box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes)]
        rid = inventory.add_region(source, 0, [source["righe"][i]["id"] for i in members],
                                   box, "fixture", kind="test")
        blocks.append({"sorgente": {"regione": rid}, "_render": {
            "scale": scale, "fill_poly": [[[v / scale for v in p] for p in structure.rectangle(b)]
                                          for b in contours]}})
    return {"indice": 0, "blocchi": blocks, "sorgente": inventory.seal(source)}


class StructureTest(unittest.TestCase):
    def test_fixture_ownership_order_and_no_semantic_invention(self):
        for case in CASES:
            with self.subTest(case=case["name"]):
                page = page_for(case)
                before = copy.deepcopy(page)
                result = structure.build_structure(page)
                ids = [r["id"] for r in page["sorgente"]["righe"]]
                self.assertEqual([[ids.index(r) for r in c["righe"]] for c in result["candidati"]],
                                 case["expected"])
                self.assertEqual([ids.index(r["riga"]) for r in result["irrisolte"]], case["unresolved"])
                self.assertEqual(result["lettura"]["regioni"], [c["id"] for c in result["candidati"]])
                self.assertTrue(all(c["tipo_testo"] == "non_classificato" for c in result["candidati"]))
                self.assertEqual(page, before)

    def test_physical_contours_do_not_depend_on_dpi(self):
        self.assertEqual(structure.build_structure(page_for(CASES[1], 0.5)),
                         structure.build_structure(page_for(CASES[1], 0.25)))

    def test_excluded_observation_is_not_promoted(self):
        page = page_for(CASES[0])
        page["sorgente"]["righe"][0].update(stato="esclusa", motivo="low_confidence")
        page["sorgente"] = inventory.seal(page["sorgente"])
        result = structure.build_structure(page)
        self.assertEqual(result["candidati"][0]["righe"], [])
        self.assertEqual(result["irrisolte"][0]["motivo_sorgente"], "low_confidence")

    def test_duplicate_and_lost_rows_rejected(self):
        page = page_for(CASES[0])
        data = structure.build_structure(page)
        row = data["candidati"][0]["righe"][0]
        data["candidati"][0]["righe"].append(row)
        with self.assertRaisesRegex(ValueError, "duplicata"):
            structure.validate_structure(page, data)
        data["candidati"][0]["righe"] = []
        with self.assertRaisesRegex(ValueError, "perse"):
            structure.validate_structure(page, data)

    def test_outside_row_rejected(self):
        page = page_for(CASES[0])
        data = structure.build_structure(page)
        data["candidati"][0]["righe"], data["candidati"][1]["righe"] = (
            data["candidati"][1]["righe"], data["candidati"][0]["righe"])
        with self.assertRaisesRegex(ValueError, "fuori regione"):
            structure.validate_structure(page, data)

    def test_order_cannot_repeat_omit_or_promote(self):
        page = page_for(CASES[0])
        for change in ("repeat", "omit", "promote"):
            data = structure.build_structure(page)
            if change == "repeat":
                data["lettura"]["regioni"] *= 2
            elif change == "omit":
                data["lettura"]["regioni"].pop()
            else:
                data["lettura"]["confermato"] = True
            with self.assertRaisesRegex(ValueError, "ordine"):
                structure.validate_structure(page, data)

    def test_hidden_overlap_rejected(self):
        page = page_for(CASES[7])
        data = structure.build_structure(page)
        data["candidati"][0]["conflitti"] = []
        with self.assertRaisesRegex(ValueError, "sovrapposizione"):
            structure.validate_structure(page, data)

    def test_source_hash_change_rejected(self):
        page = page_for(CASES[0])
        data = structure.build_structure(page)
        data["sorgente_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "hash"):
            structure.validate_structure(page, data)
        page["sorgente"]["righe"][0]["testo_grezzo"] = "invented"
        with self.assertRaisesRegex(ValueError, "hash"):
            structure.build_structure(page)

    def test_concave_notch_does_not_contain_whole_row(self):
        row = {"box_pt": [2, 2, 8, 8]}
        poly = [[0, 0], [10, 0], [10, 10], [6, 10], [6, 4], [4, 4], [4, 10], [0, 10]]
        self.assertFalse(structure._inside(row, poly))

    def test_legacy_page_without_structure_remains_compatible(self):
        structure.validate_structure({})


def multilobe_page(*, cross_overlap=True, same_region=False):
    """Fixture P3.2 con tre lobi fisici e associazioni OCR sigillate."""
    source = inventory.new_inventory()
    boxes = [[15, 12, 40, 22], [22, 42, 48, 52], [30, 72, 56, 82]]
    row_ids = [
        inventory.add_line(source, 0, raw=text, normalized=text, box=box, kind="test")
        for text, box in zip(("PRIMO", "SECONDO", "TERZO"), boxes)
    ]
    contours = [
        [[0, 0], [60, 0], [60, 50], [0, 50]],
        [[10, 30], [70, 30], [70, 80], [10, 80]],
        [[20, 60], [80, 60], [80, 110], [20, 110]],
    ]
    blocks = []
    if same_region:
        region_id = inventory.add_region(
            source, 0, row_ids, [15, 12, 56, 82], "fixture", kind="test"
        )
        region_ids = [region_id]
        contour_groups = [[contours[0], contours[2]]]
        block_rows = [row_ids]
    else:
        region_ids = [
            inventory.add_region(source, 0, [row_id], box, "fixture", kind="test")
            for row_id, box in zip(row_ids, boxes)
        ]
        contour_groups = [[contour] for contour in contours]
        block_rows = [[row_id] for row_id in row_ids]
    for index, (region_id, profiles, owned_rows) in enumerate(
        zip(region_ids, contour_groups, block_rows)
    ):
        overlaps = []
        if cross_overlap and not same_region:
            if index > 0:
                overlaps.append(row_ids[index - 1])
            if index + 1 < len(row_ids):
                overlaps.append(row_ids[index + 1])
        blocks.append({
            "id": "b%d" % index,
            "originale": " ".join(
                next(r["testo_normalizzato"] for r in source["righe"] if r["id"] == rid)
                for rid in owned_rows
            ),
            "automatica": "T%d" % index,
            "testo": "T%d" % index,
            "salta": True,
            "sorgente": {"regione": region_id},
            "_source_region_id": region_id,
            "_render": {
                "tipo": "immagine",
                "scale": 1.0,
                "fill_poly": profiles,
                "box_px": [min(p[0] for poly in profiles for p in poly),
                           min(p[1] for poly in profiles for p in poly),
                           max(p[0] for poly in profiles for p in poly),
                           max(p[1] for poly in profiles for p in poly)],
                "bg": [0, 0, 0],
                "light_text": True,
                "source_overlap": overlaps,
                "source_conflict": list(overlaps),
                "skip_localizzazione": True,
            },
        })
    return {"indice": 0, "numero": 1, "blocchi": blocks,
            "sorgente": inventory.seal(source)}


class StructureExecutionTest(unittest.TestCase):
    def test_three_lobes_are_validated_activated_and_adapted(self):
        page = multilobe_page()
        page["struttura"] = structure.build_structure(page)
        units = [u for u in page["struttura"]["unita"] if u["tipo"] == "multilobo"]
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["stato"], "eseguibile")
        self.assertEqual(len(units[0]["righe"]), 3)
        self.assertEqual(structure.activate_validated_units(page), 3)
        self.assertTrue(all(not block["salta"] for block in page["blocchi"]))
        entries = copy.deepcopy(page["blocchi"])
        adapted = structure.adapt_render_entries(page, entries)
        self.assertEqual(entries, page["blocchi"])
        self.assertEqual(len(adapted), 1)
        render = adapted[0]["_render"]
        self.assertEqual(render["p3_lobi"], 3)
        self.assertEqual(render["p3_blocchi"], ["b0", "b1", "b2"])
        self.assertEqual(len(render["fill_poly"]), 3)
        self.assertEqual(adapted[0]["testo"], "T0 T1 T2")
        self.assertEqual(structure.structure_issues({"pagine": [page]}), [])

    def test_touching_speakers_without_cross_ocr_are_not_merged(self):
        page = multilobe_page(cross_overlap=False)
        data = structure.build_structure(page)
        self.assertFalse(any(u["tipo"] == "multilobo" for u in data["unita"]))
        self.assertTrue(data["diagnostica"]["bloccante"])
        self.assertTrue(any(c["conflitti"] for c in data["candidati"]))

    def test_multiple_contours_with_one_text_block_fail_closed(self):
        page = multilobe_page(same_region=True)
        data = structure.build_structure(page)
        unit = next(u for u in data["unita"] if u["tipo"] == "multilobo")
        self.assertEqual(unit["stato"], "incompleta")
        self.assertIn("testo_non_segmentato_per_lobo", unit["motivi"])
        page["struttura"] = data
        self.assertEqual(structure.activate_validated_units(page), 0)
        issues = structure.structure_issues({"pagine": [page]})
        self.assertEqual([item["code"] for item in issues], ["p3_structure_incomplete"])

    def test_translation_skip_and_review_flags_do_not_invalidate_geometry(self):
        page = multilobe_page()
        page["struttura"] = structure.build_structure(page)
        page["blocchi"][0]["salta"] = True
        page["blocchi"][0]["_render"]["unreviewed"] = True
        structure.validate_structure(page)
        structure.activate_validated_units(page)
        self.assertTrue(page["blocchi"][0]["salta"])
        self.assertEqual(len(structure.adapt_render_entries(page)), 3)

    def test_tampered_unit_or_diagnostics_are_rejected(self):
        page = multilobe_page()
        data = structure.build_structure(page)
        broken = copy.deepcopy(data)
        broken["unita"][0]["stato"] = "incompleta"
        with self.assertRaisesRegex(ValueError, "unita"):
            structure.validate_structure(page, broken)
        broken = copy.deepcopy(data)
        broken["diagnostica"]["bloccante"] = not broken["diagnostica"]["bloccante"]
        with self.assertRaisesRegex(ValueError, "diagnostica"):
            structure.validate_structure(page, broken)
