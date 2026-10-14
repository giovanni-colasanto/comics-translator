import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.disegno import pdf_translate_overlay as pipeline
from comics_translator.verifica import visual_qa
from comics_translator.disegno.plan import rebuild_page, validate_merges
from comics_translator.verifica.qa_corrections import apply_corrections


def merge_plan():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    return {"pagine": [{"numero": 1, "indice": 0, "blocchi": [
        {"id": "a", "testo": "Prima", "originale": "First", "salta": False,
         "_render": {"tipo": "immagine", "box_px": [0, 0, 100, 45], "fill_poly": [poly]}},
        {"id": "b", "testo": "seconda", "originale": "second", "salta": False,
         "unisci_con": "a", "_render": {
             "tipo": "immagine", "box_px": [0, 50, 100, 100], "fill_poly": [poly]}},
    ]}]}


class RenderRegressionTest(unittest.TestCase):
    def test_two_renders_leave_plan_and_merge_validation_unchanged(self):
        source = merge_plan()
        expected = copy.deepcopy(source)
        entries = source["pagine"][0]["blocchi"]
        self.assertEqual(validate_merges(source), [])
        first = rebuild_page(entries, SimpleNamespace, SimpleNamespace)
        self.assertEqual(source, expected)
        self.assertEqual(validate_merges(source), [])
        second = rebuild_page(entries, SimpleNamespace, SimpleNamespace)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 1)
        self.assertEqual(first[1][0].translated, "Prima seconda")
        corrected, count = apply_corrections(source, {"set": {"a": {"testo": "Inizio"}}})
        self.assertEqual(count, 1)
        self.assertEqual(validate_merges(corrected), [])

    def test_text_pdf_merges_are_also_repeatable(self):
        source = merge_plan()
        for block in source["pagine"][0]["blocchi"]:
            box = block["_render"]["box_px"]
            block["_render"] = {"tipo": "testo", "box": box, "rect": list(box)}
        expected = copy.deepcopy(source)
        first = pipeline.plan_to_objects(source)
        second = pipeline.plan_to_objects(source)
        self.assertEqual(first, second)
        self.assertEqual(source, expected)


class VisualSafetyRegressionTest(unittest.TestCase):
    def test_skipped_blocks_cannot_be_reactivated_automatically(self):
        for flag in ("compound", "page_protected", "contenuto_in", None):
            with self.subTest(flag=flag):
                plan = merge_plan()
                a, b = plan["pagine"][0]["blocchi"]
                b["unisci_con"] = None
                a["salta"] = True
                if flag:
                    a["_render"][flag] = True
                with self.assertRaisesRegex(ValueError, "protetto"):
                    visual_qa.apply_visual_corrections(plan, {"set": {"a": {"salta": False}}}, {"a", "b"})
                self.assertTrue(a["salta"])

    def test_correction_cannot_change_other_batches(self):
        with self.assertRaisesRegex(ValueError, "fuori dal lotto"):
            visual_qa.apply_visual_corrections(merge_plan(), {"set": {"b": {"testo": "fuori"}}}, {"a"})

    def test_literal_newline_escape_is_not_rendered_as_text(self):
        with self.assertRaisesRegex(ValueError, "escape letterale"):
            visual_qa.apply_visual_corrections(
                merge_plan(), {"set": {"a": {"testo": r"prima\nseconda"}}}, {"a", "b"},
            )

    def test_merge_cannot_absorb_skipped_content(self):
        plan = merge_plan()
        plan["pagine"][0]["blocchi"][0]["salta"] = True
        with self.assertRaisesRegex(ValueError, "protetto"):
            visual_qa.apply_visual_corrections(plan, {"set": {"b": {"unisci_con": "a"}}}, {"a", "b"})

    def _review(self, outcomes):
        plan = merge_plan()
        for index in range(1, len(outcomes)):
            plan["pagine"].append({"numero": index + 1, "indice": index, "blocchi": []})
        session = SimpleNamespace(ai="codex", model="gpt-test", cli="codex")
        calls = iter(enumerate(outcomes, 1))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_run(*args, **kwargs):
                number, code = next(calls)
                (root / ("correzioni.visuali.lotto%02d.json" % number)).write_text(
                    json.dumps({"set": {}, "replace": {}}), encoding="utf-8",
                )
                return None if code is None else SimpleNamespace(returncode=code)

            with (
                patch("comics_translator.verifica.visual_qa.ai_review.prepare_provider", return_value=session),
                patch("comics_translator.verifica.visual_qa.export_comparisons", return_value={
                    p: (root / "a.png", root / "b.png") for p in range(len(outcomes))
                }),
                patch("comics_translator.verifica.visual_qa.ai_review.run_agent", side_effect=fake_run),
            ):
                return visual_qa.review_rendered(root / "s.pdf", root / "r.pdf", plan,
                                                 list(range(len(outcomes))), root, batch=1)

    def test_timeout_is_incomplete_even_when_a_file_exists(self):
        result = self._review([None])
        self.assertFalse(result.complete)
        self.assertEqual(result.validated_batches, 0)

    def test_failed_process_is_incomplete_even_when_json_is_valid(self):
        result = self._review([1])
        self.assertFalse(result.complete)
        self.assertEqual(result.changes, 0)

    def test_valid_empty_spec_is_a_completed_review(self):
        result = self._review([0])
        self.assertTrue(result.complete)
        self.assertEqual(result.changes, 0)

    def test_partial_review_keeps_explicit_counts(self):
        result = self._review([0, None])
        self.assertFalse(result.complete)
        self.assertEqual((result.validated_batches, result.total_batches), (1, 2))

    def test_pipeline_never_reports_incomplete_review_as_clean(self):
        plan = merge_plan()
        args = SimpleNamespace(visual_qa=True, review_dir=Path("review"), review_ai="codex",
                               review_model="gpt-test", review_timeout=5, review_batch=2)
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("comics_translator.verifica.visual_qa.review_rendered", return_value=visual_qa.ReviewResult(plan, 0, 0, 1)),
            patch("comics_translator.disegno.pdf_translate_overlay.render_plan") as render,
            redirect_stdout(stdout), redirect_stderr(stderr),
        ):
            result = pipeline.run_visual_qa(Path("s.pdf"), Path("r.pdf"), plan, [0], None, args)
        self.assertIs(result, plan)
        self.assertIn("INCOMPLETO", stderr.getvalue())
        self.assertIn("non verificato", stdout.getvalue())
        self.assertNotIn("nessuna correzione necessaria", stdout.getvalue())
        render.assert_not_called()
