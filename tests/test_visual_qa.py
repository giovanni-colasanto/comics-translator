import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.verifica import visual_qa
from comics_translator.disegno import pdf_translate_overlay as pipeline


def sample_plan():
    return {
        "pagine": [
            {
                "numero": 1,
                "indice": 0,
                "blocchi": [
                    {
                        "id": "p001-b001",
                        "originale": "THE NAME IS NOVA",
                        "testo": "Il nome e' Nuova",
                        "salta": False,
                        "_render": {
                            "tipo": "immagine",
                            "box_px": [0, 0, 100, 50],
                        },
                    }
                ],
            },
            {
                "numero": 2,
                "indice": 1,
                "blocchi": [],
            },
        ]
    }


class VisualQaPlanTest(unittest.TestCase):
    def test_selected_pages_are_slim_and_keep_source_untouched(self):
        source = sample_plan()
        selected = visual_qa.selected_plan_pages(source, [0])

        self.assertEqual(len(selected), 1)
        self.assertNotIn("_render", selected[0]["blocchi"][0])
        self.assertIn("_render", source["pagine"][0]["blocchi"][0])

    def test_batches_never_use_zero_as_step(self):
        batches = visual_qa.make_batches(sample_plan(), [0, 1], 0)
        self.assertEqual([len(batch) for batch in batches], [1, 1])

    def test_spec_rejects_unexpected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text('{"set": {}, "note": "non ammessa"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "chiavi non ammesse"):
                visual_qa._load_spec(path)


class VisualQaReviewTest(unittest.TestCase):
    def test_valid_correction_is_applied_without_mutating_source(self):
        source = sample_plan()
        session = SimpleNamespace(ai="codex", model="gpt-test", cli="codex")

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            original = workdir / "p01.originale.png"
            rendered = workdir / "p01.tradotta.png"

            def fake_run_agent(_session, _prompt, current_workdir, _images, _timeout, **kwargs):
                spec = {
                    "set": {"p001-b001": {"testo": "Il nome e' Nova"}},
                    "replace": {},
                }
                (current_workdir / "correzioni.visuali.lotto01.json").write_text(
                    json.dumps(spec), encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("comics_translator.verifica.visual_qa.ai_review.prepare_provider", return_value=session),
                patch(
                    "comics_translator.verifica.visual_qa.export_comparisons",
                    return_value={0: (original, rendered)},
                ),
                patch("comics_translator.verifica.visual_qa.ai_review.run_agent", side_effect=fake_run_agent),
            ):
                result = visual_qa.review_rendered(
                    Path("source.pdf"), Path("rendered.pdf"), source, [0], workdir,
                )

        corrected, changes = result.plan, result.changes
        self.assertTrue(result.complete)
        self.assertEqual(changes, 1)
        self.assertEqual(
            corrected["pagine"][0]["blocchi"][0]["testo"], "Il nome e' Nova"
        )
        self.assertEqual(
            source["pagine"][0]["blocchi"][0]["testo"], "Il nome e' Nuova"
        )

    def test_invalid_ai_output_is_ignored(self):
        source = sample_plan()
        session = SimpleNamespace(ai="codex", model="gpt-test", cli="codex")

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            def fake_run_agent(_session, _prompt, current_workdir, _images, _timeout, **kwargs):
                (current_workdir / "correzioni.visuali.lotto01.json").write_text(
                    '{"set": {"missing": {"salta": true}}}', encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("comics_translator.verifica.visual_qa.ai_review.prepare_provider", return_value=session),
                patch(
                    "comics_translator.verifica.visual_qa.export_comparisons",
                    return_value={0: (workdir / "a.png", workdir / "b.png")},
                ),
                patch("comics_translator.verifica.visual_qa.ai_review.run_agent", side_effect=fake_run_agent),
            ):
                result = visual_qa.review_rendered(
                    Path("source.pdf"), Path("rendered.pdf"), source, [0], workdir,
                )

        corrected, changes = result.plan, result.changes
        self.assertFalse(result.complete)
        self.assertEqual(changes, 0)
        self.assertEqual(corrected, source)


class QaProviderResolutionTest(unittest.TestCase):
    """Il QA puo' usare un modello diverso dalla traduzione (verifica incrociata)."""

    def args(self, **kw):
        base = dict(review_ai="codex", review_model="gpt-5.4", qa_ai=None, qa_model=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_senza_qa_ai_il_qa_eredita_provider_e_modello_della_traduzione(self):
        # Comportamento identico a prima del cambiamento.
        self.assertEqual(pipeline.risolvi_qa_provider(self.args()), ("codex", "gpt-5.4"))

    def test_qa_ai_diverso_usa_il_suo_modello_di_default_non_quello_di_traduzione(self):
        # glm non deve ereditare "gpt-5.4", che e' di un altro provider.
        self.assertEqual(pipeline.risolvi_qa_provider(self.args(qa_ai="glm")), ("glm", None))

    def test_qa_model_esplicito_vince_con_qualunque_provider(self):
        got = pipeline.risolvi_qa_provider(self.args(qa_ai="claude", qa_model="opus"))
        self.assertEqual(got, ("claude", "opus"))

    def test_qa_model_senza_qa_ai_resta_sul_provider_di_traduzione(self):
        got = pipeline.risolvi_qa_provider(self.args(qa_model="gpt-5.5"))
        self.assertEqual(got, ("codex", "gpt-5.5"))

    def test_run_visual_qa_passa_il_provider_qa_a_review_rendered(self):
        captured = {}

        def finto_review(**kw):
            captured.update(kw)
            return SimpleNamespace(plan=kw["plan"], changes=0, complete=True,
                                   validated_batches=1, total_batches=1)

        args = self.args(qa_ai="glm", visual_qa=True, review_dir=None,
                         review_timeout=1800, review_batch=2)
        with patch.object(pipeline.visual_qa, "review_rendered", side_effect=finto_review):
            pipeline.run_visual_qa(Path("s.pdf"), Path("o.pdf"),
                                   {"pagine": []}, [0], None, args)
        self.assertEqual(captured["ai"], "glm")
        self.assertIsNone(captured["model"])


if __name__ == "__main__":
    unittest.main()
