import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.disegno import pdf_translate_overlay as pipeline


def options(review_dir: Path):
    return SimpleNamespace(
        visual_qa=True,
        review_dir=review_dir,
        review_ai="codex",
        review_model="gpt-test",
        review_timeout=30,
        review_batch=2,
    )


class VisualQaPipelineTest(unittest.TestCase):
    def test_changed_output_preserves_pre_qa_and_renders_corrected_plan(self):
        plan = {"pagine": []}
        corrected = {"pagine": [], "corretto": True}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "finale.it.pdf"
            output.write_bytes(b"primo rendering")
            args = options(root / "revisione")

            with (
                patch(
                    "comics_translator.disegno.pdf_translate_overlay.visual_qa.review_rendered",
                    return_value=pipeline.visual_qa.ReviewResult(corrected, 2, 1, 1),
                ),
                patch("comics_translator.disegno.pdf_translate_overlay.save_plan") as save,
                patch("comics_translator.disegno.pdf_translate_overlay.render_plan") as render,
            ):
                result = pipeline.run_visual_qa(
                    root / "source.pdf", output, plan, [0], object(), args,
                )

            pre_qa = root / "finale.it.pre-qa.pdf"
            self.assertEqual(pre_qa.read_bytes(), b"primo rendering")
            self.assertIs(result, corrected)
            save.assert_called_once_with(
                args.review_dir / "piano.controllato.json", corrected
            )
            render.assert_called_once_with(
                root / "source.pdf", corrected, [0], unittest.mock.ANY, args, output
            )

    def test_clean_output_is_not_rendered_twice(self):
        plan = {"pagine": []}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "finale.it.pdf"
            output.write_bytes(b"rendering valido")
            args = options(root / "revisione")

            with (
                patch(
                    "comics_translator.disegno.pdf_translate_overlay.visual_qa.review_rendered",
                    return_value=pipeline.visual_qa.ReviewResult(plan, 0, 1, 1),
                ),
                patch("comics_translator.disegno.pdf_translate_overlay.render_plan") as render,
            ):
                result = pipeline.run_visual_qa(
                    root / "source.pdf", output, plan, [0], object(), args,
                )

            self.assertIs(result, plan)
            self.assertFalse((root / "finale.it.pre-qa.pdf").exists())
            render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
