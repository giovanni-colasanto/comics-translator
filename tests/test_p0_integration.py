"""Real local PDF pipeline; AI replaced by deterministic contract responses."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.traduzione import ai_review
from comics_translator.disegno import pdf_translate_overlay as pipeline
from comics_translator import pipeline_status as status


class P0IntegrationTest(unittest.TestCase):
    def exercise(self, missing_translation=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.pdf", root / "output.pdf"
            with pipeline.fitz.open() as document:
                page = document.new_page(width=200, height=100)
                page.insert_text((20, 30), "HELLO", fontsize=12)
                document.save(source)

            def agent(_session, _prompt, workdir, _images, _timeout, output_path=None):
                if output_path.name.startswith("piano.rivisto"):
                    if missing_translation:
                        return None
                    original = json.loads((workdir / "piano.lotto01.json").read_text("utf-8"))
                    spec = {"pagine": [{"numero": p["numero"], "blocchi": [
                        {"id": b["id"], "testo": "CIAO", "salta": False} for b in p["blocchi"]
                    ]} for p in original["pagine"]]}
                else:
                    spec = {"set": {}, "replace": {}}
                output_path.write_text(json.dumps(spec), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            command = ["pdf_translate_overlay.py", str(source), "-o", str(output),
                       "--mode", "text", "--engine", "none", "--ai-review"]
            with (patch("sys.argv", command),
                  patch("comics_translator.traduzione.ai_review.prepare_provider", return_value=ai_review.ProviderSession("codex", "test", "codex")),
                  patch("comics_translator.traduzione.ai_review.run_agent", side_effect=agent),
                  patch("comics_translator.pipeline_status.code_provenance", return_value={} )):
                code = status.cli_entry(pipeline.main)
            manifest = next(root.glob("output.run/runs/*/manifest.json"))
            report = json.loads(manifest.read_text("utf-8"))
            with pipeline.fitz.open(output) as document:
                text = document[0].get_text()
                self.assertEqual(len(document), 1)
            self.assertEqual(report["artifacts"]["source_pdf"]["pages"], 1)
            self.assertEqual(report["quality"], "draft_unverified")
            return code, report, text

    def test_complete_local_pipeline_records_all_phases(self):
        code, report, text = self.exercise()
        self.assertEqual(code, 0)
        self.assertIn("CIAO", text)
        self.assertEqual(set(report["phases"]),
                         {"analysis", "structure", "translation", "render", "qa",
                          "riparazione", "structure_audit"})
        self.assertTrue(all(p["status"] == "complete" for p in report["phases"].values()))

    def test_l_audit_raster_gira_da_solo_e_lascia_il_rapporto(self):
        """P3.4 non e' piu' un comando da ricordarsi: la run si misura da sola."""
        _, report, _ = self.exercise()
        prova = report["artifacts"]["audit_report"]
        self.assertTrue(prova["path"].endswith("audit.json"))
        self.assertGreater(prova["bytes"], 0)
        fase = report["phases"]["structure_audit"]
        self.assertEqual(fase["unit"], "pages")
        self.assertEqual(fase["issues"], [])
        self.assertEqual(fase["status"], "complete")

    def test_missing_translation_preserves_original_and_exits_partial(self):
        code, report, text = self.exercise(missing_translation=True)
        self.assertEqual(code, 2)
        self.assertIn("HELLO", text)
        self.assertEqual(report["phases"]["translation"]["status"], "failed")
        self.assertEqual(report["coverage"]["skipped"][0]["code"], "unreviewed")


if __name__ == "__main__":
    unittest.main()
