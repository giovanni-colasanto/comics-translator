"""Cassette: registrazione e replay delle chiamate al modello, tutto offline.

Nessun subprocess reale: la CLI e' vietata (AssertionError) in ogni test.
"""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.traduzione import ai_review
from comics_translator.traduzione import cassette

ROOT = Path(__file__).resolve().parents[1]


def _png(cartella: Path, nome: str, contenuto: bytes) -> Path:
    percorso = cartella / nome
    percorso.write_bytes(contenuto)
    return percorso


class BaseCassetteTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.originale = dict(cassette._stato)
        cassette._stato.update(dir=None, registra=False, scritti=0, riprodotti=0)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        cassette._stato.clear()
        cassette._stato.update(self.originale)
        super().tearDown()


class ChiaveTest(BaseCassetteTest):
    def test_stessa_richiesta_stessa_chiave(self):
        immagini = [_png(self.tmp, "p01.png", b"a"), _png(self.tmp, "p02.png", b"b")]
        sha = [hashlib.sha256(p.read_bytes()).hexdigest() for p in immagini]
        prima = cassette.chiave("glm", "glm-5.3-flash", "abc", sha)
        seconda = cassette.chiave("glm", "glm-5.3-flash", "abc", sha)
        self.assertEqual(prima, seconda)

    def test_modello_o_immagine_diversi_cambiano_la_chiave(self):
        immagini = [_png(self.tmp, "p01.png", b"a")]
        sha = [hashlib.sha256(immagini[0].read_bytes()).hexdigest()]
        base = cassette.chiave("glm", "glm-5.3-flash", "abc", sha)
        self.assertNotEqual(base, cassette.chiave("codex", "glm-5.3-flash", "abc", sha))
        self.assertNotEqual(base, cassette.chiave("glm", "haiku", "abc", sha))
        self.assertNotEqual(base, cassette.chiave("glm", "glm-5.3-flash", "xyz", sha))
        altre = [hashlib.sha256(b"diverso").hexdigest()]
        self.assertNotEqual(base, cassette.chiave("glm", "glm-5.3-flash", "abc", altre))


class RegistrazioneTest(BaseCassetteTest):
    def test_registrazione_scrive_cassette_e_manifest(self):
        corpus = self.tmp / "corpus"
        immagini = [_png(self.tmp, "p01.png", b"a")]
        output = self.tmp / "piano.rivisto.lotto01.json"
        session = ai_review.ProviderSession("glm", "glm-5.3-flash", "glm")

        def finta_cli(cmd, workdir, timeout, stdin_text=None, ai=None):
            output.write_text('{"pagine": []}', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        cassette.attiva(corpus, registra=True)
        with (
            patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=finta_cli) as esecutore,
            patch("comics_translator.traduzione.ai_review._report_outcome"),
        ):
            ai_review.run_agent(session, "prompt", self.tmp, immagini, 30, output_path=output)

        esecutore.assert_called_once()
        files = list((corpus / "cassette").glob("*.json"))
        self.assertEqual(len(files), 1)
        voce = json.loads(files[0].read_text("utf-8"))
        self.assertEqual(voce["ai"], "glm")
        self.assertEqual(voce["output"], '{"pagine": []}')
        self.assertEqual(voce["returncode"], 0)
        riepilogo = cassette.chiudi(piano_sha256="p", unito_sha256="u")
        manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
        self.assertEqual(len(manifest["cassette"]), 1)
        self.assertEqual(riepilogo["cassette_scritte"], 1)

    def test_timeout_registrato_e_riprodotto(self):
        corpus = self.tmp / "corpus"
        immagini = [_png(self.tmp, "p01.png", b"a")]
        session = ai_review.ProviderSession("glm", "glm-5.3-flash", "glm")

        output = self.tmp / "piano.rivisto.lotto01.json"
        cassette.attiva(corpus, registra=True)
        with (
            patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=subprocess.TimeoutExpired(["glm"], 30)),
            patch("comics_translator.traduzione.ai_review._report_outcome"),
        ):
            esito = ai_review.run_agent(session, "prompt", self.tmp, immagini, 30, output_path=output)
        self.assertIsNone(esito)
        self.assertFalse(output.exists())
        cassette.chiudi()

        cassette.attiva(corpus, registra=False)
        with patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=AssertionError("CLI vietata")):
            esito = ai_review.run_agent(session, "prompt", self.tmp, immagini, 30, output_path=output)
        self.assertIsNone(esito)

    def test_attesa_credito_in_registrazione_scrive_una_sola_cassette(self):
        corpus = self.tmp / "corpus"
        immagini = [_png(self.tmp, "p01.png", b"a")]
        output = self.tmp / "piano.rivisto.lotto01.json"
        session = ai_review.ProviderSession("glm", "glm-5.3-flash", "glm")
        bloccato = SimpleNamespace(returncode=1, stdout="", stderr="usage limit reached")
        completato = SimpleNamespace(returncode=0, stdout="", stderr="")

        def finta_cli(cmd, workdir, timeout, stdin_text=None, ai=None):
            output.write_text('{"pagine": []}', encoding="utf-8")
            return completato

        cassette.attiva(corpus, registra=True)
        with (
            patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=[bloccato, finta_cli(None, None, None)]),
            patch("comics_translator.traduzione.ai_review._aspetta_sblocco", return_value=True),
            patch("comics_translator.traduzione.ai_review._report_outcome"),
        ):
            esito = ai_review.run_agent(session, "prompt", self.tmp, immagini, 30, output_path=output)

        self.assertIs(esito, completato)
        files = list((corpus / "cassette").glob("*.json"))
        self.assertEqual(len(files), 1, "il tentativo bloccato non deve lasciare cassette")


class ReplayTest(BaseCassetteTest):
    def _registra(self, corpus: Path, output: Path, immagini, returncode=0, output_text='{"pagine": []}'):
        session = ai_review.ProviderSession("glm", "glm-5.3-flash", "glm")

        def finta_cli(cmd, workdir, timeout, stdin_text=None, ai=None):
            if output_text is not None:
                output.write_text(output_text, encoding="utf-8")
            return SimpleNamespace(returncode=returncode, stdout="riga", stderr="")

        cassette.attiva(corpus, registra=True)
        with (
            patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=finta_cli),
            patch("comics_translator.traduzione.ai_review._report_outcome"),
        ):
            ai_review.run_agent(session, "prompt", self.tmp, immagini, 30, output_path=output)
        cassette.chiudi()

    def test_replay_mancante_fallisce_chiuso_senza_chiamate(self):
        corpus = self.tmp / "corpus"
        (corpus / "cassette").mkdir(parents=True)
        (corpus / "manifest.json").write_text("{}", encoding="utf-8")
        immagini = [_png(self.tmp, "p01.png", b"a")]
        output = self.tmp / "piano.rivisto.lotto01.json"
        session = ai_review.ProviderSession("glm", "glm-5.3-flash", "glm")

        cassette.attiva(corpus, registra=False)
        with patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=AssertionError("CLI vietata")):
            with self.assertRaises(SystemExit):
                ai_review.run_agent(session, "prompt", self.tmp, immagini, 30, output_path=output)
        self.assertFalse(output.exists(), "nessun output senza cassette")

    def test_replay_non_tocca_cli_ne_abbonamento(self):
        corpus = self.tmp / "corpus"
        immagini = [_png(self.tmp, "p01.png", b"a")]
        output = self.tmp / "piano.rivisto.lotto01.json"
        self._registra(corpus, output, immagini)

        cassette.attiva(corpus, registra=False)
        with (
            patch("comics_translator.traduzione.ai_review.find_cli", side_effect=AssertionError("CLI non richiesta")),
            patch("comics_translator.traduzione.ai_review._verifica_login_glm", side_effect=AssertionError("auth non richiesta")),
            patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=AssertionError("CLI vietata")),
        ):
            session = ai_review.prepare_provider("glm", "glm-5.3-flash")
            self.assertEqual(session.cli, "cassette")
            esito = ai_review.run_agent(session, "prompt", self.tmp, immagini, 30, output_path=output)
        self.assertEqual(esito.returncode, 0)
        self.assertEqual(output.read_text("utf-8"), '{"pagine": []}')

    def test_replay_riproduce_fallimento_cli_senza_output(self):
        corpus = self.tmp / "corpus"
        immagini = [_png(self.tmp, "p01.png", b"a")]
        output = self.tmp / "piano.rivisto.lotto01.json"
        self._registra(corpus, output, immagini, returncode=1, output_text=None)

        cassette.attiva(corpus, registra=False)
        output.write_text("stale", encoding="utf-8")
        with patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=AssertionError("CLI vietata")):
            esito = ai_review.run_agent(
                ai_review.ProviderSession("glm", "glm-5.3-flash", "glm"),
                "prompt", self.tmp, immagini, 30, output_path=output,
            )
        self.assertEqual(esito.returncode, 1)
        self.assertEqual(output.read_text("utf-8"), "stale", "niente output per un fallimento")

    def test_replay_bytes_identici_alla_registrazione(self):
        corpus = self.tmp / "corpus"
        immagini = [_png(self.tmp, "p01.png", b"a")]
        output = self.tmp / "piano.rivisto.lotto01.json"
        testo = '{"pagine": [{"testo": "Caf\\u00e9"}]}'
        self._registra(corpus, output, immagini, output_text=testo)
        registrato = output.read_bytes()

        output.write_text("vecchio", encoding="utf-8")
        cassette.attiva(corpus, registra=False)
        with patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=AssertionError("CLI vietata")):
            ai_review.run_agent(
                ai_review.ProviderSession("glm", "glm-5.3-flash", "glm"),
                "prompt", self.tmp, immagini, 30, output_path=output,
            )
        self.assertEqual(output.read_bytes(), registrato)

    def test_cassette_corrotta_fallisce_chiuso(self):
        corpus = self.tmp / "corpus"
        immagini = [_png(self.tmp, "p01.png", b"a")]
        output = self.tmp / "piano.rivisto.lotto01.json"
        self._registra(corpus, output, immagini)
        voce = json.loads(next((corpus / "cassette").glob("*.json")).read_text("utf-8"))
        voce["output"] = voce["output"].replace("pagine", "pagineX")
        next((corpus / "cassette").glob("*.json")).write_text(
            json.dumps(voce, ensure_ascii=False), encoding="utf-8"
        )

        cassette.attiva(corpus, registra=False)
        with patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=AssertionError("CLI vietata")):
            with self.assertRaises(SystemExit):
                ai_review.run_agent(
                    ai_review.ProviderSession("glm", "glm-5.3-flash", "glm"),
                    "prompt", self.tmp, immagini, 30, output_path=output,
                )


if __name__ == "__main__":
    unittest.main()
