import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator import pipeline_status as status
from comics_translator.traduzione import ai_review


class ModelDefaultsTest(unittest.TestCase):
    def test_codex_is_default_with_gpt_5_4(self):
        self.assertEqual(ai_review.AI_DEFAULT, "codex")
        self.assertEqual(ai_review.default_model("codex"), "gpt-5.4")

    def test_claude_keeps_its_own_default(self):
        self.assertEqual(ai_review.default_model("claude"), "haiku")

    def test_glm_defaults_to_flash_model(self):
        self.assertEqual(ai_review.default_model("glm"), "glm-5.3-flash")


class CommandBuilderTest(unittest.TestCase):
    def test_codex_exec_is_headless_sandboxed_and_receives_images(self):
        workdir = Path("C:/work/revisione")
        images = [workdir / "pagine/p01.png", workdir / "pagine/p02.png"]

        cmd = ai_review._build_command(
            "codex", "codex", "traduci", "gpt-5.5", workdir, images
        )

        self.assertEqual(cmd[:2], ["codex", "exec"])
        self.assertIn("--json", cmd)
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--approve-for-me", cmd)
        self.assertNotIn("--sandbox", cmd)
        self.assertNotIn("--ask-for-approval", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "gpt-5.5")
        self.assertEqual(cmd.count("--image"), 2)
        self.assertNotIn("traduci", cmd)
        self.assertLess(cmd.index("-"), cmd.index("--image"))
        self.assertEqual(cmd[-1], str(images[-1]))

    def test_claude_command_keeps_existing_headless_contract(self):
        cmd = ai_review._build_command(
            "claude", "claude", "traduci", "haiku", Path("revisione"), []
        )

        self.assertEqual(cmd[:3], ["claude", "-p", "traduci"])
        self.assertEqual(cmd[cmd.index("--model") + 1], "haiku")
        self.assertIn("stream-json", cmd)
        self.assertIn("acceptEdits", cmd)

    def test_glm_command_keeps_flags_but_prompt_travels_on_stdin(self):
        cmd = ai_review._build_command(
            "glm", "glm", "traduci", "glm-5.3-flash", Path("revisione"), []
        )

        # Lo shim .cmd di glm passa da cmd.exe, che spezza argv sulle newlines:
        # il prompt non deve mai viaggiare sulla riga di comando.
        self.assertEqual(cmd[:2], ["glm", "-p"])
        self.assertNotIn("traduci", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "glm-5.3-flash")
        self.assertIn("stream-json", cmd)
        self.assertIn("acceptEdits", cmd)

    def test_workdir_relativa_arriva_assoluta_alla_cli(self):
        """Il processo parte gia' dentro la workdir e la risolverebbe due volte.

        Con un percorso relativo la CLI cerca `lavoro/x` dentro `lavoro/x` e
        fallisce con "os error 3": tutti i lotti persi, esito partial e zero
        campi rivisti, senza che nulla indichi il percorso come causa.
        """
        atteso = str(Path("lavoro/sessione/revisione").resolve())
        for ai, opzione in (("codex", "--cd"), ("claude", "--add-dir"), ("glm", "--add-dir")):
            with self.subTest(ai=ai):
                cmd = ai_review._build_command(
                    ai, ai, "traduci", "m", Path("lavoro/sessione/revisione"), []
                )
                passato = cmd[cmd.index(opzione) + 1]
                self.assertEqual(passato, atteso)
                self.assertTrue(Path(passato).is_absolute())


class CodexSubscriptionTest(unittest.TestCase):
    @patch("comics_translator.traduzione.ai_review.subprocess.run")
    def test_chatgpt_login_is_accepted(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["codex", "login", "status"], 0, "Logged in using ChatGPT", ""
        )

        ai_review._verifica_abbonamento_codex("codex")

    @patch("comics_translator.traduzione.ai_review.subprocess.run")
    def test_api_key_login_is_rejected(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["codex", "login", "status"], 0, "Logged in using an API key", ""
        )

        with self.assertRaises(SystemExit) as raised:
            ai_review._verifica_abbonamento_codex("codex")

        self.assertIn("abbonamento", str(raised.exception))

    @patch("comics_translator.traduzione.ai_review.subprocess.run")
    def test_missing_login_is_rejected(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["codex", "login", "status"], 1, "", "Not logged in"
        )

        with self.assertRaises(SystemExit):
            ai_review._verifica_abbonamento_codex("codex")


class GlmSubscriptionTest(unittest.TestCase):
    @patch("comics_translator.traduzione.ai_review.subprocess.run")
    def test_oauth_login_is_accepted(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["glm", "auth", "status"], 0,
            '{"loggedIn": true, "authMethod": "oauth_token"}', "",
        )

        ai_review._verifica_login_glm("glm")

    @patch("comics_translator.traduzione.ai_review.subprocess.run")
    def test_missing_login_is_rejected(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["glm", "auth", "status"], 1, "", "Not logged in"
        )

        with self.assertRaises(SystemExit):
            ai_review._verifica_login_glm("glm")

    @patch("comics_translator.traduzione.ai_review.subprocess.run")
    def test_logged_out_status_is_rejected(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["glm", "auth", "status"], 0, '{"loggedIn": false}', "",
        )

        with self.assertRaises(SystemExit):
            ai_review._verifica_login_glm("glm")

    @patch("comics_translator.traduzione.ai_review.subprocess.run")
    def test_unreadable_status_is_rejected(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["glm", "auth", "status"], 0, "non e' json", "",
        )

        with self.assertRaises(SystemExit):
            ai_review._verifica_login_glm("glm")


class OutcomeReportingTest(unittest.TestCase):
    def test_codex_error_reports_the_useful_line_not_the_help_footer(self):
        proc = SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="error: opzioni incompatibili\n\nUsage: codex exec\nFor more information, try '--help'.",
        )
        output = StringIO()

        with redirect_stderr(output):
            ai_review._report_outcome(proc, "codex")

        self.assertIn("error: opzioni incompatibili", output.getvalue())
        self.assertNotIn("For more information", output.getvalue())

    def test_successful_model_text_cannot_fake_a_usage_limit(self):
        proc = SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=(
                '{"type":"assistant","message":{"content":'
                '[{"type":"text","text":"usage limit reached nel documento"}]}}\n'
                '{"type":"result","is_error":false,"result":"completato"}'
            ),
        )

        self.assertEqual(ai_review._limite_esaurito(proc), "")

    def test_structured_usage_error_is_detected(self):
        proc = SimpleNamespace(
            returncode=0,
            stderr="",
            stdout='{"type":"result","is_error":true,"result":"usage limit reached"}',
        )

        self.assertIn("usage limit reached", ai_review._limite_esaurito(proc))


class ProviderRunnerTest(unittest.TestCase):
    def test_usage_limit_retries_the_same_turn_after_confirmation(self):
        session = ai_review.ProviderSession("codex", "gpt-5.5", "codex")
        blocked = SimpleNamespace(
            returncode=1, stdout="", stderr="usage limit reached; resets at 14:00"
        )
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch("comics_translator.traduzione.ai_review._build_command", return_value=["codex", "exec"]),
            patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=[blocked, completed]) as execute,
            patch("comics_translator.traduzione.ai_review._aspetta_sblocco", return_value=True) as wait,
            patch("comics_translator.traduzione.ai_review._report_outcome") as report,
        ):
            result = ai_review.run_agent(
                session, "prompt", Path("revisione"), [Path("p01.png")], 30,
            )

        self.assertIs(result, completed)
        self.assertEqual(execute.call_count, 2)
        wait.assert_called_once()
        report.assert_called_once_with(completed, "codex")

    def test_timeout_returns_none(self):
        session = ai_review.ProviderSession("codex", "gpt-5.5", "codex")
        timeout = subprocess.TimeoutExpired(["codex", "exec"], 30)

        with (
            patch("comics_translator.traduzione.ai_review._build_command", return_value=["codex", "exec"]),
            patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=timeout),
            patch("comics_translator.traduzione.ai_review._report_outcome") as report,
        ):
            result = ai_review.run_agent(
                session, "prompt", Path("revisione"), [], 30,
            )

        self.assertIsNone(result)
        report.assert_not_called()

    def test_glm_prompt_travels_on_stdin_like_codex(self):
        session = ai_review.ProviderSession("glm", "glm-5.3-flash", "glm")
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch("comics_translator.traduzione.ai_review._build_command", return_value=["glm", "-p"]) as build,
            patch("comics_translator.traduzione.ai_review._esegui_cli", return_value=completed) as execute,
            patch("comics_translator.traduzione.ai_review._report_outcome"),
        ):
            ai_review.run_agent(
                session, "prompt lungo", Path("revisione"), [], 30,
            )

        self.assertEqual(
            execute.call_args.kwargs.get("stdin_text"), "prompt lungo",
        )
        self.assertEqual(
            build.call_args.args[2], "prompt lungo",
        )

    def test_silent_cli_reports_stderr_instead_of_success(self):
        proc = SimpleNamespace(returncode=0, stdout="", stderr="argomenti rotti\nseconda riga")
        output = StringIO()

        with redirect_stderr(output):
            ai_review._report_outcome(proc, "glm")

        self.assertIn("non ha prodotto eventi", output.getvalue())
        self.assertIn("argomenti rotti", output.getvalue())
        self.assertNotIn("revisione completata", output.getvalue())


class EnvSenzaHostClaudeTest(unittest.TestCase):
    """Il CLI figlio non deve ereditare i marcatori di sessione Claude Code.

    Dentro una sessione Claude Code l'host inietta CLAUDECODE / CLAUDE_CODE_*:
    glm (che e' Claude Code) li eredita, si crede figlio della sessione e usa
    l'OAuth dell'host contro Z.ai -> 401 anche con chiave valida. Diagnosticato
    il 4 settembre sull'albo Imperial.
    """

    def test_toglie_le_variabili_claude_dell_host(self):
        finto = {"CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "claude-desktop",
                 "CLAUDE_CODE_SDK_HAS_OAUTH_REFRESH": "1", "CLAUDE_AGENT_SDK_VERSION": "0.3",
                 "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                 "ZAI_API_KEY": "k", "PATH": "/bin"}
        with patch.dict(ai_review.os.environ, finto, clear=True):
            env = ai_review._env_senza_host_claude()
        self.assertNotIn("CLAUDECODE", env)
        self.assertFalse(any(k.startswith("CLAUDE") for k in env))
        # cio' che serve ai provider resta
        self.assertEqual(env["ZAI_API_KEY"], "k")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://api.z.ai/api/anthropic")
        self.assertEqual(env["PATH"], "/bin")


class ClaudeSoloSuAnthropicTest(unittest.TestCase):
    """`--ai claude --review-model haiku` deve essere haiku, con l'account della CLI.

    Il 10 settembre, su doom-academy-1, le ANTHROPIC_* della sessione di lancio
    (base URL z.ai, token, alias haiku -> glm-5.3-flash) hanno fatto girare 14
    lotti su 15 come glm: run buttata e rifatta.
    """

    SESSIONE_ZAI = {
        "CLAUDECODE": "1", "CLAUDE_CODE_OAUTH_TOKEN": "host",
        "CLAUDE_CONFIG_DIR": "C:/host/.claude",
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "t", "ANTHROPIC_API_KEY": "k",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.3-flash",
        "ZAI_API_KEY": "z", "PATH": "/bin",
    }

    def test_claude_non_eredita_le_anthropic_della_sessione(self):
        with patch.dict(ai_review.os.environ, self.SESSIONE_ZAI, clear=True):
            env = ai_review._env_per_provider("claude")
        self.assertFalse(any(k.startswith(("ANTHROPIC", "CLAUDE")) for k in env))
        self.assertEqual(env["PATH"], "/bin")

    def test_glm_tiene_le_anthropic(self):
        with patch.dict(ai_review.os.environ, self.SESSIONE_ZAI, clear=True):
            env = ai_review._env_per_provider("glm")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://api.z.ai/api/anthropic")
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)

    def test_cartella_dedicata_diventa_claude_config_dir(self):
        finto = dict(self.SESSIONE_ZAI, COMICS_CLAUDE_CONFIG_DIR="C:/pipeline/.claude")
        with patch.dict(ai_review.os.environ, finto, clear=True):
            env = ai_review._env_per_provider("claude")
        # quella dell'host sparisce, resta la cartella della pipeline
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "C:/pipeline/.claude")

    def test_init_con_modello_non_claude_e_estraneo(self):
        init = {"type": "system", "subtype": "init", "model": "glm-5.3-flash"}
        self.assertEqual(ai_review._modello_estraneo("claude", init), "glm-5.3-flash")
        buono = dict(init, model="claude-haiku-4-5-20251001")
        self.assertIsNone(ai_review._modello_estraneo("claude", buono))
        # glm annuncia glm, ed e' giusto cosi'
        self.assertIsNone(ai_review._modello_estraneo("glm", init))
        self.assertIsNone(ai_review._modello_estraneo("claude", {"type": "assistant"}))

    def _stato(self, **campi):
        stato = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
                 "email": "chi@esempio.it", "subscriptionType": "pro"}
        stato.update(campi)
        return SimpleNamespace(returncode=0, stdout=json.dumps(stato), stderr="")

    def test_login_abbonamento_accettato_e_account_mostrato(self):
        uscita = StringIO()
        with (patch.dict(ai_review.os.environ, self.SESSIONE_ZAI, clear=True),
              patch.object(ai_review.subprocess, "run", return_value=self._stato()) as run,
              redirect_stdout(uscita)):
            ai_review._verifica_login_claude("claude")
        self.assertIn("chi@esempio.it", uscita.getvalue())
        # lo stato si legge con l'ambiente dei lotti, non con quello dell'host
        self.assertNotIn("ANTHROPIC_BASE_URL", run.call_args.kwargs["env"])

    def test_api_key_o_endpoint_estraneo_rifiutati(self):
        for stato in (self._stato(authMethod="api_key"), self._stato(apiProvider="bedrock"),
                      self._stato(loggedIn=False)):
            with (patch.object(ai_review.subprocess, "run", return_value=stato),
                  self.assertRaises(SystemExit)):
                ai_review._verifica_login_claude("claude")


class AttesaCreditoTest(unittest.TestCase):
    """Credito esaurito: si aspetta il via libera dell'utente, su due canali.

    Il 7 settembre, su carnage-3, la scelta del canale in base a
    `sys.stdin.isatty()` ha fermato per ore una run lanciata in background da un
    agente: stdin era ereditato da una shell vera, quindi il programma si e'
    messo ad aspettare un tasto che nessuno poteva premere, su un ramo senza
    tetto di tempo. Ora i canali si ascoltano entrambi e vince il primo che
    arriva, cosi' non c'e' piu' niente da indovinare.
    """

    def setUp(self):
        self.cartella = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.cartella, True)
        report = SimpleNamespace(path=self.cartella / "manifest.json",
                                 event=lambda *a, **k: None)
        token = status.ACTIVE_REPORT.set(report)
        self.addCleanup(status.ACTIVE_REPORT.reset, token)
        # nessun tasto premuto, se il test non ne simula uno
        for patcher in (patch.object(ai_review, "ATTESA_CREDITO_POLL_S", 0.01),
                        patch.object(ai_review, "_tasto_premuto", return_value=b"")):
            self.tasti = patcher.start()
            self.addCleanup(patcher.stop)

    def _aspetta(self):
        with redirect_stderr(StringIO()) as errori:
            return ai_review._aspetta_sblocco("usage limit reached"), errori.getvalue()

    def test_il_file_riprendi_fa_ritentare_e_viene_consumato(self):
        segnale = self.cartella / ai_review.SBLOCCO_RIPRENDI
        segnale.write_text("", encoding="utf-8")

        riprova, errori = self._aspetta()

        self.assertTrue(riprova)
        # consumato: il blocco successivo pretende un segnale nuovo
        self.assertFalse(segnale.exists())
        self.assertIn(str(segnale), errori)

    def test_il_file_vale_anche_quando_stdin_sembra_un_terminale(self):
        """La regressione di carnage-3, ridotta all'osso.

        Una run in background eredita uno stdin che dice di essere un
        terminale. Prima bastava a spegnere il canale su file: il lotto restava
        appeso a un tasto irraggiungibile.
        """
        (self.cartella / ai_review.SBLOCCO_RIPRENDI).write_text("", encoding="utf-8")

        with patch.object(ai_review.sys, "stdin", SimpleNamespace(isatty=lambda: True)):
            riprova, _ = self._aspetta()

        self.assertTrue(riprova)

    def test_il_file_rinuncia_chiude_senza_aspettare_la_scadenza(self):
        (self.cartella / ai_review.SBLOCCO_RINUNCIA).write_text("", encoding="utf-8")

        riprova, _ = self._aspetta()

        self.assertFalse(riprova)
        self.assertFalse((self.cartella / ai_review.SBLOCCO_RINUNCIA).exists())

    def test_un_tasto_qualsiasi_fa_ritentare(self):
        self.tasti.return_value = b"x"

        riprova, _ = self._aspetta()

        self.assertTrue(riprova)

    def test_esc_rinuncia(self):
        self.tasti.return_value = b"\x1b"

        riprova, _ = self._aspetta()

        self.assertFalse(riprova)

    def test_senza_segnale_rinuncia_alla_scadenza_invece_di_restare_appeso(self):
        with patch.object(ai_review, "ATTESA_CREDITO_S", 0.05):
            riprova, errori = self._aspetta()

        self.assertFalse(riprova)
        self.assertIn("Nessun segnale", errori)

    def test_il_tetto_di_tempo_vale_anche_col_terminale(self):
        """Prima il ramo del tasto non scadeva mai: appesi per sempre."""
        with (patch.object(ai_review.sys, "stdin", SimpleNamespace(isatty=lambda: True)),
              patch.object(ai_review, "ATTESA_CREDITO_S", 0.05)):
            riprova, _ = self._aspetta()

        self.assertFalse(riprova)

    def test_senza_cartella_di_run_resta_il_canale_dei_tasti(self):
        # niente run: il file non ha dove stare, ma il tasto funziona lo stesso
        token = status.ACTIVE_REPORT.set(None)
        self.addCleanup(status.ACTIVE_REPORT.reset, token)
        self.tasti.return_value = b"x"

        riprova, _ = self._aspetta()

        self.assertTrue(riprova)


class TastoPremutoTest(unittest.TestCase):
    """La lettura del tasto non deve bloccare ne' esplodere senza console."""

    def test_senza_tasti_in_coda_non_blocca(self):
        with (patch("msvcrt.kbhit", return_value=False, create=True),
              patch("msvcrt.getch", side_effect=AssertionError("non deve leggere"),
                    create=True)):
            self.assertEqual(ai_review._tasto_premuto(), b"")

    def test_un_processo_senza_console_non_fa_cadere_la_run(self):
        with patch("msvcrt.kbhit", side_effect=OSError("niente console"), create=True):
            self.assertEqual(ai_review._tasto_premuto(), b"")


if __name__ == "__main__":
    unittest.main()
