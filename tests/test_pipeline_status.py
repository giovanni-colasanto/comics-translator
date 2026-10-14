"""P0 fault injection: no network/CLI calls, synthetic contract fixture."""

import copy
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.traduzione import ai_review
from comics_translator.disegno import pdf_translate_overlay as pdf
from comics_translator import pipeline_status as status
from comics_translator.disegno import plan
from comics_translator.verifica import visual_qa


FIXTURE = json.loads((Path(__file__).parent / "fixtures/p0_contract.json").read_text("utf-8"))


def proc(code=0, stdout=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr="")


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.plan = copy.deepcopy(FIXTURE["plan"])
        self.response = copy.deepcopy(FIXTURE["response"])

    def test_complete_response_includes_empty_page(self):
        ai_review.validate_review_batch(self.plan["pagine"], self.response)

    def test_missing_or_duplicate_page_is_rejected(self):
        for pages in ([self.response["pagine"][0]], self.response["pagine"] * 2):
            with self.subTest(pages=len(pages)), self.assertRaises(ValueError):
                ai_review.validate_review_batch(self.plan["pagine"], {"pagine": pages})

    def test_missing_unknown_duplicate_block_is_rejected(self):
        for kind in ("missing", "unknown", "duplicate"):
            response = copy.deepcopy(self.response)
            blocks = response["pagine"][0]["blocchi"]
            if kind == "missing":
                blocks.pop()
            elif kind == "unknown":
                blocks[0]["id"] = "invented"
            else:
                blocks[1]["id"] = blocks[0]["id"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ai_review.validate_review_batch(self.plan["pagine"], response)

    def test_invalid_field_types_or_missing_translation_are_rejected(self):
        for fields in ({"salta": "false"}, {"testo": ""}, {"corpo": float("nan")},
                       {"testo": "testo\\naltro"}, {"testo": None}):
            response = copy.deepcopy(self.response)
            response["pagine"][0]["blocchi"][0].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                ai_review.validate_review_batch(self.plan["pagine"], response)

    def test_protected_block_still_requires_nonempty_text(self):
        source = self.plan["pagine"][0]["blocchi"][0]
        source["salta"] = True
        before = copy.deepcopy(self.plan)
        response = self.response["pagine"][0]["blocchi"][0]
        response.update(salta=True, testo="")
        with self.assertRaisesRegex(ValueError, "stringa non vuota"):
            ai_review.validate_review_batch(self.plan["pagine"], self.response)
        self.assertEqual(self.plan, before)

        response["testo"] = source["originale"]
        ai_review.validate_review_batch(self.plan["pagine"], self.response)
        self.assertEqual(self.plan, before)
        self.assertTrue(response["salta"])

    def test_blank_protected_text_is_repaired_without_mutating_inputs(self):
        self.plan["pagine"][0]["blocchi"][0]["salta"] = True
        for blank in ("", " \n\t"):
            with self.subTest(blank=blank):
                self.response["pagine"][0]["blocchi"][0].update(salta=True, testo=blank)
                before_plan = copy.deepcopy(self.plan)
                before_response = copy.deepcopy(self.response)
                repaired, repairs = ai_review.validate_and_repair_review_batch(
                    self.plan["pagine"], self.response)
                block = repaired["pagine"][0]["blocchi"][0]
                self.assertEqual(block["testo"], self.plan["pagine"][0]["blocchi"][0]["originale"])
                self.assertIs(block["salta"], True)
                self.assertEqual(repairs[0]["ids"], [block["id"]])
                self.assertEqual(len(repairs), 1)
                self.assertEqual(self.plan, before_plan)
                self.assertEqual(self.response, before_response)
                ai_review.validate_review_batch(self.plan["pagine"], repaired)

    def test_protected_repair_does_not_relax_other_contract_errors(self):
        # "merge" non e' piu' in lista: una fusione impossibile viene rifiutata
        # e registrata come riparazione, non fa piu' cadere il lotto intero.
        # "reenabled" non e' piu' in lista: la riattivazione di un'area
        # protetta viene richiusa sul singolo blocco e registrata, non fa piu'
        # cadere il lotto intero.
        for kind in ("unprotected", "null", "missing_text", "missing_block",
                     "unknown_block", "duplicate_block", "unknown_field", "empty_original"):
            with self.subTest(kind=kind):
                source = copy.deepcopy(self.plan["pagine"])
                response = copy.deepcopy(self.response)
                original = source[0]["blocchi"][0]
                revised = response["pagine"][0]["blocchi"][0]
                original["salta"] = True
                revised.update(salta=True, testo="")
                if kind == "unprotected":
                    original["salta"] = False
                elif kind == "null":
                    revised["testo"] = None
                elif kind == "missing_text":
                    revised.pop("testo")
                elif kind == "missing_block":
                    response["pagine"][0]["blocchi"].pop()
                elif kind == "unknown_block":
                    revised["id"] = "unknown"
                elif kind == "duplicate_block":
                    response["pagine"][0]["blocchi"].append(copy.deepcopy(revised))
                elif kind == "unknown_field":
                    revised["extra"] = True
                elif kind == "empty_original":
                    original["originale"] = ""
                before = copy.deepcopy(response)
                with self.assertRaises(ValueError):
                    ai_review.validate_and_repair_review_batch(source, response)
                self.assertEqual(response, before)

    def test_fusione_impossibile_non_butta_via_il_lotto(self):
        source = copy.deepcopy(self.plan["pagine"])
        response = copy.deepcopy(self.response)
        source[0]["blocchi"][0]["salta"] = True
        revised = response["pagine"][0]["blocchi"][0]
        revised.update(salta=True, unisci_con=response["pagine"][0]["blocchi"][1]["id"])
        altro = response["pagine"][0]["blocchi"][1]["testo"]
        before = copy.deepcopy(response)

        candidate, repairs = ai_review.validate_and_repair_review_batch(source, response)

        self.assertEqual(response, before)  # la risposta grezza resta intatta
        self.assertEqual([r["code"] for r in repairs], ["invalid_merge_dropped"])
        blocchi = candidate["pagine"][0]["blocchi"]
        self.assertIsNone(blocchi[0]["unisci_con"])
        self.assertTrue(blocchi[0]["salta"])  # resta protetto
        self.assertEqual(blocchi[1]["testo"], altro)  # la traduzione buona resta

    def test_riattivazione_protetta_non_butta_via_il_lotto(self):
        source = copy.deepcopy(self.plan["pagine"])
        response = copy.deepcopy(self.response)
        source[0]["blocchi"][0]["salta"] = True
        response["pagine"][0]["blocchi"][0]["salta"] = False
        mia = response["pagine"][0]["blocchi"][0]["testo"]
        altro = response["pagine"][0]["blocchi"][1]["testo"]
        before = copy.deepcopy(response)

        candidate, repairs = ai_review.validate_and_repair_review_batch(source, response)

        self.assertEqual(response, before)  # la risposta grezza resta intatta
        self.assertEqual([r["code"] for r in repairs], ["protected_reactivation_dropped"])
        self.assertEqual(repairs[0]["ids"], ["p1-a"])
        blocchi = candidate["pagine"][0]["blocchi"]
        self.assertTrue(blocchi[0]["salta"])  # torna protetto com'era
        self.assertEqual(blocchi[0]["testo"], mia)  # la sua traduzione resta scritta
        self.assertEqual(blocchi[1]["testo"], altro)  # e quella del vicino pure

    def test_riattivazione_con_testo_vuoto_ricade_sull_originale(self):
        source = copy.deepcopy(self.plan["pagine"])
        response = copy.deepcopy(self.response)
        source[0]["blocchi"][0]["salta"] = True
        response["pagine"][0]["blocchi"][0].update(salta=False, testo="")

        candidate, repairs = ai_review.validate_and_repair_review_batch(source, response)

        self.assertEqual([r["code"] for r in repairs],
                         ["protected_reactivation_dropped", "protected_empty_text_restored"])
        blocco = candidate["pagine"][0]["blocchi"][0]
        self.assertTrue(blocco["salta"])
        self.assertEqual(blocco["testo"], source[0]["blocchi"][0]["originale"])

    def test_fusione_che_tocca_l_area_richiusa_cade_da_sola(self):
        for chi in ("dalla", "verso"):
            with self.subTest(chi=chi):
                source = copy.deepcopy(self.plan["pagine"])
                response = copy.deepcopy(self.response)
                source[0]["blocchi"][0]["salta"] = True
                blocchi = response["pagine"][0]["blocchi"]
                blocchi[0]["salta"] = False
                if chi == "dalla":
                    blocchi[0]["unisci_con"] = "p1-b"
                else:
                    blocchi[1]["unisci_con"] = "p1-a"
                altro = blocchi[1]["testo"]

                candidate, repairs = ai_review.validate_and_repair_review_batch(
                    source, response)

                self.assertEqual(
                    [r["code"] for r in repairs],
                    ["protected_reactivation_dropped", "merge_on_reprotected_dropped"])
                usciti = candidate["pagine"][0]["blocchi"]
                self.assertTrue(usciti[0]["salta"])
                self.assertIsNone(usciti[0].get("unisci_con"))
                self.assertIsNone(usciti[1].get("unisci_con"))
                self.assertEqual(usciti[1]["testo"], altro)

    def test_riattivazione_non_salva_un_lotto_con_altri_errori(self):
        """La riparazione vale per quel blocco, non assolve il resto del lotto."""
        for kind in ("unknown_field", "missing_text", "unknown_block", "escape"):
            with self.subTest(kind=kind):
                source = copy.deepcopy(self.plan["pagine"])
                response = copy.deepcopy(self.response)
                source[0]["blocchi"][0]["salta"] = True
                response["pagine"][0]["blocchi"][0]["salta"] = False
                vicino = response["pagine"][0]["blocchi"][1]
                if kind == "unknown_field":
                    vicino["extra"] = True
                elif kind == "missing_text":
                    vicino.pop("testo")
                elif kind == "unknown_block":
                    vicino["id"] = "inventato"
                else:
                    vicino["testo"] = "riga\\naltra"
                with self.assertRaises(ValueError):
                    ai_review.validate_and_repair_review_batch(source, response)

    def test_fusione_valida_resta_al_suo_posto(self):
        source = copy.deepcopy(self.plan["pagine"])
        response = copy.deepcopy(self.response)
        target = response["pagine"][0]["blocchi"][1]["id"]
        response["pagine"][0]["blocchi"][0]["unisci_con"] = target
        with (patch("comics_translator.disegno.plan.validate_merges", return_value=[]),
              patch("comics_translator.verifica.qa_corrections.validate_merges", return_value=[])):
            candidate, repairs = ai_review.validate_and_repair_review_batch(source, response)
        self.assertEqual(repairs, [])
        self.assertEqual(candidate["pagine"][0]["blocchi"][0]["unisci_con"], target)

    def test_merge_repair_does_not_accept_invalid_target_types(self):
        for target in (["p1-b"], {"id": "p1-b"}, 42, True):
            with self.subTest(target=target):
                response = copy.deepcopy(self.response)
                response["pagine"][0]["blocchi"][0]["unisci_con"] = target
                before = copy.deepcopy(response)
                with self.assertRaisesRegex(ValueError, "unisci_con"):
                    ai_review.validate_and_repair_review_batch(self.plan["pagine"], response)
                self.assertEqual(response, before)

    def test_protected_block_cannot_be_reenabled(self):
        self.plan["pagine"][0]["blocchi"][0]["salta"] = True
        with self.assertRaisesRegex(ValueError, "protetta"):
            ai_review.validate_review_batch(self.plan["pagine"], self.response)

    def test_input_duplicate_id_is_not_silently_collapsed(self):
        self.plan["pagine"][0]["blocchi"][1]["id"] = "p1-a"
        with self.assertRaisesRegex(ValueError, "duplicato"):
            status.plan_index(self.plan)

    def test_empty_page_is_kept_in_build_plan(self):
        doc = [SimpleNamespace(rect=SimpleNamespace(width=600, height=800))]
        built = plan.build_plan(doc, [0], {}, {0: []}, {})
        self.assertEqual(len(built["pagine"]), 1)
        self.assertEqual(built["pagine"][0]["blocchi"], [])
        self.assertEqual(built["pagine"][0]["modo"], "immagine")

    def test_cli_error_with_zero_exit_is_not_success(self):
        self.assertTrue(ai_review.agent_failure(proc(stdout='{"type":"turn.failed"}')))
        recovered = '{"type":"error"}\n{"type":"turn.completed","usage":{}}'
        self.assertFalse(ai_review.agent_failure(proc(stdout=recovered)))

    def test_qa_empty_object_is_not_an_explicit_clean_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                visual_qa._load_spec(path)


class TranslationBatchTest(unittest.TestCase):
    def run_review(self, results, responses, stale=False, source=None, inspect=None):
        source = copy.deepcopy(FIXTURE["plan"] if source is None else source)
        session = ai_review.ProviderSession("codex", "gpt-test", "codex")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if stale:
                (root / "piano.rivisto.lotto01.json").write_text(
                    json.dumps(FIXTURE["response"]), encoding="utf-8")
                (root / "piano.rivisto.lotto01.riparato.json").write_text(
                    json.dumps(FIXTURE["response"]), encoding="utf-8")
            calls = iter(zip(results, responses))
            def fake_agent(*args, output_path=None):
                result, response = next(calls)
                if response is not None:
                    output_path.write_text(response, encoding="utf-8")
                return result
            with (patch("comics_translator.traduzione.ai_review.prepare_provider", return_value=session),
                  patch("comics_translator.traduzione.ai_review.export_pages"),
                  patch("comics_translator.traduzione.ai_review.run_agent", side_effect=fake_agent)):
                # ritenta=0: qui si fissa il verdetto su una singola risposta;
                # il nuovo tentativo ha i suoi test in test_recupero_lotti.
                result = ai_review.review(source, object(), [0, 1], root, "en", "it",
                                          batch=1, ritenta=0)
            if inspect:
                inspect(root, result)
            return result

    def test_repaired_batch_keeps_raw_output_and_records_recovery(self):
        source = copy.deepcopy(FIXTURE["plan"])
        source["pagine"][0]["blocchi"][0]["salta"] = True
        response = copy.deepcopy(FIXTURE["response"])
        response["pagine"][0]["blocchi"][0].update(salta=True, testo="")
        responses = [json.dumps({"pagine": [p]}) for p in response["pagine"]]
        def inspect(root, result):
            raw = root / "piano.rivisto.lotto01.json"
            repaired = root / "piano.rivisto.lotto01.riparato.json"
            self.assertEqual(raw.read_text("utf-8"), responses[0])
            saved = json.loads(repaired.read_text("utf-8"))
            self.assertEqual(saved["pagine"], result.plan["pagine"][:1])
            self.assertFalse((root / "piano.rivisto.lotto02.riparato.json").exists())
            self.assertEqual(saved["pagine"][0]["blocchi"][1], response["pagine"][0]["blocchi"][1])
        with patch("comics_translator.traduzione.ai_review.status.event") as event:
            result = self.run_review([proc(), proc()], responses, source=source, inspect=inspect)
        self.assertEqual(result.outcome.status, "complete")
        self.assertEqual(result.outcome.completed, 2)
        self.assertEqual(len(result.outcome.repairs), 1)
        event.assert_called_once()
        self.assertEqual(event.call_args.args[0], "translation_batch_repaired")
        self.assertEqual(event.call_args.kwargs["repairs"], result.outcome.repairs)

    def test_merge_repair_log_does_not_claim_empty_text_was_restored(self):
        response = copy.deepcopy(FIXTURE["response"])
        response["pagine"][0]["blocchi"][0]["unisci_con"] = "missing"
        responses = [json.dumps({"pagine": [p]}) for p in response["pagine"]]
        output = StringIO()
        with redirect_stdout(output):
            result = self.run_review([proc(), proc()], responses)
        self.assertEqual(result.outcome.completed, 2)
        self.assertIn("invalid_merge_dropped=1", output.getvalue())
        self.assertNotIn("vuoti ripristinati", output.getvalue())

    def test_invalid_merge_type_rejects_only_its_block_without_crashing(self):
        """Prima cadeva l'intero lotto; ora solo il blocco col tipo sbagliato.

        Il lotto resta non completo (esito parziale, blocco perso dichiarato),
        ma il vicino valido della stessa pagina non torna in inglese.
        """
        response = copy.deepcopy(FIXTURE["response"])
        response["pagine"][0]["blocchi"][0]["unisci_con"] = ["p1-b"]
        responses = [json.dumps({"pagine": [p]}) for p in response["pagine"]]
        result = self.run_review([proc(), proc()], responses)
        self.assertEqual(result.outcome.status, "partial")
        self.assertEqual(result.outcome.completed, 1)
        recuperati = [b["id"] for p in result.plan["pagine"] for b in p["blocchi"]]
        bad = response["pagine"][0]["blocchi"][0]["id"]
        self.assertNotIn(bad, recuperati)
        self.assertIn(response["pagine"][0]["blocchi"][1]["id"], recuperati)
        salvato = [i for i in result.outcome.issues if i["code"] == "translation_batch_salvaged"]
        self.assertEqual(salvato[0]["ids"], [bad])

    def test_repairable_blank_does_not_save_a_batch_with_another_error(self):
        source = copy.deepcopy(FIXTURE["plan"])
        source["pagine"][0]["blocchi"][0]["salta"] = True
        response = copy.deepcopy(FIXTURE["response"])
        response["pagine"][0]["blocchi"][0].update(salta=True, testo="")
        response["pagine"][0]["blocchi"][1]["testo"] = ""
        responses = [json.dumps({"pagine": [p]}) for p in response["pagine"]]
        def inspect(root, result):
            self.assertFalse((root / "piano.rivisto.lotto01.riparato.json").exists())
            self.assertEqual(result.plan["pagine"], response["pagine"][1:])
        with patch("comics_translator.traduzione.ai_review.status.event") as event:
            result = self.run_review([proc(), proc()], responses, source=source, inspect=inspect)
        self.assertEqual(result.outcome.status, "partial")
        self.assertEqual(result.outcome.completed, 1)
        self.assertEqual(result.outcome.repairs, [])
        event.assert_not_called()

    def test_repair_audit_survives_report_reload_without_approving_pdf(self):
        source = copy.deepcopy(FIXTURE["plan"])
        source["pagine"][0]["blocchi"][0]["salta"] = True
        response = copy.deepcopy(FIXTURE["response"])
        response["pagine"][0]["blocchi"][0].update(salta=True, testo="")
        responses = [json.dumps({"pagine": [p]}) for p in response["pagine"]]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("comics_translator.pipeline_status.code_provenance", return_value={}):
                report = status.RunReport.create(tmp, {"model": "test"}, tmp)
            token = status.ACTIVE_REPORT.set(report)
            try:
                def inspect(root, result):
                    loaded = status.RunReport.load(report.path)
                    phase = loaded.data["phases"]["translation"]
                    self.assertEqual(phase["status"], "complete")
                    self.assertEqual(phase["repairs"], result.outcome.repairs)
                    for name, filename in (
                        ("translation_raw_lotto01", "piano.rivisto.lotto01.json"),
                        ("translation_repaired_lotto01", "piano.rivisto.lotto01.riparato.json"),
                    ):
                        self.assertEqual(loaded.data["artifacts"][name]["sha256"],
                                         status.sha256(root / filename))
                    self.assertEqual(loaded.data["quality"], "draft_unverified")
                    self.assertFalse(loaded.data["coverage"]["verified"])
                self.run_review([proc(), proc()], responses, source=source, inspect=inspect)
            finally:
                status.ACTIVE_REPORT.reset(token)

    def test_one_missing_batch_is_partial_not_complete(self):
        first = json.dumps({"pagine": FIXTURE["response"]["pagine"][:1]})
        result = self.run_review([proc(), None], [first, None])
        self.assertEqual(result.outcome.status, "partial")
        self.assertEqual((result.outcome.completed, result.outcome.failed), (1, 1))
        self.assertEqual(result.outcome.issues[0]["pages"], [2])

    def test_invalid_or_stale_json_is_not_accepted(self):
        for response in (None, "{", "{}"):
            with self.subTest(response=response):
                def inspect(root, result):
                    self.assertFalse((root / "piano.rivisto.lotto01.riparato.json").exists())
                result = self.run_review([proc(), proc()], [response, response],
                                         stale=True, inspect=inspect)
                self.assertEqual(result.outcome.status, "failed")

    def test_nonzero_exit_rejects_valid_json(self):
        responses = [json.dumps({"pagine": [p]}) for p in FIXTURE["response"]["pagine"]]
        result = self.run_review([proc(1), proc(1)], responses)
        self.assertEqual(result.outcome.completed, 0)

    def test_credit_retry_discards_output_from_failed_attempt(self):
        session = ai_review.ProviderSession("codex", "gpt-test", "codex")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            def execute(*args, **kwargs):
                self.assertFalse(output.exists())
                if execute.calls == 0:
                    output.write_text("{}", encoding="utf-8")
                    execute.calls += 1
                    return SimpleNamespace(returncode=1, stdout="", stderr="usage limit reached")
                return proc()
            execute.calls = 0
            with (patch("comics_translator.traduzione.ai_review._esegui_cli", side_effect=execute),
                  patch("comics_translator.traduzione.ai_review._aspetta_sblocco", return_value=True),
                  patch("comics_translator.traduzione.ai_review._report_outcome")):
                ai_review.run_agent(session, "prompt", Path(tmp), [], 1, output_path=output)
            self.assertFalse(output.exists())


class ReportTest(unittest.TestCase):
    def make_report(self, root):
        with patch("comics_translator.pipeline_status.code_provenance", return_value={}):
            return status.RunReport.create(root, {"model": "test"}, root)

    def test_failed_qa_with_pdf_produces_partial_nonzero_and_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.make_report(tmp)
            report.data["artifacts"]["output_pdf"] = {"path": "test.pdf"}
            report.finish_phase(status.PhaseResult("qa", 2, 1))
            self.assertEqual(report.finish(), 2)
            loaded = status.RunReport.load(report.path)
            self.assertEqual(loaded.data["quality"], "draft_unverified")
            self.assertFalse(loaded.data["coverage"]["verified"])
            self.assertEqual(loaded.data["phases"]["qa"]["failed"], 1)

    def test_waiting_credit_is_persisted_before_input_and_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.make_report(tmp)
            token = status.ACTIVE_REPORT.set(report)
            try:
                def waiting(_):
                    self.assertEqual(status.RunReport.load(report.path).data["status"], "waiting_credit")
                    return False
                with (patch("comics_translator.traduzione.ai_review._esegui_cli", return_value=SimpleNamespace(
                        returncode=1, stdout="", stderr="usage limit reached")),
                      patch("comics_translator.traduzione.ai_review._aspetta_sblocco", side_effect=waiting),
                      patch("comics_translator.traduzione.ai_review._report_outcome")):
                    ai_review.run_agent(ai_review.ProviderSession("codex", "test", "codex"),
                                        "test", Path(tmp), [], 1)
                events = [json.loads(line) for line in report.path.with_name("events.jsonl").read_text().splitlines()]
                self.assertTrue(any(e["event"] == "credit_wait_ended" for e in events))
            finally:
                status.ACTIVE_REPORT.reset(token)

    def test_pending_batches_are_not_marked_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.make_report(tmp)
            report.start_phase("qa", 3)
            report.finish_phase(status.PhaseResult("qa", 3, 1), final=False)
            self.assertEqual(report.data["phases"]["qa"]["failed"], 0)
            self.assertEqual(report.data["phases"]["qa"]["status"], "running")

    def test_provenance_includes_untracked_effective_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("pass\n")
            first = status.code_provenance(root)
            (root / "new.py").write_text("pass\n")
            second = status.code_provenance(root)
            self.assertNotEqual(first["effective_sha256"], second["effective_sha256"])
            self.assertIn("new.py", second["files"])

    def test_standalone_finish_does_not_approve_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.make_report(tmp)
            report.finish_phase(status.PhaseResult("render", 1, 1, unit="pages"))
            self.assertEqual(pdf.finish_translation(report), 0)
            self.assertEqual(report.data["quality"], "draft_unverified")


if __name__ == "__main__":
    unittest.main()
