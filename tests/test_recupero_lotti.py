"""Traduzione: nessuna battuta valida persa per colpa del lotto, tutto chiesto al modello.

Misura sui 14 albi FATTO (settembre 2026): 158 blocchi riscritti a mano erano
`unreviewed` (lotto non validato per intero) e 170 traduzioni erano gia' sul
disco nei `piano.rivisto.lottoNN.json`, mai arrivate alla pagina. Altri 356
blocchi protetti non erano mai stati tradotti perche' il prompt diceva al
modello di lasciarli stare.

Qui si fissano: il recupero per blocco, il nuovo tentativo, la `natura` e il
motivo della protezione mostrato al modello.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator import manifesto_lotti
from comics_translator.traduzione import ai_review

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/p0_contract.json").read_text("utf-8"))


def piano():
    return copy.deepcopy(FIXTURE["plan"])


def ok():
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def scrivi(output_path, blocchi):
    output_path.write_text(json.dumps({"pagine": [{"numero": 1, "blocchi": blocchi}]}),
                           encoding="utf-8")


def buona(workdir, output_path, **extra):
    lotto = json.loads((workdir / output_path.name.replace("piano.rivisto.", "piano."))
                       .read_text(encoding="utf-8"))
    output_path.write_text(json.dumps({"pagine": [
        {"numero": p["numero"],
         "blocchi": [{"id": b["id"], "testo": "TRADOTTO", "salta": False, **extra}
                     for b in p["blocchi"]]}
        for p in lotto["pagine"]]}), encoding="utf-8")
    return ok()


class RecuperoTest(unittest.TestCase):
    def esegui(self, agente, **kwargs):
        sessione = ai_review.ProviderSession("codex", "gpt-test", "codex")
        with tempfile.TemporaryDirectory() as tmp:
            radice = Path(tmp)
            with (patch.object(ai_review, "prepare_provider", return_value=sessione),
                  patch.object(ai_review, "export_pages"),
                  patch.object(ai_review, "run_agent", side_effect=agente) as chiamate):
                esito = ai_review.review(piano(), object(), [0, 1], radice, "en", "it",
                                         batch=1, **kwargs)
            manifesto = manifesto_lotti.leggi(radice)
            disco = {p.name: p.read_bytes() for p in radice.glob("piano.rivisto.*.json")}
        fuso, _ = ai_review.apply_review(piano(), esito.plan)
        testi = {b["id"]: b["testo"] for p in fuso["pagine"] for b in p["blocchi"]}
        return esito, fuso, testi, manifesto, chiamate.call_count, disco

    def test_un_blocco_dimenticato_non_trascina_il_vicino(self):
        """Il caso di imperial-exiles, lotto 13: 17 blocchi buoni su 18."""
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" in output_path.name:
                scrivi(output_path, [{"id": "p1-a", "testo": "Aspetta!", "salta": False}])
                return ok()
            return buona(workdir, output_path)

        esito, _, testi, manifesto, _, _ = self.esegui(agente)
        self.assertEqual(testi["p1-a"], "Aspetta!")
        self.assertEqual(testi["p1-b"], "No!", "il blocco mancante resta com'era")
        self.assertIn("p1-a", esito.outcome.ids)
        self.assertNotIn("p1-b", esito.outcome.ids, "chi non e' tornato non e' approvato")
        self.assertFalse(esito.outcome.complete)
        salvato = [i for i in esito.outcome.issues if i["code"] == "translation_batch_salvaged"]
        self.assertEqual(salvato[0]["ids"], ["p1-b"])
        self.assertEqual(manifesto["lotto01"]["stato"], manifesto_lotti.VALIDO)
        self.assertEqual(manifesto["lotto01"]["recupero_per_blocco"]["persi"], ["p1-b"])

    def test_un_blocco_con_escape_letterale_si_perde_da_solo(self):
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" in output_path.name:
                scrivi(output_path, [{"id": "p1-a", "testo": "Aspetta!", "salta": False},
                                     {"id": "p1-b", "testo": "No!\\nNo!", "salta": False}])
                return ok()
            return buona(workdir, output_path)

        esito, _, testi, _, _, _ = self.esegui(agente, ritenta=0)
        self.assertEqual(testi["p1-a"], "Aspetta!")
        self.assertEqual(testi["p1-b"], "No!")
        self.assertNotIn("p1-b", esito.outcome.ids)

    def test_una_risposta_senza_nessun_blocco_valido_resta_un_lotto_fallito(self):
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" in output_path.name:
                scrivi(output_path, [{"id": "p1-a", "testo": "", "salta": False}])
                return ok()
            return buona(workdir, output_path)

        esito, _, testi, manifesto, _, _ = self.esegui(agente, ritenta=0)
        self.assertEqual(testi["p1-a"], "Wait!")
        self.assertEqual(manifesto["lotto01"]["stato"], manifesto_lotti.FALLITO)
        self.assertIn("translation_batch_invalid", [i["code"] for i in esito.outcome.issues])

    def test_una_riattivazione_nel_recupero_resta_richiusa(self):
        protetto = piano()
        protetto["pagine"][0]["blocchi"][1]["salta"] = True

        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" in output_path.name:
                scrivi(output_path, [{"id": "p1-b", "testo": "No!", "salta": False}])
                return ok()
            return buona(workdir, output_path)

        sessione = ai_review.ProviderSession("codex", "gpt-test", "codex")
        with tempfile.TemporaryDirectory() as tmp:
            with (patch.object(ai_review, "prepare_provider", return_value=sessione),
                  patch.object(ai_review, "export_pages"),
                  patch.object(ai_review, "run_agent", side_effect=agente)):
                esito = ai_review.review(protetto, object(), [0, 1], Path(tmp), "en", "it",
                                         batch=1, ritenta=0)
        voce = {b["id"]: b for p in esito.plan["pagine"] for b in p["blocchi"]}["p1-b"]
        self.assertTrue(voce["salta"], "nessuna area protetta si riapre dal recupero")


class NuovoTentativoTest(unittest.TestCase):
    esegui = RecuperoTest.esegui

    def test_un_lotto_con_la_cli_fallita_viene_ritentato(self):
        tentativi = {"lotto01": 0}

        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" in output_path.name:
                tentativi["lotto01"] += 1
                if tentativi["lotto01"] == 1:
                    return SimpleNamespace(returncode=1, stdout="", stderr="os error 3")
            return buona(workdir, output_path)

        esito, _, testi, _, chiamate, _ = self.esegui(agente)
        self.assertEqual(chiamate, 3, "lotto01 due volte, lotto02 una")
        self.assertTrue(esito.outcome.complete)
        self.assertEqual(testi["p1-a"], "TRADOTTO")

    def test_si_tiene_il_tentativo_migliore_anche_se_non_e_l_ultimo(self):
        risposta_parziale = [{"id": "p1-a", "testo": "Aspetta!", "salta": False}]
        tentativi = {"n": 0}

        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" not in output_path.name:
                return buona(workdir, output_path)
            tentativi["n"] += 1
            if tentativi["n"] == 1:
                scrivi(output_path, risposta_parziale)
                return ok()
            return None  # il secondo giro va in timeout e non scrive niente

        esito, _, testi, manifesto, _, disco = self.esegui(agente)
        self.assertEqual(testi["p1-a"], "Aspetta!")
        salvata = disco["piano.rivisto.lotto01.json"]
        self.assertEqual(json.loads(salvata)["pagine"][0]["blocchi"], risposta_parziale,
                         "sul disco resta la risposta a cui il manifesto si lega")

    def test_ritenta_zero_non_richiama_il_modello(self):
        def agente(*_a, **_k):
            return SimpleNamespace(returncode=1, stdout="", stderr="x")

        _, _, _, _, chiamate, _ = self.esegui(agente, ritenta=0)
        self.assertEqual(chiamate, 2, "una chiamata per lotto; col default sarebbero 4")
        _, _, _, _, chiamate, _ = self.esegui(agente)
        self.assertEqual(chiamate, 4, "controprova: di default ogni lotto fallito si ritenta")

    def test_in_replay_a_cassette_non_si_ritenta(self):
        def agente(*_a, **_k):
            return SimpleNamespace(returncode=1, stdout="", stderr="x")

        with patch.object(ai_review.cassette, "attivo", return_value=True):
            _, _, _, _, chiamate, _ = self.esegui(agente)
        self.assertEqual(chiamate, 2, "una chiamata per lotto, nessun secondo giro")


class NaturaTest(unittest.TestCase):
    esegui = RecuperoTest.esegui

    def test_la_natura_si_registra_accanto_al_blocco(self):
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            return buona(workdir, output_path, natura="dialogo")

        esito, fuso, _, _, _, _ = self.esegui(agente)
        self.assertTrue(esito.outcome.complete)
        blocchi = {b["id"]: b for p in fuso["pagine"] for b in p["blocchi"]}
        self.assertEqual(blocchi["p1-a"]["natura"], "dialogo")

    def test_una_natura_sconosciuta_si_ignora_senza_perdere_il_lotto(self):
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            return buona(workdir, output_path, natura="balloon")

        esito, fuso, testi, _, _, _ = self.esegui(agente)
        self.assertTrue(esito.outcome.complete)
        self.assertIn("unknown_nature_dropped", [r["code"] for r in esito.outcome.repairs])
        self.assertEqual(testi["p1-a"], "TRADOTTO")
        self.assertNotIn("natura", fuso["pagine"][0]["blocchi"][0])


class ProiezioneTest(unittest.TestCase):
    def test_il_modello_vede_perche_un_blocco_e_protetto(self):
        pagine = [{"numero": 1, "blocchi": [
            {"id": "a", "originale": "HI", "testo": "HI", "salta": True,
             "_render": {"source_conflict": ["r1"], "skip_localizzazione": True}},
            {"id": "b", "originale": "YO", "testo": "YO", "salta": True,
             "_render": {"scartato_come_disegno": True}},
            {"id": "c", "originale": "OK", "testo": "OK", "salta": False,
             "natura": "dialogo", "_render": {"source_overlap": ["r2"]}},
            {"id": "d", "originale": "BLAM", "testo": "BLAM", "salta": True,
             "_render": {"salta_deciso": True}},
        ]}]
        voci = {b["id"]: b for b in ai_review._senza_render(pagine)[0]["blocchi"]}
        self.assertEqual(voci["a"]["protetto"], "sovrapposto")
        self.assertEqual(voci["b"]["protetto"], "scartato_come_disegno")
        self.assertNotIn("protetto", voci["c"], "un blocco disegnato non e' protetto")
        self.assertNotIn("natura", voci["c"], "la natura non torna al modello come dato")
        self.assertNotIn("protetto", voci["d"], "una decisione presa non e' una protezione")

    def test_il_prompt_chiede_sempre_la_traduzione_e_la_natura(self):
        self.assertIn("Traduci SEMPRE", ai_review.PROMPT)
        for natura in ai_review.NATURE:
            self.assertIn("`%s`" % natura, ai_review.PROMPT)
        self.assertIn("`natura`", ai_review.PROMPT)


if __name__ == "__main__":
    unittest.main()
