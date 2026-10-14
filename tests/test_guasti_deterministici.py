"""P7.5: i guasti non devono ne' perdere lavoro valido ne' promuovere il parziale.

Il piano chiede test deterministici per timeout, output troncato, errore CLI,
disco pieno, annullamento, lock concorrente e credito esaurito, con due
proprieta' da difendere in ogni caso: **conservare i risultati gia' validi** e
**non promuovere l'output parziale**.

Timeout e attesa credito erano gia' coperti (`test_ai_review`, `test_cassette`).
Qui si fissano gli altri. Il caso dell'errore CLI viene dal vero: il 1 settembre
2026 una workdir relativa ha fatto uscire la CLI codex con `os error 3` su tutti
i lotti, e la run e' arrivata in fondo con esito `partial` e zero campi rivisti.
Il comportamento era corretto - nulla di parziale e' stato promosso - ma non era
fissato da nessun test, quindi nulla impediva a una modifica futura di romperlo.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.traduzione import ai_review

# Stesso piano valido usato dagli altri test del contratto P0. Costruirne uno a
# mano qui vorrebbe dire inventare un inventario sorgente che il caricatore
# rifiuta, e duplicare una fixture che esiste gia'.
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/p0_contract.json").read_text("utf-8"))


def piano() -> dict:
    return copy.deepcopy(FIXTURE["plan"])


def _testi(plan: dict) -> dict:
    return {b["id"]: b["testo"] for p in plan["pagine"] for b in p["blocchi"]}


class GuastiDelLottoTest(unittest.TestCase):
    """Un lotto che va storto non deve trascinarsi dietro quelli riusciti."""

    def esegui(self, agente, batch: int = 1):
        """Esegue la revisione e la fonde come fa la pipeline vera.

        `review` restituisce **solo** le pagine dei lotti riusciti: le altre non
        ci sono proprio. E' `apply_review` a rimetterle nel piano completo, ed
        e' li' che si vede se un lotto perso ha danneggiato gli altri. Provare
        `review` da sola misurerebbe meta' del contratto.
        """
        sessione = ai_review.ProviderSession("codex", "gpt-test", "codex")
        completo = piano()
        with tempfile.TemporaryDirectory() as tmp:
            with (patch.object(ai_review, "prepare_provider", return_value=sessione),
                  patch.object(ai_review, "export_pages"),
                  patch.object(ai_review, "run_agent", side_effect=agente)):
                esito = ai_review.review(completo, object(), [0, 1], Path(tmp),
                                         "en", "it", batch=batch)
        fuso, _ = ai_review.apply_review(piano(), esito.plan)
        return esito, fuso

    @staticmethod
    def buona(workdir, output_path):
        """Risposta valida: traduce ogni blocco del lotto che l'ha richiesta."""
        lotto = workdir / output_path.name.replace("piano.rivisto.", "piano.")
        originale = json.loads(lotto.read_text(encoding="utf-8"))
        output_path.write_text(json.dumps({"pagine": [
            {"numero": p["numero"],
             "blocchi": [{"id": b["id"], "testo": "TRADOTTO", "salta": False}
                         for b in p["blocchi"]]}
            for p in originale["pagine"]]}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def test_errore_cli_su_un_lotto_non_perde_i_lotti_riusciti(self):
        """Il caso reale del 1 settembre, ridotto a un lotto solo."""
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto02" in output_path.name:
                # la CLI e' partita ed e' morta: nessun file prodotto
                return SimpleNamespace(returncode=1, stdout="", stderr="os error 3")
            return self.buona(workdir, output_path)

        esito, fuso = self.esegui(agente)
        self.assertEqual(esito.outcome.failed, 1)
        self.assertFalse(esito.outcome.complete,
                         "un lotto perso rende l'esito incompleto")
        self.assertEqual(_testi(fuso)["p1-a"], "TRADOTTO",
                         "il lotto riuscito non va perso per colpa di un altro")
        self.assertIn("p1-a", esito.outcome.ids,
                      "gli id validati sono quelli che il chiamante lascera' tradotti")

    def test_output_troncato_non_viene_applicato_a_meta(self):
        """JSON incompleto: si scarta il lotto, non si applica quel che si e' letto."""
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" in output_path.name:
                output_path.write_text('{"pagine": [{"numero": 1, "blocc',
                                       encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return self.buona(workdir, output_path)

        esito, fuso = self.esegui(agente)
        self.assertEqual(esito.outcome.failed, 1)
        self.assertEqual(_testi(fuso)["p1-a"], "Wait!",
                         "un JSON troncato non viene applicato a meta'")
        self.assertNotIn("p1-a", esito.outcome.ids,
                         "il blocco non validato non risulta approvato")

    def test_risposta_che_inventa_blocchi_estranei_non_scrive_l_intruso(self):
        """Un lotto non puo' scrivere blocchi che non gli appartengono.

        L'intruso non entra mai. I blocchi validi del lotto invece si
        recuperano uno per uno: prima cadeva tutto il lotto, e la battuta buona
        tornava in inglese fino al `.finale`. L'esito resta incompleto.
        """
        def agente(_s, _p, workdir, _i, _t, output_path=None):
            if "lotto01" in output_path.name:
                output_path.write_text(json.dumps({"pagine": [
                    {"numero": 1, "blocchi": [
                        {"id": "p1-a", "testo": "TRADOTTO", "salta": False},
                        {"id": "p9-intruso", "testo": "INTRUSO", "salta": False}]}]}),
                    encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return self.buona(workdir, output_path)

        esito, fuso = self.esegui(agente)
        self.assertEqual(esito.outcome.failed, 1)
        self.assertFalse(esito.outcome.complete)
        self.assertNotIn("INTRUSO", set(_testi(fuso).values()))
        self.assertEqual(_testi(fuso)["p1-a"], "TRADOTTO",
                         "la battuta valida del lotto si recupera")
        self.assertIn("translation_batch_salvaged",
                      [i["code"] for i in esito.outcome.issues])

    def test_tutti_i_lotti_falliti_non_promuovono_nulla(self):
        """Il 1 settembre e' successo esattamente questo: 0 campi rivisti."""
        def agente(*_a, **_k):
            return SimpleNamespace(returncode=1, stdout="", stderr="os error 3")

        esito, fuso = self.esegui(agente)
        self.assertEqual(esito.outcome.completed, 0)
        self.assertFalse(esito.outcome.complete)
        self.assertEqual(esito.outcome.ids, [], "nessun id risulta approvato")
        self.assertNotIn("TRADOTTO", set(_testi(fuso).values()),
                         "niente di parziale viene promosso")


class PianoSuDiscoTest(unittest.TestCase):
    """Un piano scritto a meta' e' peggio di un piano assente: verrebbe riletto."""

    def test_un_piano_troncato_sul_disco_viene_rifiutato_alla_rilettura(self):
        """Rifiuta uscendo, non restituendo meta' piano: e' il punto che conta.

        `load_plan` esce con `SystemExit` invece di sollevare `ValueError`:
        e' pensato per essere chiamato da riga di comando. Quello che il test
        difende non e' il tipo dell'eccezione ma il fatto che un piano
        troncato non venga mai restituito come se fosse buono.
        """
        from comics_translator.disegno.plan import load_plan

        with tempfile.TemporaryDirectory() as tmp:
            percorso = Path(tmp) / "piano.json"
            intero = json.dumps(piano(), ensure_ascii=False)
            percorso.write_text(intero[: len(intero) // 2], encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_plan(percorso)

    def test_un_piano_valido_sopravvive_al_giro_completo(self):
        """Controprova: il rifiuto sopra non e' dovuto al formato in se'."""
        from comics_translator.disegno.plan import load_plan, save_plan

        with tempfile.TemporaryDirectory() as tmp:
            percorso = Path(tmp) / "piano.json"
            save_plan(percorso, piano())
            self.assertEqual(len(load_plan(percorso)["pagine"]),
                             len(piano()["pagine"]))


if __name__ == "__main__":
    unittest.main()
