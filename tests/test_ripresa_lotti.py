"""P7: manifesto per lotto, salvataggio atomico e ripresa senza rifare il lavoro.

Fino a qui un lotto validato viveva solo in memoria: se la run moriva al lotto
5 di 7, anche i primi quattro erano da rifare e la spesa ripartiva da capo. E'
successo davvero su Amazing Spider-Man '63 parte 2, dove il timeout di un lotto
ha lasciato due pagine in inglese.

Le due proprieta' da difendere sono opposte e vanno tenute insieme:
- non ripetere un lotto gia' validato, e
- non spacciare per fatto un lotto il cui esito si e' perso. Una richiesta
  partita e mai tornata e' un `tentativo`, non un risultato.

Valgono per le due fasi che spendono a lotti: la revisione di traduzione e il
QA visuale, che ha una cartella e un manifesto suoi ma le stesse regole.
"""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator import manifesto_lotti
from comics_translator import pipeline_status as status
from comics_translator.traduzione import ai_review
from comics_translator.verifica import visual_qa

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/p0_contract.json").read_text("utf-8"))


def piano() -> dict:
    return copy.deepcopy(FIXTURE["plan"])


class _Agente:
    """Risposta valida, che si conta le chiamate: e' la misura del risparmio."""

    def __init__(self, testo="TRADOTTO"):
        self.chiamate = []
        self.testo = testo

    def __call__(self, _s, _p, workdir, _i, _t, output_path=None):
        self.chiamate.append(output_path.name)
        lotto = workdir / output_path.name.replace("piano.rivisto.", "piano.")
        originale = json.loads(lotto.read_text(encoding="utf-8"))
        output_path.write_text(json.dumps({"pagine": [
            {"numero": p["numero"],
             "blocchi": [{"id": b["id"], "testo": self.testo, "salta": False}
                         for b in p["blocchi"]]}
            for p in originale["pagine"]]}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class RipresaDeiLottiTest(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workdir, True)

    def esegui(self, agente, *, riprendi=False, model="gpt-test", batch=1):
        sessione = ai_review.ProviderSession("codex", model, "codex")
        with (patch.object(ai_review, "prepare_provider", return_value=sessione),
              patch.object(ai_review, "export_pages"),
              patch.object(ai_review, "run_agent", side_effect=agente)):
            return ai_review.review(piano(), object(), [0, 1], self.workdir,
                                    "en", "it", batch=batch, riprendi=riprendi)

    def manifesto(self):
        return ai_review.leggi_manifesto_lotti(self.workdir)

    # --- manifesto ---------------------------------------------------------

    def test_un_lotto_validato_lascia_una_traccia_sul_disco(self):
        esito = self.esegui(_Agente())

        lotti = self.manifesto()
        self.assertEqual(esito.outcome.completed, len(lotti))
        voce = lotti["lotto01"]
        self.assertEqual(voce["stato"], ai_review.LOTTO_VALIDO)
        risposta = self.workdir / voce["risposta"]
        self.assertEqual(voce["risposta_sha256"], status.sha256(risposta))

    def test_un_lotto_non_validato_resta_dichiarato_fallito(self):
        def agente(_s, _p, _w, _i, _t, output_path=None):
            # la CLI parte e muore: nessun file prodotto
            return SimpleNamespace(returncode=1, stdout="", stderr="os error 3")

        self.esegui(agente)

        for voce in self.manifesto().values():
            self.assertEqual(voce["stato"], ai_review.LOTTO_FALLITO)
            self.assertIn("exit 1", voce["errore"])

    def test_un_timeout_non_si_traveste_da_lotto_ripreso(self):
        """Il lotto ripreso e il lotto perso arrivano qui uguali: senza `proc`.

        run_agent torna None sia quando salta il modello perche' il lotto era
        gia' fatto, sia quando la CLI e' andata in timeout. Se i due casi si
        confondono, un lotto mai tornato passa per riuscito: e' il difetto
        esatto che la ripresa deve non avere.
        """
        def agente_in_timeout(_s, _p, workdir, _i, _t, output_path=None):
            # file presente ma vecchio: senza il timeout sembrerebbe una risposta
            output_path.write_text('{"pagine": []}', encoding="utf-8")
            return None

        esito = self.esegui(agente_in_timeout, riprendi=True)

        self.assertEqual(esito.outcome.completed, 0)
        self.assertTrue(any("timeout" in i["reason"] for i in esito.outcome.issues))

    def test_un_lotto_rifatto_bene_non_si_porta_dietro_l_errore_di_prima(self):
        """`valido` accanto a `errore` e' una voce che si contraddice da sola."""
        def agente_rotto(_s, _p, _w, _i, _t, output_path=None):
            return SimpleNamespace(returncode=1, stdout="", stderr="os error 3")

        self.esegui(agente_rotto)
        self.assertIn("errore", self.manifesto()["lotto01"])

        self.esegui(_Agente())

        voce = self.manifesto()["lotto01"]
        self.assertEqual(voce["stato"], ai_review.LOTTO_VALIDO)
        self.assertNotIn("errore", voce)

    # --- ripresa -----------------------------------------------------------

    def test_la_ripresa_non_richiama_il_modello_e_da_lo_stesso_esito(self):
        primo = self.esegui(_Agente())

        secondo_agente = _Agente()
        secondo = self.esegui(secondo_agente, riprendi=True)

        self.assertEqual(secondo_agente.chiamate, [])  # zero token spesi
        self.assertEqual(secondo.plan, primo.plan)
        self.assertEqual(secondo.outcome.completed, primo.outcome.completed)

    def test_senza_riprendi_una_vecchia_cartella_non_conta_come_lavoro_fatto(self):
        self.esegui(_Agente())

        agente = _Agente()
        self.esegui(agente)  # riprendi=False, il default

        self.assertEqual(len(agente.chiamate), 2)

    def test_un_lotto_solo_tentato_viene_rifatto(self):
        """L'esito di una richiesta remota persa non si puo' dichiarare valido."""
        self.esegui(_Agente())
        percorso = self.workdir / ai_review.MANIFESTO_LOTTI
        dati = json.loads(percorso.read_text(encoding="utf-8"))
        dati["lotti"]["lotto01"]["stato"] = ai_review.LOTTO_TENTATIVO
        percorso.write_text(json.dumps(dati), encoding="utf-8")

        agente = _Agente()
        self.esegui(agente, riprendi=True)

        self.assertEqual(agente.chiamate, ["piano.rivisto.lotto01.json"])

    def test_un_modello_diverso_riusa_comunque_la_risposta_di_prima(self):
        """Cambiare AI (es. credito esaurito) non deve buttare via lotti gia' validati."""
        self.esegui(_Agente())

        agente = _Agente()
        self.esegui(agente, riprendi=True, model="gpt-altro")

        self.assertEqual(agente.chiamate, [])

    def test_una_risposta_manomessa_non_viene_riusata(self):
        self.esegui(_Agente())
        risposta = self.workdir / "piano.rivisto.lotto01.json"
        risposta.write_text(risposta.read_text(encoding="utf-8") + " ", encoding="utf-8")

        agente = _Agente()
        self.esegui(agente, riprendi=True)

        self.assertEqual(agente.chiamate, ["piano.rivisto.lotto01.json"])

    def test_una_risposta_sparita_non_viene_riusata(self):
        self.esegui(_Agente())
        (self.workdir / "piano.rivisto.lotto01.json").unlink()

        agente = _Agente()
        self.esegui(agente, riprendi=True)

        self.assertEqual(agente.chiamate, ["piano.rivisto.lotto01.json"])

    def test_riprendere_una_cartella_vuota_e_una_run_normale(self):
        agente = _Agente()
        esito = self.esegui(agente, riprendi=True)

        self.assertEqual(len(agente.chiamate), 2)
        self.assertEqual(esito.outcome.completed, 2)


def piano_qa() -> dict:
    """Due pagine con un blocco ciascuna: a lotti da una, due lotti distinti."""
    return {"pagine": [
        {"numero": indice + 1, "indice": indice, "blocchi": [{
            "id": "p%03d-b001" % (indice + 1),
            "originale": "THE NAME IS NOVA",
            "testo": "Il nome e' Nuova",
            "salta": False,
            "_render": {"tipo": "immagine", "box_px": [0, 0, 100, 50]},
        }]}
        for indice in (0, 1)
    ]}


class _AgenteQa:
    """Correzione valida, che si conta le chiamate: e' la misura del risparmio."""

    def __init__(self, testo="Il nome e' Nova"):
        self.chiamate = []
        self.testo = testo

    def __call__(self, _s, _p, workdir, _i, _t, output_path=None):
        self.chiamate.append(output_path.name)
        lotto = workdir / output_path.name.replace(
            "correzioni.visuali.", "piano.visuale.")
        piano = json.loads(lotto.read_text(encoding="utf-8"))
        ids = [b["id"] for p in piano["pagine"] for b in p["blocchi"]]
        output_path.write_text(
            json.dumps({"set": {i: {"testo": self.testo} for i in ids}, "replace": {}}),
            encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class RipresaDeiLottiDelQaTest(unittest.TestCase):
    """Il QA visuale e' l'altra meta' della spesa, e si perdeva allo stesso modo."""

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workdir, True)

    def esegui(self, agente, *, riprendi=False, model="gpt-test", batch=1):
        sessione = ai_review.ProviderSession("codex", model, "codex")
        coppie = {
            indice: (self.workdir / ("p%02d.originale.png" % (indice + 1)),
                     self.workdir / ("p%02d.tradotta.png" % (indice + 1)))
            for indice in (0, 1)
        }
        with (patch.object(visual_qa.ai_review, "prepare_provider", return_value=sessione),
              patch.object(visual_qa, "export_comparisons", return_value=coppie),
              patch.object(visual_qa.ai_review, "run_agent", side_effect=agente)):
            return visual_qa.review_rendered(
                Path("sorgente.pdf"), Path("tradotto.pdf"), piano_qa(), [0, 1],
                self.workdir, batch=batch, riprendi=riprendi)

    def manifesto(self):
        return manifesto_lotti.leggi(self.workdir)

    def test_un_lotto_di_qa_validato_lascia_una_traccia_sul_disco(self):
        esito = self.esegui(_AgenteQa())

        lotti = self.manifesto()
        self.assertEqual(esito.validated_batches, len(lotti))
        voce = lotti["lotto01"]
        self.assertEqual(voce["stato"], manifesto_lotti.VALIDO)
        self.assertEqual(voce["risposta_sha256"],
                         status.sha256(self.workdir / voce["risposta"]))

    def test_la_ripresa_non_richiama_il_modello_e_da_lo_stesso_piano(self):
        primo = self.esegui(_AgenteQa())

        agente = _AgenteQa()
        secondo = self.esegui(agente, riprendi=True)

        self.assertEqual(agente.chiamate, [])  # zero token spesi
        self.assertEqual(secondo.plan, primo.plan)
        self.assertEqual(secondo.changes, primo.changes)
        self.assertEqual(secondo.reused, 2)
        self.assertTrue(secondo.complete)

    def test_senza_riprendi_una_vecchia_cartella_non_conta_come_lavoro_fatto(self):
        self.esegui(_AgenteQa())

        agente = _AgenteQa()
        self.esegui(agente)  # riprendi=False, il default

        self.assertEqual(len(agente.chiamate), 2)

    def test_solo_il_lotto_perso_viene_rifatto(self):
        """Il caso vero: la run muore a meta', e si ripaga solo il lotto in corso."""
        self.esegui(_AgenteQa())
        # il lotto 2 e' quello su cui e' morta: richiesta partita, esito perso
        percorso = self.workdir / manifesto_lotti.NOME_FILE
        dati = json.loads(percorso.read_text(encoding="utf-8"))
        dati["lotti"]["lotto02"]["stato"] = manifesto_lotti.TENTATIVO
        percorso.write_text(json.dumps(dati), encoding="utf-8")

        agente = _AgenteQa()
        esito = self.esegui(agente, riprendi=True)

        self.assertEqual(agente.chiamate, ["correzioni.visuali.lotto02.json"])
        self.assertEqual(esito.reused, 1)
        self.assertTrue(esito.complete)

    def test_un_lotto_di_qa_fallito_viene_rifatto(self):
        def agente_rotto(_s, _p, _w, _i, _t, output_path=None):
            output_path.write_text(
                '{"set": {"inesistente": {"salta": true}}, "replace": {}}',
                encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        self.esegui(agente_rotto)
        for voce in self.manifesto().values():
            self.assertEqual(voce["stato"], manifesto_lotti.FALLITO)

        agente = _AgenteQa()
        self.esegui(agente, riprendi=True)

        self.assertEqual(len(agente.chiamate), 2)

    def test_una_correzione_manomessa_non_viene_riusata(self):
        self.esegui(_AgenteQa())
        risposta = self.workdir / "correzioni.visuali.lotto01.json"
        risposta.write_text(risposta.read_text(encoding="utf-8") + " ", encoding="utf-8")

        agente = _AgenteQa()
        self.esegui(agente, riprendi=True)

        self.assertEqual(agente.chiamate, ["correzioni.visuali.lotto01.json"])

    def test_un_modello_diverso_riusa_comunque_le_correzioni_di_prima(self):
        """Cambiare AI (es. credito esaurito) non deve buttare via lotti gia' validati."""
        self.esegui(_AgenteQa())

        agente = _AgenteQa()
        self.esegui(agente, riprendi=True, model="gpt-altro")

        self.assertEqual(agente.chiamate, [])


class ScritturaAtomicaTest(unittest.TestCase):
    """Un file scritto a meta' da un crash sembra leggibile: e' il caso peggiore."""

    def setUp(self):
        self.cartella = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.cartella, True)

    def test_sostituisce_il_contenuto_senza_lasciare_temporanei(self):
        percorso = self.cartella / "piano.json"
        status.scrivi_atomico(percorso, '{"a": 1}')
        status.scrivi_atomico(percorso, '{"a": 2}')

        self.assertEqual(percorso.read_text(encoding="utf-8"), '{"a": 2}')
        self.assertEqual([p.name for p in self.cartella.iterdir()], ["piano.json"])

    def test_una_scrittura_interrotta_lascia_intatto_il_file_di_prima(self):
        percorso = self.cartella / "piano.json"
        status.scrivi_atomico(percorso, '{"a": 1}')

        with patch.object(status.os, "replace", side_effect=OSError("disco pieno")):
            with self.assertRaises(OSError):
                status.scrivi_atomico(percorso, '{"a": 2}')

        self.assertEqual(percorso.read_text(encoding="utf-8"), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
