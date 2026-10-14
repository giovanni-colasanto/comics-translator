"""QA visuale: non storpia i nomi copiati dall'originale e non butta un lotto intero.

Due danni misurati sui FATTO: su 5-doom-storm-5 il QA ha rovinato quattro
crediti giusti ("Ayodele" -> "Anodele", "VC's" -> "DC's"); su thunderbolts e
deadpool-10 una sola `replace` non combaciante faceva cadere 9 lotti su 14 e 5
su 10, con tutte le correzioni buone dentro.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comics_translator.verifica import visual_qa


def piano(*blocchi):
    return {"pagine": [{"numero": 1, "indice": 0, "blocchi": [
        {"id": bid, "originale": orig, "testo": testo, "salta": False,
         "_render": {"tipo": "immagine", "box_px": [0, 0, 100, 50]}}
        for bid, orig, testo in blocchi]}]}


class NomiAlteratiTest(unittest.TestCase):
    def test_i_crediti_storpiati_vengono_scartati(self):
        p = piano(("a", "WRITER MUREWA AYODELE", "Testi Murewa Ayodele"),
                  ("b", "LETTERER VC'S TRAVIS LANHAM", "Lettering VC's Travis Lanham e"))
        spec = {"set": {}, "replace": {
            "a": [{"old": "Murewa Ayodele", "new": "Murewa Anodele"}],
            "b": [{"old": "VC's Travis Lanham e", "new": "DC's Travis Lanham e"}],
        }}
        filtrata, riparazioni = visual_qa.filtra_nomi_alterati(p, spec)
        self.assertEqual(filtrata["replace"], {})
        self.assertEqual(sorted(i for r in riparazioni for i in r["ids"]), ["a", "b"])

    def test_tradurre_una_parola_rimasta_in_inglese_resta_permesso(self):
        p = piano(("a", "WHAT?!", "WHAT?!"))
        spec = {"set": {}, "replace": {"a": [{"old": "WHAT", "new": "COSA"}]}}
        filtrata, riparazioni = visual_qa.filtra_nomi_alterati(p, spec)
        self.assertEqual(filtrata, spec)
        self.assertEqual(riparazioni, [])

    def test_tradurre_un_blocco_rimasto_in_inglese_resta_permesso(self):
        """Prova A2 (spidey-63-9): "IMPOSSIBLE" -> "IMPOSSIBILE" non e' un nome storpiato."""
        frase = "THIS IS IMPOSSIBLE! THE DOCTOR IS HERE."
        p = piano(("a", frase, frase), ("b", frase, frase))
        spec = {"set": {"a": {"testo": "È impossibile! Il dottore è qui."}},
                "replace": {"b": [{"old": "THIS IS IMPOSSIBLE!", "new": "È IMPOSSIBILE!"}]}}
        filtrata, riparazioni = visual_qa.filtra_nomi_alterati(p, spec)
        self.assertEqual(filtrata, spec)
        self.assertEqual(riparazioni, [])

    def test_correggere_un_nome_tradotto_per_sbaglio_resta_permesso(self):
        """Il caso del test storico: "Nova" tradotto "Nuova" va rimesso a posto."""
        p = piano(("a", "THE NAME IS NOVA", "Il nome e' Nuova"))
        spec = {"set": {"a": {"testo": "Il nome e' Nova"}}, "replace": {}}
        filtrata, _ = visual_qa.filtra_nomi_alterati(p, spec)
        self.assertEqual(filtrata, spec)

    def test_un_set_che_storpia_un_nome_perde_solo_il_testo(self):
        p = piano(("a", "HERE COMES ROCKET!", "Arriva Rocket!"))
        spec = {"set": {"a": {"testo": "Arriva Racket!", "corpo": 9}}, "replace": {}}
        filtrata, riparazioni = visual_qa.filtra_nomi_alterati(p, spec)
        self.assertEqual(filtrata["set"], {"a": {"corpo": 9}})
        self.assertEqual(len(riparazioni), 1)


class RecuperoPerBloccoTest(unittest.TestCase):
    def esegui(self, spec, sorgente):
        sessione = SimpleNamespace(ai="codex", model="gpt-test", cli="codex")
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            def agente(_s, _p, cartella, _i, _t, **_k):
                (cartella / "correzioni.visuali.lotto01.json").write_text(
                    json.dumps(spec), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (patch("comics_translator.verifica.visual_qa.ai_review.prepare_provider",
                        return_value=sessione),
                  patch("comics_translator.verifica.visual_qa.export_comparisons",
                        return_value={0: (workdir / "a.png", workdir / "b.png")}),
                  patch("comics_translator.verifica.visual_qa.ai_review.run_agent", side_effect=agente)):
                return visual_qa.review_rendered(Path("s.pdf"), Path("r.pdf"), sorgente,
                                                 [0], workdir)

    def test_una_replace_sbagliata_non_butta_le_correzioni_buone(self):
        sorgente = piano(("a", "HELLO", "Cioa"), ("b", "BYE", "Ciao ciao"))
        spec = {"set": {"a": {"testo": "Ciao"}},
                "replace": {"b": [{"old": "non c'e'", "new": "x"}]}}
        esito = self.esegui(spec, sorgente)
        testi = {b["id"]: b["testo"] for b in esito.plan["pagine"][0]["blocchi"]}
        self.assertEqual(testi, {"a": "Ciao", "b": "Ciao ciao"})
        self.assertEqual(esito.changes, 1)
        self.assertFalse(esito.complete, "un lotto recuperato non e' un lotto pulito")
        codici = [i["code"] for i in esito.outcome.issues]
        self.assertIn("qa_batch_salvaged", codici)

    def test_senza_nessuna_correzione_valida_il_lotto_resta_fallito(self):
        sorgente = piano(("a", "HELLO", "Ciao"))
        spec = {"set": {}, "replace": {"a": [{"old": "non c'e'", "new": "x"}]}}
        esito = self.esegui(spec, sorgente)
        self.assertEqual(esito.changes, 0)
        self.assertIn("qa_batch_invalid", [i["code"] for i in esito.outcome.issues])


if __name__ == "__main__":
    unittest.main()
