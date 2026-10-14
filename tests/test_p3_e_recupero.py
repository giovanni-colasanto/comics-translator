"""Due difetti trovati dalla prova reale su imperial-guardians (15 settembre 2026).

1. Un blocco riattivato da un'unita' P3 multilobo eseguibile veniva
   riprotetto da `enforce_localization`; la validazione del lotto leggeva il
   salto come una riattivazione del modello e buttava il lotto intero (pagine
   17-18, due volte, con la risposta del modello identica ai `salta` iniziali).
2. Il recupero per blocco rifaceva la stessa prova finale tutto-o-niente:
   un blocco rifiutato bastava a perdere anche tutti gli altri.
"""

import copy
import unittest
from unittest.mock import patch

from comics_translator.sorgente import source_inventory as sources
from comics_translator.sorgente import text_localization as loc
from comics_translator.traduzione import ai_review

from test_sblocco_automatico import analyzed, line


def _pagina_protetta():
    # riga senza lettere dentro il balloon: protegge il blocco all'analisi
    return analyzed([line(), line("0400", (22, 22, 78, 30), reason="no_letters")])


class P3NonRiproteggeTest(unittest.TestCase):
    def test_un_blocco_riattivato_da_p3_non_torna_protetto_al_ricalcolo(self):
        built = _pagina_protetta()
        (blocco,) = built["pagine"][0]["blocchi"]
        self.assertTrue(blocco["salta"])
        # quello che fa activate_validated_units su un'unita' eseguibile
        render = blocco["_render"]
        render["p3_conflitti_precedenti"] = render.pop("source_conflict")
        render["p3_multilobo"] = True
        blocco["salta"] = False

        loc.enforce_localization(built)

        self.assertFalse(blocco["salta"], "il ricalcolo non revoca una decisione di P3")
        self.assertNotIn("source_conflict", render)
        sources.validate_sources(built)

    def test_la_revisione_che_conferma_i_salta_non_invalida_il_lotto(self):
        """Il caso del lotto 9: risposta identica, lotto valido."""
        built = _pagina_protetta()
        (blocco,) = built["pagine"][0]["blocchi"]
        render = blocco["_render"]
        render["p3_conflitti_precedenti"] = render.pop("source_conflict")
        render["p3_multilobo"] = True
        blocco["salta"] = False
        risposta = {"pagine": [{"numero": 1, "blocchi": [
            {"id": blocco["id"], "testo": "CIAO", "salta": False, "natura": "dialogo"}]}]}

        parziale, _ = ai_review.validate_and_repair_review_batch(built["pagine"], risposta)

        self.assertEqual(parziale["pagine"][0]["blocchi"][0]["salta"], False)

    def test_senza_p3_la_protezione_resta(self):
        """Controprova: lo stesso blocco senza la decisione di P3 resta protetto."""
        built = _pagina_protetta()
        (blocco,) = built["pagine"][0]["blocchi"]
        blocco["salta"] = False
        loc.enforce_localization(built)
        self.assertTrue(blocco["salta"])


class RecuperoScartaSoloIColpevoliTest(unittest.TestCase):
    def test_un_blocco_rifiutato_dalla_prova_finale_non_trascina_gli_altri(self):
        built = analyzed([line("UNO", (20, 20, 80, 32)), line("DUE", (20, 200, 80, 212))])
        primo, secondo = built["pagine"][0]["blocchi"]
        risposta = {"pagine": [{"numero": 1, "blocchi": [
            {"id": primo["id"], "testo": "ALFA", "salta": False},
            {"id": secondo["id"], "testo": "BETA", "salta": False}]}]}
        vera = ai_review.apply_corrections

        def rifiuta_il_primo(plan, spec, decisioni=True):
            if primo["id"] in spec.get("set", {}):
                raise ValueError("blocco protetto da source_conflict riattivato: %s" % primo["id"])
            return vera(plan, spec, decisioni=decisioni)

        with patch.object(ai_review, "apply_corrections", side_effect=rifiuta_il_primo):
            parziale, riparazioni, persi = ai_review.recupera_blocchi_validi(
                copy.deepcopy(built["pagine"]), risposta)

        tenuti = [b["id"] for p in parziale["pagine"] for b in p["blocchi"]]
        self.assertEqual(tenuti, [secondo["id"]])
        self.assertEqual(persi, [primo["id"]])
        self.assertIn("batch_salvaged_per_block", [r["code"] for r in riparazioni])

    def test_un_errore_che_non_nomina_blocchi_resta_tutto_o_niente(self):
        built = analyzed([line("UNO", (20, 20, 80, 32))])
        (blocco,) = built["pagine"][0]["blocchi"]
        risposta = {"pagine": [{"numero": 1, "blocchi": [
            {"id": blocco["id"], "testo": "ALFA", "salta": False}]}]}
        with patch.object(ai_review, "apply_corrections",
                          side_effect=ValueError("fusioni non valide")):
            parziale, _, persi = ai_review.recupera_blocchi_validi(built["pagine"], risposta)
        self.assertEqual(parziale, {"pagine": []})
        self.assertEqual(persi, [blocco["id"]])


if __name__ == "__main__":
    unittest.main()
