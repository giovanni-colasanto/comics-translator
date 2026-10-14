"""Il banco di prova sui FATTO misura senza toccare niente e senza chiamare il modello."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from comics_translator import manifesto_lotti
from comics_translator.verifica import banco_fatto

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/p0_contract.json").read_text("utf-8"))


class BancoFattoTest(unittest.TestCase):
    def test_un_blocco_perso_dal_lotto_e_ora_recuperato_viene_contato(self):
        with tempfile.TemporaryDirectory() as tmp:
            albo = Path(tmp) / "FATTO prova"
            rev = albo / "prova.it.revisione"
            rev.mkdir(parents=True)
            base = copy.deepcopy(FIXTURE["plan"])
            (rev / "piano.json").write_text(json.dumps(base), encoding="utf-8")
            # la run originale: lotto01 fallito (blocco p1-b mancante), tutto unreviewed
            automatico = copy.deepcopy(base)
            for b in automatico["pagine"][0]["blocchi"]:
                b.update(salta=True, _render={"unreviewed": True})
            (rev / "piano.applicato.json").write_text(json.dumps(automatico), encoding="utf-8")
            (rev / "piano.rivisto.lotto01.json").write_text(json.dumps({"pagine": [
                {"numero": 1, "blocchi": [{"id": "p1-a", "testo": "Aspetta!", "salta": False}]}]}),
                encoding="utf-8")
            manifesto_lotti.registra(rev, "lotto01", manifesto_lotti.TIPO_REVISIONE,
                                     stato=manifesto_lotti.FALLITO, pagine=[1])
            manuale = copy.deepcopy(base)
            manuale["pagine"][0]["blocchi"][0]["testo"] = "Aspetta!"
            (rev / "piano.manuale.json").write_text(json.dumps(manuale), encoding="utf-8")
            prima = {p.name: p.read_bytes() for p in rev.iterdir()}

            esito = banco_fatto.misura_albo(albo)

            dopo = {p.name: p.read_bytes() for p in rev.iterdir()}
        self.assertEqual(prima, dopo, "il banco non scrive nella cartella dell'albo")
        self.assertEqual(esito["lotti"], {"recuperato_per_blocco": 1})
        self.assertEqual(esito["blocchi_recuperati_dai_lotti"], 1)
        self.assertEqual(esito["unreviewed_sbloccati_a_mano"], 1)
        self.assertEqual(esito["unreviewed_a_mano_ora_recuperati"], 1)

    def test_una_cartella_incompleta_si_dichiara_saltata(self):
        with tempfile.TemporaryDirectory() as tmp:
            esito = banco_fatto.misura_albo(Path(tmp))
        self.assertIn("saltato", esito)


if __name__ == "__main__":
    unittest.main()
