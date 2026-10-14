"""Chi disegna dichiara cosa la protezione sta zittendo, non solo chi traduce.

`render_plan` ricalcola la protezione prima di disegnare (`enforce_localization`):
e' il posto giusto, perche' i `salta` possono essere cambiati dopo il piano da
una correzione QA, da un riempimento ritoccato a mano o da un `--plan-in` di un
piano gia' fatto. Quel ricalcolo pero' puo' rendere invisibile un blocco che nel
piano era attivo e TRADOTTO, e fino a qui nessuno lo diceva: la dichiarazione
stava solo sul percorso di traduzione.

Misurato su imperial-guardians p26 (prova A1, 16 settembre 2026): il balloon
"AT YOUR BIDDING. / HOW DO YOU WANT 'EM?", tradotto e attivo nel piano, e'
sparito al rendering per un conflitto sorgente nato li' - e la corsa ha chiuso
con esito "complete".
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz

from comics_translator import pipeline_status as status
from comics_translator.disegno import pdf_translate_overlay as pipeline
from comics_translator.sorgente import source_inventory as sources


def piano(motivo_riga="low_confidence"):
    """Un blocco tradotto e attivo, con una riga sorgente che nessuno riscrive.

    Come su p26: la riga e' una rilettura incerta scartata dall'OCR, quindi non
    appartiene a nessun blocco, e sporge dal riquadro che il blocco ricopre.
    """
    inventario = sources.new_inventory()
    sources.add_line(inventario, 1, raw="AT YOUR BIDDING.", normalized="AT YOUR BIDDING.",
                     box=[12, 12, 88, 38], kind="ocr")
    sources.add_line(inventario, 1, raw="'EM?", normalized="'EM?",
                     box=[80, 38, 95, 50], kind="ocr", reason=motivo_riga)
    propria, sporgente = inventario["righe"]
    sources.add_region(inventario, 1, [propria["id"]], [12, 12, 88, 38],
                       "AT YOUR BIDDING.", kind="gruppo_ocr")
    regione = inventario["regioni"][0]
    inventario["lettura"] = {"metodo": "geometrico_y_x", "confermato": False,
                             "regioni": [regione["id"]]}
    inventario["localizzazione"] = {"aree": [
        {"riga": propria["id"], "blocco": "p1-b1", "box_pt": propria["box_pt"],
         "quad_pt": [[12, 12], [88, 12], [88, 38], [12, 38]]},
        {"riga": sporgente["id"], "blocco": None, "box_pt": sporgente["box_pt"],
         "quad_pt": [[80, 38], [95, 38], [95, 50], [80, 50]]},
    ]}
    return {
        "versione": 2,
        "pagine": [{
            "numero": 1,
            "blocchi": [{
                "id": "p1-b1",
                "originale": "AT YOUR BIDDING.",
                "testo": "Ai tuoi ordini.",
                "salta": False,
                "sorgente": {"regione": regione["id"], "righe": [propria["id"]]},
                "_render": {"tipo": "testo", "rect": [10, 10, 90, 40],
                            "source_overlap": [sporgente["id"]]},
            }],
            "sorgente": sources.seal(inventario),
        }],
    }


class Args:
    image_font = "arial"
    min_size = 4.5
    grow = 1.0


class SoppressioneAlRenderingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.radice = Path(self.tmp.name)
        self.sorgente = self.radice / "src.pdf"
        with fitz.open() as doc:
            doc.new_page(width=100, height=60)
            doc.save(self.sorgente)
        with patch("comics_translator.pipeline_status.code_provenance", return_value={}):
            self.report = status.RunReport.create(self.tmp.name, {"model": "test"}, self.tmp.name)
        token = status.ACTIVE_REPORT.set(self.report)
        self.addCleanup(status.ACTIVE_REPORT.reset, token)

    def rendi(self, plan):
        uscita = self.radice / "out.pdf"

        def disegna(doc, selected, per_page, image_jobs, fonts, image_fonts, args, out_path):
            doc.save(out_path)

        with patch("comics_translator.disegno.pdf_translate_overlay._render_document", side_effect=disegna):
            pipeline.render_plan(self.sorgente, plan, [0], None, Args(), uscita)

    def codici(self):
        return [v["code"] for v in status.RunReport.load(self.report.path).data.get("issues", [])]

    def test_una_traduzione_zittita_dal_ricalcolo_finisce_nel_manifest(self):
        plan = piano()
        self.rendi(plan)
        blocco = plan["pagine"][0]["blocchi"][0]
        self.assertTrue(blocco["salta"], "il ricalcolo deve aver protetto il blocco")
        self.assertIn("traduzione_soppressa_da_protezione", self.codici())

    def test_senza_soppressioni_il_manifest_resta_pulito(self):
        # la riga e' vuota: non c'e' inchiostro da difendere, il blocco si disegna
        plan = piano(motivo_riga="empty_text")
        self.rendi(plan)
        self.assertFalse(plan["pagine"][0]["blocchi"][0]["salta"])
        self.assertNotIn("traduzione_soppressa_da_protezione", self.codici())

    def test_la_stessa_perdita_non_si_conta_due_volte(self):
        """Traduzione e rendering la dichiarano entrambi: e' un guasto solo."""
        plan = piano()
        blocco = plan["pagine"][0]["blocchi"][0]
        blocco["salta"] = True
        blocco["_render"].update(
            source_conflict=blocco["_render"]["source_overlap"], skip_localizzazione=True)
        pipeline.segnala_traduzioni_soppresse(plan)   # percorso di traduzione
        pipeline.segnala_traduzioni_soppresse(plan)   # percorso di rendering
        self.assertEqual(self.codici().count("traduzione_soppressa_da_protezione"), 1)
        # Anche l'evento e la riga a schermo vanno detti una volta: sul registro
        # di superman-1 le stesse 25 battute risultavano perse due volte.
        registro = self.report.path.with_name("events.jsonl").read_text(encoding="utf-8")
        eventi = [json.loads(r)["event"] for r in registro.splitlines() if r.strip()]
        self.assertEqual(eventi.count("traduzioni_soppresse"), 1)


if __name__ == "__main__":
    unittest.main()
