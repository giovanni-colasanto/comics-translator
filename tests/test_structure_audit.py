"""P3.4: il confronto raster trova i difetti, non li dichiara assenti."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from comics_translator.disegno import plan as plan_module
from comics_translator.sorgente import source_inventory as inventory
from comics_translator.verifica import structure_audit as audit

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


BALLOON = [20, 20, 120, 60]
ART = [150, 100, 190, 130]
OUTSIDE = [150, 20, 190, 50]


def write_source(path):
    with fitz.open() as document:
        page = document.new_page(width=200, height=140)
        page.insert_text((30, 45), "HELLO", fontsize=12)
        page.draw_rect(fitz.Rect(*ART), color=None, fill=(0, 0, 0))
        document.save(path)


def build_plan(orfana=False):
    """`orfana`: una riga acquisita che nessun blocco del piano rivendica.

    E' il caso reale delle righe che l'OCR vede ma che restano fuori da ogni
    blocco: nessuna maschera le cancella e l'inglese resta in chiaro.
    """
    source = inventory.new_inventory()
    row = inventory.add_line(
        source, 0, raw="HELLO", normalized="HELLO", box=[28, 34, 92, 47], kind="test"
    )
    region = inventory.add_region(
        source, 0, [row], BALLOON, "HELLO", kind="test"
    )
    if orfana:
        libera = inventory.add_line(
            source, 0, raw="DOES", normalized="DOES", box=[28, 50, 60, 58], kind="test"
        )
        inventory.add_region(
            source, 0, [libera], [26, 48, 62, 60], "DOES", kind="test"
        )
    page = {
        "numero": 1,
        "indice": 0,
        "blocchi": [{
            "id": "b0",
            "originale": "HELLO",
            "automatica": "CIAO",
            "testo": "CIAO",
            "salta": False,
            "_source_region_id": region,
            "_render": {
                "tipo": "immagine",
                "scale": 1.0,
                "box_px": list(BALLOON),
                "fill_poly": [[[20, 20], [120, 20], [120, 60], [20, 60]]],
                "bg": [255, 255, 255],
                "light_text": False,
            },
        }],
    }
    inventory.attach_sources(page, source, preserve_ids=True)
    return {"versione": 2, "pagine": [page]}, row


def render_candidate(source_pdf, out, *, text="CIAO", fill=True, extra=None):
    """Disegna come farebbe il renderer, senza dipendere dal renderer."""
    with fitz.open(source_pdf) as document:
        page = document[0]
        if fill:
            page.draw_rect(fitz.Rect(*BALLOON), color=None, fill=(1, 1, 1), overlay=True)
        if text:
            page.insert_text((30, 45), text, fontsize=12)
        if extra:
            page.draw_rect(fitz.Rect(*extra), color=None, fill=(0, 0, 0), overlay=True)
        document.save(out)


class AuditPureFunctionTest(unittest.TestCase):
    def test_differences_below_the_channel_tolerance_are_not_changes(self):
        before = np.full((10, 10, 3), 200, np.uint8)
        after = before + audit.CHANNEL_TOLERANCE
        self.assertEqual(int((audit.changed_mask(before, after) > 0).sum()), 0)
        louder = before + audit.CHANNEL_TOLERANCE + 1
        self.assertEqual(int((audit.changed_mask(before, louder) > 0).sum()), 100)

    def test_pages_of_different_size_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensioni"):
            audit.changed_mask(
                np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8)
            )

    def test_a_row_owned_by_two_drawn_entries_is_reported(self):
        plan, row = build_plan()
        page = plan["pagine"][0]
        region = page["blocchi"][0]["sorgente"]["regione"]
        twin = copy.deepcopy(page["blocchi"][0])
        twin["id"] = "b1"
        twin["_render"]["p3_regioni"] = [region]
        page["blocchi"].append(twin)
        result = audit.coverage(page)
        self.assertEqual(
            result["righe_duplicate"], [{"riga": row, "blocchi": ["b0", "b1"]}]
        )

    def test_a_merged_entry_still_owns_the_rows_it_had_drawn(self):
        plan, row = build_plan()
        page = plan["pagine"][0]
        source = page["sorgente"]
        second_row = inventory.add_line(
            source, 0, raw="BYE", normalized="BYE", box=[28, 70, 92, 83], kind="test"
        )
        region = inventory.add_region(
            source, 0, [second_row], [20, 65, 120, 90], "BYE", kind="test"
        )
        source["lettura"]["regioni"].append(region)
        page["sorgente"] = inventory.seal(source)
        # Due frammenti OCR della stessa nuvoletta: profilo condiviso,
        # rettangoli di scrittura adiacenti. E' la fusione che il piano ammette.
        shared = [[[20, 20], [120, 20], [120, 90], [20, 90]]]
        absorbed = copy.deepcopy(page["blocchi"][0])
        absorbed.update(id="b1", originale="BYE", testo="ADDIO", unisci_con="b0",
                        sorgente={"regione": region, "righe": [second_row]})
        absorbed["_render"].update(box_px=[20, 55, 120, 90], fill_poly=shared)
        page["blocchi"][0]["_render"].update(box_px=[20, 20, 120, 55], fill_poly=shared)
        page["blocchi"].append(absorbed)
        self.assertEqual(
            [entry["id"] for entry in plan_module.drawable_entries(
                page["blocchi"], page=page)],
            ["b0"],
        )
        result = audit.coverage(page)
        self.assertEqual(result["righe_coperte"], 2)
        self.assertEqual(result["righe_non_coperte"], [])
        self.assertEqual(result["righe_duplicate"], [])

    def test_a_plan_without_source_rows_is_not_declared_clean(self):
        page = {"numero": 1, "blocchi": []}
        self.assertEqual(
            audit.coverage(page),
            {"misurabile": False, "motivo": "sorgente_senza_righe"},
        )

    def test_skipping_a_block_lowers_coverage_instead_of_raising_it(self):
        plan, row = build_plan()
        page = plan["pagine"][0]
        self.assertEqual(audit.coverage(page)["righe_coperte"], 1)
        page["blocchi"][0]["salta"] = True
        page["blocchi"][0]["_render"]["testo_illeggibile"] = True
        skipped = audit.coverage(page)
        self.assertEqual(skipped["righe_acquisite"], 1)
        self.assertEqual(skipped["righe_coperte"], 0)
        self.assertEqual(skipped["righe_non_coperte"], [row])
        self.assertEqual(
            skipped["blocchi_saltati"],
            [{"blocco": "b0", "motivo": "testo_illeggibile"}],
        )


    def test_una_riga_che_nessun_blocco_rivendica_e_orfana(self):
        """Il caso del grawlix di Captain America #2 p2: righe acquisite che
        non appartengono a nessun blocco, quindi nessuna maschera le cancella
        e l'inglese resta in chiaro sotto la traduzione del vicino."""
        plan, row = build_plan(orfana=True)
        page = plan["pagine"][0]
        orfana = next(r["id"] for r in page["sorgente"]["righe"]
                      if r["testo_normalizzato"] == "DOES")
        misura = audit.coverage(page)
        self.assertEqual(misura["righe_orfane"], [orfana])
        self.assertEqual(misura["righe_escluse_per_scelta"], [])
        self.assertIn(orfana, misura["righe_non_coperte"])
        self.assertNotIn(row, misura["righe_orfane"])

    def test_una_riga_di_un_blocco_saltato_non_e_orfana(self):
        """Un'esclusione dichiarata ha comunque un blocco che la copre: e' una
        scelta editoriale, non un difetto di rendering."""
        plan, row = build_plan()
        page = plan["pagine"][0]
        page["blocchi"][0]["salta"] = True
        page["blocchi"][0]["_render"]["testo_illeggibile"] = True
        misura = audit.coverage(page)
        self.assertEqual(misura["righe_orfane"], [])
        self.assertEqual(misura["righe_escluse_per_scelta"], [row])


class RigheScoperteTest(unittest.TestCase):
    """Inglese che nessuna maschera cancella, dentro un balloon riscritto.

    Su Captain America #2 pagina 2 l'OCR aveva spezzato un balloon lasciando
    tre righe senza padrone ('#', '#%&', 'DOES'): la traduzione e' stata
    scritta sopra, ma l'originale e' rimasto leggibile accanto.
    """

    #: il riempimento copre solo la meta' sinistra del riquadro di scrittura
    MEZZO = [[20, 20], [70, 20], [70, 60], [20, 60]]

    def pagina(self, box_riga, testo="DOES"):
        source = inventory.new_inventory()
        riga = inventory.add_line(source, 0, raw="HELLO", normalized="HELLO",
                                  box=[28, 34, 60, 47], kind="test")
        regione = inventory.add_region(source, 0, [riga], BALLOON, "HELLO", kind="test")
        libera = inventory.add_line(source, 0, raw=testo, normalized=testo,
                                    box=list(box_riga), kind="test")
        inventory.add_region(source, 0, [libera], list(box_riga), testo, kind="test")
        page = {
            "numero": 1, "indice": 0,
            "blocchi": [{
                "id": "b0", "originale": "HELLO", "automatica": "CIAO",
                "testo": "CIAO", "salta": False, "_source_region_id": regione,
                "_render": {
                    "tipo": "immagine", "scale": 1.0, "box_px": list(BALLOON),
                    "fill_poly": [list(self.MEZZO)], "bg": [255, 255, 255],
                    "light_text": False,
                },
            }],
        }
        inventory.attach_sources(page, source, preserve_ids=True)
        return page, libera

    def misura(self, box_riga, testo="DOES"):
        page, libera = self.pagina(box_riga, testo)
        rect = fitz.Rect(0, 0, 200, 140)
        dpi = audit.AUDIT_DPI
        size = (int(rect.height * dpi / 72.0), int(rect.width * dpi / 72.0))
        return audit.righe_scoperte(page, rect, dpi, size), libera

    def test_una_riga_scoperta_dentro_il_balloon_riscritto_e_un_residuo(self):
        scoperte, libera = self.misura([85, 30, 115, 45])
        self.assertEqual([r["riga"] for r in scoperte], [libera])
        self.assertTrue(scoperte[0]["leggibile"])
        self.assertLess(scoperte[0]["coperta"], 0.5)

    def test_una_riga_dentro_il_riempimento_e_gia_cancellata(self):
        scoperte, _ = self.misura([28, 30, 60, 45])
        self.assertEqual(scoperte, [])

    def test_una_riga_lontana_dalle_voci_disegnate_e_grafica_non_residuo(self):
        scoperte, _ = self.misura([150, 100, 190, 130])
        self.assertEqual(scoperte, [])

    def test_la_spazzatura_dell_ocr_viene_riportata_ma_non_blocca(self):
        scoperte, _ = self.misura([85, 30, 115, 45], testo="***")
        self.assertEqual(len(scoperte), 1)
        self.assertFalse(scoperte[0]["leggibile"])


class AuditRenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.source = self.tmp / "source.pdf"
        write_source(self.source)
        self.plan, self.row = build_plan()

    def run_audit(self, candidate, name, **kwargs):
        return audit.audit(
            self.source, self.plan, candidate, self.tmp / name, **kwargs
        )

    def test_a_render_inside_the_declared_area_is_clean(self):
        candidate = self.tmp / "ok.pdf"
        render_candidate(self.source, candidate)
        report = self.run_audit(candidate, "audit-ok")
        page = report["pagine"][0]
        self.assertEqual(report["esito"], "pulito")
        self.assertEqual(page["pixel_fuori_area"], 0)
        self.assertEqual(page["lobi_vuoti"], [])
        self.assertEqual(page["copertura"]["righe_coperte"], 1)
        self.assertGreater(page["pixel_cambiati"], 0)
        self.assertEqual(
            json.loads((self.tmp / "audit-ok/audit.json").read_text("utf-8")), report
        )

    def test_pixels_changed_outside_the_declared_area_are_reported(self):
        candidate = self.tmp / "sbava.pdf"
        render_candidate(self.source, candidate, extra=OUTSIDE)
        report = self.run_audit(candidate, "audit-sbava")
        self.assertEqual(report["esito"], "bloccante")
        self.assertGreater(report["pagine"][0]["pixel_fuori_area"], 0)
        self.assertTrue((self.tmp / "audit-sbava/p01-fuori-area.png").exists())

    def test_erasing_the_source_art_is_reported_even_when_nothing_is_written(self):
        candidate = self.tmp / "cancella.pdf"
        with fitz.open(self.source) as document:
            document[0].draw_rect(
                fitz.Rect(*ART), color=None, fill=(1, 1, 1), overlay=True
            )
            document.save(candidate)
        report = self.run_audit(candidate, "audit-cancella")
        self.assertGreater(report["pagine"][0]["pixel_fuori_area"], 0)

    def test_a_filled_lobe_left_without_ink_is_reported(self):
        candidate = self.tmp / "vuoto.pdf"
        render_candidate(self.source, candidate, text=None)
        report = self.run_audit(candidate, "audit-vuoto")
        page = report["pagine"][0]
        self.assertEqual(page["pixel_fuori_area"], 0)
        self.assertEqual(
            [(lobe["blocco"], lobe["lobo"]) for lobe in page["lobi_vuoti"]], [("b0", 0)]
        )
        self.assertEqual(page["voci_non_disegnate"], [])
        self.assertEqual(report["esito"], "bloccante")

    def test_an_entry_that_changed_nothing_is_not_called_an_empty_lobe(self):
        """Se la pagina e' identica alla sorgente il rendering non ha disegnato:
        e' un'omissione, non un balloon cancellato e lasciato vuoto."""
        candidate = self.tmp / "nulla.pdf"
        render_candidate(self.source, candidate, text=None, fill=False)
        page = self.run_audit(candidate, "audit-nulla")["pagine"][0]
        self.assertEqual(page["pixel_cambiati"], 0)
        self.assertEqual(page["lobi_vuoti"], [])
        self.assertEqual(
            page["voci_non_disegnate"], [{"blocco": "b0", "testo": "CIAO"}]
        )

    def test_the_reference_pdf_separates_drift_from_identity(self):
        candidate = self.tmp / "cand.pdf"
        render_candidate(self.source, candidate)
        drift = self.run_audit(
            candidate, "audit-drift", reference_pdf=self.source
        )["pagine"][0]["riferimento"]
        self.assertFalse(drift["identica"])
        self.assertGreater(drift["pixel_diversi"], 0)
        same = self.run_audit(
            candidate, "audit-same", reference_pdf=candidate
        )["pagine"][0]["riferimento"]
        self.assertTrue(same["identica"])
        self.assertEqual(same["pixel_diversi"], 0)

    def test_an_existing_output_directory_is_refused(self):
        candidate = self.tmp / "ok.pdf"
        render_candidate(self.source, candidate)
        (self.tmp / "occupata").mkdir()
        with self.assertRaisesRegex(ValueError, "esiste gia'"):
            self.run_audit(candidate, "occupata")

    def test_the_audit_does_not_mutate_the_plan(self):
        candidate = self.tmp / "ok.pdf"
        render_candidate(self.source, candidate)
        before = copy.deepcopy(self.plan)
        self.run_audit(candidate, "audit-intatto")
        self.assertEqual(self.plan, before)



class AuditNellaPipelineTest(unittest.TestCase):
    """L'audit agganciato al rendering: quello che trova deve arrivare al manifest."""

    def _misura(self, orfana=False, saltato=False, residuo=False, **rendering):
        from types import SimpleNamespace
        from unittest.mock import patch

        from comics_translator.disegno import pdf_translate_overlay as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, candidate = root / "source.pdf", root / "out.pdf"
            write_source(source)
            if saltato:
                rendering.setdefault("text", "")
                rendering.setdefault("fill", False)
            render_candidate(source, candidate, **rendering)
            piano, _ = build_plan(orfana=orfana)
            pagina = piano["pagine"][0]
            if residuo:
                # il riempimento copre meta' riquadro: nell'altra meta' resta
                # una riga inglese che nessun blocco rivendica
                blocco, riga = RigheScoperteTest().pagina([85, 30, 115, 45])
                pagina.update(blocco)
                del riga
            if saltato:
                pagina["blocchi"][0]["salta"] = True
                pagina["blocchi"][0]["_render"]["testo_illeggibile"] = True
            registrati, esiti = {}, []
            report = SimpleNamespace(
                artifact=lambda nome, percorso, **extra: registrati.update(
                    {nome: (Path(percorso), extra)})
            )
            with patch.object(pipeline.status, "finish_phase",
                              lambda risultato, final=True: esiti.append(risultato)):
                pipeline.esegui_audit_struttura(
                    source, candidate, piano, SimpleNamespace(audit=True), report)
            self.assertEqual(len(esiti), 1)
            return esiti[0], registrati

    def test_una_resa_dentro_le_aree_dichiarate_non_produce_difetti(self):
        esito, registrati = self._misura()
        self.assertEqual(esito.issues, [])
        self.assertEqual(esito.status, "complete")
        self.assertIn("audit_report", registrati)

    def test_i_pixel_fuori_area_diventano_un_difetto_del_manifest(self):
        esito, _ = self._misura(extra=OUTSIDE)
        self.assertEqual([i["code"] for i in esito.issues], ["audit_pixel_fuori_area"])
        self.assertEqual(esito.issues[0]["pages"], [1])
        self.assertEqual(esito.status, "failed")

    def test_un_lobo_riempito_e_lasciato_vuoto_diventa_un_difetto(self):
        esito, _ = self._misura(text="")
        self.assertIn("audit_lobi_vuoti", [i["code"] for i in esito.issues])
        self.assertEqual(esito.status, "failed")

    def test_un_residuo_leggibile_diventa_un_difetto_del_manifest(self):
        """La riga scoperta arriva fino al manifest, non resta nel rapporto."""
        esito, _ = self._misura(residuo=True)
        self.assertIn("audit_righe_scoperte", [i["code"] for i in esito.issues])
        self.assertEqual(esito.status, "failed")

    def test_una_riga_orfana_diventa_un_difetto_del_manifest(self):
        esito, _ = self._misura(orfana=True)
        self.assertIn("audit_righe_orfane", [i["code"] for i in esito.issues])
        self.assertEqual(esito.status, "failed")

    def test_una_riga_esclusa_per_scelta_non_e_un_difetto(self):
        esito, _ = self._misura(saltato=True)
        self.assertEqual(esito.issues, [])
        self.assertEqual(esito.status, "complete")

    def test_l_audit_non_puo_girare_due_volte_sulla_stessa_cartella(self):
        """La cartella porta l'orario: due render nello stesso secondo non
        devono far sparire il rapporto del primo in silenzio."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from comics_translator.disegno import pdf_translate_overlay as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, candidate = root / "source.pdf", root / "out.pdf"
            write_source(source)
            render_candidate(source, candidate)
            piano, _ = build_plan()
            esiti = []
            report = SimpleNamespace(artifact=lambda *a, **k: None)
            with (patch.object(pipeline.status, "finish_phase",
                               lambda risultato, final=True: esiti.append(risultato)),
                  patch.object(pipeline.time, "strftime", return_value="stessa-ora")):
                for _ in range(2):
                    pipeline.esegui_audit_struttura(
                        source, candidate, piano, SimpleNamespace(audit=True), report)
            self.assertEqual([i["code"] for i in esiti[1].issues], ["audit_non_eseguito"])
            self.assertEqual(esiti[1].status, "failed")

if __name__ == "__main__":
    unittest.main()
