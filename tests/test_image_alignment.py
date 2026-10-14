"""Regression for B00: image captions must honor the plan's alignment."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf as fitz

from comics_translator.disegno import pdf_translate_overlay as pipeline
from comics_translator.disegno.image_overlay import ImageJob, draw_group
from comics_translator.disegno.plan import _image_block_entry, rebuild_page


def entry(alignment=None):
    block = {
        "id": "caption", "originale": "A caption", "testo": "Una didascalia",
        "_render": {"tipo": "immagine", "box_px": [10, 10, 150, 80], "scale": 1},
    }
    if alignment is not None:
        block["allinea"] = alignment
    return block


class ImageAlignmentTest(unittest.TestCase):
    def test_rebuild_honors_left_and_keeps_legacy_centered(self):
        for alignment, centered in (("sinistra", False), ("centro", True), (None, True)):
            with self.subTest(alignment=alignment):
                _, jobs = rebuild_page([entry(alignment)], SimpleNamespace, ImageJob)
                self.assertEqual(jobs[0].centered, centered)

    def test_image_alignment_survives_serialization(self):
        for centered in (True, False):
            with self.subTest(centered=centered):
                job = ImageJob(0, [10, 10, 150, 80], "Caption", False, 10,
                               (255, 255, 255), False, 1, centered=centered)
                saved = _image_block_entry(job, 0, fitz.Rect(0, 0, 200, 200))
                self.assertEqual(saved["allinea"], "centro" if centered else "sinistra")

    def positions(self, **alignment):
        font = fitz.Font("helv")
        page = SimpleNamespace(rect=fitz.Rect(0, 0, 200, 200))
        fonts = SimpleNamespace(pick=lambda **kwargs: font)
        with patch("comics_translator.disegno.image_overlay.fitz.TextWriter") as writer:
            draw_group(page, [10, 10, 150, 80], "Wide phrase I", 1,
                       (255, 255, 255), False, 10, fonts, 8,
                       lambda *args: (10, ["Wide phrase", "I"]),
                       riempi=False, **alignment)
            return [call.args[0].x for call in writer.return_value.append.call_args_list]

    def test_left_alignment_starts_every_line_at_inner_left(self):
        self.assertEqual(self.positions(centered=False), [11.5, 11.5])

    def test_default_preserves_centered_balloon_layout(self):
        first, second = self.positions()
        self.assertGreater(first, 11.5)
        self.assertGreater(second, first)

    def test_pipeline_passes_alignment_in_both_render_phases(self):
        _, jobs = rebuild_page([entry("sinistra")], SimpleNamespace, ImageJob)
        doc = MagicMock()
        with patch("comics_translator.disegno.pdf_translate_overlay.draw_group") as draw:
            pipeline._render_document(doc, [0], {}, {0: jobs}, None, None,
                                      SimpleNamespace(min_size=8), "unused.pdf")
        self.assertEqual(draw.call_count, 2)
        self.assertTrue(all(call.kwargs["centered"] is False for call in draw.call_args_list))



class TestoCheNonEntra(unittest.TestCase):
    """Un riquadro molto piu' piccolo del testo: meglio l'originale."""

    def _job(self, testo, larghezza, altezza):
        return dict(
            box_px=[0, 0, larghezza, altezza], text=testo, scale=1.0,
            bg=(255, 255, 255), light_text=False, base_size=9.0,
            min_size=4.5, fit_fn=pipeline.fit, fill_poly=[],
        )

    def test_non_disegna_e_non_copre_quando_il_testo_non_entra(self):
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        fonts = pipeline.load_fontset("arial")
        lungo = "Quanto a zia May, lei pensava che il sole sorgesse e tramontasse su suo nipote!"
        prima = len(page.get_drawings())
        esito = draw_group(page=page, fonts=fonts, **self._job(lungo, 18, 6))
        self.assertFalse(esito)
        self.assertEqual(len(page.get_drawings()), prima)  # nessun riempimento
        self.assertEqual(page.get_text().strip(), "")      # nessuna scritta
        doc.close()

    def test_un_testo_che_ci_sta_viene_disegnato(self):
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        fonts = pipeline.load_fontset("arial")
        esito = draw_group(page=page, fonts=fonts, **self._job("CIAO", 120, 40))
        self.assertTrue(esito)
        self.assertIn("CIAO", page.get_text())
        doc.close()


class CorpoDiRiferimentoTest(unittest.TestCase):
    """`base_size` non deve dipendere da una riga fusa dall'OCR.

    La soglia di leggibilita' e' relativa a `base_size`: se un logo o un
    titolo disegnato finisce nel gruppo, la media delle altezze si alza, il
    testo vero risulta "troppo piccolo" e il blocco resta in inglese.
    """

    def gruppo(self, altezze):
        from comics_translator.disegno.image_overlay import Group, OcrLine

        righe = [OcrLine(10, y, 90, y + h, "RIGA", 0.9)
                 for y, h in zip(range(0, 1000, 100), altezze)]
        return Group(lines=righe)

    def test_righe_omogenee_non_cambiano_corpo(self):
        gruppo = self.gruppo([10, 10, 10, 10])
        self.assertEqual(gruppo.corpo_righe, gruppo.avg_height)

    def test_una_riga_fusa_alza_la_media_ma_non_la_mediana(self):
        gruppo = self.gruppo([10, 10, 10, 40])
        self.assertEqual(gruppo.corpo_righe, 10)
        self.assertGreater(gruppo.avg_height, 15)

    def test_col_corpo_gonfiato_la_traduzione_verrebbe_rifiutata(self):
        """Lo stesso testo, lo stesso riquadro: cambia solo il riferimento."""
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        fonts = pipeline.load_fontset("arial")
        gruppo = self.gruppo([10, 10, 10, 40])
        comune = dict(box_px=[0, 0, 80, 22],
                      text="Una battuta di prova un po' piu' lunga del solito",
                      scale=1.0, bg=(255, 255, 255), light_text=False,
                      min_size=4.5, fit_fn=pipeline.fit, fill_poly=[])
        gonfio = draw_group(page=page, fonts=fonts,
                            base_size=gruppo.avg_height * 0.78, **comune)
        giusto = draw_group(page=page, fonts=fonts,
                            base_size=gruppo.corpo_righe * 0.78, **comune)
        self.assertFalse(gonfio, "con la media il testo viene giudicato illeggibile")
        self.assertTrue(giusto, "con la mediana la traduzione entra e viene scritta")
        doc.close()


if __name__ == "__main__":
    unittest.main()
