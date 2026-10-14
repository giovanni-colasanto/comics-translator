import unittest
from types import SimpleNamespace

from comics_translator.disegno.image_overlay import _profili_sicuri
from comics_translator.disegno.plan import (
    _image_block_entry,
    _protect_contained_blocks,
    _protect_page_if_wide_compound,
    merge_review,
    validate_merges,
)
from comics_translator.verifica.qa_corrections import apply_corrections


def sample_plan():
    return {
        "pagine": [{
            "numero": 1,
            "blocchi": [
                {"id": "a", "testo": "Testo brutto", "_render": {"tipo": "immagine", "box_px": [0, 0, 100, 50]}},
                {"id": "b", "testo": "Segue", "_render": {"tipo": "immagine", "box_px": [0, 55, 100, 100]}},
            ],
        }]
    }


class CorrectionTest(unittest.TestCase):
    def test_set_and_exact_replace_do_not_mutate_source(self):
        source = sample_plan()
        corrected, changes = apply_corrections(source, {
            "set": {"a": {"salta": False}},
            "replace": {"a": [{"old": "brutto", "new": "naturale"}]},
        })
        self.assertEqual(changes, 2)
        self.assertEqual(corrected["pagine"][0]["blocchi"][0]["testo"], "Testo naturale")
        self.assertNotIn("salta", source["pagine"][0]["blocchi"][0])

    def test_unknown_id_fails(self):
        with self.assertRaisesRegex(ValueError, "id sconosciuti"):
            apply_corrections(sample_plan(), {"set": {"missing": {"salta": True}}})

    def test_replace_must_match_exactly_once(self):
        with self.assertRaisesRegex(ValueError, "trovato 0 volte"):
            apply_corrections(sample_plan(), {"replace": {"a": [{"old": "assente", "new": "x"}]}})

    def test_invalid_merge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fusioni non valide"):
            apply_corrections(sample_plan(), {"set": {"a": {"unisci_con": "missing"}}})

    def test_adjacent_fragments_can_merge(self):
        plan = sample_plan()
        plan["pagine"][0]["blocchi"][0]["unisci_con"] = "b"
        self.assertEqual(validate_merges(plan), [])

    def test_contained_blocks_cannot_merge(self):
        plan = sample_plan()
        plan["pagine"][0]["blocchi"][1]["_render"]["box_px"] = [10, 10, 80, 40]
        plan["pagine"][0]["blocchi"][1]["unisci_con"] = "a"
        self.assertTrue(validate_merges(plan))

    def test_oversized_fill_profile_is_discarded(self):
        local = [[0, 0], [100, 0], [100, 50], [0, 50]]
        leaked = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
        self.assertEqual(_profili_sicuri([local, leaked], [0, 0, 100, 50]), [local])

    def test_long_multi_profile_ocr_group_is_kept_untouched(self):
        job = SimpleNamespace(
            scale=0.18, box_px=[0, 0, 1000, 800], text="x" * 120,
            translated="traduzione", fill_poly=[[[0, 0], [1, 0], [1, 1]],
                                                  [[2, 0], [3, 0], [3, 1]]],
            bg=(255, 255, 255), light_text=False, upper=False, base_size=10,
        )
        entry = _image_block_entry(job, 0, SimpleNamespace(width=1000, height=1000))
        self.assertTrue(entry["salta"])
        self.assertTrue(entry["_render"]["compound"])

        plan = {"pagine": [{"blocchi": [entry]}]}
        reviewed = {"pagine": [{"blocchi": [{"id": entry["id"], "salta": False}]}]}
        changed, unknown = merge_review(plan, reviewed)
        self.assertEqual((changed, unknown), (0, 0))
        self.assertTrue(entry["salta"])

    def test_single_profile_covering_the_page_is_kept_untouched(self):
        # Caso reale: il clustering OCR ha unito il riassunto editoriale e le
        # colonne dei credits in un gruppo solo, con un unico profilo largo
        # quasi un quarto della pagina. Tradurlo cancella i credits.
        job = SimpleNamespace(
            scale=0.18, box_px=[0, 0, 3351, 2582], text="x" * 531,
            translated="traduzione", fill_poly=[[[0, 0], [3351, 0], [3351, 2582]]],
            bg=(255, 255, 255), light_text=False, upper=False, base_size=10,
        )
        page = SimpleNamespace(width=5120 * 0.18, height=7413 * 0.18)
        entry = _image_block_entry(job, 1, page)
        self.assertTrue(entry["salta"])
        self.assertTrue(entry["_render"]["compound"])

        plan = {"pagine": [{"blocchi": [entry]}]}
        reviewed = {"pagine": [{"blocchi": [{"id": entry["id"], "salta": False}]}]}
        self.assertEqual(merge_review(plan, reviewed), (0, 0))
        self.assertTrue(entry["salta"])

    def test_ordinary_balloon_stays_translatable(self):
        job = SimpleNamespace(
            scale=0.18, box_px=[0, 0, 872, 476], text="x" * 200,
            translated="traduzione", fill_poly=[[[0, 0], [872, 0], [872, 476]]],
            bg=(255, 255, 255), light_text=False, upper=False, base_size=10,
        )
        page = SimpleNamespace(width=5120 * 0.18, height=7413 * 0.18)
        entry = _image_block_entry(job, 4, page)
        self.assertFalse(entry["salta"])
        self.assertFalse(entry["_render"]["compound"])

    def test_wide_compound_protects_the_whole_editorial_page(self):
        entries = [
            {"id": "wide", "salta": True, "_render": {
                "tipo": "immagine", "compound": True,
                "box_pt": [50, 10, 850, 160],
            }},
            {"id": "neighbor", "salta": False, "_render": {
                "tipo": "immagine", "compound": False,
                "box_pt": [50, 170, 350, 500],
            }},
        ]
        self.assertTrue(_protect_page_if_wide_compound(entries, 1000))
        self.assertTrue(all(entry["salta"] for entry in entries))
        self.assertTrue(all(
            entry["_render"]["page_protected"] for entry in entries
        ))

        plan = {"pagine": [{"blocchi": entries}]}
        reviewed = {"pagine": [{"blocchi": [
            {"id": "neighbor", "salta": False},
        ]}]}
        self.assertEqual(merge_review(plan, reviewed), (0, 0))
        self.assertTrue(entries[1]["salta"])

    def test_narrow_compound_does_not_protect_the_page(self):
        entries = [{"id": "narrow", "salta": True, "_render": {
            "tipo": "immagine", "compound": True,
            "box_pt": [50, 10, 650, 300],
        }}]
        self.assertFalse(_protect_page_if_wide_compound(entries, 1000))
        self.assertNotIn("page_protected", entries[0]["_render"])

    def _coppia(self, poly_a, poly_b, box_a=(0, 0, 100, 50), box_b=(0, 55, 100, 100)):
        return {"pagine": [{"blocchi": [
            {"id": "a", "testo": "Prima meta'", "originale": "First half",
             "unisci_con": "b",
             "_render": {"tipo": "immagine", "box_px": list(box_a), "fill_poly": [poly_a]}},
            {"id": "b", "testo": "Seconda meta'", "originale": "Second half",
             "_render": {"tipo": "immagine", "box_px": list(box_b), "fill_poly": [poly_b]}},
        ]}]}

    def test_fragments_of_the_same_balloon_can_merge(self):
        # Stessa nuvoletta spezzata in due dall'OCR: i profili coincidono.
        poly = [[0, 0], [120, 0], [120, 110], [0, 110]]
        self.assertEqual(validate_merges(self._coppia(poly, list(poly))), [])

    def test_two_touching_balloons_cannot_merge(self):
        # Nuvolette diverse: fondendole il testo viene riscritto nell'unione
        # dei due rettangoli, sborda dai bordi e lascia vuota quella sotto.
        alto = [[0, 0], [120, 0], [120, 60], [0, 60]]
        basso = [[0, 52], [120, 52], [120, 115], [0, 115]]
        problemi = validate_merges(self._coppia(alto, basso))
        self.assertEqual(len(problemi), 1)
        self.assertIn("nuvolette diverse", problemi[0])

    def test_short_multi_profile_group_is_kept_untouched(self):
        # Nei piani raccolti nessun gruppo a due profili si e' reso bene: il
        # caso da 108 caratteri di pagina 3 sfuggiva alla soglia dei 120.
        job = SimpleNamespace(
            scale=0.18, box_px=[0, 0, 1022, 329], text="x" * 108,
            translated="traduzione", fill_poly=[[[0, 0], [1, 0], [1, 1]],
                                                [[2, 0], [3, 0], [3, 1]]],
            bg=(255, 255, 255), light_text=False, upper=False, base_size=10,
        )
        page = SimpleNamespace(width=5120 * 0.18, height=7413 * 0.18)
        entry = _image_block_entry(job, 2, page)
        self.assertTrue(entry["salta"])
        self.assertTrue(entry["_render"]["compound"])

    def _blocco(self, id_, box, testo="Testo"):
        return {"id": id_, "testo": testo, "originale": testo, "salta": False,
                "_render": {"tipo": "immagine", "box_px": list(box),
                            "fill_poly": [[[box[0], box[1]], [box[2], box[1]],
                                           [box[2], box[3]]]]}}

    def test_fragment_inside_another_block_is_kept_untouched(self):
        # Caso reale: l'OCR estrae "I'll" sia dentro il balloon grande sia come
        # frammento a se'. Scritti entrambi, il testo finisce uno sull'altro.
        grande = self._blocco("grande", (0, 0, 400, 300), "I'll pack you a bag.")
        frammento = self._blocco("frammento", (80, 30, 260, 130), "I'll")
        entries = [grande, frammento]
        self.assertEqual(_protect_contained_blocks(entries), 1)
        self.assertFalse(grande["salta"])
        self.assertTrue(frammento["salta"])
        self.assertEqual(frammento["_render"]["contenuto_in"], "grande")

        plan = {"pagine": [{"blocchi": entries}]}
        reviewed = {"pagine": [{"blocchi": [{"id": "frammento", "salta": False}]}]}
        self.assertEqual(merge_review(plan, reviewed), (0, 0))
        self.assertTrue(frammento["salta"])

    def test_balloons_that_only_touch_stay_translatable(self):
        # Due nuvolette vicine che si sfiorano non sono un frammento duplicato.
        primo = self._blocco("primo", (0, 0, 200, 100))
        secondo = self._blocco("secondo", (180, 90, 380, 190))
        self.assertEqual(_protect_contained_blocks([primo, secondo]), 0)
        self.assertFalse(primo["salta"])
        self.assertFalse(secondo["salta"])


if __name__ == "__main__":
    unittest.main()
