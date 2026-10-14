"""Il tetto dell'attesa credito non deve scadere prima del reset dichiarato.

Osservato sul primo blocco credito reale (GLM, 9 settembre 2026):

    API Error: Request rejected (429) [1310][Weekly/Monthly Limit Exhausted.
    Your limit will reset at 2026-09-09 16:23:17]

Il reset era fra 7h34m, il tetto fisso e' 6h: la pipeline si sarebbe arresa
un'ora e mezza prima che il credito tornasse, pur avendo l'orario in mano.
"""

import unittest
from datetime import datetime

from comics_translator.traduzione.ai_review import (
    ATTESA_CREDITO_MARGINE_S,
    ATTESA_CREDITO_MAX_S,
    ATTESA_CREDITO_S,
    attesa_dal_messaggio,
)


def istante(anno, mese, giorno, ore, minuti, secondi=0):
    return datetime(anno, mese, giorno, ore, minuti, secondi).timestamp()


class AttesaDalMessaggioTest(unittest.TestCase):
    def test_il_caso_reale_glm(self):
        """Reset fra 7h34m: si aspetta fino a li', non 6h."""
        messaggio = ("API Error: Request rejected (429) [1310][Weekly/Monthly "
                     "Limit Exhausted. Your limit will reset at "
                     "2026-09-09 16:23:17][20260909144928]")
        adesso = istante(2026, 9, 9, 8, 49)
        attesa = attesa_dal_messaggio(messaggio, adesso)
        self.assertGreater(attesa, ATTESA_CREDITO_S,
                           "il tetto fisso avrebbe fatto rinunciare troppo presto")
        atteso = istante(2026, 9, 9, 16, 23, 17) - adesso + ATTESA_CREDITO_MARGINE_S
        self.assertAlmostEqual(attesa, atteso, delta=1.0)

    def test_senza_orario_resta_il_tetto_fisso(self):
        for messaggio in ("usage limit reached", "", "429 Too Many Requests"):
            self.assertEqual(attesa_dal_messaggio(messaggio, istante(2026, 9, 9, 8, 0)),
                             float(ATTESA_CREDITO_S))

    def test_reset_gia_passato_non_accorcia_ne_allunga(self):
        """Un messaggio riciclato non deve far rinunciare subito."""
        messaggio = "Your limit will reset at 2026-09-09 06:00:00"
        attesa = attesa_dal_messaggio(messaggio, istante(2026, 9, 9, 8, 49))
        self.assertEqual(attesa, float(ATTESA_CREDITO_S))

    def test_reset_vicino_resta_il_tetto_fisso(self):
        """Se il reset e' fra un'ora, il tetto di 6h e' gia' abbondante."""
        messaggio = "Your limit will reset at 2026-09-09 09:49:00"
        attesa = attesa_dal_messaggio(messaggio, istante(2026, 9, 9, 8, 49))
        self.assertEqual(attesa, float(ATTESA_CREDITO_S))

    def test_reset_assurdo_viene_limitato(self):
        """Un reset fra tre giorni non deve tenere il lock per tre giorni."""
        messaggio = "Your limit will reset at 2026-09-12 08:00:00"
        attesa = attesa_dal_messaggio(messaggio, istante(2026, 9, 9, 8, 49))
        self.assertEqual(attesa, float(ATTESA_CREDITO_MAX_S))

    def test_formati_accettati(self):
        adesso = istante(2026, 9, 9, 8, 0)
        for messaggio in (
            "Your limit will reset at 2026-09-09 20:00:00",
            "limit resets at 2026-09-09 20:00",
            "Rate limit reached. Resets: 2026-09-09T20:00:00",
        ):
            attesa = attesa_dal_messaggio(messaggio, adesso)
            self.assertGreater(attesa, ATTESA_CREDITO_S, messaggio)

    def test_data_impossibile_non_esplode(self):
        messaggio = "Your limit will reset at 2026-13-45 99:99:99"
        self.assertEqual(attesa_dal_messaggio(messaggio, istante(2026, 9, 9, 8, 0)),
                         float(ATTESA_CREDITO_S))


class ResetStrutturatoTest(unittest.TestCase):
    """Il rate_limit_event JSON della CLI di Claude (prova A2, spidey-63-9)."""

    EVENTO = ('{"type":"rate_limit_event","rate_limit_info":{"status":"rejected",'
              '"resetsAt":1789599000,"rateLimitType":"five_hour"}}')

    def test_resets_at_in_epoch_si_legge(self):
        from comics_translator.traduzione.ai_review import reset_dichiarato
        self.assertEqual(reset_dichiarato(self.EVENTO, 1789599000 - 3600), 1789599000.0)

    def test_la_sveglia_suona_al_reset_e_non_al_tetto(self):
        from unittest.mock import patch
        from comics_translator.traduzione import ai_review
        with (patch("comics_translator.traduzione.ai_review.time.time", return_value=1789599000 - 3600),
              patch("comics_translator.traduzione.ai_review._aspetta_via_libera",
                    return_value=True) as attesa):
            self.assertTrue(ai_review._aspetta_sblocco(self.EVENTO))
        self.assertEqual(attesa.call_args.kwargs["riprova_alle"],
                         1789599000 + ai_review.ATTESA_CREDITO_MARGINE_S)

    def test_il_reset_oltre_il_taglio_del_messaggio_resta(self):
        from types import SimpleNamespace
        from comics_translator.traduzione import ai_review
        stdout = ("x" * 300 + " You've hit your limit · resets 12:50am (Europe/Rome) "
                  + self.EVENTO)
        messaggio = ai_review._limite_esaurito(SimpleNamespace(returncode=1, stdout=stdout,
                                                               stderr=""))
        self.assertTrue(messaggio)
        self.assertIn("resetsAt", messaggio)


class ResetSoloOraTest(unittest.TestCase):
    """Il formato di Claude visto su batman: "resets 2:30am (Europe/Rome)"."""

    def test_ora_con_fuso_si_legge_come_prossima_occorrenza(self):
        from zoneinfo import ZoneInfo
        from comics_translator.traduzione.ai_review import reset_dichiarato

        roma = ZoneInfo("Europe/Rome")
        adesso = datetime(2026, 9, 14, 0, 5, tzinfo=roma).timestamp()
        reset = reset_dichiarato("You've hit your limit · resets 2:30am (Europe/Rome)", adesso)
        self.assertEqual(reset, datetime(2026, 9, 14, 2, 30, tzinfo=roma).timestamp())

    def test_un_ora_gia_passata_oggi_vale_domani(self):
        from zoneinfo import ZoneInfo
        from comics_translator.traduzione.ai_review import reset_dichiarato

        roma = ZoneInfo("Europe/Rome")
        adesso = datetime(2026, 9, 14, 19, 0, tzinfo=roma).timestamp()
        reset = reset_dichiarato("limit reached, resets 7:50pm (Europe/Rome)", adesso)
        self.assertEqual(reset, datetime(2026, 9, 14, 19, 50, tzinfo=roma).timestamp())
        reset = reset_dichiarato("resets 6pm (Europe/Rome)", adesso)
        self.assertEqual(reset, datetime(2026, 9, 15, 18, 0, tzinfo=roma).timestamp())

    def test_un_reset_lontano_allunga_l_attesa_oltre_il_tetto(self):
        from zoneinfo import ZoneInfo

        roma = ZoneInfo("Europe/Rome")
        adesso = datetime(2026, 9, 14, 8, 0, tzinfo=roma).timestamp()
        attesa = attesa_dal_messaggio("resets 11:00pm (Europe/Rome)", adesso)
        self.assertGreater(attesa, ATTESA_CREDITO_S)

    def test_numeri_senza_forma_di_orario_non_sono_un_reset(self):
        from comics_translator.traduzione.ai_review import reset_dichiarato

        for messaggio in ("resets in 5 hours", "usage limit reached 429", "reset 14"):
            self.assertIsNone(reset_dichiarato(messaggio, istante(2026, 9, 9, 8, 0)), messaggio)


class RiprovaAllOrarioTest(unittest.TestCase):
    """Il credito torna all'ora dichiarata: non serve un segnale da nessuno."""

    def test_passato_il_reset_dichiarato_si_riprova_da_soli(self):
        from unittest.mock import patch
        from comics_translator.traduzione import ai_review

        orologio = iter(range(0, 10_000, 10))
        with (patch.object(ai_review.status, "clock", side_effect=lambda: next(orologio)),
              patch.object(ai_review.time, "time", side_effect=[900.0, 1001.0]),
              patch.object(ai_review.time, "sleep"),
              patch.object(ai_review, "_tasto_premuto", return_value=b"")):
            self.assertTrue(ai_review._aspetta_via_libera(None, 3600, riprova_alle=1000.0))

    def test_senza_reset_dichiarato_resta_il_tetto(self):
        from unittest.mock import patch
        from comics_translator.traduzione import ai_review

        orologio = iter(range(0, 10_000, 1000))
        with (patch.object(ai_review.status, "clock", side_effect=lambda: next(orologio)),
              patch.object(ai_review.time, "sleep"),
              patch.object(ai_review, "_tasto_premuto", return_value=b"")):
            self.assertFalse(ai_review._aspetta_via_libera(None, 3000))


if __name__ == "__main__":
    unittest.main()
