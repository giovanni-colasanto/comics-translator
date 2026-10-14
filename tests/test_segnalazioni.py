"""Gap C e Gap B: quello che la pipeline buttava via o lasciava in inglese.

I casi non sono inventati: testi e forme vengono dai piani gia' salvati
(Imperial parte 2, Daredevil #2, Captain America #2), misurati con
`python -m comics_translator.verifica.segnalazioni --radice .`.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from comics_translator.verifica import segnalazioni as seg


def _regione(testo, righe=1, motivo="background_noise", rumore=25.0, rid="p1-r1"):
    return {"id": rid, "tipo": "gruppo_ocr", "righe": ["r%d" % i for i in range(righe)],
            "testo_normalizzato": testo, "stato": "esclusa" if motivo else "acquisita",
            "motivo": motivo, "rumore_sfondo": rumore}


def _piano_regioni(*regioni, numero=7):
    return {"pagine": [{"numero": numero, "sorgente": {"regioni": list(regioni)}}]}


class TestoScartatoComeDisegnoTest(unittest.TestCase):
    """L'esclusione `background_noise` toglie il gruppo PRIMA della traduzione.

    Quello che scarta non arriva al modello, non arriva al QA e non manca
    nemmeno all'audit, che misura la copertura sulla sorgente sigillata: una
    riga mai acquisita non manca a nessuno. Su Imperial ha portato via interi
    balloon. Non si allenta la soglia: si dichiara.
    """

    def test_un_balloon_multiriga_scartato_viene_segnalato(self):
        # caso reale: Imperial parte 2 p7, 12 righe scartate a rumore 21.7
        testo = ("IHAVEDECLARED A PERIOD OF MOURNING.FIVE DAYSFOR THEFIVEFORMERWIVES "
                 "OFHIRO-KALA")
        got = seg.testo_scartato_come_disegno(_piano_regioni(_regione(testo, righe=12)))

        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["forma"], "blocco_di_testo")
        self.assertEqual(got[0]["pagina"], 7)
        self.assertEqual(got[0]["righe"], 12)

    def test_una_didascalia_breve_col_punto_viene_segnalata(self):
        # Imperial parte 2 p6 'LATER.', Daredevil #2 p4 'NEW YORK.'
        got = seg.testo_scartato_come_disegno(
            _piano_regioni(_regione("LATER.", rid="a"), _regione("NEW YORK.", rid="b")))

        self.assertEqual([x["id"] for x in got], ["a", "b"])
        self.assertEqual({x["forma"] for x in got}, {"didascalia"})

    def test_un_logo_di_una_riga_resta_disegno(self):
        # Daredevil #2 p1 'DAREDEVIL' a rumore 56.2: e' il logo della testata
        self.assertEqual(
            seg.testo_scartato_come_disegno(_piano_regioni(_regione("DAREDEVIL"))), [])

    def test_un_onomatopea_col_punto_esclamativo_resta_disegno(self):
        # su una riga sola il punto esclamativo e' marchio di copertine e SFX
        for testo in ("KRAK", "TOMBSTDNE!", "HA", "SPIDEY CUTSLOOSE!"):
            with self.subTest(testo=testo):
                self.assertEqual(
                    seg.testo_scartato_come_disegno(_piano_regioni(_regione(testo))), [])

    def test_una_battuta_esclamativa_su_due_righe_viene_segnalata(self):
        # caso reale: carnage-3 p8, balloon vero perso perche' chiude con "!"
        # e sta sotto la soglia del blocco di testo (2 righe, 24 caratteri)
        got = seg.testo_scartato_come_disegno(
            _piano_regioni(_regione("IDON'TLIKE THAT ONE BIT!", righe=2)))

        self.assertEqual([x["forma"] for x in got], ["esclamativa"])

    def test_un_onomatopea_ripetuta_su_due_righe_resta_disegno(self):
        # "BANG! BANG! BANG!" ha tre parole ma una sola distinta: e' un suono,
        # non una battuta. E' il caso che tiene onesta la regola esclamativa.
        for testo in ("BANG! BANG! BANG!", "MARVEL ENOW!"):
            with self.subTest(testo=testo):
                self.assertEqual(seg.testo_scartato_come_disegno(
                    _piano_regioni(_regione(testo, righe=2))), [])

    def test_un_numero_col_punto_non_e_una_didascalia(self):
        # prezzi, date e misure chiudono col punto ma restano disegno
        self.assertEqual(
            seg.testo_scartato_come_disegno(_piano_regioni(_regione("12.50."))), [])

    def test_gli_scarti_per_altri_motivi_non_c_entrano(self):
        # `too_short` e' un'altra decisione, con un'altra misura dietro
        lungo = "UNA FRASE LUNGA ABBASTANZA DA SEMBRARE UN BLOCCO"
        self.assertEqual(seg.testo_scartato_come_disegno(
            _piano_regioni(_regione(lungo, righe=4, motivo="too_short"))), [])

    def test_una_regione_tenuta_non_e_uno_scarto(self):
        lungo = "UNA FRASE LUNGA ABBASTANZA DA SEMBRARE UN BLOCCO"
        self.assertEqual(seg.testo_scartato_come_disegno(
            _piano_regioni(_regione(lungo, righe=4, motivo=None))), [])

    def test_un_piano_senza_inventario_non_esplode(self):
        # piani v1 e proiezioni non hanno la sorgente: niente da dire, non un errore
        self.assertEqual(seg.testo_scartato_come_disegno({"pagine": [{"numero": 1}]}), [])
        self.assertEqual(seg.testo_scartato_come_disegno({}), [])


def _blocco(testo, originale, salta=False, bid="b1"):
    return {"id": bid, "testo": testo, "originale": originale, "salta": salta}


def _piano_blocchi(*blocchi, numero=2):
    return {"pagine": [{"numero": numero, "blocchi": list(blocchi)}]}


class TraduzioneUgualeAllOriginaleTest(unittest.TestCase):
    """Gap B: sotto degrado il modello ricopia l'inglese invece di tradurre.

    Su una parola o due copiare e' la scelta giusta (onomatopee, nomi,
    insegne); su una frase intera e' traduzione mancata, e nel PDF si legge
    l'inglese senza che niente lo dica.
    """

    def test_una_frase_ricopiata_viene_segnalata(self):
        # caso reale: 'You are too late to help the man-monster, avengers!'
        frase = "You are too late to help the man-monster, avengers!"
        got = seg.traduzione_uguale_all_originale(_piano_blocchi(_blocco(frase, frase)))

        self.assertEqual([x["id"] for x in got], ["b1"])
        self.assertEqual(got[0]["pagina"], 2)

    def test_la_stessa_frase_in_maiuscolo_resta_una_copia(self):
        # rimettere in maiuscolo l'inglese non e' tradurre: il confronto
        # esatto lo lascerebbe passare
        got = seg.traduzione_uguale_all_originale(
            _piano_blocchi(_blocco("YOU ARE TOO  LATE, AVENGERS!",
                                   "You are too late, avengers!")))

        self.assertEqual(len(got), 1)

    def test_una_traduzione_vera_non_e_una_copia(self):
        self.assertEqual(seg.traduzione_uguale_all_originale(
            _piano_blocchi(_blocco("Sei arrivato troppo tardi, vendicatore!",
                                   "You are too late, avenger!"))), [])

    def test_una_o_due_parole_possono_essere_una_scelta(self):
        # nomi propri, insegne, onomatopee: copiare l'originale e' legittimo
        for testo in ("KRAK", "PETER PARKER", "DAILY BUGLE"):
            with self.subTest(testo=testo):
                self.assertEqual(seg.traduzione_uguale_all_originale(
                    _piano_blocchi(_blocco(testo, testo))), [])

    def test_un_blocco_saltato_non_viene_segnalato(self):
        # il salto e' una decisione dichiarata: li' l'inglese e' voluto
        frase = "You are too late to help the man-monster, avengers!"
        self.assertEqual(seg.traduzione_uguale_all_originale(
            _piano_blocchi(_blocco(frase, frase, salta=True))), [])

    def test_un_blocco_vuoto_non_e_una_copia(self):
        self.assertEqual(seg.traduzione_uguale_all_originale(
            _piano_blocchi(_blocco("", ""))), [])


class DialogoProtettoNonTradottoTest(unittest.TestCase):
    """Gap D: la protezione zittisce il blocco PRIMA che il modello lo traduca.

    Il modello lo riceve gia' con `salta: true`, fa quello che il prompt gli
    dice - ricopia l'originale - e l'inglese arriva nel PDF. Nessuno degli
    altri tre segnali lo vede. Misurato su carnage-3: 25 blocchi, di cui 19
    dialogo vero; il danno si leggeva a meta' balloon.
    """

    def _protetto(self, testo, bid="b1", **render):
        base = {"source_conflict": ["r1"], "skip_localizzazione": True}
        base.update(render)
        return {"id": bid, "testo": testo, "originale": testo,
                "salta": True, "_render": base}

    def test_un_balloon_protetto_e_mai_tradotto_viene_segnalato(self):
        # caso reale: carnage-3 p15, meta' del balloon di cui l'altra meta'
        # era gia' tradotta in "Decine"
        got = seg.dialogo_protetto_non_tradotto(
            _piano_blocchi(self._protetto("Peoplediedinthat policestation")))

        self.assertEqual([x["id"] for x in got], ["b1"])
        self.assertEqual(got[0]["pagina"], 2)

    def test_l_ocr_senza_spazi_non_inganna_il_riconoscimento(self):
        """Contare le parole qui non funziona: l'OCR dei fumetti perde gli spazi."""
        for testo in ("Youhired", "Alive?", "Mostof", "Why?Why", "I'llquit."):
            with self.subTest(testo=testo):
                self.assertEqual(len(seg.dialogo_protetto_non_tradotto(
                    _piano_blocchi(self._protetto(testo)))), 1)

    def test_un_suono_senza_vocali_resta_una_scelta_giusta(self):
        # "Ktk k-clk" e "Hnh." sono rumori disegnati: li' l'inglese ci va
        for testo in ("Ktk k-clk", "Hnh.", "SKKRK", "K-CLK"):
            with self.subTest(testo=testo):
                self.assertEqual(seg.dialogo_protetto_non_tradotto(
                    _piano_blocchi(self._protetto(testo))), [])

    def test_una_parola_corta_ma_vera_viene_comunque_segnalata(self):
        # caso reale: doom-1-2 p24, meta' del balloon "YOU DARE?!" - il blocco
        # protetto era solo "You" (3 lettere, 2 vocali), invisibile finche' la
        # soglia scartava tutto sotto le 4 lettere
        for testo in ("You", "Yes", "Yea"):
            with self.subTest(testo=testo):
                self.assertEqual(len(seg.dialogo_protetto_non_tradotto(
                    _piano_blocchi(self._protetto(testo)))), 1)

    def test_una_sigla_di_due_lettere_resta_esclusa(self):
        # il taglio netto non sparisce, si sposta: sotto le 3 lettere niente
        # supera comunque VOCALI_MINIME=2 su un totale cosi' corto
        for testo in ("Oh", "Ow", "Hm"):
            with self.subTest(testo=testo):
                self.assertEqual(seg.dialogo_protetto_non_tradotto(
                    _piano_blocchi(self._protetto(testo))), [])

    def test_un_blocco_con_traduzione_vera_e_del_gap_a(self):
        # qui l'italiano esiste, e' la protezione a nasconderlo: lo dice il Gap A
        b = self._protetto("WHERE AM I")
        b["testo"] = "Dove sono?"
        self.assertEqual(seg.dialogo_protetto_non_tradotto(_piano_blocchi(b)), [])

    def test_un_blocco_gia_sbloccato_a_mano_non_si_ripete(self):
        b = self._protetto("Peoplediedinthat policestation",
                           sbloccato_a_mano="area verificata")
        self.assertEqual(seg.dialogo_protetto_non_tradotto(_piano_blocchi(b)), [])

    def test_un_salto_deciso_dalla_revisione_non_e_una_protezione(self):
        # la revisione ha guardato la tavola e ha scelto: non e' lo stesso caso
        b = self._protetto("Peoplediedinthat policestation")
        b["_render"] = {"salta_deciso": True}
        self.assertEqual(seg.dialogo_protetto_non_tradotto(_piano_blocchi(b)), [])

    def test_un_blocco_disegnato_non_c_entra(self):
        b = self._protetto("Peoplediedinthat policestation")
        b["salta"] = False
        self.assertEqual(seg.dialogo_protetto_non_tradotto(_piano_blocchi(b)), [])


class ContieneTraduzioniTest(unittest.TestCase):
    """Un piano non tradotto ha `testo == originale` ovunque per costruzione.

    Vale per il piano dell'OCR e per le proiezioni mandate al modello:
    contarle come copie segnalerebbe l'albo intero. Misurato: 535 finte copie
    prima di questo controllo, 15 vere dopo.
    """

    def test_un_piano_non_tradotto_e_riconosciuto(self):
        self.assertFalse(seg.contiene_traduzioni(
            _piano_blocchi(_blocco("WHERE AM I", "WHERE AM I"),
                           _blocco("KRAK", "KRAK", bid="b2"))))

    def test_basta_una_traduzione_vera(self):
        self.assertTrue(seg.contiene_traduzioni(
            _piano_blocchi(_blocco("WHERE AM I", "WHERE AM I"),
                           _blocco("Dove sono?", "WHERE AM I", bid="b2"))))


class MisuraTest(unittest.TestCase):
    """La misura sui piani gia' salvati: nessuna cattura, nessuna chiamata AI."""

    def setUp(self):
        self.cartella = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.cartella, True)

    def _scrivi(self, nome, plan):
        percorso = self.cartella / nome
        percorso.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        return percorso

    def test_conta_scarti_sospetti_e_copie(self):
        frase = "You are too late to help the man-monster, avengers!"
        plan = {"pagine": [{
            "numero": 1,
            "sorgente": {"regioni": [_regione("LATER."), _regione("KRAK", rid="p1-r2")]},
            "blocchi": [_blocco(frase, frase), _blocco("Ciao", "HI", bid="b2")],
        }]}
        esito = seg.misura([self._scrivi("piano.json", plan)])

        self.assertEqual(esito["scartati"], 2)
        self.assertEqual(esito["scartati_sospetti"], 1)
        self.assertEqual(esito["copie"], 1)
        self.assertEqual(esito["non_tradotti"], 0)

    def test_un_piano_non_tradotto_non_conta_copie(self):
        frase = "You are too late to help the man-monster, avengers!"
        plan = _piano_blocchi(_blocco(frase, frase))
        esito = seg.misura([self._scrivi("piano.json", plan)])

        self.assertEqual(esito["copie"], 0)
        self.assertEqual(esito["non_tradotti"], 1)
        self.assertFalse(esito["piani"][0]["tradotto"])

    def test_un_file_illeggibile_e_dichiarato_non_saltato(self):
        percorso = self.cartella / "rotto.json"
        percorso.write_text("{non json", encoding="utf-8")
        esito = seg.misura([percorso])

        self.assertIn("errore", esito["piani"][0])


if __name__ == "__main__":
    unittest.main()
