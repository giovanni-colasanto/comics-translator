"""P4.2: la maschera di cancellazione non deve colare fuori dal balloon.

Il bianco dentro un balloon e il bianco del margine fra due vignette sono lo
stesso colore. Dove il contorno del balloon tocca il bordo della vignetta, o
lo sfiora, la componente connessa di sfondo esce dal balloon, attraversa il
margine e raggiunge lo sfondo chiaro della vignetta accanto: la maschera
cancella il bordo della tavola e fonde due balloon in un'unica macchia.

Misurato su `captain-america-2-codex-5-4-16p-r1` il 1 settembre 2026:
16 coppie di maschere sovrapposte su 9 pagine di 16.
"""

import unittest

import numpy as np

from comics_translator.disegno import plan
from comics_translator.disegno.image_overlay import _balloon_mask

SFONDO = (255, 255, 255)


def tavola(ponte: int) -> np.ndarray:
    """Due balloon bianchi su vignette scure, uniti da un ponte di `ponte` px.

    Il ponte e' il varco per cui la componente connessa cola: nella pagina vera
    e' il margine chiaro fra le vignette, qui e' un corridoio verticale.
    """
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    img[40:150, 40:260] = 255      # balloon in alto
    img[250:360, 40:260] = 255     # balloon in basso, altra vignetta
    if ponte:
        meta = 150
        img[150:250, meta - ponte // 2: meta + ponte // 2 + ponte % 2] = 255
    return img


def righe_del_testo():
    """Una riga OCR dentro il solo balloon superiore."""
    return [[70, 80, 230, 110]]


def copre(mask, offset, box) -> float:
    """Frazione di `box` coperta dalla maschera, in coordinate immagine."""
    ox, oy = offset
    x0, y0, x1, y1 = box
    ritaglio = mask[max(0, y0 - oy):max(0, y1 - oy), max(0, x0 - ox):max(0, x1 - ox)]
    if ritaglio.size == 0:
        return 0.0
    return float(np.count_nonzero(ritaglio)) / ritaglio.size


class BalloonMaskLeakTest(unittest.TestCase):
    def test_balloon_isolato_viene_coperto_per_intero(self):
        """Senza ponte non c'e' niente da sigillare: il caso normale non cambia."""
        img = tavola(ponte=0)
        mask, offset = _balloon_mask(img, [60, 70, 240, 120], SFONDO,
                                     righe=righe_del_testo())
        self.assertIsNotNone(mask)
        self.assertGreater(copre(mask, offset, [45, 45, 255, 145]), 0.95,
                           "il balloon che contiene il testo va coperto tutto")

    def test_la_maschera_non_raggiunge_il_balloon_della_vignetta_accanto(self):
        """Il ponte chiaro non deve portare la cancellazione sull'altro balloon."""
        img = tavola(ponte=8)
        mask, offset = _balloon_mask(img, [60, 70, 240, 120], SFONDO,
                                     righe=righe_del_testo())
        self.assertIsNotNone(mask)
        self.assertGreater(copre(mask, offset, [45, 45, 255, 145]), 0.95,
                           "il balloon con il testo resta coperto per intero")
        self.assertLess(copre(mask, offset, [45, 255, 255, 355]), 0.05,
                        "il balloon accanto non va toccato: e' di un'altra voce")

    def test_un_ponte_largo_quanto_il_balloon_non_e_un_ponte(self):
        """Sigillare non deve spezzare un balloon davvero fatto a clessidra.

        Se il collegamento e' largo quasi quanto le due parti non e' una
        colatura ma la forma del balloon: va conservata, altrimenti si
        cancellerebbe meta' della battuta.
        """
        img = tavola(ponte=180)
        mask, offset = _balloon_mask(img, [60, 70, 240, 120], SFONDO,
                                     righe=righe_del_testo())
        self.assertIsNotNone(mask)
        self.assertGreater(copre(mask, offset, [45, 255, 255, 355]), 0.9,
                           "una strozzatura larga fa parte della stessa area")



class MascheraDentroIMuriTest(unittest.TestCase):
    """Il profilo da cancellare deve fermarsi ai muri dei vicini.

    `prepara_area` riceve gia' i limiti di crescita e li usa per il riquadro di
    scrittura, ma non per il profilo. Due balloon separati da un margine chiaro
    producono cosi' due maschere che se lo contendono: sovrapponendosi
    cancellano il bordo della vignetta e fondono i balloon in una macchia sola.
    Misurato su `captain-america-2-codex-5-4-16p-r1`: 16 coppie su 9 pagine.

    Le proporzioni qui sotto tengono la macchia unita sotto le 5 aree del
    riquadro, altrimenti scatta `_profili_sicuri` e il caso non si riproduce.
    """

    BOX = [60, 50, 240, 100]
    NOSTRE = [[70, 60, 230, 90]]
    ALTRUI = [[70, 160, 230, 190]]
    MURO = [0, 0, 300, 125]      # a meta' fra la nostra riga e quella del vicino

    def tavola(self):
        img = np.zeros((260, 300, 3), dtype=np.uint8)
        img[40:110, 40:260] = 255     # balloon con il nostro testo
        img[110:140, 40:260] = 250    # margine chiaro fra le vignette
        img[140:210, 40:260] = 255    # balloon del vicino
        return img

    def profilo(self, limits):
        from comics_translator.disegno.image_overlay import prepara_area

        _, profili = prepara_area(self.tavola(), self.BOX, SFONDO, limits=limits,
                                  vietati=self.ALTRUI, righe=self.NOSTRE)
        return profili

    def test_senza_muri_ferma_comunque_l_intorno_delle_righe(self):
        """Due difese indipendenti: i muri dei vicini e l'intorno delle righe.

        Prima questa prova fotografava il difetto (senza muri la macchia
        arrivava sul balloon accanto, a y > 140). Ora la macchia si ferma comunque
        a meno di una riga dalle proprie righe: e' il ritaglio che tiene il
        riempimento lontano dalla cornice della vignetta, dove i muri non ci
        sono perche' non c'e' nessun blocco vicino (Absolute Superman p5).
        """
        profili = self.profilo(None)
        self.assertTrue(profili, "la macchia unita esiste ed e' sotto la soglia di sicurezza")
        self.assertLess(max(p[1] for pr in profili for p in pr), 140,
                        "l'intorno delle righe deve fermare la macchia prima del vicino")

    def test_il_profilo_non_supera_il_muro_verso_il_vicino(self):
        profili = self.profilo(self.MURO)
        self.assertTrue(profili, "un profilo va prodotto: il balloon esiste")
        self.assertLessEqual(max(p[1] for pr in profili for p in pr), 132,
                             "la cancellazione ha superato il muro e invade il vicino")

    def test_il_profilo_copre_comunque_tutto_il_proprio_testo(self):
        """Il muro non deve mai tagliare via l'originale che stiamo coprendo."""
        profili = self.profilo([0, 0, 300, 70])   # muro assurdo, dentro il nostro testo
        self.assertTrue(profili)
        self.assertGreaterEqual(max(p[1] for pr in profili for p in pr), 90,
                                "il proprio testo va coperto anche se il muro e' piu' stretto")


class RiquadroSulProprioTestoTest(unittest.TestCase):
    """Il riquadro di scrittura non puo' essere piu' stretto del proprio testo.

    Misurato su `captain-america-2-codex-5-4-9p-r2` pagina 4, didascalia
    `p4-beba05ef39515f988`: riquadro 486..740 px (254 di larghezza) contro
    righe OCR 346..848 (502), con il profilo da cancellare che copriva
    307..892, cioe' TUTTO il testo. La traduzione non entrava in meta' spazio
    e il blocco e' rimasto in inglese.

    Qui il balloon e' un rombo: il rettangolo piu' grande che ci sta dentro e'
    molto piu' piccolo dell'ingombro delle righe, che pero' stanno comodamente
    dentro la forma. E' la stessa geometria, ridotta all'osso.
    """

    RIGHE = [[110, 195, 160, 215], [240, 195, 290, 215], [190, 100, 210, 120]]
    BOX = [110, 100, 290, 215]        # ingombro del gruppo OCR

    def tavola(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        y, x = np.ogrid[:400, :400]
        rombo = (np.abs(x - 200) / 110.0 + np.abs(y - 200) / 110.0) <= 1.0
        img[rombo] = 255
        return img

    def riquadro(self):
        from comics_translator.disegno.image_overlay import prepara_area

        return prepara_area(self.tavola(), list(self.BOX), SFONDO,
                            limits=None, vietati=(), righe=self.RIGHE)

    def test_il_riquadro_copre_tutte_le_righe_del_blocco(self):
        box, profili = self.riquadro()
        self.assertTrue(profili, "il balloon esiste e viene riempito")
        ingombro = [min(r[0] for r in self.RIGHE), min(r[1] for r in self.RIGHE),
                    max(r[2] for r in self.RIGHE), max(r[3] for r in self.RIGHE)]
        self.assertLessEqual(box[0], ingombro[0])
        self.assertLessEqual(box[1], ingombro[1])
        self.assertGreaterEqual(box[2], ingombro[2],
                                "il riquadro e' piu' stretto del testo che sostituisce")
        self.assertGreaterEqual(box[3], ingombro[3])

    def test_il_riquadro_resta_dentro_cio_che_viene_cancellato(self):
        """Allargare non e' una scusa per scrivere dove non si cancella."""
        box, profili = self.riquadro()
        xs = [q[0] for pr in profili for q in pr]
        ys = [q[1] for pr in profili for q in pr]
        self.assertGreaterEqual(box[0], min(xs))
        self.assertGreaterEqual(box[1], min(ys))
        self.assertLessEqual(box[2], max(xs))
        self.assertLessEqual(box[3], max(ys))




class MascheraDegliSbloccatiTest(unittest.TestCase):
    """Chi sblocca un blocco si prende anche le righe che lo proteggevano.

    `source_overlap` elenca le righe che stanno nell'area del blocco senza
    appartenergli: sono la ragione della protezione, e restavano visibili
    accanto all'italiano quando l'operatore sbloccava il blocco. Misurato su
    carnage-3 il 7 settembre 2026: "ME TO FIND" sopra "MI HAI ASSUNTO PER
    TROVARE", "YOU LEAVE THEM" dietro "PERCHÉ LASCIARLI".
    """

    def _pagina(self, render):
        return {
            "numero": 1,
            "sorgente": {"righe": [
                {"id": "r-vicina", "box_px": [10.0, 90.0, 60.0, 100.0]},
                {"id": "r-lontana", "box_px": [500.0, 500.0, 540.0, 510.0]},
            ]},
            "blocchi": [{"id": "b1", "_render": render}],
        }

    def _render(self, **extra):
        base = {"tipo": "immagine", "box_px": [10.0, 100.0, 200.0, 180.0],
                "source_overlap": ["r-vicina"]}
        base.update(extra)
        return base

    def test_un_blocco_sbloccato_copre_anche_la_riga_sovrapposta(self):
        render = self._render(sbloccato_a_mano="area verificata")
        profili = plan.profili_di_copertura(render, self._pagina(render))

        # niente fill_poly: entra il riquadro del blocco piu' la riga
        self.assertEqual(len(profili), 2)
        self.assertIn([[10.0, 90.0], [60.0, 90.0], [60.0, 100.0], [10.0, 100.0]], profili)

    def test_senza_sblocco_la_maschera_non_cambia(self):
        """Li' la protezione difende ancora inchiostro che nessuno riscrive."""
        render = self._render()
        self.assertEqual(plan.profili_di_copertura(render, self._pagina(render)), [])

    def test_i_profili_del_balloon_restano_e_si_aggiungono(self):
        contorno = [[10.0, 100.0], [200.0, 100.0], [200.0, 180.0], [10.0, 180.0]]
        render = self._render(sbloccato_a_mano="ok", fill_poly=[contorno])
        profili = plan.profili_di_copertura(render, self._pagina(render))

        self.assertEqual(profili[0], contorno)   # il balloon non si perde
        self.assertEqual(len(profili), 2)

    def test_una_riga_che_non_c_e_piu_non_fa_cadere_il_rendering(self):
        render = self._render(sbloccato_a_mano="ok", source_overlap=["r-sparita"])
        self.assertEqual(plan.profili_di_copertura(render, self._pagina(render)), [])


class BalloonApertoSullaCorniceTest(unittest.TestCase):
    """Un balloon aperto sulla cornice bianca della vignetta non la cancella.

    Su Absolute Superman #1 (superman-1, prova A3) i balloon poggiano sul bordo
    della vignetta e il bianco dei due e' lo stesso: la macchia di sfondo passava
    dal balloon alla cornice, `_profili_sicuri` scartava il profilo e al suo posto
    si copriva il rettangolo cresciuto. Risultato sulla tavola: toppe bianche sui
    bordi delle vignette e sull'arte a p4, p5 e p16-p18, cioe' quasi ogni pagina
    ridisegnata a mano nel `.finale`. Il ritaglio all'intorno delle righe lo
    impedisce, e nessun muro puo' farlo: di la' non c'e' un altro blocco.
    """

    RIGHE = [[70, 60, 230, 92]]

    def tavola(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[20:290, 20:280] = (40, 60, 90)   # arte dentro la vignetta
        img[20:290, 20:32] = 255             # cornice bianca a sinistra
        img[20:32, 20:280] = 255             # cornice bianca in alto
        img[45:110, 32:250] = 255            # balloon bianco, aperto sulla cornice
        img[60:92, 70:230] = 0               # lettering
        return img

    def test_il_riempimento_non_arriva_alla_cornice(self):
        from comics_translator.disegno.image_overlay import prepara_area

        _, profili = prepara_area(self.tavola(), [60, 50, 240, 100], (255, 255, 255),
                                  righe=self.RIGHE)
        self.assertTrue(profili, "il balloon esiste: un profilo va prodotto")
        xs = [p[0] for pr in profili for p in pr]
        ys = [p[1] for pr in profili for p in pr]
        self.assertGreater(min(xs), 32, "il riempimento ha invaso la cornice sinistra")
        self.assertGreater(min(ys), 32, "il riempimento ha invaso la cornice in alto")

    def test_il_testo_originale_resta_coperto_per_intero(self):
        from comics_translator.disegno.image_overlay import prepara_area

        _, profili = prepara_area(self.tavola(), [60, 50, 240, 100], (255, 255, 255),
                                  righe=self.RIGHE)
        xs = [p[0] for pr in profili for p in pr]
        ys = [p[1] for pr in profili for p in pr]
        self.assertLessEqual(min(xs), self.RIGHE[0][0])
        self.assertLessEqual(min(ys), self.RIGHE[0][1])
        self.assertGreaterEqual(max(xs), self.RIGHE[0][2])
        self.assertGreaterEqual(max(ys), self.RIGHE[0][3])


if __name__ == "__main__":
    unittest.main()
