"""Motore di traduzione dei fumetti: sorgente, traduzione, disegno, verifica.

Le sottocartelle seguono i tre oggetti separati dell'architettura piu' la
verifica finale:

    sorgente/    sorgente immutabile: inventario OCR, localizzazione, struttura
    traduzione/  decisione semantica: provider AI e cassette di replay
    disegno/     piano derivato, composizione e rendering del PDF tradotto
    verifica/    controlli automatici e correzioni riproducibili

`pipeline_status` resta qui perche' trasversale a tutte le fasi.
"""

from pathlib import Path

__version__ = "0.1.0"

# Cartella del pacchetto: la provenienza delle esecuzioni (RunReport) misura
# il codice da qui. Installato con pip, e' la cartella in site-packages.
RADICE = Path(__file__).resolve().parent
