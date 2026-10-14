"""Il manifesto dei lotti: cosa, in questa cartella, risulta gia' fatto.

Traduzione e QA visuale spendono entrambi soldi veri, e li spendono a lotti. Un
lotto concluso che vive solo in memoria e' una spesa che il primo crash
cancella: se la run muore al lotto 5 di 7, anche i primi quattro si ripagano.
Qui ogni lotto lascia una traccia sul disco appena e' concluso, e una ripresa
puo' saltarlo senza richiamare il modello.

Le due proprieta' da tenere insieme sono opposte:

- non ripetere un lotto gia' validato;
- non spacciare per fatto un lotto il cui esito si e' perso. Una richiesta
  partita e mai tornata e' un `tentativo`, non un risultato.

Il file e' uno per cartella di lavoro: la revisione ha la sua, il QA visuale ne
ha un'altra (`qa-visuale/`), e i due manifesti non si incrociano mai.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from comics_translator import pipeline_status as status

NOME_FILE = "lotti.json"

# Uno stato che distingue il tentativo dal risultato: una richiesta partita e
# mai tornata NON e' un lotto fatto, e ripeterla e' l'unica cosa onesta.
TENTATIVO = "tentativo"
VALIDO = "valido"
FALLITO = "fallito"

# Che lavoro racconta il manifesto: serve solo a chi lo legge a mano, ma un
# file che non dice di quale fase parla e' un file che prima o poi si confonde.
TIPO_REVISIONE = "lotti_revisione"
TIPO_QA_VISUALE = "lotti_qa_visuale"


def leggi(workdir: Path) -> dict:
    """Cosa risulta gia' fatto in questa cartella. Illeggibile = niente."""
    percorso = Path(workdir) / NOME_FILE
    try:
        dati = json.loads(percorso.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    lotti = dati.get("lotti") if isinstance(dati, dict) else None
    return lotti if isinstance(lotti, dict) else {}


def registra(workdir: Path, etichetta: str, tipo: str, **campi) -> None:
    """Aggiorna il manifesto dei lotti, in modo atomico e senza perdere il resto."""
    percorso = Path(workdir) / NOME_FILE
    lotti = leggi(workdir)
    voce = dict(lotti.get(etichetta) or {})
    voce.update(campi)
    # Un lotto valido non ha un errore: l'`errore` del tentativo precedente
    # resterebbe accanto allo stato nuovo, e chi legge il manifesto a mano si
    # troverebbe una voce che dice `valido` e insieme perche' e' fallita.
    # Visto per davvero su cap-3, lotti 02 e 11, ripresi al secondo giro.
    if campi.get("stato") == VALIDO:
        voce.pop("errore", None)
    voce["aggiornato"] = status.utc_now()
    lotti[etichetta] = voce
    status.scrivi_atomico(percorso, json.dumps(
        {"manifesto": tipo, "lotti": lotti}, ensure_ascii=False, indent=2))


def impronta(prompt: str, piano_lotto: Path) -> str:
    """Cosa e' stato chiesto per questo lotto, in una riga.

    Lega il risultato salvato al suo input: piano del lotto e prompt. Se cambia
    anche solo uno, la risposta di prima non risponde piu' alla domanda di
    adesso e va rifatta. Il provider e il modello NON contano: cambiare AI a
    meta' run (tipicamente per credito esaurito su un provider) non deve
    buttare via lotti gia' validati, che rispondevano gia' alla stessa identica
    domanda. Il prompt cita per nome le immagini del lotto, quindi un lotto
    composto da pagine diverse ha comunque un'impronta diversa.
    """
    impasto = hashlib.sha256()
    impasto.update(status.sha256(piano_lotto).encode("ascii"))
    impasto.update(prompt.encode("utf-8"))
    return impasto.hexdigest()


def riusabile(voce, valore_impronta: str, out_path: Path) -> bool:
    """True solo per un risultato VALIDATO, con lo stesso input, ancora sul disco.

    Un lotto in stato `tentativo` e' una richiesta il cui esito si e' perso: non
    si puo' dichiarare fatta. E la risposta deve essere quella registrata, non
    un file con lo stesso nome scritto da un'altra prova.
    """
    if not isinstance(voce, dict) or voce.get("stato") != VALIDO:
        return False
    if voce.get("impronta") != valore_impronta:
        return False
    if not Path(out_path).is_file():
        return False
    return voce.get("risposta_sha256") == status.sha256(out_path)
