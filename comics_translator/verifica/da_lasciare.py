"""Cosa resta in inglese per scelta, uguale per tutti gli albi.

Ogni chat che chiudeva un albo FATTO decideva da se' la propria lista KEEP:
loghi, crediti, onomatopee, insegne. Le liste si somigliavano ma non erano
uguali ("LIVE" tradotto in IN DIRETTA su woman-3 e woman-4, lasciato altrove),
e il controllo OCR del PDF finito le contava come inglese rimasto: sulle bozze
di woman-5 erano una parte delle 100 righe segnalate.

Le regole stanno qui, una volta sola. Le usano il controllo dell'inglese
(`inglese_render`), per dichiarare queste righe invece di segnalarle, e il
ciclo di riparazione, per non spenderci il modello. Lo stesso elenco, in
parole, sta nel prompt della traduzione (`ai_review.PROMPT`).

Si traducono sempre, e quindi qui NON compaiono: riassunti, titoli, NEXT,
TO BE CONTINUED, bug TV, luoghi e tempi, rubriche.
"""

from __future__ import annotations

import re

# Crediti, ruoli e marchi: il nome d'arte e il ruolo restano come in copertina.
# Solo ruoli che in un dialogo non compaiono quasi mai, o un ruolo seguito da
# "by": "THE ART OF WAR" e "COVER ME!" sono battute, "ART BY" no.
CREDITI = re.compile(
    r"(\b(written|story|art|cover|design|edited|created|lettered|colou?red|drawn|"
    r"script|plot|layouts?)\s+by\b"
    r"|\b(writer|penciler|penciller|pencils|inker|inks|colou?rist|colou?rs|letterer|"
    r"letters|variant cover|assistant editor|associate editor|senior editor|"
    r"editor in chief|editor-in-chief|production design|logo design|vc's)\b)",
    re.IGNORECASE)
LEGALE = re.compile(
    r"(©|\(c\)|\btm\b|®|all rights reserved|printed in|marvel\.com|dccomics\.com|"
    r"published by|no similarity|issn|isbn|\bupc\b|manufactured)", re.IGNORECASE)
MARCHI = re.compile(
    r"^\W*(marvel|dc|dc comics|absolute|ultimate|all in|nation|finest|image|"
    r"marvel comics|comics code|approved by the comics code authority)\W*$",
    re.IGNORECASE)
VOCALI = re.compile(r"[aeiouyAEIOUY]")
RIPETUTA = re.compile(r"(\w)\1{2,}")


def onomatopea(testo: str) -> bool:
    """Suoni: nessuna vocale (KRKK, THWP), o una lettera ripetuta tre volte
    (AAAGH, SKREEE) in una riga di una o due parole."""
    parole = re.findall(r"[^\W\d_]+", testo or "")
    if not parole or len(parole) > 2:
        return False
    if all(not VOCALI.search(p) for p in parole):
        return True
    return all(RIPETUTA.search(p) for p in parole)


def motivo(testo: str) -> str | None:
    """Perche' questa riga resta in inglese per scelta, o None."""
    t = (testo or "").strip()
    if not t:
        return None
    if LEGALE.search(t):
        return "legale"
    if MARCHI.match(t):
        return "marchio"
    if onomatopea(t):
        return "onomatopea"
    # Un credito e' corto e porta un ruolo: "WRITTEN BY AL EWING", "COLORS".
    if CREDITI.search(t) and len(t.split()) <= 8:
        return "crediti"
    return None
