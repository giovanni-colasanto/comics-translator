"""Revisione del piano affidata a Codex, Claude Code o GLM, che guardano le pagine.

Non usa API a consumo: invoca `codex exec`, la CLI `claude` o la CLI `glm` in
modalita' headless, usando l'abbonamento gia' autenticato sulla macchina. Il
modello riceve i PNG delle pagine e il piano, e ne restituisce una versione rivista.

Il confine e' netto: l'AI scrive un documento, non esegue la pipeline. Puo'
cambiare solo i campi modificabili del piano (`testo`, `salta`, `unisci_con`,
`corpo`, `allinea`); rettangoli, colori e id restano quelli calcolati
dall'analisi, e merge_review() scarta tutto il resto.
"""

from __future__ import annotations

import copy
from collections import Counter
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib

from comics_translator import manifesto_lotti
from comics_translator import pipeline_status as status
from comics_translator.traduzione import cassette
from comics_translator.verifica.qa_corrections import apply_corrections


MAX_EDGE = 1600  # lato lungo dell'anteprima: oltre non aggiunge leggibilita'
AI_DEFAULT = "codex"
MODELLI_DEFAULT = {"codex": "gpt-5.4", "claude": "haiku", "glm": "glm-5.3-flash"}
AI_SUPPORTATE = tuple(MODELLI_DEFAULT)

# Credito esaurito in una run senza terminale (agente, scheduler): invece di
# rinunciare subito si aspetta un segnale sul disco, cosi' il lotto si puo'
# riprendere quando il limite si e' resettato. I file stanno nella cartella
# della run, non del lavoro: due run in parallelo si sbloccano separatamente.
SBLOCCO_RIPRENDI = "riprendi.txt"
SBLOCCO_RINUNCIA = "rinuncia.txt"
ATTESA_CREDITO_S = 6 * 60 * 60  # tetto: appesi per sempre e' peggio che fermarsi
ATTESA_CREDITO_POLL_S = 5.0
# Ma se il messaggio del blocco dice QUANDO il limite si resetta, rinunciare
# prima di quell'ora e' solo uno spreco: il tetto si allunga fino al reset.
# Osservato il 9 settembre 2026 su GLM: reset dichiarato fra 7h34m, tetto fisso
# a 6h, e la pipeline si sarebbe arresa un'ora e mezza troppo presto pur avendo
# l'orario in mano. Il massimo assoluto resta, perche' un messaggio che
# annuncia un reset fra tre giorni non deve tenere un lock aperto per tre giorni.
ATTESA_CREDITO_MAX_S = 24 * 60 * 60
ATTESA_CREDITO_MARGINE_S = 5 * 60

# P7: il manifesto dei lotti vive in comics_translator.manifesto_lotti, perche' da P7.1
# lo usa anche il QA visuale. Qui restano i nomi storici, che erano gia'
# l'interfaccia di questa fase e non c'e' motivo di far cambiare a chi legge.
MANIFESTO_LOTTI = manifesto_lotti.NOME_FILE
LOTTO_TENTATIVO = manifesto_lotti.TENTATIVO
LOTTO_VALIDO = manifesto_lotti.VALIDO
LOTTO_FALLITO = manifesto_lotti.FALLITO

# Il piano puo' arrivare con o senza una bozza automatica: senza motore di
# traduzione (`--engine none`) il campo `automatica` ripete l'originale, e
# spacciarlo per una traduzione manderebbe il modello fuori strada.
BOZZA_PRESENTE = "- `automatica`: una traduzione automatica grezza, fatta senza vedere la pagina;"
BOZZA_ASSENTE = """- `automatica`: ignoralo, ripete l'originale. Qui non c'e' nessuna bozza da
  correggere: la traduzione la fai tu da zero, ed e' l'unica che verra' usata;"""

PROMPT = """Sei un traduttore professionista di fumetti e documenti illustrati.

Nella cartella di lavoro trovi:
- `%(piano)s`: l'elenco dei blocchi di testo delle pagine da tradurre;
- %(immagini)s: le immagini di quelle pagine, a cui i blocchi si riferiscono.

Leggi quelle immagini e il piano, poi scrivi il file `%(out)s`. Leggi ogni
immagine una volta sola: rileggerla non aggiunge niente e costa parecchio.
Non consultare lotti precedenti o altri file: il contesto necessario e' qui.
Per ogni blocco decidi la traduzione definitiva da %(src)s a %(dst)s.

Ogni blocco del piano ha:
- `originale`: il testo letto da un OCR, che sbaglia spesso;
%(bozza)s
- `pos`: dove si trova il blocco nella pagina, per riconoscerlo nell'immagine.
- `salta`: se e' gia' `true`, lascialo `true`. Non decidi tu se quel blocco
  verra' disegnato: lo decide il programma DOPO, verificando l'area sulla
  pagina. Ma la traduzione ti serve lo stesso, vedi sotto;
- `protetto` (solo su alcuni blocchi): perche' il blocco e' arrivato con
  `salta: true` - `sovrapposto` (tocca righe OCR vicine), `scartato_come_disegno`
  (l'analisi l'ha creduto parte del disegno), `piu_aree`, `contenuto_in_altro`,
  `pagina_protetta`. Spesso e' comunque un balloon o una didascalia vera.

Traduci SEMPRE `testo`, anche nei blocchi con `salta: true`, quando sono dialogo
o didascalia: un blocco protetto che resta in inglese perche' nessuno l'ha
tradotto e' il difetto piu' frequente di questo programma. Copia `originale`
in `testo` solo per onomatopee, loghi, insegne disegnate, crediti o rumore OCR.

Per ogni blocco scrivi anche `natura`, guardando l'immagine e non l'OCR:
- `dialogo`: balloon, pensieri, parlato di un personaggio;
- `didascalia`: riquadri narrativi, titoli di capitolo, testo redazionale;
- `grafica`: onomatopee, loghi, insegne o scritte disegnate, crediti, firme;
- `rumore`: l'OCR ha letto lettere dove non c'e' testo (texture, riflessi).

Come lavorare:
- Guarda la vignetta. Chi parla, a chi, con che tono, e cosa e' appena successo:
  e' quello che la traduzione automatica non poteva sapere.
- L'OCR unisce parole ("ANDNOWI" = "and now I"), perde apostrofi e scambia
  lettere. Ricostruisci quello che c'era scritto e traduci quello, non l'errore.
- Conserva nomi di personaggi e marchi: non tradurli perche' somigliano a
  sostantivi comuni. Per luoghi, istituzioni e termini usa invece il nome
  italiano consolidato (oceano Atlantico, Pentagono, Nazioni Unite, supersoldato).
- Registro parlato e naturale, stessa intensita' dell'originale.
- Prima di consegnare rileggi tutto l'italiano: correggi grammatica, ortografia,
  genere del parlante, calchi letterali e parole rimaste per errore in inglese.
  Scrivi in UTF-8: usa accenti italiani veri, non apostrofi al posto degli accenti.
- Conserva esattamente simboli e operatori visibili (per esempio `≠` non e' `=`),
  oltre alla punteggiatura che cambia il senso.
- La fedelta' viene prima di tutto: non sacrificare una sfumatura, una battuta
  o un pezzo di senso per far entrare il testo nel balloon. Ma a parita' di resa
  scegli sempre la piu' corta, e non aggiungere parole che l'originale non ha:
  quello che non entra nel balloon viene rimpicciolito fino a entrarci, e un
  balloon pieno di caratteri minuscoli si legge male quanto una battuta
  tradotta male.

Cosa resta in inglese e cosa no (vale per tutti gli albi, sempre uguale):
- restano in originale, con `natura: grafica`: loghi e testate, crediti e firme
  (autori, disegnatori, editor, copertine variant), avvisi legali, onomatopee e
  suoni disegnati, insegne e scritte dipinte nella scena, testo specchiato o
  illeggibile, codici e numeri di catalogo;
- si traducono sempre, con `natura: didascalia`: riassunti iniziali
  ("PREVIOUSLY..."), titoli di capitolo e di storia, "NEXT:", "TO BE
  CONTINUED", "THE END", le scritte dei notiziari e dei bug TV ("LIVE" ->
  "IN DIRETTA"), luoghi e tempi ("NOW.", "GATEWAY CITY."), lettere e rubriche.

Oltre al testo puoi correggere il lavoro dell'algoritmo, che raggruppa i blocchi
guardando solo i pixel:
- `salta: true` SOLO se quel testo NON va sostituito (onomatopee e scritte che
  fanno parte del disegno: coprirle rovinerebbe la tavola). Mai su un blocco
  `dialogo` o `didascalia`: un dialogo saltato resta in inglese sulla pagina;
- `unisci_con: "<id>"` se due blocchi sono in realta' lo stesso balloon spezzato
  in due e sono visivamente adiacenti; il testo verra' fuso nell'altro blocco.
  Ogni blocco deve contenere solo la propria parte: NON copiare anche il testo
  del vicino nella traduzione, altrimenti la fusione lo duplica.
  Non unire mai colonne, titoli, sottotitoli, indici o paragrafi distinti;
- se un singolo blocco comprende piu' balloon o lobi, NON saltarlo: traduci
  tutto, nell'ordine di lettura, chiudendo la parte di ogni balloon con la sua
  punteggiatura (una frase o piu' per balloon, come nell'originale). Il
  programma divide il testo fra i balloon contando le frasi: non unire in una
  frase sola battute che stanno in balloon diversi;
- `corpo: <numero>` per forzare il corpo del carattere, se serve;
- `allinea: "centro"` o `"sinistra"`.

Se l'OCR ha letto male un frammento, ricostruiscilo guardando l'immagine: il
balloon e' li', e lo leggi meglio dell'OCR. Se una parola resta davvero
illeggibile anche nell'immagine, traduci il resto della battuta senza
inventare quella parola: una battuta tradotta quasi tutta vale piu' di una
battuta lasciata in inglese.

Scrivi `%(out)s` con la STESSA struttura di `piano.json`, ma includendo per ogni
blocco solo i campi: `id`, `testo`, `salta`, `unisci_con`, `corpo`, `allinea`,
`natura`. Includi TUTTI i blocchi del piano, nessuno escluso.
Il campo `testo` deve essere sempre una stringa non vuota, anche con `salta: true`.
Per un blocco lasciato originale, copia `originale` in `testo`: non usare "" o null.
Non copiare `_render`, non inventare blocchi, non cambiare gli `id`.
Non scrivere altri file e non modificare `piano.json`."""


# Cosa il modello dice di un blocco guardando la tavola. E' l'informazione che
# la protezione geometrica non ha: una riga OCR sovrapposta puo' essere il resto
# dello stesso balloon o un'onomatopea disegnata, e solo l'immagine lo dice.
# Serve allo sblocco automatico (`comics_translator.disegno.sblocco_automatico`), che la
# combina con la verifica dell'area: nessuna delle due basta da sola.
NATURE = ("dialogo", "didascalia", "grafica", "rumore")

# Campi che il modello puo' restituire per un blocco. `natura` non e' un campo
# EDITABLE del piano: non cambia cosa si disegna, lo si registra accanto.
CAMPI_RISPOSTA = frozenset({"id", "testo", "salta", "unisci_con", "corpo", "allinea",
                            "natura"})

# Il motivo di una protezione, detto al modello con una parola. Senza, un blocco
# con `salta: true` e' indistinguibile da un'onomatopea gia' riconosciuta, e il
# prompt storico gli diceva di lasciarlo stare: e' cosi' che nasceva il Gap D.
MOTIVI_PROTEZIONE = (
    ("scartato_come_disegno", "scartato_come_disegno"),
    ("page_protected", "pagina_protetta"),
    ("contenuto_in", "contenuto_in_altro"),
    ("compound", "piu_aree"),
    ("source_conflict", "sovrapposto"),
    ("skip_localizzazione", "sovrapposto"),
)

# Nuovi tentativi per un lotto che non ha prodotto un risultato completo
# (timeout, CLI morta, JSON rotto, blocchi mancanti). Un lotto perso costava
# fino a due pagine tradotte a mano nel `.finale`; un secondo giro costa un lotto.
RITENTA_LOTTI = 1


def default_model(ai: str) -> str:
    """Il modello adatto al provider, se l'utente non ne sceglie uno."""
    try:
        return MODELLI_DEFAULT[ai]
    except KeyError:
        raise ValueError("AI non supportata: %s" % ai)


def find_cli(ai: str = AI_DEFAULT) -> str | None:
    """La CLI del provider: prima il PATH, poi le installazioni note su Windows."""
    if ai not in AI_SUPPORTATE:
        raise ValueError("AI non supportata: %s" % ai)
    found = shutil.which(ai)
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WindowsApps",
        Path(os.environ.get("APPDATA", "")) / "npm",
    ]
    names = ("%s.exe" % ai, "%s.cmd" % ai, ai)
    for base in candidates:
        if not base.is_dir():
            continue
        for name in names:
            for hit in base.rglob(name):
                if hit.is_file():
                    return str(hit)
    return None


def _verifica_abbonamento_codex(cli: str) -> None:
    """Accetta solo il login ChatGPT: una API key renderebbe l'uso a consumo."""
    status = subprocess.run(
        [cli, "login", "status"], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    messaggio = "\n".join((status.stdout or "", status.stderr or "")).strip()
    modo = messaggio.lower()
    if status.returncode != 0:
        sys.exit("Codex non autenticato. Esegui 'codex login' e accedi con ChatGPT.")
    if "api key" in modo or "chatgpt" not in modo:
        sys.exit(
            "Codex non sta usando il login ChatGPT dell'abbonamento.\n"
            "Esegui 'codex logout', poi 'codex login' senza --with-api-key.\n"
            "Stato rilevato: %s" % (messaggio or "modalita' sconosciuta")
        )


def _verifica_login_glm(cli: str) -> None:
    """Accetta solo il login OAuth dell'abbonamento GLM Coding, non una API key."""
    esito = subprocess.run(
        [cli, "auth", "status"], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if esito.returncode != 0:
        sys.exit("GLM non autenticato. Lancia 'glm' e usa /login, poi riprova.")
    try:
        stato = json.loads((esito.stdout or "").strip())
    except json.JSONDecodeError:
        sys.exit(
            "Stato di login GLM non leggibile. Verifica a mano con "
            "'glm auth status' che sia attivo il login OAuth dell'abbonamento."
        )
    if not isinstance(stato, dict) or not stato.get("loggedIn"):
        sys.exit(
            "GLM non sta usando il login dell'abbonamento (GLM Coding plan).\n"
            "Esegui 'glm auth logout', poi 'glm' e '/login', come da piano z.ai."
        )


def _verifica_login_claude(cli: str) -> None:
    """Accetta solo il login claude.ai dell'abbonamento, diretto su Anthropic.

    Gira con lo stesso ambiente dei lotti, cosi' dice quale account paghera' la
    run prima che parta: un login con API key sarebbe uso a consumo.
    """
    cartella = os.environ.get(CLAUDE_CONFIG_PIPELINE)
    come_login = (
        "Lancia 'claude' con CLAUDE_CONFIG_DIR=%s e usa /login." % cartella
        if cartella else "Lancia 'claude' e usa /login."
    )
    esito = subprocess.run(
        [cli, "auth", "status"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=_env_per_provider("claude"),
    )
    try:
        stato = json.loads((esito.stdout or "").strip())
    except json.JSONDecodeError:
        stato = None
    if esito.returncode != 0 or not isinstance(stato, dict) or not stato.get("loggedIn"):
        sys.exit("Claude non autenticato. %s" % come_login)
    if stato.get("apiProvider") != "firstParty" or stato.get("authMethod") != "claude.ai":
        sys.exit(
            "Claude non sta usando il login claude.ai dell'abbonamento.\n"
            "%s\nStato rilevato: provider=%s, metodo=%s"
            % (come_login, stato.get("apiProvider"), stato.get("authMethod"))
        )
    print("  Claude: account %s (%s)%s" % (
        stato.get("email") or "?", stato.get("subscriptionType") or "?",
        " da %s" % cartella if cartella else "",
    ))


@dataclass(frozen=True)
class ProviderSession:
    """Provider AI risolto una volta e condiviso dalle diverse revisioni."""

    ai: str
    model: str
    cli: str


def prepare_provider(ai: str = AI_DEFAULT, model: str | None = None) -> ProviderSession:
    """Trova e valida una CLI prima di avviare qualunque tipo di revisione."""
    ai = ai.lower()
    selected_model = model or default_model(ai)
    if cassette.attivo() and not cassette.in_registrazione():
        # Replay: nessuna CLI, nessun controllo di abbonamento, nessun subprocess.
        return ProviderSession(ai=ai, model=selected_model, cli="cassette")
    cli = find_cli(ai)
    if not cli:
        sys.exit(
            "CLI '%s' non trovata. Installala oppure applica un piano gia' "
            "rivisto con --plan-in." % ai
        )
    if ai == "codex":
        _verifica_abbonamento_codex(cli)
    if ai == "glm":
        _verifica_login_glm(cli)
    if ai == "claude":
        _verifica_login_claude(cli)
    if status.ACTIVE_REPORT.get():
        try:
            version = subprocess.run([cli, "--version"], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=10)
            version_text = version.stdout.strip()[:200] if version.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            version_text = None
        status.event("provider", provider=ai, model=selected_model, cli=cli, version=version_text)
    return ProviderSession(ai=ai, model=selected_model, cli=cli)


def run_agent(
    session: ProviderSession,
    prompt: str,
    workdir: Path,
    images: list[Path],
    timeout: int,
    output_path: Path | None = None,
):
    """Esegue un turno AI, aspettando il ripristino dei crediti se necessario."""
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if cassette.attivo() and not cassette.in_registrazione():
        return _riproduci_agent(session, prompt_sha, images, timeout, output_path)
    cmd = _build_command(
        session.ai, session.cli, prompt, session.model, workdir, images,
    )
    while True:
        if output_path and output_path.exists():
            output_path.unlink()
        started = status.clock()
        status.event("agent_started", provider=session.ai, model=session.model,
                     prompt_sha256=prompt_sha,
                     output=str(output_path) if output_path else None)
        try:
            proc = _esegui_cli(
                cmd, workdir, timeout,
                stdin_text=prompt if session.ai in ("codex", "glm") else None,
                ai=session.ai,
            )
        except subprocess.TimeoutExpired:
            status.event("agent_finished", status="timeout", elapsed_s=status.clock() - started)
            if cassette.in_registrazione():
                cassette.registra_timeout(session.ai, session.model, prompt_sha, images)
            return None

        usage = status.timed_call_usage(proc)
        status.event("agent_finished", returncode=proc.returncode,
                     elapsed_s=status.clock() - started, usage=usage)
        blocco = _limite_esaurito(proc)
        if blocco:
            report = status.ACTIVE_REPORT.get()
            wait_started = status.clock()
            if report:
                report.wait_credit(True)
            status.event("waiting_credit", requires_user_key=True)
            retry = _aspetta_sblocco(blocco)
            status.event("credit_wait_ended", elapsed_s=status.clock() - wait_started, retry=retry)
            if report:
                report.wait_credit(False)
            if retry:
                continue
            # Rinuncia con credito esaurito: la cassette registra il fallimento
            # cosi' il replay ripercorre lo stesso lotto invalido.
        if cassette.in_registrazione() and output_path is not None:
            cassette.registra(session.ai, session.model, prompt_sha, images,
                              output_path, proc, usage=usage,
                              elapsed_s=status.clock() - started)
        try:
            _report_outcome(proc, session.ai)
        except Exception as exc:  # pragma: no cover
            print("  (esito della CLI non leggibile: %s)" % exc, file=sys.stderr)
        return proc


def _riproduci_agent(session, prompt_sha, images, timeout, output_path):
    """Riproduce un turno dalla cassette: nessun subprocess, fail-closed."""
    if output_path is None:
        sys.exit("Cassette: chiamata senza output_path non riproducibile.")
    started = status.clock()
    status.event("agent_started", provider=session.ai, model=session.model,
                 prompt_sha256=prompt_sha, cassette="replay", output=str(output_path))
    voce = cassette.riproduci(session.ai, session.model, prompt_sha, images, output_path)
    if voce.get("timeout"):
        status.event("agent_finished", status="timeout", cassette="replay",
                     elapsed_s=status.clock() - started)
        return None
    proc = _Esito(voce.get("returncode") or 0, voce.get("stdout_tail") or "", "")
    status.event("agent_finished", returncode=proc.returncode, cassette="replay",
                 elapsed_s=status.clock() - started, usage=voce.get("usage") or {})
    return proc


def export_pages(doc, pages, workdir: Path, max_edge: int = MAX_EDGE) -> list:
    """Salva le anteprime che il modello dovra' guardare."""
    outdir = workdir / "pagine"
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for pno in pages:
        page = doc[pno]
        longest = max(page.rect.width, page.rect.height)
        dpi = max(36, min(220, int(72.0 * max_edge / max(longest, 1))))
        path = outdir / ("p%02d.png" % (pno + 1))
        page.get_pixmap(dpi=dpi).save(path)
        written.append(path)
    return written


def _build_command(ai, cli, prompt, model, workdir, images):
    """Costruisce una chiamata headless equivalente per le CLI.

    La CLI `glm` e' un fork di Claude Code: parlano lo stesso dialetto di
    opzioni (`-p`, `--output-format stream-json`, `--tools`, ...) ed emettono
    gli stessi eventi. Con una differenza sostanziale: il prompt NON va in
    argv. Il binario di Claude e' un .exe e riceve gli argomenti direttamente,
    ma `glm` e' uno shim .cmd npm: CreateProcess lo esegue via `cmd.exe /c`,
    che spezza la riga di comando sulle newlines e rompe le virgolette — un
    prompt lungo e multilinea arriva troncato o malformato. Il prompt viaggia
    quindi su stdin (come Codex): `-p` senza argomento lo legge da li'.

    La workdir viene risolta prima di finire in argv. Il processo parte gia'
    dentro quella cartella (`cwd` in `_esegui_cli`): se il percorso e'
    relativo, la CLI lo risolve una seconda volta rispetto a se stessa e non
    trova piu' niente. Sintomo osservato il 1 settembre: tutti i lotti falliti
    con "Impossibile trovare il percorso specificato (os error 3)", run
    `partial` e zero campi rivisti, senza che nulla indicasse il percorso.
    """
    workdir = Path(workdir).resolve()
    if ai == "claude":
        return [
            cli, "-p", prompt,
            "--output-format", "stream-json", "--verbose",
            "--model", model,
            "--tools", "Read,Write,Glob",
            "--permission-mode", "acceptEdits",
            "--add-dir", str(workdir),
            "--no-session-persistence",
        ]

    if ai == "glm":
        return [
            cli, "-p",
            "--output-format", "stream-json", "--verbose",
            "--model", model,
            "--tools", "Read,Write,Glob",
            "--permission-mode", "acceptEdits",
            "--add-dir", str(workdir),
            "--no-session-persistence",
        ]

    cmd = [
        cli, "exec",
        "--json",
        "--ephemeral",
        "--approve-for-me",
        "--model", model,
        "--cd", str(workdir),
        "-",
    ]
    for image in images:
        cmd += ["--image", str(image)]
    return cmd


# Nome storico della lettura del manifesto: e' quello che usa chi ispeziona una
# cartella di revisione, dentro e fuori dai test.
leggi_manifesto_lotti = manifesto_lotti.leggi


def review(
    plan: dict,
    doc,
    pages,
    workdir: Path,
    source: str,
    target: str,
    ai: str = AI_DEFAULT,
    model: str | None = None,
    timeout: int = 1800,
    batch: int = 2,
    riprendi: bool = False,
    ritenta: int = RITENTA_LOTTI,
) -> TranslationResult:
    """Restituisce il piano rivisto insieme all'esito verificato dei lotti.

    Il lavoro e' spezzato in lotti di poche pagine: una sola chiamata per tutto
    l'albo si porterebbe dietro ogni immagine gia' letta a ogni turno, e il
    conto cresce molto piu' che linearmente. A lotti il contesto resta piccolo,
    quello che e' fatto e' fatto, e un lotto che va storto non trascina gli altri.
    """
    status.plan_index(plan, [p + 1 for p in pages])
    session = prepare_provider(ai, model)

    workdir.mkdir(parents=True, exist_ok=True)
    from comics_translator.disegno.plan import save_plan  # import locale: evita cicli all'avvio

    save_plan(workdir / "piano.json", plan)
    export_pages(doc, pages, workdir)

    senza_bozza = str(plan.get("motore", "")).lower() in ("", "none")
    pagine = plan.get("pagine", [])
    passo = max(1, batch)
    lotti = [pagine[i:i + passo] for i in range(0, len(pagine), passo)]

    n_blocks = sum(len(p["blocchi"]) for p in pagine)
    print("Revisione con %s (%s): %d pagine, %d blocchi, %d lotti da %d pagine"
          % (session.ai, session.model, len(pagine), n_blocks, len(lotti), passo))

    rivisti = []
    ripresi = 0
    # Solo con `riprendi`: senza, la cartella vecchia non deve MAI passare per
    # lavoro fatto. Riusare per sbaglio la risposta di un'altra prova sarebbe
    # peggio del doppio costo.
    manifesto = manifesto_lotti.leggi(workdir) if riprendi else {}
    if riprendi and not manifesto:
        # Meglio dirlo: una ripresa che non trova niente da riprendere e' una
        # run intera da pagare, e chi l'ha lanciata deve saperlo prima.
        print("  ripresa richiesta ma qui non risulta nessun lotto salvato: "
              "sara' una revisione completa")
    outcome = status.PhaseResult("translation", len(lotti))
    status.start_phase("translation", len(lotti))
    for numero, lotto in enumerate(lotti, start=1):
        etichetta = "lotto%02d" % numero
        piano_lotto = "piano.%s.json" % etichetta
        out_name = "piano.rivisto.%s.json" % etichetta
        out_path = workdir / out_name
        repaired_path = workdir / ("piano.rivisto.%s.riparato.json" % etichetta)

        save_plan(workdir / piano_lotto, {**plan, "pagine": _senza_render(lotto)}, projection=True)
        nomi_immagini = ["pagine/p%02d.png" % p["numero"] for p in lotto]
        immagini = ", ".join("`%s`" % nome for nome in nomi_immagini)
        prompt = PROMPT % {
            "src": source, "dst": target, "out": out_name, "piano": piano_lotto,
            "immagini": immagini,
            "bozza": BOZZA_ASSENTE if senza_bozza else BOZZA_PRESENTE,
        }

        print("\n  [%d/%d] pagine %s  (%d blocchi)"
              % (numero, len(lotti), ", ".join(str(p["numero"]) for p in lotto),
                 sum(len(p["blocchi"]) for p in lotto)))

        impronta = manifesto_lotti.impronta(prompt, workdir / piano_lotto)
        riuso = manifesto_lotti.riusabile(manifesto.get(etichetta), impronta, out_path)
        pagine_lotto = [p["numero"] for p in lotto]
        id_lotto = [b["id"] for p in lotto for b in p["blocchi"]]
        # In replay le cassette rispondono sempre allo stesso modo: ritentare
        # rileggerebbe lo stesso fallimento e sfaserebbe la registrazione.
        tentativi = 1 + (0 if riuso or cassette.attivo() else max(0, ritenta))
        migliore = None  # (recuperati, parziale, repairs, errore, persi)
        grezzo_migliore = None
        for tentativo in range(1, tentativi + 1):
            if riuso:
                # La risposta salvata viene comunque rivalidata sotto: il file
                # durevole e' quello che il modello ha scritto, non il verdetto.
                print("  gia' fatto e salvato: riprendo senza richiamare il modello")
                status.event("translation_batch_reused", batch=numero,
                             pages=pagine_lotto, path=str(out_path))
                ripresi += 1
                proc = None
            else:
                if tentativo > 1:
                    print("  nuovo tentativo del lotto %d (%d/%d)"
                          % (numero, tentativo, tentativi))
                    status.event("translation_batch_retry", batch=numero,
                                 attempt=tentativo, pages=pagine_lotto)
                for previous in (out_path, repaired_path):
                    if previous.exists():
                        previous.unlink()
                manifesto_lotti.registra(
                    workdir, etichetta, manifesto_lotti.TIPO_REVISIONE,
                    stato=LOTTO_TENTATIVO, numero=numero, impronta=impronta,
                    risposta=out_name, pagine=pagine_lotto)
                proc = run_agent(
                    session, prompt, workdir,
                    [workdir / nome for nome in nomi_immagini], timeout,
                    output_path=out_path,
                )
            esito = _valuta_lotto(lotto, out_path, "" if riuso else agent_failure(proc))
            if migliore is None or esito[0] > migliore[0]:
                migliore = esito
                if out_path.is_file():
                    grezzo_migliore = out_path.read_bytes()
            if esito[3] is None:
                break  # lotto valido per intero: niente da ritentare
        recuperati, parziale, repairs, errore, persi = migliore
        if recuperati and grezzo_migliore is not None and (
                not out_path.is_file() or out_path.read_bytes() != grezzo_migliore):
            # Il tentativo migliore non e' l'ultimo: sul disco va la sua risposta,
            # perche' e' a quella che il manifesto lega il risultato.
            status.scrivi_atomico(out_path, grezzo_migliore.decode("utf-8"))
        if repairs and parziale is not None and (errore is None or recuperati):
            status.scrivi_atomico(
                repaired_path, json.dumps(parziale, ensure_ascii=False, indent=2))
            report = status.ACTIVE_REPORT.get()
            if report:
                report.artifact("translation_raw_%s" % etichetta, out_path)
                report.artifact("translation_repaired_%s" % etichetta,
                                repaired_path, repairs=repairs)
            status.event("translation_batch_repaired", batch=numero, repairs=repairs,
                         raw_path=str(out_path), raw_sha256=status.sha256(out_path),
                         repaired_path=str(repaired_path),
                         repaired_sha256=status.sha256(repaired_path))
            counts = Counter(repair["code"] for repair in repairs)
            print("  Riparazioni del lotto: " + ", ".join(
                "%s=%d" % (code, count) for code, count in sorted(counts.items())
            ))
        if errore is not None and not recuperati:
            problem = status.issue("translation_batch_invalid", errore,
                                   pagine_lotto, id_lotto)
            outcome.issues.append(problem)
            manifesto_lotti.registra(
                workdir, etichetta, manifesto_lotti.TIPO_REVISIONE,
                stato=LOTTO_FALLITO, errore=errore)
            print("  ! lotto %d non validato: %s" % (numero, errore), file=sys.stderr)
        else:
            outcome.repairs.extend(repairs)
            rivisti.extend(parziale["pagine"])
            outcome.pages.extend(pagine_lotto)
            outcome.ids.extend(b["id"] for p in parziale["pagine"] for b in p["blocchi"])
            if errore is None:
                outcome.completed += 1
            else:
                # Recupero per blocco: le battute valide del lotto restano
                # tradotte, quelle perse si dichiarano una per una. Il lotto non
                # conta come completo, quindi l'esito resta onestamente parziale.
                outcome.issues.append(status.issue(
                    "translation_batch_salvaged",
                    "lotto non valido per intero (%s): %d blocchi recuperati, "
                    "%d persi" % (errore, recuperati, len(persi)),
                    pagine_lotto, persi))
                print("  ! lotto %d recuperato per blocco: %d validi, %d persi (%s)"
                      % (numero, recuperati, len(persi), errore), file=sys.stderr)
            # Solo qui il lotto diventa un risultato: validato, con la risposta
            # ancora sul disco e legata al suo input.
            manifesto_lotti.registra(
                workdir, etichetta, manifesto_lotti.TIPO_REVISIONE,
                stato=LOTTO_VALIDO, numero=numero, impronta=impronta,
                risposta=out_name, risposta_sha256=status.sha256(out_path),
                pagine=pagine_lotto,
                riparazioni=[r["code"] for r in repairs],
                **({"recupero_per_blocco": {"errore": errore, "persi": persi}}
                   if errore is not None else {}))
        status.finish_phase(outcome, final=False)
    if ripresi:
        # Una run che ha ripreso non ha rifatto il lavoro: dirlo, perche' il
        # conto dei lotti "completati" da solo lascerebbe credere il contrario.
        print("\n  %d lotti su %d ripresi dal salvataggio, senza chiamare il modello"
              % (ripresi, len(lotti)))
        status.event("translation_resumed", reused=ripresi, batches=len(lotti))
    status.finish_phase(outcome)
    return TranslationResult({"pagine": rivisti}, outcome)


@dataclass(frozen=True)
class TranslationResult:
    plan: dict
    outcome: status.PhaseResult


def _valuta_lotto(lotto, out_path, errore_agente):
    """Il verdetto su una risposta: (recuperati, parziale, repairs, errore, persi).

    `errore` e' None per un lotto valido per intero. Altrimenti si prova il
    recupero per blocco, e `recuperati` dice quante battute si salvano: e' il
    numero con cui si sceglie il migliore fra piu' tentativi.
    """
    totale = sum(len(p["blocchi"]) for p in lotto)
    tutti = [b["id"] for p in lotto for b in p["blocchi"]]
    if errore_agente:
        # Una CLI uscita male non firma il file che lascia, anche se e' un JSON
        # valido (contratto P0): niente recupero, al piu' un nuovo tentativo.
        return 0, None, [], errore_agente, tutti
    grezzo = _leggi_rivisto(out_path)
    try:
        parziale, repairs = validate_and_repair_review_batch(lotto, grezzo)
        return totale, parziale, repairs, None, []
    except (ValueError, OSError) as exc:
        errore = str(exc)
    parziale, repairs, persi = recupera_blocchi_validi(lotto, grezzo)
    recuperati = sum(len(p["blocchi"]) for p in parziale["pagine"])
    return recuperati, parziale, repairs, errore, persi


def recupera_blocchi_validi(expected, reviewed):
    """Le battute valide di una risposta che non supera la validazione intera.

    Il fail-closed sul lotto intero buttava via tutto per un difetto locale: un
    blocco dimenticato (imperial-exiles, lotto 13: 17 blocchi buoni su 18), un
    campo estraneo, un testo vuoto. Le traduzioni finivano `unreviewed` e
    tornavano in inglese: 158 blocchi riscritti a mano sui 14 albi FATTO, 170
    traduzioni gia' sul disco e mai arrivate alla pagina.

    Il contratto non si allenta sui singoli blocchi: un blocco si recupera solo
    se, preso da solo, rispetta tutto quello che la validazione intera chiede
    (testo non vuoto senza escape, tipi giusti, nessuna area riattivata). Le
    fusioni cadono tutte: dipendono da un vicino che puo' essere perso.
    Restituisce (piano parziale, riparazioni, id persi).
    """
    before = status.plan_index({"pagine": expected})
    risposte = {}
    for page in (reviewed or {}).get("pagine", []) or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocchi", []) or []:
            if isinstance(block, dict) and isinstance(block.get("id"), str):
                # Un id ripetuto nella risposta e' ambiguo: non si sceglie.
                risposte[block["id"]] = None if block["id"] in risposte else block
    from comics_translator.verifica.qa_corrections import _validate_value

    pagine, repairs, persi = [], [], []
    for number, blocks in before.items():
        tenuti = []
        for bid, block in blocks.items():
            revised = risposte.get(bid)
            voce = _voce_recuperabile(block, revised, _validate_value)
            if voce is None:
                persi.append(bid)
                continue
            if block.get("salta") is True and voce["salta"] is False:
                voce["salta"] = True
                repairs.append(status.issue(
                    "protected_reactivation_dropped",
                    "area protetta riaperta dal modello e richiusa; "
                    "traduzione del blocco conservata", [number], [bid]))
            tenuti.append(voce)
        if tenuti:
            pagine.append({"numero": number, "blocchi": tenuti})
    if not pagine:
        # Niente da salvare: nessuna riparazione da dichiarare, e' un lotto perso.
        return {"pagine": []}, [], persi
    # Stessa prova finale della validazione intera, sui soli blocchi tenuti.
    # Se rifiuta blocchi precisi (una riattivazione che la protezione non
    # consente), cadono quelli e si riprova: tutto-o-niente qui rifaceva, un
    # livello piu' in basso, lo stesso errore del lotto intero.
    while True:
        sets = {v["id"]: {k: x for k, x in v.items() if k not in ("id", "natura")}
                for p in pagine for v in p["blocchi"]}
        try:
            apply_corrections({"pagine": expected}, {"set": sets}, decisioni=False)
            break
        except ValueError as exc:
            colpevoli = {bid for bid in sets if bid in str(exc)}
            if not colpevoli:
                return {"pagine": []}, [], [b for bs in before.values() for b in bs]
            persi.extend(sorted(colpevoli))
            pagine = [{"numero": p["numero"],
                       "blocchi": [v for v in p["blocchi"] if v["id"] not in colpevoli]}
                      for p in pagine]
            pagine = [p for p in pagine if p["blocchi"]]
            if not pagine:
                return {"pagine": []}, [], persi
    parziale = {"pagine": pagine}
    if persi or repairs:
        repairs.append(status.issue(
            "batch_salvaged_per_block",
            "risposta non valida per intero: tenuti i blocchi validi presi uno per uno",
            sorted(before), persi))
    return parziale, repairs, persi


def _voce_recuperabile(block, revised, valida):
    if not isinstance(revised, dict):
        return None
    testo = revised.get("testo")
    if not isinstance(testo, str) or not testo.strip():
        return None
    if any(escape in testo for escape in ("\\n", "\\r")):
        return None
    salta = revised.get("salta", block.get("salta", False))
    if not isinstance(salta, bool):
        return None
    if revised.get("unisci_con") is not None and not isinstance(revised["unisci_con"], str):
        return None  # un tipo sbagliato e' una violazione di contratto, non una fusione
    voce = {"id": block["id"], "testo": testo.strip(), "salta": salta, "unisci_con": None}
    for campo in ("corpo", "allinea"):
        if campo in revised:
            try:
                valida(campo, revised[campo])
            except ValueError:
                return None
            voce[campo] = revised[campo]
    if revised.get("natura") in NATURE:
        voce["natura"] = revised["natura"]
    return voce


def agent_failure(proc):
    if proc is None:
        return "timeout della CLI"
    if proc.returncode != 0:
        return "CLI fallita (exit %s)" % proc.returncode
    terminal = None
    for line in (getattr(proc, "stdout", "") or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") in (
            "turn.completed", "turn.failed", "result",
        ):
            terminal = event
    if terminal and (terminal.get("type") == "turn.failed" or terminal.get("is_error")):
        return "errore strutturato della CLI anche con exit 0"
    return ""


def validate_and_repair_review_batch(expected, reviewed):
    """Richiude le aree riattivate, ripara testi vuoti protetti e scarta
    fusioni impossibili, poi valida.

    La risposta grezza resta immutata. Nessun testo viene inventato e nessuna
    area resta riattivata; tipi errati e altri errori invalidano ancora il lotto.
    """
    candidate = copy.deepcopy(reviewed)
    before = status.plan_index({"pagine": expected})
    after = status.plan_index(candidate, before)
    repairs = []
    for number, blocks in before.items():
        if set(blocks) != set(after[number]):
            raise ValueError("p%d: blocchi mancanti o estranei" % number)
    repairs.extend(_richiude_aree_riattivate(before, after))
    for number, blocks in before.items():
        for bid, block in blocks.items():
            revised = after[number][bid]
            text, original = revised.get("testo"), block.get("originale")
            if (block.get("salta") is True and revised.get("salta") is True
                    and isinstance(text, str) and not text.strip()
                    and isinstance(original, str) and original.strip()):
                revised["testo"] = original
                repairs.append(status.issue(
                    "protected_empty_text_restored",
                    "testo vuoto ripristinato dall'originale; blocco ancora protetto",
                    [number], [bid],
                ))
    repairs.extend(_scarta_fusioni_impossibili(expected, candidate, after))
    for number, blocks in after.items():
        for bid, revised in blocks.items():
            # Una natura sconosciuta non e' una traduzione sbagliata: la si
            # toglie, e il blocco resta semplicemente non classificato.
            if "natura" in revised and revised["natura"] not in NATURE:
                revised.pop("natura")
                repairs.append(status.issue(
                    "unknown_nature_dropped", "natura non riconosciuta, ignorata",
                    [number], [bid]))
    validate_review_batch(expected, candidate)
    return candidate, repairs


def _richiude_aree_riattivate(before, after):
    """Richiude le sole aree protette che il modello ha riaperto.

    Riaprire un'area protetta e' una richiesta, non una traduzione sbagliata:
    e' lo stesso caso della fusione impossibile, e va trattato allo stesso
    modo. Invalidare il lotto intero per un blocco butta via le battute buone
    di tutte le altre pagine del lotto - misurato su Captain America #2, dove
    due lotti su otto sono caduti cosi' lasciando quattro pagine su sedici in
    inglese e la copertura al 72.7%.

    Il blocco torna protetto esattamente com'era; la traduzione che il modello
    gli ha dato resta scritta nel piano. Non viene disegnata, perche' il blocco
    e' saltato, ma e' il punto di partenza di un eventuale sblocco motivato -
    cancellarla butterebbe via lavoro senza proteggere niente di piu'.
    """
    riparazioni, richiusi = [], set()
    for number, blocks in before.items():
        for bid, block in blocks.items():
            revised = after.get(number, {}).get(bid)
            if not isinstance(revised, dict):
                continue
            # Solo il booleano esatto: un tipo sbagliato resta una violazione
            # di contratto e deve continuare a invalidare il lotto.
            if block.get("salta") is True and revised.get("salta") is False:
                revised["salta"] = True
                richiusi.add(bid)
                riparazioni.append(status.issue(
                    "protected_reactivation_dropped",
                    "area protetta riaperta dal modello e richiusa; "
                    "traduzione del blocco conservata",
                    [number], [bid],
                ))
    if not richiusi:
        return riparazioni

    # Una fusione che tocca un blocco appena richiuso e' diventata impossibile
    # per effetto di questa riparazione, non per una scelta incoerente del
    # modello: farla cadere e' la conseguenza, non un'assoluzione. La fusione
    # con un blocco che il modello stesso aveva dichiarato saltato resta una
    # violazione di contratto e invalida ancora tutto il lotto.
    for number, blocks in after.items():
        for bid, revised in blocks.items():
            target = revised.get("unisci_con")
            if not isinstance(target, str) or not target:
                continue
            if bid in richiusi or target in richiusi:
                revised["unisci_con"] = None
                riparazioni.append(status.issue(
                    "merge_on_reprotected_dropped",
                    "fusione che toccava un'area richiusa, rimossa; "
                    "traduzioni dei blocchi conservate",
                    [number], [bid],
                ))
    return riparazioni


def _scarta_fusioni_impossibili(expected, candidate, after):
    """Toglie le sole fusioni che il programma rifiuterebbe, e lo dichiara.

    Una fusione impossibile e' una richiesta, non una traduzione sbagliata:
    farla cadere non inventa niente e non riattiva niente, mentre invalidare
    tutto il lotto butterebbe via le battute buone di due pagine intere.
    """
    from comics_translator.disegno.plan import validate_merges  # import locale: evita cicli all'avvio

    scartate = []
    for number, blocks in after.items():
        for bid, revised in blocks.items():
            target = revised.get("unisci_con")
            # Una richiesta malformata non e' una fusione geometricamente
            # impossibile: rifiutarla prima di usarla come chiave o ripararla.
            if target is not None and not isinstance(target, str):
                raise ValueError("%s: 'unisci_con' deve essere una stringa o null" % bid)
            if not target:
                continue
            trial = {"pagine": copy.deepcopy(expected)}
            by_id = {b["id"]: b for p in trial["pagine"] for b in p["blocchi"]}
            if bid not in by_id:
                continue
            for other, campi in ((bid, revised), (target, after.get(number, {}).get(target))):
                if other in by_id and campi:
                    if "salta" in campi:
                        by_id[other]["salta"] = campi["salta"]
            by_id[bid]["unisci_con"] = target
            if not validate_merges(trial):
                continue
            revised["unisci_con"] = None
            scartate.append(status.issue(
                "invalid_merge_dropped",
                "fusione rifiutata dal programma, traduzione del blocco conservata",
                [number], [bid],
            ))
    return scartate


def validate_review_batch(expected, reviewed):
    before = status.plan_index({"pagine": expected})
    after = status.plan_index(reviewed, before)
    sets = {}
    for number, blocks in before.items():
        if set(blocks) != set(after[number]):
            raise ValueError("p%d: blocchi mancanti o estranei" % number)
        for bid, block in blocks.items():
            revised = after[number][bid]
            if "testo" not in revised or "salta" not in revised:
                raise ValueError("%s: testo/salta mancanti" % bid)
            if set(revised) - CAMPI_RISPOSTA:
                raise ValueError("%s: campi estranei alla revisione" % bid)
            if "natura" in revised and revised["natura"] not in NATURE:
                raise ValueError("%s: natura non valida" % bid)
            if block.get("salta") and not revised["salta"]:
                raise ValueError("%s: riattivazione di area protetta" % bid)
            if isinstance(revised["testo"], str) and any(
                escape in revised["testo"] for escape in ("\\n", "\\r")
            ):
                raise ValueError("%s: escape letterale" % bid)
            sets[bid] = {k: v for k, v in revised.items() if k not in ("id", "natura")}
    # Il lotto rimanda indietro tutti i blocchi: un `salta` invariato e'
    # una conferma, non una decisione dell'operatore.
    corrected, _ = apply_corrections({"pagine": expected}, {"set": sets}, decisioni=False)
    by_id = {b["id"]: b for p in corrected["pagine"] for b in p["blocchi"]}
    for bid, block in by_id.items():
        target = block.get("unisci_con")
        if target and (block.get("salta") or by_id[target].get("salta")):
            raise ValueError("%s: fusione con blocco saltato" % bid)


def apply_review(plan, reviewed):
    """Applica atomicamente solo lotti gia' validati, senza scarti silenziosi.

    La `natura` detta dal modello si registra accanto al blocco: non e' un
    campo del piano che cambia il disegno, ma la prova che lo sblocco
    automatico usera' insieme alla verifica dell'area.
    """
    voci = [block for page in reviewed["pagine"] for block in page["blocchi"]]
    result, changes = apply_corrections(plan, {"set": {
        block["id"]: {k: v for k, v in block.items() if k not in ("id", "natura")}
        for block in voci
    }}, decisioni=False)
    nature = {block["id"]: block["natura"] for block in voci if block.get("natura") in NATURE}
    for page in result["pagine"]:
        for block in page["blocchi"]:
            if block["id"] in nature:
                block["natura"] = nature[block["id"]]
    return result, changes


def motivo_protezione(block):
    """Il motivo di un `salta` automatico, con la parola che vede il modello."""
    render = block.get("_render") or {}
    if not block.get("salta"):
        return None
    for marcatore, parola in MOTIVI_PROTEZIONE:
        if render.get(marcatore):
            return parola
    return None


def _senza_render(pagine):
    """Le stesse pagine, ma senza lo stato tecnico dei blocchi.

    `_render` (rettangoli, colori, scale) e' il 90% abbondante del piano, e al
    modello non serve: puo' cambiare solo i campi in chiaro, e merge_review
    scarta comunque tutto il resto. Dandoglielo lo si costringe a paginare
    centinaia di KB per arrivare a poche righe di testo, pagandole a peso.

    Del `_render` resta una parola sola, `protetto`: perche' il blocco arriva
    saltato. E' quella che permette al modello di tradurlo lo stesso.
    """
    snelle = []
    for pagina in pagine:
        blocchi = []
        for b in pagina.get("blocchi", []):
            voce = {k: v for k, v in b.items()
                    if k not in ("_render", "sorgente", "legacy_id", "natura")}
            motivo = motivo_protezione(b)
            if motivo:
                voce["protetto"] = motivo
            blocchi.append(voce)
        snelle.append({**{k: v for k, v in pagina.items() if k not in ("blocchi", "sorgente", "struttura")},
                       "blocchi": blocchi})
    return snelle


def _leggi_rivisto(out_path: Path) -> dict:
    """Il piano rivisto di un lotto, se il modello l'ha davvero scritto."""
    if not out_path.is_file():
        print("  ! %s non prodotto: quelle pagine restano non riviste" % out_path.name,
              file=sys.stderr)
        return {}
    try:
        with open(out_path, encoding="utf-8") as fh:
            rivisto = json.load(fh)
    except json.JSONDecodeError as exc:
        print("  ! %s illeggibile (%s)" % (out_path.name, exc), file=sys.stderr)
        return {}
    return rivisto if isinstance(rivisto, dict) else {}


# Come si presenta il credito finito: la CLI risponde subito, con un errore che
# parla di limite raggiunto e quasi sempre dell'ora in cui si sblocca.
BLOCCO = re.compile(
    r"out of extra usage|usage limit|limit reached|rate limit|too many requests|"
    r"quota|resets? (at |in )?\d",
    re.IGNORECASE,
)


class _Esito:
    """Quel poco di CompletedProcess che serve al resto del modulo."""

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _sicuro(testo):
    """Vedi `pipeline_status.testo_console`: serve anche all'OCR, non solo qui."""
    return status.testo_console(testo)


def _riga(inizio, testo, simbolo="."):
    minuti, secondi = divmod(int(time.monotonic() - inizio), 60)
    print(_sicuro("    %s [%02d:%02d] %s" % (simbolo, minuti, secondi, testo)), flush=True)


def _mostra(evento, inizio):
    """Traduce un evento della CLI in una riga leggibile, o niente."""
    tipo = evento.get("type")

    # Stream JSON di Claude Code.
    if tipo == "assistant":
        for pezzo in evento.get("message", {}).get("content", []):
            genere = pezzo.get("type")
            if genere == "text":
                testo = " ".join(str(pezzo.get("text", "")).split())
                if testo:
                    _riga(inizio, testo[:160], simbolo=" ")
            elif genere == "tool_use":
                dati = pezzo.get("input") or {}
                bersaglio = dati.get("file_path") or dati.get("pattern") or dati.get("path") or ""
                if bersaglio:
                    bersaglio = Path(str(bersaglio)).name
                _riga(inizio, ("%s %s" % (pezzo.get("name", "?"), bersaglio)).strip())
        return

    # JSONL di `codex exec --json`.
    if tipo in ("item.started", "item.completed"):
        item = evento.get("item") or {}
        genere = item.get("type")
        if genere == "agent_message":
            testo = " ".join(str(item.get("text", "")).split())
            if testo:
                _riga(inizio, testo[:160], simbolo=" ")
        elif genere == "command_execution" and tipo == "item.started":
            comando = " ".join(str(item.get("command", "")).split())
            _riga(inizio, ("comando " + comando[:140]).strip())
        elif genere == "file_change" and tipo == "item.completed":
            cambi = item.get("changes") or []
            nomi = [Path(str(c.get("path", ""))).name for c in cambi if c.get("path")]
            _riga(inizio, "scrive " + (", ".join(nomi) or "file"))
    elif tipo in ("error", "turn.failed"):
        errore = evento.get("message") or evento.get("error") or "errore Codex"
        _riga(inizio, str(errore)[:160], simbolo="!")


def _env_senza_host_claude():
    """Ambiente per il CLI figlio, senza i marcatori di sessione dell'host.

    Se la pipeline gira DENTRO una sessione Claude Code (Claude Code stesso o un
    agente come questo), l'host inietta `CLAUDECODE=1` e una famiglia di
    `CLAUDE_CODE_*` / `CLAUDE_AGENT_*` nell'ambiente. Un provider che e' a sua
    volta Claude Code (claude, glm) li eredita, si crede figlio della sessione
    host e prova ad autenticarsi con l'OAuth dell'host invece che con la propria
    credenziale: contro l'endpoint Z.ai di glm questo produce
    401 "token expired or incorrect" ANCHE con una ZAI_API_KEY validissima
    (la stessa che in un terminale pulito funziona). Diagnosticato il 4
    settembre: glm interattivo dal terminale dell'utente rispondeva, glm
    headless lanciato da dentro l'agente no; togliere le `CLAUDE*` risolveva.

    Si toglie solo la famiglia `CLAUDE*` (marcatori di sessione host): restano
    `ANTHROPIC_*`, `ZAI_API_KEY`, PATH e tutto il resto, che i wrapper dei
    provider gestiscono da soli. codex non usa queste variabili, quindi per lui
    e' un no-op innocuo.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}


# Cartella di configurazione dedicata alla CLI `claude` della pipeline, col suo
# login. Non si chiama CLAUDE_CONFIG_DIR apposta: la famiglia CLAUDE* viene tolta
# dall'ambiente del figlio, e sarebbe comunque quella dell'host.
CLAUDE_CONFIG_PIPELINE = "COMICS_CLAUDE_CONFIG_DIR"


def _env_per_provider(ai):
    """Ambiente del CLI figlio per un provider preciso.

    Per `claude` si tolgono anche tutte le `ANTHROPIC_*`: la CLI deve usare solo
    il login salvato nella propria cartella di configurazione, qualunque sia la
    sessione che lancia la pipeline. Il 10 settembre, su doom-academy-1, la
    sessione di lancio aveva `ANTHROPIC_BASE_URL` di z.ai, un
    `ANTHROPIC_AUTH_TOKEN` e `ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.3-flash`:
    `--model haiku` e' girato come glm per 14 lotti su 15, e la run e' stata
    buttata. Una `ANTHROPIC_API_KEY` farebbe di peggio: uso a consumo.

    Con `COMICS_CLAUDE_CONFIG_DIR` la CLI legge login e impostazioni da quella
    cartella invece che da `~/.claude`: un account diverso da quello dell'app
    desktop, fisso per ogni modo di lancio.

    glm e codex restano come prima: glm usa le `ANTHROPIC_*` a modo suo.
    """
    env = _env_senza_host_claude()
    if ai != "claude":
        return env
    env = {k: v for k, v in env.items() if not k.startswith("ANTHROPIC_")}
    cartella = os.environ.get(CLAUDE_CONFIG_PIPELINE)
    if cartella:
        env["CLAUDE_CONFIG_DIR"] = cartella
    return env


def _modello_estraneo(ai, evento):
    """Il modello annunciato dall'init se il provider `claude` non gira su Claude.

    E' l'unico segnale che non si fida della config: `--model haiku` viene
    risolto dalla CLI, e un alias rimappato (variabili d'ambiente, `env` nei
    settings) si vede solo qui, come `"model":"glm-5.3-flash"`.
    """
    if ai != "claude" or not isinstance(evento, dict):
        return None
    if evento.get("type") != "system" or evento.get("subtype") != "init":
        return None
    modello = str(evento.get("model") or "")
    if not modello or modello.startswith("claude-"):
        return None
    return modello


def _esegui_cli(cmd, workdir, timeout, stdin_text=None, ai=None):
    """Lancia la CLI e ne mostra il lavoro mentre avviene.

    Con --output-format json si resta davanti a un terminale fermo per minuti
    senza sapere se sta lavorando o se e' piantato. In streaming ogni mossa
    arriva subito, e il tempo accanto a ogni riga dice dove se ne va.
    """
    if cassette.attivo() and not cassette.in_registrazione():
        # Cintura di sicurezza: in replay nessun subprocess puo' partire.
        raise AssertionError("CLI vietata in modalita' cassette replay")
    # Assoluta anche qui: log e messaggi d'errore devono dire dove ha girato
    # davvero il processo, non un percorso ambiguo relativo a chi ha lanciato.
    workdir = Path(workdir).resolve()
    proc = subprocess.Popen(
        cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        env=_env_per_provider(ai),
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    if stdin_text is not None:
        proc.stdin.write(stdin_text)
        proc.stdin.close()
    righe = []
    coda = queue.Queue()

    def leggi():
        for riga in proc.stdout:
            coda.put(riga)
        coda.put(None)

    threading.Thread(target=leggi, daemon=True).start()

    inizio = time.monotonic()
    scaduto = False
    while True:
        rimasto = timeout - (time.monotonic() - inizio)
        if rimasto <= 0:
            scaduto = True
            break
        try:
            riga = coda.get(timeout=min(rimasto, 1.0))
        except queue.Empty:
            continue
        if riga is None:
            break
        riga = riga.strip()
        if not riga:
            continue
        righe.append(riga)
        try:
            evento = json.loads(riga)
        except json.JSONDecodeError:
            continue
        _mostra(evento, inizio)
        estraneo = _modello_estraneo(ai, evento)
        if estraneo:
            # Si ferma all'init, prima che il lotto costi o scriva qualcosa.
            proc.kill()
            proc.wait()
            sys.exit(
                "La CLI claude sta girando come '%s', non come un modello Claude.\n"
                "Qualcosa rimappa i modelli (variabili ANTHROPIC_* o 'env' nei "
                "settings della CLI): controlla ~/.claude/settings.json o la "
                "cartella di %s." % (estraneo, CLAUDE_CONFIG_PIPELINE)
            )

    if scaduto:
        proc.kill()
        proc.wait()
        raise subprocess.TimeoutExpired(cmd, timeout)

    proc.wait()
    errori = proc.stderr.read() if proc.stderr else ""
    return _Esito(proc.returncode, "\n".join(righe), errori)


def _esito(proc) -> dict:
    """L'ultima riga di stdout, che con --output-format json e' il resoconto."""
    try:
        result = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}
    return result if isinstance(result, dict) else {}


def _limite_esaurito(proc) -> str:
    """Il messaggio del blocco se e' finito il credito, altrimenti stringa vuota.

    Su un processo riuscito si guardano solo gli eventi di errore strutturati:
    cercare nel testo libero del modello produce falsi positivi quando il
    documento stesso parla di limiti o quote. Su un processo fallito si
    controllano invece stdout e stderr, perche' la CLI puo' fermarsi prima di
    produrre il resoconto JSON finale.
    """
    candidati = []
    if proc.returncode != 0:
        candidati.extend((proc.stdout or "", proc.stderr or ""))
    else:
        for riga in (proc.stdout or "").splitlines():
            try:
                evento = json.loads(riga)
            except json.JSONDecodeError:
                continue
            if evento.get("type") in ("error", "turn.failed"):
                candidati.append(json.dumps(evento, ensure_ascii=False))
            elif evento.get("type") == "result" and evento.get("is_error"):
                candidati.append(str(evento.get("result", "")))
    for testo in candidati:
        testo = testo.strip()
        if testo and BLOCCO.search(testo):
            compatto = " ".join(testo.split())
            messaggio = compatto[:200]
            # L'orario del reset puo' stare oltre il taglio (la CLI di Claude
            # mette il testo "resets 12:50am" in coda al JSON): si riporta,
            # altrimenti l'attesa non sa quando riprovare.
            for schema in (_RESET_EPOCH, _RESET_ORA):
                trovato = schema.search(compatto)
                if trovato and trovato.group(0) not in messaggio:
                    messaggio += " " + trovato.group(0)
            return messaggio
    return ""


def _cartella_sblocco() -> Path | None:
    """La cartella della run in corso: e' li' che si lasciano i file di sblocco."""
    percorso = getattr(status.ACTIVE_REPORT.get(), "path", None)
    return Path(percorso).parent if percorso else None


def _consuma(segnale: Path) -> bool:
    """True se il segnale c'era; in ogni caso non resta, cosi' il blocco dopo
    ne pretende uno nuovo invece di ripartire da solo."""
    try:
        segnale.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:  # c'e' ma non si cancella: il segnale vale lo stesso
        return True


def _tasto_premuto() -> bytes:
    """Il tasto appena premuto, se c'e' una console e qualcuno l'ha premuto.

    Non blocca, ed e' la differenza che conta: `isatty()` dice che un terminale
    c'e', non che qualcuno lo stia guardando. Una run lanciata in background da
    un agente eredita lo stdin di una shell vera, quindi `isatty()` risponde
    "si'" e l'attesa bloccante non finisce mai.
    """
    try:
        import msvcrt  # solo Windows: altrove resta il file, che basta
    except ImportError:
        return b""
    try:
        return msvcrt.getch() if msvcrt.kbhit() else b""
    except OSError:  # processo senza console vera
        return b""


_RESET_DICHIARATO = re.compile(
    r"reset(?:s|ted)?\s*(?:at|on)?\s*[:\s]\s*"
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?",
    re.IGNORECASE,
)


# Il formato di Claude, visto in produzione il 14 settembre su batman:
# "resets 2:30am (Europe/Rome)". Niente data, fuso fra parentesi. Non capirlo
# faceva cadere l'attesa sul tetto fisso di 6 ore: il credito era tornato da
# 1h30 quando la run ha rinunciato, e il lotto 12 e' stato rifatto a mano.
# Serve un'ora vera (con i minuti o con am/pm): "resets in 5" non e' un orario.
_RESET_ORA = re.compile(
    r"resets?\s*(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    r"(?:\s*\(([A-Za-z_]+/[A-Za-z_/+-]+|UTC)\))?",
    re.IGNORECASE,
)


def _reset_solo_ora(messaggio: str, adesso: float) -> float | None:
    """Il prossimo istante con l'ora dichiarata, nel fuso dichiarato se c'e'."""
    from datetime import timedelta

    for trovato in _RESET_ORA.finditer(messaggio):
        ore, minuti, meridiano, fuso = trovato.groups()
        if minuti is None and meridiano is None:
            continue
        ore, minuti = int(ore), int(minuti or 0)
        if meridiano:
            if not 1 <= ore <= 12:
                continue
            ore = ore % 12 + (12 if meridiano.lower() == "pm" else 0)
        if ore > 23 or minuti > 59:
            continue
        zona = None
        if fuso:
            try:
                from zoneinfo import ZoneInfo
                zona = ZoneInfo(fuso)
            except Exception:  # fuso sconosciuto: vale l'ora locale
                zona = None
        ora = datetime.fromtimestamp(adesso, zona)
        reset = ora.replace(hour=ore, minute=minuti, second=0, microsecond=0)
        if reset.timestamp() <= adesso:
            reset += timedelta(days=1)
        return reset.timestamp()
    return None


# Il formato strutturato della CLI di Claude, visto nella prova A2 (16
# settembre, spidey-63-9): {"type":"rate_limit_event","rate_limit_info":
# {"status":"rejected","resetsAt":1789599000,...}}. Non leggerlo ha fatto
# aspettare il tetto di 6 ore invece di 1h35, e un lotto di QA e' andato perso.
_RESET_EPOCH = re.compile(r'"?resets_?at"?\s*:\s*"?(\d{10})(?:\.\d+)?', re.IGNORECASE)


def reset_dichiarato(messaggio: str, adesso: float | None = None) -> float | None:
    """L'istante (epoch) in cui il messaggio dice che il limite si resetta, o None."""
    adesso = time.time() if adesso is None else adesso
    epoch = _RESET_EPOCH.search(messaggio or "")
    if epoch:
        return float(epoch.group(1))
    trovato = _RESET_DICHIARATO.search(messaggio or "")
    if trovato:
        anno, mese, giorno, ore, minuti = (int(trovato.group(i)) for i in range(1, 6))
        secondi = int(trovato.group(6) or 0)
        try:
            return datetime(anno, mese, giorno, ore, minuti, secondi).timestamp()
        except ValueError:
            return None
    return _reset_solo_ora(messaggio or "", adesso)


def attesa_dal_messaggio(messaggio: str, adesso: float | None = None) -> float:
    """Quanto aspettare, letto dall'orario di reset che il blocco dichiara.

    Le CLI dicono spesso quando il limite tornera' disponibile
    ("Your limit will reset at 2026-09-09 16:23:17"). Quell'ora e' l'unica
    informazione affidabile che abbiamo: rinunciare prima significa buttare via
    una run che sarebbe ripartita da sola poco dopo.

    Si torna il tetto fisso quando non c'e' nessun orario, quando l'orario e'
    gia' passato (blocco vecchio, messaggio riciclato) o quando aspettare
    costerebbe meno del tetto. L'orario si legge come ora locale, perche' e'
    quello che la CLI mostra e che l'utente vede sul proprio pannello.
    """
    adesso = time.time() if adesso is None else adesso
    reset = reset_dichiarato(messaggio, adesso)
    if reset is None:
        return float(ATTESA_CREDITO_S)
    manca = reset - adesso + ATTESA_CREDITO_MARGINE_S
    if manca <= ATTESA_CREDITO_S:
        return float(ATTESA_CREDITO_S)
    return float(min(manca, ATTESA_CREDITO_MAX_S))


def _aspetta_via_libera(cartella: Path | None, attesa_s: float | None = None,
                        riprova_alle: float | None = None) -> bool:
    """Aspetta un tasto O un file, quello che arriva prima. True per riprovare.

    Due canali insieme invece di sceglierne uno: chi ha il terminale davanti
    preme un tasto, chi ha lanciato la run da un agente lascia un file, e il
    programma non deve indovinare in quale dei due mondi si trova - indovinare
    e' esattamente quello che gli e' riuscito male.

    Il terzo canale e' l'orologio: se il messaggio del blocco dichiarava un
    reset FUTURO (`riprova_alle`), passato quell'istante si riprova da soli. Su
    batman il credito era tornato alle 2:30 e la run ha aspettato un segnale
    che nessuno poteva dare fino al tetto di 6 ore, poi ha rinunciato. Se il
    limite non e' davvero tornato, il tentativo riceve un nuovo blocco con un
    nuovo orario: nessun giro a vuoto.
    """
    riprendi = cartella / SBLOCCO_RIPRENDI if cartella else None
    rinuncia = cartella / SBLOCCO_RINUNCIA if cartella else None
    attesa_s = float(ATTESA_CREDITO_S if attesa_s is None else attesa_s)
    minuti = int(attesa_s // 60)
    print("  Quando il limite si e' resettato premi un tasto qualsiasi per",
          file=sys.stderr)
    print("  riprovare  (ESC per rinunciare e tenere la traduzione automatica).",
          file=sys.stderr)
    if riprendi is not None:
        print("  Senza nessuno al terminale valgono anche due file:", file=sys.stderr)
        print("  per riprovare:   %s" % riprendi, file=sys.stderr)
        print("  per rinunciare:  %s" % rinuncia, file=sys.stderr)
    if riprova_alle is not None:
        print("  Il limite dichiara il reset alle %s: a quell'ora riprovo da solo."
              % datetime.fromtimestamp(riprova_alle).strftime("%H:%M"), file=sys.stderr)
    print("  Senza segnale rinuncio da solo dopo %d minuti." % minuti, file=sys.stderr)
    print("!" * 70 + "\n", file=sys.stderr)
    status.event("credit_wait_signal",
                 riprendi=str(riprendi) if riprendi else None,
                 rinuncia=str(rinuncia) if rinuncia else None,
                 timeout_s=attesa_s, riprova_alle=riprova_alle)

    scadenza = status.clock() + attesa_s
    while status.clock() < scadenza:
        if riprova_alle is not None and time.time() >= riprova_alle:
            print("  Orario di reset dichiarato passato: riprovo la revisione...")
            return True
        if riprendi is not None and _consuma(riprendi):
            print("  Segnale di ripresa: riprovo la revisione...")
            return True
        if rinuncia is not None and _consuma(rinuncia):
            print("  Segnale di rinuncia: tengo la traduzione automatica.",
                  file=sys.stderr)
            return False
        tasto = _tasto_premuto()
        if tasto:
            if tasto in (b"\x1b", b"\x03"):
                print("  Rinuncio alla revisione.", file=sys.stderr)
                return False
            print("  Riprovo la revisione...")
            return True
        time.sleep(ATTESA_CREDITO_POLL_S)
    print("  Nessun segnale entro il tempo massimo: rinuncio.", file=sys.stderr)
    return False


def _aspetta_sblocco(messaggio: str) -> bool:
    """Segnala il blocco e aspetta il via libera. True se si deve riprovare.

    Solo l'utente sa quando il credito e' tornato, quindi si aspetta lui: un
    tasto se e' al terminale, un file nella cartella della run se la run gira
    dentro un agente o uno scheduler. Non si sceglie fra i due canali - si
    ascoltano entrambi - perche' distinguere i due casi da dentro il processo
    non si e' rivelato affidabile. Il tetto di tempo vale in ogni caso: restare
    appesi per sempre e' peggio che fermarsi.
    """
    print("\n" + "!" * 70, file=sys.stderr)
    print("  CREDITO ESAURITO: la revisione non e' partita.", file=sys.stderr)
    print("  %s" % messaggio, file=sys.stderr)
    print("  Il lavoro fatto finora e' salvo, niente e' da rifare.", file=sys.stderr)
    # Se il messaggio dichiara l'ora del reset, si aspetta almeno fino a li':
    # il tetto fisso da solo faceva rinunciare prima che il credito tornasse.
    adesso = time.time()
    reset = reset_dichiarato(messaggio, adesso)
    # Solo un reset futuro vale come sveglia: un orario gia' passato (messaggio
    # riciclato) farebbe riprovare subito, in un giro stretto contro il limite.
    riprova_alle = reset + ATTESA_CREDITO_MARGINE_S if reset and reset > adesso else None
    return _aspetta_via_libera(_cartella_sblocco(), attesa_dal_messaggio(messaggio, adesso),
                               riprova_alle=riprova_alle)


def _report_outcome(proc, ai) -> None:
    """Traduce l'esito della CLI in un messaggio comprensibile."""
    uscita = proc.stdout or ""
    errore = (proc.stderr or "").strip()
    if proc.returncode != 0:
        righe = errore.splitlines() or uscita.strip().splitlines()
        utili = [
            riga.strip() for riga in righe
            if riga.strip()
            and not riga.strip().lower().startswith("usage:")
            and not riga.strip().lower().startswith("for more information")
        ]
        dettaglio = (": " + (utili[0] if utili else righe[-1])[:300]) if righe else ""
        print("  ! la CLI %s e' uscita con codice %d%s"
              % (ai, proc.returncode, dettaglio), file=sys.stderr)
        testo = (uscita + "\n" + errore).lower()
        if "authenticate" in testo or "not logged" in testo or "401" in testo:
            istruzione = "codex login" if ai == "codex" else "claude, poi /login"
            print("    Autentica la CLI con l'abbonamento: %s." % istruzione,
                  file=sys.stderr)
        return

    if ai == "codex":
        eventi = []
        for riga in uscita.splitlines():
            try:
                eventi.append(json.loads(riga))
            except json.JSONDecodeError:
                continue
        completato = next(
            (e for e in reversed(eventi) if e.get("type") == "turn.completed"), {}
        )
        usage = completato.get("usage") or {}
        detail = []
        ingresso = usage.get("input_tokens")
        uscita_token = usage.get("output_tokens")
        if ingresso is not None:
            detail.append("%s token input" % ingresso)
        if uscita_token is not None:
            detail.append("%s token output" % uscita_token)
        print("  revisione completata%s"
              % ((" (" + ", ".join(detail) + ")") if detail else ""))
        return

    result = _esito(proc)
    if not result:
        # Un turno stream-json emette sempre almeno l'evento di init: stdout
        # vuoto significa che la CLI non e' mai partita davvero (argomenti
        # rotti, shim .cmd, auth). Il motivo e' quasi sempre su stderr.
        errori = (proc.stderr or "").strip()
        if errori:
            coda = " | ".join(errori.splitlines()[-3:])[:300]
            print(_sicuro("  ! la CLI %s non ha prodotto eventi: %s" % (ai, coda)),
                  file=sys.stderr)
        else:
            print("  ! la CLI %s non ha prodotto eventi su stdout" % ai, file=sys.stderr)
        return

    if result.get("is_error"):
        message = str(result.get("result", "")).strip()
        print(_sicuro("  ! revisione fallita: %s" % message[:300]), file=sys.stderr)
        if "authenticate" in message.lower() or "401" in message:
            print("    Autentica la CLI: lancia 'claude' e usa /login.",
                  file=sys.stderr)
        return

    cost = result.get("total_cost_usd")
    turns = result.get("num_turns")
    detail = []
    if turns:
        detail.append("%d turni" % turns)
    if cost:
        detail.append("$%.4f" % cost)
    print("  revisione completata%s" % ((" (" + ", ".join(detail) + ")") if detail else ""))
