"""Controllo visuale automatico del PDF gia' renderizzato.

La prima revisione AI vede la tavola originale e produce la traduzione. Questa
fase vede invece, affiancate, la sorgente e la pagina tradotta: e' l'unico
momento in cui si possono rilevare testo cancellato, balloon vuoti, scritte
duplicate, sovrapposizioni e corpi illeggibili introdotti dal rendering.

L'AI non modifica il piano direttamente. Produce una piccola specifica
``set``/``replace`` che viene validata da qa_corrections.apply_corrections.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from comics_translator.traduzione import ai_review
from comics_translator import manifesto_lotti
from comics_translator import pipeline_status as status
from comics_translator.verifica.qa_corrections import apply_corrections

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


PROMPT = """Sei il controllo qualita' finale di una traduzione di fumetti.

Per ogni pagina hai due immagini:
- `%(originals)s`: le pagine originali;
- `%(rendered)s`: le stesse pagine dopo traduzione e rendering.

Il file `%(plan)s` contiene soltanto i blocchi di quelle pagine e i relativi id.
Confronta attentamente ogni coppia. Cerca:
- senso tradotto male, italiano innaturale, errori OCR, genere o tono errati;
- nomi propri tradotti per sbaglio, simboli o parole importanti perduti;
- inglese residuo dentro balloon che dovevano essere tradotti;
- balloon svuotati, testo duplicato, troncato, sovrapposto o fuori area;
- testo troppo piccolo per essere leggibile;
- scritte parte del disegno che sono state coperte senza poter essere riprodotte.

Non correggere differenze puramente stilistiche e non inventare contenuto. Usa
l'originale per verificare il senso e il PDF tradotto per verificare la resa.
Mantieni le frasi volutamente interrotte nell'originale. Non completarle.
Non tentare di separare i lobi con sequenze di escape nel testo.
I blocchi con `salta: true` devono restare tali. Non riattivare gruppi protetti
e non trasferire il loro testo in blocchi vicini. Non mettere `salta: true` su
un dialogo o una didascalia: lascerebbe l'inglese sulla pagina. Se un gruppo
unico copre piu' lobi e il testo esce dai bordi o lascia un lobo vuoto,
riscrivi `testo` con le battute di ogni lobo come frasi separate, nell'ordine
di lettura: il programma le divide fra i lobi contando le frasi. L'inglese
rimasto fuori dai blocchi lo ripara una fase successiva: qui non serve
segnalarlo. Non inventare id per testo assente dal piano.
Scrivi il testo in UTF-8 con accenti corretti, non apostrofi al posto degli accenti.
Leggi soltanto il piano e le coppie di immagini di questo lotto.

Scrivi esclusivamente `%(out)s`, in JSON valido con questa struttura:
{
  "set": {
    "id-blocco": {
      "testo": "testo definitivo opzionale",
      "salta": false,
      "unisci_con": null,
      "corpo": null,
      "allinea": null
    }
  },
  "replace": {
    "id-blocco": [{"old": "frammento esatto", "new": "correzione"}]
  }
}

In `set` includi soltanto i campi che devono cambiare. Usa `replace` solo se
`old` compare esattamente una volta nel testo del blocco. Non inventare id e
non modificare coordinate o `_render`. Per un balloon corretto non scrivere
nulla. Se non trovi difetti, scrivi `{\"set\": {}, \"replace\": {}}`.
"""


def selected_plan_pages(plan: dict, pages: list[int]) -> list[dict]:
    """Restituisce le pagine richieste, senza il pesante stato di rendering."""
    wanted = set(pages)
    selected = []
    for page in plan.get("pagine", []):
        pno = int(page.get("indice", page.get("numero", 1) - 1))
        if pno not in wanted:
            continue
        blocks = [
            {key: value for key, value in block.items() if key not in ("_render", "sorgente", "legacy_id")}
            for block in page.get("blocchi", [])
        ]
        selected.append({
            **{key: value for key, value in page.items() if key not in ("blocchi", "sorgente", "struttura")},
            "blocchi": blocks,
        })
    return selected


def make_batches(plan: dict, pages: list[int], batch: int) -> list[list[dict]]:
    selected = selected_plan_pages(plan, pages)
    step = max(1, batch)
    return [selected[index:index + step] for index in range(0, len(selected), step)]


def _export_page(page, path: Path, max_edge: int) -> None:
    longest = max(page.rect.width, page.rect.height)
    dpi = max(36, min(220, int(72.0 * max_edge / max(longest, 1))))
    page.get_pixmap(dpi=dpi).save(path)


def export_comparisons(
    source_pdf: Path,
    rendered_pdf: Path,
    pages: list[int],
    workdir: Path,
    max_edge: int = ai_review.MAX_EDGE,
) -> dict[int, tuple[Path, Path]]:
    """Esporta coppie original/rendered con nomi stabili per numero pagina."""
    outdir = workdir / "confronto"
    outdir.mkdir(parents=True, exist_ok=True)
    source = fitz.open(source_pdf)
    rendered = fitz.open(rendered_pdf)
    try:
        if source.page_count != rendered.page_count:
            raise ValueError(
                "il PDF renderizzato ha %d pagine, la sorgente %d"
                % (rendered.page_count, source.page_count)
            )
        result = {}
        for pno in pages:
            original_path = outdir / ("p%02d.originale.png" % (pno + 1))
            rendered_path = outdir / ("p%02d.tradotta.png" % (pno + 1))
            _export_page(source[pno], original_path, max_edge)
            _export_page(rendered[pno], rendered_path, max_edge)
            result[pno] = (original_path, rendered_path)
        return result
    finally:
        source.close()
        rendered.close()


def _load_spec(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("file di correzione non prodotto")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("JSON di correzione non valido: %s" % exc) from exc
    if not isinstance(spec, dict):
        raise ValueError("la correzione deve essere un oggetto JSON")
    if set(spec) != {"set", "replace"}:
        raise ValueError("chiavi non ammesse o mancanti: richiesti set e replace")
    return spec


@dataclass(frozen=True)
class ReviewResult:
    plan: dict
    changes: int
    validated_batches: int
    total_batches: int
    outcome: status.PhaseResult | None = None
    #: lotti ripresi dal disco invece di essere richiesti di nuovo al modello.
    reused: int = 0

    @property
    def complete(self) -> bool:
        if self.outcome is not None:
            return self.outcome.complete
        return self.validated_batches == self.total_batches


_PAROLA = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
SOMIGLIANZA_NOME = 0.5
# Sotto questa quota di parole copiate che sopravvivono nel testo nuovo, la
# correzione riscrive la frase: e' una traduzione, non il ritocco di un nome.
QUOTA_TRADUZIONE = 0.5


def _traduce_la_frase(vecchio, nuovo, parole_originale):
    """True se la correzione traduce un pezzo rimasto in inglese.

    Un credito storpiato tiene quasi tutte le parole ("VC's Travis Lanham" ->
    "DC's Travis Lanham": due su tre restano); una frase inglese tradotta non ne
    tiene quasi nessuna ("THIS IS IMPOSSIBLE!" -> "E' IMPOSSIBILE!"). Senza
    questa distinzione "IMPOSSIBLE" -> "IMPOSSIBILE" sembrava un nome storpiato
    e il QA non poteva tradurre i blocchi lasciati in inglese (prova A2,
    spidey-63-9). Una parola sola resta sotto il filtro: non basta per dire.
    """
    copiate = [p.casefold() for p in _PAROLA.findall(vecchio)
               if p.casefold() in parole_originale]
    if len(copiate) < 2:
        return False
    nuove = {p.casefold() for p in _PAROLA.findall(nuovo)}
    rimaste = sum(p in nuove for p in copiate)
    return rimaste / len(copiate) < QUOTA_TRADUZIONE


def _altera_nome(vecchio, nuovo, parole_originale):
    """True se `nuovo` storpia una parola che il testo aveva copiato dall'originale.

    E' la forma del danno misurato su 5-doom-storm-5, pagina 4: il QA ha
    "corretto" quattro crediti giusti in sbagliati ("Ayodele" -> "Anodele",
    "VC's" -> "DC's", "Guimaraes" -> "Buimaraes"). Una parola ripresa tale e
    quale dall'originale e' un nome, una sigla o un marchio; sostituirla con una
    parola QUASI uguale non e' una traduzione (quella cambia la parola intera,
    "WHAT" -> "COSA") ma un refuso introdotto. Il QA non vede meglio dell'OCR
    un nome di persona, quindi nel dubbio vince l'originale.
    """
    if not isinstance(vecchio, str) or not isinstance(nuovo, str):
        return False
    if _traduce_la_frase(vecchio, nuovo, parole_originale):
        return False
    nuove = [p.casefold() for p in _PAROLA.findall(nuovo)]
    for parola in _PAROLA.findall(vecchio):
        chiave = parola.casefold()
        if chiave not in parole_originale or chiave in nuove:
            continue
        if any(SequenceMatcher(None, chiave, n).ratio() >= SOMIGLIANZA_NOME for n in nuove):
            return True
    return False


def filtra_nomi_alterati(plan: dict, spec: dict):
    """Toglie dalla specifica le correzioni che storpiano nomi copiati dall'originale.

    Restituisce (specifica filtrata, riparazioni). Non invalida il lotto: le
    altre correzioni dello stesso lotto restano buone.
    """
    blocchi = {b["id"]: b for p in plan.get("pagine", []) for b in p.get("blocchi", [])
               if isinstance(b, dict) and b.get("id")}
    filtrata = copy.deepcopy(spec)
    riparazioni = []

    def parole(bid):
        return {p.casefold() for p in _PAROLA.findall((blocchi.get(bid) or {}).get("originale") or "")}

    for bid, regole in list((filtrata.get("replace") or {}).items()):
        if not isinstance(regole, list) or bid not in blocchi:
            continue
        tenute = [r for r in regole if not (
            isinstance(r, dict) and _altera_nome(r.get("old"), r.get("new"), parole(bid)))]
        if len(tenute) != len(regole):
            riparazioni.append(status.issue(
                "qa_nome_alterato_scartato",
                "correzione che storpia una parola copiata dall'originale, ignorata",
                [], [bid]))
            if tenute:
                filtrata["replace"][bid] = tenute
            else:
                filtrata["replace"].pop(bid)
    for bid, campi in list((filtrata.get("set") or {}).items()):
        if not isinstance(campi, dict) or "testo" not in campi or bid not in blocchi:
            continue
        if _altera_nome(blocchi[bid].get("testo"), campi["testo"], parole(bid)):
            riparazioni.append(status.issue(
                "qa_nome_alterato_scartato",
                "testo del QA che storpia una parola copiata dall'originale, ignorato",
                [], [bid]))
            campi.pop("testo")
            if not campi:
                filtrata["set"].pop(bid)
    return filtrata, riparazioni


def applica_per_blocco(plan: dict, spec: dict, allowed_ids: set[str]):
    """Le correzioni di un lotto applicate una per blocco: (piano, interventi, perse).

    Una sola `replace` che non combacia faceva cadere l'intero lotto di QA
    (`qa_batch_invalid`): su thunderbolts 9 lotti su 14, su deadpool-10 5 su
    10. Ogni blocco passa per la stessa validazione di prima, da solo.
    """
    corrente, interventi, perse = plan, 0, []
    ids = sorted(set(spec.get("set") or {}) | set(spec.get("replace") or {}))
    for bid in ids:
        parziale = {"set": {}, "replace": {}}
        if bid in (spec.get("set") or {}):
            parziale["set"][bid] = spec["set"][bid]
        if bid in (spec.get("replace") or {}):
            parziale["replace"][bid] = spec["replace"][bid]
        try:
            corrente, fatti = apply_visual_corrections(corrente, parziale, allowed_ids)
        except ValueError as exc:
            perse.append({"id": bid, "errore": str(exc)})
            continue
        interventi += fatti
    return corrente, interventi, perse


def apply_visual_corrections(plan: dict, spec: dict, allowed_ids: set[str]):
    """Il QA automatico non puo' uscire dal lotto o riattivare aree saltate."""
    corrected, changes = apply_corrections(plan, spec)
    requested = set(spec.get("set", {})) | set(spec.get("replace", {}))
    if requested - allowed_ids:
        raise ValueError("correzioni fuori dal lotto: %s" % ", ".join(sorted(requested - allowed_ids)))
    before = {
        block["id"]: block
        for page in plan.get("pagine", []) for block in page.get("blocchi", [])
    }
    for page in corrected.get("pagine", []):
        for block in page.get("blocchi", []):
            if block["id"] in requested and any(
                escape in block.get("testo", "") for escape in ("\\n", "\\r")
            ):
                raise ValueError("escape letterale nel testo: %s" % block["id"])
            if before[block["id"]].get("salta") and not block.get("salta"):
                raise ValueError("riattivazione di blocco protetto: %s" % block["id"])
            target = block.get("unisci_con")
            if target and block["id"] in requested and (
                block.get("salta") or before[target].get("salta")
                or target not in allowed_ids
            ):
                raise ValueError("fusione con blocco protetto o fuori lotto: %s" % block["id"])
    return corrected, changes


def review_rendered(
    source_pdf: Path,
    rendered_pdf: Path,
    plan: dict,
    pages: list[int],
    workdir: Path,
    ai: str = ai_review.AI_DEFAULT,
    model: str | None = None,
    timeout: int = 1800,
    batch: int = 2,
    riprendi: bool = False,
) -> ReviewResult:
    """Distingue esplicitamente QA completato e lotti non validati.

    Come la revisione, il QA lascia sul disco la traccia di ogni lotto appena e'
    concluso: se la run muore al lotto 5 di 7, con `riprendi` i primi quattro
    tornano dal loro file di correzioni invece di essere ricomprati. La
    correzione ripresa viene rivalidata e riapplicata come se arrivasse ora: il
    file durevole e' quello che il modello ha scritto, non il verdetto.
    """
    status.plan_index(plan)
    status.plan_index({"pagine": selected_plan_pages(plan, pages)}, [p + 1 for p in pages])
    session = ai_review.prepare_provider(ai, model)
    workdir.mkdir(parents=True, exist_ok=True)
    pairs = export_comparisons(source_pdf, rendered_pdf, pages, workdir)
    batches = make_batches(plan, pages, batch)
    current = copy.deepcopy(plan)
    total_changes = 0
    validated_batches = 0
    reused = 0
    # Solo con `riprendi`: senza, una cartella rimasta da una prova precedente
    # non deve MAI passare per lavoro fatto.
    manifesto = manifesto_lotti.leggi(workdir) if riprendi else {}
    if riprendi and not manifesto:
        print("  ripresa richiesta ma qui non risulta nessun lotto di QA salvato: "
              "sara' un controllo completo")
    outcome = status.PhaseResult("qa", len(batches))
    status.start_phase("qa", len(batches))

    print(
        "QA visuale con %s (%s): %d pagine, %d lotti"
        % (session.ai, session.model, len(pages), len(batches))
    )
    for number, page_batch in enumerate(batches, start=1):
        label = "lotto%02d" % number
        plan_name = "piano.visuale.%s.json" % label
        out_name = "correzioni.visuali.%s.json" % label
        plan_path = workdir / plan_name
        out_path = workdir / out_name
        # Atomica perche' l'impronta del lotto si calcola su questo file: un
        # piano troncato da un crash a meta' scrittura sembra ancora leggibile,
        # e legherebbe la risposta salvata a un input mai mandato a nessuno.
        status.scrivi_atomico(
            plan_path,
            json.dumps({"pagine": page_batch}, ensure_ascii=False, indent=2),
        )

        page_numbers = [
            int(page.get("indice", int(page.get("numero", 1)) - 1))
            for page in page_batch
        ]
        originals = [pairs[pno][0] for pno in page_numbers]
        rendered = [pairs[pno][1] for pno in page_numbers]
        prompt = PROMPT % {
            "originals": ", ".join("`%s`" % p.relative_to(workdir) for p in originals),
            "rendered": ", ".join("`%s`" % p.relative_to(workdir) for p in rendered),
            "plan": plan_name,
            "out": out_name,
        }
        print(
            "\n  [QA %d/%d] pagine %s"
            % (number, len(batches), ", ".join(str(pno + 1) for pno in page_numbers))
        )
        impronta = manifesto_lotti.impronta(prompt, plan_path)
        if manifesto_lotti.riusabile(manifesto.get(label), impronta, out_path):
            print("  gia' fatto e salvato: riprendo senza richiamare il modello")
            status.event("qa_batch_reused", batch=number,
                         pages=[pno + 1 for pno in page_numbers], path=str(out_path))
            reused += 1
            riuso, proc = True, None
        else:
            riuso = False
            if out_path.exists():
                out_path.unlink()
            manifesto_lotti.registra(
                workdir, label, manifesto_lotti.TIPO_QA_VISUALE,
                stato=manifesto_lotti.TENTATIVO, numero=number, impronta=impronta,
                risposta=out_name, pagine=[pno + 1 for pno in page_numbers])
            proc = ai_review.run_agent(
                session, prompt, workdir, originals + rendered, timeout,
                output_path=out_path,
            )
        try:
            # `riuso`, non `proc is None`: run_agent torna None anche in timeout,
            # e quello e' il caso che agent_failure deve poter dichiarare perso.
            error = "" if riuso else ai_review.agent_failure(proc)
            if error:
                raise ValueError(error)
            spec = _load_spec(out_path)
            allowed_ids = {block["id"] for page in page_batch for block in page.get("blocchi", [])}
            spec, nomi = filtra_nomi_alterati(current, spec)
            if nomi:
                outcome.repairs.extend(nomi)
                status.event("qa_nomi_alterati_scartati", batch=number,
                             ids=[i for r in nomi for i in r["ids"]])
            try:
                current, changes = apply_visual_corrections(current, spec, allowed_ids)
            except ValueError as intero:
                current, changes, perse = applica_per_blocco(current, spec, allowed_ids)
                if not changes:
                    raise intero
                outcome.issues.append(status.issue(
                    "qa_batch_salvaged",
                    "correzioni non valide per intero (%s): %d interventi applicati, "
                    "%d blocchi scartati" % (intero, changes, len(perse)),
                    [pno + 1 for pno in page_numbers], [p["id"] for p in perse]))
                print("  ! correzioni visuali recuperate per blocco: %d scartate (%s)"
                      % (len(perse), intero), file=sys.stderr)
        except (ValueError, OSError) as exc:
            outcome.issues.append(status.issue("qa_batch_invalid", str(exc),
                                               [pno + 1 for pno in page_numbers]))
            manifesto_lotti.registra(
                workdir, label, manifesto_lotti.TIPO_QA_VISUALE,
                stato=manifesto_lotti.FALLITO, errore=str(exc))
            status.finish_phase(outcome, final=False)
            print("  ! correzioni visuali ignorate: %s" % exc, file=sys.stderr)
            continue
        # Solo qui il lotto diventa un risultato: validato, applicato, con la
        # correzione ancora sul disco e legata al suo input.
        manifesto_lotti.registra(
            workdir, label, manifesto_lotti.TIPO_QA_VISUALE,
            stato=manifesto_lotti.VALIDO, numero=number, impronta=impronta,
            risposta=out_name, risposta_sha256=status.sha256(out_path),
            pagine=[pno + 1 for pno in page_numbers], interventi=changes)
        total_changes += changes
        validated_batches += 1
        outcome.completed += 1
        outcome.pages.extend(pno + 1 for pno in page_numbers)
        outcome.ids.extend(sorted(allowed_ids))
        status.finish_phase(outcome, final=False)
        print("  %d interventi visuali validati" % changes)

    if reused:
        # Una run che ha ripreso non ha rifatto il lavoro: dirlo, perche' il
        # conto dei lotti "validati" da solo lascerebbe credere il contrario.
        print("\n  %d lotti di QA su %d ripresi dal salvataggio, "
              "senza chiamare il modello" % (reused, len(batches)))
        status.event("qa_resumed", reused=reused, batches=len(batches))
    status.finish_phase(outcome)
    return ReviewResult(current, total_changes, validated_batches, len(batches),
                        outcome, reused)
