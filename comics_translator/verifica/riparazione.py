"""Ciclo di riparazione: l'inglese che resta sul PDF finito si ripara qui.

Fino al 28 ottobre 2026 la run finiva dicendo "OCR del PDF finito: 100 righe
in inglese su 13 pagine" (woman-5) e si fermava con esito `partial`. Poi una
chat rifaceva ogni volta lo stesso giro: uno sweep OCR copiato albo per albo,
i ritagli guardati a occhio, uno script `.finale` con le operazioni di
`finale_helpers`. Su 38 albi FATTO, circa 8.600 comandi.

Qui quel giro diventa una fase, con due mani:

1. **deterministica**, senza modello: una riga inglese rimasta accanto a un
   blocco gia' tradotto e' quasi sempre un pezzo dello stesso balloon che
   l'OCR ha perso o un fantasma fra due riempimenti. Si copre (`copri`) o si
   allarga il blocco (`estendi_fill`), come facevano i `.finale`;
2. **il modello**, per il resto: vede la pagina originale e quella resa, le
   righe residue e i blocchi vicini, e risponde SOLO con operazioni di un
   vocabolario chiuso (tradurre un residuo, svegliare un blocco, cambiare un
   testo che non entra, aggiungere testo che l'OCR non ha mai visto, lasciare
   per scelta). Ogni operazione si valida da sola: una sbagliata cade senza
   portarsi dietro le altre.

Dopo ogni giro si ridisegna e si rilegge. Massimo `giri` volte; quello che
resta e' dichiarato, con il motivo, ed e' il punto da cui riparte un umano.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comics_translator import pipeline_status as status
from comics_translator.disegno.operazioni import Officina, unione
from comics_translator.sorgente import text_localization as localization
from comics_translator.traduzione import ai_review
from comics_translator.verifica import inglese_render

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

GIRI = 2
NATURE_TESTO = ("dialogo", "didascalia")
# Distanza (in altezze di riga) entro cui una riga orfana e' "dello stesso balloon".
VICINANZA_RIGHE = 1.2

PROMPT = """Sei l'ultimo controllo di una traduzione di fumetti dall'inglese
all'italiano. Il programma ha gia' tradotto e ridisegnato le pagine, ma
rileggendo il PDF finito trova ancora testo in inglese.

Per ogni pagina hai:
- `%(originali)s`: le pagine originali;
- `%(rese)s`: le stesse pagine tradotte, come sono adesso;
- `%(dati)s`: per ogni pagina la dimensione delle immagini, i `residui` (righe
  ancora in inglese rilette sulla pagina resa, con `box` [x0, y0, x1, y1] in
  pixel dell'immagine) e i `blocchi` vicini (testo originale, traduzione, se
  sono disegnati o saltati, la `natura`, il `box`). Un blocco con
  `illeggibile: true` ha una traduzione che non entra nel suo riquadro.

Guarda ogni residuo sulle due immagini e decidi. Scrivi `%(out)s`, JSON valido:
{"operazioni": [ ... ]}

Operazioni ammesse (nessun'altra):
- {"op": "traduci_residuo", "residuo": "<id>", "testo": "<italiano>",
   "sfondo": "balloon" | "banda"}
  il residuo e' una battuta, una didascalia o una scritta da tradurre: viene
  coperto e riscritto in italiano. "balloon" se sta in un balloon o riquadro a
  fondo uniforme, "banda" se e' stampato sul disegno. Se piu' residui sono la
  stessa battuta, usa il primo con il testo intero e "lascia" gli altri con
  motivo "assorbito";
- {"op": "sveglia", "blocco": "<id>", "testo": "<italiano>"}
  un blocco saltato che e' un dialogo o una didascalia: viene disegnato;
- {"op": "testo", "blocco": "<id>", "testo": "<italiano>"}
  cambia la traduzione di un blocco (per esempio una piu' corta per un blocco
  `illeggibile`, o una battuta completata con le parole che l'OCR aveva perso);
- {"op": "nuovo", "pagina": <numero>, "box": [x0, y0, x1, y1], "testo": "<italiano>",
   "sfondo": "balloon" | "banda"}
  testo in inglese visibile sulla pagina resa che NON e' fra i residui (l'OCR
  non l'ha letto: scritte stilizzate, verticali, in un burst). `box` in pixel
  dell'immagine, stretto attorno alle lettere;
- {"op": "lascia", "residuo": "<id>", "motivo": "logo" | "crediti" | "onomatopea"
   | "insegna" | "nome proprio" | "assorbito" | "falso positivo" | "altro"}

Regole: restano in inglese loghi, crediti, avvisi legali, onomatopee e suoni,
insegne e scritte dipinte nella scena, nomi propri. Si traducono riassunti,
titoli, "NEXT:", "TO BE CONTINUED", bug TV ("LIVE" -> "IN DIRETTA"), luoghi e
tempi, dialoghi e didascalie. Italiano naturale, accenti veri, maiuscole come
l'originale. Non inventare id. Leggi solo i file di questo lotto.
"""


# --------------------------------------------------------------------- residui
def residui_della_pagina(rapporto, numero):
    for pagina in rapporto.get("pagine", []):
        if pagina["pagina"] == numero:
            return [r for r in pagina["righe"] if not r.get("dichiarata")]
    return []


def _px(box_pt, scala):
    return [v / scala for v in box_pt]


def _altezza_tipica(officina, numero):
    altezze = sorted(r["box_px"][3] - r["box_px"][1] for r in officina.righe(numero).values()
                     if r.get("box_px"))
    return altezze[len(altezze) // 2] if altezze else 60.0


def _si_toccano(a, b, margine=0.0):
    return not (a[2] + margine < b[0] or b[2] + margine < a[0]
                or a[3] + margine < b[1] or b[3] + margine < a[1])


def _sovrapposizione_x(a, b):
    larghezza = min(a[2], b[2]) - max(a[0], b[0])
    return max(0.0, larghezza) / max(1.0, min(a[2] - a[0], b[2] - b[0]))


def attribuisci(officina, numero, residuo_px):
    """(operazione, blocco) per una riga residua accanto a un blocco disegnato.

    - dentro il riquadro del blocco: un fantasma che il riempimento non copre
      -> `copri` (il testo italiano si scrive comunque sopra);
    - appena sopra o sotto, nella stessa colonna: una riga dello stesso balloon
      che l'OCR non aveva legato al blocco -> `estendi` (riempimento e
      riquadro si allargano, il testo si riadatta).
    """
    altezza = _altezza_tipica(officina, numero)
    margine = VICINANZA_RIGHE * altezza
    migliore = None
    for block in officina.pagine[numero].get("blocchi", []):
        if not isinstance(block, dict) or block.get("salta"):
            continue
        render = block.get("_render") or {}
        if render.get("tipo") != "immagine" or not render.get("box_px"):
            continue
        box = render["box_px"]
        if _si_toccano(box, residuo_px):
            dentro = (residuo_px[0] >= box[0] - 4 and residuo_px[2] <= box[2] + 4
                      and residuo_px[1] >= box[1] - 4 and residuo_px[3] <= box[3] + 4)
            candidato = ("copri" if dentro else "estendi", block, 0.0)
        elif _si_toccano(box, residuo_px, margine) and _sovrapposizione_x(box, residuo_px) >= 0.5:
            distanza = max(box[1] - residuo_px[3], residuo_px[1] - box[3], 0.0)
            candidato = ("estendi", block, distanza)
        else:
            continue
        if migliore is None or candidato[2] < migliore[2]:
            migliore = candidato
    return (migliore[0], migliore[1]) if migliore else (None, None)


def ripara_deterministico(officina, rapporto, numeri):
    """Prima mano: residui accanto a blocchi disegnati. Ritorna (fatti, restanti)."""
    fatti, restanti = [], {}
    for numero in numeri:
        scala = officina.scala(numero)
        for residuo in residui_della_pagina(rapporto, numero):
            box = _px(residuo["box_pt"], scala)
            op, block = attribuisci(officina, numero, box)
            if op == "copri":
                officina.copri(block["id"], [box], "riparazione", "fantasma %r" % residuo["testo"])
            elif op == "estendi":
                officina.estendi_fill(block["id"], [box], "riparazione",
                                      "riga orfana %r" % residuo["testo"])
            else:
                restanti.setdefault(numero, []).append(residuo)
                continue
            fatti.append({"pagina": numero, "op": op, "blocco": block["id"],
                          "testo": residuo["testo"]})
    return fatti, restanti


def allarga_illeggibili(officina):
    """Gap E senza modello: un blocco che non entra prova il riquadro di tutto
    il suo riempimento, prima di tornare all'inglese."""
    fatti = []
    for bid, block in list(officina.blocchi.items()):
        render = block.get("_render") or {}
        if not render.get("testo_illeggibile"):
            continue
        profili = [p for p in render.get("fill_poly") or [] if len(p) >= 3]
        if not profili:
            continue
        xs = [pt[0] for p in profili for pt in p]
        ys = [pt[1] for p in profili for pt in p]
        pieno = [min(xs), min(ys), max(xs), max(ys)]
        box = render["box_px"]
        if (pieno[2] - pieno[0]) * (pieno[3] - pieno[1]) <= (box[2] - box[0]) * (box[3] - box[1]) * 1.05:
            continue
        render["box_px"] = pieno
        render.pop("testo_illeggibile", None)
        block["salta"] = False
        render["sbloccato_auto"] = "riparazione: riquadro allargato al riempimento"
        officina.adatta_corpo(block)
        fatti.append({"op": "allarga", "blocco": bid})
    return fatti


# ---------------------------------------------------------------- il modello
def _esporta(doc, pno, path, max_edge):
    page = doc[pno]
    lato = max(page.rect.width, page.rect.height)
    dpi = max(36, min(220, int(72.0 * max_edge / max(lato, 1))))
    page.get_pixmap(dpi=dpi).save(path)
    return dpi


def _dati_pagina(officina, numero, residui, dpi):
    """Residui e blocchi vicini in pixel dell'immagine esportata."""
    scala = officina.scala(numero)
    k = scala * dpi / 72.0  # px del piano -> px dell'immagine

    def img(box):
        return [round(v * k) for v in box]

    zona = [unione([_px(r["box_pt"], scala) for r in residui])] if residui else []
    blocchi = []
    for block in officina.pagine[numero].get("blocchi", []):
        if not isinstance(block, dict):
            continue
        render = block.get("_render") or {}
        if render.get("tipo") != "immagine" or not render.get("box_px"):
            continue
        illeggibile = bool(render.get("testo_illeggibile"))
        vicino = any(_si_toccano(render["box_px"], z, 400) for z in zona)
        saltato_testo = block.get("salta") and block.get("natura") in NATURE_TESTO
        if not (vicino or illeggibile or saltato_testo):
            continue
        voce = {"id": block["id"], "originale": block.get("originale"),
                "testo": block.get("testo"), "salta": bool(block.get("salta")),
                "natura": block.get("natura"), "box": img(render["box_px"])}
        if illeggibile:
            voce["illeggibile"] = True
        blocchi.append(voce)
    return {
        "pagina": numero,
        "residui": [{"id": "p%d-r%d" % (numero, i + 1), "testo": r["testo"],
                     "box": img(_px(r["box_pt"], scala))} for i, r in enumerate(residui)],
        "blocchi": blocchi,
    }


def _aggancia_all_ocr(officina, numero, box_px):
    """Il riquadro detto dal modello e' approssimato: lo si stringe sulle righe
    che l'OCR legge li' attorno, se ce ne sono."""
    from comics_translator.disegno.image_overlay import ocr_lines

    img = officina.immagine(numero)
    if img is None:
        return box_px
    w, h = box_px[2] - box_px[0], box_px[3] - box_px[1]
    x0, y0 = max(0, int(box_px[0] - 0.3 * w)), max(0, int(box_px[1] - 0.3 * h))
    x1, y1 = min(img.shape[1], int(box_px[2] + 0.3 * w)), min(img.shape[0], int(box_px[3] + 0.3 * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return box_px
    try:
        righe = ocr_lines(img[y0:y1, x0:x1], 0.3)
    except Exception:
        return box_px
    boxes = [[r.x0 + x0, r.y0 + y0, r.x1 + x0, r.y1 + y0] for r in righe]
    boxes = [b for b in boxes if _si_toccano(b, box_px)]
    return unione(boxes) if boxes else box_px


def applica_operazioni(officina, spec, dati, dpi_per_pagina):
    """Applica le operazioni del modello una per una; ritorna (fatte, scartate)."""
    if not isinstance(spec, dict) or not isinstance(spec.get("operazioni"), list):
        raise ValueError("serve {\"operazioni\": [...]}")
    residui = {r["id"]: (p["pagina"], r) for p in dati for r in p["residui"]}
    pagine = {p["pagina"] for p in dati}
    ammessi = {b["id"] for p in dati for b in p["blocchi"]}
    fatte, scartate = [], []
    for op in spec["operazioni"]:
        try:
            if not isinstance(op, dict):
                raise ValueError("operazione non e' un oggetto")
            tipo = op.get("op")
            testo = (op.get("testo") or "").strip() if "testo" in op else None
            if tipo in ("traduci_residuo", "sveglia", "testo", "nuovo") and not testo:
                raise ValueError("testo mancante")
            if tipo == "lascia":
                if op.get("residuo") not in residui:
                    raise ValueError("residuo sconosciuto")
            elif tipo in ("sveglia", "testo"):
                bid = op.get("blocco")
                if bid not in ammessi or bid not in officina.blocchi:
                    raise ValueError("blocco fuori dal lotto: %s" % bid)
                block = officina.blocchi[bid]
                if tipo == "sveglia":
                    officina.banda(bid, "riparazione", "svegliato dal modello", testo)
                else:
                    block["testo"] = testo
                    render = block["_render"]
                    if render.pop("testo_illeggibile", None):
                        block["salta"] = False
                        render["sbloccato_auto"] = "riparazione: testo accorciato"
                    officina.adatta_corpo(block)
            elif tipo in ("traduci_residuo", "nuovo"):
                sfondo = op.get("sfondo") if op.get("sfondo") in ("balloon", "banda") else "banda"
                if tipo == "traduci_residuo":
                    if op.get("residuo") not in residui:
                        raise ValueError("residuo sconosciuto")
                    numero, residuo = residui[op["residuo"]]
                    box = _px(residuo["box_pt"], officina.scala(numero))
                    inglese = residuo["testo"]
                else:
                    numero = int(op.get("pagina"))
                    if numero not in pagine:
                        raise ValueError("pagina fuori dal lotto")
                    box_img = op.get("box")
                    if (not isinstance(box_img, list) or len(box_img) != 4
                            or not box_img[0] < box_img[2] or not box_img[1] < box_img[3]):
                        raise ValueError("box non valido")
                    k = officina.scala(numero) * dpi_per_pagina[numero] / 72.0
                    box = _aggancia_all_ocr(officina, numero, [v / k for v in box_img])
                    inglese = ""
                riga = officina.nuova_riga(numero, "rip|%s|%s" % (box, testo), box,
                                           inglese or testo)
                officina.nuovo_blocco(numero, [riga], testo, "riparazione",
                                      "%s dal modello" % tipo, sfondo=sfondo,
                                      originale=inglese or None, natura="didascalia"
                                      if sfondo == "banda" else "dialogo")
            else:
                raise ValueError("operazione non ammessa: %r" % tipo)
            fatte.append(op)
        except (ValueError, KeyError, TypeError) as exc:
            scartate.append({"op": op, "errore": str(exc)})
    return fatte, scartate


def chiedi_al_modello(officina, sorgente_pdf, resa_pdf, restanti, workdir, giro, sessione,
                      timeout, batch=2):
    """Un lotto di pagine alla volta: immagini, dati, risposta, applicazione."""
    workdir.mkdir(parents=True, exist_ok=True)
    numeri = sorted(set(restanti) | {officina.pagina_di[b]["numero"]
                                     for b, bl in officina.blocchi.items()
                                     if (bl.get("_render") or {}).get("testo_illeggibile")})
    fatte_tot, scartate_tot = [], []
    with fitz.open(sorgente_pdf) as src, fitz.open(resa_pdf) as resa:
        for inizio in range(0, len(numeri), max(1, batch)):
            lotto = numeri[inizio:inizio + batch]
            nome = "riparazione.giro%d.lotto%02d" % (giro, inizio // batch + 1)
            originali, rese, dati, dpi = [], [], [], {}
            for numero in lotto:
                pno = int(officina.pagine[numero].get("indice", numero - 1))
                po = workdir / ("p%02d.originale.png" % numero)
                pr = workdir / ("p%02d.resa.giro%d.png" % (numero, giro))
                dpi[numero] = _esporta(src, pno, po, ai_review.MAX_EDGE)
                _esporta(resa, pno, pr, ai_review.MAX_EDGE)
                originali.append(po)
                rese.append(pr)
                dati.append(_dati_pagina(officina, numero, restanti.get(numero, []), dpi[numero]))
            dati_path = workdir / (nome + ".json")
            out_path = workdir / (nome + ".risposta.json")
            status.scrivi_atomico(dati_path, json.dumps({"pagine": dati}, ensure_ascii=False,
                                                        indent=2))
            if out_path.exists():
                out_path.unlink()
            prompt = PROMPT % {
                "originali": ", ".join("`%s`" % p.name for p in originali),
                "rese": ", ".join("`%s`" % p.name for p in rese),
                "dati": dati_path.name, "out": out_path.name,
            }
            print("\n  [riparazione giro %d] pagine %s" % (giro, ", ".join(map(str, lotto))))
            proc = ai_review.run_agent(sessione, prompt, workdir, originali + rese, timeout,
                                       output_path=out_path)
            errore = ai_review.agent_failure(proc)
            if errore or not out_path.is_file():
                scartate_tot.append({"lotto": nome, "errore": errore or "nessuna risposta"})
                print("  ! lotto di riparazione perso: %s" % (errore or "nessuna risposta"),
                      file=sys.stderr)
                continue
            try:
                spec = json.loads(out_path.read_text(encoding="utf-8"))
                fatte, scartate = applica_operazioni(officina, spec, dati, dpi)
            except (ValueError, OSError) as exc:
                scartate_tot.append({"lotto": nome, "errore": str(exc)})
                continue
            fatte_tot.extend(fatte)
            scartate_tot.extend(scartate)
            print("  %d operazioni applicate, %d scartate" % (len(fatte), len(scartate)))
    return fatte_tot, scartate_tot


# ------------------------------------------------------------------- la fase
def ripara(sorgente_pdf, out_path, plan, pagine, rendi, workdir, *, ai, model, timeout,
           giri=GIRI, batch=2, modello=True):
    """Il ciclo completo. `rendi(plan)` ridisegna il PDF in `out_path`.

    Restituisce il piano riparato e un riepilogo. Il PDF prima della
    riparazione resta accanto, come `.pre-riparazione.pdf`.
    """
    out_path = Path(out_path)
    # Una fase sola: i giri sono un tetto, non un lavoro promesso. Finire prima
    # perche' non c'e' piu' niente da riparare e' l'esito migliore.
    status.start_phase("riparazione", 1, "albi")
    esito = status.PhaseResult("riparazione", 1, unit="albi")
    riepilogo = {"giri": []}
    pre = out_path.with_name(out_path.stem + ".pre-riparazione.pdf")
    if out_path.is_file():
        pre.write_bytes(out_path.read_bytes())
    sessione = ai_review.prepare_provider(ai, model) if modello else None
    numeri_tutti = [p + 1 for p in pagine]
    for giro in range(1, giri + 1):
        rapporto = inglese_render.verifica_pdf(out_path, numeri_tutti, piano=plan)
        prima = rapporto["righe_non_dichiarate"]
        officina = Officina(plan, sorgente_pdf)
        try:
            illeggibili = allarga_illeggibili(officina)
            fatti, restanti = ripara_deterministico(officina, rapporto,
                                                    rapporto["pagine_con_inglese"])
            fatte_mod, scartate = [], []
            if modello and (restanti or any((b.get("_render") or {}).get("testo_illeggibile")
                                            for b in officina.blocchi.values())):
                fatte_mod, scartate = chiedi_al_modello(
                    officina, sorgente_pdf, out_path, restanti, Path(workdir), giro,
                    sessione, timeout, batch)
            toccate = sorted({officina.pagina_di[t["id"]]["numero"] for t in officina.toccati
                              if t["id"] in officina.pagina_di})
            if toccate:
                officina.sigilla(toccate)
                localization.enforce_localization(plan)
        finally:
            officina.chiudi()
        voce = {"giro": giro, "righe_prima": prima, "deterministiche": len(fatti),
                "allargati": len(illeggibili), "dal_modello": len(fatte_mod),
                "scartate": scartate}
        riepilogo["giri"].append(voce)
        status.event("riparazione_giro", **{k: v for k, v in voce.items() if k != "scartate"},
                     scartate=len(scartate))
        print("  riparazione giro %d: %d righe inglesi; %d riparate senza modello, "
              "%d riquadri allargati, %d operazioni del modello"
              % (giro, prima, len(fatti), len(illeggibili), len(fatte_mod)))
        if not (fatti or illeggibili or fatte_mod):
            break
        rendi(plan)
    finale = inglese_render.verifica_pdf(out_path, numeri_tutti, piano=plan)
    esito.completed = 1
    riepilogo["righe_finali"] = finale["righe_non_dichiarate"]
    riepilogo["pagine_finali"] = finale["pagine_con_inglese"]
    if finale["pagine_con_inglese"]:
        esito.issues.append(status.issue(
            "riparazione_incompleta",
            "%d righe ancora in inglese dopo la riparazione"
            % finale["righe_non_dichiarate"], finale["pagine_con_inglese"]))
    status.finish_phase(esito)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "riparazione.json").write_text(
        json.dumps(riepilogo, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  riparazione: %d righe in inglese restano su %d pagine"
          % (finale["righe_non_dichiarate"], len(finale["pagine_con_inglese"])))
    return plan, riepilogo
