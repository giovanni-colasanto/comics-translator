# comics-translator

Translate comic pages and keep the artwork intact. Give it a PDF, get back the same PDF with every balloon and caption rewritten in another language, in a comic font, inside its own balloon.

```bash
comics-translator my-comic.pdf --to it --ai-review --review-ai claude
```

## How it works

Comic lettering is pixels, not text, so the translator works on the image in four stages:

1. **Source** (`comics_translator/sorgente`): renders each page, runs OCR (RapidOCR) and builds an immutable inventory of the text lines. Each line is then located inside its balloon or caption box, and the lines are grouped into reading units, telling dialogue apart from sound effects and lettering that belongs to the art.
2. **Translation** (`comics_translator/traduzione`): an AI model (Claude, Codex or GLM via their CLIs) translates the units while looking at the page images, so it knows who is talking and what is drawn. Work runs in batches that can be resumed after an interruption. Responses can be recorded as *cassettes* and replayed offline, so the whole pipeline is testable without calling a model.
3. **Rendering** (`comics_translator/disegno`): builds a translation plan, erases the original text with a fill that follows the balloon shape (multi-lobe balloons included), and fits the translated text inside it.
4. **Checks** (`comics_translator/verifica`): finds English left on the rendered page, dropped lines and broken structure. Optionally an AI model reviews the finished pages, and its corrections are applied in a reproducible way.

Each run writes a report with the outcome of every phase and the provenance of the code that produced it.

## Install

```bash
pip install "comics-translator @ git+https://github.com/giovanni-colasanto/comics-translator.git"
```

Optional extras: `wordninja` (restores spaces lost by OCR on all-caps lettering), `mt` (DeepL, Google or LibreTranslate draft translation), `argos` (offline draft translation).

For the AI review you need the CLI of the provider you choose (`claude`, `codex` or `glm`), logged in.

## Usage

```bash
# Layout test, fully offline: redraws the original text in place
comics-translator page.pdf --engine none

# Draft with DeepL (needs DEEPL_API_KEY), only some pages
comics-translator comic.pdf --engine deepl --pages 1-5 --to it

# Translation and visual review by an AI model
comics-translator comic.pdf --to it --ai-review --review-ai claude --visual-qa
```

Run `comics-translator --help` for all the options. It works on comics you own or that are free to use.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

The test suite (450+ tests) runs offline: model answers come from recorded cassettes.

The code, comments and command-line help are written in Italian, the language the project was developed in.

## License

MIT, see [LICENSE](LICENSE).
