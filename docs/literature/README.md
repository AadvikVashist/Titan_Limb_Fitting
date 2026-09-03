# Titan literature library

The PDF files in `data/papers/` are the source of record. Do not edit, rewrite,
or replace them in place. Searchable working copies live in `data/papers-md/`.
Each working copy names its source PDF, records its SHA-256 digest, and marks
every source page.

The source ledger is `docs/literature/source-ledger.csv`. Its citation key links
the BibTeX entry, PDF, and Markdown file. A blank local path means that the
library has metadata only. `metadata_only` does not mean that the paper is open
access or that the bibliographic facts have received an outside check.

## Naming

- BibTeX key: stable project citation key, kept for draft compatibility.
- PDF: `data/papers/<key>.pdf`.
- Markdown: `data/papers-md/<key>.md`.
- New keys: lower-case ASCII surname, four-digit year, and a short title word.
  Add `a`, `b`, and so on only when the first two parts would collide.

Do not rename an old key without changing every citation and recording the old
key. Some old keys contain capital or accented letters. They remain valid until
a planned migration can keep the draft and ledger in sync.

## Build and check

Run:

```bash
python3 scripts/literature/build_library.py
python3 scripts/literature/build_library.py --check
```

The builder uses `pdftotext -layout` and `pdfinfo`. When a page has no useful
text layer, it renders that page and uses Tesseract OCR for the working copy. It
checks the PDF page count against the Markdown page sections, rejects Unicode
replacement characters, and requires a small amount of text per page. Scans,
dense formulas, and image-only pages can still need a visual check. Before
quoting, compare the Markdown text with the named PDF page.

## Adding a source

1. Check the title, authors, year, journal or book, pages or article number, and
   DOI against the publisher, Crossref, NASA ADS, or another primary record.
2. Add one BibTeX entry. Reuse an existing key for the same work. Do not add a
   second key for a formatting variant.
3. Record a DOI without a URL prefix. Use the publisher page or DOI URL as the
   source URL.
4. If a lawful PDF is available, save the original bytes as
   `data/papers/<key>.pdf`. Do not bypass access controls.
5. Run the builder, inspect the ledger row, and compare at least the title page,
   one body page, and the references page against the PDF.
6. If no lawful PDF is available, leave the local paths blank and record a
   library, author-copy, preprint, or publisher access route in the ledger.

## Limits of automated validation

A matching page count and readable text do not prove that every symbol, table,
or column order is correct. The Markdown copy supports search and first-pass
reading. The PDF controls quotations, page references, figures, tables, and
equations.
