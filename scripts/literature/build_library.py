"""Build the Titan literature ledger and page-marked Markdown copies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIB_PATH = PROJECT_ROOT / "LaTex" / "NSAreferences.bib"
PDF_DIR = PROJECT_ROOT / "data" / "papers"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "papers-md"
LEDGER_PATH = PROJECT_ROOT / "docs" / "literature" / "source-ledger.csv"
PAGE_BREAK = "\f"
MIN_PAGE_CHARACTERS = 40
MIN_WRAPPED_VALUE_LENGTH = 2
FIELD_PATTERN = re.compile(
    r"(?m)^\s*([A-Za-z][\w-]*)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)\s*,?"
)
ENTRY_PATTERN = re.compile(r"@(?P<entry_type>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,")
DOI_PREFIX_PATTERN = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)
LEDGER_COLUMNS = (
    "key",
    "entry_type",
    "title",
    "authors",
    "container_title",
    "publication_year",
    "doi",
    "stable_identifier",
    "source_url",
    "local_pdf_path",
    "local_markdown_path",
    "conversion_status",
    "access_status",
    "relevance",
    "verification_notes",
    "pdf_sha256",
    "pdf_pages",
    "markdown_pages",
    "text_characters",
)
DUPLICATE_KEYS = {
    "achterberg2008observation": "Same work and PDF as achterberg2008titanb.",
    "achterberg2008titanb": "Same work and PDF as achterberg2008observation.",
    "VIMSclouds": "Same DOI as Rodriguez2011Clouds.",
    "Rodriguez2011Clouds": "Same DOI as VIMSclouds.",
    "2010Icar..207..485D": "Same DOI as dekok2010titan.",
    "dekok2010titan": "Same DOI as 2010Icar..207..485D.",
    "2017Icar..290..134A": "Same DOI as adámkovics2017titan.",
    "adámkovics2017titan": "Same DOI as 2017Icar..290..134A.",
    "tomasko2005rain": "Same work as Tomasko2005Titan; the latter has a DOI.",
    "Tomasko2005Titan": "Same work as tomasko2005rain; this entry has the DOI.",
}
SOURCE_NOTES = {
    "clark2018vims": (
        "Metadata checked against the NASA PDS record; PDF downloaded from the "
        "PDS record."
    ),
    "cooper2025forward": (
        "Journal facts checked against the publisher record; PDF is the arXiv "
        "author manuscript."
    ),
    "lemouelic2019archive": (
        "Metadata checked against the journal DOI and university records; PDF "
        "is the arXiv author manuscript."
    ),
    "nixon2025jwst": (
        "Journal facts checked against the publisher record; PDF is the arXiv "
        "author manuscript."
    ),
    "snell2024titan": (
        "Metadata checked against the open-access publisher record; PDF is the "
        "publisher version."
    ),
    "vinatier2015seasonal": (
        "The local PDF has corrupt compressed streams and renders as blank pages "
        "with Poppler; replace only after preserving this file and checking a "
        "lawful copy."
    ),
    "west2018seasonal": (
        "Metadata checked against the journal DOI; PDF is the arXiv author manuscript."
    ),
}
RELEVANCE_OVERRIDES = {
    "clark2018vims": (
        "Direct: final VIMS wavelength and radiometric calibration and known limits."
    ),
    "cooper2025forward": (
        "Direct: full-disk VIMS processing and haze-scattering effects across "
        "phase angle."
    ),
    "lemouelic2019archive": (
        "Direct: complete Titan VIMS archive, calibration, geometry, and image "
        "processing."
    ),
    "nixon2025jwst": (
        "Direct context: post-Cassini northern-summer atmosphere and "
        "altitude-sensitive imaging."
    ),
    "snell2024titan": (
        "Direct: Cassini ISS north-south albedo asymmetry and seasonal variability."
    ),
    "vinatier2015seasonal": (
        "Direct: CIRS limb retrievals of seasonal temperature, gas, and aerosol "
        "structure."
    ),
    "west2018seasonal": "Direct: mission-long seasonal cycle of Titan's detached haze.",
}
OCR_DISABLED_KEYS = {"vinatier2015seasonal"}


@dataclass(frozen=True)
class BibEntry:
    """One parsed BibTeX entry."""

    key: str
    entry_type: str
    fields: dict[str, str]


@dataclass(frozen=True)
class ConversionResult:
    """Checks recorded for one PDF-to-Markdown conversion."""

    status: str
    pdf_sha256: str
    pdf_pages: int
    markdown_pages: int
    text_characters: int
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that current Markdown files and ledger are reproducible.",
    )
    return parser.parse_args()


def matching_brace(text: str, open_index: int) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index in range(open_index, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
        if quoted:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unclosed BibTeX entry starting at character {open_index}")


def strip_bib_value(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    if (
        len(value) >= MIN_WRAPPED_VALUE_LENGTH
        and value[0] in '{"'
        and value[-1] in '}"'
    ):
        return value[1:-1].strip()
    return value


def parse_bibtex(path: Path) -> list[BibEntry]:
    text = path.read_text(encoding="utf-8")
    entries: list[BibEntry] = []
    for match in ENTRY_PATTERN.finditer(text):
        open_index = text.find("{", match.start())
        close_index = matching_brace(text, open_index)
        block = text[match.end() : close_index]
        fields = {
            field.casefold(): strip_bib_value(value)
            for field, value in FIELD_PATTERN.findall(block)
        }
        entries.append(
            BibEntry(
                key=match.group("key"),
                entry_type=match.group("entry_type").casefold(),
                fields=fields,
            )
        )
    return entries


def normalize_doi(value: str) -> str:
    return DOI_PREFIX_PATTERN.sub("", value.strip())


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", maxsplit=1)[1].strip())
    raise ValueError(f"pdfinfo did not report a page count for {path}")


def extract_pages(path: Path, citation_key: str) -> tuple[list[str], int]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    pages = text.split(PAGE_BREAK)
    if pages and not pages[-1].strip():
        pages.pop()
    ocr_pages = 0
    if citation_key in OCR_DISABLED_KEYS:
        return pages, ocr_pages
    with tempfile.TemporaryDirectory(prefix="titan-literature-ocr-") as temp_dir:
        temp_path = Path(temp_dir)
        for page_index, page in enumerate(pages):
            if len(page.strip()) >= MIN_PAGE_CHARACTERS:
                continue
            page_number = page_index + 1
            image_prefix = temp_path / f"page-{page_number}"
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-r",
                    "220",
                    "-singlefile",
                    "-png",
                    str(path),
                    str(image_prefix),
                ],
                check=True,
                capture_output=True,
            )
            ocr_result = subprocess.run(
                ["tesseract", str(image_prefix.with_suffix(".png")), "stdout"],
                check=True,
                capture_output=True,
                text=True,
            )
            pages[page_index] = ocr_result.stdout
            ocr_pages += 1
    return pages, ocr_pages


def markdown_text(
    entry: BibEntry, pdf_path: Path, pages: list[str], ocr_pages: int
) -> str:
    page_blocks = []
    for page_number, page in enumerate(pages, start=1):
        cleaned_page = page.rstrip()
        page_blocks.append(
            f"<!-- PDF_PAGE: {page_number} -->\n\n"
            f"## PDF page {page_number}\n\n"
            f"```text\n{cleaned_page}\n```"
        )
    source_hash = sha256(pdf_path)
    title = entry.fields.get("title", entry.key).replace("\n", " ")
    header = (
        "---\n"
        f'citation_key: "{entry.key}"\n'
        f'title: "{title.replace(chr(34), chr(39))}"\n'
        f'source_pdf: "{relative_path(pdf_path)}"\n'
        f'source_pdf_sha256: "{source_hash}"\n'
        'conversion_tool: "pdftotext -layout; tesseract OCR fallback"\n'
        f"ocr_pages: {ocr_pages}\n"
        'page_marker_scheme: "PDF_PAGE and PDF page heading"\n'
        "---\n\n"
        "This is a searchable working copy. Use the linked PDF as the source of record "
        "and check it before quoting.\n\n"
    )
    return header + "\n\n".join(page_blocks) + "\n"


def validate_conversion(
    pdf_path: Path, markdown_path: Path, pages: list[str], ocr_pages: int
) -> ConversionResult:
    pdf_pages = pdf_page_count(pdf_path)
    page_characters = [len(page.strip()) for page in pages]
    markdown_pages = len(pages)
    text_characters = sum(page_characters)
    sparse_pages = sum(count < MIN_PAGE_CHARACTERS for count in page_characters)
    replacement_characters = sum(page.count("�") for page in pages)
    checks = [
        pdf_pages == markdown_pages,
        markdown_path.exists(),
        text_characters >= pdf_pages * MIN_PAGE_CHARACTERS,
        replacement_characters == 0,
    ]
    status = "validated" if all(checks) else "needs_review"
    notes = (
        f"Page count {'matches' if pdf_pages == markdown_pages else 'does not match'}; "
        f"{sparse_pages} sparse page(s); {replacement_characters} replacement "
        "character(s); "
        f"OCR used on {ocr_pages} page(s)."
    )
    return ConversionResult(
        status=status,
        pdf_sha256=sha256(pdf_path),
        pdf_pages=pdf_pages,
        markdown_pages=markdown_pages,
        text_characters=text_characters,
        notes=notes,
    )


def source_url(fields: dict[str, str]) -> str:
    doi = normalize_doi(fields.get("doi", ""))
    if doi:
        return f"https://doi.org/{doi}"
    for key in ("url", "adsurl", "eprint"):
        if fields.get(key, "").startswith(("http://", "https://")):
            return fields[key]
    return ""


def source_relevance(entry: BibEntry) -> str:
    if entry.key in RELEVANCE_OVERRIDES:
        return RELEVANCE_OVERRIDES[entry.key]
    text = " ".join((entry.key, entry.fields.get("title", ""))).casefold()
    if "vims" in text or "mapping spectrometer" in text:
        return "Direct: Cassini VIMS instrument, processing, or Titan VIMS result."
    if "limb" in text or "occultation" in text or "transit" in text:
        return (
            "Direct: limb behavior, vertical structure, or line-of-sight "
            "interpretation."
        )
    if "haze" in text or "albedo asymmetry" in text or "season" in text:
        return "Direct: Titan haze asymmetry or seasonal change."
    if "titan" in text:
        return "Context: Titan atmosphere, circulation, clouds, or mission science."
    return "Method context: limb-darkening model or fitting background."


def build_row(entry: BibEntry, result: ConversionResult | None) -> dict[str, str | int]:
    fields = entry.fields
    pdf_path = PDF_DIR / f"{entry.key}.pdf"
    markdown_path = MARKDOWN_DIR / f"{entry.key}.md"
    doi = normalize_doi(fields.get("doi", ""))
    url = source_url(fields)
    identifier = f"doi:{doi}" if doi else url or f"project-bibkey:{entry.key}"
    notes = "; ".join(
        part
        for part in (DUPLICATE_KEYS.get(entry.key, ""), SOURCE_NOTES.get(entry.key, ""))
        if part
    )
    if result is not None:
        notes = "; ".join(part for part in (notes, result.notes) if part)
    elif not pdf_path.exists():
        notes = "; ".join(
            part
            for part in (
                notes,
                "Metadata from the existing BibTeX entry; local PDF not audited.",
            )
            if part
        )
    if not doi and not url:
        notes = "; ".join(
            part
            for part in (
                notes,
                "No outside stable identifier or source URL has been verified; "
                "the ledger uses the stable project key.",
            )
            if part
        )
    container = fields.get(
        "journal",
        fields.get("booktitle", fields.get("institution", fields.get("publisher", ""))),
    )
    return {
        "key": entry.key,
        "entry_type": entry.entry_type,
        "title": fields.get("title", ""),
        "authors": fields.get("author", ""),
        "container_title": container,
        "publication_year": fields.get("year", ""),
        "doi": doi,
        "stable_identifier": identifier,
        "source_url": url,
        "local_pdf_path": relative_path(pdf_path) if pdf_path.exists() else "",
        "local_markdown_path": relative_path(markdown_path)
        if markdown_path.exists()
        else "",
        "conversion_status": result.status if result else "not_converted",
        "access_status": "local_pdf" if pdf_path.exists() else "metadata_only",
        "relevance": source_relevance(entry),
        "verification_notes": notes,
        "pdf_sha256": result.pdf_sha256 if result else "",
        "pdf_pages": result.pdf_pages if result else "",
        "markdown_pages": result.markdown_pages if result else "",
        "text_characters": result.text_characters if result else "",
    }


def render_ledger(rows: list[dict[str, str | int]]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_or_check(path: Path, expected: str, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"Generated file is missing or stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def main() -> None:
    args = parse_args()
    entries = parse_bibtex(BIB_PATH)
    keys = [entry.key for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("BibTeX keys are not unique")

    rows = []
    for entry in entries:
        pdf_path = PDF_DIR / f"{entry.key}.pdf"
        markdown_path = MARKDOWN_DIR / f"{entry.key}.md"
        result = None
        if pdf_path.exists():
            pages, ocr_pages = extract_pages(pdf_path, entry.key)
            expected_markdown = markdown_text(entry, pdf_path, pages, ocr_pages)
            write_or_check(markdown_path, expected_markdown, args.check)
            result = validate_conversion(pdf_path, markdown_path, pages, ocr_pages)
        rows.append(build_row(entry, result))

    expected_ledger = render_ledger(rows)
    write_or_check(LEDGER_PATH, expected_ledger, args.check)
    local_pdf_count = sum(row["access_status"] == "local_pdf" for row in rows)
    validated_count = sum(row["conversion_status"] == "validated" for row in rows)
    print(
        f"Processed {len(entries)} entries: "
        f"{local_pdf_count} local PDFs, {validated_count} validated conversions."
    )


if __name__ == "__main__":
    main()
