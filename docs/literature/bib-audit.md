# BibTeX audit

Audit date: 2026-09-03.

## Current state

- `LaTex/NSAreferences.bib` contains 81 entries and 81 unique keys after four
  checked additions. The starting collection had 77 entries.
- The current paper uses 48 unique citation keys. All 48 resolve exactly.
- All 52 local PDFs match a BibTeX key by filename.
- Twenty-nine entries do not appear in the current paper. The four new entries
  also remain uncited until the manuscript work reviews them.
- BibTeX accepts the file and reports one warning: the book chapter
  `lebonnois2014general` has no publisher.
- Twenty-two entries have no DOI field. This pass recovered 21 exact DOI values
  from the source PDFs or the publisher and NASA records. Some older works and
  meeting abstracts may not have a DOI; each remaining case still needs a
  source-record check.
- Most entries have no `url` field. Many have an `adsurl`, and the ledger uses a
  DOI URL first, then another stable source URL when present.
- Twenty older or lightly described records still lack a checked outside
  identifier and source URL. The ledger assigns their stable project keys and
  states the gap rather than inventing an external record.

The BibTeX file already had uncommitted edits at the start of this audit. This
work does not reformat or reorder those entries. It adds four checked records at
the end.

## Duplicate works to resolve with aliases, not silent deletion

| Keys | Evidence in the current files | Safe next step |
| --- | --- | --- |
| `achterberg2008observation`, `achterberg2008titanb` | Same title; both PDF files have the same byte size and content hash. | Pick one canonical key, update citations if any, then retain an alias record in this audit. |
| `VIMSclouds`, `Rodriguez2011Clouds` | Same title and DOI `10.1016/j.icarus.2011.07.031`. | Keep the key used by the draft until a planned citation-key migration. |
| `2010Icar..207..485D`, `dekok2010titan` | Same title and DOI `10.1016/j.icarus.2009.10.021`. | Prefer the readable key after checking all citations. |
| `2017Icar..290..134A`, `adámkovics2017titan` | Same title and DOI `10.1016/j.icarus.2017.02.015`. | Prefer the readable key after checking all citations. |
| `tomasko2005rain`, `Tomasko2005Titan` | Same title, authors, journal, year, volume, and pages; the latter records DOI `10.1038/nature04126`. | Merge verified fields into one canonical entry after checking citations. |

## Format and record issues

- DOI values mix bare identifiers and full `https://doi.org/` URLs. The ledger
  normalizes them without changing the source file.
- Keys mix lower-case ASCII, capital letters, accented letters, and NASA ADS
  bibcodes. This works now but makes file links and manual citation entry harder.
- A standalone closing brace follows `Turtle2018Titan`. BibTeX ignores it in the
  current check, but it should be removed in a small, reviewed repair.
- `sotin2016characteristics` is a meeting abstract, not a peer-reviewed article.
  The entry correctly uses `booktitle`, but it should not stand in for a full
  method paper.
- Several entries rely on journal macros such as `\\icarus`, `\\nat`, and
  `\\psj`. The manuscript style may define them, but tools outside that LaTeX
  setup may not.

## Policy for later repairs

Keep code-only changes separate from scholarly-method changes. Make each BibTeX
repair small enough to review, cite the authority used to verify it, and rebuild
the ledger after every change. Never delete a key until every use has moved and
the old-to-new relation has been recorded.
