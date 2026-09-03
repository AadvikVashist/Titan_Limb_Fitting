# Literature gap review

Review date: 2026-09-03.

This review compares the current draft and source library with primary journal,
publisher, author-manuscript, and NASA PDS records. It does not propose text for
the paper.

## Priority sources added

| Key | Gap covered | Record checked | Local copy |
| --- | --- | --- | --- |
| `clark2018vims` | Final VIMS wavelength and radiometric calibration, time-dependent wavelength shifts, and known unresolved limits | [NASA PDS document record](https://pds.nasa.gov/ds-view/pds/viewDocument.jsp?identifier=urn:nasa:pds:cassini_vims_saturn:document:vims-wavelength-and-radiometric-calibration-report&version=1.0), DOI `10.17189/1504137` | Official PDS PDF |
| `lemouelic2019archive` | Complete Titan VIMS archive, ISIS calibration, geometry, observing modes, and image-processing context | [University of Arizona publication record](https://experts.arizona.edu/en/publications/the-cassini-vims-archive-of-titan-from-browse-products-to-global-/), DOI `10.1016/j.icarus.2018.09.017` | arXiv author manuscript `1809.06545` |
| `cooper2025forward` | Modern full-disk VIMS selection and processing plus the effect of Titan haze scattering on disk brightness | [The Planetary Science Journal record](https://doi.org/10.3847/PSJ/ae071f), DOI `10.3847/PSJ/ae071f` | arXiv author manuscript `2507.00924` |
| `nixon2025jwst` | Post-Cassini northern-summer atmosphere, JWST/Keck imaging, and altitude-sensitive near-infrared interpretation | [Nature Astronomy record](https://www.nature.com/articles/s41550-025-02537-3), DOI `10.1038/s41550-025-02537-3` | arXiv author manuscript `2505.10655` |
| `snell2024titan` | Mission-long ISS north-south albedo asymmetry and seasonal variability | [The Planetary Science Journal record](https://doi.org/10.3847/PSJ/ad0bec), DOI `10.3847/PSJ/ad0bec` | Open-access publisher PDF |
| `west2018seasonal` | Mission-long seasonal cycle of the detached haze | [Nature Astronomy record](https://www.nature.com/articles/s41550-018-0434-z), DOI `10.1038/s41550-018-0434-z` | arXiv author manuscript `1804.10842` |

## Remaining gaps

1. **Calibration provenance in the method.** The draft cites the 2004 VIMS
   instrument paper but not the final RC19 calibration report. A later method
   review should state which calibration made the local cubes and compare it
   with the final PDS record. The final report says that VIMS-IR wavelengths
   shifted over time and lists unresolved calibration differences.
2. **Exact cube-processing lineage.** The Le Mouélic archive paper and Nantes
   portal describe ISIS calibration and navigation. The project must still
   record the version and steps used for each local cube; a citation alone does
   not establish that lineage.
3. **Limb and line-of-sight interpretation.** The library has occultation and
   detached-haze work, but it lacks a single checked source that turns this
   paper's fitted full-disk `u1 + u2` values into haze altitude, abundance, or
   single-scattering albedo. The paper should keep such claims limited unless a
   radiative-transfer retrieval tests them.
4. **Phase-angle effects.** The new Cooper paper shows that Titan's disk
   brightness departs strongly from a simple diffuse reflector, most clearly at
   high phase. The local study uses low-phase cubes, but it should still test or
   bound phase-angle effects rather than treat them as absent.
5. **Seasonal state after Cassini.** Nixon et al. add late northern-summer JWST
   and Keck observations. They give useful context but do not extend the same
   VIMS limb-profile measure. The paper should keep cross-instrument comparison
   descriptive.
6. **Current seasonal synthesis.** The library has strong primary studies but
   no recent, atmosphere-wide review that covers the full Cassini record plus
   JWST. A 2025 book chapter was found, but its access and scope need a full
   check before adding it.
7. **Unavailable or damaged local files.** `vinatier2015seasonal.pdf` has corrupt
   compressed streams and renders as blank pages in Poppler. A University of
   Bristol record lists an accepted manuscript, but its file endpoint denied
   automated access. Preserve the current bytes and obtain a lawful repository
   or library copy before replacing the working source.

## Stop rule for the first expansion

The first pass stops here because it now covers each named gap with at least one
high-value record or a clear open issue. A second pass should focus on the
remaining method questions, not add broad Titan background papers.
