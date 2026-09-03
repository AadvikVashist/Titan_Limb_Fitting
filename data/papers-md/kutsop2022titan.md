---
citation_key: "kutsop2022titan"
title: "Titan Stratospheric Haze Bands Observed in Cassini VIMS as Tracers of Meridional Circulation"
source_pdf: "data/papers/kutsop2022titan.pdf"
source_pdf_sha256: "f4a51aaacb0f83fbf3f1abef0c9f676dea8e161bf34c622fc20fb8724f5cc574"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                     https://doi.org/10.3847/PSJ/ac582d
© 2022. The Author(s). Published by the American Astronomical Society.




  Titan Stratospheric Haze Bands Observed in Cassini VIMS as Tracers of Meridional
                                    Circulation
     N. W. Kutsop1              , A. G. Hayes1,3 , P. M. Corlies2 , S. Le Mouélic4 , J. I. Lunine1 , C. A. Nixon5                     , P. Rannou6     ,
                                         S. Rodriguez7 , M. T. Roman8 , C. Sotin4,9 , and T. Tokano10
                                                              The Cassini VIMS Team
                                              1
                                     Cornell University, 300 Day Hall, Ithaca, NY 14853-2801, USA; nwk25@cornell.edu
                                    2
                             Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139-4307, USA
                       3
                         Cornell Center for Astrophysics and Planetary Science, 104 Space Sciences Building, Ithaca, NY 14853, USA
                         4
                           Laboratoire De Planétologie Et Géosciences, 2 Chem. de la Houssinière Bâtiment 4, 44300 Nantes, France
                                  5
                                    NASA Goddard Space Flight Center, 8800 Greenbelt Road, Greenbelt, MD 20771, USA
6
  Groupe de Spectrométrie Moléculaire et Atmosphérique (GSMA), UMR CNRS 7331, Université de Reims, U.F.R. Sciences Exactes et Naturelles, Moulin de la
                                                        Housse, B.P. 1039, 51687 Reims Cedex 2, France
                           7
                             Université de Paris, Institut de physique du Globe de Paris, CNRS, 1 Rue Jussieu, 75005 Paris, France
                                              8
                                                University of Leicester, University Road, Leicester, LE1 7RH, UK
                                       9
                                         Jet Propulsion Laboratory, 4800 Oak Grove Drive, Pasadena, CA 91109 , USA
                                             10
                                                Universität zu Köln, Albertus-Magnus-Platz 50923, Köln, Germany
                          Received 2021 August 26; revised 2022 February 21; accepted 2022 February 22; published 2022 May 20

                                                                Abstract
             We analyzed Cassini data to derive the nature and evolution of circumglobal annuli observed in the stratosphere
             of Titan, Saturnʼs largest moon. The annuli were observed between 2004 and 2017 in data acquired by the
             Visual and Infrared Mapping Spectrometer on board the Cassini spacecraft. We observed a north polar annulus,
             an equatorial annulus, and several secondary annuli. Pre-Cassini telescopic observations by the Hubble Space
             Telescope and Keck reported an atmospheric feature consistent with the presence of a south polar annulus
             between 1999 and 2001, although this feature was not observed by Cassini. Relative to the atmosphere near the
             annuli, they appear dark at 300–500 nm and bright in methane absorption channels such as the ones at 900 and
             1150 nm. The stratosphere seems to rotate around the north pole. Alternatively, it seems to rotate about a point
             offset from solid-body rotation axis by a few degrees; this point in turn rotates around the solid-body
             rotation axis.
             Uniﬁed Astronomy Thesaurus concepts: Stratosphere (1640); Titan (2186); Atmospheric circulation (112);
             Spectrophotometry (1556); Seasonal phenomena (1437)


                                    1. Introduction                                      links between the annuli seen in VIMS and clouds detected by
                                                                                         ISS (Turtle et al. 2018; Figure 2).
   Titan has a thick organic haze that obscures its surface at
                                                                                            Circumglobal bands have been previously identiﬁed in
visible and infrared wavelengths and is produced through the
                                                                                         Titanʼs atmosphere. Like the north–south asymmetry, these
dissociation of CH4 and N2 by UV light and charged particles
                                                                                         have been observed to be dark at wavelengths shorter than
(Yung et al. 1984; Lavvas et al. 2008a, 2008b). Noteworthy
                                                                                         600 nm and bright in methane absorption channels such as
haze features include a globally extensive thin layer of
                                                                                         those at 900 and 1150 nm. North polar atmospheric bands have
detached haze (the detached haze layer; Smith et al. 1981;
                                                                                         been previously observed (Smith et al. 1982; Grifﬁth et al.
Rages & Pollack 1983; Rannou 2000; Teanby et al. 2009; West
                                                                                         2008; Jennings et al. 2015; Le Mouélic et al. 2018), and the
et al. 2018; Seignovert et al. 2021), a hemispheric asymmetry
in brightness (the north–south asymmetry; Sromovsky et al.                               north polar annulus (NPA) has likely been observed before
1981; Smith et al. 1982; Tomasko & Smith 1982), and polar                                (Sromovsky et al. 1981; Le Mouélic et al. 2012; Rannou et al.
hoods that occur during local winter (i.e., the north and south                          2012), but not with the morphology and spectral characteristics
polar hoods; Lorenz et al. 1997; West et al. 2016; Le Mouélic                            we report for the ﬁrst time, which were only possible with near
et al. 2018; Seignovert et al. 2021; Penteado et al. 2010).                              nadir observations of the north pole taken around 2014. The
Tracking the distribution and evolution of Titanʼs haze is a key                         equatorial annulus (EQA) has been previously identiﬁed in
tool for studying Titanʼs atmospheric photochemistry,                                    other works, either directly or indirectly (Roman et al. 2009; de
dynamics, and circulation. Herein we describe circumglobal                               Kok et al. 2010). A south polar annulus (SPA) was observed in
bands, or annuli (Figures 1 and 2), in Titanʼs atmosphere using                          Titanʼs south polar region by Keck and the Hubble Space
observations from Cassini’s Visual and Infrared Mapping                                  Telescope (HST; Lorenz et al. 2001; Roe et al. 2002;) prior to
Spectrometer (VIMS) and discuss their relevance to the nature                            Cassini’s arrival at Saturn. Throughout the course of the
of Titanʼs stratosphere and circulation in general. We also used                         Cassini mission, we did not observe the SPA seen previously.
the Cassini Imaging Science Subsystem (ISS) to investigate                                  Circumglobal atmospheric bands are found on all solar
                                                                                         system bodies with a substantial atmosphere and are valuable
                 Original content from this work may be used under the terms
                                                                                         observables for investigating a variety of planetary processes.
                 of the Creative Commons Attribution 4.0 licence. Any further            On Earth and Venus these bands are found in the jet streams,
distribution of this work must maintain attribution to the author(s) and the title       which can be observed by tracking clouds or infrared radiance
of the work, journal citation and DOI.                                                   (Horinouchi et al. 2017). These jets reveal the location of

                                                                                     1
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                          Kutsop et al.




Figure 1. Orthographic mosaics of T096, 2013-12-01, LS = 51°. 2. (A) 1155 nm mosaic. (B) 1155 nm mosaic with manifold correction. (C) Context using VIMS-ISS
map (Seignovert et al. 2019).




Figure 2. Top: T061, 2009-08-25, LS = 0°. 49; bottom: T096, 2013-12-01, LS = 51°. 2. (A) 494 nm mosaic with manifold correction. (B) 1155 nm mosaic with
manifold correction. (C) ISS NAC observations taken during the same respective ﬂyby. Both are taken with ﬁlters CL1 and CB3 with an effective wavelength of
938.03 nm (Knowles 2016). Image names are N1630071434_1 (top) and N1764434226_1 (bottom).

meridional convergence zones and can be used to understand                       dynamics separated by strong meridional and vertical gradients
the pattern of zonal circulation. Annular modes of variability                   in the zonal (i.e., east–west) winds (Fletcher et al. 2020).
have recently been identiﬁed in the atmospheres of Mars and                      Auroras are particularly bright bands found on the giant
Titan (Battalio & Lora 2021a). Annular modes such as these                       planets, Earth, Mars, and Ganymede. Auroras at Ganymede
explain much of the internal variability of Titanʼs troposphere                  reveal that the largest moon of the solar system has an internal
as simulated in a global circulation model (GCM). The gas                        ocean (Saur et al. 2015). Jupiterʼs auroras are used as evidence
giants (Jupiter and Saturn) and the ice giants (Uranus and                       in explaining why the gas giantʼs upper atmosphere is much
Neptune) all display patterns of planetary banding, with regions                 hotter than expected from sunlight alone. Spectroscopic
of different temperatures, composition, aerosol properties, and                  observations of these bands from Keck II suggest that the

                                                                             2
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                          Kutsop et al.

                                                                                        a time series set of measurements of the tilt, as well as its
                                                                                        azimuthal offset, throughout the Cassini mission (2004–2017),
                                                                                        which provides an important database that can be used for
                                                                                        future modeling of the origin and evolutions of the tilt, the
                                                                                        superrotation of the middle atmosphere, and the link
                                                                                        between them.
                                                                                           Saturnʼs eccentricity (0.055) and obliquity (26°. 7) dictate the
                                                                                        variation in the amount and location of haze production
                                                                                        throughout a Titan year. The obliquity also causes the
                                                                                        meridional circulation to be asymmetric about the equator
                                                                                        and to reverse semiannually (Lebonnois et al. 2012; Teanby
                                                                                        et al. 2012). The ﬂow rises in the summer polar region and
                                                                                        descends in winter polar regions. As the cells redistribute the
                                                                                        heat in Titanʼs atmosphere, it also vertically transports gases
                                                                                        and aerosols to high altitudes, transports them horizontally
                                                                                        across the globe, and carries them back down to lower
                                                                                        altitudes. During the fall and spring seasons, the single pole-to-
                                                                                        pole cell splits into two symmetric cells, where the atmosphere
                                                                                        ascends at the equator and descends at each of the poles
Figure 3. Transect of pixels from mosaic of T61 ﬂyby [2009-08-25] (see
Figure 2). The deviation in each of the three curves is attributed to the EQA.          (Mitchell et al. 2006; Tokano 2011; Newman et al. 2016; Lora
The inﬂection point of each deviation is attributed to the center of the annulus,       et al. 2019; Battalio et al. 2022). We discuss in Section 4 the
and the Gaussian tail on either side of the inﬂection point is the uncertainty in       apparent correlation of the annuli presence and location with
locating the annulus. The north–south asymmetry appears as the smooth
parabolic curve between the hemispheres. The EQA adds a sharp deviation in              the predicted meridional circulation cell cycle.
the curve, without which the north–south asymmetry would appear as a
gradient between the hemispheres, rather than the asymmetry we see.
                                                                                                          1.1. The Equatorial Annulus
excess heat is produced by the redistribution of auroral energy
                                                                                           The EQA occurs at the boundary of the north–south
(O’Donoghue et al. 2021).
                                                                                        asymmetry (Roman et al. 2009; de Kok et al. 2010). The
   Observations of Titan show that the stratosphere is super-
                                                                                        north–south asymmetry is a feature of Titanʼs atmosphere
rotating with wind speeds of ∼200 m s−1 (Flasar & Achterberg
                                                                                        where one hemisphere is darker than the other; which
2009). Several Titan GCMs are able to reproduce the
                                                                                        hemisphere is darker varies as a function of time, and the
superrotation, though not always with wind speeds matching
                                                                                        degree of contrast varies with wavelength. The north–south
observations. The results of these GCMs are in agreement
with the Gierasch−Rossow−Williams (GRW) mechanism for                                   asymmetry was ﬁrst observed in 1980-11-11 in Voyager 1
producing superrotations (Hörst 2017). In the GWR mech-                                 observations (and retrospectively observed in Pioneer 11
anism, angular momentum is transported to higher altitudes                              observations, 1979-08-31) and has since been observed
and then poleward by mean meridional circulation and is                                 continuously by ground-based telescopes, HST, and Cassini
transported down and to the equator by barotropic waves                                 (Smith et al. 1981; Sromovsky et al. 1981; Caldwell et al.
generated by instabilities on the edges of the high-latitude jets                       1992). Previous analysis of ISS data from 2004 to 2007
(Gierasch 1975; Rossow & Williams 1979; Hourdin et al.                                  indicates that the EQA/north–south asymmetry was present at
1995; Lebonnois et al. 2014).                                                           around 80 km altitude with an axial tilt of 3°. 8 ± 0.9 relative to
   Analysis of the stratospheric superrotation by the Cassini                           the spin axis, with the vector directed 79° ± 24° to the west of
Composite Infrared Spectrometer (CIRS) revealed the strato-                             the subsolar longitude (Roman et al. 2009). A height of 80 km,
sphere to be tilted 4° with respect to the solid-body rotation                          with possible additional contributions between 50 and 150 km,
axis (Achterberg et al. 2008). Multiple observations of the tilt                        was also inferred for the EQA from Cassini VIMS (de Kok
showed that it was directed 76° west of the subsolar longitude.                         et al. 2010), while CIRS measurements of HCN indicated that
Further observations using CIRS suggest that the tilt is not                            the hemispheric asymmetry extended to at least 125 km
ﬁxed in a solar reference frame, but rather ﬁxed in an inertial                         (Teanby et al. 2010). In images of scattered light, the EQA
reference frame (Achterberg et al. 2011). The tilt has been                             helps to exaggerate the difference between the hemispheres
conﬁrmed by multiple follow-up investigations using composi-                            (Figure 3). The variability in altitude estimations above can be
tion and tracking atmospheric features (Achterberg et al. 2008;                         attributed to the variety of instruments used, changes in the
Roman et al. 2009; Teanby et al. 2010; West et al. 2016).                               EQA over time, and targets accessed, i.e., the HCN used by
   While mechanisms that explain the tilt and its relationship                          Teanby et al. (2010) may be a part of the EQA but at an altitude
with superrotation have been put forth (Achterberg et al. 2008;                         different from the haze in the north–south asymmetry used by
Tokano 2010), there remains a lack of consensus. Achterberg                             Roman et al. (2009).
et al. (2008) proposed that the tilt, by feedback between the                              The EQA is one of the few features observed at visible
circulation and the heating, facilitates the vertical transport of                      wavelengths that has been observed between Titan years, as the
angular momentum to balance the heat ﬂow and insolation at                              EQA can be inferred from observations of the north–south
low latitudes. Tokano (2010) proposed that the tilt is the result                       asymmetry seen by Pioneer 11 (1979 September, Tomasko &
of thermal tides and is only possible if atmospheric waves                              Smith 1982), Voyager 1 (1980 November, Sromovsky et al.
perturb the circulation. With the annuli, we are able to produce                        1981), and Voyager 2 (1981 August, Smith et al. 1982).

                                                                                    3
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                          Kutsop et al.

                 1.2. The North Polar Annulus                           means that closest-approach observations, with their rapidly
                                                                        changing viewing geometry, were not used.
   The NPA is, on average, spectrally and morphologically
                                                                           VIMS image cubes from the Vis detector have stripes of
identical to the EQA. The spectra, size, and position of the
                                                                        varying brightness at each sample running across the lines. The
NPA change more profoundly than the EQA, which we discuss
                                                                        stripes are due to offsets of about 100 DN introduced by the
in Section 4.2. Because more than half of the NPA can be seen
                                                                        readout electronics on the signal (Filacchione et al. 2007). We
in a single observation and there are multiple observations with
                                                                        developed a destriping routine, described in Appendix A,
the same illumination geometry taken close in time to one
                                                                        which we applied to every VIMS-Vis image cube.
another, we are capable of triangulating the NPA (Section 2.4).
                                                                           We leverage Titanʼs smoothly scattering atmosphere to
We use the vector normal to the plane of the modeled ellipses
                                                                        improve contrast in the image cubes and increase spectral
to describe its tilt with respect to the rotation axis of Titanʼs
                                                                        clarity (see Appendix B). Manifold corrections were applied to
solid body (θNP), its axial precession west of the subsolar
                                                                        all image cubes from both detectors, although we only used the
longitude (fSol), and its declinations in an inertial reference
                                                                        manifold-corrected data for making mosaics and examining
frame. Parameters of interest for the two models are shown in
                                                                        images (Figures 1 and 2). Data used for spectral analysis, like
Table 1.
                                                                        the transects discussed below, have no corrections applied
                                                                        except the VIMS Radiometric Calibration, RC19 (Clark et al.
                 1.3. The South Polar Annulus                           2018), which is applied to all VIMS PDS data, and the
   The SPA was not observed in the Cassini data, due to the             destriping applied to VIMS-Vis.
timing of the event. The SPA appeared as a weak dark ring
using HST at 336 nm (HST Wide Field and Planetary Camera                         2.2. Identifying the Annuli through Transects
ﬁlter F336W; Lorenz et al. 2001) and a bright collar using
Keck II at 1158 and 1702 nm (W. M. Keck II KCAM and                        Taking a north–south transect of the pixels in a mosaic at a
SCAM, ﬁlters J1158 and H1702; Roe et al. 2002) around 60°               single longitude and over all latitudes produces a curve of
south latitude and centered on the south pole from 1999 to              brightness for the queried wavelengths and longitudes
2001. Lorenz et al. (2001) concluded that at least some of the          (Figure 3). In these meridional transects the band can be seen
material responsible for this feature must be at altitudes of           as a deviation in the normally smooth and featureless curves.
above 150 km. Roe et al. (2002) used the presence of the SPA            For wavelengths where the annuli are darker than the
in the J1158 Keck ﬁlter to conclude that SPA must be at or              surrounding atmosphere, the deviation is concave, while for
above 40–50 km altitude. Both Roe et al. (2002) and Lorenz              wavelengths where the band is bright, the deviation is convex.
et al. (2001) note the similarity of the SPA they observed with            We used the transects to determine the presence and location
the dark northern collar observed by Voyager 2 around the               of the annuli. We propose that the center of the annulus is at the
north pole and suggested that the SPA has a seasonal origin.            inﬂection point for the observed deviations and that the width
The altitude difference between Roe et al. (2002) and Lorenz            of the annulus is where the deviation returns to the smooth
et al. (2001) can be attributed to the different wavelengths used       path. The annuli in general appear darkest at ∼500 nm and
to interpret the altitude (i.e., 336 nm versus 1702 nm).                brightest at ∼1150 nm, and they are also very bright at
                                                                        ∼900 nm. We use these three channels to investigate the annuli
                                                                        because they show the greatest contrast and span both the
                       2. Data/Methods                                  VIMS-Vis and VIMS-IR detectors. Using both detectors to
   Our processing workﬂow starts with the publicly available            identify the annuli is necessary for redundancy because
VIMS data set on the Planetary Data System (PDS; Le Mouélic             occasionally one detector will have bad data. We took transects
et al. 2019, https://pds-imaging.jpl.nasa.gov/data/cassini/             of each ﬂyby on the ingress and egress hemisphere, every 10°
cassini_orbiter/covims_0001/data/). VIMS consisted of two               of longitude (i.e., 60°, 70°, 80°, etc.) and spanning all visible
imaging spectrometers: a visual detector (VIMS-Vis) with 96             latitudes. This produced a data set of 4829 transects, and each
channels between 0.35 and 1.05 μm, and an infrared detector             of these was examined to determine whether deviations
(VIMS-IR) with 256 spectroscopic channels between 0.89 and              matching the ones associated with the annuli were present.
5.13 μm (Brown et al. 2004). The absorption and scattering by              We created a data set of annuli detections with information
methane, nitrogen, and haze mean that some channels in VIMS             on the viewing geometry, coordinates, and time of observation
can see the surface, while for other channels the atmosphere is         by analyzing the transects. Transects were analyzed manually
completely opaque (Corlies et al. 2021). For every pixel, we            in a random order (randomizing over both ﬂyby and longitude)
determined the viewing geometry and location information                to prevent unintentionally biasing our search to patterns seen
using the SPICE toolkit from NASA’s Navigation and                      when looking at consecutive longitudes in the same ﬂyby. We
Ancillary Information Facility (Acton et al. 2018).                     then reorganized our data and displayed those detections over
                                                                        the appropriate ﬂyby mosaics. With the context of the mosaics
                                                                        we found that our initial search results produced many false
             2.1. Data Correction and Mosaicking
                                                                        positives. These false positives were caused by several sources,
   We created supersampled mosaic images of Titan for ingress           including the north polar hood, the south polar vortex,
and egress observations of each ﬂyby, weighting higher spatial          observation of the limb at irregular angles, and seams between
resolution pixels more in the mosaicking. We rasterized the             VIMS image cubes (Kelland et al. 2018; Le Mouélic et al.
pixels to a spatial resolution of 5 km2 and calculated the              2018). We identiﬁed and removed these false positives by
weighted average of all individual image pixels that ﬁlled each         reinvestigating each transect, this time in ﬂyby and longitude
pixel of the mosaic. The primary constraints on data                    order, as well as viewing the relevant mosaics. We further
incorporated into the mosaics were a pixel scale of less than           reﬁned and expanded our detections by explicitly looking for
250 km and a change in the phase angle, θ, of less than 3°. This        patterns in the location of the inﬂection point between the

                                                                    4
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                                   Kutsop et al.




Figure 4. The average latitude of the observed primary and secondary annuli. The black regions are all the locations in time that were at a viewing geometry shared by
the majority (95%) of our detections of the annuli (incidence angle < ∼ 90° and emission < ∼ 60°). The red circle at ∼750 latitude and LS = ∼40° is a direct detection
of the NPA in ﬂyby T85.

transects that correlated to features that had the morphology of                      third-degree polynomial, as well as splines, and the results are
the annuli. This is the data set we use for tracking the annuli                       very similar. We therefore use the straight line, as anything
over the course of the Cassini mission and producing the                              more complicated is not justiﬁable. The differential spectrum is
sinusoidal ﬁts and triangulated models. Our ﬁnal data set                             then the maximum difference between the deviation and the
contained 552 transects for the NPA and 686 transects for the                         interpolated data. Figure 5 shows the average differential
EQA, as well as a total of 256 transects for the four secondary                       spectrum of the T061 EQA. This is a good representative of the
annuli. Figure 4 shows our range of detections through the                            EQA and NPA spectra, except for early and late spectra of the
average latitude of the annuli. Information on the location                           NPA, which we discuss in Section 4.2.
timing and aspects, such as tilt of the averages of our transects,
is provided in Table 1.                                                                            2.3. Altitude from Spectra/Spectraltimetry
   We use the transects to acquire spectra of the annuli. The
                                                                                         We estimated the altitude of the annuli using spectraltimetry.
spectrum of the annuli has no distinguishing features when
                                                                                      Features entrained in an absorbing medium will become visible
compared to spectra taken from pixels located just north or                           at different pressure levels dependent on the wavelengths.
south of the annuli (Figure C). This suggests that the annulus is                     Using superposition, we can infer the shape and altitude/depth
compositionally indistinct from the rest of the atmosphere and                        of a feature by its appearance in the spectra. This technique has
that observed brightness differences are the result of a local                        already been used extensively at Titan for measuring the height
increase in haze optical depth. Instead of a typical VIMS                             of clouds using observations from Cassini and ground-based
spectrum, we investigated the differential spectrum of the                            telescopes (Brown et al. 2002; Le Mouélic et al. 2012;
annuli, ΔI/F. Using the transects, we identify the northern and                       Ádámkovics et al. 2016; Corlies et al. 2021). As methane is the
southern edge of the annuli, i.e., where the smooth featureless                       primary absorber in Titanʼs atmosphere at IR wavelengths, we
atmosphere transitions to the deviation of the annuli and back                        used a methane opacity proﬁle from Rannou et al. (2016) to
again. We use these bounds to remove the deviation from the                           estimate the altitude of the EQA and the NPA. We calculated
curve, and we then interpolate the data spanning the resulting                        the altitude where the atmosphere becomes opaque in two
hole. We use a straight line (ﬁrst-degree polynomial) to                              ways. First, we determined the altitude, H, where the sum of
interpolate the data. We have investigated using a second- and                        the weighted (with Gauss coefﬁcients) effective optical depth

                                                                                  5
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                                     Kutsop et al.




Figure 5. Averaged differential spectrum of the EQA from LS = 345° to 15° (ﬂybys T044–T072). The differential spectra shown with the cyan curve are produced
from the difference between model spectra with typical haze abundance and model spectra where the haze abundance is increased at 100–130 km by 10%.




Figure 6. The averaged differential spectra of the NPA (blue) and the EQA (orange) from LS = 55° to 65° and from LS = 345° to 15°, respectively. The purple line
indicates the average median altitude of Model A and Model B from our triangulation efforts of the NPA (Section 2.4). The green curves show the modeled altitudes
where the atmosphere is opaque when averaged over the different terms of the k-correlated description. Solid green curve: the altitudes where the average column
opacity is equal to 1. Dashed green curve: the average altitude where the column opacity is 1. Yellow, gold, and brown are the 50th, 85th, and 98th percentile values of
the difference between neighboring pixels across all ﬂybys, respectively. Features in the differential spectra are signiﬁcant if they are outside these bounds. The gray
area shows the range of expected minimum altitudes based on the detection at ∼1.2 μm with > P98% and the detection at ∼2.3 μm with ≈P85%.


(following Pollack & McKay 1985) calculated with four k-                                  In order to determine the altitude of a feature, we must ﬁrst
correlated coefﬁcients where t (z ) = åi4= 1 wi ti (z ) is equal to 1                  determine whether the difference of the feature from the rest of
(Goody et al. 1989); this is the dotted green curve in Figure 6.                       the observation is signiﬁcant. To determine the signiﬁcance of a
Second, we determined the average H as the weighted altitude                           feature, we used a similar approach outlined in McCord et al.
(with Gauss coefﬁcients) H = åi4= 1 wi Hi calculated from the                          (2008 and references therein). We found the ΔI/F between
altitudes Hi where the effective τi(z) are equal to 1; this is the                     neighboring pixels in our mosaics and determined their nth
solid green curve in Figure 6. We believe that the second                              percentile values, Pn%. We found that our 125 mosaics all had
method is more appropriate for spectraltimetry.                                        similar Pn% values, and so we opted to use the average across all


                                                                                   6
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                            Kutsop et al.




Figure 7. T095, 2013-10-14, LS= 49°. 7. (A) Our triangulated model of the NPA (orange) and the contemporaneous VIMS observation (black and white) projected
onto Titan as seen from the side. (B) The projection of the modeled annulus onto Titan (orange) ﬁtting our observation of the annulus (blue) in Lat–Lon space.

mosaics, P̄n%. We suggest that a feature i is signiﬁcant (meaning                 which we refer to as Model A and Model B, respectively. The
that its ΔI/F is outside the range of ΔI/F between neighboring                    characteristics of each model are shown in Table 1.
pixels that come about via noise) if DI F (i ) > P¯98% (DI F ). A                    The true/physical location of the annuli exists along a cone
feature with DI F (i ) > P¯85% (DI F ) is considered to be a                      connecting the observations of the annuli with Cassini
positive detection as well, but it is considered with scrutiny. We                (Figure 7(A)). We used the location of the inﬂection points
chose P̄85% and P̄98%, as these percentiles represent a statistical               on the transects to deﬁne one point on a vector extending from
signiﬁcance similar to 1σ and 2σ, respectively, for normal and                    the position of Cassini at the time of observations of the
nonnormal distributions. As can be seen in Figure 6, the                          transect to and through the annulus. We interpolated between
differential spectra of the annuli are signiﬁcant from the                        the inﬂection points using a sinusoidal ﬁt. This produced a
background up to the methane absorption channels at around                        higher-density vector ﬁeld to ﬁnd intersections between cones.
∼1.4 μm. At these wavelengths Titan is essentially opaque, at                     The width of the deviation of the transects and the longitudinal
∼100 km. At the next series of absorption channels at around                      bin width of the transects produce uncertainty in the location of
2.4 μm, the differential brightness falls within P̄85%, suggesting an             the center of the annuli. To compensate for this, we performed
upper limit of the minimum height of ∼130 km. Regardless of                       a Monte Carlo simulation with 75 runs, where we randomly
how bright annuli are at other wavelengths, they all fail our criteria            varied the location of our interpolated points by half of the
beyond 2.6 μm. In images of the annulus at these wavelengths, it                  average latitudinal width of all transect deviations (3°) and by
is very difﬁcult to discern the annulus from the rest of the                      the longitudinal separation between interpolated points (2°). In
atmosphere. By 3.0 μm it becomes difﬁcult to interpret the                        each run of the Monte Carlo simulation, we found the closest
spectraltimetry. We suggest that this is not due to the annulus                   point between any and all two vectors from multiple ﬂybys in a
being below a certain opaque altitude, but rather because of the                  longitude bin with a width of 2°.
low signal-to-noise ratio at these wavelengths due to lack of solar                  For each ﬂyby where we detected the NPA, we projected our
illumination and because the scattering intensity has diminished                  modeled annuli onto Titan from the perspective of Cassini
following Beer–Lambertʼs law. Titanʼs stratosphere is between                     (Figure 7(B)). We then used the χ2 goodness-of-ﬁt statistic to
∼50 and ∼300 km, while the main haze layer is between ∼100                        determine how well our model matched our observations. The
and ∼400/500 km (Hörst 2017). Depending on the ﬂyby, the                          χ2 goodness-of-ﬁt statistic is given by
differential spectrum of the NPA is identical to the EQA, and so
                                                                                                                  N
we can conclude that it is at the same altitude for part of a Titan                                                    (fi - F (l i ))2
                                                                                                          c2 = å             s2
                                                                                                                                          ,               (1 )
year. In Section 4.2, we will discuss the altitude implications of                                               i=1
the change in differential spectra of the NPA.
                                                                                  where f is the latitude of the projected annuli, λ is longitude of the
                  2.4. Altitude from Triangulation                                projected annuli, F is the function for the expected latitude, and σ2
                                                                                  is the variance related to the measurement error for f. As we
   We triangulate the location of the NPA by ﬁnding the
                                                                                  expected, we saw a lower goodness of ﬁt to our observations the
intersection between multiple observations of the annulus at
                                                                                  farther in time a ﬂyby was from either Model Aʼs group (T092–
different times with similar illumination geometries. We
produced two models where the opposite hemisphere was                             T096) or Model Bʼs group (T103–T107), suggesting that the
illuminated between the two groups. The ﬁrst used observa-                        annulus changes over time. We used a grid search, to investigate
tions from ﬂybys T092–T096 (LS = 46°. 71 to LS = 51°. 2), and                     what changes were necessary in order to match the annulus we
the second used observations from ﬂybys T103–T107                                 predicted with our observations. We varied several parameters,
(LS = 58°. 37 to LS = 62°. 79). These produced modeled annuli,                    including tilt and altitude, but most of these could be neglected.

                                                                              7
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                                  Kutsop et al.




Figure 8. (A) Latitudinal transect of the mosaic for the T100 ﬂyby [2014-04-07, LS = 55°. 17]. One primary (the NPA) and two secondaries are shown. The
nomenclature is consistent with Figure 7. The annuli are highlighted at 1155 nm, where a deviation in the otherwise smooth curve is found. Similar convex deviations
are found in the 894 nm transect, and concave deviations are found in the 498 nm transect. (B) 1155 nm mosaic with manifold correction. The colored lines identify
the annuli and are consistent with panel (A). (C) Context using VIMS-ISS map (Seignovert et al. 2019).

The two signiﬁcant parameters we varied were the radius of the                       features are nearly identical to the primary annuli spectrally and
ellipse (preserving the eccentricity) from −250 to +250 km and                       morphologically, the difference being that they are less intense
the direction of the annulusʼs normal vector with respect to the                     and distinct. Secondary annuli 2 and 3 (using the nomenclature
subsolar longitude from −180° to +180°. Changing the radius of                       from Figure 4) occur within 30° of the equator, and sometimes
the annulus also changed the altitude.                                               both secondary annuli and the equatorial annuli can be seen
                                                                                     together. Secondary annuli 1 and 4 occur at roughly 55° north and
                                                                                     55° south, respectively. It is interesting that we see the southern
                     3. Results and Seasonality                                      secondary annuli so low in latitude, while not observing the SPA.
   The annuli appear roughly sinusoidal in latitude–longitude space                  It is possible that the secondary annuli have less of a seasonal
(Figure 7), and, similar to Roman et al. (2009), we use a sinusoidal                 dependence in a similar way to the EQA while the SPA and the
ﬁt to determine the tilt with respect to the rotation axis of Titanʼs                NPA are seasonal features. The secondary annuli are more
solid body (θNP), the axial precession west of the subsolar longitude                sporadic than the primary annuli. This could be due to a more
(fSol), and the average latitude (l̄ ). The average values from our                  sensitive viewing geometry requirement. Alternatively, it may
sinusoidal ﬁt coefﬁcients (with 95% conﬁdence bounds) to the                         indicate that the increase in haze is especially low and that the
EQA over all our observations are θNP = 2°. 78 ± 1°. 63, fSol =                      variation in abundance can vary enough to make the secondary
110°. 27 ± 54°, and l¯ = -3 . 13  2 . 09. The average values for                  annuli detectable or not between observations.
the sinusoidal ﬁt coefﬁcients of the NPA (with 95% conﬁdence
bounds) over all our observations are θ = 2°. 05 ± 1°. 27;
fSol = 162°. 38 ± 85°. 83, and l¯ = 66 . 39  2 . 01. Our results for                                3.2. When Do We See the Annuli?
determining the altitude of the EQA and the NPA suggest that the
annuli are at least 100 km in altitude, which puts our ﬁndings in                       In Figure 4, we plot the location of the mean latitude against
line with Lorenz et al. (2001).                                                      the time of the observation in units of degrees of planetocentric
                                                                                     longitude of the Sun relative to vernal equinox, LS. The
                                                                                     observations of the annuli are grouped largely by their mean
                         3.1. Secondary Annuli                                       latitude and perception of repeat observations. We found that
   We found four secondary annuli (Figure 8). We have grouped                        the presence of the annuli is not correlated with the presence of
them together based on common latitude ranges (Figure 4). These                      tropospheric clouds, as can be seen in Figure 2(c).

                                                                                 8
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                           Kutsop et al.

   The annuli are seen over a wide range of viewing
geometries. But a closer inspection of each observation
reveals that the vast majority (95%) occur at incidence
angles, i, between 9° and 91° and at emission angles, e,
between ∼4° and ∼57°. We mapped where VIMS observed
Titan within these ranges, suggesting that if an annulus was
in the area when these observations were taken, there would
be a higher probability of detection. In Figure 4 gaps in the
detection of the annuli, in particular the EQA starting at
LS = 45°. 21 and the NPA starting at LS = 63°. 77, correlate
well to the areas we suggest would not show an annulus,
whether it is present or not.
   We see the EQA during the entirety of the Cassini mission and
secondary annuli for essentially the same length of time. Our
earliest observation of the NPA comes from ﬂyby T085 [2012-07-
24, LS = 35°. 61], where the annulus can be seen as a solitary arc
extending off the disk of Titan (Figure 4, red dot), and we ﬁrst see
the NPA as the band on the disk of Titan in ﬂyby T087, [2012-11-
13, LS = 39°. 16]. This is the same time that Cassini begins its
inclined series of orbits to focus on Titanʼs north polar region. It is
therefore difﬁcult to determine whether the NPA is a seasonal
feature brought about by the increase in insolation or is always
present and we simply lacked the ability to observe it. The absence
of the SPA gives us a clue in breaking the degeneracy, leading to
our preferred hypothesis that the NPA is a seasonal feature.
   We know that the SPA was observed prior to Cassini’s arrival.
Yet despite the existence of south polar observations with good
resolution and good viewing geometry (e.g., −32°. 1–0°LS relative
to vernal equinox), we did not observe the SPA. From this we
conclude that the SPA is absent owing to the changes in season. It
was predicted by Roe (2012) that the SPA would vanish as
summer moved toward fall in the southern hemisphere. Titanʼs
stratosphere has already been shown to be periodic as the north–
south asymmetry switches between either hemisphere being bright
or dark. We propose that the NPA is similarly periodically
symmetric to the SPA. Since the SPA is seasonal, and we also see
the NPA change with the seasons as discussed below, we propose
that the NPA appears according to the season. We suggest that if
the NPA was present before our earliest detections, it was not
signiﬁcantly earlier on a seasonal timescale.
   If we assume that the SPA has a similar seasonal timescale to
the NPA, and given our detection of the NPA range from 2012-
07-24 to 2017-09-11 (LS = 39°. 17–93°. 3), then we could expect
the SPA to be present as early as 1998 January and as late as 2003
March (Ls ≈ 220°–280°). If we continue to extrapolate, we should
expect to see the SPA again starting around 2027 February. If,
however, Earth-based observations of Titan are taken before 2027
and the SPA is observed, this would imply that the NPA should
have been visible as early as 2006 December (LS = 315°. 29), or
Cassini ﬂyby T021. As this is not the case, an observation of the             Figure 9. The colored solid lines in each ﬁgure are spline ﬁts to the
                                                                              corresponding data with an R2 ≈ 0.95. These curves are nonphysical and are
SPA before 2027 would imply that the southern stratosphere does               intended to facilitate comprehension. The three vertical gray bars in each ﬁgure
not mirror the northern stratosphere. This might be unsurprising,             are located at 498, 894, and 1155 nm. These are the wavelengths used for our
as asymmetries are already observed between the atmospheres of                transects and mosaics. (A) The EQA differential spectra taken from three time
the north/south hemispheres, such as the lag in the expected onset            periods of equal length centered about LS = 15° (the peak of the blue sinusoid
                                                                              in Figure 10(A)); blue is from ﬂybys T028–T061, green from T061–T082, and
of methane clouds in the north following equinox (Rodriguez                   red from T082–T105. (B) The NPA differential spectra taken from three time
et al. 2009, 2011; Turtle et al. 2018).                                       periods of equal length spanning the range of observable ﬂybys; blue is from
                                                                              T088–T095, green from T098–T108, and red from T119–nT262. (C)
                                                                              Differential spectra produced by creating a modeled spectra where the haze
                   3.3. Changes during Cassini                                abundance at a range of altitudes is increased by 10% and then subtracting a
                                                                              modeled spectrum with typical haze abundance.
   The differential spectra of the EQA and NPA change as the
season progresses (Figures 9 and 10). The shape of the                        in Figure 9(A). The amplitude, however, changes a great deal
differential spectrum of the EQA remains the same as                          and has a very symmetric pattern, with the peak amplitude
evidenced by the positions of the peak brightness and darkness                centered about 10° after the vernal equinox (Figure 11(A)). The

                                                                          9
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                                  Kutsop et al.




Figure 10. (A) Blue: maximum value of ΔI/F of the EQA differential spectrum. Orange: minimum value of ΔI/F of the EQA differential spectrum. Green: maximum
value of ΔI/F of the NPA differential spectrum. Purple: minimum value of ΔI/F of the NPA differential spectrum. (B) Blue: the wavelength of the maximum ΔI/F
of the EQA differential spectrum. Orange: the wavelength of the minimum ΔI/F of the EQA differential spectrum. Green: the wavelength of the maximum ΔI/F of
the NPA differential spectrum. Purple: the wavelength of the minimum ΔI/F of the NPA differential spectrum. The opacity of the data points correlates to the relative
weight of the data. Data are weighted according to the number of transect detections. For details on the curves, see Table 2.




                                                                                      nature of the spectral changes for the NPA is different from that
                                                                                      of the EQA. The entire spectrum of the NPA becomes darker as
                                                                                      we approach summer solstice. The bright part of the spectrum
                                                                                      moves redward, while the dark component shows a small shift
                                                                                      to bluer wavelengths (Figures 9(B) and 10(B)).
                                                                                         Our triangulated models indicate that the semimajor axis of
                                                                                      the NPA increases as Titan approaches northern summer,
                                                                                      potentially reaching a maximum size of about 1500 km around
                                                                                      LS = 72° (Figure 11(A)). This may be due to the meridional
                                                                                      circulation transporting the haze that had been lofted by the
                                                                                      onset of the pole-to-pole circulation cell, toward the south pole.
                                                                                      As the annulus increases in radius, it also increases in altitude,
                                                                                      assuming a constant tilt (Figure 11(B)). This may imply that the
                                                                                      change in altitude we suggest based on the changing spectra of
                                                                                      the NPA is caused by the NPA increasing in size, rather than
                                                                                      the NPA being lofted higher itself. After LS = 72°, the size of
                                                                                      the annulus seems to decrease, returning to its original size just
                                                                                      before summer solstice.

                                                                                                                3.4. The Tilted Pole
                                                                                         In a review of open questions at Titan following Cassini,
                                                                                      Nixon et al. (2018) ask, “Is the [azimuthal and tilt offset of the
                                                                                      stratosphere] ﬁxed in magnitude and direction, or does it
                                                                                      wander on seasonal or longer timescales?” In Figures 12, 13,
                                                                                      and 14 we show the change in the tilt and the offset as a
                                                                                      function of time in solar and inertial reference frames. We note
Figure 11. Both panels show the results of the χ2 minimization routine to             that the data for the EQA and NPA are contiguous but not
determine the change in parameters of the NPA. Model A was constructed                continuous; however, where appropriate we analyzed and
using ﬂybys T094–T098, and Model B was constructed with ﬂybys T101–
T105. (A) Semimajor axis of the NPA as a function of LS. The curves are               interpreted the data as if they were one continuous set.
Gaussian ﬁts, with (B) median altitude of the NPA vs. LS. The change in                  Achterberg et al. (2011) suggested that a longer time base of
altitude is a product of the change in semimajor axis.                                stratospheric tilt offset may reveal the offset to be ﬁxed in an


                                                                                10
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                                         Kutsop et al.




Figure 12. Top: the R.A. offset of the normal vector of the annuli from the solid-body rotation axis in an inertial reference frame (ICRF). The R.A. of Titanʼs solid-
body rotation axis is R. A. = 39°. 48 (Stiles et al. 2008). The green sinusoid through the ﬂyby has a goodness-of-ﬁt metric of R2 = 0.411. Bottom: the decl. offset of the
normal vector of the annuli from the solid-body rotation axis in an inertial reference frame (ICRF). The decl. of Titanʼs solid-body rotation axis is R.A. = 83°. 43 (Stiles
et al. 2008).




Figure 13. Top: western azimuthal offset of the normal vector of the annulus from the subsolar vector. The yellow squares are taken from Roman et al. (2009),
Table 1, Column (6). The black crosses are taken from Tokano (2010), Figure 6(b). Bottom: magnitude of the polar offset of the normal vector of the annulus from the
solid-body rotation axis (north pole). The red line is taken from Tokano (2010), Figure 5(a).




                                                                                    11
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                          Kutsop et al.

                                                                                        causes a simultaneous change in the components AAMx and
                                                                                        AAMy, which manifests itself as a 180° shift of the tilt. Tokano
                                                                                        (2010) did not emphasize that AAMx and AAMy also depend
                                                                                        on the zonal wind u. If u is very large (as is the case in Titanʼs
                                                                                        stratosphere), a sign change of v may have little inﬂuence on
                                                                                        AAMx and AAMy according to Equations (1) and (2) of
                                                                                        Tokano (2010). From a comparison with Achterberg et al.
                                                                                        (2008), the zonal wind in the stratosphere produced by
                                                                                        Tokanoʼs GCM is too weak, and this could bias the seasonal
                                                                                        variation in the AAM tilt angle. It is probable that the weight of
                                                                                        v of the equatorial AAM is too large relative to that of u. An
                                                                                        additional reason may be an underestimation of the axial
                                                                                        component of AAM, AAMz, in the calculation of the tilt angle
                                                                                        by Tokano (2010). According to Tokano & Neubauer (2005),
                                                                                        AAMz strongly varies with season owing to seasonal reversal
                                                                                        of the zonal wind direction in the lower troposphere. However,
                                                                                        since the stratospheric superrotation in their GCM is greatly
Figure 14. The R.A. and the decl. offset of the normal vector of the annuli from        underestimated, it is likely that the seasonal variation in AAMz
the solid-body rotation axis. Diamonds are data from the EQA, and circles are           relative to the annual-mean AAMz is too large. In reality, the
data from the NPA. The colors correspond to the time of observation.                    mean AAMz may be much larger, while its seasonal variation
                                                                                        due to tropospheric winds remains unchanged. This implies
inertial (star-ﬁxed) reference frame. We determined the R.A.                            that the lack of a systematic seasonal tilt angle of the
and decl. difference between the normal vector of the annulus                           stratosphere in the observational data is evidence of the relative
and the Titan solid-body rotation axis (Stiles et al. 2008)                             seasonal invariance of the axial AAM due to perennial
aligned to the X-axis of the International Celestial Reference                          stratospheric superrotation. Our results, however, do seem to
Frame (ICRF; Charlot et al. 2020). In Figure 12, we see that,                           match Tokanoʼs (2010) predictions for the NPA between 40°
instead of being ﬁxed, it appears that the stratospheric tilt offset                    and 60° past the vernal equinox. This might indicate that the
in an inertial frame is a function of the Titan season. We                              GCM from Tokano (2010) could predict the rotation of the
modeled the oscillation of the R.A. of the stratosphere about                           stratosphere by using observations such as those presented in
the north pole fNPRA, as a function of LS,                                              this work to further constrain the model.
                                                                                           Tokano (2010) shows in his Figure 4 that the tilt of the
                                    L       2p ⎞                                        stratosphere with respect to the north pole should vary from 0°
       f NPRA (Ls) = f + M sin ⎛⎛360 s + y⎞       ,                         (2 )        to 8° biannually, with the maximum tilt peaking around 30° of
                               ⎝⎝   P     ⎠ 360 ⎠                                       planetocentric longitude of the Sun after summer and winter
                                                                                        solstice. We ﬁnd that the tilts of the EPA and NPA stay within
where f is an offset in R.A. of the north pole of Titan, M is the
                                                                                        the bounds predicted by Tokano (2010), but with much greater
amplitude, P is the period, and ψ is the phase shift. We used the
                                                                                        variability (Figure 13). We found a weak correlation between
bisquare weights method to minimize the inﬂuence of outliers.                           the variability and differences in the longitudinal extent of
We found that the coefﬁcients (with 95th percentile bounds) for                         some of our observations, which may indicate inaccuracies in
the EQA and NPA combined are f = −0°. 9071 ± 0°. 38,                                    our tilt estimations. The upper limit of our tilt measurements
M = 1°. 38 ± 0°. 35, P = 148°. 3 ± 36°. 3, and ψ = −146° ± 18°. 5.                      follows the same path predicted by Tokano (2010), decreasing
The curve in Figure 12 ﬁts our data with an R2 = 0.411.                                 at the same rate between −60° and 65 ° relative to the vernal
   Additionally, we ﬁnd that the tilt offset in a solar reference                       equinox. That the rate of change predicted by Tokano (2010) is
frame (i.e., the subsolar longitude) is also a function of the                          consistent with what was observed over the Cassini mission
season (Figure 13). We assume that the stratosphere is a triaxial                       indicates that the forces that change the tilt were accurately
ellipsoid, although hemispheric asymmetries of the atmo-                                modeled.
spheric angular momentum (AAM) are possible and perhaps
necessary to explain our observations (Tokano 2010). Since                                                       4. Discussion
each group can be characterized by a ﬁrst-degree polynomial
and considering both would require a more complicated model,                               We propose that the annuli are features of enhanced aerosols
for now we will describe their progression separately. We ﬁnd                           that are corralled/conﬁned by gradients in the winds, or at least
that the EQA rotation axis precesses westward at a rate                                 mark locations where aerosols are in a stable quasi-equilibrium
dfSol/dt = 0.21 ± .23(+W°/LS), with a starting point at vernal                          for an extended time period. Given their location, circumglobal
equinox fSol(0) = 99°. 22 ± 7°. 51. We ﬁnd that the NPA                                 nature, and seasonal behavior, we suggest that the annuli are
rotation axis precesses westward at a rate dfSol/dt =                                   caused by the meridional cell circulations, with different
3.85 ± .99(+W°/LS), with a starting point at vernal equinox                             mechanisms for the NPA and EQA, respectively.
fSol(0) = −71°. 3 ± 66°. 41. This contradicts the prediction from
Tokano (2010) showing a time series of the angular distance                                        4.1. Detailed Explanation of the Spectra
from the subsolar longitude that moves eastward as the season
                                                                                                                4.1.1. 300–500 nm
progresses (Figure 13).
   Tokano (2010) explains the movement of the equatorial                                   The difference spectrum is negative for wavelengths shorter
AAM as a result of the seasonal reversal of the meridional                              than ∼700 nm, indicating that the annuli are darker than the
circulation. The reversal of the meridional circulation cell                            surrounding atmosphere (Figure 5). The local darkening could

                                                                                   12
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                              Kutsop et al.

be produced by multiple scattering in a higher optical depth                                      4.1.3. 700–2000 nm
environment removing net-outbound photons, or by absorption
                                                                             The backscatter component of weak-particle/aerosol scatter-
of inbound photons that never get a chance to scatter from the            ing is not particularly intense owing to the lower extinction
low-altitude gases. Rayleigh scattering is the dominant                   efﬁciency of the relatively small scatterers. The extra brightness
mechanism of scattering by particles much smaller than the                produced by the annuli is therefore lost in the brightness
wavelength of light (x = 2πr/λ = 1, for particles of radius r             produced by the highly reﬂective surface. Not only is the
and wavelength λ) and has an intensity dependence of                      surface brightness (I/F ≈ 10−1) two to three orders of
I ∝ 1/λ4. In the lower atmosphere scattering by gaseous                   magnitude greater than the ΔI/F for the annuli (∼10−3), but
nitrogen and methane dominates; since both have a kinetic                 the surface is much more variegated than the atmosphere,
radius of r ≈ 0.2 nm, the lower atmosphere is highly back-                making the detection of the small difference between the annuli
scattering. The annuli, however, if predominantly made of haze            and the rest of Titan very difﬁcult. Instead, we utilize the
particles, will have effective radii around 1.0–2.0 μm (Lavvas            methane absorption channels, including ∼900 and ∼1150 nm
et al. 2010), which leads to aerosol/particle scattering at these         used for our detections. The majority of photons are absorbed
wavelengths with only a small backscatter component (in the               at these wavelengths. Those that are not absorbed are scattered
limit of spherical particles this is Mie scattering). Outbound            by the haze. As before, the increased optical depth at a higher
photons scattered by the gases that interact with the annuli must         altitude increases backscatter and reduces multiple scattering,
contend with a higher optical depth, which is not encountered             producing a localized bright feature.
outside the annuli. Alternatively, the haze or a companion                   At longer wavelengths, several complementary effects
species may have absorption features around 500 nm. In this               produce a slow drop-off in differential brightness. Primarily,
scenario photons are absorbed more effectively in the higher              the intensity of the scattered light drops off as 1 − e− λ
optical depth environment of the annuli, producing the contrast           following Beer–Lambertʼs law. Additionally, the longer-
                                                                          wavelength methane absorption channels are more effective,
we observe.
                                                                          requiring a lower density of methane to achieve opaqueness.
   To test this hypothesis, we use a model that predicts the
                                                                          This means that the atmosphere becomes opaque at higher
photometry in UV and visible (e.g., Rannou et al. 2016). This
                                                                          altitudes as wavelength increases. At ∼2.4 μm the annulus
model is tuned to ﬁt Titanʼs spectra near the equator, as
                                                                          becomes indiscernible from the rest of the atmosphere. The
observed in a near-nadir viewing (e.g., spectra from VIMS                 atmosphere becomes essentially opaque at ∼100 km for 1.2 μm
observation CM_1477457253_1 in ﬂyby TA). We then                          and at ∼130 km for ∼2.4 μm, suggesting that the annulus has a
successively increase by 10% the haze opacity in different                minimum altitude of 100–130 km. Altitude constraints for the
layers in order to assess the effect of this change on the                NPA and EQA are discussed further in the next section.
outgoing intensity relative to the reference model, producing a           Finally, solar intensity drops off at longer wavelengths, leading
modeled differential spectrum. As can be seen in Figure 5, an             to a signal-to-noise ratio so low that it is not possible to discern
increase of 10% haze opacity at 100–130 km produces                       the annulus in the images or the deviation in the transects.
differential spectra nearly identical to the averaged differential
spectra of the EQA from 345° to 15° LS. A more complete
                                                                                      4.2. Origin and Evolution of the Annuli
radiative transfer model is needed to determine whether the
mechanism of obscuration only arises from an increase in haze                        4.2.1. Formation of the North Polar Annulus
opacity at a speciﬁc altitude or changes in scattering and                   The NPA occurs at the boundary where the north polar hood
absorption properties are also needed to explain the data in ﬁne          was seen until it dissipated during Titanʼs spring (Le Mouélic
detail.                                                                   et al. 2012). At that time, a bright ring of haze and mist whose
                                                                          opacity smoothly rises from about 51° to 68° north was seen by
                                                                          VIMS bordering the north polar hood (Grifﬁth et al. 2008).
                        4.1.2. 500–700 nm                                 While this band is not spectrally or morphologically consistent
   Beginning with 500 nm and progressing redward, the                     with our observation of the NPA, it is circumglobal at certain
brightness of the annulus starts to increase (Figure 5). Nitrogen         wavelengths and was estimated to be at about 40 km of altitude
and methane move further into the Rayleigh scattering regime,             according to radiative transfer modeling. This band is evidence
but the intensity from the lower atmosphere, optically thick              of the potential for haze to build up at the border of the north
                                                                          polar hood. When the north polar hood began to break up, this
because of these gases, decreases as I ∝ 1/λ4. Meanwhile, the
                                                                          band did not immediately dissipate, leaving behind a zone clear
haze moves toward Rayleigh scattering, which leads to an
                                                                          of haze and clouds near the pole (Le Mouélic et al. 2012;
increase in its backscatter component. Together, this means that
                                                                          Rannou et al. 2012).
the brightness from the lower atmosphere decreases while the                 We propose that the NPA is another manifestation of haze
brightness at the top of the annulus increases. The increased             collecting at the mixing boundary encircling the north polar
optical depth of the annuli produces more backscattering than             hood. In this scenario, haze is produced in the summer
the rest of the haze throughout the atmosphere. The higher                hemisphere and transported to the winter hemisphere. This
altitude also presents a shorter two-way path through the                 process forms a polar hood (Rannou et al. 2004; Larson et al.
atmosphere, reducing multiple scattering events that could                2015) and gradually builds up an annulus at the boundary of a
remove the photons from the beam of VIMS. The increase in                 polar jet or a sharp change in the polar vertical velocities.
localized backscatter and the reduction of multiple scattering            Titanʼs polar hoods contain high concentrations of ethane,
lead to the annuli being brighter than the surrounding                    which is consistent with their location above the arctic circle
atmosphere.                                                               (Mayo & Samuelson 2005). The polar annuli, however, exist


                                                                     13
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                              Kutsop et al.

equatorward of the arctic circle and may not be able to maintain          which must await constraint by higher-resolution observations
condensed ethane aerosols. As spring comes to the north, the              than are currently available. Alternatively, as upwelling at the
ethane begins to deplete from the polar hoods, entraining                 north pole peaks at summer solstice, the increase in velocity
accumulated haze with it. Because the polar annuli lack ethane,           may act to move the bounds of NPA farther south. It could also
they persist longer, until they are eventually transported away           be indicative of the upwelling moving to lower latitudes. This
by the summer meridional circulation.                                     could be a sign of how Titan moves from pole-to-pole
   We also investigated whether the NPA (and the other annuli)            circulation at solstice to equator-to-pole circulation at the
could be explained by methane clouds, as nearly global arcs of            equinoxes.
cloud have been observed with ISS and VIMS (Turtle et al.
2018). We ﬁnd that there is no obvious correlation between the                        4.2.3. Evolution of the Equatorial Annulus
presence of clouds and the presence of the annuli. In
Figure 2(c), we see that the annuli are present while no clouds              We proposed in Section 4.1 that the dark component of the
are visible. We have also seen that the presence, location, and           annuliʼs differential spectra (shorter than 600 nm) is produced
morphology of clouds when seen by VIMS or ISS do not                      by the haze in the annulus obscuring the bright Rayleigh
correlate with the presence, location, or morphology of the               scattering from the lower atmosphere. If the haze in the annulus
annuli. We suggest that the annuli and clouds seen in Turtle              is more abundant, it will have a higher optical depth, which will
et al. (2018) may be related by the same (or a similar)                   make it more efﬁcient at blocking the illumination from below.
mechanism, which acts to corral them into global circular                 Meanwhile, the bright component of the differential spectra
features.                                                                 (near-IR) of the annuli is produced by the scattering from the
                                                                          top of the annuli. Once again, more haze in the annuli leads to a
                                                                          higher optical depth, which facilitates more backscattering.
           4.2.2. Evolution of the North Polar Annulus                       We propose that the dark and bright components of the EQA
   The differential spectrum of the NPA changes as Titan                  spectra in Figures 9(A) and 10 change intensity in unison from
moves further toward northern summer. We see that the entire              a change of optical depth. The change in optical depth is due to
spectrum gets darker and the bright component shifts redward              changing haze abundances as a function of season. The EQA is
(Figures 9(B) and 10). We note the similarities between the               at its brightest and darkest around the vernal equinox. From this
differential spectra of the NPA over time (Figure 9(B)) and the           we infer that the haze at the equator increases until it reaches a
modeled differential spectra where the haze abundance is                  maximum abundance in the annuli around the vernal equinox,
increased at several altitudes (Figure 9(C)). As the altitude of          and then it begins to decrease. It appears that the EQA becomes
the modeled differential spectra increases, the dark component            least distinguishable from the rest of the atmosphere around the
short of 700 nm decreases while shifting spectrally very little.          summer and winter solstices. This suggests that the haze
We see the same behavior in the NPA differential spectrum                 abundance in the EQA is similar to the rest of the stratosphere.
over time. The dark components of the differential spectra of
the NPA at LS = 40°–50°, LS = 53°–63°, and LS = 78°–88°                               4.2.4. Formation of the Equatorial Annulus
most closely resemble the dark components of the modeled                     If the haze abundance is responsible for the change in
differential spectra at 0–30 km, 20–50 km, and 100–130 km,                intensity of the EQA, then the haze reaches its maximum
respectively. We propose that these similarities are indicative of        abundance at the vernal equinox, which coincides with the
the NPA increasing in altitude as the season progresses. The              circulation pattern of upwelling at the equator from the equator-
bright component of the modeled differential spectra also                 to-pole meridional circulation cell circulation. We suggest that
shows some similarities to the observed spectra, but not as               the haze becomes suspended at the convergence of the north
much as the dark component. A more complete radiative                     and south cells. As the circulation transitions from equator-to-
transfer model is needed to fully quantify how the spectrum of            pole to pole-to-pole, the haze that had been accumulating and
the NPA changes in the way that it does.                                  suspended begins to disperse. What is curious, however, is that
   This scenario is consistent with the expected change in the            we do not see the EQA go away; rather, it persists all the way
meridional circulation (Battalio et al. 2022). As spring turns to         up to and including at summer solstice. This might suggest that
summer, the two meridional circulation cells that converge in             some part of the haze that had been built up during fall
an upwelling event at the equator transition to a single cell,            remained at the equator. Perhaps this is because of some area of
which rises in the north and subsides in the south                        quiescence brought about by gravity or pressure waves, or
(Tokano 2011). We propose that haze originally transported                maybe the circulation in the middle stratosphere is just not
to the north polar hood during the winter and spring months is            effective. In this case horizontal circulation may happen in the
lofted higher by the summer upwelling at the north. It is                 troposphere and near the stratopause, ﬂowing in opposite
possible that the band observed at the boundary of the north              directions, while the center of the stratosphere remains
polar hood did not simply dissipate, but rather moves to higher           unmixed and undisturbed.
altitudes while also changing in abundance, size, fractal
dimension, and/or single scattering albedo.
                                                                                 4.3. Using the Annuli to Track the Stratosphere
   The increase in the semimajor axis of the NPA in Figure 11
may be related to variations in the polar temperature. As the                Given their altitude (>∼130 km), the annuli are tracers of
temperature gradient decreases during summer, the weakening               Titanʼs zonal stratospheric superrotation and seasonal mer-
strength of the jet and polar vortex could result in larger-              idional circulations, as well as the orientation of the strato-
amplitude waves dipping farther south (Newman et al. 2011;                sphere with respect to the solid body. The annuli can be
Lora et al. 2015, 2019) This could also lead to an increase in            observed from Earth-based observatories, allowing us to extend
the ellipticity of the annuli. Determining the ellipticity of the         the understanding of the stratosphere and meridional circula-
annuli requires additional terms in the sinusoidal function,              tion a full Titan year. The distribution of secondary annuli

                                                                     14
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                              Kutsop et al.

bears a resemblance to zonal wind patterns produced by                      angles facilitates this process or is a product of the process.
barotropic waves in Titanʼs stratosphere based on shallow                   Additionally in Figure 14, the points seem to cluster around R.
water modeling (Luz & Hourdin 2003). The secondary annuli                   A. = −2° and decl. = 2°, suggesting that the stratosphere
may offer accessible observables of Titanʼs barotropic waves.               rotates around an axis offset from the north polar axis. The
                                                                            three peaks in the bottom panel of Figure 12 between −15 and
                                                                            60° relative to the vernal equinox might indicate that this axis is
               4.3.1. The Tilt of Titan’s Stratosphere
                                                                            orbiting around the north polar axis and is presenting itself in
   The tilt of Titanʼs atmosphere, relative to the tilt of the solid        our data as a second-order waveform.
surface, is one of the remaining questions left open after the
end of the Cassini mission (Nixon et al. 2018). The mechanism
                                                                                               4.3.2. Testing Rotation Rates
of the tilt and superrotation of Titanʼs stratosphere are not well
understood. Tokano (2010) proposes that the tilt is the result of              The large difference in the rotation rate of the EQA and the
thermal tides and is only possible if atmospheric waves perturb             NPA (0.14 ± .4 versus 3.35 ± .54) would suggest that the
the circulation. From Figure 13 we see that the observed                    equatorial and polar stratospheres rotate independently of one
temporal variation in the azimuthal offset of the normal vector             another. GCMs show that the different latitudes experience
of the annulus from the subsolar vector does not follow the path            different zonal wind velocities in the stratosphere (Newman
predicted by Tokano (2010). The most likely explanation is that             et al. 2011; Lebonnois et al. 2012; Lora et al. 2015). The
the weak zonal wind in his GCM affects all three components                 simulations show, however, that the peak wind speed is in the
of AAM in such a way that the azimuth of the tilt turns out to              winter hemisphere, with the slowest speed experienced in the
be wrong most of the time. Another possible explanation is that             summer hemisphere. We suspect that the EQA and NPA
the tilt is not directly responding to thermal tides, unlike the            precession rates are not decoupled. In the top panel of
assertion by Tokano (2010). Tokano (2010) discarded the                     Figure 13 we can see that from −60° to 10° relative to vernal
possibility of mixed Rossby-gravity waves as a cause of                     equinox the EQA does not precess. Then, from 10° to 40° after
westward migration of the tilt because the superrotating winds              the vernal equinox, the precession rate of the EQA accelerates.
in the upper stratosphere should turn the phase speed of such               If the EQA precession rate were to remain on its course after
waves relative to the surface to eastward. However, it is worth             LS = 40°, we might expect it to look exactly like the precession
mentioning that Battalio & Lora (2021b) predicted the presence              rate of the NPA. We therefore suggest that the EQA and NPA
of mixed Rossby-gravity waves and equatorial Rossby waves                   precess together at the same rate. Furthermore, we suggest that
in their GCM. Therefore, the likelihood that Rossby waves                   the EQA and NPA are nearly parallel with each other and move
affect the precession of the tilt of the annuli cannot be fully             in lock step. A more detailed investigation of nearly
dismissed. Achterberg et al. (2008) propose that the tilt                   contemporaneous observations of the EQA and NPA will be
facilitates the vertical transport of angular momentum to                   needed to verify this hypothesis.
balance the heat ﬂow and insolation at low latitudes by
feedback between the circulation and the heating.
                                                                                                     5. Conclusion
   The azimuthal position of the stratosphere shows some
correlation with the subsolar longitude. We suggest that the                   The NPA, the EQA, and the SPA are stratospheric seasonal
orientation and position of the stratosphere are strongly                   features that provide insight into Titanʼs circulation mechan-
correlated with the seasons when examined in an inertial                    isms and patterns. The annuli are unique among Titan features
reference frame (Figures 12 and 14). There is some indication               in their reﬁned morphology (as compared to the polar hoods
that the stratosphere reorientates itself throughout the year               and the north–south asymmetry), their predictability (as
around a centroid offset by a few degrees from the north polar              compared to clouds), and their scale, which allows for Earth-
axis. Assuming that this is correct, we suggest that the centroid           based observations. From the differential spectra of the annuli,
about which the stratosphere reorientates itself is itself rotating         we conclude that the annuli are areas of increased haze. Using
around the north polar axis. In this scenario the centroid is a             spectraltimetric techniques, we determined the minimum
seasonal feature (Figure 12, top), while the stratosphere around            altitude of the annuli to be between 100 and 130 km.
the centroid may be an orbital or diurnal feature with a much               Triangulation of the NPA is consistent with the spectraltimetry
shorter period (Figure 12, bottom). We suggest that our data set            with a modeled median altitude of H = 217 ± 2 km. We
supports the proposed mechanism for the tilt suggested in                   tracked the EQA and the NPA over the course of the Cassini
Achterberg et al. (2008).                                                   mission and investigated their changes in position and spectra.
   In Figure 12 we see how the azimuthal position of the                       The annuli are easily observable features that can be used to
stratospheric tilt (the angle in the X-Y plane) is oriented with            track the evolution of Titanʼs stratosphere on seasonal and
respect to the north polar axis in an inertial (star-ﬁxed)                  yearly timescales. The annuli have already been observed with
reference frame. The tilt follows a sinusoid, with a maximum                Keck (Roe et al. 2002), and with an expected spatial resolution
deﬂection of about 2° clockwise at Ls ≈ 25°. We propose that                of 200 km and a spectral range of 600–2300 nm, the annuli
this sinusoid indicates that the orientation of the stratosphere is         should be readily observable with JWST (Nixon et al. 2016).
dependent on the season rather than on the solar position. We               The timing and position of the annuli are controlled by Titanʼs
propose that the stratosphere rotates around the north polar axis           circulation and dynamic and chemical mixing boundaries. An
in response to the subsolar latitude and true anomaly. These                understanding of the annuli, including how they form and what
results support and expand on the interpretation of Achterberg              they are made of, provides insight into the driving forces that
et al. (2011), which suggested that the stratosphere is ﬁxed in             control Titanʼs stratosphere.
an inertial reference frame. These results imply that the                      The EQA is visible for the entirety of the Cassini mission.
mechanism for the tilt proposed in Achterberg et al. (2008) is              The spectra of the EQA increase in absolute contrast to a
seasonally dependent and reorientation at different azimuthal               maximum of about 10° of LS after the vernal equinox. We

                                                                       15
```

<!-- PDF_PAGE: 16 -->

## PDF page 16

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                            Kutsop et al.

suggest that this is evidence of the haze increasing in                    seasons when examined in an inertial reference frame
abundance in the EQA as Titan approaches the vernal equinox.               (Figures 12 and 14). There is some indication that the
We propose that the EQA is formed by upwelling at the equator              stratosphere reorientates itself throughout the year around a
where the north and south equator-to-pole meridional circula-              centroid offset by a few degrees from the north polar axis.
tion cells converge around the vernal equinox. When the                    Assuming that this is correct, we suggest that the centroid about
meridional circulation cells transition to a single pole-to-pole           which the stratosphere reorientates itself is itself rotating
cell, most of the haze accumulated at the equator is transported           around the north polar axis. In this scenario the centroid is a
poleward. Given the continuous observation of the EQA, we                  seasonal feature (Figure 12, top), while the stratosphere around
propose that some haze must remain in the middle stratosphere,             the centroid may be an orbital or diurnal feature with a much
where wind speeds are possibly low and meridional circulation              shorter period (Figure 12, bottom). We suggest that our data set
is stagnant. This is supported by the meridional stream                    supports the proposed mechanism for the tilt suggested in
functions shown in Figure 7 of Lebonnois et al. (2012). We                 Achterberg et al. (2008).
see that at LS = 123°. 8 (the nearest time to summer solstice) the            In order to analyze the annuli, we developed (1) a
meridional circulation crosses the equator at ∼300 km.                     mosaicking routine with aspects of subpixel superresolution,
Meanwhile, in the lower stratosphere near 100 km, the                      (2) a technique for correcting the striping in the VIMS-Vis data
meridional circulation is limited to a small cell spanning 20°             set, and (3) an empirical technique for improving the contrast in
south and 20° north. In this area we might expect that this small          a Titan observation. Any data used in this work or any of the
cell continuously circulates any haze that had been lofted                 above-described techniques can be made available by contact-
during the convergence events seen at LS = 8°. 9 and 179°. 3.              ing the ﬁrst author.
   We did not observe the SPA despite many opportunities
where the viewing geometry would have allowed detection.                     The authors thank the reviewers for the in-depth comments
Because it was detected in 1999–2001, but not during the                   and suggestions, which greatly improved the paper. We thank
Cassini mission, we propose that the SPA is a seasonal feature.            Phil Nicholson and Tom Loredo for their advice and insights.
Thus, if we assume that the polar stratospheres and the                    T.T. is supported by Deutsche Forschungsgemeinschaft (Ger-
circulation are mostly symmetric between the hemispheres,                  man Research Foundation, DFG), grant TO269/5-1. This work
then we can also assume that the NPA is a seasonal feature.                was supported by the NESSF grant 80NSSC18K1319.
   The NPA becomes detectable about 35° of LS after the vernal
equinox and remains present up to the end of the Cassini
mission. The spectrum of the NPA is nominally identical to the                                    Appendix A
EQA, but it changes in ways the EQA does not. The darkening                                    VIMS-Vis Destriping
of the spectra of the NPA is consistent with our models that                  To remove the striping, we reconstruct what the offsets
suggest that the altitude of the NPA is increasing throughout              introduced by the readout electronics would have been. We ﬁrst
our observations. The NPA shares its morphology and location               sample each line of data that runs perpendicular to the striping
with several other north polar features, including the north               on the sample (Figure A1). We apply a smoothing spline based
polar hood. The timing/sequence of the other polar features                on the LOESS (Appendix B) method to each line. We use a
and the NPA leads us to suggest that the NPA is haze, which                second-degree polynomial to model the local regression and a
was on the border of the north polar hood. As the north pole               span of 2/3. The span is the fraction of the total number of data
enters summer, the ethane in the north polar hood falls out,               points used for calculating the smoothed value. We subtract the
carrying the haze along with it, leaving the NPA as remnant                smoothed spline from the data and approximate the spikiness of
haze occupying the stratosphere on the arctic circle.                      the data from the stripes in the lines (Figure A2). Stitching the
   An important result of this work is the tracking of the tilt and        lines back together, with each of their respective smoothed
azimuthal offset of the annuli, and by extension the strato-               curves removed, produces an approximation of the stripes
sphere. We found that the tilt of the stratosphere is in the range         Figure A3). We know that the offset applied is constant for
predicted by Tokano (2010) but does not follow any distinct                each sample. We approximate this constant value by taking the
pattern. We found that the azimuthal position of the strato-               median of data along the line dimension (Figure A4). This
sphere does not match the prediction from Tokano (2010). The               gives us a single line of data, which we then expand to the
azimuthal position of the stratosphere shows some correlation              original image size, producing a mask for the initial offset
with the subsolar longitude. We suggest that the orientation and           (Figure A5). To destripe the data, we subtract Figure A5 from
position of the stratosphere are strongly correlated with the              A1 to produce a VIMS image with greatly improved clarity.




                                                                      16
```

<!-- PDF_PAGE: 17 -->

## PDF page 17

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                           Kutsop et al.




                                   Figure A. Schematic showing the VIMS destriping routine described in Appendix A.

                          Appendix B                                            twice as wide as Menrva crater, one of the largest features on
                       Manifold Correction                                      Titanʼs surface. Doing this 100 times through bootstrapping
                                                                                provides a good estimate of the underlying parameters for the
   The goal of the manifold technique is to remove global scale
                                                                                brightness trend on Titan. This technique is ﬂexible in many ways,
brightness trends from our observations. We want to remove
                                                                                including in the number of data points sampled, how many Monte
effects due to illumination geometry and scattering from the haze
                                                                                Carlo runs are made, the wavelengths corrected, and the polynomial
while preserving compositional and physical properties like grain               degree and weighting of the LOESS function.
size and roughness. The predicted brightness due to these effects
resembles a curvy surface, which we call a “manifold.” To
remove effects like viewing geometry and scattering from the                                                  Appendix C
VIMS data, we subtract the manifold from the observations. This
technique is analogous to ﬂat-ﬁeld corrections, which are
ubiquitous throughout astronomy.
   First, we produce subpixel-style superresolution mosaics and
project them into an orthographic view. We have to produce a
manifold for each channel in VIMS because the amount and type of
scattering in the atmosphere vary with wavelength. For each
channel we randomly sample 100 pixels from the mosaic. We ﬁt a
locally estimated scatterplot smoothing function (LOESS) to the
data points, where the independent variables are the row and
column location of the pixel in the mosaic, and the dependent
variable is the pixel brightness. The LOESS method ﬁts a low-
degree polynomial to a subset of the data at each point in the range
of the data set. The polynomial is ﬁtted using weighted least
squares, giving more weight to points near the point whose
response is being estimated and less weight to points farther away
(Savitzky & Golay 1964; Cleveland 1979). We then repeat this
process 100 times, each time selecting 100 random pixels from the
mosaic, for each channel. This produces 100 manifolds, which we
take the average of to produce the ﬁnal manifold.
   By sampling only 100 pixels from our mosaic (which are at least
400,000 pixels in total), we nullify all small-scale variations in data,        Figure C. Zoom-in of the modeled atmospheric spectra and the modeled
like those caused by surface composition. The randomly sampled                  annulus spectra from Figure 3. The difference between the two spectra is less
pixels will be separated by about 730 km on average, which is                   than the line width at certain wavelengths.

                                                                           17
```

<!-- PDF_PAGE: 18 -->

## PDF page 18

```text
                                                                                                                                                                                                      The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                               Appendix D


                                                                                                  Table 1
                                                                                  Details of Every Detection of the Annuli

                                                                                                                                                           Tilt
                                                                                                                                                Average   from      Prime      Subsolar
                                                                 Planetocentric    Average     Average        Average    Central     Average     Pixel    North    Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence    Emission        Phase    Longitude    Latitude    Scale     Pole     Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)       (deg)          (deg)      (deg)       (deg)      (km2)    (deg)   (deg, E+)   (deg, E+)     Bins

                                                                                                North Polar

     T079    TI158    2011-12-     3 583.241 3   377 079 149.2       28.41         54.69         68.8          19.26     160.27       66.96      77.37                                       1
                     13T20:11:23
     T087    TI174    2012-11-     973.650 52     406074195          39.17         86.21         49.6         131.11     −54.23       65.28      82.11                                       4
                     13T10:22:08
     T088    TI175    2012-11-     1 014.693 4   407 451 486.1       39.66         82.54        43.81         126.19     −68.06       68.85     109.08                                       2
                     29T08:56:59
     T089    TI181    2013-02-     1 978.042 3    414338262          42.19         84.54        35.21          117       −46.05       71.35      157.9    4.76    −148.86       42.8         11
                     17T01:56:35
     T090    TI185    2013-04-     1 400.076 1   418 470 277.6       43.7          86.63        37.13         108.86     −36.78       67.95      59.31    2.63     36.93      −133.18        17
                     05T21:43:30
     T091    TI190    2013-05-     970.052 55    422 602 441.7       45.21          63.7        45.17         105.01     −113.93      69.63     120.39    1.45     145.81      −26.03        8
                     23T17:32:55
18   T092    TI194    2013-07-     963.786 73    426 734 573.9       46.7          76.96         29.5          97.53     −73.07       68.79     100.32    1.04     48.07      −125.51        17
                     10T13:21:47
     T093    TI195    2013-07-     1 399.697 9    428111849          47.2          81.42        20.11          90.41     −49.04       66.1       61.42    1.64     52.23      −121.96        25
                     26T11:56:22
     T094    TI197    2013-09-     1 396.765 3    432243903          48.71         72.86        21.35          83.1      −44.12       67.57      44.02    2.13     38.62      −137.35        28
                     12T07:43:56
     T095    TI198    2013-10-     960.758 29     434998654          49.7          63.33        36.86          76.56         95.32     67        86.54    2.67      30.1      −147.04        35
                     14T04:56:27
     T096    TI199    2013-12-     1 400.035 4   439 130 545.8       51.19         61.83        23.53          66.13     −77.15       65.5       43.51    3.73      24.2      −154.76        36
                     01T00:41:19
     T097    TI200    2014-01-     1 399.810 4    441885648          52.18         62.89        19.95          59.94     −92.06       65.69      78.62    3.61     55.66      −124.39        36
                     01T21:59:41
     T098    TI201    2014-02-     1 235.470 6    444640425          53.17         71.86        34.72          51.74     −34.07       66.41      98.89    3.59     48.03      −133.19        36
                     02T19:12:38
     T099    TI202    2014-03-     1 499.789 9    447395274          54.16         60.41        29.14          43.04     −66.15       69.17     134.98    3.38     68.39      −113.99        22
                     06T16:26:47
     T100    TI203    2014-04-     963.449 34     450150141          55.16         61.96         38.6          34.27     −51.81       68.04      58.05    3.35     42.69      −140.85        28
                     07T13:41:14
     T101    TI204    2014-05-     2 991.882 4    453615202          56.4          87.01        29.86          109.5         99.55    66.71      110.8    1.19    −142.66     −142.04        11
                     17T16:12:15
     T102    TI205    2014-06-     3 658.690 5    456370172          57.39         84.11        34.45         104.14         70.52    66.15      79.78    0.85    −152.88     −153.39        17
                     18T13:28:25
     T103    TI206    2014-07-     5 103.283 1   459 124 925.1       58.38         74.41        37.43          99.95         30.52    66.73      83.72    3.03    −138.85     −140.55        24



                                                                                                                                                                                                           Kutsop et al.
                     20T10:40:58
     T104    TI207    2014-08-     964.103 3      461880616          59.36         77.77        31.85         103.26         95.2     64.14      44.64    1.65    −127.36     −130.01        17
                     21T08:09:09
```

<!-- PDF_PAGE: 19 -->

## PDF page 19

```text
                                                                                                                                                                                                     The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from       Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North     Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole      Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)    (deg, E+)   (deg, E+)     Bins

     T105    TI208    2014-09-     1 401.501 6    464635466          60.34         62.97        42.2         110.48      59.9       65.89      39.78    0.28      −42.95      −46.78        18
                     22T05:23:19
     T106    TI209    2014-10-     1 013.168 4    467390497          61.32           82        42.51         119.77      112.2      66.38      60.52    1.02      −101.1     −106.05        13
                     24T02:40:30
     T107    TI210    2014-12-     980.572 19    471 522 462.4       62.79          94.3       35.54         125.85     153.02      67.83      89.29    0.06      −0.03       −6.83         5
                     10T22:26:35
     T119    TI235    2016-05-     969.201 5     515 825 744.7       78.42         83.96        42.9         121.75     104.13       65        54.45    1.58       70.48      103.41        10
                     06T16:54:37
     T120    TI236    2016-06-     974.555 42    518 580 444.9       79.39         85.08       34.14         114.78     110.67      64.5       71.57    1.47       78.43      110.11        8
                     07T14:06:17
     T121    TI238    2016-07-     975.407 36    522 712 770.8       80.84         82.05       36.62         112.29      97.54      63.17      64.4     0.66        69         98.88        16
                     25T09:58:23
     T122    TI239    2016-08-     1 698.358 2   524 089 920.8       81.34         88.69       29.92         111.84     106.87      63.95     143.08    1.13       25.46       54.66        11
                     10T08:30:53
     T123    TI243    2016-09-     1 775.659 2    528221887          82.77         67.77       45.93          112        61.43      64.83      69.29    0.62       146.1      173.41        8
                     27T04:16:59
     T124    TI248    2016-11-     1 585.323 9   532 353 423.8       84.21         64.59       49.77         112.65      50.88      61.26     107.16    1.15      −40.36      −15.05        5
                     13T23:55:56
19
     nT261   TI261    2017-02-     186 795.47    540 609 095.4       87.09         52.84        36.6          78.03     −66.31      66.57     109.41    1.87        88        107.36        14
                     17T13:10:26
     nT262   TI262    2017-02-     220 499.08    540 632 554.7       87.1          62.33       35.93          85.05     −22.58      65.17      133.8     3.3       69.19       94.66        25
                     17T19:41:26
     nT283   TI283    2017-07-     264 321.55    552 966 722.3       91.4          48.91       23.94          55.67     −15.45      66.54     155.08    3.63        73         76.7         19
                     10T13:50:53
     nT288   TI288    2017-08-     194 993.55    555 699 920.5       92.36         55.89       30.81          71.73      31.21      66.47     118.68    0.57       83.39       80.23        11
                     11T05:04:11
     nT292   TI292    2017-09-     119 734.92    558 428 758.2       93.3          69.87       29.57          84.68     −68.04      67.29      72.39    3.53     −103.57     −114.72        14
                     11T19:04:49

     Flyby   Rev      CA UTC        CA Alti-      CA Ephe-       Planetocentric    Average    Average        Average    Central    Average    Average     Tilt     Prime      Subsolar     # of
                                   tude (km)      meris Time     Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Pixel     from     Meridian   Longitude   Longitude
                                                                  the Sun (LS)      (Deg)      (Deg)          (Deg)      (Deg)      (Deg)      Scale    North      Offset      Offset      Bins
                                                                                                                                               (km2)     Pole    (Deg, E+)   (Deg, E+)
                                                                                                                                                        (Deg)

                                                                                                Equatorial

     T00A    TI00A    2004-10-     1 174.330 6    152076669         297.26         40.84       26.45          19.97    −127.87      −7.54      70.95    1.53      106.02      −98.09        8
                     26T15:30:05
     T00B    TI00B    2004-12-     1 192.669 9   156 209 959.6      299.03         44.16       20.53          56.83     −52.61      −0.53      56.37    0.12     −141.42      −61.56        6
                     13T11:38:15
     T003    TI003    2005-02-     1 578.974 6   161 722 737.3       301.4         33.54       24.86          13.25    −137.36      −5.31      46.56    4.19      110.84      −89.49        9


                                                                                                                                                                                                          Kutsop et al.
                     15T06:57:53
     T004    TI005    2005-03-     2 403.893 6   165 571 580.1      303.06         37.27       25.61          15.82     −140.4      −4.8       28.03    5.63      115.77      −86.38        10
                     31T20:05:16
```

<!-- PDF_PAGE: 20 -->

## PDF page 20

```text
                                                                                                                                                                                                    The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from      Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North    Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole     Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)   (deg, E+)   (deg, E+)     Bins

     T005    TI006    2005-04-     1 359.185 5    166950500         303.65         54.11       23.32          57.12     −32.98      −3.1       28.12    0.35     51.89       131.46        8
                     16T19:07:16
     T006    TI013    2005-08-     3 660.731 8   177 972 881.9      308.36         48.76       23.57          52.66     −27.62      −0.9       41.36    0.65    −117.04      −42.08        7
                     22T08:53:38
     T007    TI014    2005-09-     83 235.56     179 337 751.7      308.95         47.63       24.07          51.93     −31.65      −1.42      74.69    2.34    −135.32      −64.31        8
                     07T04:01:28
     T008    TI017    2005-10-     1 352.876 1   183 744 988.8      310.81         32.39       24.59          23.54     −138.2      −5.36      65.17    3.34     120.95      −98.26        11
                     28T04:15:25
     T009    TI019    2005-12-     10 411.572     188895630         313.01           25         24.2          27.13     −42.06      −1.93      44.49    2.17    −113.75      −69.27        10
                     26T18:59:26
     T010    TI020    2006-01-     2 043.032 2   190 597 350.9      313.73         43.96       24.51          37.79    −151.46      −5.65      24.62    2.61     118.01     −113.56        11
                     15T11:41:26
     T011    TI021    2006-02-     1 812.311 4   194 300 783.6      315.29         29.83       25.75          18.17     −14.14      −1.86     112.98    3.79     −101.6      −86.97        13
                     27T08:25:18
     T012    TI022    2006-03-     1 949.351     195 998 820.6      316.02         51.77       24.99          65.01    −142.81      −3.17      32.31    6.01     141.32     −121.09        11
                     19T00:05:55
     T013    TI023    2006-04-     5 735.424 2   199 701 513.3      317.59         36.53       20.29          39.7      −10.78      −2.56      16.94    0.22     80.93        64.59        6
                     30T20:37:28
20
     T014    TI024    2006-05-     1 879.178 5    201399556          318.3         63.07       33.15          91.51    −119.98      −3.06     105.32                                       1
                     20T12:18:11
     T015    TI025    2006-07-     1 906.160 9   205 104 112.1      319.86         47.05       25.06          61.53      10.59      −4.36      93.78    0.84     −76.12     −122.92        6
                     02T09:20:47
     T016    TI026    2006-07-     950.385 67     206799991          320.6         52.87       30.76          59.39      2.03        1.9       79.66     4.1     −89.86      −54.3         7
                     22T00:25:26
     T017    TI028    2006-09-     999.713 19     210932276         322.33         46.94       23.72          62.92      7.47         0        47.34    1.66    −100.42      −66.74        5
                     07T20:16:51
     T018    TI029    2006-09-     960.198 92    212 309 993.5      322.92         35.17       41.28          64.49     −8.17       −4.97      73.94                                       4
                     23T18:58:48
     T019    TI030    2006-10-     980.047 47     213687072          323.5         46.57       40.81          65.03      5.44       −1.83      81.26    0.92     −98.11      −65.7         5
                     09T17:30:07
     T020    TI031    2006-10-     1 029.512 2    215063952         324.07          36.2       45.98          71.81     −4.31       1.87       74.86    1.75     90.75        122.4        5
                     25T15:58:07
     T021    TI035    2006-12-     1 000.284 7    219195756         325.81         58.62       34.79          67.41      21.76       1.3       95.85     3.2     −80.38      −50.71        5
                     12T11:41:31
     T025    TI039    2007-02-     86 205.694    225 401 139.6      328.38         24.79       55.63          69.36     126.85      −3.17     112.45                                       2
                     22T07:24:34
     T026    TI040    2007-03-     980.607 62     226763405         328.95         30.35       49.26           62       127.31      −5.45      68.68    3.95     50.28      −105.51        7
                     10T01:49:00
     T027    TI041    2007-03-     1 010.089 8   228 140 672.1      329.53         24.87        46.2          53.24     135.84      −6.95     115.47     0.7      0.02      −156.42        7
                     26T00:23:27
                                                                                                                                    −6.15                                   −111.34


                                                                                                                                                                                                         Kutsop et al.
     T028    TI042    2007-04-     991.146 26    229 517 944.8       330.1          28.7       42.45          45.81     128.52                  74      3.05     45.75                     10
                     10T22:58:00
     T029    TI043    2007-04-     981.179 96     230895243         330.67         30.56       32.86          37.25     129.21      −4.42      89.58    4.43     34.81      −122.91        8
                     26T21:32:58
```

<!-- PDF_PAGE: 21 -->

## PDF page 21

```text
                                                                                                                                                                                                    The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from      Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North    Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole     Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)   (deg, E+)   (deg, E+)     Bins

     T030    TI044    2007-05-     959.584 86     232272663         331.25         31.07       27.95          29.29     133.41      −2.6       82.7     6.44     54.11      −104.23        12
                     12T20:09:58
     T031    TI045    2007-05-     2 299.088 5   233 650 380.2      331.81         35.39       32.69          27.47     127.37      −4.86      29.45     3.4     50.88      −107.98        11
                     28T18:51:55
     T032    TI046    2007-06-     965.396 65     235028836         332.39         32.08       25.94          16.15     131.63      −3.3       58.83    1.56     68.97       −90.22        9
                     13T17:46:11
     T033    TI047    2007-06-     1 932.782 6    236408451         332.95         26.72       19.52          13.69     137.71      −3.14      23.28    4.17     60.02       −99.2         11
                     29T16:59:46
     T034    TI048    2007-07-     1 331.914 5   238 079 545.2      333.65         38.73       23.51          58.26      43.25      −1.15      25.18                                       3
                     19T01:11:20
     T035    TI049    2007-08-     3 324.615 5   241 814 020.8      335.19         26.82       17.15          26.87      159.3      −1.97      26.5     2.64     74.92      −113.88        10
                     31T06:32:36
     T036    TI050    2007-10-     973.489 1     244 572 228.1      336.33         35.94        26.9          32.54     134.06      −3.48      91.04    3.53     65.33      −123.77        10
                     02T04:42:43
     T037    TI052    2007-11-     999.652 91     248705310         338.03         36.22       16.89          40.66     142.73      −2.75      49.56    4.55     65.53      −125.14        8
                     19T00:47:25
     T038    TI053    2007-12-     1 298.834 1   250 085 274.9      338.59         31.41       19.37          44.55     158.82      −3.48      17.42     1.9     73.28      −117.31        9
                     05T00:06:50
21
     T039    TI054    2007-12-     969.950 23     251463540         339.17         46.04       32.55          50.13     134.06      −5.92      93.88    0.07     120.53      −70.43        8
                     20T22:57:55
     T040    TI055    2008-01-     1 014.118 2   252 840 684.6      339.72         38.06       21.11          55.98     156.25      −6.22      14.92    5.77      59.6      −132.02        9
                     05T21:30:19
     T041    TI059    2008-02-     999.872 23    256 973 591.9      341.42         46.94        35.5          66.19     139.27      −6.33      82.74    5.16     59.89      −133.31        7
                     22T17:32:07
     T042    TI062    2008-03-     999.447 82     259727333         342.54          53.8       23.75          72.98     141.59      −6.17      24.96    0.69    −132.47       32.88        7
                     25T14:27:48
     T044    TI069    2008-05-     1 399.989 7   265 235 137.1      344.79         44.41       51.07          89.45     146.68       0.1      105.68                                       3
                     28T08:24:32
     T046    TI091    2008-11-     1 105.099 8   279 005 787.9      350.33         51.51       44.65          81.13     −98.42      0.17       90.41    3.03     165.25      −38.85        6
                     03T17:35:23
     T047    TI093    2008-11-     1 023.290 7   280 382 252.9      350.89          49.5        45.2          84.76    −101.92      −7.91      57.73                                       3
                     19T15:56:28
     T048    TI095    2008-12-     960.510 53     281759210         351.44         46.96       47.03          77.68    −101.48      −5.38      85.57                                       3
                     05T14:25:45
     T049    TI097    2008-12-     970.886 78     283136457         351.99         42.61       45.63          71.26    −104.01      −1.53      89.57     3.9     154.79      −51.43        6
                     21T12:59:52
     T050    TI102    2009-02-     967.024 19    287 268 717.8      353.64         42.42       49.62          70.66    −101.92      −0.9      118.46    0.88       0          152.1        5
                     07T08:50:52
     T051    TI106    2009-03-     962.880 46    291 401 082.3      355.29          44.2       53.04          75.21     −98.58      −5.04     127.55                                       3
                     27T04:43:36
                                                                                                                                    −3.46


                                                                                                                                                                                                         Kutsop et al.
     T053    TI109    2009-04-     3 598.575 5   293 458 911.1      356.14         37.63       56.47          68.02      62.41                110.16                                       3
                     20T00:20:45
     T054    TI110    2009-05-     3 242.262 8   294 836 120.9      356.68          24.4       55.23          62.78      47.87      0.88      106.01    4.72     −40.04      −73.18        7
                     05T22:54:15
```

<!-- PDF_PAGE: 22 -->

## PDF page 22

```text
                                                                                                                                                                                                    The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from      Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North    Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole     Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)   (deg, E+)   (deg, E+)     Bins

     T055    TI111    2009-05-     965.623 78    296 213 267.1      357.24         23.79       49.97          52.13      29.55      −1.55     194.95    1.44     −60.71      −94.47        7
                     21T21:26:41
     T056    TI112    2009-06-     967.717 2     297 590 466.1      357.79         33.54       48.82          44.21      49.64      −2.47     163.46    3.56     −57.21      −91.59        10
                     06T20:00:00
     T057    TI113    2009-06-     955.214 66     298967621         358.32         29.42       44.67          36.87      35.92      −1.9       91.92    3.41     −53.53      −88.53        10
                     22T18:32:35
     T058    TI114    2009-07-     966.055 3     300 344 709.3      358.86         27.86       38.62          29.9       51.37      −1.94      49.59    4.16     −60.44      −96.09        11
                     08T17:04:03
     T059    TI115    2009-07-     956.582 69    301 721 709.1      359.41         18.39       27.89          22.52      30.89      −4.11      36.5     3.98     −65.48      −101.8        6
                     24T15:34:03
     T061    TI117    2009-08-     960.837 48    304 476 763.6        0.5          26.14       32.34          12.95      30.41      −3.18      22.72    2.76     −62.07      −99.44        13
                     25T12:51:37
     T062    TI119    2009-10-     1 299.964 9   308 608 650.2       2.14          31.55        30.4          10.5       10.19      −3.17     129.31    3.12     −67.91     −107.04        11
                     12T08:36:24
     T063    TI122    2009-12-     4 847.710 8   313 851 859.8       4.19          42.19        23.3          46.81     145.57      −1.91      55.65    1.98     40.56       −70.32        9
                     12T01:03:14
     T064    TI123    2009-12-     951.817 66    315 231 484.6       4.73          40.78       22.31          44.29     147.41      −2.15      30.55    3.33     34.49       −76.37        8
                     28T00:16:58
22
     T065    TI124    2010-01-     1 074.516     316 609 901.9       5.27          27.33       38.94          44.17     113.07      −2.02      56.58    4.93     40.89       −70.26        9
                     12T23:10:36
     T066    TI125    2010-01-     7 486.939 1   317 989 795.7       5.81           38.7        25.2          43.37     140.44      −0.31      75.38    0.89       0        −111.05        9
                     28T22:28:50
     T067    TI129    2010-04-     7 437.318 4   323 754 719.8       8.06          21.12       21.04          13.61      36.75      −3.68      51.67     2.1     −65.1      −111.75        12
                     05T15:50:54
     T068    TI131    2010-05-     1 398.006 7   327 597 926.3       9.56          31.66        25.9          31.55     133.37      −2.03     118.34    2.52     46.14       −77.6         10
                     20T03:24:20
     T069    TI132    2010-06-     2 042.953      328976853          10.09         31.33       35.09          31.05      117.8      −1.26      71.79    3.84     50.77       −73.12        11
                     05T02:26:27
     T070    TI133    2010-06-     878.567 51    330 355 728.9       10.62         33.88       25.04          32.24     146.06       −2        36.56    3.05     48.82       −75.25        11
                     21T01:27:43
     T071    TI134    2010-07-     1 004.078 5   331 734 231.1       11.16         36.43       23.32          35.03     143.59      −2.41      76.2     2.08     51.21       −73.12        9
                     07T00:22:45
     nT136   TI136    2010-08-     417 252.75    334 805 769.2       12.35         48.94       23.86          56.38      78.24      −1.47      178.1     2.1      0.01       −42.77        6
                     11T13:35:03
     T072    TI138    2010-09-      8 177.99     338 625 587.1       13.81         27.09       30.66          24.09     114.76      −3.06      17.12    0.54      0.01      −125.96        9
                     24T18:38:41
     T074    TI145    2011-02-     3 651.096 8   351 317 117.2       18.69         33.33       29.51          31.36      16.38      −4.23     150.65    1.71     −70.15     −124.13        11
                     18T16:04:11
     T075    TI147    2011-04-     10 052.942     356461305          20.64         31.88       32.02          18.21     127.75      −2.7       77.18    1.54     58.84       −92.75        11
                     19T05:00:39
                                                                                                                                    −4.94                        −60.78     −127.14


                                                                                                                                                                                                         Kutsop et al.
     T076    TI148    2011-05-     1 872.807 1   358 167 290.5       21.28         31.33       22.26          42.38      36.87                 24.59    3.48                               9
                     08T22:53:44
     T077    TI149    2011-06-     1 358.992 4   361 866 786.6       22.68         33.57       31.88          21.17     126.38      −3.95      51.2     2.13     76.03      −104.96        10
                     20T18:32:00
```

<!-- PDF_PAGE: 23 -->

## PDF page 23

```text
                                                                                                                                                                                                    The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from      Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North    Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole     Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)   (deg, E+)   (deg, E+)     Bins

     T078    TI153    2011-09-     5 821.661 5   369 067 871.9       25.4          52.85        30.6          75.85      46.71      −5.86      57.86    4.22     −34.8      −136.56        6
                     12T02:50:06
     T079    TI158    2011-12-     3 583.241 3   377 079 149.2       28.41         33.33       22.99          23.23     144.81      −4.65      16.84    3.15      69.1       −102.1        12
                     13T20:11:23
     T081    TI160    2012-01-     31 130.577    381 202 854.1       29.96         34.31       38.73          28.17    −140.56      −3.89      73.62    3.36      74.4       −100.7        12
                     30T13:39:48
     T082    TI161    2012-02-     3 803.185     382 913 062.9       30.59         53.04       21.81          63.85      35.22       −4        38.83    3.63     −35.13     −123.89        6
                     19T08:43:17
     T083    TI166    2012-05-     953.990 5     390 921 076.9       33.58         32.06       27.24          21.93     134.49      −4.56      71.74    1.92     50.92      −108.15        11
                     22T01:10:11
     T084    TI167    2012-06-     959.347 58    392 299 706.8       34.09         37.44       37.38          28.4      119.55      −4.26     103.69    4.02     60.22       −99.09        11
                     07T00:07:21
     T085    TI169    2012-07-     1 012.449 8   396 432 253.9       35.61         26.95       20.75          33.04     155.36      −3.89      15.92    1.45     71.49       −89.43        11
                     24T20:03:07
     T086    TI172    2012-09-     956.550 55     401942205          37.66         34.94       32.96          47.11     136.86      −3.54     128.58    0.64     49.84      −113.24        8
                     26T14:35:38
     T087    TI174    2012-11-     973.650 52     406074195          39.17         31.65       38.91          52.92     146.05      −4.43      75.31    2.43     53.44       −111.4        10
                     13T10:22:08
23
     T088    TI175    2012-11-     1 014.693 4   407 451 486.1       39.66         29.98       28.63          58.32     157.04      −3.17      19.94    1.95     55.57      −109.87        6
                     29T08:56:59
     T089    TI181    2013-02-     1 978.042 3    414338262          42.19         40.74       38.25          68.48     113.47      −5.54     181.36                                       2
                     17T01:56:35
     T090    TI185    2013-04-     1 400.076 1   418 470 277.6       43.7          41.84       40.49          73.85     136.55      −7.03      49.82    0.06       0        −170.11        5
                     05T21:43:30
     T091    TI190    2013-05-     970.052 55    422 602 441.7       45.21          35.4        46.7          80.58     140.05      −6.98      94.14                                       1
                     23T17:32:55
     T101    TI204    2014-05-     2 991.882 4    453615202          56.4          40.69       38.55          67.89     −15.37      −6.79     130.88                                       3
                     17T16:12:15
     T102    TI205    2014-06-     3 658.690 5    456370172          57.39         36.77        44.8          75.21     −15.73      −7.18     120.87                                       2
                     18T13:28:25
     T106    TI209    2014-10-     1 013.168 4    467390497          61.32         33.76       35.88          66.44     −4.72       −5.55      50.22                                       3
                     24T02:40:30
     T107    TI210    2014-12-     980.572 19    471 522 462.4       62.79         40.16       31.86          59.21     −7.63       −1.5       90.91    7.58     80.23        73.43        5
                     10T22:26:35
     T108    TI211    2015-01-     970.385 16     474277782          63.77         39.07       27.95          52.07     −10.92      −4.77      31.13    2.13    −100.52     −108.37        5
                     11T19:48:35
     T109    TI212    2015-02-     1 200.149 1   477 032 951.1       64.75         24.41       22.94          42.12     −1.28       6.67       13.6                                        4
                     12T17:08:04
     T112    TI218    2015-07-     10 952.196    489 528 658.1       69.17         37.48       19.64          24.85     −26.09      −4.98      42.5                                        3
                     07T08:09:50
                                                                                                                        −25.85      −7.69


                                                                                                                                                                                                         Kutsop et al.
     T114    TI225    2015-11-     11 927.57     500 665 658.9       73.1          35.63       19.64          28.11                            39.82                                       3
                     13T05:46:31
     T116    TI231    2016-02-     1 398.535 8   507 560 472.7       75.52         32.56       11.54          32.21     −25.06      −5.26      36.88                                       2
                     01T01:00:05
```

<!-- PDF_PAGE: 24 -->

## PDF page 24

```text
                                                                                                                                                                                                     The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from       Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North     Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole      Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)    (deg, E+)   (deg, E+)     Bins

                                                                                               Secondary 1

     T062    TI119    2009-10-     1 299.964 9   308 608 650.2       2.14          55.31       56.25          10.69      9.58       57.77     177.46    3.85       19.01      −20.12        10
                     12T08:36:24
     T067    TI129    2010-04-     7 437.318 4   323 754 719.8       8.06          49.14       50.53          14.42      21.25      47.94      79.58                                        3
                     05T15:50:54
     T071    TI134    2010-07-     1 004.078 5   331 734 231.1       11.16          49.9       39.91          34.91      159        32.32      84.03                                        3
                     07T00:22:45
     T076    TI148    2011-05-     1 872.807 1   358 167 290.5       21.28         46.32       51.75          46.18      34.07      47.25      64.11                                        3
                     08T22:53:44
     T079    TI158    2011-12-     3 583.241 3   377 079 149.2       28.41         43.06       52.61          18.97     142.38      52.31      55.65    1.56       60.19     −111.01        7
                     13T20:11:23
     T082    TI161    2012-02-     3 803.185     382 913 062.9       30.59         54.82        47.6          62.79      31.77      43.86      62.07                                        4
                     19T08:43:17
     T098    TI201    2014-02-     1 235.470 6    444640425          53.17         35.97       18.94          52.16    −148.08      51.55      94.63                                        4
                     02T19:12:38
     T100    TI203    2014-04-     963.449 34     450150141          55.16         33.88       14.63          34.86    −153.57      49.28      54.18    2.42       118.8      −64.74        6
                     07T13:41:14
24   T114    TI225    2015-11-     11 927.57     500 665 658.9       73.1          29.19       51.91          27.2      −3.79       39.47      87.69                                        3
                     13T05:46:31
     nT264   TI264    2017-03-     489 890.93    541 986 902.8       87.57         29.73       18.25          42.71     −28.68      51.78     197.65     2.5     −114.84      −95.99        6
                     05T11:44:03
     TI126   TI270    2017-04-     980.236 27    546 113 356.1       89.01         61.52       52.98         111.94      52.75      47.52      99.6                                         4
                     22T05:52:13

     Flyby   Rev      CA UTC        CA Alti-      CA Ephe-       Planetocentric    Average    Average        Average    Central    Average    Average     Tilt     Prime      Subsolar     # of
                                   tude (km)      meris Time     Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Pixel     from     Meridian   Longitude   Longitude
                                                                  the Sun (LS)      (Deg)      (Deg)          (Deg)      (Deg)      (Deg)      Scale    North      Offset      Offset      Bins
                                                                                                                                               (km2)     Pole    (Deg, E+)   (Deg, E+)
                                                                                                                                                        (Deg)

                                                                                               Secondary 2

     T00A    TI00A    2004-10-     1 174.330 6    152076669         297.26          43.1       23.85          19.58    −159.81      18.68      89.13                                        1
                     26T15:30:05
     T008    TI017    2005-10-     1 352.876 1   183 744 988.8      310.81         42.52       21.73          23.16    −163.23      9.42       62.06                                        3
                     28T04:15:25
     T036    TI050    2007-10-     973.489 1     244 572 228.1      336.33          53.8       26.32          32.28     136.42      18.44      89.87                                        4
                     02T04:42:43
     T037    TI052    2007-11-     999.652 91     248705310         338.03         48.99       16.04          39.58     145.28      17.64      36.84                                        2
                     19T00:47:25
     T071    TI134    2010-07-     1 004.078 5   331 734 231.1       11.16          49.9       39.91          34.91      159        32.32      84.03                                        3



                                                                                                                                                                                                          Kutsop et al.
                     07T00:22:45
     T083    TI166    2012-05-     953.990 5     390 921 076.9       33.58         23.06       27.05          22.14     130.19      11.54      70.46                                        1
                     22T01:10:11
```

<!-- PDF_PAGE: 25 -->

## PDF page 25

```text
                                                                                                                                                                                                     The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from       Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North     Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole      Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)    (deg, E+)   (deg, E+)     Bins

     T098    TI201    2014-02-     1 235.470 6    444640425          53.17         40.17        47.3          52.26    −127.24      15.44      120.7    0.55      −0.04       178.74        6
                     02T19:12:38
     T099    TI202    2014-03-     1 499.789 9    447395274          54.16         28.64       38.65          43.33    −133.43      19.24     169.07    0.17      −0.01       177.61        7
                     06T16:26:47
     T100    TI203    2014-04-     963.449 34     450150141          55.16         27.94       37.78          35.19    −135.75      15.4       61.86    0.04        0         176.46        10
                     07T13:41:14
     T107    TI210    2014-12-     980.572 19    471 522 462.4       62.79         20.67       51.04          59.5       −5.6       18.36     109.36                                        2
                     10T22:26:35
     T108    TI211    2015-01-     970.385 16     474277782          63.77         27.07       45.83          52.46     −10.82      20.07      33.53    1.65     −104.06     −111.91        9
                     11T19:48:35
     T109    TI212    2015-02-     1 200.149 1   477 032 951.1       64.75         45.53       29.88          41.8      −32.66      8.94       17.44                                        4
                     12T17:08:04
     T110    TI213    2015-03-     2 274.981 7   479 788 255.4       65.72         33.43       19.31          36.4      −18.59      17.67      13.92    4.38      −119.6     −129.61        5
                     16T12:59:36
     T111    TI215    2015-05-     2 722.237 4   484 311 090.4       67.33          49.1       23.16          65.69    −134.97      17.69      22.26    2.79      135.65      −134.3        8
                     07T20:38:58
     T112    TI218    2015-07-     10 952.196    489 528 658.1       69.17         25.45       24.21          25.75     −15.88      19.14      58.49    2.19     −104.01      −92.53        6
                     07T08:09:50
25
     T114    TI225    2015-11-     11 927.57     500 665 658.9       73.1          28.62       17.41          29.66     −38.52      −1.47      34.56                                        3
                     13T05:46:31
     T116    TI231    2016-02-     1 398.535 8   507 560 472.7       75.52         19.62       27.91          32.81     −23.67      14.43      37.39    6.03     −109.79      −73.44        5
                     01T01:00:05
     T119    TI235    2016-05-     969.201 5     515 825 744.7       78.42         27.94       35.09          53.8      −0.16       8.45       119                                          1
                     06T16:54:37
     T124    TI248    2016-11-     1 585.323 9   532 353 423.8       84.21         41.82       40.87          69.7      −55.84      2.88       44.31                                        4
                     13T23:55:56
     T125    TI250    2016-11-     3 159.020 5    533729740          84.69         47.79       37.41          68.32     −66.49      10.38      32.4     1.33     −150.44     −126.03        5
                     29T19:16:29
     TI126   TI270    2017-04-     980.236 27    546 113 356.1       89.01          41.7       35.65          72.13     −35.69      −0.04      96.66                                        4
                     22T05:52:13
     nT275   TI275    2017-05-     117 952.41    548 857 163.5       89.97         48.78       36.63          84.56      15.09      8.46      105.94    0.67      −108.8      −97.37        8
                     24T03:17:28

     Flyby   Rev      CA UTC        CA Alti-      CA Ephe-       Planetocentric    Average    Average        Average    Central    Average    Average     Tilt     Prime      Subsolar     # of
                                   tude (km)      meris Time     Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Pixel     from     Meridian   Longitude   Longitude
                                                                  the Sun (LS)      (Deg)      (Deg)          (Deg)      (Deg)      (Deg)      Scale    North      Offset      Offset      Bins
                                                                                                                                               (km2)     Pole    (Deg, E+)   (Deg, E+)
                                                                                                                                                        (Deg)

                                                                                               Secondary 3

     T003    TI003    2005-02-     1 578.974 6   161 722 737.3       301.4         10.35       20.03          13.59     −148.7     −29.78      72.81                                        3


                                                                                                                                                                                                          Kutsop et al.
                     15T06:57:53
     T011    TI021    2006-02-     1 812.311 4   194 300 783.6      315.29         15.55       32.18          18.38      7.64      −25.94     125.32                                        3
                     27T08:25:18
```

<!-- PDF_PAGE: 26 -->

## PDF page 26

```text
                                                                                                                                                                                                    The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                 Table 1
                                                                                               (Continued)

                                                                                                                                                         Tilt
                                                                                                                                              Average   from      Prime      Subsolar
                                                                 Planetocentric    Average    Average        Average    Central    Average     Pixel    North    Meridian   Longitude     # of
                                    CA Alti-      CA Ephe-       Longitude of     Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole     Offset      Offset    Longitude
     Flyby   Rev      CA UTC       tude (km)      meris Time      the Sun (LS)      (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)   (deg, E+)   (deg, E+)     Bins

     T012    TI022    2006-03-     1 949.351     195 998 820.6      316.02         31.28       41.31          64.45    −122.88     −26.71      48.22                                       3
                     19T00:05:55
     T016    TI026    2006-07-     950.385 67     206799991          320.6         45.94       28.08          59.47      4.49      −31.95      82.84                                       2
                     22T00:25:26
     T017    TI028    2006-09-     999.713 19     210932276         322.33         41.31       25.19          62.02      6.07      −23.94      21.08                                       3
                     07T20:16:51
     T018    TI029    2006-09-     960.198 92    212 309 993.5      322.92         47.84       18.47          64.22      10.72     −23.87      63.13                                       3
                     23T18:58:48
     T051    TI106    2009-03-     962.880 46    291 401 082.3      355.29         42.79       36.92          75.28    −107.51     −24.36      80.09                                       2
                     27T04:43:36
     T057    TI113    2009-06-     955.214 66     298967621         358.32         24.51       18.63          36.77      20.08     −23.85      77.08                                       1
                     22T18:32:35
     T058    TI114    2009-07-     966.055 3     300 344 709.3      358.86         35.98       24.14          30.34      58.67     −24.23      30.11                                       2
                     08T17:04:03
     T059    TI115    2009-07-     956.582 69    301 721 709.1      359.41         28.23       15.84          22.8       49.48     −23.33      22.78                                       1
                     24T15:34:03
     T061    TI117    2009-08-     960.837 48    304 476 763.6        0.5          30.99        28.2          12.2       45.19     −26.49      32.74                                       4
                     25T12:51:37
26
     T063    TI122    2009-12-     4 847.710 8   313 851 859.8       4.19          30.28        53.4          46.94     106.41      −27.7      87.95                                       2
                     12T01:03:14
     T065    TI124    2010-01-     1 074.516     316 609 901.9       5.27          29.97       62.21          45.54      90.52     −21.79      47.31                                       1
                     12T23:10:36
     T066    TI125    2010-01-     7 486.939 1   317 989 795.7       5.81          39.02       24.26          43.33     133.47     −23.01      74.77                                       2
                     28T22:28:50
     T069    TI132    2010-06-     2 042.953      328976853          10.09          34.6       25.14          30.69     139.53     −21.67      65.6                                        1
                     05T02:26:27
     T077    TI149    2011-06-     1 358.992 4   361 866 786.6       22.68         41.51       27.69          19.81     154.84     −24.27      40.69                                       2
                     20T18:32:00
     T079    TI158    2011-12-     3 583.241 3   377 079 149.2       28.41         58.13       41.55          17.39     114.06     −20.86      53.9                                        3
                     13T20:11:23
     T081    TI160    2012-01-     31 130.577    381 202 854.1       29.96         33.83       13.47          26.56     168.42      −19.1      71.5                                        2
                     30T13:39:48
     T083    TI166    2012-05-     953.990 5     390 921 076.9       33.58         42.22       24.31          21.56     128.95     −19.66      71.22                                       4
                     22T01:10:11
     T084    TI167    2012-06-     959.347 58    392 299 706.8       34.09         38.28       12.95          28.22     157.57     −21.09      96.53                                       3
                     07T00:07:21
     T085    TI169    2012-07-     1 012.449 8   396 432 253.9       35.61         36.79       10.44          32.69     158.09     −20.02      18.46                                       4
                     24T20:03:07
     T087    TI174    2012-11-     973.650 52     406074195          39.17         39.67       17.95          52.68     148.72     −20.43      66.26    0.01       0        −164.84        5
                     13T10:22:08



                                                                                                                                                                                                         Kutsop et al.
                                                                                               Secondary 4

     T009    TI019    2005-12-     10 411.572     188895630         313.01          38.4       50.36          27.69     −6.32      −47.47      91.9      1.5     −95.39      −50.92        6
                     26T18:59:26
```

<!-- PDF_PAGE: 27 -->

## PDF page 27

```text
                                                                                                                                                                                                                The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                             Table 1
                                                                                                           (Continued)

                                                                                                                                                                     Tilt
                                                                                                                                                          Average   from      Prime      Subsolar
                                                                         Planetocentric        Average    Average        Average    Central    Average     Pixel    North    Meridian   Longitude     # of
                                          CA Alti-      CA Ephe-         Longitude of         Incidence   Emission        Phase    Longitude   Latitude    Scale     Pole     Offset      Offset    Longitude
     Flyby     Rev        CA UTC         tude (km)      meris Time        the Sun (LS)          (deg)      (deg)          (deg)      (deg)      (deg)      (km2)    (deg)   (deg, E+)   (deg, E+)     Bins

     T067     TI129      2010-04-       7 437.318 4    323 754 719.8          8.06             53.98       50.28          15.17      18.87     −49.93      93.38    2.03     118.44       71.79        5
                        05T15:50:54
     T119     TI235      2016-05-        969.201 5     515 825 744.7         78.42              68.6       30.91          53.39     −31.99     −42.65     102.99    0.08      0.02        32.94        9
                        06T16:54:37
     T120     TI236      2016-06-       974.555 42     518 580 444.9         79.39             78.03       26.04          61.43     −30.34     −45.19      91.13    0.05      0.01        31.69        8
                        07T14:06:17
     T121     TI238      2016-07-       975.407 36     522 712 770.8         80.84             74.49       26.02           67       −55.69     −42.04      68.73    4.08     40.08        69.96        5
                        25T09:58:23
27   T123     TI243      2016-09-       1 775.659 2      528221887           82.77             61.07       19.08          69.09     −41.39     −44.94      74.66    0.34    −127.27      −99.96        6
                        27T04:16:59
     T124     TI248      2016-11-       1 585.323 9    532 353 423.8         84.21             67.14        19.3          68.96     −39.1      −45.93      48.12    2.55     60.25        85.55        7
                        13T23:55:56

     Note. A digital version of this table can be provided upon request to the ﬁrst author.




                                                                                                                                                                                                                     Kutsop et al.
```

<!-- PDF_PAGE: 28 -->

## PDF page 28

```text
                                                                                                                                                                                                               The Planetary Science Journal, 3:114 (29pp), 2022 May
                                                                                                          Table 2
                                                                                    Coefﬁcients for the Functions Displayed in Figure 9

     Figure 9(A)                                                                                                   Coefﬁcients
                                 R2              ΔI/F0                     P95%                     dΔI/F                        P95%           Period (P)        P95%          Phase (ψ)         P95%
     EQA Max (EQ 1)        0.567 134 084     0.001 045 799        (0.0009755, 0.001 116)        0.000 505 815       (0.0004142, 0.000 597 4)   108.714 098 7   (96.42, 121)    46.645 061 2   (34.78, 58.51)
     EQA Min (EQ 1)        0.555 180 381     −0.000800664       (−0.0008791, −0.0007223)        −0.000584556       (−0.0006903, −0.0004788)    97.275 427 31   (88.58, 106)   62.814 466 72   (51.65, 73.98)

                                               d(I/F)/dLS                   P95%                 ΔI/F(LS(0))                     P95%

     NPA Max (EQ 2)        0.831 186 229       −2.10E-05         (−2.777e-05, −1.419e-05)       0.002 185 821         (0.001763, 0.002 608)
     NPA Min (EQ 2)        0.383 779 961       −2.79E-05         (−3.853e-05, −1.729e-05)       0.001 040 415        (0.0003801, 0.001 701)

     Figure 9(B)                                                                                                   Coefﬁcients

                                 R2              dλ/dLS                    P95%                    λ(LS(0))                      P95%

28   EQA Max (EQ 3)        0.534 429 624        9.02E-05         (−0.0002733, 0.000 453 8)      0.880 850 658           (0.8689, 0.892 8)
     EQA Min (EQ 3)        0.952 738 203       −2.19E-05          (−0.000729, 0.000 685 3)      0.519 985 156           (0.4967, 0.543 3)
     NPA Max (EQ 3)        0.618 118 166     0.004 308 898          (0.000602, 0.008 016)       0.625 063 359           (0.3945, 0.855 6)
     NPA Min (EQ 3)        0.898 577 491        5.36E-06           (−0.005123, 0.005 134)       0.497 419 496           (0.1785, 0.816 4)

     Note. The columns labeled P95% are the 95th percentile conﬁdence bounds of the coefﬁcients in the respective columns to the left.
     EQ1) DI F = DI F0 + d DI F sin (LS P + y )
     EQ2) DI F = DI F (LS (0)) + d (I F ) dLS ´ LS
     EQ3) l = l (LS (0)) + dl (d L S) ´ LS




                                                                                                                                                                                                                    Kutsop et al.
```

<!-- PDF_PAGE: 29 -->

## PDF page 29

```text
The Planetary Science Journal, 3:114 (29pp), 2022 May                                                                                                      Kutsop et al.

                                ORCID iDs                                                Le Mouélic, S., Rodriguez, S., Robidel, R., et al. 2018, Icar, 311, 371
                                                                                         Lebonnois, S., Burgalat, J., Rannou, P., & Charnay, B. 2012, Icar, 218, 707
N. W. Kutsop https://orcid.org/0000-0001-7188-9044                                       Lebonnois, S., Flasar, F. M., Tokano, T., & Newman, C. E. 2014, in Titan, ed.
A. G. Hayes https://orcid.org/0000-0001-6397-2630                                            I Müller-Wodarg et al. (Cambridge: Cambridge Univ. Press), 122
P. M. Corlies https://orcid.org/0000-0002-6417-9316                                      Lora, J. M., Lunine, J. I., & Russell, J. L. 2015, Icar, 250, 516
                                                                                         Lora, J. M., Tokano, T., Vatant d’Ollone, J., Lebonnois, S., & Lorenz, R. D.
S. Le Mouélic https://orcid.org/0000-0001-5260-1367                                          2019, Icar, 333, 113
J. I. Lunine https://orcid.org/0000-0003-2279-4131                                       Lorenz, R. D., Smith, P. H., Lemmon, M. T., et al. 1997, Icar, 127, 173
C. A. Nixon https://orcid.org/0000-0001-9540-9121                                        Lorenz, R. D., Young, E. F., & Lemmon, M. T. 2001, GeoRL, 28, 4453
P. Rannou https://orcid.org/0000-0003-0836-723X                                          Luz, D., & Hourdin, F. 2003, Icar, 166, 328
S. Rodriguez https://orcid.org/0000-0003-1219-0641                                       Mayo, L. A., & Samuelson, R. E. 2005, Icar, 176, 316
                                                                                         McCord, T. B., Hayne, P., Combe, J.-P., et al. 2008, Icar, 194, 212
M. T. Roman https://orcid.org/0000-0001-8206-2165                                        Mitchell, J. L., Pierrehumbert, R. T., Frierson, D. M. W., & Caballero, R. 2006,
C. Sotin https://orcid.org/0000-0003-3947-1072                                               PNAS, 103, 18421
T. Tokano https://orcid.org/0000-0002-7518-9245                                          Newman, C. E., Lee, C., Lian, Y., Richardson, M. I., & Toigo, A. D. 2011,
                                                                                             Icar, 213, 636
                                                                                         Newman, C. E., Richardson, M. I., Lian, Y., & Lee, C. 2016, Icar, 267, 106
                                 References                                              Nixon, C. A., Achterberg, R. K., Ádámkovics, M., et al. 2016, PASP, 959,
                                                                                             018007
Achterberg, R. K., Conrath, B. J., Gierasch, P. J., Flasar, F. M., & Nixon, C. A.        Nixon, C. A., Lorenz, R. D., Achterberg, R. K., et al. 2018, P&SS, 155, 50
   2008, Icar, 197, 549                                                                  O’Donoghue, J., Moore, L., Bhakyapaibul, T., et al. 2021, Natur, 596, 54
Achterberg, R. K., Gierasch, P. J., Conrath, B. J., Michael Flasar, F., &                Penteado, P. F., Grifﬁth, C. A., Tomasko, M. G., et al. 2010, Icar,
   Nixon, C. A. 2011, Icar, 211, 686                                                         206, 352
Acton, C., Bachman, N., Semenov, B., & Wright, E. 2018, P&SS, 150, 9                     Pollack, J. B., & McKay, C. P. 1985, JAtS, 42, 245
Ádámkovics, M., Mitchell, J. L., Hayes, A. G., et al. 2016, Icar, 270, 376               Rages, K., & Pollack, J. B. 1983, Icarus, 55, 50
Battalio, J. M., & Lora, J. M. 2021a, NatAs, 5, 1139                                     Rannou, P. 2000, Icar, 147, 267
Battalio, J. M., & Lora, J. M. 2021b, GeoRL, 48, e94244                                  Rannou, P., Hourdin, F., McKay, C. P., & Luz, D. 2004, Icar, 170, 443
Battalio, J. M., Lora, J. M., Rafkin, S., & Soto, A. 2022, Icar, 373, 114623             Rannou, P., Le Mouélic, S., Sotin, C., & Brown, R. H. 2012, ApJ, 748, 4
Brown, M. E., Bouchez, A. H., & Grifﬁth, C. A. 2002, Natur, 420, 795                     Rannou, P., Toledo, D., Lavvas, P., et al. 2016, Icar, 270, 291
Brown, R. H., Baines, K. H., Bellucci, G., et al. 2004, SSRv, 115, 111                   Rodriguez, S., Le Mouelic, S., Rannou, P., et al. 2009, Natur, 459, 678
Caldwell, J., Cunningham, C. C., Anthony, D., et al. 1992, Icar, 97, 1                   Rodriguez, S., Le Mouélic, S., Rannou, P., et al. 2011, Icar, 216, 89
Charlot, P., Jacobs, C. S., Gordon, D., et al. 2020, A&A, 644, A159                      Roe, H. G. 2012, AREPS, 40, 355
Clark, R. N. B., Robert, H., Lytle, D. M., & Hedman, M. 2018, PDSS, http://              Roe, H. G., de Pater, I., Macintosh, B. A., et al. 2002, Icar, 157, 254
   atmos.nmsu.edu/data_and_services/atmospheres_data/Cassini/vims.html                   Roman, M. T., West, R. A., Banﬁeld, D. J., et al. 2009, Icar, 203, 242
Cleveland, W. S. 1979, Journal of American Statistical Association, 74, 829              Rossow, W. B., & Williams, G. P. 1979, JAtS, 36, 377
Corlies, P., McDonald, G. D., Hayes, A. G., et al. 2021, Icar, 357, 114228               Saur, J., Duling, S., Roth, L., et al. 2015, JGRA, 120, 1715
de Kok, R., Irwin, P. G. J., Teanby, N. A., et al. 2010, Icar, 207, 485                  Savitzky, A., & Golay, M. J. E. 1964, AnaCh, 36, 1627
Filacchione, G., Capaccioni, F., McCord, T. B., et al. 2007, Icar, 186, 259              Seignovert, B., Le Mouélic, S., Brown, R. H., et al. 2019, CaltechDATA,
Flasar, F. M., & Achterberg, R. K. 2009, RSPTA, 367, 649                                     Titan’s Global Map Combining VIMS and ISS Mosaics, https://data.
Fletcher, L. N., Kaspi, Y., Guillot, T., & Showman, A. P. 2020, SSRv, 216, 30                caltech.edu/records/1173
Gierasch, P. J. 1975, JAtS, 32, 1038                                                     Seignovert, B. t., Rannou, P., West, R. A., & Vinatier, S. 2021, ApJ, 907, 36
Goody, R., West, R., Chen, L., & Crisp, D. 1989, JQSRT, 42, 539                          Smith, B. A., Soderblom, L., Batson, R. M., et al. 1982, Sci, 215, 504
Grifﬁth, C. A., McKay, C. P., & Ferri, F. 2008, ApJ, 687, L41
                                                                                         Smith, B. A., Soderblom, L., Beebe, R. F., et al. 1981, Sci, 212, 163
Horinouchi, T., Murakami, S. Y., Satoh, T., et al. 2017, NatGe, 10, 646
                                                                                         Sromovsky, L. A., Suomi, V. E., Pollack, J. B., et al. 1981, Natur,
Hörst, S. M. 2017, JGRE, 122, 432
                                                                                             292, 698
Hourdin, F., Talagrand, O., Sadourny, R., et al. 1995, Icar, 117, 358
Jennings, D. E., Achterberg, R. K., Cottini, V., et al. 2015, ApJ, 804, L34              Stiles, B. W., Kirk, R. L., Lorenz, R. D., et al. 2008, AJ, 135, 1669
Kelland, J., Turtle, E., Rodriguez, S., Hayes, A., & Corlies, P. 2018, in 42nd           Teanby, N. A., de Kok, R., & Irwin, P. G. J. 2009, Icar, 204, 645
   COSPAR Scientiﬁc Assembly (Pasadena, CA), B5.2-43-18                                  Teanby, N. A., Irwin, P. G., Nixon, C. A., et al. 2012, Natur, 491, 732
Knowles, B. 2016, Planetary Data System, Ring-Moon Systems Node, https://                Teanby, N. A., Irwin, P. G. J., & de Kok, R. 2010, P&SS, 58, 792
   pds-rings.seti.org/cassini/iss/                                                       Tokano, T. 2010, P&SS, 58, 814
Larson, E. J. L., Toon, O. B., West, R. A., & Friedson, A. J. 2015, Icar,                Tokano, T. 2011, Sci, 331, 1393
   254, 122                                                                              Tokano, T., & Neubauer, F. M. 2005, GeoRL, 32, L24203
Lavvas, P., Yelle, R. V., & Grifﬁth, C. A. 2010, Icar, 210, 832                          Tomasko, M. G., & Smith, P. H. 1982, Icar, 51, 65
Lavvas, P. P., Coustenis, A., & Vardavas, I. M. 2008a, P&SS, 56, 27                      Turtle, E. P., Perry, J. E., Barbara, J. M., et al. 2018, GeoRL, 45, 5320
Lavvas, P. P., Coustenis, A., & Vardavas, I. M. 2008b, P&SS, 56, 67                      West, R. A., Del Genio, A. D., Barbara, J. M., et al. 2016, Icar, 270, 399
Le Mouélic, S., Cornet, T., Rodriguez, S., et al. 2019, Icar, 319, 121                   West, R. A., Seignovert, B., Rannou, P., et al. 2018, NatAs, 2, 495
Le Mouélic, S., Rannou, P., Rodriguez, S., et al. 2012, P&SS, 60, 86                     Yung, Y. L., Allen, M., & Pinto, J. P. 1984, ApJS, 55, 465




                                                                                    29
```
