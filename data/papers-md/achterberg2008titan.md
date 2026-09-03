---
citation_key: "achterberg2008titan"
title: "Titan's middle-atmospheric temperatures and dynamics observed by the Cassini Composite Infrared Spectrometer"
source_pdf: "data/papers/achterberg2008titan.pdf"
source_pdf_sha256: "3eff4498ec14137379db62d4d769bceb57c64c9a81177fcb1982ddb0c23e46fc"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                                                              Icarus 194 (2008) 263–277
                                                                                                                        www.elsevier.com/locate/icarus




            Titan’s middle-atmospheric temperatures and dynamics observed
                    by the Cassini Composite Infrared Spectrometer
           Richard K. Achterberg a,∗ , Barney J. Conrath b , Peter J. Gierasch b , F. Michael Flasar c ,
                                             Conor A. Nixon a
                                       a University of Maryland, Department of Astronomy, College Park, MD 20742, USA
                                             b Department of Astronomy, Cornell University, Ithaca, NY 14853, USA
                                               c NASA Goddard Space Flight Center, Greenbelt, MD 20771, USA

                                                       Received 31 May 2007; revised 13 September 2007
                                                                Available online 10 January 2008




Abstract
   The Composite Infrared Radiometer–Spectrometer (CIRS) instrument, on the NASA Cassini Saturn orbiter, has been acquiring thermal emission
spectra from the atmosphere of Titan since orbit insertion in 2004. Observation sequences for measuring stratospheric temperatures have been
obtained using both a nadir mapping mode and a limb viewing mode. The limb observations give better vertical resolution, and give information
from higher altitudes, while the nadir observations provide more complete longitude coverage. Because the scale height of Titan’s atmosphere is
large enough so that emission from a grazing ray is influenced by horizontal temperature variations in the atmosphere, we have developed a two-
dimensional temperature retrieval algorithm for reducing the limb spectra, which solves simultaneously for meridional and vertical temperature
variations. The analyzed nadir mapping data have sampled nearly all longitudes at latitudes from about 90◦ S to 60◦ N, providing temperatures
between pressure levels of about 5 to 0.2 mbar. The limb data covers latitudes between about 75◦ S and 85◦ N, and yields temperatures between
about 1 and 0.005 mbar, at a small number of longitudes. The retrieved temperatures are consistent with early results from nadir observations
[Flasar, F.M., and 44 colleagues, 2005. Science 308, 975–978] between 0.5 and 5 mbar where both results are valid, with the warmest temperatures
at the equator, and much stronger meridional temperature gradients in the northern (winter) hemisphere than in the southern. At higher altitudes
not probed by nadir viewing, the limb data reveal that the stratopause is nearly 20 K warmer in the northern polar regions than at the equator and
southern hemisphere, and that the altitude of the stratopause shifts from ≈0.1 mbar (300 km) near the equator to 0.01 mbar (400 km) poleward of
about 40◦ N. When the gradient wind equation is used to construct a zonal mean wind, the reversal in sign of the temperature leads to capping of
the winter westerly flow. The core of the resulting jet is about 190 m s−1 in magnitude, spans between 30◦ N and 60◦ N, and peaks near 0.1 mbar.
Estimates of the radiative heating associated with the radiative disequilibrium lead to a meridional overturning timescale of about three Earth years.
© 2007 Elsevier Inc. All rights reserved.

Keywords: Titan; Atmospheres, structure; Atmospheres, dynamics; Infrared observations




1. Introduction                                                                   equinox (February 1980). The latitudinal temperature gradient
                                                                                  was found to be too large to be dynamically supported unless
   The earliest temperature determinations for Titan’s strato-                    centrifugal acceleration balanced the latitudinal pressure gra-
sphere were made by the Voyager 1 infrared spectrometer                           dient, leading to the conclusion that Titan’s atmosphere, like
(IRIS) (Flasar et al., 1981). The latitude variation of tempera-                  that of Venus, exhibits rapid spin. Here we report new tempera-
ture from about 50◦ S latitude to about 60◦ N, at pressure levels                 ture determinations inferred from mid infrared spectra obtained
between about 0.4 and 1.0 mbar in Titan’s stratosphere was de-                    by the NASA Cassini orbiter Composite Infrared Spectrome-
termined in November 1980, just after Titan’s northern spring                     ter (CIRS). The spectra were obtained between July 2004 and
                                                                                  September 2006, after the northern hemisphere winter solstice
 * Corresponding author. Fax: +1 301 286 0212.                                    on Titan (October 2002). Observations were made both of the
   E-mail address: richard.k.achterberg@nasa.gov (R.K. Achterberg).               limb, to collect information from high elevations, and in a nadir
0019-1035/$ – see front matter © 2007 Elsevier Inc. All rights reserved.
doi:10.1016/j.icarus.2007.09.029
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
264                                             R.K. Achterberg et al. / Icarus 194 (2008) 263–277


mapping mode, to give good spatial coverage at deeper lev-
els. Altogether, the coverage reported here permits construc-
tion of a height, latitude cross-section from about 75◦ S to
75◦ N and between pressure levels approximately from 0.01
to 5 mbar, a range of over six pressure scale heights in the up-
per stratosphere and lower mesosphere. In addition, the maps
may be used to study longitudinal structure, such as propagating
waves. This report is a follow-up to a preliminary presentation
of early mapping results (Flasar et al., 2005). At that time there
were no limb observations and, as a result, the height coverage
was more restricted.
    Other techniques have been used to observe the structure
and dynamics of Titan’s atmosphere. Kostiuk et al. (2005) re-
view wind determinations made by Doppler heterodyne spec-
troscopy, and report on recent measurements made immediately
prior to the arrival of Cassini at Saturn. This type of measure-
ment is particularly valuable because it gives an unambiguous              Fig. 1. Season and time for previous wind determinations. The year of the obser-
determination of the sign of stratospheric zonal winds, which              vation is indicated at the tail of the arrow, along with the first author’s shortened
cannot be obtained from temperature measurements or stellar                name. FLA80 (Flasar et al., 1981) and ACH06 (the present work) are based on
                                                                           thermal winds from infrared spectrometers on Voyager and Cassini. HUB89
occultations (to be discussed below). The first successful mea-            (Hubbard et al., 1993), BOU01 (Bouchez, 2003) and SIC03 (Sicardy et al.,
surements were in 1993 (Fast et al., 1994). From that time                 2006) are from stellar occultation campaigns.
through to the present (Kostiuk et al., 2005), Doppler spec-
troscopy has consistently shown that Titan’s stratospheric rota-
tion is in the same direction as that of the satellite itself, which
is locked in a 1:1 spin–orbit resonance with one side facing Sat-
urn (Lemmon et al., 1993).
    Central flash events in stellar occultations give information
about the spin of Titan’s upper stratosphere by revealing an
oblate shape to density surfaces, which cannot be in static equi-
librium with the satellite’s gravitational field (see, e.g., Hubbard
et al., 1993). This technique applies at altitudes where pressures
are on the order of 0.2 to 0.8 mbar, but it has relatively poor hor-
izontal resolution, restricted by the number of observing sites.
Titan’s winds at these levels are expected to vary seasonally:
the radiative relaxation time in the upper stratosphere is rel-
atively short (Flasar et al., 1981; Flasar and Conrath, 1990),
implying that temperatures vary seasonally, and the zonal winds
and zonal mean temperatures are coupled through the ther-
mal wind equation, as discussed in Section 3. Wind profiles
or partial profiles exist for July 1989 (Hubbard et al., 1993;             Fig. 2. Schematic indication of Voyager 1 thermal wind estimates (Flasar et al.,
Sicardy et al., 1999), December 2001 (Bouchez, 2003), and No-              1981) and stellar occultation wind determinations from occultations in 1989
                                                                           (Hubbard et al., 1993), 2001 (Bouchez, 2003) and 2003 (Sicardy et al., 2006).
vember 2003 (Sicardy et al., 2006). Fig. 1 indicates the seasonal
                                                                           Points are plotted only every 30◦ for easy comparison of the global patterns.
times when wind determinations have been made, and Fig. 2                  The 1989 wind profile of Hubbard et al. (1993) is plotted only in the southern
displays a comparison of wind profiles from these four sets of             hemisphere where information exists (Bouchez, 2003).
data. The seasonal trends are complicated, but there is a sugges-
tion that wind velocities are highest in the winter hemisphere.
    Doppler shift of the radio signals from the Huygens Probe              the stratospheric region where our temperature determinations
gave direct in situ determination of the vertical profile of zonal         exist, and we shall refer back to these results in Section 3, where
wind at about 10◦ S latitude, between the surface (1.5 bar)                we will need a boundary condition for a thermal wind integra-
and 140 km altitude (∼3 mbar pressure) (Bird et al., 2005;                 tion.
Folkner et al., 2006). At altitudes above 8 km, the measure-                  Other seasonal changes, less directly connected to temper-
ments show that the winds are westerly (in the same direc-                 ature observations, are observed on Titan. At the time of the
tion as Titan’s rotation) and increase with altitude. However,             Voyager encounters (LS close to zero, northern spring equinox)
above 65 km there is a region of strong negative shear, and the            the southern hemisphere appeared brightest at visible wave-
winds decrease and nearly vanish at 75 km. At higher levels,               lengths. Lorenz et al. (1997) showed that the asymmetry was
they increase strongly with altitude, reaching ∼60 m s−1 by                one point on a complicated seasonal cycle involving the global
100 km altitude (10 mbar). This is approximately the base of               mean albedo as well as a latitudinal asymmetry. Furthermore,
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                       Temperatures in Titan’s stratosphere                                             265


the hemispheric brightness difference reverses sign between the            different placements of the array, one above the other, with ap-
blue and the near infrared. These changes in Titan’s haze prop-            proximately a 3-pixel overlap. This ensured adequate vertical
erties may be related to chemical transports caused by the gen-            coverage even in the presence of spacecraft pointing errors. For
eral circulation and its seasonal changes. Hourdin et al. (2004)           each array placement, about ten to twelve spectra were acquired
have studied the behavior of a Titan climatological model with             per pixel with a spectral resolution of 15.5 cm−1 . During the
parameterized eddy transports, and find that with appropriate              time each set of spectra were taken, the instrument pointing
parameter choices, the seasonal variations of hazes and hydro-             remained essentially fixed, with the tangent points for each de-
carbons seem consistent with observations. The seasonal cycle              tector forming tight clusters in tangent height and tangent point
on Titan is particularly complicated, in large part because the            latitude. These clusters are spaced approximately 5◦ in latitude
radiative and chemical time constants for adjustment to equi-              and 30 to 50 km in tangent height, depending on the range of
librium vary with height. Furthermore, coupling with dynamics              the spacecraft from Titan. In addition, spectra were taken as the
introduces major uncertainties because neither modeling nor                array was slewed from one latitude to the next; these were ex-
observations have yet identified the dominant dynamical heat               cluded from the analysis. The spectra within each cluster were
transport modes, not to mention their seasonal variations (Flasar          averaged together for each detector to improve the signal-to-
and Conrath, 1990; Tokano et al., 1999). The temperature de-               noise ratio. Together, the limb maps provide fairly complete
terminations reported here will provide a much more detailed               latitude coverage, but are sparse in longitude distribution as
and stringent test for models.                                             shown in Fig. 3. The characteristics of the nadir and limb data
    Because of the geometric thickness of Titan’s atmosphere,              sets are listed in Tables 1 and 2, respectively.
interpretation of the spectra taken in limb viewing mode must
take into account horizontal temperature variations in the at-             2.2. Retrieval methodology
mosphere. The largest of these are due to latitudinal temper-
ature gradients. In Section 2 a two-dimensional temperature                   To retrieve information on the atmospheric temperature
retrieval algorithm is developed for inversion of limb spectra.            structure, it is necessary to formulate an appropriate radiative
In addition, a correction procedure is introduced for the one-             transfer model. If local thermodynamic equilibrium is assumed,
dimensional inversion of spectra taken on the disk but at an               the spectral radiance I (ν) observed at the spacecraft can be
angle, in the presence of horizontal temperature gradients. In             written in the form
                                                                                   
Section 3 zonal mean results are presented and discussed, in-                                         ∂T (ν, s)
cluding display of thermal winds consistent with the tempera-              I (ν) = B ν, T (r, φ, λ)               ds,                   (1)
                                                                                                           ∂s
ture field. In Section 4, the implications for the mean meridional                  C
circulation of Titan’s stratosphere are discussed.                         where B(ν, T ) is the Planck radiance at wavenumber ν and
                                                                           temperature T , s is the distance from the spacecraft along the
2. Data and retrievals                                                     observation ray path, and T (ν, s) is the atmospheric transmit-
                                                                           tance along this distance. The atmospheric temperature field
2.1. Data                                                                  is a function of the radial distance r from the center of Ti-
                                                                           tan, the latitude φ, and the longitude λ. For quasi-nadir ob-
   The stratospheric thermal structure analyzed in this investi-           servations, the integration path C extends from the surface of
gation was retrieved from spectra acquired with Focal Plane 4              Titan to the spacecraft (in practice to the top of the sensible
(FP4) of the CIRS instrument in the 1100 to 1400 cm−1 spectral             atmosphere), while for limb observations, the path passes tan-
region with a selectable apodized spectral resolution between              gentially through the atmosphere. In the absence of atmospheric
0.5 and 15.5 cm−1 . Each pixel of the 10-element linear array              refractivity, the ray paths would be linear. The spectral radi-
has a field of view of 0.28 mrad. The instrument and its calibra-          ance observed by CIRS is the average from all ray paths within
tion are described in detail by Flasar et al. (2004).                      the solid angle of the sensor, weighted by the spatial response
   Both nadir-viewing and limb-viewing measurements are                    across the field of view. For computational expediency, we ap-
used. The nadir data were acquired using mapping sequences                 proximate this average with a single ray at the center of the
taken typically within a range of 250,000 to 400,000 km, giving            field of view. Comparisons with test calculations with detailed
a spatial resolution of 1.5◦ to 2.5◦ of great circle arc. Using a se-      averaging over the field of view indicate this approximation is
ries of continuous slews of the instrument, each map covers es-            adequate for use with the FP4 measurements used in this study.
sentially the entire Titan disk visible from the spacecraft. Four-             The atmospheric transmittance appearing in (1) is an aver-
teen maps with 2.8 cm−1 spectral resolution were completed                 age of the monochromatic transmittance over a spectral res-
during the time period included in this study; collectively, they          olution element with central wavenumber ν, weighted by the
provide quasi-global coverage. Five limb-viewing maps were                 apodized spectral response function of the instrument. Specifi-
also obtained at a typical spacecraft range of 120,000 km,                 cation of the transmittance requires knowledge of the absorber
yielding a geometric vertical resolution on the limb for an in-            distributions along the ray path. Absorption in the spectral re-
dividual pixel of 30 to 40 km. Sequences of data were taken                gion used here is dominated by the ν4 CH4 band, with addi-
at points along the limb as viewed from the spacecraft corre-              tional contributions by CH3 D. A stratospheric CH4 mole frac-
sponding to approximately 5◦ steps in latitude. The array was              tion of 0.0141 is assumed (Niemann et al., 2005), along with
positioned perpendicular to the limb, with data taken for two              a [CH3 D]/[CH4 ] ratio of 6.1 × 10−4 . For rapid calculation of
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
266                                                      R.K. Achterberg et al. / Icarus 194 (2008) 263–277




Fig. 3. Latitude and longitude of limb spectra. The circled clusters of points are the sub-spacecraft location for each observation. See Table 2 for further information.


Table 1
Summary of nadir mapping observations
Encounter              Start time                           Duration              Mean longitude (◦ west)               Mean latitude              Resolution (◦ of arc)
T0                     2004 Jul 02 03:30:21                 13:30                 356                                   38◦ S                      2.1
Tb                     2004 Dec 13 15:12:29                  8:25                 138                                   07◦ S                      2.1
T3                     2005 Feb 14 09:57:53                  9:00                 134                                   02◦ S                      2.1
T3                     2005 Feb 15 18:57:53                  4:20                 356                                   00◦ N                      1.8
T4                     2005 Apr 01 08:05:16                  6:30                 232                                   02◦ S                      1.9
T6                     2005 Aug 22 20:53:37                  6:47                 228                                   13◦ S                      1.9
T8                     2005 Oct 27 01:24:00                  7:00                 126                                   00◦ N                      2.9
T8                     2005 Oct 28 16:15:25                  7:49                 340                                   01◦ N                      1.9
T9                     2005 Dec 27 14:04:00                 10:07                 228                                   00◦ N                      2.9
T10                    2006 Jan 14 14:23:27                  9:13                 131                                   00◦ N                      2.1
T14                    2006 May 21 01:18:11                  2:00                  11                                   40◦ N                      1.7
T14                    2006 May 21 06:18:11                  2:58                 358                                   06◦ S                      2.4
T15                    2006 Jul 02 23:50:47                  7:54                 228                                   01◦ S                      2.3
T17                    2006 Sep 06 21:56:51                  7:20                 145                                   07◦ N                      2.4
T18                    2006 Sep 22 20:58:49                  7:00                 137                                   10◦ N                      2.3


Table 2
Summary of limb observations
Encounter               Start time                           Duration               Latitude range               Mean longitude (◦ west)                 Resolution (km)
T4                      2005 Apr 01 00:25:12                 3:30                   3◦ N–86◦ N                   132                                     35
T6                      2005 Aug 22 01:10:00                 2:30                   34◦ S–26◦ N                  295                                     35
T6                      2005 Aug 22 13:40:00                 2:30                   80◦ S–38◦ S                  318                                     35
T8                      2005 Oct 27 09:38:09                 3:20                   11◦ N–86◦ N                   72                                     40
T13                     2006 Apr 30 11:53:31                 5:00                   1◦ N–41◦ N                    99                                     28
T16                     2006 Jul 22 05:25:13                 2:15                   31◦ N–76◦ N                   56                                     33



transmittance, a correlated-k approach (Lacis and Oinas, 1991)                          servational geometry can be used to infer information on the
is used, which incorporates the spectral response function of                           stratospheric temperature field.
the instrument. The required molecular parameters are taken
from the GEISA 2003 spectral line atlas (Jacquinet-Husson                               2.2.1. Limb retrieval
et al., 2005). Aerosols can also contribute to the atmospheric                             We first consider analysis of the limb data. The locations
absorption as discussed further below. Once the transmittance                           of the tangent points for the five limb maps used in this study
is specified, measured radiance as a function of ν and/or ob-                           are shown in Fig. 3. The encircled clusters of points represent
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                               Temperatures in Titan’s stratosphere                                                           267


the sub-spacecraft locations corresponding to each sequence of                     perature field, we have formulated a 2-dimensional (height,
measurements. In all cases, the data were taken with the space-                    latitude) inversion algorithm. Work has been done previously
craft near Titan’s equatorial plane. For low latitude spectra,                     on 2-dimensional algorithms for application to terrestrial limb-
projections of the ray paths onto the surface of Titan are es-                     viewing thermal emission spectra. Worden et al. (2004) investi-
sentially in the zonal direction, while at higher latitudes, the ray               gated the possibility of retrieving CO, while Steck et al. (2005)
paths traverse a substantial range of latitudes. We anticipate that                studied temperature retrieval. In both cases, feasibility studies
meridional temperature gradients are stronger than zonal gradi-                    were done using synthetic data and a geometry in which space-
ents. As a consequence, the variation of temperature along a                       craft and tangent points were in the same meridional plane.
ray path associated with a low-latitude tangent point results pri-                 In each case, a profile of a single atmospheric parameter was
marily from the projection of the vertical temperature gradient                    retrieved with all necessary constraints assumed known, includ-
onto the ray path, while at higher latitudes, both the vertical and                ing exact pointing geometry and knowledge of the pressure
latitude components of the gradient contribute to the variation.                   field.
For the data sets used here, the latitude gradients are poten-                         In formulating our limb retrieval algorithm, dependence on
tially important because of two factors. First, the ratio of the                   longitude is neglected, with the atmospheric temperature as-
depth of Titan’s atmosphere to its radius is relatively large. For                 sumed to be a function of altitude and latitude only. A 2-dimen-
a spectral region of moderate opacity, the maximum contribu-                       sional grid is established with n layers in the vertical between
tion to the outgoing radiance comes from a portion of√the ray                      Titan’s surface and an altitude of 600 km with a spacing z in
path centered near the tangent point with a width ∼2 2rt H ,                       geometric height, and q latitudes with equal spacing φ. Val-
where H is the pressure scale height, and rt is the distance of                    ues of n = 100 and φ = 5◦ are used in most cases with the
the tangent point from the center of Titan. For a tangent height                   value of q based on the latitude range spanned by the particular
of 200 km above Titan’s surface and H = 40 km, this distance                       limb map used. Temperatures are defined at these grid points by
is ∼670 km or a great circle arc of 20◦ . Second, in the spectral                  the array
region near 1300 cm−1 and for the temperature range relevant
                                                                                   T (ri , φj ) = Tij ;    i = 1, . . . , n; j = 1, . . . , q.                (2)
here, the Planck radiance is very strongly dependent on T with
d ln B/d ln T ∼ 14, so temperature gradients along the ray path                    It is necessary to first formulate a finite difference, 2-dimen-
become greatly magnified. Ray paths for measurements ac-                           sional forward radiative transfer model based on (1). The ray
quired on the T4 Titan flyby, projected onto a meridional plane,                   path is divided into ns segments, with the segment boundaries at
are shown in Fig. 4.                                                               the points where the ray path crosses altitude layer boundaries.
    In order to retrieve temperatures from these data, while prop-                 If the distance along the ray path measured from the top of the
erly taking into account the latitude dependence of the tem-                       atmosphere to the boundary between segments i and i + 1 is




Fig. 4. Ray paths for limb measurements acquired on the T4 Titan flyby. The equatorward branch of each trajectory extends from the spacecraft to the tangent point
of the ray, while the poleward branch is the extension of the ray path beyond the tangent point.
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
268                                                      R.K. Achterberg et al. / Icarus 194 (2008) 263–277


denoted by si , then the radiative transfer equation can be written                 because of errors in knowledge of the spacecraft orientation.
                                                                                    Second, the pressure is not precisely known on a constant grav-
          
          ns
                                                   
I (ν) =         B(ν, T̂i ) T (ν, si−1 ) − T (ν, si ) .                     (3)      itational potential surface nor are necessary portions of the
          i=1
                                                                                    temperature profile which may lie outside the region for which
                                                                                    information can be retrieved from the measurements. Inference
The temperature T̂i at the point where the ray path intersects                      of ξl attempts to compensate for both of these effects by prop-
the center of the ith layer is obtained by linear interpolation                     erly aligning the tangent heights with the pressure field.
in latitude of the primary temperature array Tij . To accomplish                        In practice, offsets to the tangent point altitude can have a
this, the latitude of the point of intersection is calculated from                  very similar affect on the spectrum as scaling the aerosol opti-
knowledge of the locations of the tangent point and the sub-                        cal depth profile, so that there can be an ambiguity between the
spacecraft point. It is assumed that refraction of the ray path                     retrieved values for the tangent height correction and aerosol
can be neglected. Estimates indicate that at a tangent height of                    optical depth. Therefore, although we retrieve aerosol optical
150 km, for example, the actual ray path near the tangent point                     depth profiles along with the temperature, we do not present
is refracted downward by only ∼10 m relative to the adopted                         the retrieved aerosol profiles in this paper. We have examined
linear path. The latitude of each point along the ray path can                      the effect of the aerosol opacity on our temperature retrievals
be calculated by noting that the ray path projection on a ref-                      by also performing temperature retrievals assuming no aerosol
erence sphere centered on Titan follows a great circle passing                      opacity, using a more restricted wavenumber range (1270 to
through both the sub-spacecraft and the tangent point locations.                    1300 cm−1 ) where the effect of the aerosol on the spectrum
The meridional ray path projections shown in Fig. 4 were ob-                        is lowest. Results from the temperature only retrievals typically
tained in this way.                                                                 differ from the combined temperature and aerosol retrievals by
    Calculation of the transmittances in (3) requires a knowledge                   up to about 1 K over the altitude range where the retrievals are
of the distribution of absorbers along the ray path. The distrib-                   valid.
utions of CH4 and CH3 D can be obtained from their mole frac-                           Spectra with 0.5 cm−1 resolution would give better access
tions, providing the pressure and temperature along the ray path                    to the continuum between gaseous absorption lines, permitting
are known. The pressure profile at each latitude grid point is ob-                  better separation of the retrieved aerosol profile and tangent
tained by integration of the hydrostatic equation. This requires                    height shift. Averages of 0.5 cm−1 limb spectra at 15◦ S and
knowledge of pressures at each latitude on a reference surface                      85◦ N have been successfully inverted by Vinatier et al. (2007)
of constant gravitational potential, along with the temperature                     to obtain temperature, gas abundances, and aerosol opacity. Be-
profile at all levels between the reference surface and the top                     cause of the relatively large amount of integration time required
of the atmospheric model. An interpolation of the pressure in                       to obtain an adequate signal-to-noise ratio, extensive spatial
latitude for each point along the ray path is carried out similar                   coverage is not possible with this high-resolution mode. Since
to that used for temperatures. An alternative formulation could                     spectra taken at 15 cm−1 resolution provide good signal-to-
be developed using a pressure-related vertical coordinate such                      noise in a relatively short time, this observing mode was used
as − ln p. However, it is still necessary to invoke the hydrosta-                   in limb temperature maps where good latitude coverage was de-
tic equation, along with the appropriate boundary conditions,                       sired.
to relate the pressure field to the geometry of the limb tangent                        Two approaches to the 2-dimensional inversion problem
observations.                                                                       were examined. In the first, the 2-dimensional forward model is
    Aerosols may also contribute significant absorption with the                    used, but retrievals are carried out at each tangent point latitude
long paths associated with the limb tangent geometry. We as-                        individually, progressing from low to high latitudes. At a given
sume the aerosols can be treated as pure absorbers, and specify                     latitude, only temperatures at that latitude are retrieved, with
profiles of (dτa /dr)ij at each grid point where τa is the normal                   temperatures along the ray paths at all lower latitudes specified
aerosol optical depth, taken to be independent of wavenumber                        from the previous retrievals. Temperatures poleward of the tan-
over the spectral region used here. Again, interpolation in lati-                   gent point latitude are obtained by extrapolation. This approach
tude is used to specify this quantity at each required point along                  takes advantage of the fact that the ray paths at low latitudes
the ray path.                                                                       are primarily zonal, and coupling in latitude gradually increases
    We now turn to the development of an appropriate re-                            moving poleward. In the second approach, atmospheric para-
trieval algorithm. Radiances from eight spectral points spaced                      meters at all latitudes are retrieved simultaneously from the
15 cm−1 apart between 1210 and 1315 cm−1 are used in the                            entire set of measurements. The two methods were found to
retrievals, providing a sampling of the widest range of at-                         give similar results. Only the second approach will be consid-
mospheric opacity available in this spectral region with this                       ered further here.
spectral resolution.                                                                    The measured spectral radiance is a function of ν, rt , and φt .
    Since both the temperature and aerosol distributions are un-                    For convenience, these measurements are incorporated into a
known, it is necessary to attempt to retrieve both parameters                       single m-element vector y whose elements are the radiances
from the measurements. In addition, we also infer a single                          corresponding to the sets [ν, rt , φt ]k (k = 1, . . . , m). For the
shift ξl in all tangent heights at a given tangent point lati-                      larger limb sequences, m exceeds 1000. In a similar fashion,
tude (φt )l (l = 1, . . . , p). The reason for this is two-fold. First,             we can construct a single vector x containing all of the quan-
the tangent heights provided in the data sets can be incorrect                      tities to be retrieved. Define the sets of vectors Tj and bj
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                        Temperatures in Titan’s stratosphere                                                 269


(j = 1, . . . , q) representing the columns of the arrays Tij and           the error covariance matrix is assumed to be diagonal, with
ln(dτc /dr)ij , respectively. These n-element vectors represent             values equal to the square of the noise equivalent spectral ra-
the vertical temperature and aerosol profiles at the j th lati-             diance (NESR) of each measurement. Matrix transposition is
tude grid point. Combining these vectors with the tangent point             denoted by the superscript T . The first term in (11) is χ 2 ,
shifts ξl (l = 1, . . . , p) we obtain                                      as used in standard least squares estimation, while the second
     ⎡ ⎤                                                                    term represents a constraint that must be included to obtain
       T1
                                                                            a physically meaningful solution from the ill-posed problem.
     ⎢ .. ⎥
     ⎢ . ⎥                                                                  The matrix S is determined by the specific constraints chosen.
     ⎢ ⎥
     ⎢ Tq ⎥                                                                 The penalty function Q is similar in form to that frequently
     ⎢ ⎥
     ⎢ b1 ⎥                                                                 employed in 1-dimensional inversion problems, and minimiza-
     ⎢ . ⎥
x=⎢        ⎥                                                                tion is straight forward (see, for example, Press et al., 1994;
     ⎢ .. ⎥ .                                                  (4)
     ⎢ ⎥                                                                    Rodgers, 2000), yielding
     ⎢ bq ⎥
     ⎢ ⎥
     ⎢ ξ1 ⎥                                                                 x̂ = SKT H−1 y,                                               (12)
     ⎢ . ⎥
     ⎣ .. ⎦
                                                                            where
        ξp
    A forward spectral radiance model vector f(x) with m-                   H = KSKT + E .                                                  (13)
elements is defined such that the kth element is the spectral               The matrix H is large (m×m), but its inversion is accomplished
radiance calculated using (3) for the set [ν, rt , φt ]k . The sensi-       efficiently using Cholesky decomposition and back substitution
tivities of the radiances to the temperature and aerosol profiles           (see, for example, Press et al., 1994). Because of the nonlinear
at the j th latitude grid point can be written as the m × n Jaco-           dependence of the measurements on the atmospheric parame-
bian matrices                                                               ters, it is necessary to iterate the solution. Note that we do not
      ∂f                                                                    constrain the final solution to necessarily lie near the reference
Lj =      ; j = 1, . . . , q,                                      (5)      value x0 —since we have little a priori knowledge of Titan’s
     ∂Tj
                                                                            thermal structure and aerosol distribution, at each iteration we
       ∂f
Mj =      ; j = 1, . . . , q.                                      (6)      replace x0 with the x from the previous iteration.
      ∂bj                                                                       The constraint matrix S is chosen to permit filtering of the
These partial derivatives are calculated analytically from the              retrieved temperature and aerosol fields in both height and lat-
discretized forward model. The partial derivatives of the radi-             itude. This is accomplished by imposing correlations of the
ances with respect to the shifts in tangent height at each tangent          profiles between atmospheric levels at each grid point latitude
point latitude can be written as a set of m-element vectors                 and correlations between latitudes at each atmospheric level.
                                                                            Correlations between latitudes for the retrieved tangent height
       ∂f
ul =       ;   l = 1, . . . , p,                                   (7)      shifts are also included. S can be displayed schematically in
       ∂ξl                                                                  block form
and are calculated by numerical perturbation. The matrices (5)                   ⎡                                                            ⎤
                                                                                    U11 · · · U1q        O ··· O            O ··· O
and (6) along with the vectors (7) can be combined into a single                 ⎢ ..               ..     ..           ..    ..          .. ⎥
array K of dimensions m × (2nq + p), which can be written in                     ⎢ .
                                                                                 ⎢                   .      .            .     .           . ⎥⎥
block form                                                                       ⎢ U1q · · · Uqq
                                                                                 ⎢                       O ··· O            O ··· O ⎥         ⎥
                                                                                 ⎢ O · · · O V11 · · · V1q
                                                                                 ⎢                                          O ··· O ⎥         ⎥
K = [ L1 · · · Lq      M1 · · · Mq   u1 · · · u p ] .              (8)           ⎢
                                                                            S = ⎢ ...             ..     ..           ..    ..          .. ⎥
                                                                                 ⎢                 .      .            .     .           . ⎥  ⎥
                                                                                                                                                ,
   The forward model f(x) can be expanded about a reference                      ⎢ O · · · O V1q · · · Vqq                  O ··· O ⎥         ⎥
                                                                                 ⎢
set of atmospheric parameters x0 , which to first order can be                   ⎢ O ··· O
                                                                                 ⎢                       O · · · O w11 · · · w1p ⎥            ⎥
written                                                                          ⎢ .               ..     ..           ..    ..          .. ⎥
                                                                                 ⎣ ..               .      .            .     .           . ⎦
f(x) = f(x0 ) + Kx,                                               (9)                O ··· O            O · · · O w1p · · · wpp
where                                                                                                                                        (14)
                                                                            where null matrices of appropriate dimensions are represented
x = x − x0 .                                                    (10)
                                                                            by the O symbols, and the Ujj  are n×n matrices which impose
   We proceed to formulate a constrained inversion algorithm                2-point correlation in the vertical as well as between grid point
by minimizing the penalty function                                          latitudes φj and φj  . The form chosen is
Q = (y − Kx̂)T E−1 (y − Kx̂) + x̂T S−1 x̂,                 (11)       Ujj  = αz Cδjj  + αφ In Djj  ;   j, j  = 1, . . . , q,      (15)
where y = y − f(x0 ) with y the measured radiance, E is                    where the n × n matrix C specifies the correlation in the vertical
the measurement error covariance matrix, and x̂ = x0 + x̂                  at a given latitude, and Djj  is the correlation in latitude at a
is the constrained solution sought. Under the assumption that               given height. In is an n × n unit matrix, and the scalars αz and
the random errors in the m measurements are uncorrelated,                   αφ determine the strength of the filtering. Adopting Gaussian
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
270                                                      R.K. Achterberg et al. / Icarus 194 (2008) 263–277


correlation filters,                                                                measurements. Examples are shown in Figs. 5 and 6. Spec-
                                                                                  tral radiances were calculated using our 2-D forward model,
Cii  = exp −(i − i  )2 z2 /2cz2 ;        i, i  = 1, . . . , n,        (16)      assuming an observation geometry corresponding to the T4
where cz is the vertical correlation length or filter width. The                    limb map. Two different model temperature cross-sections
filter in latitude is                                                               were used, one with a relatively warm upper atmosphere at
                                                                                  high latitudes and the other with a cold polar region. Ran-
Djj  = exp −(j − j  )2 φ 2 /2cφ2 ; j, j  = 1, . . . , q, (17)                   dom noise was added to the radiances with a rms value of
where cφ is the latitude correlation length. The matrices Vjj                      3 × 10−10 W cm−2 ster−1 /cm−1 , which is approximately the
provide similar filtering on the aerosol field and are written                      effective Noise Equivalent Spectral Radiance (NESR) for the
                                                                                    averages of ∼10 spectra as used in the limb map retrievals.
Vjj  = βz Cδjj  + βφ In Djj  ;      j, j  = 1, . . . , q.             (18)      As a basis for comparison, retrievals were first carried out us-
                                                                                    ing a 1-dimensional inversion algorithm, treating the data at
Smoothing in latitude of the tangent height shifts is accom-                        each limb point latitude independently and assuming horizontal
plished using                                                                       homogeneity (Figs. 5 and 6, top). At the higher northern lati-
                                     
wll  = γφ exp −(l − l  )2 φt2 /2cφ2 ; l, l  = 1, . . . , p, (19)                tudes, there are substantial errors in the retrieved cross-sections
                                                                                    because of the inability of the 1-dimensional retrieval to distin-
where φt is the characteristic tangent point latitude spacing                      guish between latitude and height components of the temper-
of 5◦ .                                                                             ature gradients. The 2-dimensional retrieval algorithm yields
    A vertical correlation length cz = 40 km was used, and the                      substantially better results in both cases (Figs. 5 and 6, bot-
latitude correlation length cφ was set equal to the latitude grid                   tom). As the edges of the retrieval domain are approached both
spacing. The parameters αz , αφ , βz , βφ , and γφ were deter-                      vertically and horizontally, the information content decreases,
mined empirically to yield stable solutions, which are also con-                    and the retrieved thermal structure becomes increasingly less
sistent with χ 2 ∼ m.                                                               well constrained. This accounts for the deterioration in the qual-
    Using (13)–(19) in the solution (12), and explicitly display-                   ity of the 2-dimensional retrieval at the northernmost latitudes
ing the retrieved temperature and aerosol profiles for each grid                    and higher atmospheric levels. For this reason, in the analyses
point latitude, we obtain                                                           that follow we will not use retrieved temperatures from limb
         q                                                                        measurements poleward of 75◦ N nor for pressures less than
           
T̂j =         Ujj  Lj  H−1 y; j = 1, . . . , q
                      T
                                                             (20)                   0.005 mbar.
             j  =1
                                                                                    2.2.2. Nadir retrieval
and                                                                                     Inversion of the nadir-viewing spectra is accomplished with
                              
             
             q                                                                      a 1-dimensional retrieval algorithm of a type extensively dis-
b̂j =                 Vjj  MTj H−1 y;   j = 1, . . . , q.             (21)      cussed previously (Conrath et al., 1998). In this case ẑ = − ln p
             j  =1                                                                 is used as the vertical coordinate. A vertical temperature profile
The solution for the shift in tangent height is                                     is retrieved from each individual spectrum. Minimization of a
      p                                                                           penalty function analogous to (11) gives
      
                                                                                                                  −1
ξl =      wll  ul  H−1 y; l = 1, . . . , p.
                 T
                                                                          (22)      T̂ = SKT KSKT + E                 I                                (24)
       l  =1
                                                                                    and, for the 1-dimensional retrieval, K and S are, in component
The random error in the retrieved temperature profile resulting                     form,
from the propagation of random error in the measured radiances
                                                                                           ∂B[νk , T (ẑi )] ∂T (νk , ẑi )
can be calculated from (20). The resulting retrieved temperature                    Kki =                                   ;
error covariance matrices for the profile at grid point latitudes                                   ∂T              ∂ ẑ
φj and φj  become                                                                     k = 1, . . . , m; i = 1, . . . , n,                               (25)
         q                        q                                             and
                                    
Rjj  =               T     −1  −1
                                          Lj  Uj  j  ;                                                               
              Ujj  Lj  H EH                                                     Sii  = αẑ exp −(i − i  )2 ẑ2 /cẑ2 ;   i, i  = 1, . . . , n,   (26)
             j  =1                          j  =1
                                                                                    where ẑ is the vertical layer increment, cẑ is vertical correla-
  j, j  = 1, . . . , q.                                                  (23)
                                                                                    tion length (usually chosen to be 1 pressure scale height), and
In this expression, we have made use of the fact C and H are                        αẑ controls the strength of the vertical filter. I is the difference
symmetric matrices. In the present analysis, we are interested                      between the measured spectral radiance and that calculated with
only in the diagonal elements of the n × n matrices Rjj , which                     the forward model, using the reference temperature profile T0 .
represent the error variance at each atmospheric level for the                      The retrieved vertical temperature profile is
profile at latitude φj . The error covariance matrices for the re-
                                                                                    T̂ = T0 + T̂.                                                       (27)
trieved aerosol profiles are similar in form to (23).
    To investigate the behavior of the 2-dimensional retrieval                      As in the 2-dimensional limb case, the non-linearity of the prob-
algorithm outlined above, we have applied it to synthetic                           lem requires iteration.
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                                                Temperatures in Titan’s stratosphere                                                            271




Fig. 5. One- and two-dimensional retrieval results on synthetic data with a warm    Fig. 6. One- and two-dimensional retrieval results on synthetic data with a cold
pole, as discussed in the text. The one-dimensional retrieval does not recover      pole, as discussed in the text.
the polar warm spot well.
                                                                                    on both the emission angle of the observations and the ther-
   Because of Titan’s relatively thick atmosphere, the lati-                        mal structure itself, being more heavily weighted toward warm
tude and longitude of points along the ray path at higher at-                       regions. The temperature variance resulting from the propaga-
mospheric levels can differ significantly from the coordinates of                   tion of random measurement error serves as an indicator of the
the point of intersection of the ray path with Titan’s surface, es-                 relative information content. Fig. 7 shows a meridional cross-
pecially for data taken at large emission angles. Consequently,                     section of the square root of the error variance, calculated using
the retrieved temperature profile is not strictly representative                    the 1-dimensional nadir equivalent of (23). In regions where
of the local vertical thermal structure when horizontal tem-                        there is strong sensitivity of the measured radiances to the ther-
perature gradients are present. Rather than attempting a full                       mal structure, the error variance is relatively low. The error
2-dimensional nadir retrieval as was done with the limb data,                       increases moving toward both lower and higher pressures until
a correction is introduced by calculating the latitude and longi-                   maxima are reached. Finally, as the sensitivity decreases fur-
tude associated with each atmospheric level along the ray path.                     ther at the both the lowest pressures and deepest levels, the
These level-dependent coordinates are used in constructing the                      error once again decreases. In these regions, the algorithm re-
cross-sections and maps discussed in the following sections.                        trieves values approaching the reference profile, but does so
                                                                                    with high precision. Thus, the upper and lower maxima of the
3. Results                                                                          estimated error provide bounds for the region of useful infor-
                                                                                    mation on the thermal structure. A similar error cross-section
   The first step in our analysis of the retrieved temperatures is                  for the limb retrievals is shown in Fig. 8. In this case, the full
to calculate a zonal mean meridional cross-section. We use a                        2-dimensional version of (23) was used. These results indicate
combination of nadir and limb retrievals for this purpose. The                      that the nadir retrievals give useful results from approximately
inversion algorithms discussed in the previous section were ap-                     0.2 to 5 mbar, except at high northern latitudes where the deep
plied to all of the data sets listed in Table 1, using as an initial                pressure limit decreases to ∼2 mbar. Inclusion of the limb re-
guess temperature profile the 15◦ S profile from Flasar et al.                      sults moves the lower pressure limit approximately two decades
(2005).                                                                             to ∼0.003 mbar.
   The pressure range over which useful information on the                              A composite mean meridional temperature cross-section
temperature structure is obtained from the nadir data depends                       was constructed, using all of the nadir and limb maps. The
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
272                                                     R.K. Achterberg et al. / Icarus 194 (2008) 263–277


temperatures at each pressure level were averaged in 5◦ lati-                      temperatures are consistent with the preliminary, nadir-only re-
tude bins, and the resulting averages were then smoothed with                      sults of Flasar et al. (2005) with the warmest temperatures at
three passes of a sliding 10◦ rectangular window. For pressures                    the equator, and much larger gradients in the northern (winter)
greater than ∼0.2 mbar, the averages are dominated by the nadir                    hemisphere than in the southern hemisphere. At lower pres-
retrievals, which provide good sampling in longitude at most                       sures, seen only with the limb observations, we find that the
latitudes. However, they are distributed over an approximately                     stratopause is almost 20 K warmer and a decade lower in pres-
2-year time interval as indicated in Table 1. At lower pressures,                  sure at high northern latitudes than at the equator and in the
the results depend predominately on the limb retrievals, with                      southern hemisphere. At 10◦ S, our retrieved temperatures are
extremely limited coverage in longitude (see Fig. 3). Because                      up to 10 K colder than the temperatures measured by the Huy-
of these limitations, the results should be regarded as represen-                  gens Atmospheric Structure Instrument (HASI) in the upper
tative of the large scale meridional structure at this particular                  stratosphere between 0.1 and 1 mbar (Fulchignoni et al., 2005).
season rather than as a detailed zonal mean meridional cross-                      Conversely, synthetic spectra calculated using the HASI mea-
section for a given point in time. The resulting temperatures                      sured temperature profile have radiances in the ν4 CH4 band
are shown in Fig. 9. For pressures greater than ∼0.5 mbar, the                     that are over 30% larger than the radiances measured by CIRS.
                                                                                   The reason for the discrepancy between the two instruments is
                                                                                   currently unknown.
                                                                                       Using the assumption that the meridional pressure gradient
                                                                                   is balanced by the sum of the horizontal components of the
                                                                                   Coriolis and centrifugal forces, which is expected to hold for
                                                                                   Titan, the zonal wind velocity u is related to the meridional
                                                                                   temperature gradient through the gradient wind equation
                                                                                                                      
                                                                                     ∂               u2         g 1 ∂T
                                                                                          2Ωu +             =−              ,                 (28)
                                                                                   ∂z            r cos φ       T r ∂φ p
                                                                                   where Ω is the rotation rate of Titan, φ is latitude, g is grav-
                                                                                   itational acceleration, and for a thick atmosphere, the “verti-
                                                                                   cal” derivative is taken along cylinders parallel to the rota-
                                                                                   tion axis (see, for example, Flasar et al., 2005). To calculate
                                                                                   a wind velocity from (28), it is necessary to specify a bound-
                                                                                   ary condition. Currently, few constraints on the wind field
                                                                                   exist. We have chosen to assume solid-body rotation at the
Fig. 7. Uncertainty in temperatures retrieved from a single spectrum, from nadir   10 mbar level at an angular velocity of four times Titan’s solid-
data from the T4 flyby, resulting from propagation of random errors in the mea-    body rotation rate, consistent with the Huygens Doppler Wind
sured radiances.                                                                   Experiment measurements at that altitude (Bird et al., 2005;




      Fig. 8. Uncertainty in temperatures retrieved from limb data from the T4 flyby, resulting from propagation of random errors in the measured radiances.
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                                                                Temperatures in Titan’s stratosphere                                                              273




Fig. 9. Zonal mean temperatures from all limb and nadir maps. Retrieved temperatures were averaged in 5◦ latitude bins, then smoothed with a 10◦ boxcar function
applied three times. Contours are labeled in K.




Fig. 10. Zonal winds calculated from the temperatures in Fig. 9 from the gradient wind equation, assuming solid-body rotation at the 10 mbar level at four times
Titan’s rotation rate. Wind speed contours (black lines) are labeled in m s−1 . The gray lines indicate cylindrical surfaces parallel to the rotation axis along which
the gradient wind equation is integrated. Equatorward and above the gray line tangent to the equator at 10 mbar, the winds are unconstrained by the gradient wind
equation, and have been linearly interpolated on constant pressure surfaces.


Folkner et al., 2006). The resulting wind cross-section is shown                     M = (Ωr cos φ + ū)r cos φ,                                                (29)
in Fig. 10. Equatorward and above the line defining the cylin-
der tangent to the equator at 10 mbar, the wind field is un-                         and the zonal mean of the Ertel potential vorticity
constrained by (28), and u has been linearly interpolated on                                                                              
                                                                                          ∇ × v · ∇θ           1       ∂θ  ∂M ∂θ ∂M 
constant pressure surfaces. The resulting wind profile has a sin-                    Q̄ =            = −g 2                      −            ,
                                                                                               ρ            a cos φ ∂φ p ∂p         ∂p ∂φ p
gle strong jet at northern midlatitudes, with peak winds of about                                                                              (30)
190 m s−1 between 30◦ N and 50◦ N at a pressure of 0.01 mbar.
   From the calculated zonal mean winds, we can also estimate                        which is a conserved quantity for adiabatic, frictionless motion.
the zonal mean angular momentum per unit mass                                        Here v is the velocity vector, and θ is the potential temperature.
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
274                                                     R.K. Achterberg et al. / Icarus 194 (2008) 263–277

                                                                                                1
                                                                                      =                ∇ · F,                                  (31)
                                                                                           ρ0 a cos φ
                                                                                         1     ∂                 1 ∂
                                                                                                  (v̄ cos φ) +         (ρ0 w̄) = 0,            (32)
                                                                                    a cos φ ∂φ                   ρ0 ∂z
                                                                                                                      
                                                                                    ∂ T̄    v̄ ∂ T̄         ∂ T̄    g          T − Teq
                                                                                          +         + w̄         +       =−            .       (33)
                                                                                    ∂t      a ∂φ            ∂z      Cp            τr
                                                                                       These equations include assumptions which cannot yet be
                                                                                   verified for Titan. In the heat equation we have omitted eddy
                                                                                   diffusivity, on the grounds that the temperature structure is
                                                                                   stable. In these equations, the meridional and vertical veloc-
                                                                                   ities v̄ and w̄ are the residual mean velocities. We assume
                                                                                   that these are appropriate for estimation of the zonal mean
                                                                                   transport of passive tracers from both advection and eddy
                                                                                   fluxes. This also follows terrestrial practice (Pedlosky, 1987;
                                                                                   Holton, 2004), but may not apply to Titan. The notation is
                                                                                   φ = latitude, f = 2Ω sin φ, where Ω is Titan’s rotation fre-
                                                                                   quency, and Cp is the specific heat at constant pressure. The
                                                                                   radiative forcing is approximated by linear relaxation to a ra-
                                                                                   diative equilibrium temperature Teq with time scale τr . F is the
                                                                                   Eliassen–Palm (EP) flux, the divergence of which is the eddy
                                                                                   momentum forcing of the mean zonal flow.
                                                                                       The energy equation (33) allows us to estimate the vertical
                                                                                   wind velocity from the observed temperature if we can assume
                                                                                   that the first two terms on the left are negligible, so that
                                                                                                                    
                                                                                             ∂ T̄    g −1 T − Teq
                                                                                   w̄ = −         +                      .                      (34)
                                                                                              ∂z    Cp           τr
Fig. 11. (Top) Potential temperature calculated from the temperatures in Fig. 9.   The radiative timescale in the middle stratosphere was calcu-
(Middle) Zonal angular momentum per unit mass, calculated from the zonal           lated by Flasar et al. (1981) to be τr = 3. × 107 s at 1 mbar,
mean winds in Fig. 10. (Bottom) Ertel potential vorticity of the zonal mean
winds of Fig. 10.
                                                                                   decreasing somewhat with increasing altitude, which is much
                                                                                   shorter than the Titan year, so strong seasonal effects are ex-
                                                                                   pected. Calculations of radiative equilibrium temperatures for
   The resulting cross-sections of zonal mean angular momen-
                                                                                   the season appropriate for our observations have not been done
tum and Ertel potential vorticity are shown in Fig. 11. Because
                                                                                   with realistic haze distributions. However, Hourdin et al. (1995)
the potential vorticity involves the second derivative of the ob-
                                                                                   give the results of radiative–convective calculations for p >
served temperatures, the result is very noisy (particularly at low
                                                                                   0.3 mbar, albeit with a globally uniform haze at northern winter
pressures where the data is only from limb observations) and
                                                                                   solstice. Comparison of this calculation with our results allows
only the large scale variations are reliable.
                                                                                   us to estimate that T − Teq near the 1 mbar level varies from
                                                                                   ∼ −15 K near the south pole to ∼25 K around 60◦ N, and
4. Discussion
                                                                                   probably larger at the north pole, although the uncertainty in
                                                                                   these numbers is large. This gives an order of magnitude es-
4.1. Zonal mean circulation                                                        timate of the vertical velocities of w ∼ 0.5 mm s−1 , with ris-
                                                                                   ing motion in the southern hemisphere and subsidence in the
    We can use a standard approach that is applied to the Earth’s                  north. This is qualitatively consistent with general circulation
middle atmosphere (Holton, 2004) to discuss, in an order-of-                       models of Titan, which predict a single meridional cell in the
magnitude sense, the physics of the flow. In this approach the                     stratosphere, with rising motion at the summer pole and subsi-
observed meridional temperature structure (Fig. 9) allows us to                    dence at the winter pole, for times near the solstices (Hourdin
estimate the magnitude of the zonally averaged meridional cir-                     et al., 1995, 2004; Rannou et al., 2004).
culation associated with the observed stratospheric jet. Using                         Given an estimate of the vertical velocity, the order of mag-
the transformed Eulerian-mean approximation [see, e.g., Sec-                       nitude of the meridional velocity can be estimated by a scaling
tion 3.5 of Andrews et al. (1987) and Chapter 12 of Holton                         analysis of the continuity equation (32)
(2004)], the zonally averaged zonal momentum, continuity and
energy equations can be written as                                                  v̄   w̄
                                                                                       ∼ ,                                                    (35)
                                                                                 L D
∂ ū        1     ∂                        ∂ ū
     +              (ū cos φ) − f v̄ + w̄                                         where L and D are the horizontal and vertical scales on which
 ∂t      a cos φ ∂φ                        ∂z                                      the velocity varies. Using the vertical scale height for D, and
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
                                                   Temperatures in Titan’s stratosphere                                                            275


the radius of Titan as the horizontal scale L gives an estimate            The structure of Titan’s winter polar stratosphere is remark-
of the magnitude of the meridional velocity of v̄ ∼ 3 cm s−1 .         ably similar in many respects to the winter polar vortex on
The estimated velocities also allow us to determine a dynamical        Earth—a strong jet in the stratosphere which forms in au-
timescale for the meridional circulation tdyn = H /w̄ = L/v̄ ≈         tumn and decays in spring surrounding and isolating a po-
9 × 107 s (about 3 years), about three times larger than the ra-       lar stratosphere with anomalous composition—enhanced ni-
diative timescale but still somewhat shorter than the seasonal         triles on Titan, depleted CH4 and N2 O and enchanced HF on
timescales.                                                            Earth (Abrams et al., 1996a, 1996b)—and an elevated polar
   Given the meridional velocity scale, we can then estimate the       stratopause. In the terrestrial winter polar stratosphere, the ele-
order of magnitude of the EP flux divergence near the center           vated, warm stratopause is caused by downward motion driven
of the observed jet where the spatial derivatives of the zonal         by drag from breaking gravity waves (Hitchman et al., 1989),
velocity can be ignored, and assuming that the time derivative         while the composition is determined by a combination of the
of the zonal velocity is also small. With these assumptions,           downward motion and meridional mixing by planetary waves
                                                                     (Bacmeister et al., 1996). Models of Titan’s atmosphere by
     1                  ū tan φ                                       Hourdin et al. (2004) indicate that a similar process can explain
           ∇ · F = v̄            −f ,                       (36)
ρ0 a cos φ                  a                                          the enhancement of HCN in Titan’s winter polar stratosphere.
                                                                       With higher polar altitudes in permanent sunlight, and the in-
which gives us ∇ · F/(aρ0 ) ∼ 2 × 10−6 m s−2 at northern mid-
                                                                       creased aerosol abundance in the winter polar hood, heating
latitudes, giving a timescale of ∼9 × 107 s for the eddy forcing
                                                                       from absorption of sunlight by aerosols may also contribute to
of the mean zonal flow. Flasar and Conrath (1990) first calcu-
                                                                       the warm polar stratopause.
lated that the dynamical timescales on Titan are significantly
longer than the thermal timescale, based on Voyager IRIS ob-
                                                                       Acknowledgments
servations. They pointed out that, because of the coupling of
the temperature and winds through gradient wind balance, the
                                                                          We thank M.E. Segura, M.H. Elliott, J.S. Tingley, S. Al-
stratospheric temperatures will lag the thermal forcing by more        bright, E. Lellouch and P.N. Romani for their work on CIRS in-
than would expected simply from the thermal timescale.                 strument commanding, D.E. Jennings, A. Mamoutkin, R. Carl-
                                                                       son and V. Kunde for data calibration, and P.J. Schinder for
4.2. Winter stratospheric jet                                          calculating pointing information. This work has been supported
                                                                       by the NASA Cassini Project and by the NSF Planetary Astron-
    The zonal mean wind structure implied by the observed tem-         omy Program.
peratures (Fig. 10) is dominated by a single strong jet in the
middle stratosphere at northern midlatitudes, with only weak           References
winds in the southern hemisphere. The meridional structure of
the zonal mean winds is consistent with wind measurements              Abrams, M.C., Manney, G.L., Gunson, M.R., Abbas, M.M., Chang, A.Y., Gold-
from the November 2003 stellar occultations by Sicardy et al.             man, A., Irion, F.W., Michelsen, H.A., Newchurch, M.J., Rinsland, C.P.,
                                                                          Salawitch, R.J., Stiller, G.P., Zander, R., 1996a. ATMOS/ATLAS-3 obser-
(2006). Thermal infrared observations by Voyager, just after
                                                                          vations of long-lived tracers and descent in the Antarctic vortex in Novem-
the northern spring equinox, also showed strong winds at mid-             ber 1994. Geophys. Res. Lett. 23, 2341–2344.
northern latitudes, albeit considerably weaker than the Cassini        Abrams, M.C., Manney, G.L., Gunson, M.R., Abbas, M.M., Chang, A.Y., Gold-
data (∼80 m s−1 ) and with strong winds (∼60 m s−1 ) also at              man, A., Irion, F.W., Michelsen, H.A., Newchurch, M.J., Rinsland, C.P.,
high southern latitudes (Flasar and Conrath, 1990). Analysis of           Salawitch, R.J., Stiller, G.P., Zander, R., 1996b. Trace gas transport in the
                                                                          Arctic vortex inferred from ATMOS ATLAS-2 observations during April
the 28 Sgr stellar occultation, during early northern summer,
                                                                          1993. Geophys. Res. Lett. 23, 2345–2348.
showed a very strong jet (∼175 m s−1 ) in the southern hemi-           Andrews, D.G., Holton, J.R., Leovy, C.B., 1987. Middle Atmosphere Dynam-
sphere, but the data was not sensitive to winds in the northern           ics. Academic Press, Orlando.
hemisphere (Hubbard et al., 1993; Sicardy et al., 1999). Despite       Bacmeister, J.T., Schoeberl, M.R., Summers, M.E., Rosenfield, J.R., Zhu, X.,
the sparse temporal coverage, there is an apparent pattern of a           1996. Descent of long-lived trace gases in the winter polar vortex. J. Geo-
                                                                          phys. Res. 100, 11669–11684.
strong jet in the winter hemisphere, forming during the fall and       Bird, M.K., Allison, M., Asmar, S.W., Atkinson, D.H., Avruch, I.M., Dutta-
dissipating sometime after the spring equinox.                            Roy, R., Dzierma, Y., Edenhofer, P., Folkner, W.M., Gervits, L.I., Johnston,
    Poleward of the winter hemispheric jet, temperatures in the           D.V., Plettemeier, D., Pogrebenko, S.V., Preston, R.A., Tyler, G.L., 2005.
lower stratosphere are at least 25 K colder than at the equa-             The vertical profile of winds on Titan. Nature 438, 800–802.
tor, whereas the stratopause is roughly 20 K warmer than at the        Bouchez, A.H., 2003. Seasonal trends in Titan’s atmosphere: Haze, wind, and
                                                                          clouds. Ph.D. dissertation, California Institute of Technology. URL: http://
equator and is elevated by about two scale heights. Composition           resolver.caltech.edu/CaltechETD:etd-10272003-092206.
determinations from CIRS data also show a strong enhance-              Conrath, B.J., Gierasch, P.J., Ustinov, E.A., 1998. Thermal structure and para
ment in the abundance of nitriles and some trace hydrocar-                hydrogen fraction on the outer planets from Voyager IRIS measurements.
bons at latitudes within and poleward of the jet (Teanby et al.,          Icarus 135, 501–517.
2006, 2007; Coustenis et al., 2007; Vinatier et al., 2007). The        Coustenis, A., Bézard, B., 1995. Titan’s atmosphere from Voyager infrared
                                                                          observations. IV. Latitudinal variations of temperature and composition.
enhancement of nitriles was also seen by Voyager (Coustenis               Icarus 115, 126–140.
and Bézard, 1995), indicating that the enhancement persists into       Coustenis, A., Achterberg, R.K., Conrath, B.J., Jennings, D.E., Marten, A.,
early spring.                                                             Gautier, D., Nixon, C.A., Flasar, F.M., Teanby, N.A., Bézard, B., Samuel-
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
276                                                       R.K. Achterberg et al. / Icarus 194 (2008) 263–277


    son, R.E., Carlson, R.C., Lellouch, E., Bjoraker, G.L., Romani, P.N., Tay-            1993. The occultation of 28 SGR by Titan. Astron. Astrophys. 269, 541–
    lor, F.W., Irwin, P.G.J., Fouchet, T., Hubert, A., Orton, G.S., Kunde, V.G.,          563.
    Vinatier, S., Mondellini, J., Abbas, M.M., Courtin, R., 2007. The com-            Jacquinet-Husson, N., Scott, N.A., Chedin, A., Garceran, K., Armante, R.,
    position of Titan’s stratosphere from Cassini/CIRS mid-infrared spectra.              Chursin, A.A., Barbe, A., Birk, M., Brown, L.R., Camy-Peyret, C., Cler-
    Icarus 189, 35–62.                                                                    baux, C., Coheur, P.F., Dana, V., Daumont, L., Debaker-Barilly, M.R.,
Fast, K.E., Kostiuk, T., Espenak, F., Buhl, D., Livengood, T.A., Goldstein, J.,           Glaud, J.M., Goldman, A., Hamdouni, A., Hess, M., Jacquenmart, D.,
    1994. Direct measurement of Doppler shifts due to zonal winds on Titan.               Kipke, P., Mandin, J.Y., Massie, S., Mickhailenko, S., Nemchinov, V.,
    Bull. Am. Astron. Soc. 26, 1183.                                                      Nikitin, A., Newnham, D., Perrin, A., Perevalov, V.I., Regalia-Jarlot, L.,
Flasar, F.M., Conrath, B.J., 1990. Titan’s stratospheric temperatures: A case for         Rublev, A., Schreier, F., Schult, I., Smith, K.M., Tashkun, S.A., Teffo,
    dynamical inertia? Icarus 85, 346–354.                                                J.L., Toth, R.A., Tyuterev, V.G., Vander Auwera, J., Varanasi, P., Wagner,
Flasar, F.M., Samuelson, R.E., Conrath, B.J., 1981. Titan’s atmosphere: Tem-              G., 2005. The 2003 edition of the GEISA/IASI spectroscopic data base.
    perature and dynamics. Nature 292, 693–698.                                           J. Quant. Spectrosc. Radiat. Trans. 95, 429–467.
Flasar, F.M., Kunde, V.G., Abbas, M.M., Achterberg, R.K., Ade, P., Barucci,           Kostiuk, T., Livengood, T.A., Hewagama, T., Sonnabend, G., Fast, K.E., Mu-
    A., Bézard, B., Bjoraker, G.L., Brasunas, J.C., Calcutt, S., Carlson, R., Ce-         rakawa, K., Tokunaga, A.T., Annen, J., Buhl, D., Schmülling, F., 2005.
    sarsky, C.J., Conrath, B.J., Coradini, A., Courtin, R., Gautier, D., Gierasch,        Titan’s stratospheric zonal wind, temperature, and ethane abundance a
    P.J., Grossman, K., Irwin, P., Jennings, D.E., Lellouch, E., Mamoutkine,              year prior to Huygens insertion. Geophys. Res. Lett. 32, doi:10.1029/
    A.A., Marten, A., Meyer, J.P., Nixon, C.A., Orton, G.S., Owen, T.C., Pearl,           2005GL023897. 22205.
    J.C., Prange, R., Raulin, F., Read, P.L., Romani, P.N., Samuelson, R.E., Se-      Lacis, A.A., Oinas, V., 1991. A description of the correlated k distribution
    gura, M.E., Showalter, M.R., Simon-Miller, A.A., Smith, M.D., Spencer,                method for modeling nongray gaseous absorption, thermal emission, and
    J.R., Spilker, L.J., Taylor, F.W., 2004. Exploring the Saturn system in the           multiple scattering in vertically inhomogeneous atmospheres. J. Geophys.
    thermal infrared: The composite infrared spectrometer. Space Sci. Rev. 115,           Res. 96, 9027–9063.
    169–297.                                                                          Lemmon, M.T., Karkoschka, E., Tomasko, M., 1993. Titan’s rotation: Surface
                                                                                          feature observed. Icarus 103, 329–332.
Flasar, F.M., Achterberg, R.K., Conrath, B.J., Gierasch, P.J., Kunde, V.G.,
                                                                                      Lorenz, R.D., Smith, P.H., Lemmon, M.T., Karkoschka, E., Lockwood, G.W.,
    Nixon, C.A., Bjoraker, G.L., Jennings, D.E., Romani, P.N., Simon-Miller,
                                                                                          Caldwell, J., 1997. Titan’s north–south asymmetry from HST and Voy-
    A.A., Bézard, B., Coustenis, A., Irwin, P.G.J., Teanby, N.A., Brasunas, J.,
                                                                                          ager imaging: Comparison with models and ground-based photometry.
    Pearl, J.C., Segura, M.E., Carlson, R.C., Mamoutkine, A., Schinder, P.J.,
                                                                                          Icarus 127, 173–189.
    Barucci, A., Courtin, R., Fouchet, T., Gautier, D., Lellouch, E., Marten,
                                                                                      Niemann, H.B., Atreya, S.K., Bauer, S.J., Carignan, G.R., Demick, J.E., Frost,
    A., Prangè, R., Vinatier, S., Strobel, D.F., Calcutt, S.B., Read, P.L., Taylor,
                                                                                          R.L., Gautier, D., Haberman, J.A., Harpold, D.N., Hunten, D.M., Israel, G.,
    F.W., Bowles, N., Samuelson, R.E., Orton, G.S., Spilker, L.J., Owen, T.C.,
                                                                                          Lunine, J.I., Kasprzak, W.T., Owen, T.C., Paulkovich, M., Raulin, F., Raaen,
    Spencer, J.R., Showalter, M.R., Ferrari, C., Abbas, M.M., Raulin, F., Edg-
                                                                                          E., Way, S.H., 2005. The abundances of constituents of Titan’s atmosphere
    ington, S., Ade, P., Wishnow, E.H., 2005. Titan’s atmospheric temperatures,
                                                                                          from the GCMS instrument on the Huygens probe. Nature 438, 779–784.
    winds, and composition. Science 308, 975–978.
                                                                                      Pedlosky, J., 1987. Geophysical Fluid Dynamics, second ed. Springer-Verlag,
Folkner, W.M., Asmar, S.W., Border, J.S., Franklin, G.W., Finley, S.G., Gore-
                                                                                          New York, pp. 399–400.
    lik, J., Johnston, D.V., Kerzhanovich, V.V., Lowe, S.T., Preston, R.A., Bird,
                                                                                      Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 1994. Numerical
    M.K., Dutta-Roy, R., Allison, M., Atkinson, D.H., Edenhofer, P., Plette-
                                                                                          Recipes, second ed. Cambridge Univ. Press, Cambridge, Chapter 18.
    meier, D., Tyler, G.L., 2006. Winds on Titan from ground-based tracking
                                                                                      Rannou, P., Hourdin, F., McKay, C.P., Luz, D., 2004. A coupled dynamics–
    of the Huygens probe. J. Geophys. Res. 111, doi:10.1029/2005JE002649.
                                                                                          microphysics model of Titan’s atmosphere. Icarus 170, 443–462.
    E07S02.
                                                                                      Rodgers, C.D., 2000. Inverse Methods for Atmospheric Sounding. World Sci-
Fulchignoni, M., Ferri, F., Angrilli, F., Ball, A.J., Barn-Nun, A., Barucci, M.A.,
                                                                                          entific, London.
    Bettanini, C., Bianchini, G., Borucki, W., Colombatti, G., Coradini, M.,
                                                                                      Sicardy, B., Ferri, F., Roques, F., Brosh, N., Nevo, Y., Hubbard, W.B., Reitsema,
    Coustenis, A., Debei, S., Falkner, P., Fanti, G., Flamini, E., Gaborit, V.,
                                                                                          H.R., Blanco, C., Cristaldi, S., Carreira, E., Rossi, F., Lecacheux, J., Pau,
    Grard, R., Hamelin, M., Harri, A.M., Hathi, B., Jernej, I., Leese, M.R.,
                                                                                          S., Beisker, W., Bittner, C., Bode, H.-J., Bruns, M., Denzau, H., Nezel, M.,
    Lehto, A., Lion Stoppato, P.F., Lópes-Moreno, J.J., Mäkinen, T., McDon-
                                                                                          Riedel, E., Struckmann, H., Appleby, G., Forrest, R.W., Nicolson, I.K.M.,
    nell, J.A.M., McKay, C.P., Molina-Cuberos, G., Neubauer, F.M., Pirronello,
                                                                                          Miles, R., Hollis, A.J., 1999. The structure of Titan’s stratosphere from the
    V., Rodrigo, R., Saggin, B., Schwingenschuh, K., Seiff, A., Simões, F.,
                                                                                          28 Sgr occultation. Icarus 142, 357–390.
    Svedhem, H., Tokano, T., Towner, M.C., Trautner, R., Withers, P., Zarnecki,
                                                                                      Sicardy, B., Colas, F., Widemann, T., Bellucci, A., Beisker, W., Kretlow, M.,
    J.C., 2005. In situ measurements of the physical characteristics of Titan’s
                                                                                          Ferri, F., Lacour, S., Lecacheux, J., Lellouch, E., Pau, S., Renner, S.,
    environment. Nature 438, 785–791.
                                                                                          Roques, F., Fienga, A., Etienne, C., Martinez, C., Glass, I.S., Baba, D., Na-
Hitchman, M.H., Gille, J.C., Rodgers, C.D., Brasseur, G., 1989. The sepa-                 gayama, T., Nagata, T., Itting-Enke, S., Bath, K.-L., Bode, H.-J., Bode, F.,
    rated polar winter stratopause: A gravity wave driven climatological feature.         Lüdemann, H., Lüdemann, J., Neubauer, D., Tegtmeier, A., Tegtmeier, C.,
    J. Atmos. Sci. 46, 410–422.                                                           Thomé, B., Hund, F., deWitt, C., Fraser, B., Jansen, A., Jones, T., Schoe-
Holton, J., 2004. An Introduction to Dynamic Meteorology, fourth ed. Elsevier,            nau, P., Turk, C., Meintjies, P., Fiel, D., Frappa, E., Peyrot, A., Teng, J.P.,
    San Diego.                                                                            Vignand, M., Hesler, G., Payet, T., Howell, R.R., Kidger, M., Ortiz, J.L.,
Hourdin, F., Talagrand, O., Sadourny, R., Courtin, R., Gautier, D., McKay, C.P.,          Naranjo, O., Rosenzweig, P., Rapaport, M., 2006. The two Titan stellar
    1995. Numerical simulation of the general circulation of the atmosphere of            occultations of 14 November 2003. J. Geophys. Res. 111, doi:10.1029/
    Titan. Icarus 117, 358–374.                                                           2005JE002624. E11S91.
Hourdin, F., Lebonnois, S., Luz, D., Rannou, P., 2004. Titan’s stratospheric          Steck, T., Höpfner, M., von Clarmann, T., Grabowski, U., 2005. Tomographic
    composition driven by condensation and dynamics. J. Geophys. Res. 109,                retrieval of atmospheric parameters from infrared limb emission observa-
    doi:10.1029/2004JE002282. E12005.                                                     tions. Appl. Opt. 44, 3291–3301.
Hubbard, W.B., Sicardy, B., Miles, R., Hollis, A.J., Forrest, R.W., Nicolson,         Teanby, N.A., Irwin, P.G.J., de Kok, R., Nixon, C.A., Coustenis, A., Bézard, B.,
    I.K.M., Appleby, G., Beisker, W., Bittner, C., Bode, H.-J., Bruns, M., Den-           Calcutt, S.B., Bowles, N.E., Flasar, F.M., Fletcher, L., Howett, C., Taylor,
    zau, H., Nezel, M., Riedel, E., Struckmann, H., Arlot, J.E., Roques, F.,              F.W., 2006. Latitudinal variations of HCN, HC3 N, and C2 N2 in Titan’s
    Sevre, F., Thuillot, W., Hoffmann, M., Geyer, E.H., Buil, C., Colas, F.,              stratosphere derived from Cassini CIRS data. Icarus 181, 243–255.
    Lecacheux, J., Klotz, A., Thouvenot, E., Vidal, J.L., Carreira, E., Rossi, F.,    Teanby, N.A., Irwin, P.G.J., de Kok, R., Vinatier, S., Bézard, B., Nixon, C.A.,
    Blanco, C., Cristaldi, S., Nevo, Y., Reitsema, H.J., Brosch, N., Cernis, K.,          Flasar, F.M., Calcutt, S.B., Bowles, N.E., Fletcher, L., Howett, C., Tay-
    Zdanavicius, K., Wasserman, L.H., Hunten, D.M., Gautier, D., Lellouch, E.,            lor, F.W., 2007. Vertical profiles of HCN, HC3 N, and C2 N2 in Titan’s
    Yelle, R.V., Rizk, B., Flasar, F.M., Porco, C.C., Toublanc, D., Corugedo, G.,        atmosphere derived from Cassini/CIRS data. Icarus 186, 364–384.
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
                                                               Temperatures in Titan’s stratosphere                                                        277


Tokano, T., Neubauer, F.M., Laube, M., McKay, C.P., 1999. Seasonal variation         Vertical abundance profiles of hydrocarbons in Titan’s atmosphere at 15◦ S
   of Titan’s atmospheric structure simulated by a general circulation model.        and 80◦ N retrieved from Cassini/CIRS spectra. Icarus 188, 120–138.
   Planet. Space Sci. 47, 493–520.                                                 Worden, J.R., Bowman, K.W., Jones, D.B., 2004. Two-dimensional character-
Vinatier, S., Bézard, B., Fouchet, T., Teanby, N.A., de Kok, R., Irwin, P.G.J.,      ization of atmospheric profile retrievals from limb sounding observations.
   Conrath, B.J., Nixon, C.A., Romani, P.N., Flasar, F.M., Coustenis, A., 2007.      J. Quant. Spectrosc. Radiat. Trans. 86, 45–71.
```
