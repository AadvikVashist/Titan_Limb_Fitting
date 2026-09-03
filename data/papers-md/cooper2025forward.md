---
citation_key: "cooper2025forward"
title: "Extreme Forward Scattering Observed in Disk-Averaged Near-Infrared Phase Curves of Titan"
source_pdf: "data/papers/cooper2025forward.pdf"
source_pdf_sha256: "0090a3ecc4e11a09bc89f04c6a2d153ba25425c17d045a116143f1e9c98acc5d"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                                                Draft version September 11, 2025
                                                Typeset using LATEX twocolumn style in AASTeX631




                                                  Extreme Forward Scattering Observed in Disk-Averaged Near-Infrared Phase Curves of Titan
                                                         Chase Cooper       ,1, 2, 3 Tyler D. Robinson ,2, 3, 4 Jason W. Barnes        ,5 L. C. Mayorga       ,6 and
                                                                                                  Lily Robinthal 2, 3
                                                                         1 Department of Astronomy, University of Arizona, Tucson, AZ 85721, USA
                                                                      2 Lunar and Planetary Laboratory, University of Arizona, Tucson, AZ 85721, USA
                                                           3 Habitability, Atmospheres, and Biosignatures Laboratory, University of Arizona, Tucson, AZ 85721, USA
                                              4 NASA Nexus for Exoplanet System Science Virtual Planetary Laboratory, University of Washington, Box 351580, Seattle, WA 98195,

                                                                                                            USA




arXiv:2507.00924v3 [astro-ph.EP] 9 Sep 2025
                                                                            5 Department of Physics; University of Idaho; Moscow, ID 83844, USA
                                                                            6 Johns Hopkins Applied Physics Laboratory, Laurel, MD, 20723, USA



                                                                                                       ABSTRACT
                                                         Titan, with its thick and hazy atmosphere, is a key world in our solar system for understanding
                                                      light scattering processes. NASA’s Cassini mission monitored Titan between 2004 and 2017, where
                                                      the derived dataset includes a large number of whole disk observations. Once spatially integrated,
                                                      these whole disk observations reveal Titan’s phase-dependent brightness which can serve as an analog
                                                      for how hazy worlds might appear around other stars. To explore Titan’s phase curve, we present a
                                                      pipeline for whole disk Titan observations acquired by the Cassini Visual and Infrared Mapping Spec-
                                                      trometer (VIMS) spanning 0.9–5.1 µm. Application of the pipeline finds over 4,400 quality spatially-
                                                      and spectrally-resolved datacubes that were then integrated over Titan’s disk to yield phase curves
                                                      spanning 2–165° in phase angle. Spectra at near-full phase provide a useful approximation for Titan’s
                                                      geometric albedo, thus extending the spectral coverage of previous work. Crescent phase brightness
                                                      enhancements in the Cassini /VIMS phase curves are often more extreme than analogous results seen
                                                      at optical wavelengths, which can be explained by atmospheric transparency and haze scattering pro-
                                                      cesses. These results provide validation opportunities for exoplanet-focused spectral models and also
                                                      shed light on how extreme aerosol forward scattering could influence exoplanet observations and inter-
                                                      pretations.


                                                      Keywords: Titan, Phase Curves, Atmospheric Effects, Planetary Atmospheres

                                                                1. INTRODUCTION                                   and circulation (for a convenient summary of targets,
                                                Observations of exoplanets have revealed an impres-               see Table 1 in Parmentier & Crossfield 2018).
                                              sive diversity in planetary atmospheres. The atmo-                    In reflected-light direct imaging, current telescopes
                                              spheres of many Jupiter- to Neptune-sized exoplanets                are generally unable to resolve planetary companions
                                              have been successfully studied by looking at features in            and study phase curves, especially for rocky exoplanets
                                              transit (e.g., Madhusudhan et al. 2020; Kesseli et al.              (see, e.g., Wang et al. 2017). In the future, though,
                                              2022; Zhang et al. 2022; Madhusudhan et al. 2023;                   NASA’s under-development Habitable Worlds Obser-
                                              Rustamkulov et al. 2023), thermal emission (Charbon-                vatory (HWO; Feinberg et al. 2024) will provide the
                                              neau et al. 2005; Knutson et al. 2008; Kreidberg et al.             high-contrast imaging capabilities required to study a
                                              2014; Stevenson et al. 2014), and/or reflected light (Es-           wide range of planet types, including clement terrestrial
                                              teves et al. 2015; Hoeijmakers et al. 2018; Lendl et al.            worlds. Critically, repeat visits to planetary systems
                                              2020; Hooton et al. 2022; Brandeker et al. 2022) spec-              and targets will enable HWO to build up phase curves
                                              tra. For a subset of worlds — especially hot Jupiters —             for many worlds over the duration of the mission.
                                              observations of their changing thermal brightness with                Studying reflected-light phase curves can constrain
                                              star-planet-observer (i.e., phase) angle has helped to              surface and/or atmospheric properties of a planet. In
                                              constrain key aspects of atmospheric thermal structure              general, an object’s brightness will decrease as phase an-
                                                                                                                  gle increases, as progressively less of the planetary disk
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
2

is illuminated from the observer’s perspective. Moving        data of the moon in the ultraviolet, visible, infrared,
beyond simple illumination effects, the shape of phase        and radio wavelength regimes. Among the instruments
curves can be affected by phenomena that can reveal           on board the probe was the Visual and Infrared Mapping
details of the atmospheric and/or surface state. For          Spectrometer (VIMS) instrument, which took spatially
example, phase curves of Venus contain a feature at           resolved images of Titan in the optical (0.3–0.9 µm) and
high phase angles due to sulfuric acid droplets in the        near-infrared (0.9–5.1 µm) wavelength regimes (Brown
atmosphere (Arking & Potter 1968), while optical phase        et al. 2004).
curves of Titan show that the moon’s haze is responsi-          The primary focus of this work is to use
ble for forward scattering of optical light (Garcı́a Muñoz   Cassini /VIMS observations of Titan to generate phase
et al. 2017). Phase curves of solar system objects can        curves that span the near-infrared wavelength regime,
further serve as testing grounds for future direct obser-     thereby complementing the optical studies of Garcı́a
vations of exoplanets and the production of their phase       Muñoz et al. (2017). Section 2 covers data acquisi-
curves, as has been done with both Jupiter (Mayorga           tion and analysis using available resources, as well as
et al. 2016; Heng & Li 2021) and Saturn (Hedman &             the pipeline we produced to reduce spatially resolved
Stark 2015; Dyudina et al. 2016).                             images to disk-averaged reflectivity measures. In Sec-
   Modeling focused studies of exoplanet reflected-light      tion 3, near-infrared reflected-light phase curves of Ti-
phase curves have emphasized the potential for detect-        tan are presented and key features are noted. Section
ing specular reflection from liquid water oceans (i.e.,       4 explores possible explanations of the two phenomena
glint), thus revealing a habitable surface environment        mentioned in Section 3, and compares our findings to
(Williams & Gaidos 2008). Robinson et al. (2010) used         those of Garcı́a Muñoz et al. (2017). Finally, Section 5
Earth models to create phase curves of Earth both with        will summarize the findings of this work, as well as con-
and without glint contributions, and found that Earth         nect these findings to ongoing research of exoplanets.
with glint can appear twice as bright as Earth without
                                                                                   2. METHODS
glint at crescent phase. Detecting glint effects in nearby
exoplanet phase curves could be within the capabilities         Section 2.1 details the process of acquiring and filter-
of HWO (Vaughan et al. 2023). The presence of surface         ing VIMS data. Section 2.2 describes our automated
oceans may also be inferred from measuring reflected          disk detection process. Section 2.3 gives an overview
light polarization as a function of phase, indicated by       of how disk-averaged measurements are derived from
a peak polarization just past half-phase (Zugger et al.       VIMS pixel data.
2010).                                                                   2.1. File Vetting and Calibration
   Most fundamentally, an ocean glint signature in a
planetary phase curve is revealed through forward scat-
tering. Thus, the previously mentioned haze forward             Data were acquired using the PDS Image Atlas on
scattering detections from Garcı́a Muñoz et al. (2017)       NASA’s Planetary Data System. In the rest of this pa-
could present a potential false positive for glint. This      per, downloaded files are referred to as “cubes.” Cubes
connection between glint and Titan is particularly in-        contain general information about the time and duration
teresting given the collection of studies that use glint           Table 1. Filters used to remove unusable data
from Titan to study its seas (Stephan et al. 2010; Barnes
et al. 2011; Soderblom et al. 2012; Barnes et al. 2013,                 Filter Criteria      # of cubes removed
2014). The work presented here further explores aerosol
                                                                           Empty file                 35
forward scattering in Titan phase curves.
                                                                         Likely swaths              45,333
   In the solar system, Titan stands out among plane-
                                                                       Missing band data               97
tary objects as an example of a rocky/icy body with a
                                                                     Failed edge detection           4,104
thick atmosphere. Due to the large semi-major axis of
the Saturn system in comparison to that of the Earth,                  Failed calibration             83
both ground and Earth-orbiting telescopes can only view             Titan disk exceeds FOV           6,643
Titan at phase angles below about 6.5°. Fortunately,                    Incomplete disk              1,217
access to viewing angles not obtainable from Earth was                  Manual sorting                372
provided by NASA’s Cassini spacecraft, which made or-                 Total invalid cubes           57,884
bital observations of Saturn, its rings, and its moons                    Valid cubes               4,492
from 2004 to 2017 (Titan Discipline Working Group
2019). Cassini made regular flybys of Titan, taking
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                                                                                      3

of the observation, as well as the location and orientation   & Astrogeology Science Center 2024). The calibration
of Cassini at time of acquisition. Each cube includes         process formats each cube to be ISIS3-compliant, per-
352 two-dimensional arrays—one array for each VIMS            forms background subtraction and noise filtering, re-
band—and each array contains spatially-resolved counts        moves sources of error such as imaging artifacts and
accumulated during the cube’s exposure time. Arrays 1–        cosmic rays, and adds SPICE data. SPICE data in-
96 correspond to VIMS bands in optical light, while ar-       cludes information on the location and orientation of
rays 97–352 correspond to bands at infrared (IR) wave-        the spacecraft during its mission, as well as instrument
lengths (Brown et al. 2004). Our analyzed cubes were          details, target information and more (Acton 1996; Ac-
acquired throughout the full duration of the Cassini          ton et al. 2018). The calibration also converted data
mission (2004–2017), were taken at phase angles from          numbers, which are correlated with photon counts, to
2° to 165°, and contained imaging data at all infrared        spatially-resolved I/F values, a unit of measurement of
bands covered by the VIMS-IR camera (0.9–5.1 µm). A           an object’s reflectivity per steradian (see Section 2.3).
total of 62,376 cubes of Titan were downloaded.               Cubes were calibrated this way in order to be compat-
  We placed a constraint that cubes be at least 12 pix-       ible with the pyvims Python library, the primary tool
els to a side in size. Critically, this criterion removed     used for the analysis of cube data. The pyvims soft-
the many swaths in our dataset — long, narrow scans           ware suite (described in Le Mouélic et al. 2019) facili-
of Titan’s surface taken during close approaches. We          tates the analysis of VIMS cubes and can extract from
also exclude cubes taken at distances where the appar-        a calibrated cube the full suite of ephemeris and point-
ent diameter of Titan exceeds the maximum cube size           ing quantities, including sub-spacecraft coordinates and
of 64 pixels to a side. The diameter of Titan in pixels is    phase angle.
calculated as:                                                  VIMS data taken in the 0.9–1.3 µm wavelength range
                               2      R                       suffered from extensive saturation issues, especially at
                    2Rpixel = tan( )                          high phase angles. The cause is likely longer integra-
                               S      d
                                                              tion times – cubes with longer exposure times showed a
where S = 5 · 10−4 radians is the field of view of a VIMS
                                                              greater degree of saturation. The phase curves produced
pixel on a side, R is the radius of Titan in km, and d is
                                                              from these data show greater spread and do not closely
the distance of the spacecraft at time of observation. By
                                                              resemble a continuous curve. As a result, we are un-
letting 2Rpixel = 64 and solving for d, we get a lower dis-
                                                              able to properly analyze most of these data. We include
tance limit of approximately 180,000 km. Cubes missing
                                                              these data in later figures for comparisons with previ-
data from more than 10% of VIMS bands were also re-
                                                              ous works. Similar saturation issues extended to a few
moved.
                                                              cubes at longer wavelengths at higher phase angles, and
   We then used a simple Canny edge detection method
                                                              these cubes were excluded from our dataset via manual
to test for structures in cube data, and excluded cubes
                                                              sorting.
showing no structure. Canny edge detection algorithms
are a family of algorithms that identify edges and bound-
aries in pixelated images (Canny 1986). In the case of                    2.2. Automated Disk Detection
Titan images, Canny edge detection returns an array              The automatic detection of Titan within images was
that identifies pixels which either lie on the edge of the    achieved using a pipeline written in Python, and mirrors
disk of Titan or on the boundaries between geographi-         the disk detection process used by Strauss et al. (2024).
cal regions. By requiring that cubes have structure as        A visual summary of the process can be seen in Fig-
determined by this algorithm, we exclude cubes that do        ure 1. First, the radius of Titan RTitan is determined by
not contain the disk of Titan, that zoom in on a subsec-      adding the wavelength-dependent atmospheric height of
tion of the disk, or have erroneous data. Edge detection      Titan (Robinson et al. 2014) to the solid body radius of
was also used to remove cubes not containing the full         2, 575 km (Zebker et al. 2009). Effective height data at
disk of Titan. Finally, remaining cubes were manually         wavelengths corresponding to VIMS-IR bands 345–352
inspected to ensure they met our criteria. Table 1 breaks     were absent, so the effective height used for band 344 was
down the different criteria used to sort the cubes, in the    applied to these bands as well. Then Titan’s radius in
order they were applied, and the amount of cubes re-          units of VIMS pixels is calculated as Rpxl = ⌈RTitan /S⌉,
moved by each criterion. Once the filtering process was       where S is the average surface resolution of the image in
finished, our dataset contained 4,492 cubes.                  kilometers per pixel as determined with pyvims. This
   All cubes were passed through the multi-step calibra-      radius is increased by 1 in cases where the phase angle
tion process described by Le Mouélic et al. (2019) us-       of the image is above 120°, because the atmosphere ap-
ing the USGS ISIS3 software package (Kelvin Rodriguez         pears much brighter due to forward scattering of light
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
4




Figure 1. Major steps in the disk detection process on two cubes: (far left) an unaltered cube; (center left) edge pixels
identified using Canny edge detection – some pixels along terrain boundaries are falsely identified as edge pixels; (center right)
the accumulator array after applying a circle Hough transform; (far right) the final array, containing only data from the disk of
Titan. Data were taken at 2 µm.

by atmospheric aerosols, which increases the apparent               described above converts data numbers from the raw
radius of Titan. Increasing this radius by more than                Cassini data to I/F values, with units of inverse steradi-
1 pixel did not substantially change the disk-averaged              ans. Because we aim to simulate point-like observations
measurements.                                                       of Titan rather than the spatially resolved observations
   Next, we used a Canny edge detection algorithm to                provided by Cassini, we convert the spatial I/F values
identify pixels on the edge of Titan’s visible disk. A              across Titan’s disk to a single “disk averaged” quantity.
circle Hough transform is then used to locate the (ap-              The disk-averaged Ag Φ(α) for Titan is calculated as,
proximate) center of the disk (Xie & Ji 2002). In the
circle Hough transform, an accumulator array with the                                          I¯  d2 Ω X
                                                                                 Ag Φ(α) = π      = sc 2  (I/F )i             (1)
same dimensions as the image is created. Circles with                                          F    πR i
a radius of Rpxl pixels are overlaid on the array, cen-
tered on each edge pixel location in the edge pixel array,          where ¯I/F is the disk-averaged intensity-to-incident flux
with each pixel covered by a circle having its accumula-            ratio, dsc is the distance from Cassini to the center of
tor array value incremented. After each circle has been             Titan in km, Ω = 2.5 × 10−7 is the solid angle of a
overlaid, the pixel with the highest value is taken to be           VIMS pixel in steradians, R is the solid body radius
the center of the disk, and an array is created where               of Titan in km, and the sum is over individual I/F
each pixel within Rpxl pixels of the center pixel iden-             values of all pixels on the disk. pyvims does not provide
tified above has a value of 1, and all other pixels have            information on errors in individual pixel measurements,
a value of 0. By multiplying the original image array               however integrating over the disk significantly reduces
by this masking array, the resulting array only contains            random errors.
data from pixels lying on the disk of Titan.
                                                                                           3. RESULTS
                2.3. Ag Φ(α) Calculation                              A disk-averaged Ag Φ(α) spectrum recorded at the
   The quantity Ag Φ(α) is the product of an object’s ge-           smallest phase angle in our dataset (2°) is shown in Fig-
ometric albedo, Ag , and its planetary phase function,              ure 2. As the variation of Ag Φ(α) with phase at these
Φ(α). In planetary science, Ag Φ(α) is often used as a              low phase angles is small, this spectrum is a reasonable
metric of a planetary object’s reflectivity, as it quan-            stand-in for a geometric albedo spectrum (which is for-
tifies the ratio of the radiance received from an object            mally defined at full phase). Also included for compar-
to the irradiance that object receives. As a reflectance,           ison, and extension, are optical data from Karkoschka
it is a unitless quantity. The cube calibration process             (1998), which corroborate the full-phase values derived
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                                                                                             5




Figure 2. A near-full phase Ag Φ(α) spectrum of Titan. Data from Karkoschka (1998) are included for comparison. The
spectrum was obtained by taking the disk-averaged spectra of 10 cubes with phase angle α < 3°. Data were acquired at very
low phase angle and, thus, approximate the geometric albedo spectrum of Titan. Certain spectral features and their sources are
also identified. Features at 3.31 µm and 4.54 µm are discussed in Section 4.2. Features in the Cassini data are consistent with
the model results from Es-sayeh et al. (2023) and recent James Webb Space Telescope data from Nixon et al. (2025).

                                                                   creasing Ag Φ(α) with wavelength, consistent with the
                                                                   trend of decreasing haze single scattering albedo with
                                                                   increasing wavelength (Tomasko et al. 2008). Notable
                                                                   atmospheric windows with sensitivity to the deeper at-
                                                                   mosphere/surface occur at 0.94, 1.08, 1.28, 1.6, 2.0, 2.7,
                                                                   2.8, and 5.0 µm.
                                                                     Figure 3 shows the value of Ag Φ(α) for Titan as a
                                                                   function of wavelength and phase angle. These disk-
                                                                   averaged Ag Φ(α) values are not normalized to full phase,
                                                                   so are not formal phase functions. Cubes were binned
                                                                   by phase angle into one-degree bins, and for bins with
                                                                   multiple cubes an average Ag Φ(α) value was computed.
                                                                   Vertical black stripes represent phase angle bins with no
                                                                   viable cubes. The region on the right edge of the figure
                                                                   indicates where significant forward scattering causes Ti-
                                                                   tan to appear bright, as discussed above. These regions
                                                                   largely correspond to continuum outside of Titan’s at-
                                                                   mospheric absorption features.
                                                                     Figure 4 shows a selection of phase curves at se-
                                                                   lect wavelengths produced by our pipeline. The phase
Figure 3. Ag Φ(α) of Titan as a function of wavelength             curve of Titan is decidedly not Lambertian, in contrast
and phase angle. Vertical black lines indicate phase an-           to those of other major Saturnian satellites which are
gles at which there were no cubes to consider. Bright hori-        better-approximated by a Lambert phase function (Bu-
zontal bands are associated with continuum between strong          ratti & Veverka 1984). At every wavelength, the bright-
methane absorption bands.
                                                                   ness of Titan follows a smooth curve that initially de-
                                                                   creases as the phase angle increases from roughly 10°
by Garcı́a Muñoz et al. (2017). The discrepancy be-               to 100°. At phase angles above about 100°, the bright-
tween the findings of this work and that of Karkoschka             ness of the disk then increases due to forward scattering
is explained by the saturation issues mentioned at the             of sunlight by the atmosphere. At phase angles above
end of Section 2.1. Figure 2 highlights just the VIMS              about 140°, Titan appears brighter than when observed
(near) geometric albedo spectrum with notable bands                at near-full phase. While this effect has been observed in
due to CH4 , CO, and CH3 D indicated. Continuum re-                the 0.3–1 µm range before (Tomasko et al. 2008; Doose
gions between absorption bands show a generally de-                et al. 2016; Garcı́a Muñoz et al. 2017), our analysis con-
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
6

firms that significant forward scattering occurs at most     tion (West & Smith 1991) results in stronger brightening
infrared VIMS bands. The larger spread in Ag Φ(α) at         between quadrature phase and full phase.
smaller phase angles in the 2 µm plot in Figure 4 is ap-        The wavelength dependence of the normalized cres-
parent in all phase curves recorded at wavelengths with      cent phase peak Ag Φ(α) is primarily controlled by
sensitivity to the deep atmosphere and/or surface and        aerosol scattering and gas absorption optical depths.
will be discussed later.                                     Previous estimates on the size of Titan aerosols are on
   Polynomial fits of degree 10 to select phase curves       the order of 1 µm (Rages et al. 1983; Waite et al. 2007;
at wavelengths between 0.88µm and 5.12µm are shown           Tomasko et al. 2008), and strong forward-scattering is
in Figure 5, normalized to lowest-phase Ag Φ(α) val-         to be expected when particle sizes are approximately
ues, thereby approximating the planetary phase function      the same size as the wavelength of incident light. How-
Φ(α). All phase curves show a gradual decrease from          ever, a complete description of the scattering behavior
roughly 2°–80° before leveling out. Significant bright-      of Titan’s hazes requires consideration of the fractal ag-
ness surges occur beginning around 130°. The effect is       gregate nature of these hazes (see West & Smith 1991;
weakest at shorter wavelengths, though for some bands        Tomasko et al. 2008; Lavvas et al. 2010). The strong for-
this muting may be due to aforementioned saturation          ward peak in the aerosol single-scattering phase function
issues. Some normalized phase curves taken at wave-          explains the general behavior of larger disk-averaged
lengths between 3 and 4µm show nearly an order of mag-       Ag Φ(α) values at crescent phases versus near full phase
nitude increase between near-full and crescent phases as     given the ubiquity of haze aerosols in Titan’s atmo-
a result of significant forward scattering by atmospheric    sphere (Garcı́a Muñoz et al. 2017). Increasing at-
aerosols.                                                    mospheric aerosol transparency at longer wavelengths
   Figures 6 and 7 emphasize the value of disk-averaged      causes scattering to tend towards the single-scattering
Titan observations as an analog for a hazy exoplanet         regime, which explains the general trend with wave-
via comparisons to an Earth phase curve and Earth            length in Figure 5. The figure breaks from this trend
color-color data, respectively. Published measurements       around 4.3 µm, though the exact reason for this is un-
of Earth’s phase curve at wavelengths corresponding          known. At wavelengths with strong gas absorption,
to VIMS do not exist, so the broadband visible (0.4–         photons incident at the near-full phase geometry would
0.7 µm) Earthshine data are shown (Qiu et al. 2003;          typically require multiple scatterings to escape the at-
Pallé et al. 2003). Similarly, phase-dependent color mea-   mosphere so are, instead, absorbed along such a path.
surements for Earth at VIMS-analogous wavelengths do         At crescent phases, though, only a limited number of
not exist, so we adopt high-fidelity, phase-dependent        scattering events are required to direct photons towards
simulations from the Virtual Planetary Laboratory 3-         the observer (spacecraft), making observations at these
D Spectral Earth Model (Robinson et al. 2010).               wavelengths and phase less sensitive to gas absorption
                                                             in the deeper atmosphere and more sensitive to single-
                                                             scattered radiation. A comprehensive spectral model of
                    4. DISCUSSION                            aerosol scattering could further explain these results but
    4.1. Phase Curve Structure and Comparisons to            falls beyond the scope of this work.
                      Optical Results                           The general shape of near-infrared phase curves pro-
                                                             duced with our pipeline matches that of the optical
  At all wavelengths, the phase curves presented here
                                                             phase curves produced by Garcı́a Muñoz et al. (2017). A
share the same general structure. Near full-phase, disk-
                                                             comparison of our results to data from this earlier work
averaged Ag Φ(α) starts modest and decreases as phase
                                                             is shown in Figure 8. Both sets demonstrate intense for-
angle increases, with minimum disk-averaged Ag Φ(α)
                                                             ward scattering from Titan’s atmospheric aerosols. The
measurements occurring near ∼ 100°. Curves then
                                                             onset of brightness surges occur at similar phase angles
sharply increase, beginning around ∼140° phase. Maxi-
                                                             in our phase curves (∼135°) as in those of Garcı́a Muñoz
mum disk-averaged Ag Φ(α) measurements occur at high
                                                             et al. (2017) (∼150°). However, the relative strength of
phase at nearly all wavelengths; at the shortest wave-
                                                             forward scattering is evidently much different between
lengths accessible to VIMS, disk-averaged Ag Φ(α) at
                                                             the two wavelength regimes. Several near-infrared phase
lowest and highest phase are comparable, though this is
                                                             curve fits across our sampled phase angle range (2°–165°)
partially due to aforementioned saturation problems at
                                                             show approximate Φ(α) values of 10 or more at high
these wavelengths. At atmospheric window wavelengths
                                                             phase; optical light phase curve values across the same
(e.g., the 2 µm plot in Figure 4), photons are scattered
                                                             intervals are of order unity (consistent with results at
fewer times overall, in many cases only once, and the
                                                             the shortest VIMS-IR wavelengths). The differences in
back-scattering peak in the haze’s scattering phase func-
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                                                                                             7




Figure 4. A suite of Titan phase curves taken at different wavelengths. At each wavelength, extreme forward scattering is
apparent at high phase angles, even at wavelengths where the disk of Titan appears dim when observing at low phase angles
(e.g. 4.0 µm). The effect is strong enough that Titan’s disk-averaged brightness is greater at high phase than at low phase for
almost all wavelengths. The top left phase curve, taken at 0.88 µm, suffers from saturation-related issues at high phase.




Figure 5. Left: A selection of approximate planetary phase functions from across the near-infrared spectrum accessible to
VIMS-IR. The color of fits are indicative of the VIMS-IR band at which they were generated. Curves are normalized to 10° due
to a sparse set of data at phase angles below 10°. Right: The value of Titan’s planetary phase function Φ(α) at the max phase
angle in our dataset, α = 165°. The magnitude of this value broadly scales with wavelength. Some dips within the structure
correspond to spectral continuum regions, similar to features in the infrared spectrum of Titan (grey; spectrum averaged over
all cubes)
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
8




Figure 6. Comparison of a Titan phase curve at a contin-
uum wavelength (0.9 µm) to Earth’s broadband visible phase
curve (data from Qiu et al. 2003; Pallé et al. 2003).




                                                               Figure 8. Top: A comparison of two phase curves from the
                                                               present work (3.00 µm in teal; 5.00 µm in red) and two phase
                                                               curves from Garcı́a Muñoz et al. (2017) (569 nm in light gray;
                                                               938 nm in dark gray). Phase curves are normalized to their
                                                               respective lowest-phase observation and offset from one an-
                                                               other for clarity. Bottom: A comparison of the 938 nm phase
                                                               curve from Garcı́a Muñoz et al. (2017) with our phase curve
                                                               at 933 nm. The curves share a similar shape, though our
                                                               phase curve has much more noise because of saturation is-
                                                               sues.
Figure 7. Comparison of Titan (orange) and Earth (blue)
phase-dependent brightness in color-color space. Earth val-    passing these features from the color-contour diagram
ues are from spectral models in Robinson et al. (2010).
                                                               in Figure 3, demonstrating how both features main-
Adopted spectral elements at 1.08 µm, 1.28 µm, and 1.60 µm
are continuum for both worlds and within the anticipated       tain a near-constant Ag Φ(α) with increasing phase an-
spectral coverage for HWO. Point saturation indicates phase    gle, which further suggests this feature is due to emis-
angle, indicating that Earth and Titan separate well in this   sion rather than phase-dependent forward scattering.
color-color space except at very large phase angles.           Garcı́a-Comas et al. (2011) explain that the 3.31 µm fea-
                                                               ture is non-local thermodynamic equilibrium (non-LTE)
the extent of the crescent phase peak in Titan’s phase         emission from upper-atmospheric methane driven by ab-
curves at optical versus near-infrared wavelengths aligns      sorption of solar radiation. The isotropic emission from
with the earlier physical explanation rooted in haze scat-     this non-LTE source results in a phase curve shape that
tering.                                                        is distinctly less-structured than at wavelengths domi-
                                                               nated by absorption or scattering processes, as seen in
             4.2. Notable Spectral Features                    Figure 3. This emission is prevalent enough to produce
  Notable features in apparent emission are seen in            a spectral feature even when averaged over the disk of
Figure 2 at 3.31 µm and 4.54 µm. For context, Fig-             Titan. While both CO and CH3 D contribute to non-
ure 9 shows two high-quality examples of whole disk            LTE emission near 4.54 µm, where the ν2 fundamental
images of Titan at the wavelengths with these no-              of CH3 D gives rise to a sharp feature at 4.54 µm (Baines
table emission features. The 3.31 µm emission primarily        et al. 2006), emission from these species is not strong
comes from the dayside limb while the 4.54 µm emis-            enough to explain a sharp increase in Ag Φ(α) seen in
sion is structured and concentrated near the southern          some cubes at this wavelength. Thus we suspect that
pole. Figure 10 highlights the spectral region encom-          a subset of cubes are affected by residual instrument or
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                                                                                                            9

                                                                     The 2 µm phase curve in Figure 4 reveals a spread
                                                                  whose strength increases with decreasing phase angle.
                                                                  The spread far exceeds the anticipated error in disk-
                                                                  averaged values, especially at these longer wavelengths
                                                                  that do not suffer from saturation issues. This behavior
                                                                  is apparent in phase curves from all wavelengths with
                                                                  deep atmosphere and/or surface sensitivity, thus indi-
                                                                  cating that processes in the deep atmosphere or on the
                                                                  surface are driving the phase curve variability. For ex-
                                                                  ample, the visibility of Xanadu — a large, spectrally dis-
                                                                  tinct region on the surface of Titan centered at about
Figure 9. Images of Titan at 3.31 µm (left) and 4.54 µm           100°S longitude and 10°S latitude (Coustenis et al. 1995;
(right), showing source locations of emission.                    Lellouch et al. 2004; Negrão et al. 2006) — should intro-
                                                                  duce variability at surface-sensitive wavelengths. How-
                                                                  ever, we found no substantive correlation between the
                                                                  sub-spacecraft coordinates of Cassini at the time of cube
                                                                  acquisitions and the relative Ag Φ(α) of the cubes (Fig-
                                                                  ure 12).
                                                                     An indication that weather plays some role in causing
                                                                  variability in phase curves with deep atmospheric sensi-
                                                                  tivity is highlighted by flybys 253TI-255TI, flyby 264TI,
                                                                  and flyby 273TI (hereafter referred to as groups 1, 2,
                                                                  and 3, respectively), which dominate the brightest, low-
                                                                  phase observations in the 2 µm curve in Figure 4. Look-
                                                                  ing at visualizations of cubes from each group reveals a
                                                                  set of clouds above Titan’s north pole, surrounded by
                                                                  a ring of circumpolar clouds at about 50°N (Figure 11).
                                                                  All three groups took place during early- to mid-spring
                                                                  2017. The cloud ring is most easily viewed in early group
                                                                  3 cubes, while the polar cloud can easily be seen in cubes
Figure 10. Non-local thermodynamic equilibrium emission           from groups 1 and 2. The location and shapes of cloud
signatures in the spectrally-resolved phase curves of Titan.
                                                                  structures are in agreement with the findings of Yahn
As explained in the text, the feature at 4.54 µm is likely af-
fected by issues related to the instrument or cube calibration.
                                                                  et al. (2025), who discovered that, from September 2016
                                                                  to September 2017, clouds on Titan were concentrated
                                                                  around 0-120°W longitude, 50°N latitude, and had large
calibration issues at this wavelength, such as an order-
                                                                  aspect ratios indicating long, thin structures.
sorting filter change, hot pixel, or other impediment in
channel 317.

                                                                            4.4. Applications and Future Work
               4.3. Phase Curve Variability
                                                                    The shape of Titan’s phase curves are directly related
                                                                  to the physical properties of its hazes, such as their size,
                                                                  opacities, and single scattering albedo (Tomasko et al.
                                                                  2008; Doose et al. 2016; Garcı́a Muñoz et al. 2017).
                                                                  Their wavelength-properties have been deduced previ-
                                                                  ously from similar measurements at visible and some
                                                                  near-infrared wavelengths. The findings of this work
                                                                  could be used to extend the results of past studies by
                                                                  further constraining haze parameters at near-infrared
                                                                  wavelengths. This, in turn, would find applications in
                                                                  determining Titan’s energy budget, modeling its atmo-
Figure 11. Images from identified “bright” flyby groups dis-      spheric thermodynamics, or validating models thereof
playing likely cloud structures. Images are taken at 2.8 µm.
                                                                  (Garcı́a Muñoz et al. 2017).
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
10

                                                               a characteristic inflection point not seen in the Earth
                                                               data. Phase-dependent color-color comparisons of Earth
                                                               and Titan at red/NIR wavelengths, as shown in Fig-
                                                               ure 7, also demonstrate that, except at extreme cres-
                                                               cent phases, brightness measurements at a few contin-
                                                               uum wavelengths well-separates Titan-like worlds from
                                                               potential exo-Earths.

                                                                                 5. CONCLUSIONS
                                                                 In this work, we used disk-averaged measurements of
                                                               Titan’s Ag Φ(α) to produce the first reflected-light phase
                                                               curves of the moon that span the full near-infrared wave-
                                                               length range. Key features of these phase curves are tied
                                                               to atmospheric properties of Titan, and an understand-
                                                               ing of these relations will be applicable to future efforts
Figure 12. The spatial distribution of cubes by their sub-
                                                               to directly image rocky exoplanets. Our key findings are
spacecraft coordinates. Most cubes used in the present study
were acquired in quick succession by Cassini during Titan      as follows:
passes, hence the clustering.
                                                                  • We developed a pipeline that takes Cassini images
                                                                    of Titan and automatically determines the disk-
  The presence of aerosols in Titan’s atmosphere ev-
                                                                    averaged Ag Φ(α) of Titan at all VIMS-IR bands.
idently dominates near-infrared reflected-light phase
                                                                    Our dataset of 4,492 images spanned a phase angle
curves. It is reasonable, then, to expect similar mech-
                                                                    range of 2° to 165°. We produced phase curves of
anisms to occur in the atmospheres of hazy exoplan-
                                                                    Titan in the near-infrared regime (0.88–5.12 µm)
ets. Models (Hu et al. 2013; Adams et al. 2019; Gao
                                                                    that show strong non-Lambertian effects.
et al. 2020) and laboratory results (He et al. 2018; Hörst
et al. 2018) have shown that exoplanets with a wide               • Our near-infrared reflectance spectrum of Titan
range of atmospheric conditions are capable of hosting              at 2° is a useful approximation to Titan’s geomet-
aerosols, and many strong detections of aerosols come               ric albedo. This observation extends important
from warm/hot giant planets (e.g. Estrela et al. 2021;              previous results at primarily optical wavelengths
Malsky et al. 2025). Observations of Titan can serve                (Karkoschka 1998) and also agrees with very re-
as a rare opportunity to study the phase curves of cool,            cent, high-quality near-full phase Ag Φ(α) spectral
terrestrial exoplanets with thick atmospheres, and espe-            observations at near-infrared and infrared wave-
cially the impact of atmospheric aerosols thereon.                  lengths from James Webb Space Telescope (Nixon
  The identification and study of aerosol forward-                  et al. 2025, their Figure 4b).
scattering can also avoid false positive detections of
glint, whose strong specular scattering in the forward            • Phase curves of Titan at wavelengths with sur-
direction has been proposed as an avenue towards sur-               face sensitivity show enhanced backscattering at
face ocean detection (Williams & Gaidos 2008; Robin-                low phase angles as well as increased variability at
son et al. 2010; Vaughan et al. 2023). Misidentifica-               these phase angles due to clouds in the lower/deep
tion of atmospheric forward scattering as glint could               atmosphere. Additionally, spectra show evidence
lead to an ocean false positive detection as both phe-              of a feature at 3.31 µm attributable to CH4 non-
nomena contribute at larger phase angles. Fortunately,              LTE emission. While this feature has been ob-
Figure 6 shows that, at least for Titan-like hazes, such            served and studied before in spatially-resolved ob-
false positive scenarios are unlikely. Importantly, while           servations (Baines et al. 2005; Garcı́a-Comas et al.
ocean glint increases Earth’s crescent phase brightness             2011), we found that this feature is observable even
by as much as 50% at these wavelengths (Robinson et al.             when averaged over the entire disk.
2010), this glint enhancement is markedly smaller than
haze forward scattering effects in Titan’s phase curve.           • Phase curves of Titan at most wavelengths are
Distinguishing Titan haze-like effects for an analogous             dominated by forward scattering due to atmo-
exoplanet from glint requires access to only modest cres-           spheric aerosols. At all investigated wavelengths
cent phases (110–120°), where the Titan phase curve has             above about 1.3 µm, forward scattering by hazes
                                                                    causes Titan’s disk-averaged Ag Φ(α) to be greater
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                                                                                                                          11

      at crescent phase than at near-full phase. At cer-       with later editing support from TR, JB, and LM. We
      tain wavelengths with significant atmospheric ab-        thank A. Garcı́a Muñoz for sharing optical phase curve
      sorption, crescent phase Ag Φ(α) can be an order of      data for Titan. We also thank the two anonymous re-
      magnitude greater than at near-full phase (Figure        viewers for their insightful comments and suggestions
      5).                                                      for improvement.

    • Future reflected light direct imaging of rocky exo-        Software:
      planets would yield phase curves for these distant       astropy (Astropy Collaboration et al. 2013, 2018, 2022),
      worlds, and a detailed understanding of forward          ISIS3 (Rodriguez & Astrogeology Science Center 2024),
      scattering effects on phase curves can reduce the        matplotlib (Hunter 2007), numpy (Harris et al. 2020),
      chances of false positive detections of other phe-       pyvims (Le Mouélic et al. 2019), scipy (Virtanen et al.
      nomena (e.g., ocean glint) that occur at similar         2020)
      phase angles.
                                                                   7. APPENDIX - DATA ACQUISITION AND
              6. ACKNOWLEDGMENTS                                                      CALIBRATION
  The present work was initially supported by a seed             Data used in this work were acquired using the PDS
grant from the Arizona Astrobiology Center, led by             Image Atlas. The data were part of the ”VIMS Ob-
CC. All authors acknowledge support through an                 servations from the Cassini Tour of the Saturn System”
award from NASA’s Exoplanets Research Program                  dataset (Brown & VIMS Science Team 2020). Data were
(No. 80NSSC25K7149). CC and TR conceived of this               selected from the dataset by requiring that they target
study, which was further developed by CC, TR, JB, and          Titan. The ISIS3 package is developed by the United
LM. All pipeline materials, results, and figures were cre-     States Geological Survey and can be accessed from their
ated by CC. CC wrote all early drafts of this manuscript,      GitHub page.


                                                        REFERENCES
Acton, C., Bachman, N., Semenov, B., & Wright, E. 2018,        Barnes, J. W., Soderblom, J. M., Brown, R. H., et al. 2011,
  Planetary and Space Science, 150, 9,                           Icarus, 211, 722,
  doi: https://doi.org/10.1016/j.pss.2017.02.013                 doi: https://doi.org/10.1016/j.icarus.2010.09.022
Acton, C. H. 1996, Planetary and Space Science, 44, 65,        Barnes, J. W., Clark, R. N., Sotin, C., et al. 2013, The
  doi: https://doi.org/10.1016/0032-0633(95)00107-7              Astrophysical Journal, 777, 161,
Adams, D., Gao, P., de Pater, I., & Morley, C. V. 2019,          doi: 10.1088/0004-637X/777/2/161
 ApJ, 874, 61, doi: 10.3847/1538-4357/ab074c                   Brandeker, A., Heng, K., Lendl, M., et al. 2022, A&A, 659,
Arking, A., & Potter, J. 1968, Journal of Atmospheric            L4, doi: 10.1051/0004-6361/202243082
  Sciences, 25, 617 , doi: 10.1175/1520-0469(1968)025⟨0617:
                                                               Brown, R. H., & VIMS Science Team. 2020, VIMS
  TPCOVA⟩2.0.CO;2
                                                                 Observations from the Cassini Tour of the Saturn
Astropy Collaboration, Robitaille, T. P., Tollerud, E. J.,
                                                                 System, doi: 10.17189/1504134
  et al. 2013, A&A, 558, A33,
                                                               Brown, R. H., Baines, K. H., Bellucci, G., et al. 2004, Space
  doi: 10.1051/0004-6361/201322068
                                                                 Science Reviews, 115, 111–168,
Astropy Collaboration, Price-Whelan, A. M., Sipőcz, B. M.,
                                                                 doi: 10.1007/s11214-004-1453-x
  et al. 2018, AJ, 156, 123, doi: 10.3847/1538-3881/aabc4f
                                                               Buratti, B., & Veverka, J. 1984, Icarus, 58, 254,
Astropy Collaboration, Price-Whelan, A. M., Lim, P. L.,
                                                                 doi: 10.1016/0019-1035(84)90042-3
  et al. 2022, ApJ, 935, 167, doi: 10.3847/1538-4357/ac7c74
Baines, K. H., Drossart, P., Momary, T. W., et al. 2005,       Canny, J. 1986, Pattern Analysis and Machine Intelligence,
  Earth, Moon, and Planets, 96, 119,                             IEEE Transactions on, PAMI-8, 679 ,
  doi: 10.1007/s11038-005-9058-2                                 doi: 10.1109/TPAMI.1986.4767851
Baines, K. H., Drossart, P., Lopez-Valverde, M. A., et al.     Charbonneau, D., Allen, L. E., Megeath, S. T., et al. 2005,
  2006, Planet. Space Sci., 54, 1552,                            The Astrophysical Journal, 626, 523, doi: 10.1086/429991
  doi: 10.1016/j.pss.2006.06.020                               Coustenis, A., Lellouch, E., Maillard, J., & McKay, C.
Barnes, J., Sotin, C., Soderblom, J., et al. 2014, Planetary     1995, Icarus, 118, 87,
  Science, 3, 3, doi: 10.1186/s13535-014-0003-4                  doi: https://doi.org/10.1006/icar.1995.1179
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
12

Doose, L. R., Karkoschka, E., Tomasko, M. G., &                  Kesseli, A. Y., Snellen, I. A. G., Casasayas-Barris, N.,
 Anderson, C. M. 2016, Icarus, 270, 355,                           Mollière, P., & Sánchez-López, A. 2022, AJ, 163, 107,
 doi: 10.1016/j.icarus.2015.09.039                                 doi: 10.3847/1538-3881/ac4336
Dyudina, U., Zhang, X., Li, L., et al. 2016, The                 Knutson, H. A., Charbonneau, D., Allen, L. E., Burrows,
 Astrophysical Journal, 822, 76,                                   A., & Megeath, S. T. 2008, ApJ, 673, 526,
 doi: 10.3847/0004-637X/822/2/76                                   doi: 10.1086/523894
Es-sayeh, M., Rodriguez, S., Coutelier, M., et al. 2023, The     Kreidberg, L., Bean, J. L., Désert, J.-M., et al. 2014, ApJL,
  Planetary Science Journal, 4, 44,                                793, L27, doi: 10.1088/2041-8205/793/2/L27
  doi: 10.3847/PSJ/acbd37                                        Lavvas, P., Yelle, R., & Griffith, C. 2010, Icarus, 210, 832,
                                                                   doi: https://doi.org/10.1016/j.icarus.2010.07.025
Esteves, L. J., De Mooij, E. J. W., & Jayawardhana, R.
                                                                 Le Mouélic, S., Cornet, T., Rodriguez, S., et al. 2019,
  2015, ApJ, 804, 150, doi: 10.1088/0004-637X/804/2/150
                                                                   Icarus, 319, 121,
Estrela, R., Swain, M. R., Roudier, G. M., et al. 2021, AJ,
                                                                   doi: https://doi.org/10.1016/j.icarus.2018.09.017
  162, 91, doi: 10.3847/1538-3881/ac0c7c
                                                                 Lellouch, E., Schmitt, B., Coustenis, A., & Cuby, J.-G.
Feinberg, L., Ziemer, J., Ansdell, M., et al. 2024, in Space
                                                                   2004, Icarus, 168, 209,
  Telescopes and Instrumentation 2024: Optical, Infrared,
                                                                   doi: https://doi.org/10.1016/j.icarus.2003.12.001
  and Millimeter Wave, ed. L. E. Coyle, S. Matsuura, &
                                                                 Lendl, M., Csizmadia, S., Deline, A., et al. 2020, A&A, 643,
  M. D. Perrin, Vol. 13092, International Society for Optics
                                                                   A94, doi: 10.1051/0004-6361/202038677
  and Photonics (SPIE), 130921N, doi: 10.1117/12.3018328
                                                                 Madhusudhan, N., Nixon, M. C., Welbanks, L., Piette, A.
Gao, P., Thorngren, D. P., Lee, E. K. H., et al. 2020, Nature      A. A., & Booth, R. A. 2020, The Astrophysical Journal
 Astronomy, 4, 951, doi: 10.1038/s41550-020-1114-3                 Letters, 891, L7, doi: 10.3847/2041-8213/ab7229
Garcı́a-Comas, M., López-Puertas, M., Funke, B., et al.         Madhusudhan, N., Sarkar, S., Constantinou, S., et al. 2023,
 2011, Icarus, 214, 571, doi: 10.1016/j.icarus.2011.03.020         The Astrophysical Journal Letters, 956, L13,
Garcı́a Muñoz, A., Lavvas, P., & West, R. A. 2017, Nature         doi: 10.3847/2041-8213/acf577
 Astronomy, 1, 0114, doi: 10.1038/s41550-017-0114                Malsky, I., Rauscher, E., Stevenson, K., et al. 2025, AJ,
Harris, C. R., Millman, K. J., van der Walt, S. J., et al.         169, 221, doi: 10.3847/1538-3881/adb7e8
  2020, Nature, 585, 357–362,                                    Mayorga, L. C., Jackiewicz, J., Rages, K., et al. 2016, The
  doi: 10.1038/s41586-020-2649-2                                   Astronomical Journal, 152, 209,
He, C., Hörst, S. M., Lewis, N. K., et al. 2018, AJ, 156, 38,     doi: 10.3847/0004-6256/152/6/209
  doi: 10.3847/1538-3881/aac883                                  Negrão, A., Coustenis, A., Lellouch, E., et al. 2006,
Hedman, M. M., & Stark, C. C. 2015, The Astrophysical              Planetary and Space Science, 54, 1225,
  Journal, 811, 67, doi: 10.1088/0004-637X/811/1/67                doi: https://doi.org/10.1016/j.pss.2006.05.031
Heng, K., & Li, L. 2021, ApJL, 909, L20,                         Nixon, C. A., Bézard, B., Cornet, T., et al. 2025, Nature
  doi: 10.3847/2041-8213/abe872                                    Astronomy, doi: 10.1038/s41550-025-02537-3
                                                                 Pallé, E., Goode, P. R., Yurchyshyn, V., et al. 2003,
Hoeijmakers, H. J., Snellen, I. A. G., & van Terwisga, S. E.
                                                                   Journal of Geophysical Research (Atmospheres), 108,
 2018, A&A, 610, A47, doi: 10.1051/0004-6361/201731192
                                                                   4710, doi: 10.1029/2003JD003611
Hooton, M. J., Hoyer, S., Kitzmann, D., et al. 2022, A&A,
                                                                 Parmentier, V., & Crossfield, I. J. M. 2018, Exoplanet
 658, A75, doi: 10.1051/0004-6361/202141645
                                                                   Phase Curves: Observations and Theory (Springer
Hörst, S. M., He, C., Lewis, N. K., et al. 2018, Nature
                                                                   International Publishing), 1419–1440,
  Astronomy, 2, 303, doi: 10.1038/s41550-018-0397-0
                                                                   doi: 10.1007/978-3-319-55333-7 116
Hu, R., Seager, S., & Bains, W. 2013, The Astrophysical
                                                                 Qiu, J., Goode, P. R., Pallé, E., et al. 2003, Journal of
 Journal, 769, 6, doi: 10.1088/0004-637X/769/1/6                   Geophysical Research (Atmospheres), 108, 4709,
Hunter, J. D. 2007, Computing in Science & Engineering, 9,         doi: 10.1029/2003JD003610
 90, doi: 10.1109/MCSE.2007.55                                   Rages, K., Pollack, J. B., & Smith, P. H. 1983,
Karkoschka, E. 1998, Icarus, 133, 134,                             J. Geophys. Res., 88, 8721,
 doi: 10.1006/icar.1998.5913                                       doi: 10.1029/JA088iA11p08721
Kelvin Rodriguez, & Astrogeology Science Center. 2024,           Robinson, T. D., Maltagliati, L., Marley, M. S., & Fortney,
  Integrated Software for Imagers and Spectrometers (ISIS)         J. J. 2014, Proceedings of the National Academy of
  8.3.0, U.S. Geological Survey, doi: 10.5066/P13TADS5             Science, 111, 9042, doi: 10.1073/pnas.1403473111
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
                                                                                                                          13

Robinson, T. D., Meadows, V. S., & Crisp, D. 2010, ApJL,        Vaughan, S. R., Gebhard, T. D., Bott, K., et al. 2023,
  721, L67, doi: 10.1088/2041-8205/721/1/L67                      Monthly Notices of the Royal Astronomical Society, 524,
                                                                  5477, doi: 10.1093/mnras/stad2127
Rodriguez, K., & Astrogeology Science Center. 2024,
                                                                Virtanen, P., Gommers, R., Oliphant, T. E., et al. 2020,
  Integrated software for imagers and spectrometers (ISIS)
                                                                  Nature Methods, 17, 261, doi: 10.1038/s41592-019-0686-2
  8.3.0, U.S. Geological Survey                                 Waite, J. H., Young, D. T., Cravens, T. E., et al. 2007,
Rustamkulov, Z., Sing, D. K., Mukherjee, S., et al. 2023,         Science, 316, 870, doi: 10.1126/science.1139727
  Nature, 614, 659, doi: 10.1038/s41586-022-05677-y             Wang, J., Mawet, D., Ruane, G., Hu, R., & Benneke, B.
                                                                  2017, The Astronomical Journal, 153, 183,
Soderblom, J. M., Barnes, J. W., Soderblom, L. A., et al.
                                                                  doi: 10.3847/1538-3881/aa6474
  2012, Icarus, 220, 744,
                                                                West, R. A., & Smith, P. H. 1991, Icarus, 90, 330,
  doi: https://doi.org/10.1016/j.icarus.2012.05.030               doi: https://doi.org/10.1016/0019-1035(91)90113-8
Stephan, K., Jaumann, R., Brown, R. H., et al. 2010,            Williams, D. M., & Gaidos, E. 2008, Icarus, 195, 927,
  Geophys. Res. Lett., 37, L07104,                                doi: https://doi.org/10.1016/j.icarus.2008.01.002
                                                                Xie, Y., & Ji, Q. 2002, in 2002 International Conference on
  doi: 10.1029/2009GL042312
                                                                  Pattern Recognition, Vol. 2, IEEE, 957–960
Stevenson, K. B., Désert, J.-M., Line, M. R., et al. 2014,     Yahn, Z., Trent, D. M., Duncan, E., et al. 2025, Journal of
  Science, 346, 838, doi: 10.1126/science.1256758                 Geophysical Research: Machine Learning and
Strauss, R. H., Robinson, T. D., Trilling, D. E., Cummings,       Computation, 2, doi: 10.1029/2024jh000366
  R., & Smith, C. J. 2024, The Astronomical Journal, 167,       Zebker, H. A., Stiles, B., Hensley, S., et al. 2009, Science,
                                                                  324, 921, doi: 10.1126/science.1168905
  87, doi: 10.3847/1538-3881/ad1bd1
                                                                Zhang, M., Knutson, H. A., Wang, L., Dai, F., & Barragán,
Titan Discipline Working Group. 2019, Titan, Tech. rep.,          O. 2022, AJ, 163, 67, doi: 10.3847/1538-3881/ac3fa7
  NASA                                                          Zugger, M. E., Kasting, J. F., Williams, D. M., Kane, T. J.,
Tomasko, M. G., Doose, L., Engel, S., et al. 2008,                & Philbrick, C. R. 2010, The Astrophysical Journal, 723,
  Planet. Space Sci., 56, 669, doi: 10.1016/j.pss.2007.11.019     1168, doi: 10.1088/0004-637X/723/2/1168
```
