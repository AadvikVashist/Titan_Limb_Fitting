---
citation_key: "lemouelic2019archive"
title: "The Cassini VIMS archive of Titan: From browse products to global infrared color maps"
source_pdf: "data/papers/lemouelic2019archive.pdf"
source_pdf_sha256: "eeb7017a55846c466a75cc07c8f62805ca7cf7cb4d4abd6d861df4fdc66e01bd"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                                                                                 The Cassini VIMS archive of Titan:
                                                                          from browse products to global infrared color maps

                                                       Stéphane Le Mouélica,∗, Thomas Cornetb , Sébastien Rodriguezc , Christophe Sotind , Benoı̂t Seignoverta ,
                                                            Jason W. Barnese , Robert H. Brownf , Kevin H. Bainesd , Bonnie J. Burattid , Roger N. Clarkg ,
                                                                     Philip D. Nicholsonh , Jérémie Lasuei , Virginia Pasekf , Jason M. Soberblomj
                                                                    a LPG, UMR 6112, CNRS, Université de Nantes, 2 rue de la Houssinière, 44322 Nantes, France
                                                                      b European Space Astronomy Centre (ESA/ESAC), Villanueva de la Canada, Madrid, Spain
                                                                              c IPGP, CNRS-UMR 7154, Université Paris-Diderot, USPC, Paris, France
                                                                      d Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA 91109, USA
                                                                 e Department of Physics, University of Idaho, Engineering-Physics Building, Moscow, ID 83844, USA
                                                                          f Department of Planetary Sciences, University of Arizona, Tucson, AZ 85721, USA
                                                                                             g Planetary Science Institute, Tucson, USA




arXiv:1809.06545v1 [astro-ph.EP] 18 Sep 2018
                                                                               h Department of Astronomy, Cornell University, Ithaca, NY 14853, USA
                                                                                                      i IRAP, Toulouse, France
                                                                    j MIT, Department of Earth, Atmospheric and Planetary Sciences, Cambridge, MA 02139, USA




                                               Abstract
                                               We have analyzed the complete Visual and Infrared Mapping Spectrometer (VIMS) data archive of Titan. Our objective
                                               is to build global surface cartographic products, by combining all the data gathered during the 127 targeted flybys of
                                               Titan into synthetic global maps interpolated on a grid at 32 pixels per degree (∼1.4 km/pixel at the equator), in seven
                                               infrared spectral atmospheric windows. Multispectral summary images have been computed for each single VIMS cube
                                               in order to rapidly identify their scientific content and assess their quality. These summary images are made available
                                               to the community on a public website (vims.univ-nantes.fr). The global mapping work faced several challenges due to
                                               the strong absorbing and scattering effects of the atmosphere coupled to the changing observing conditions linked to
                                               the orbital tour of the Cassini mission. We determined a surface photometric function which accounts for variations in
                                               incidence, emergence and phase angles, and which is able to mitigate brightness variations linked to the viewing geometry
                                               of the flybys. The atmospheric contribution has been reduced using the subtraction of the methane absorption band
                                               wings, considered as proxies for atmospheric haze scattering. We present a new global three color composite map of band
                                               ratios (red: 1.59/1.27 µm; green: 2.03/1.27 µm; blue: 1.27/1.08 µm), which has also been empirically corrected from an
                                               airmass (the solar photon path length through the atmosphere) dependence. This map provides a detailed global color
                                               view of Titan’s surface partially corrected from the atmosphere and gives a global insight of the spectral variability,
                                               with the equatorial dunes fields appearing in brownish tones, and several occurrences of bluish tones localized in areas
                                               such as Sinlap, Menvra and Selk craters. This kind of spectral map can serve as a basis for further regional studies and
                                               comparisons with radiative transfer outputs, such as surface albedos, and other additional data sets acquired by the
                                               Cassini Radar (RADAR) and Imaging Science Subsystem (ISS) instruments.
                                               Keywords: Titan, Titan surface, Image processing, Infrared observations
                                               DOI: 10.1016/j.icarus.2018.09.017


                                               1. Introduction                                                      13 years between 2004 and 2017 in the Saturnian system.
                                                                                                                    The Radar instrument onboard Cassini was able to observe
                                                 Titan has been recognized since the era of the Voyager             directly through Titan’s atmosphere using a centimetric
                                               space missions as one of the most interesting bodies in              wavelength (Elachi et al., 2005). Optical observations were
                                               the field of comparative planetology. Although the sur-              performed by two other instruments. The Imaging Science
                                               face is totally masked in the visible wavelength by scat-            Subsystem (ISS), composed of two multispectral framing
                                               tering and absorptions in the atmosphere, the geological             cameras, provided information on the surface thanks to
                                               diversity of Titan has been progressively revealed by the            its 0.93 µm CB3 filter (Porco et al., 2005). The Visual and
                                               instruments onboard the Cassini spacecraft, which spent              Infrared Mapping Spectrometer (VIMS) acquired hyper-
                                                                                                                    spectral images which gave access to the surface through
                                                 ∗ Corresponding author                                             partially transparent atmospheric windows in the infrared
                                                    Email address: stephane.lemouelic@univ-nantes.fr                at 1.08, 1.27, 1.59, 2.01, 2.69, 2.78 and 5 µm (Brown et al.,
                                               (Stéphane Le Mouélic)

                                               Preprint submitted to Icarus                                                                                        September 19, 2018
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
Fig. 1. Pixel size (in km) of the cubes used to build our global mosaic. A global map at 2 µm is shown in transparency. The cumulative
fractional coverage is shown on the right. Only 5 % of the surface has been observed with a pixel size better than 6 km. 60 % was observed
in conditions better than 15 km/pixel.



2004; Sotin et al., 2005). The spectral dimension of VIMS               section, we describe the global VIMS data set, its observ-
provides the possibility to retrieve information on compo-              ing modes and radiometric calibration, and discuss the
sitional and/or physical state (grain size) variations at the           production of multispectral summary images designed to
surface, in addition to giving access to clouds and aerosols            catch the scientific content of each observation in an op-
properties. With observations collected year after year,                timized way. In a second step, we present how the data
the geological diversity of Titan proved to be exceeding                were merged into global maps, after implementing empir-
the most optimistic expectations. Earth-like processes and              ical corrections for the surface photometry and for the at-
landforms such as cloud formation, river flowing at the sur-            mospheric effects. We discuss in particular the use of band
face (implying rainfalls), polar lakes and seas, mountain               ratios, before concluding with series of orthographic views
chains, equatorial dunes fields, impact craters, were pro-              and a focus on the Huygens landing site.
gressively discovered and characterized (Tomasko et al.,
2005; Stofan et al., 2007; Radebaugh et al., 2007, 2008;
                                                                        2. Description of the VIMS dataset
Wood et al., 2010; Aharonson et al., 2014). The main
difference with Earth comes from the nature of the ma-                  2.1. VIMS observing modes
terials: with an average surface temperature of −180 ◦C,
                                                                           The Visual and Infrared Mapping Spectrometer (VIMS)
methane is close to its triple point, playing on Titan the
                                                                        onboard Cassini acquired up to 64 pixels × 64 pixels im-
role of water on Earth. Only very few impact craters have
                                                                        ages in 352 spectral channels from 0.35 to 5.12 µm (Brown
been observed on the entire surface (Wood et al., 2010),
                                                                        et al., 2004). VIMS was composed of two separate instru-
which indicates that the surface is geologically relatively
                                                                        ments. The first was a two-dimension CCD array that
young, probably reprocessed by tectonic events, erosion
                                                                        covers the visible range (0.35–1.04 µm) with 96 spectral
of the bedrock, and deposition of sediments from air fall
                                                                        channels. The second covered the infrared range (0.88–
or slope/fluvial transport processes (Neish et al., 2016;
                                                                        5.12 µm) with 256 channels on a linear detector array and
Brossier et al., 2018).
                                                                        a bidirectional mirror (whisk-broom). The visible part has
   In this paper, we focus our study on the VIMS global                 proved to be very challenging to observe the surface of Ti-
archive, with the objective of producing global color mo-               tan, due to the strongly absorbing and scattering atmo-
saics of the complete data set of Titan acquired between                sphere (Tomasko et al., 2005; Hirtzig et al., 2009; Vixie
T0 (July 2004) and the last targeted flyby, T126, in April              et al., 2012). Since we focus our present study on sur-
2017. The correspondence between the Cassini flybys of                  face observations, we therefore center our efforts on the
Titan and the Cassini orbits around Saturn can be found                 infrared detector. Between 2004 and 2017, hyperspectral
in Seignovert (2015). We consolidate a previous study,                  data have been gathered during 127 targeted Titan close
which was limited to data acquired up to June 2010 only                 encounters, in addition to more distant untargeted obser-
(Le Mouélic et al., 2012a). Merging data acquired in very              vations. The spatial size of data cubes were optimized
different viewing conditions into global homogeneous maps               to take advantage of any acquisition opportunities, which
is a challenge due to the presence of the atmosphere, which             relied on which instrument was driving the pointing of
induces strong absorbing and scattering effects when cou-               Cassini during the closest approach phase. Occasionally,
pled with the changing geometry of the flybys. In a first               cubes were acquired on a line mode (noodle), letting the
                                                                    2
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
Fig. 2. Example of VIMS multispectral summary image on cube CM 1536367827 (08 September 2006). Black and white images and color
composites have been designed to catch variations linked both to surface and atmospheric features.



spacecraft drift build the second dimension of the image           2.2. Radiometric calibration
after concatenation of a series of hundreds of line cubes.            All the data cubes have been calibrated in reflectance
An occultation mode was also designed to catch the sig-            factor I/F , where Fis the solar flux (irradiance) andI is the
nal of stars crossing Titan’s atmosphere. The time expo-           calibrated radiance measured by VIMS. We followed the
sure generally ranged from 13 ms (used mostly at closest           VIMS pipeline described in Brown et al. (2004) and Barnes
approach to compensate the fast drift of the surface) to           et al. (2007), and further refined using a time-dependent
640 ms (when a higher S/N ratio was desired to look for            radiometric calibration aimed at correcting a small wave-
subtle spectral signatures at long wavelengths). More than         length shift that has been identified during the last years
60,000 hyperspectral cubes of Titan have been acquired             of the mission (Clark et al., 2018). Up to ∼10 nm of pro-
during the entire Cassini mission, with a pixel size as fine       gressive shift is observed when comparing data taken in
as 500 meters at best when VIMS was operating right at             2004 and data taken in 2017 (Clark et al., 2018). Despite
closest approach in very few occasions. Fig. 1 shows the           this shift being small, it can dramatically alter the surface
spatial coverage obtained with all cubes acquired within           information in the sharp atmospheric windows (especially
thresholds that we describe in a later stage to build the          at short wavelength), and produce significant seams if left
mosaics, with pixel sizes smaller than 30 km and with time         uncorrected. In the last calibration step, all spectra of the
exposures in the 20–300 ms range to avoid low signal to            mosaic have therefore been converted to a common refer-
noise ratios and saturated cubes. The panel on the right           ence wavelength of 2004 with a spline interpolation, using
displays the corresponding cumulative fractional coverage.         the shifts evaluated by (Clark et al., 2018).
The region around (80°S, 120°E), which represents ∼1 %
of the surface, was never observed within these thresh-
                                                                   2.3. VIMS multispectral summary products
olds. Only 5 % of the surface was covered with a pixel
scale lower than 6 km/pixel. The cumulative global cov-               For each VIMS hyperspectral cube of Titan (except the
erage raises to 20 % when considering observations better          single line cubes and the cubes taken in occultation mode),
than the 10 km/pixel scale, and 60 % for observations bet-         we have setup a multispectral summary image designed to
ter than the 15 km/pixel scale.                                    highlight the spectral diversity of the observation using
                                                                   specific combination of channels, and displaying the cubes
                                                                   under different map projections. This strategy is similar
                                                               3
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
to the one used by the Compact Reconnaissance Imag-               non-negligible contribution of Titan’s absorbing and scat-
ing Spectrometer for Mars (CRISM) team to automati-               tering atmosphere even in the methane optical windows,
cally search for mineral signatures on Mars (Pelkey et al.,       the huge variations in observing conditions encountered
2007). It allows us to visually find the most interesting         throughout the mission, as well as the temporal changes
data, in addition to identify corrupted cubes. Fig. 2 shows       that could have occurred in the surface and atmosphere
an example on a typical multispectral summary product             during 13 years (Barnes et al., 2013a; Solomonidou et al.,
acquired at T17 in September 2006. We choose to dis-              2016).
play both enhanced black and white and color composites
which are dedicated either to surface or atmospheric fea-         3.1. Data fusion strategy
tures. We use for example the 2.01 µm channel (Fig. 2a)
                                                                     In order to build global maps, individual data cubes
to emphasize the details of the surface. We also calcu-
                                                                  have been sorted by increasing spatial resolution, with the
lated three RGB composites of single bands inspired from
                                                                  high resolution images on top of the mosaic and the low
previous studies (Barnes et al., 2007; Soderblom et al.,
                                                                  resolution images used as background. Other strategies
2009b; Le Mouélic et al., 2012a), which reveal at the
                                                                  could be envisaged to give more weight to other parame-
same time clouds and surface features (Fig. 2b/e/g). The
                                                                  ters influencing data quality, such as time exposure or low
1.59/1.27 µm ratio (Fig. 2c) emphasizes spectral variations
                                                                  airmass (that we defined by 1/cos i + 1/cos e in the plane-
of the surface. The image acquired at2.1 µm (Fig. 2d) is
                                                                  parallel atmosphere approximation), instead of the spatial
used to detect clouds (Rodriguez et al., 2009, 2011; Tur-
                                                                  resolution only (i.e., Barnes et al., 2007). After testing
tle et al., 2018), which appear bright at this wavelength
                                                                  this approach, we decided to keep the spatial resolution
where the surface is not seen. Fig. 2f corresponds to an
                                                                  as the main criterion to emphasize the finest details of
RGB color composite of the 1.59/1.27 µm, 2.03/1.27 µm
                                                                  the surface. We filtered out the observing geometry in
and 1.27/1.08 µm band ratios respectively, which provide
                                                                  order to remove the pixels acquired in too extreme illu-
the most sensitivity to surface heterogeneities. We in-
                                                                  minating and viewing conditions, which produce strong
cluded in Fig. 2h a color composite with the surface, tro-
                                                                  seams in the VIMS mosaics due to enhanced surface and
pospheric and stratospheric parameters of Brown et al.
                                                                  atmospheric photometric effects. We used thresholds of
(2010). The 2.03/2.10 µm ratio (Fig. 2i) provides a tenta-
                                                                  80° both on the incidence and emergence angles, 110° on
tive normalization of the illuminating conditions. The im-
                                                                  the phase angle, and 7 on the airmass. These thresholds
age acquired at 0.98 µm (Fig. 2j) shows a pure atmospheric
                                                                  correspond to a trade-off between the surface coverage (in
scattering observation. We have added two color compos-
                                                                  particular in polar areas, most often viewed at extreme
ites (R=2.78 µm, G=3.26 µm, B=3.31 µm in Fig. 2k and
                                                                  geometries) and the mosaic quality. The lowest values of
R=5 µm, G=3.31 µm, B=3.21 µm in Fig. 2n) which are
                                                                  the incidence, emergence, phase and airmass in the mo-
sensitive to the methane fluorescence and reveal the layers
                                                                  saic are 0.12°, 0.02°, 11.1°, 2.01 respectively. The exposure
of the atmosphere. A true color image is also displayed
                                                                  time has been restrained to the 20–300 ms range in order
(Fig. 2l). Finally, the image acquired at 5 µm (Fig. 2m)
                                                                  to avoid cubes with low signal-to-noise ratio and saturated
corresponds to the average of all channels between 4.90 and
                                                                  data.
5.12 µm. This wavelength range is the least affected by
                                                                     Fig. 3 shows the resulting global mosaic of relevant ge-
atmospheric scattering. Information regarding the flyby
                                                                  ometric viewing parameters (incidence, emergence, phase
number, date, distance range of Cassini to Titan’s surface
                                                                  and airmass), the I/F at 1.08, 2.03 and 5 µm (surface win-
at the time of the cube acquisition, pixel exposure time,
                                                                  dows), the I/F at 1.95 µm (where the atmosphere is not
ranges for the phase, incidence and emergence angles is
                                                                  transparent) with no correction for geometry nor for the
also displayed.
                                                                  atmospheric effects. Many boundaries or seams appear be-
   The browse products contain a wealth of information
                                                                  tween individual images in these raw mosaics. They are
that could potentially stimulate further focused atmo-
                                                                  mainly caused by the varying viewing angles (incidence,
spheric and surface studies. A dedicated website has
                                                                  emergence, phase) between data acquired during the dif-
been setup to provide a user-friendly access to all these
                                                                  ferent flybys, which induce strong atmospheric and surface
multispectral summary images, produced from the Plane-
                                                                  photometric effects. A dependence with airmass is also ob-
tary Data System archive and covering the entire mission
                                                                  served. Other discrepancies might exist due to surface and
(vims.univ-nantes.fr).
                                                                  atmospheric temporal variations, and residual calibrations
                                                                  artifacts.
3. Merging data into global maps
   Our final objective is to produce VIMS synthetic global        3.2. Photometric correction at 5 microns
maps interpolated on a grid at 32 pixels per degree (cor-            In order to account for the variations of solar illumina-
responding to a spatial sampling of ∼1.4 km at the equa-          tion (incidence i) and viewing angle (emergence e) between
tor), using different combinations of wavelengths empha-          different flybys, a surface photometric correction had to be
sizing surface spectral heterogeneities. Merging 13 years         implemented. We focused on the 5 µm atmospheric win-
of data represents a significant challenge considering the        dow, which is the least affected by atmospheric scattering
                                                              4
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
Fig. 3. Raw global mosaics in equidistant cylindrical projection at 1.08, 2.03, 5 µm (surface windows) and 1.95 µm (atmosphere only)
compared to global mosaics of incidence, emergence, phase, and airmass, containing all the VIMS cubes acquired during the entire Cassini
mission, within the data filters described in the text. The observing conditions vary widely during the mission, which causes significant
seams or boundaries to appear in the uncorrected mosaics. The white dashed rectangle on the 5 µm map corresponds to the test area used
to determine the corrections.



and absorption, and thus is the most sensitive to surface              mospheric absorption and haze scattering in this window,
photometric effects. We selected a test area presenting                most of the variations seen in this portion of the mosaic
a rather homogeneous brightness at 5 µm, located in the                come from the viewing conditions of the surface.
Northern mid-latitudes between 37.5°N and 52.5°N (white                  We have tested several surface photometric corrections
dashed rectangle in the 5 µm map of Fig. 3), outside of the            commonly found in the literature. These include the Lam-
lakes, seas, or dune fields. Given the relatively weak at-             bert, Lommel-Seeliger and Lunar Lambert disc functions,
                                                                   5
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
coupled with different particle phase functions P (φ) such
as the Rayleigh, Henyey-Greenstein and Hapke lunar func-
tions Hapke (2012). While particle phase functions are
usually used to infer particle shape and size properties
(e.g., Henyey-Greenstein two lobes phase function), we
rather focus on the mosaic enhancement, the phase func-
tion giving us an indication of the degree of anisotropy in
scattering. Whereas a pure Lambert function gave satis-
factory results to perform a first order correction of the
incidence angle (i) in an earlier version of the maps (Le
Mouélic et al., 2012a), we realized that a supplementary
correction of the emergence (e) and phase (φ) angles was
needed to account for the extreme diversity of the view-
ing conditions, in particular on the northern polar regions,
which were only seen during the second half of the Cassini
mission after the dissipation of the northern cloud and
haze (Le Mouélic et al., 2012b, 2018). These areas were
not included in our previous maps. Our best result was ob-
tained with a Lunar Lambert type function (Eqs. (1) and
(2)) with a lunar-like weighting factor A = 0.285 (Fig. 4
and 5).

                                                                             Fig. 5. (a) Uncorrected global map at 5 µm. (b) 5 µm map corrected
                    cos i                                                    using the surface photometric function described in equations (1)
         f =A·                · P (φ) + (1 − A) · cos i           (1)
                cos i + cos e                                                and (2). Most of the seams have been smoothed out, except on two
             "
                                                    2
                                                      #                      cubes in northern regions, one of which exhibiting a broad specular
          4 π sin φ + (π − φ) cos φ (1 − cos φ)                              reflexion on Kraken Mare.
  P (φ) =                             +                           (2)
           5            π                    10

   P (φ) in equation (1) is the single-particle phase func-
tion of the surface. The lunar theoretical particle phase
function of Hapke (1963) provided satisfactory results to                       The uncorrected map at 5 µm is shown in Fig. 5a. The
describe this term (equation (2)), as it was already noted                   map at 5 µm corrected for the photometry with the fac-
by Cornet et al. (2012). The fact that the point cloud                       tor described in equations (1) and (2) is shown in Fig. 5b.
in Fig. 4 is aligned with the origin of the graph confirms                   We see that the level of seams has significantly decreased
the hypothesis that the additive scattering term at 5 µm                     in almost all regions, except two cubes in northern lat-
is negligible.                                                               itudes taken in an extreme geometry and which contain
                                                                             in particular a broad specular reflexion on Kraken Mare.
                                                                             Very bright features in this map correspond to possible
                                                                             evaporites (Barnes et al., 2011; MacKenzie et al., 2014),
                                                                             specular reflections on the northern seas (Sotin et al., 2012;
                                                                             Soderblom et al., 2012; Barnes et al., 2013b, 2014), possible
                                                                             cryovolcanic candidates (Lopes et al., 2013), or unfiltered
                                                                             clouds (Turtle et al., 2018).

                                                                                In the following, we will use the same photometric func-
                                                                             tion for all other surface windows, assuming that the sur-
                                                                             face properties are not wavelength-dependent. This is a
                                                                             shortcoming, as a complete solution would require to de-
                                                                             rive different photometric parameters for each wavelength,
                                                                             as it is commonly done for example on bodies such as Mars
                                                                             (Binder and Jones, 1972) and the Moon (Lane and Irvine,
                                                                             1973). However, to achieve this, the contribution of the
Fig. 4. I/F at 5 µm versus the photometric function described in
                                                                             atmosphere on Titan has to be fully removed prior to the
equation (1). The linear correlation shows that this function can be         photometric parameters computation, which still makes it
used to correct at first order from the effect of incidence, emergence       very challenging at this stage, and falls beyond the scope
and phase.                                                                   of this paper. We leave this issue to further studies based
                                                                             on complete radiative transfer approaches.
                                                                         6
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
Fig. 6. Mosaics at 1.08, 1.27, 1.59, 2.03, 2.69, 2.78 µm without correction (left), with a photometric correction only (center), and with the
first order correction of the additive scattering term prior to the photometric correction (right).



3.3. Short wavelengths case                                              ing contribution of the aerosols. To mitigate this effect,
  Whereas the 5 µm methane window is almost free of at-                  we use the wings of the atmospheric windows as a proxy
mospheric scattering, this is not the case for wavelengths               to correct for the amount of additive scattering present
shorter than 3 microns, which contain an additive scatter-               in the center of these windows, where the surface is seen

                                                                     7
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
by VIMS. This process is already described in Le Mouélic                   green and blue channels controlled respectively by the 2.01,
et al. (2012a) and will therefore not be fully reproduced                   1.59 and 1.27 µm mosaics empirically corrected from atmo-
here. We used the same set of k-factor (values of 1.15,                     spheric scattering and photometry with the method de-
1.50, 1.60, 1.29 and 1.14 respectively for the 1.08, 1.27,                  scribed above. Fig. 7b corresponds to a global mosaic
1.59, 2.03 and 2.78 µm atmospheric windows, e.g. Tab. 1                     with the red, green and blue channels controlled by the
in Le Mouélic et al. (2012a), which account for the differ-                5, 2.01 and 1.27 µm images respectively. Introducing the
ence of transparency between the center of the windows                      5 µm window decreases the level of the greenish trend seen
and their wings. The wings were taken at 1.03, 1.14, 1.22,                  in Fig. 7a due to the residual absorption and scattering
1.32, 1.49, 1.65, 1.95, 2.13, 2.64 and 2.83 µm. These wave-                 in the atmosphere, mainly in the polar regions. A broad
lengths correspond the first images (departing from the                     specular reflexion is seen on Kraken Mare in both cases.
center of the windows) for which no surface feature is vi-                     These color composites have been widely used in re-
sually detectible in the global mosaic, even at the lowest                  gional studies (e.g., Barnes et al., 2007, 2011; Soderblom
airmass conditions. The 2.69 µm image is corrected with                     et al., 2009a,b; Cornet et al., 2012; Rodriguez et al., 2014).
the same wings and k-factor as the 2.78 µm image, as these                  Our objective is now to go one step further by investigat-
two windows correspond to a double peak rather than a                       ing band ratios, which are extremely sensitive to subtle
single narrow one.                                                          spectral variations, as it has already been shown in the
   Fig. 6 presents a comparison of the maps in the sur-                     case of airless bodies.
face windows before (left column) and after this empirical
atmospheric correction process (right column). The mid-
dle column shows partial results, where only the surface                    3.4. Color composites of band ratios
photometric correction described in section 3.2 has been
applied without the subtraction of the band wings, which
would be the typical correction for data acquired on an air-                   To better emphasize spectral heterogeneities, we also
less body. The level of residual seams has been decreased                   computed RGB composites of band ratios. Band ra-
in all the windows in most cases after the complete correc-                 tios, which cancel out all multiplicative factors in absence
tion process (right column).                                                of additive components, is a powerful technique widely
                                                                            used in planetary sciences. For Titan, the 1.59/1.27 µm,
                                                                            2.03/1.27 µm and 1.27/1.08 µm ratios proved to be use-
                                                                            ful for localized regional studies (Le Mouélic et al., 2008;
                                                                            Brossier et al., 2018). However, using band ratios on global
                                                                            maps of Titan still remains very challenging, as ratios are
                                                                            generally much more sensitive to atmospheric effects and
                                                                            any residual calibration artifacts than RGB composites
                                                                            of single bands only. Producing fully artifact-free global
                                                                            maps of band ratios would still be some sort of ultimate
                                                                            cartographic product that requires a thorough investiga-
                                                                            tion of the residuals present in the corrected mosaics.
                                                                               In order to make progress in this direction, we inves-
                                                                            tigated the dependence of the ratios with geometric pa-
                                                                            rameters. We observed in particular that the logarithm of
                                                                            each ratio appears correlated with our airmass parameter,
                                                                            so with the amount of atmosphere that the light has been
                                                                            crossing. This is illustrated in the scatter plots of Fig. 8,
                                                                            corresponding to all the points located in our test latitu-
                                                                            dinal belt between 37.5°N and 52.5°N, containing mostly
                                                                            homogeneously bright terrains. The dependence of the
                                                                            ratios with the airmass is also apparent when compar-
                                                                            ing the RGB composite of the 1.59/1.27 µm, 2.03/1.27 µm
                                                                            and 1.27/1.08 µm ratios in Fig. 9a and the airmass map in
Fig. 7. (a) RGB global map with the red, green and blue controlled          Fig. 3. This is particularly the case in fuzzy pinkish areas
by the 2.0, 1.59 and 1.27 µm channels respectively. (b) RGB global
map with the red, green and blue controlled by the 5, 2.0 and 1.27 µm       seen near the equator in Fig. 9a.
channels respectively.                                                         Following the systematic trends observed in Fig. 8, we
                                                                            decided to empirically remove this dependence with air-
  In order to emphasize spectral variations linked to com-                  mass using a second order polynomial fit derived on the
positional heterogeneities, the maps in Fig. 6 can be com-                  scatter plot of the logarithm of the ratios versus the air-
bined into RGB color composites. Fig. 7 shows two RGB                       mass. The corresponding correction formula are given be-
global color maps. Fig. 7a has been coded with the red,                     low:
                                                                        8
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                                                            where Rλ is the reflectance at the wavelength λ and a
                                                                         is the airmass defined by 1/cos i + 1/cos e. After this purely
                                                                         empirical step, the resulting RGB color composite of band
                                                                         ratios appears much less dependent on the geometry of
                                                                         observations, as shown in Fig. 9b. Whereas this map still
                                                                         contains some dependence with atmospheric contributions
                                                                         and pure brightness variations, subtle color differences are
                                                                         strongly emphasized compared to previous RGB maps due
                                                                         to the use of the ratios. The strongly diffusing atmosphere
                                                                         still hampers the study of the polar areas (appearing in
                                                                         pink), but the ratios nicely reveal the extent of equatorial
                                                                         dune fields, which appear in brownish tones.
                                                                            Fig. 9b is so far the most advanced global cartographic
                                                                         product that we have been able to automatically produce.
                                                                         Fig. 10 shows a series of orthographic views centered on
                                                                         the equator, with a point of perspective at infinite dis-
                                                                         tance, derived from this map after a last minor cosmetic
                                                                         hand cleaning step. The six panels correspond to hemi-
                                                                         spheric views where Titan is rotated by 60° in longitude
                                                                         eastward from left to right and from top to bottom. One
                                                                         of the most striking feature is the equatorial dune fields
                                                                         (Radebaugh et al., 2008; Rodriguez et al., 2014), which
                                                                         appear readily in brownish tones. The second main spec-
                                                                         tral type of interest corresponds to dark blue areas such
                                                                         as the ones we see around Sinlap and Menvra craters,
                                                                         or north east of Hotei Regio. This color difference can
                                                                         be spectrally explained by a local enrichment in water
                                                                         ice, decreasing the reflectance at 1.59 and 2.01 µm com-
                                                                         pared to the 1.27 µm channel (Rodriguez et al., 2006; Mc-
                                                                         Cord et al., 2008; Brossier et al., 2018; Solomonidou et al.,
                                                                         2018). However, we point out that this interpretation is
                                                                         not unique, as several organic compounds could poten-
                                                                         tially create the same spectral effect. Indeed, many organ-
                                                                         ics show the downward trend with increasing wavelength
                                                                         like water ice (e.g., Clark et al., 2009, 2010; Kokaly et al.,
                                                                         2017). NH-bearing compounds show an even stronger
                                                                         downward trend than water ice. Discriminating between
                                                                         these compositional signatures will require a very precise
                                                                         atmospheric removal and analysis of the detailed spectral
                                                                         structure within each window, which is still an ongoing
                                                                         field of research. Our main objective here was rather to
                                                                         show the global distribution of spectral heterogeneities it-
                                                                         self. We leave the identification of individual constituents
                                                                         to further dedicated studies, which could rely both on lab-
Fig. 8. Dependence of band ratios with airmass on the test area          oratory spectra and detailed radiative transfer modeling,
containing mostly homogeneously bright terrains located in the lat-      and which fall beyond the scope of this paper. We now
itudinal belt between 37.5°N and 52.5°N. A positive correlation is
observed for the upper and lower panels, whereas the middle panel
                                                                         give a regional example of the color map in one of the
presents a negative correlation at low airmass values. This depen-       most important spot on Titan: the Huygens landing site.
dence can be empirically corrected at first order using a second order
polynomial fit.                                                          3.5. Zoom on the Huygens landing site
                                                                           In order to illustrate the accuracy of the final band ratio
                                                                         map, Fig. 11 shows a zoom on the Huygens Landing site.
  
    R1.59
                     
                        R1.59
                              
                                                      2
                                                                         This area is the only spectral measurement acquired from
                    =           exp−(0.0387a−0.00187a ) (3)              the surface and is therefore of particular interest. The first
    R1.27 corrected     R1.27
                                                                     VIMS observation acquired at Ta in October 2004 (cube
    R2.03               R2.03                          2
                    =           exp−(−0.1237a−0.0123a ) (4)              CM 1477491859) had a spatial sampling of 14 km/pixel
    R1.27 corrected     R1.27                                            and provided the general context (Rodriguez et al., 2006).
                           
    R1.27               R1.27                       2
                    =           exp−(0.0415a−0.0032a ) (5) 9
    R1.08 corrected     R1.08
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
Fig. 9. (a) RGB color map with the red, green and blue channels controlled by the 1.59/1.27 µm, 2.03/1.27 µm and 1.27/1.08 µm ratios
respectively. (b) Same mosaic after the empirical correction of the airmass dependence (equations (3), (4) and (5)). The correction decreases
the contrast at the seams due to the atmosphere at the equator and near the poles. The white rectangle shows the location of the zoom on
the Huygens landing site displayed in Fig. 11



The best observation of the landing site itself has then                  0.75 and 1.4 km/pixel. Other late observations provided
been acquired at T47 in November 2008 (cube labeled                       the intermediate context. They were acquired at T88 in
CM 1605804042), when VIMS was operating at closest                        November 2012 (cube CM 1732874866, between 2.1 and
approach. This allowed to acquire an observation in a                     2.7 km/pixel) and T85 in July 2012 (cube CM 1721856031,
spot pointing mode, with the whole spacecraft spinning                    between 3.2 and 5.5 km/pixel).
progressively to compensate for the fast drift of the sur-
                                                                           In Fig. 11, we see that all these data have been ho-
face. The spatial sampling of the T47 cube ranged between
                                                                          mogeneously merged into the color band ratio map, de-
                                                                     10
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
Fig. 10. Selection of orthographic views derived from the RGB corrected color ratio map of Fig. 9b after a last cosmetic hand cleaning step.
The upper left image is centered at (0°, 170°E), close to the Huygens landing site. The six panels correspond to views where Titan is rotated
by 60 deg. in longitude eastward from left to right and from top to bottom. The equatorial dune fields appear readily in brown. Bright
terrains and several patches of bluish areas also show up in specific locations.



spite being acquired with different geometries and with                   Based on automatic filters and manual refinement in the
a time span of eight years. It appears that the Huy-                      cube selection using these summary images, ∼19,000 indi-
gens probe landed in an area which corresponds to mod-                    vidual data cubes have then been merged to produce global
erately bluish tones. The accuracy of the VIMS obser-                     color maps at 32 pixels per degree (∼1.4 km/pixel at the
vation is sufficiently high to easily recognize the bright                equator) in the seven atmospheric methane optical win-
dissected terrain that was imaged by the DISR camera at                   dows. We implemented a correction for the surface pho-
an altitude of ∼34 km during the descent of Huygens un-                   tometric function which takes into account the incidence,
der its parachute (Tomasko et al., 2005; Karkoschka and                   emergence, and phase variations. An empirical subtrac-
Schröder, 2016). Further studies could be considered to                  tion of the band wings is used to mitigate the effects of
perform a detailed comparison between the VIMS cubes                      the additive scattering aerosols at short wavelengths. We
mentioned here and the DISR data, following the work of                   also investigated band ratios, a powerful technique to em-
Karkoschka and Schröder (2016).                                          phasize subtle spectral variations, by implementing an em-
                                                                          pirical correction of the absorption difference between the
                                                                          ratioed channels. This process allowed us to build global
4. Conclusion
                                                                          maps which integrate data from the complete mission and
   We have reduced the global VIMS hyperspectral archive                  which strongly emphasize the global distribution of the
of Titan integrating data from T0 to T126 flybys in order                 main spectral units. In particular, the band ratio global
to map the spectral heterogeneities at the surface. Mul-                  map readily shows the extent of the equatorial dune fields
tispectral summary images have been computed for each                     (which appears in brown tones in band ratio RGB compos-
hyperspectral VIMS cube, in order to give an easy access                  ites of the 1.59/1.27 µm, 2.03/1.27 µm and 1.27/1.08 µm
to the scientific content of each observation. These browse               channels). Several areas show a dark bluish color, such as
products are available on the vims.univ-nantes.fr website.                near Sinlap, Menvra or Selk craters for example, due to a

                                                                     11
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                                           Acknowledgment

                                                                              Authors are very grateful to two anonymous reviewers
                                                                           for their very detailed comments. This work has been
                                                                           partly funded by the French spatial agency (CNES). We
                                                                           also acknowledge the financial support from Région Pays
                                                                           de la Loire, project GeoPlaNet (convention 2016-10982).
                                                                           S.R. is supported by the Institut Universitaire de France
                                                                           and acknowledges support from the UnivEarthS LabEx
                                                                           program of Sorbonne Paris Cité (ANR-10-LABX-0023,
                                                                           ANR-11-IDEX-0005-02) and the French National
                                                                           Research        Agency      (ANR-APOSTIC-11-BS56-002,
                                                                           ANR-12-BS05-001-3/EXO-DUNES).


                                                                           References

                                                                           Aharonson O. and 5 colleagues. Titan’s surface geology. In Muller-
                                                                             Wodarg I. and 3 colleagues, editors, Titan, pp 63–101. Cambridge
                                                                             University Press, Cambridge, 2014.
                                                                           Barnes J. W. and 9 colleagues. Global-scale surface spectral varia-
                                                                             tions on Titan seen from Cassini/VIMS. Icarus, 186(1)242–258,
                                                                             2007.
                                                                           Barnes J. W. and 16 colleagues. Organic sedimentary deposits in
                                                                             Titan’s dry lakebeds: Probable evaporite. Icarus, 216(1)136–140,
Fig. 11. Detail of the VIMS color ratio map corresponding to the
                                                                             2011.
area in the white square in Fig. 9b. The Huygens landing site is
                                                                           Barnes J. W. and 19 colleagues. Precipitation-induced surface bright-
marked by a red cross. A black and white panorama acquired by
                                                                             enings seen on Titan by Cassini VIMS and ISS. Planetary Science,
DISR on Huygens from an altitude of ∼34 km is shown for compar-
                                                                             2(1)1, 2013a.
ison (top). We can easily recognize the bright feature north of the
                                                                           Barnes J. W. and 11 colleagues. A transmission spectrum of titan’s
landing site in both data sets. VIMS suggests that Huygens landed
                                                                             north polar atmosphere from a specular reflection of the sun. As-
in an area corresponding to the moderate bluish tone units.
                                                                             trophysical Journal, 777(2), 2013b.
                                                                           Barnes J. W. and 9 colleagues. Cassini/VIMS observes rough surfaces
                                                                             on Titan’s Punga Mare in specular reflection. Planetary Science,
                                                                             3(1)3, 2014.
change in composition and/or grain size.                                   Binder A. B. and Jones J. C. Spectrophotometric studies of the
                                                                             photometric function, composition, and distribution of the surface
   The residual discrepancies in the maps are due to several                 materials of Mars. Journal of Geophysical Research, 77(17)3005–
                                                                             3020, 1972.
factors. One of the challenges comes from temporal vari-                   Brossier J. F. and 12 colleagues. Geological Evolution of Titan’s
ations at the surface (Barnes et al., 2013a; Solomonidou                     Equatorial Regions: Possible Nature and Origin of the Dune Mate-
et al., 2016) and moreover in the atmosphere (haze and                       rial. Journal of Geophysical Research: Planets, 123(5)1089–1112,
clouds). This is particularly true at the poles, where sig-                  2018.
                                                                           Brown M. E., Roberts J. E. and Schaller E. L. Clouds on Titan during
nificant changes occurred during the mission (Le Mouélic                    the Cassini prime mission: A complete analysis of the VIMS data.
et al., 2018). The north pole was fully covered by haze and                  Icarus, 205(2)571–580, 2010.
cloud up to ∼55°N at the beginning of the mission. We                      Brown R. H. and 21 colleagues. The Cassini Visual and Infrared
had to wait for the circulation turnover after the equinox                   Mapping Spectrometer (VIMS) Investigation. In The Cassini-
                                                                             Huygens Mission, pp 111–168. Kluwer Academic Publishers, Dor-
in 2009 to get clearer skies in the north. The south pole                    drecht, 2004.
experienced a reverse situation, with clear skies at the be-               Clark R. N. and 3 colleagues. Reflectance spectroscopy of organic
ginning of the mission and a polar cloud appearing after                     compounds: 1. Alkanes. Journal of Geophysical Research, 114
                                                                             (E3)E03001, 2009.
2012 and growing in size up to 2017. In addition to these
                                                                           Clark R. N. and 15 colleagues. Detection and mapping of hydrocar-
polar events, sporadic methane clouds have been observed                     bon deposits on Titan. Journal of Geophysical Research, 115(10),
throughout all the mission (e.g., Rodriguez et al., 2009,                    2010.
2011; Turtle et al., 2018).                                                Clark R. N. and 3 colleagues. The VIMS Wavelength and Radio-
                                                                             metric Calibration 19, Final Report. The Planetary Atmospheres
   The surface photometric behavior can be improved in                       Node, 2018.
further studies using a more complex photometric func-                     Cornet T. and 13 colleagues. Geomorphological significance of On-
                                                                             tario Lacus on Titan: Integrated interpretation of Cassini VIMS,
tion using wavelength-dependent parameters. This will                        ISS and RADAR data and comparison with the Etosha Pan
require a better decorrelation between atmospheric and                       (Namibia). Icarus, 218(2)788–806, 2012.
surface contributions in the methane windows. More in-                     Cornet T. and 16 colleagues. Radiative Transfer Modelling in Titan’s
                                                                             Atmosphere: Application to Cassini/VIMS Data. In 48th Lunar
puts derived from a complete radiative transfer analysis                     and Planetary Science Conference, volume 48, Texas, 2017.
could also provide another way to improve the homogene-                    Elachi C. and 34 colleagues. Cassini Radar Views the Surface of
ity of the maps in future works (Cornet et al., 2017).                       Titan. Science, 308(5724)970–974, 2005.

                                                                      12
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
Hapke B. Theory of Reflectance and Emittance Spectroscopy. Cam-              Sotin C. and 25 colleagues. Release of volatiles from a possible cry-
   bridge University Press, Cambridge, 2nd edition, 2012.                      ovolcano from near-infrared imaging of Titan. Nature, 435(7043)
Hapke B. W. A theoretical photometric function for the lunar sur-              786–789, 2005.
   face. Journal of Geophysical Research, 68(15)4571–4586, 1963.             Sotin C. and 15 colleagues. Observations of Titan’s Northern lakes
Hirtzig M. and 4 colleagues. A review of Titan’s atmospheric phe-              at 5µm: Implications for the organic cycle and geology. Icarus,
   nomena. The Astronomy and Astrophysics Review, 17(2)105–147,                221(2)768–786, 2012.
   2009.                                                                     Stofan E. R. and 37 colleagues. The lakes of Titan. Nature, 445
Karkoschka E. and Schröder S. E. Eight-color maps of Titan’s sur-             (7123)61–64, 2007.
   face from spectroscopy with Huygens’ DISR. Icarus, 270260–271,            Tomasko M. G. and 39 colleagues. Rain, winds and haze during the
   2016.                                                                       Huygens probe’s descent to Titan’s surface. Nature, 438(7069)
Kokaly R. F. and 10 colleagues. USGS Spectral Library Version 7.               765–778, 2005.
   Data Series, p 61, 2017.                                                  Turtle E. P. and 17 colleagues. Titan’s Meteorology Over the Cassini
Lane A. P. and Irvine W. M. Monochromatic phase curves and                     Mission: Evidence for Extensive Subsurface Methane Reservoirs.
   albedos for the lunar disk. The Astronomical Journal, 78(1962)              Geophysical Research Letters, 45(11)5320–5328, 2018.
   267, 1973.                                                                Vixie G. and 12 colleagues. Mapping Titan’s surface features within
Le Mouélic S. and 13 colleagues. Mapping polar atmospheric features           the visible spectrum via Cassini VIMS. Planetary and Space Sci-
   on Titan with VIMS: From the dissipation of the northern cloud              ence, 60(1)52–61, 2012.
   to the onset of a southern polar vortex. Icarus, 311371–383, 2018.        Wood C. A. and 5 colleagues. Impact craters on Titan. Icarus, 206
Le Mouélic S. and 17 colleagues. Mapping and interpretation of                (1)334–344, 2010.
   Sinlap crater on Titan using Cassini VIMS and RADAR data.
   Journal of Geophysical Research, 113(E4)E04003, 2008.
Le Mouélic S. and 10 colleagues. Global mapping of Titan’s surface
   using an empirical processing method for the atmospheric and
   photometric correction of Cassini/VIMS images. Planetary and
   Space Science, 73(1)178–190, 2012a.
Le Mouélic S. and 12 colleagues. Dissipation of Titans north polar
   cloud at northern spring equinox. Planetary and Space Science,
   60(1)86–92, 2012b.
Lopes R. M. C. and 15 colleagues. Cryovolcanism on Titan: New
   results from Cassini RADAR and VIMS. Journal of Geophysical
   Research: Planets, 118(3)416–435, 2013.
MacKenzie S. M. and 10 colleagues. Evidence of Titan’s climate
   history from evaporite distribution. Icarus, 243191–207, 2014.
McCord T. B. and 13 colleagues. Titan’s surface: Search for spectral
   diversity and composition using the Cassini VIMS investigation.
   Icarus, 194(1)212–242, 2008.
Neish C. D. and 7 colleagues. Fluvial erosion as a mechanism for
   crater modification on Titan. Icarus, 270114–129, 2016.
Pelkey S. M. and 11 colleagues. CRISM multispectral summary prod-
   ucts: Parameterizing mineral diversity on Mars from reflectance.
   Journal of Geophysical Research, 112(E8)E08S14, 2007.
Porco C. C. and 35 colleagues. Imaging of Titan from the Cassini
   spacecraft. Nature, 434159–168, 2005.
Radebaugh J. and 6 colleagues. Mountains on Titan observed by
   Cassini Radar. Icarus, 192(1)77–91, 2007.
Radebaugh J. and 15 colleagues. Dunes on Titan observed by Cassini
   Radar. Icarus, 194(2)690–703, 2008.
Rodriguez S. and 9 colleagues. Cassini/VIMS hyperspectral obser-
   vations of the HUYGENS landing site on Titan. Planetary and
   Space Science, 54(15)1510–1523, 2006.
Rodriguez S. and 11 colleagues. Titan’s cloud seasonal activity from
   winter to spring with Cassini/VIMS. Icarus, 216(1)89–110, 2011.
Rodriguez S. and 20 colleagues. Global mapping and characterization
   of Titan’s dune fields with Cassini: Correlation between RADAR
   and VIMS observations. Icarus, 230168–179, 2014.
Rodriguez S. and 13 colleagues. Global circulation as the main source
   of cloud activity on Titan. Nature, 459(7247)678–682, 2009.
Seignovert B. Cassini Titan flyby, 2015.
Soderblom J. M. and 11 colleagues. Modeling specular reflections
   from hydrocarbon lakes on Titan. Icarus, 220(2)744–751, 2012.
Soderblom L. A. and 7 colleagues. Composition of Titan’s Surface,
   pp 141–175. Springer Netherlands, Dordrecht, 2009a.
Soderblom L. A. and 12 colleagues. The geology of Hotei Regio,
   Titan: Correlation of Cassini VIMS and RADAR. Icarus, 204(2)
   610–618, 2009b.
Solomonidou A. and 26 colleagues. The Spectral Nature of Titan’s
   Major Geomorphological Units: Constraints on Surface Compo-
   sition. Journal of Geophysical Research: Planets, 123(2)489–507,
   2018.
Solomonidou A. and 12 colleagues. Temporal variations of Titan’s
   surface with Cassini/VIMS. Icarus, 27085–99, 2016.


                                                                        13
```
