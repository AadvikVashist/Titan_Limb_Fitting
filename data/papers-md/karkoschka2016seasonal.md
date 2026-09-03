---
citation_key: "karkoschka2016seasonal"
title: "Seasonal variation of Titan’s haze at low and high altitudes from HST-STIS spectroscopy"
source_pdf: "data/papers/karkoschka2016seasonal.pdf"
source_pdf_sha256: "176f735ca657e7fb66c0dabb3fb80869a7f4742a89918a2aff1e265d0c831a2e"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
Accepted Manuscript

Seasonal variation of Titan’s haze at low and high altitudes from HST-STIS
spectroscopy

Erich Karkoschka

PII:                    S0019-1035(15)00298-5
DOI:                    http://dx.doi.org/10.1016/j.icarus.2015.07.007
Reference:              YICAR 11639

To appear in:           Icarus

Received Date:          16 March 2015
Revised Date:           24 June 2015
Accepted Date:          1 July 2015


Please cite this article as: Karkoschka, E., Seasonal variation of Titan’s haze at low and high altitudes from HST-
STIS spectroscopy, Icarus (2015), doi: http://dx.doi.org/10.1016/j.icarus.2015.07.007




This is a PDF file of an unedited manuscript that has been accepted for publication. As a service to our customers
we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and
review of the resulting proof before it is published in its final form. Please note that during the production process
errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                               1


    Seasonal variation of Titan’s haze at low and high
         altitudes from HST-STIS spectroscopy

                                     Erich Karkoschka

Erich Karkoschka, Lunar and Planetary Laboratory, University of Arizona, Tucson, AZ 85721-

0092, USA, phone 1 (520) 621-3994, fax 1 (520) 621-4933, e-mail erich@lpl.arizona.edu


Pages:    61
Figures: 15
Tables:    6

Proposed Running Head: Seasonal variation of Titan’s haze using HST-STIS
Highlights:
> HST-STIS image cubes in five years provide a unique probe of atmospheric variations.
> The first principal component describes haze opacity variations below 80 km altitude.
> The second one has similar variations above 150 km but its phase is 1-2 years ahead.
> The tropics have higher (>150 km) and lower (<80km) opacities than the global mean.
> The north-south asymmetry may reverse in 2016 (>150 km) and 2017-2018 (<80 km).

Key-words:     Titan; Titan atmosphere; Atmospheres, structure; Atmospheres, dynamics;

Satellites, atmospheres.


Submitted to Icarus, Special Issue on Titan: 2015-03-15       Revised: 2015-06-04, 2015-06-24
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                 2


                                          Abstract
      The Space Telescope Imaging Spectrograph accumulated image cubes of Titan in five years

between 1997 and 2004 that we calibrated and analyzed. The observations probe Titan’s early

northern fall to early winter. Methane bands between 543 and 990 nm wavelength are well

resolved spectrally, and Titan’s latitudinal and center-to-limb reflectivity variations are resolved

spatially.   A principal component analysis revealed two large components and two small

components of less significance. The first principal component describes a variation of Titan’s

haze below 80±20 km altitude. Haze particles change their size, opacity, and/or shape of the

single scattering phase function. The largest and smallest opacities occurred both in 1997 at high

southern latitudes and northern latitudes, respectively. The hemispherical asymmetry switched

sign in 2002 at low latitudes, in 2003 at mid latitudes, and in early 2004 at high latitudes. The

seasonal amplitude increased almost linearly with distance from the Equator. Tropical latitudes

had slightly lower opacities than the annual and global average if the observed variation is

seasonally symmetric and shaped like a sine curve.          The cause for the variation may be

condensation of gases onto aerosols seasonally driven by atmospheric dynamics. The second

principal component describes a variation of haze opacity at altitudes above 150±50 km. Largest

and smallest opacities both occurred in 2004 at northern and high southern latitudes, respectively.

The asymmetry switched in late 2001. Tropical latitudes had significantly higher haze opacity

than the annual and global average, opposite to the case at low altitudes. The cause for the high-

altitude variation may be aerosols transported at varying speeds driven by atmospheric dynamics.

We present a seasonal model that completely describes the haze parameters at each altitude,

latitude, and time. It compares fairly well with Cassini results obtained since 2004. The north-

south asymmetry may reverse in 2016 at high altitudes and 2017 through 2018 at low altitudes.

The observed variations are significant for modeling photometric data of Titan’s surface. They
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                3


describe several characteristics of Titan’s haze variations that can be compared with results from

global circulation models.



                                 Graphical Abstract
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                4


                                       1. Introduction
     Photometry of Titan since 1972 at 472 and 551 nm wavelength indicated a variation that

was first correlated with the 11-year solar cycle (Lockwood 1977) but later revealed to be part of

an almost periodic variation with a period of 15 years, which is half a Titan year and thus

suggests a seasonal variation (Lockwood and Thompson 2009). The phase of the variation

excluded the possibility of a pure geometric effect due to changing sub-Earth latitudes.

     The Voyager 1 and 2 spacecraft saw Titan obstructed by an almost featureless haze. Most

noticeable was that the northern hemisphere was darker than the southern one (Smith et al. 1981,

1982). Early Hubble Space Telescope (HST) images in 1990 revealed that the asymmetry had

reversed at wavelengths similar to Voyager’s filters, but that the asymmetry was opposite in the

889 nm methane filter (Caldwell et al. 1992). This wavelength dependence was consistent with

models by Toon et al. (1992) showing that an increase in haze optical depth causes darkening at

wavelengths below 600 nm but brightening at longer wavelengths.

     The observed asymmetry explained the phase, but not the amplitude of Titan’s photometric

variations (Sromovsky et al. 1981, 1986, Lockwood et al. 1986). Imaging by HST refined the

spectral characteristics and caught the next asymmetry reversal (Lorenz et al. 1996, 2004).

     While there has been work in understanding Titan’s seasonal variation through Global

Circulation Models (GCM) more recently (Rannou et al. 2004, Crespin et al. 2008, Lora et al.

2015), the best observational constraints on Titan’s seasonal variability come from data sets over

a sufficiently long time period that spatially resolve Titan’s disk. Three data sets are prominent

in this respect, data from the Hubble Space Telescope since 1990, from ground-based telescopes

with adaptive optics acquiring sufficient spatial resolution about 10 years later (Coustenis et al.

2001, Roe et al. 2002, Ádámkovics et al. 2006), and from the Cassini mission since 2004
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                  5



(Teanby et al. 2012, Vinatier et al. 2015). Ground-based telescopes miss out in recording the

north-south asymmetry at short wavelengths where adaptive optics does not work yet. Cassini

missed the last reversal that occurred around 2002, but may just catch the next one at the end of

its mission. Cassini greatly excels in spatial resolution and complete phase angle coverage.

Titan’s consistent phase angle seen from HST and HST’s photometric stability have advantages

in detecting small seasonal variations.

      HST imaging in a few filters recorded the seasonal variation well (Lorenz et al. 1997, 1999,

2001, 2004, 2006). A spectrally superior coverage comes from the Space Telescope Imaging

Spectrograph (STIS) at HST that offers observations of 1024 wavelengths simultaneously, a

wealth of spectral information. Methane spectroscopy allows probing many atmospheric levels.

STIS records spatial information along the slit in a single exposure, and multiple offsets of the

slit allow sampling of the second spatial dimension. Such data sets were taken in five years.

They provide a four-dimensional record of Titan’s seasonal change, one spectral dimension, one

temporal dimension, and two spatial dimensions: one for latitudinal sampling and one for the

center-to-limb variation of reflectivity. These data sets are the topic of this work.

      In the following section we describe the data and our method of obtaining calibrated image

cubes. Section 3 present a principal component analysis of the full data set. Section 4 explains

the observed spectral signatures with radiative transfer modeling. Section 5 fits the observed

latitudinal and temporal signatures with a seasonal model. Section 6 summarizes our work.



                                      2. Data reduction
      2.1 Observations

      HST took images cubes of Titan with STIS in five years between 1997 and 2004: 1997,
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                 6


1999, 2000, 2002, and 2004. In some years, multiple images cubes were taken within one orbit

of Titan around Saturn. We picked the closest one observing Titan’s trailing hemisphere. The

probed northern seasons were early fall to early winter. The last observation occurred six months

before Cassini arrived at Saturn. The G750L grating provided wavelengths between 530 and

1020 nm at 0.49 nm/pixel over 1024 pixels, of which the central ~1010 pixels gave useful data.

The spatial sampling was 0.05 arc-sec/pixel along the slit, providing up to 17 pixels across the

disk of Titan. The CCD can capture 1024 spatial pixels, but they are generally only partially

saved. We used the central 100 spatial pixels to include plenty of sky background data. Spatial

sampling perpendicular to the slit was accomplished through multiple exposures with slight

positional offsets spaced by ~3 minutes. The successive offset for the five data sets was -0.125,

0.05, -0.05, 0.1, and -0.05 arc-sec, respectively, where positive offsets indicate a motion toward

the evening hemisphere. The five data sets include between 3 and 19 different slit positions each.

      Table 1 lists the basic observational parameters. Included is Titan’s season with winter

solstice corresponding to Ls=270°. The rotation of Titan during each data set was less than ~1º

and thus negligible. The median exposure level on Titan was about 1500 electrons per pixel,

corresponding to a signal-to-noise ratio near 40. The principal investigators for these programs

were Mark Lemmon (GO 7321, 9385 and 9745), Caitlin Griffith (GO 8229, year 1999), and Eliot

Young (GO 8580, year 2000). Fig. 1 shows the central parts of the nine Titan images for the

observations of 2002, including the flatfield taken afterwards. Fig. 2 shows the slit locations on

Titan’s disk according to our calibration, as will be explained in Section 2.8.

                                      [Table 1 approximately here]

                                       [Fig. 1 approximately here]

                                       [Fig. 2 approximately here]



      2.2 Raw images
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                 7


      We used the standard calibration pipeline by the Space Telescope Science Institute (STScI).

This includes bias subtraction, flatfielding, and linearity correction (Ely et al., 2011). These

images are the basis for our further calibrations. We flagged ~50 pixels that had consistently low

or high data numbers. An automated routine also found about 13,000 pixels on Titan or nearby

that had high data numbers on a single exposure, most likely due to cosmic ray hits. We replaced

these “bad” data numbers using linear interpolation between adjacent “good” data numbers.



      2.3 Distortion correction

      In STIS exposures, the spectral and spatial directions are not perfectly aligned with the

column and row directions of the CCD, respectively. Furthermore, the wavelength is not a linear

function of wavelength, but has quadratic and cubic terms. We rectified all exposures using the

parameters from STScI (McGrath et al., 1998), except for the tilt between the spectral direction

and the CCD row axis, which was based on our measurements using flatfields taken right after

the Titan exposures. The flatfields display dark streaks due to locations of decreased slit width,

and the tilt of the streaks, about 0.3 pixels over the 1024 pixel extent, was accurately measured.



      2.4 CCD fringes

      Toward the longer wavelengths, STIS exposures display significant fringing as recorded in

flatfields (bottom of Fig. 1) with intensity variations of up to 30 % peak-to-peak. We averaged

each flatfield along columns and smoothed it spectrally by up to 30 pixels, which is larger than

the spacing of fringes. We then divided the original flatfield by the smoothed flatfield, which

created a fringe-flatfield for each observing date. Division by these fringe-flatfields reduced the

fringes in the Titan spectra. A comparison of albedo spectra (cf. Section 2.11) yielded that the

fringes were best reduced by increasing the amplitude of the fringes by 25 % at wavelength

below 960 nm and then linearly increasing to 45 % at 1020 nm. The original flatfields recorded
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                                   8


the fringes with muted amplitudes due to scattering of the CCD discussed in Section 2.6.



      2.5 Charge transfer efficiency

      The charge transfer efficiency in STIS exposures is dependent on the exposure level

(Goudfrooij and Kimble 2002, Goudfrooij et al., 2006). Based on Karkoschka and Tomasko

(2009) using a similar STIS image cube of Uranus, we adopted that 10 % of the signal was lost

for exposures with levels of 30 electrons/pixel, that the loss is proportional to the exposure level

to the power -0.4, and that the spatial deposition of charge toward lower columns was according

to an exponential decay of 1/e every 20 pixels. We deconvolved the images using the Wiener

method, which puts the lost charge back to its origin. The initial intensity asymmetry in the sky

background on both sides of Titan vanished after deconvolution.



      2.6 Scattered light

      The STIS CCD is a backside illuminated chip, which creates wide halos of scattered light,

especially longward of 750 nm, smoothing out spectral and spatial features. For example, at 905

nm wavelength, the fractional energy in the halo is ~30 % (Quijano et al., 2007). We used the

same deconvolution method as in Karkoschka and Tomasko (2009) to put all scattered light back

to its origin. Signal levels in the sky on both sides of Titan got strongly reduced, to levels

roughly consistent with telescope diffraction. The scattered light is up to ~10 times larger than

typical albedo variations on Titan. Thus accurate deconvolution of scattered light is essential.

Our method was sufficient for detecting variations as small as 0.3 % within STIS data, but does

not allow cross calibration with other data sets to similar levels, which is inherently difficult.



      2.7 Spectral calibration

      For each exposure, we co-added all data numbers along columns to obtain one-dimensional
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                  9


spectra. We used the solar Fraunhofer sodium D and hydrogen H-α lines (cf. Fig. 1) to calibrate

the wavelength scale in each exposure. We employed an automated routine that accurately

locates the line using the whole line profile. We averaged data for both lines with weights of

0.85 for the strong hydrogen and 0.15 for the weak sodium line. All measurements are displayed

in Fig. 3 (top). The lines can shift by 3 pixels between observing sessions, but only by a fraction

of a pixel within each observing session. Variations are close to linear with shifts of 0.015 pixels

per exposure number root-mean-square (rms), which is a measure of flexure within the

instrument. Our adopted straight lines in Fig. 3 differ from the measurements by 0.05 pixels rms,

which is probably our measurement precision. This corresponds to 0.025 nm. We used our fits

and the dispersion of 0.48836 nm/pixel provided by STScI to calibrate the wavelength scale.

                                       [Fig. 3 approximately here]

      The observations taken with the wide slit (~2 pixels wide in 1997, 2002, 2004) had a poorer

spectral resolution than the ones taken with the narrow slit (~1 pixel in 1999, 2000), which is

evident when measuring the depths of the solar lines. Additionally, observations taken near the

center of Titan with a fully illuminated slit had a poorer spectral resolution than those near

Titan’s limb, where only part of the slit was significantly illuminated, especially for the wide slit.

We convolved all spectra by 3-point averages with a weighing of the three pixels of x, 1-2x, x,

where we adjusted x for each slit width and location on Titan to produce spectra of identical

resolution, about 1 nm. The adjustment was small with typically |x| < 0.1. This allowed accurate

removal of solar features in the division by a solar spectrum of 1 nm spectral resolution.



      2.8 Spatial calibration

      For the spatial calibration, we co-added all data in each row to create one spatial profile for

each exposure. We located Titan’s limb by finding the profile’s steepest gradients using cubic

interpolation between data points (solid dots in the bottom panel of Fig. 3). The half-way point
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                                                10


between both measurements is shown by open circles. Linear fits to these data give shifts of

0.018 pixels per exposure number rms, which is similar to the instrument flexure in the spectral

direction (0.015). These shifts could also be due to a tracking error of 0.0009 arc-sec per

exposure number rms. The rms offset between open circles and linear fits is 0.035 pixels (0.0018

arc-sec), indicating our measurement precision. We adopted the linear fits as our navigation

along the slit.

      For the navigation perpendicular to the slit, we first adopted the commanded offset from

one exposure to the next (0.05, 0.1, or 0.125 arc-sec). In order to find Titan’s center, we used the

distance between the curves and central lines of Fig. 3. The distance as function of exposure

number could be fitted by a quadratic function with a small fourth-order term added, to 0.014

pixels rms (0.0007 arc-sec), except for the last two exposures in 2000 where the offset is ~40

times the rms value, maybe due to a tracking error (cf. Fig. 3). For those two exposures we used

estimates from our best-fit curve to adjust the commanded offset by 0.026 and 0.036 arc-sec,

respectively. The maximum of the distance function is our adopted navigation for Titan’s center.

      We refined this initial navigation in both axes with three improvements. First, Titan’s sharp

limbs in strong methane bands provides superior navigation.             Thus, we co-added only

wavelengths with methane absorption coefficients larger than 0.5 (km-am)-1. This modification

changed our navigation by 0.06 pixels rms, corresponding to 0.003 arc-sec rms.

      Second, at non-zero phase angles, automated navigation is typically biased toward the

bright limb due to uneven illumination. Simulations with synthetic images indicated that our

routine is biased by 1/6 of the distance between sub-solar and sub-Earth points, which we

adopted for our correction. This modification changed our navigation by 0.002 arc-sec rms.

      Third, our north-south navigation is also influenced by asymmetries on Titan’s disk.

Simulations using our routine suggested adjustments of 0.002 arc-sec rms which we adopted. We

estimate that this value is the accuracy of our navigation. Since the largest intensity gradients in
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                 11


our data up to 1 pixel inside of Titan’s limb are 25 %/pixel, the maximum photometric error due

to imperfect navigation is ~ 1 %, and but much less closer to Titan’s center.

      The orientation of the slit is provided by STScI (Fig. 2). We resampled all data using cubic

interpolation to a constant wavelength scale at 0.4 nm/pixel and a constant spatial scale of 0.06

Titan radii per pixel with the central column aligned with Titan’s central meridian.



      2.9 Telescope diffraction

      HST has a point spread functions (PSF) with a narrow, ~0.1 arc-sec core, but ~25 % of the

signal is diffracted by more than 0.2 arc-sec, which is about half of Titan’s radius.             We

deconvolved the wings of the PSF with the Wiener method using monochromatic PSFs from the

TinyTim software by STScI (Krist, 2004) every 40 nm wavelength and linear interpolation in

between. We smeared the PSFs perpendicular to the slit according to the slit width, and then

scaled and rotated them in the same way as our Titan image cubes. Our deconvolved data set has

a PSF with a full width at half maximum (FWHM) of 0.12 arc-sec. This value is similar to that

of the original PSF, but now the core contains 100 % of the signal instead of ~75 %.

Deconvolution needs the whole image of Titan which was not always observed. We estimated

the missing part by interpolation along concentric circles, linear interpolation in position angle.



      2.10 Albedo calibration

      Our albedo calibration uses the solar flux spectrum by Colina et al. (1996) and the

conversion between data number and flux for each wavelength adopted by Karkoschka and

Tomasko (2009, their Fig. 5) for a similar STIS image cube of Uranus. This calibration applies

for the first, fourth and fifth image cube, which used the same slit width as the Uranus data. For

the other two data sets with the narrower slit we determined additional calibration factors from

the flatfields.   This calibration is different from the STScI calibration because of different
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
                                                 12


methods in deconvolving the significant scattered light of the CCD.

      We adjusted the absolute calibration using the ground based photometry of Titan in the y

filter by Lockwood and Thompson (2009), which includes adjustment to constant distance from

sun and Earth. For comparison, we also used the geometric albedo spectra of Titan provided by

Karkoschka (1994, 1998).       The y filter is centered on 551 nm with a FWHM of 23 nm.

Lockwood and Thompson (2009) give a phase coefficient of Titan in the y band of 0.004

mag/deg and corrected their observations to zero phase angle. We did the same, assuming a

constant coefficient with wavelength, which increased reflectivities by up to 1.2 %. For each

image cube and each wavelength, we co-added all data spatially to calculate geometric albedos.

Then we convolved all spectra with the transmission curve of the y filter (Table 5 in Lockwood

and Thompson 2009) to calculate y-filter albedos.

      In order to plot albedos on the magnitude scale of Fig. 4, we adjusted three calibration

constants to get the best match with Lockwood’s data, two constants for the narrow and wide slit

and one for the spectra of Karkoschka (1994, 1998). The three constants fix the geometric albedo

in the y band, and the same calibration factor is adopted for the whole spectrum. Fig. 4 shows

that all our data is consistent within Lockwood’s error bars, which correspond to typically 0.3 %

in albedo. The rms deviation between their and the STIS data points is 0.2 %. The temporal

variation in STIS data is even smoother than in Lockwood’s data. We estimate that our absolute

calibration is good to 4 %, and our relative calibration within the data set to 0.2 %.

                                       [Fig. 4 approximately here]

      Fig. 5 shows a comparison of the average geometric albedo spectrum from this work with

the ground-based spectrum by Karkoschka (1998). In the visible, there is good agreement.

Toward infrared wavelengths our spectrum is consistently higher, similar to brightening observed

between 1993 and 1995 (Karkoschka 1998).

                                       [Fig. 5 approximately here]
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
                                               13




     2.11 Spectral reduction

     Titan’s spectrum is influenced smooth spectral variations and the steep methane spectrum.

Smooth variations include Rayleigh scattering, aerosol optical depth, aerosol optical parameters,

and Titan’s surface albedo. In our spectral range, smooth variations are sampled sufficiently well

at 5-10 wavelengths with cubic or spline interpolation in between. Likewise, the change of

reflectivity with the methane absorption coefficient is sampled sufficiently well at 5-10 methane

strengths. We divided our spectral range into eight wavelength regions roughly centered on a

methane band each (Fig. 5), and we divided methane absorption coefficients into seven regions

centered on coefficients that are spaced by factors of √10 based on coefficients for 100 K

temperature by Karkoschka and Tomasko (2010).

     This divides our spectral information into 25 spectral bands. Each spectral band represents

a specific range of wavelength and methane absorption coefficient. Each band gives the average

of some 40 original data points, which significantly decreases noise. Our averaging used weights

and correction factors to the ~40 data points in such a way that their average represents Titan’s

reflectivity for the nominal wavelength and methane absorption coefficient. We corrected for the

curvature of Titan’s spectrum since it affects reflectivity averages by about 1 %. Our reduction

from 1024 to 25 spectral data points retains all essential information about Titan. Spectral bands

with the same wavelength but different methane strengths and others with the opposite way

simplify the distinction between methane and smooth spectral variations (Sections 3-5). While

each wavelength has its methane absorption coefficient, different spectral bands with the same

central wavelength have different methane absorption coefficients.



     2.12 Titan’s surface

     At the longest wavelengths and smallest methane absorption coefficients, our images probe
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
                                                 14


Titan’s surface. We took a surface map of Titan (# PIA 14908) made from Cassini’s Imaging

Surface Subsystem (ISS) camera near 940 nm wavelength, smoothed it to our spatial resolution,

and projected it according to our sub-Earth longitudes and latitudes (Fig. 6, right side). Our

surface features correlate with the features seen by ISS. For each spectral band, we did a linear

regression of our reflectivity values with the general limb darkening subtracted versus the data

numbers in the ISS map. We averaged data from our five observing dates with weights according

to the total signal received from Titan. The slope of each linear fit indicates how much Titan’s

geometric albedo changes with surface albedo between the darkest and brightest spot in the ISS

map since the map was normalized between 0 and 1. This is shown as the dashed curve in Fig. 5.

Our curve confirms that Titan’s surface can be easily probed at the longest wavelengths and

smallest methane strengths. Our curve is noisy because of limited signal from the surface.

                                       [Fig. 6 approximately here]

      We used the projected ISS maps and the linear fit for each wavelength to correct all our

reflectivity data to a constant average surface albedo, which removed Titan’s surface features

from our images. This process was not perfect, but reduced features by a factor of three at which

point they were close to negligible. Remaining features may be due to imperfections of the ISS

map, which has intensity discontinuities at boundaries of original ISS images.



      2.13 Spatial reduction

      Our original data sample up to 17 latitudes along each slit. We divided Titan’s globe into

10 latitude regions so that each region contains one or two such data points. The 10 latitude

regions are divided by the latitude circles every 10° from 60º South to 20º North (cf. Fig. 3).

      The spatial resolution of HST is only four times smaller than Titan’s radius, so that the

center-to-limb variation of reflectivity within a latitude region can be approximated well with

simple functions. We used the linear limb darkening function:
```

<!-- PDF_PAGE: 16 -->

## PDF page 16

```text
                                                 15


      I/F = S μ0/(μ + μ0) + L μ0                                                               (1)

where I/F is the reflectivity, μ0 and μ are the cosines of the angles of incidence and emission,

respectively, and S and L are two fitted parameters. For S = 0 Eq. (1) becomes the Lambert Law

(Minnaert exponent 1), and for L = 0 the Lommel Seeliger Law (Minnaert exponent 0.5).

      We calculated scattering geometries based on the altitude where the aerosol optical depth

reaches 0.5 from Tomasko et al. (2008). This means that we used an optical radius of Titan

between 2600 and 2800 km with the radius increasing toward shorter wavelengths.

      We sampled our linear limb darkening function at μ = 0.6 and 0.8 since this is within the

typical range of observations of μ between about 0.5 and 0.9. The lower limit of 0.5 comes from

our exclusion of data points within one pixel of the limb. We call the reflectivity ratio I/F(μ=0.8)

/ I/F(μ=0.6) the limb darkening ratio; the larger the ratio, the stronger the limb darkening. In

every wavelength range, the limb darkening ratio decreases with increasing methane absorption

since I/F(μ=0.8) probes deeper into Titan’s atmosphere and is thus more affected by methane

absorption. Fig. 5 shows the strong correlation between albedo and limb darkening.

      In summary, we reduced the data from about 1 million original data points to 2500 data

points, 25 spectral bands, 10 latitudes, two center-to-limb locations, and five observing dates.



                          3. Principal component analysis
      3.1 Method

      The principal component analysis is a useful tool to understand the main features in a data

set if the data set is sufficiently complex so that other methods may miss some features, and if the

variations are sufficiently small so that the superposition of different effects is close to linear.

Both conditions are met in our data set.

      The principal component analysis has the goal of splitting a function of several parameters
```

<!-- PDF_PAGE: 17 -->

## PDF page 17

```text
                                                 16


into a product of two functions A and B where the first function depends on some parameters and

the other function depends on the remaining parameters. Both functions are typically much

simpler than the original one so that the characteristics of the data can be understood. The

functions are defined so that their product is as close as possible to the original one. Typically,

the sum of the squares of all residuals is minimized. In the second step, the residuals are

approximated by a product of two new functions as well as possible. In the third step, the same is

done for the remaining residuals, etc. Typically, the average with respect to the second set of

parameters is subtracted from the original function which gives averages of zero for the functions

B, which are then normalized. The solution is unique except for factors of -1 for each pair of

functions. The functions A have the unit of the original function, while the functions B have

scalar values. The procedure of finding the functions is facilitated by existing software packages.

      In our implementation, the first set of parameters represent different observations of the

same atmospheric structure: wavelength, methane strength, and viewing angle. The atmospheric

structure depends on the remaining parameters: latitude and time. Thus we expand our data set in

the following way:

      I/F(λ,κ,μ,φ,t) = C(λ,κ,μ) + ∑i Ai(λ,κ,μ) Bi(φ,t)                                         (2)

where λ is the spectral band’s central wavelength, κ the methane absorption coefficient, φ the

latitude on Titan, t the observing date, C the spatially and temporally averaged reflectivity, and Ai

and Bi the principal component functions, and i=1,2,… denoting the first, second principal

component, etc.      The functions Ai describes the spectral and center-to-limb dependence of

reflectivity for a certain physical change of the atmosphere relative to its average state described

by the parameter C. A1 describes the most significant physical change observed, A2 the second

most significant, etc. The functions Bi describe the size of the physical change for each location

and time, normalized to the rms change over the whole data set. We call individual values of the

functions Ai amplitudes since they indicate the rms variation of reflectivity over the whole data
```

<!-- PDF_PAGE: 18 -->

## PDF page 18

```text
                                                17


set for each principal component. For Bi we used the term values since they indicate the change

for a certain location and time relative to the rms variation for each principal component.

      We assigned weights to our 2500 data points according to the observed area on Titan’s disk

of each latitude section and reduced weights for the shortest and longest wavelengths where the

spectrograph has low sensitivity. The weighted average difference in reflectivity between I/F and

C is 0.011. Its square we call the 100 % variance for reference.

      We used iterations to find the first five principal components. They explain 83.8 %, 11.8

%, 2.6 %, 0.9 %, and 0.4 % of the variance, respectively, while the remaining 45 components

share the remaining 0.5 % of the variance. Since the data set contains 50 combinations of λ,κ,μ

and 50 combinations of φ,t, 50 principal components are needed to fit the data perfectly. The

fifth principal component has its highest amplitudes A5 at the extreme wavelengths, where noise

can be seen in Fig. 6. This suggests that the fifth and further components describe mostly noise.

The total noise corresponds to 1 % of the variance and thus a reflectivity precision of 0.001.

      The fourth principal component has its two highest amplitudes A4 at the 890 and 793 nm

continuum bands, respectively, and the next highest amplitudes at the next methane absorption

coefficients of both bands. This is exactly where the surface features of Titan are most visible

(Fig. 6). We do not consider it further since we focus on Titan’s atmosphere in this work.

      The first three principal component data are the main results of this work and displayed in

Fig. 7, showing A1 through A3 ordered first with methane absorption coefficient and then with

spectral band. Fig. 8 shows A1 and A2 ordered the opposite way. Values B1 through B3 ordered

first with latitude and then with observing date are displayed in Fig. 9, and B1 and B2 ordered in

the in opposite way in Fig. 10. The contributions of the first and second principal components to

the reflectivity across Titan’s disk for 1999 and 2002 are displayed at the bottom of Fig. 6. The

amplitudes A1 and A2 are listed in Table 2, the values B1 in Table 3, and B2 in Table 4.

                                    [Figs. 7-10 approximately here]
```

<!-- PDF_PAGE: 19 -->

## PDF page 19

```text
                                                18


                                    [Tables 2-4 approximately here]



      3.2 Interpretation

      The first and second principal components in Fig. 10 display north-south asymmetries that

change with time and are thus related to Titan’s seasonally changing north-south asymmetry,

described by Lorenz et al. (1997). The fact that two principal components explain essentially all

observed atmospheric variations is interesting. It suggests that the north-south asymmetry has

two different components that together explain essentially all recorded variations.

      The functions Bi describe changes as function of latitude and time. For now we only

consider the plotted data points in Fig. 10, the curves will become important in Section 5. Both

principal components show a strong north-south asymmetry in the beginning of our observing

period (1997). By the end of the period (2004), the first principal component reached symmetry

between north and south, while the second component reached symmetry already around 2001.

Note that Titan’s northern winter solstice occurred in late 2002, roughly half way between the

symmetry dates of both components. Thus, the components are phase delayed with respect to

seasons by a little more than 90° for the first and second principal component, and a little less

than 90° for the second one. The curves in Fig. 10 show a dip at the tropics for the first

component but a bulge for the second component. The conditions at tropical latitudes are not

close to Titan’s average. For the second principal component, tropical latitudes have a condition

close to those of southern latitudes before the contrast reversal. An enhanced haze around the

equator was detected by Cassini ISS and VIMS with high spatial resolution (Kok et al. 2010) that

probably corresponds to the feature seen in the second principal component as we will see later.

The strongest latitudinal gradients occur at mid latitudes, while beyond 50º South, the curves get

more shallow. All curves are smooth near the scale of spatial resolution. At smaller scales,

Titan’s north-south asymmetry has a relatively sharp edge (Kok et al. 2010), and at even smaller
```

<!-- PDF_PAGE: 20 -->

## PDF page 20

```text
                                                  19


scales of ~1° in latitude Cassini ISS has detected intricate structure of narrow strips of very low

contrast (Porco et al. 2005). In our data, the strongest small-scale feature, a little dip at 25º South

in 1997, is probably noise due to low data volume of the 1997 observations.

      Fig. 9 shows the same data with a little different emphasis. At high southern latitudes, the

first principal component was probably close to its maximum value at the first observation and

then close to zero at the last observation. Values decreased toward lower latitudes and at 5º

North, there was essentially no change, with a constant negative value as noted above. The

slightly rugged curve at 5º North has a peak-to-peak variations of about 0.4. Multiplied by the

typical size of A1 values of around 0.008 yields a rugged reflectivity of about 0.003, which is

close to the expected photometric stability and thus not a significant effect. For the second

principal component, the high southern latitudes show small positive values early on and then

very negative values after 2001. At 5º South, the atmosphere was stationary at a positive value.

At northern latitudes, values increased with almost constant speed.

      The functions A1 and A2 describe the spectral shape of reflectivity and center-to-limb

variations for the most and second most significant physical change on Titan. For the first

principal component, amplitudes are essentially zero at the spectral bands with the shortest

wavelengths and also at the strongest methane absorption coefficient observed in the 890 nm

band (Fig. 8). This indicates that the physical change described by the first component is

probably located at lower altitudes, where light is absorbed either by dark aerosols at short

wavelengths, or by methane gas in methane bands. The center-to-limb variation also indicates a

low altitude location. Everywhere, amplitudes are larger at μ = 0.8 (solid dots) that probes deeper

into the atmosphere than at μ = 0.6 (open circles). This is especially true at weak methane

absorptions at the longer wavelengths that probe the lower atmosphere.                 Essentially all

amplitudes of the first principal component are positive indicating a consistent effect.

      The situation is very different for the second principal component. The amplitudes are
```

<!-- PDF_PAGE: 21 -->

## PDF page 21

```text
                                                 20


negative for wavelengths below 700 nm and generally positive beyond (Fig. 8). The physical

change associated with the second principal component acts in different directions at short versus

long wavelengths. Almost all data points indicate more positive amplitudes for larger methane

absorption coefficients that probe the upper atmosphere. The center-to-limb variation goes the

same way everywhere. The open circles probing higher altitudes are above the solid dots probing

lower altitudes further suggesting that the second principal component is rooted at high altitudes.

Similar conclusions can be drawn from Fig. 7 displaying the same data in different order.



                           4. Radiative transfer modeling
      4.1 Altitude determination

      We based our radiative transfer calculation on the method and model by Doose et al.

(2015), the revised DISR haze model. It provides the optical depth, single scattering albedo and

phase function of the aerosols as function of wavelength and altitude. The model’s haze optical

depth has a scale height of around 50 km above ~100 km altitude and then gradually transitions

into a roughly linear increase of optical depth with decreasing altitude, corresponding to roughly

constant extinction coefficient (opacity). Since the model was based on data taken during the

descent of the Huygens probe, it is valid for a latitude of -10° in January 2005.

      Our data constrain the vertical optical depth profile of the haze relative to the methane

profile, but not both independently.      Since we expect that variations in the haze strongly

dominate those in the methane profile, we assumed the methane profile to be constant and

investigated only haze variations.

      To compare radiative transfer models with observations, we selected 22 spectral bands for

the µ = 0.8 scattering geometry. The µ = 0.6 data are neglected for now since they are more

affected by the third principal component. We did not consider the three spectral bands that are
```

<!-- PDF_PAGE: 22 -->

## PDF page 22

```text
                                                 21


most sensitive to the surface features, the continuum data at the 793 and 890 nm spectral bands

and the next methane absorption coefficient (0.10 (km-am)-1) at 890 nm because elimination of

surface features was not perfect. With these 22 data points, the first and second principal

components together capture 99 % of the variance, so that further components can be neglected.

      These 22 data points have an average wavelength of 780 nm and a standard deviation of 80

nm. Thus, our data can constrain optical parameters at 780 nm and can constrain significant

variations over an interval of 80 nm. The measured amplitudes of the two principal components

are too small to do more, such as determining a curvature in an optical parameter.

      At first we studied how variations of the haze opacity influence model reflectivities.

Within a selected altitude range, all haze opacities (haze extinction coefficients) were multiplied

by the same factor, which we call the opacity factor, while opacities were left unchanged outside

the selected altitude range. The 890 nm data points are best suitable for this purpose since they

include the spectral bands that probe the lowest and the highest altitudes in our data.

      Fig. 11 shows three models with the opacity increased with respect to the revised DISR

haze model at 890 nm below 60, 80, and 100 km altitude (top panel left to right). The solid

curves are from the models, plotting the difference of the model with larger opacity minus the

reference model. The solid dots are the observed amplitudes A1 (first principal component). The

first two dots are plotted in smaller size since they are not part of the selected 22 spectral bands

and thus do not influence the fits. Likewise, the dotted curves and open circles are model and

observed data for the µ = 0.6 scattering geometry that are not used for fits, but still should fit

roughly. In each panel, the opacity factor was adjusted to minimize least-square residuals. For

the second panel at top, the haze opacity within the 0-80 km altitude region was multiplied by

1.46. Larger or smaller opacity factors move the model curves proportionally away or closer to

the zero level, but do not change the shape. The 0-80 km model provides a good fit to the data

with residuals below 0.001. The curve for the 0-60 km case is too steep, and for the 0-100 km
```

<!-- PDF_PAGE: 23 -->

## PDF page 23

```text
                                                 22


case to shallow.

                                       [Fig. 11 approximately here]

      We also varied the bottom altitude limit of the opacity change and found always good fits

whenever the top limit was near 80 km. The data are not very sensitive to the bottom limit.

However, observed values of B1 of -3 or smaller require the removal of more haze than there is in

a shallow layer such as 60-80 km. Thus, our nominal altitude range for the first principal

component is 0-80 km.

      Fig. 11 shows a similar investigation for the A2 amplitudes. In this case, we changed the

haze opacity above 200, 150, and 100 km (bottom panel left to right). The first case has model

curves too steep, the last case too shallow. The middle case with opacities changed above 150

km provides the best fit and thus is our nominal case for the second principal component. We

tested extra models with the altitude range limited at various top levels and found little sensitivity

as long as the top limit was above 200 km, mostly because optical depths of Titan’s aerosols

above 200 km altitude are only on the order of 0.1 at 890 nm wavelength. Our models with sharp

boundaries at 80 or 150 km altitude are rough large-scale approximations, while real altitude

profiles may be smooth and more complicated on smaller scales.

      In our nominal cases, the data points not used for fitting are fairly close to the model

curves. For the first principal component, open circles are consistently slightly below the dashed

curve. This is probably due to the radiative transfer code that assumes Titan’s atmospheric layers

to be plane parallel. In reality, the light beam for a µ = 0.6 geometry has the first and last

scattering roughly at 120 km altitude. At the point where the beam reaches 80 km altitude, the

surface has curved away and the geometry has changed to µ = 0.58. Thus, the altitude region of

change should be calculated with µ < 0.58 but the simple code for a flat Titan uses 0.6 for all

altitudes. It underestimates the difference with respect to the µ = 0.8 geometry.

      The third principal component has values of about twice the noise which is insufficient to
```

<!-- PDF_PAGE: 24 -->

## PDF page 24

```text
                                                23


obtain a significant result, but indicates that something other than noise probably contributes.

The large A3 amplitudes for µ = 0.6 compared to µ = 0.8 (Fig. 7) cannot be reproduced in

radiative transfer models, suggesting that it does not correspond to a change on Titan. Its spectral

dependence seems to be correlated with that of the first principal component indicating that it

may be some non-linearity effect associated with the first component.

      Our radiative transfer calculations showed that the linearity of reflectivity according to Eq.

(1) was almost valid. For zero phase angle Eq. (1) means that the reflectivity is linear in µ, but

there was generally a small negative curvature. This means that reflectivity values at µ = 0.8 are

slightly overestimated in our fit if the observations do not include this scattering geometry but

only provide a linear extrapolation from smaller values of µ. This is the case for the -65° latitude

region at the first two observations dates, for -55° also at the first one, and for the 25° latitude

region at the last four dates. The overestimation implies that the principal component analysis

cannot fit the data perfectly and needs an extra component to describe these systematic effects for

the strongest component(s). Based on our implementation, one expects negative B3 values for

the latitudes listed above. This is roughly true (Fig. 9). Thus, the third principal component is

partially due to the linearity approximation of Eq. (1), and otherwise noise or other minor

systematic effects. We estimate that this effect caused overestimation of the B1 values by about

10 % for the latitudes listed above, which is barely significant since both extreme latitude regions

have results of lower quality anyway.



      4.2 Physical parameter determination

      In Fig. 8 we display similar models as in Fig. 11, but display all spectral bands. With the

added data points, the opacity factor for the lowest residual is 1.33, somewhat smaller than the

1.46 obtained above for the 890 nm band. Across all spectral bands, the model catches the main

features of the data reasonably well. The residuals are slightly reduced if the opacity factor
```

<!-- PDF_PAGE: 25 -->

## PDF page 25

```text
                                                24


increases with wavelength as would be expected when aerosols increase in size, not just in

numbers. Thus, the first principal component may characterize an increase of aerosol sizes that

directly causes an increase of opacity. Our data are not good enough to clearly distinguish the

contributions from increased numbers of aerosols and increased sizes.

      In Fig. 12, we display a model with the same altitude ranges as before, but with a changed

aerosol phase function instead of opacity: adding f times an isotropic phase function to the

original one multiplied by 1 – f. A spectrally constant f did not provide the best fit for the first

principal component, but a linear trend accomplished it when using f = 0.057 at 700 nm and f =

0.086 at 860 nm.       For the second principal component, our change of phase function is

completely inconsistent with the data.

                                      [Fig. 12 approximately here]

      Fig. 13 displays a model for a change of single scattering albedo at the same altitude ranges

as before. For both principal components, such models can be ruled out easily. Since the single

scattering albedo of the DISR haze model reaches values close to unity, we reduced it for our

changed values, but still plot the reflectivity difference of brighter minus darker model. This

figure emphasizes the power of spectroscopic data. Imaging data is often constrained to the

continuum and the center of a methane band, for example the first and last data points of the 890

nm spectral region. With those data points, the first principal component can be fit well with a

change of single scattering albedo. But the data points in between are critical to judge the model.

                                      [Fig. 13 approximately here]



      4.3 Discussion

      The approximate altitude regions for the first and second principal components are 0-80 km

and 150 km to the top of the atmosphere, respectively. Both can be explained by a change of

haze opacity. The first one can also be fit by a change of phase function, and a change increasing
```

<!-- PDF_PAGE: 26 -->

## PDF page 26

```text
                                                  25


with wavelength. The small residuals of the two possible cases for the first principal component

behave the opposite way. For the opacity case, the curves are above the data at middle bands

(around 727 nm) and below the data at the spectral ends. For the phase function case, the

opposite is true. A combined case with both parameters changing by about half as much as in

Fig. 11 and 12 fits the data even better. It is conceivable that as the aerosols fall through the

altitude region around 80 km, they change due to condensation of gases. The condensation could

change the aerosol size, the opacity, and the phase function of aerosols. Several gases are

expected to condense between 70 and 90 km altitude. Lavvas et al. (2011) argued that HCN

dominates and that aerosols in the 30-80 km altitude range should be strongly coated by HCN.

This altitude range fits the range determined for the first principal component since the data are

sensitive to the top altitude but not to the bottom altitude of the layer.

      It makes sense that for aerosol at low altitudes, opacities and phase functions have similar

effects for observed reflectivities. In both cases, more or less light is reflected back to the upper

atmosphere, by a variation of haze opacity or back scattering part of the phase function.

      The combination of the first two principal components represents Titan’s north-south

asymmetry that switches back and forth every Titan year. It was detected by Voyager (Smith et

al. 1981, 1982) and refined by HST (Lorenz et al. 1997, 1999, 2001, 2004, 2006), Cassini ISS

imaging (e.g. Porco et al. 2005), and Cassini VIMS observations (e.g. Brown et al. 2006). While

its latitudinal structure has been known well, its interpretation and vertical location has been

difficult to determine. Our STIS data probing multiple methane absorption strengths within the

same spectral bands are among the best current observations for that purpose.

      Cassini data have provided excellent constraints on reflectivity ratios over large temporal,

latitudinal, and spectral ranges (e.g. Penteado et al. 2010), but detection of small variations in

absolute reflectivities have been hampered by the constantly changing geometry. Ground-based

observations have a constantly low phase angle, but a variable Earth’s atmosphere.              HST
```

<!-- PDF_PAGE: 27 -->

## PDF page 27

```text
                                                 26


combines the advantages of both kinds of observations yielding to a photometric calibration of

0.2 % between the five STIS data sets that allowed accurate characterization of temporal change.

      The first principal components have been difficult to distinguish from each other because

their spatial and temporal variations are somewhat similar. Lorenz et al. (1999) noted a time

dependence of the north-south asymmetry with wavelength that became clearer later on (Lorenz

et al. 2001) based on imaging in about eight filters. It indicated that high and low altitudes do not

change in tandem. Ground-based studies using adaptive optics systems also indicated an earlier

reversal of the north-south asymmetry at high altitudes (~130 km) compared to altitudes around

50 km (Hirtzig et al. 2006). These studies of the hemispherical asymmetry ignored latitudinal

variations within each hemisphere. Our spectroscopic data and more comprehensive analysis

showed that both components are separated and have their own characteristics such as the

latitudinal profile as function of time.

      An explanation for the optical depth change at high altitudes is the transfer of aerosols by

atmospheric dynamics. Perhaps, there is a contribution from variable speeds of coagulation

changing aerosol sizes, also influenced by atmospheric dynamics.            Optical depths at our

wavelengths dependent strongly on aerosol sizes.

      Our second principal component probing mostly altitudes just above 150 km has a north-

south asymmetry that is opposite to the measurements of Titan’s shadow in 1995 that suggested

that aerosols around 300 km altitude are about half the size at high southern latitudes compared to

northern latitudes (Karkoschka and Lorenz 1996). This suggests that our results cannot be

extrapolated to altitudes 3-4 scale heights higher than the region our data probe.



                              5. Time variability modeling
      5.1 Seasonal model
```

<!-- PDF_PAGE: 28 -->

## PDF page 28

```text
                                                 27


      The functions Bi describe a scale factor for the characteristic change of each principal

component for various times and latitudes. Figs. 9 and 10 and Tables 3 and 4 show and list these

functions for five dates and 10 latitudes. In this section, we assume that all variations are purely

seasonal, and that the reflectivity at each latitude and scattering geometry varies with time as a

sine curve with the period of Titan’s year (29.46 years). We suspect that a major part of the

observed variations are seasonal, but probably not 100 %. Light curves in blue and green light

are similar to sine curves (Lockwood and Thompson 2009), but they are not perfectly periodic.

We approximate the temporal variation of the functions B1 and B2 by the following seasonal

variation:

      Bi(φ,t)= Mi(φ) + Di(φ) sin [(t – Ei(φ)) 2π/29.46]                                    (3)

where Mi is the mean value, Di is the amplitude of the seasonal variation, Ei is the epoch (in

decimal years) when values go through their annual mean, t is the date, and the index i indicates

the first or second principal component. The parameters Mi, Di, and Ei are the three parameters

for the sine curves. They are all functions of latitude.

      The temporal range of the observations, almost a Titan season, is insufficient to constrain

all three parameters of sine curves. This is especially the case since the three observing sessions

with a large number of slit positions and thus high quality data cover only a period of 3 years,

only 10 % of a Titan year. Thus we needed additional constraints for unique solutions. We

assumed that both hemispheres on Titan act symmetrically except for a phase shift of 180°, half a

Titan year. This provided adequate constraints for latitudes within 30° of the Equator but no

additional constraints at higher latitudes. Thus, beyond ±30°, we assigned the parameter Ei based

on measured values at low latitudes, at least for the second principal component where this

assumption was consistent with the data. For the first principal component, residuals decreased

significantly by increasing Ei by about a year for higher latitudes. Any further increase would

not have reduced residuals significantly. Thus, our data suggest that for the first principal
```

<!-- PDF_PAGE: 29 -->

## PDF page 29

```text
                                                 28


component, high latitudes have a slightly delayed seasonal change compared to the tropics.

      Our parameters of seasonal change are displayed in Fig. 14. As we noticed in Section 3,

the phase of the second principal component is ahead relative to the first one.            Seasonal

amplitudes Di for both components increase almost linearly away from the Equator.                The

temporal mean Mi is different at low and high latitudes. Our model describes an approximation

of the state of Titan for every time and all latitudes except close to the poles. The parameters

describe the changes of Titan’s haze that provide critical data for Global Circulation Models.

                                      [Fig. 14 approximately here]

      Fig. 10 displays the Bi values calculated from Eq. 3 as curves with the observations as dots.

Most data are fit well by the curves. Large deviations occur only where little data exist, indicated

by a small dot size in Fig. 10. The smallest dots indicate uncertainties five times larger as for the

largest dots. The good fit means the seasonal model is consistent with the data, but the validity at

other latitudes and times than observed has to be tested elsewhere.



      5.2 Haze opacity factors

      Section 4 showed physical parameter changes in Titan’s atmosphere that correspond to the

functions A1 and A2. Since they are multiplied by B1 or B2 in Eq. 2, they are appropriate

wherever the values B1 or B2 increase by unity. If B1 or B2 increase by 2, the reflectivity change

is twice as much, implying twice the change in the haze opacity if reflectivities go linear with

haze opacity. Using radiative transfer models, we determined the relationships between B1 and

B2 and opacity factors. They are almost but not perfectly linear (Table 5).

                                      [Table 5 approximately here]

      Our reference model is the DISR haze model valid for -10° latitude in early 2005. From

Figs. 9 and 10, we estimated parameters B1 = -1 and B2 = 0 for this latitude and time. Thus, our

mapping between opacity factors and B1 or B2 is anchored at these values. The parameter B1
```

<!-- PDF_PAGE: 30 -->

## PDF page 30

```text
                                                   29


refers to the opacity factor at altitude below 80 km, and B2 to altitudes above 150 km.

      The opacity factors of Table 5 go over wide ranges. Both go almost to zero and go up over

2 (B1) and to about 1.5 (B2). This suggests that the aerosol opacity at the low and high altitudes

undergoes large variations. Cassini observed large variations (e.g. Vinatier et al. 2015) that are

roughly consistent with our HST data, although Cassini data catch variations on much smaller

scales in altitude, latitude, and time.

      Fig. 15 displays the relationship between opacity factors on the right side and the values of

B1 and B2 on the left side. The seasonal model (Eq. 3) is plotted for a full Titan year. Tropical

latitudes that were observed on both sides of the Equator are plotted as solid curves, observed

latitudes further south as dashed curves, and the unobserved latitudes in the north as dotted

curves. They are based on the assumption of the seasonal model that southern and northern

latitudes behave symmetrically except for a two-season phase shift.

                                          [Fig. 15 approximately here]

      The characteristics of both principal components are evident in Fig. 15.            The annual

averages M1 do not vary much with latitude (top of Fig. 14). Thus, the top panel of Fig. 15 is

almost symmetric in the vertical direction. On the other hand, M2 is quite dependent on latitude

and thus the curves in the middle panel of Fig. 15 are quite asymmetric in the vertical direction.

The epoch parameter E2 is constant with latitude, creating a figure symmetric in time (middle

panel of Fig. 15). This is not the case for E1, which shows up as a horizontal shift of phasing

between low (solid curves) and high latitudes (dashed and dotted curves). The earlier phasing of

the second principal component with respect to the first one is also obvious.

      The accuracy of the seasonal model can be estimated from the rugged variations in Fig. 9

with a size of about 0.5 for the values of B1 and B2. Fig. 10 suggests similar differences between

observations (dots) and the seasonal model (curves), although larger dots for B1 differ typically

by only about 0.2 from the curves, while some smaller dots for B2 differ by unity from the
```

<!-- PDF_PAGE: 31 -->

## PDF page 31

```text
                                                30


curves. We conservatively estimate an accuracy of 0.5 for B1 and 1 for B2, which corresponds to

an accuracy of 0.2 for opacity factors. However, we suspect that the opacity variations could be

significantly smaller than in our model. At low altitudes, phase function variations may explain

part of the observed variation. At high altitudes, lowering the boundary of change from 150 to

100 km would decrease the required opacity deviations from unity by a factor of two.



     5.3 Discussion

     Fig. 15 shows that near the 2009 spring equinox, the B2 values for the northern hemisphere

cluster tightly near the maximum. Just south of the Equator, curves are spaced far apart but

clustered somewhat further south. This means that the northern and southern latitudes are very

roughly uniform with a relatively narrow transition region just south of the Equator. This is just

what Voyager observed shortly after the previous spring equinox, a bright southern and a dark

northern hemisphere with a transition region near 10° South (Smith et al. 1981, 1982). Since

higher haze opacities correspond to darker reflectivities, Voyager images fit the seasonal model.

     The variations observed by HST are consistent with ground-based photometry by

Lockwood and Thompson (2009) to 0.002 mag rms, corresponding to 0.2 % (Fig. 4). The

previous cycle of the 1970’s and 1980’s had a 40 % larger amplitude than the cycle where the

HST data were obtained. The simplest adjustment of our seasonal model for the earlier cycle is

increasing the D2 amplitude parameter by 40 %. In the earlier cycle, the brightness at green

wavelengths (551 nm) stayed six years above its mean but more than eight years below

(Lockwood and Thompson 2009) indicating deviations from our idealized sine curve model.

     Our seasonal model includes high latitudes during the winter darkness that are difficult to

observe, in particular the strongly negative values of B1 and B2 near -4. It seems unlikely that

aerosol opacities get as small as suggested by the seasonal model. The observed ranges of B1

and B2 displayed in Figs. 9 and 10 stay above -1 and -2, respectively. The seasonal model is an
```

<!-- PDF_PAGE: 32 -->

## PDF page 32

```text
                                                 31


extrapolation of limited data that needs to be verified. In the darkness of winter, aerosols can still

be probed in emission with Cassini’s Composite Infrared Spectrometer (CIRS). Observations by

Vinatier et al. (2015) indicated low haze extinction coefficients at high southern latitudes from

2009 to 2011, in qualitative agreement with our seasonal model, although it is unknown whether

the aerosol opacities at wavelengths separated by a factor of 10 go hand in hand. Earlier CIRS

data between 2004 and 2008 (Vinatier et al. 2010), probing near 160 km altitude, indicated a

roughly constant haze extinction coefficient northward of 5° latitude but linearly decreasing to

the south, qualitatively consistent with our high altitude model for the same period.

      Cassini VIMS data taken at the end of 2006 analyzed by Rannou et al. (2010) show a

latitudinal profile of the haze opacity that increases almost linearly from -80° to the equator, then

is levelled to 30° and decreases slightly to 60°. This is very similar to the latitudinal shape of the

second principal component at that date except that the model does not have the decrease above

30°. The observed variation is only half as large as ours, which corresponds to altitudes above

150 km. Rannou et al. (2010) estimate that their data corresponds to the optical depth above 100

km altitude. The altitude range of 100-150 km does not vary with time in our model and contains

about as much haze optical depth as all altitudes above 150 km. Thus, even the apparent size

discrepancy by a factor of two is explained. These VIMS profiles come from data between 1.7

and 3.5 µm wavelength and are thus closer to our spectral range than the CIRS data near 9 µm.

      Since the haze is influenced by the dynamics in Titan’s atmosphere, one can compare our

simple seasonal model of the haze with other parameters probing the circulation. Tokano et al.

(1999) used a GCM to show that seasonal temperature variations at 1 mb pressure (~180 km

altitude) are expected to have a flatter maximum and a sharper minimum.                 The measured

distribution of trace gases with Cassini CIRS (Teanby et al. 2008, 2009) indicated latitudinal

structure that is more complicated than the simple model presented here, but also showed very

little temporal variation in the lower stratosphere where the presented model has no change in the
```

<!-- PDF_PAGE: 33 -->

## PDF page 33

```text
                                                 32


haze either, at least within the 80-150 km altitude range.



                                          6. Summary
      We analyzed spatially resolved Titan spectrophotometry collected by STIS in five years

over a 7-year period to create five image cubes, each consisting of an image of Titan at ~1000

wavelengths between 530 and 1020 nm. We took special effort in the calibration process to

extract variations as small as 0.001 in reflectivity. This is a strength of our data set compared to

ground-based observations taken through a variable Earth’s atmosphere, and compared to Cassini

observation with constantly changing phase angles.

      The ~1 million original data points were grouped and averaged into 2500 reduced data

points, a four-dimensional data set of 25 spectral bands, 10 latitude regions, five dates, and two

viewing angles. A principal component analysis characterized the significant features in this data

set. We found two large, distinct components indicating major changes of Titan’s haze. A small

third component indicated a non-linearity effect of the first one. A fourth component described

changing surface features due to Titan’s rotation.

      Radiative transfer models revealed the physical variations in Titan’s atmosphere:

      1. The first principal component is due to a change in haze opacity or a change in the shape

of the phase function below 80±20 km altitude.          Residuals are very low if both physical

parameters contribute.     The variation appears to be stronger toward longer wavelengths

suggesting a change of aerosol size too. A possible physical explanation is that gases, perhaps

HCN, condense at around 80 km altitude onto aerosols and increase in size, and that the speed of

condensation varies with location and seasons based on atmospheric dynamics.

      2. The second principal component is due to a change of haze opacity at altitudes above

150±50 km. It is probably driven by atmospheric dynamics moving aerosols. Variable speeds of
```

<!-- PDF_PAGE: 34 -->

## PDF page 34

```text
                                                  33


coagulation of aerosols may contribute, which is also influenced by dynamics. Both principal

components together characterize Titan’s seasonal north-south asymmetry that is well known,

including its wavelength dependent phasing (Lorenz et al. 1997, 1999, 2001, 2004, 2006), but its

separation into two distinct components is new.

      3. We did not detect any temporal or latitudinal change of the haze between 80 and 150 km

altitude.

      4. The tropics have significantly larger haze opacity at high altitudes (>150 km) but

slightly reduced opacity at low altitudes (<80 km) compared to the global average.

      5. The high-altitude haze opacity was symmetric with respect to the Equator at the end of

2001, at which point its north-south asymmetry reversed. The data are consistent with all

latitudes switching asymmetry at the same time, but the constraint is weak. The opposite season

will occur in spring 2016, when the next reversal may take place.

      6. The asymmetry reversal for the low-altitude haze opacity occurred later than at high

altitudes, for tropical latitudes in the middle of 2002, for mid latitudes in 2003, and high latitudes

about early 2004. The next reversal may occur throughout 2017 and 2018. The last reversal in

this direction occurred sometimes between 1981 (Voyager 2) and 1990 (HST).

      7.    Amplitudes of temporal variations for both principal components increase almost

linearly with distance from the Equator.

      8. The data can be fit well by assuming that all observed variations are purely seasonal

with shapes of sine curves. This seasonal model completely describes Titan’s haze parameters at

each altitude, latitude, and time (Table 6). While the model fits the data between 1997 and 2004,

it also is roughly consistent with a number of Cassini observations made since 2004 (cf. Sections

4.3 and 5.3). The model may be an approximate description of Titan’s seasonal haze cycle that

changed significantly from the last cycle to the current one (Lockwood and Thompson 2009).

                                      [Table 6 approximately here]
```

<!-- PDF_PAGE: 35 -->

## PDF page 35

```text
                                                34


     Our model provides more realistic atmospheric corrections in photometric investigations of

Titan’s surface than using a constant model such as the DISR haze model that was determined at

a specific date and latitude.    The three parameters of our seasonal model, mean values,

amplitudes, and phasing, and their variations with latitude can be compared with studies of

Global Circulation Models leading to a better understanding of Titan’s seasonal cycle.



                                    Acknowledgements
     This work was supported by the STScI through Archive Research program AR 12624. The

contribution of Caitlin Griffith, the principal investigator for this program, is appreciated. This

work was based on HST data from observations with Principal Investigators Mark Lemmon,

Caitlin Griffith, and Eliot Young. Support for HST was provided by NASA through the Space

Telescope Science Institute, which is operated by the Association of Universities for Research in

Astronomy, Incorporated, under NASA contract NAS5-26555.
```

<!-- PDF_PAGE: 36 -->

## PDF page 36

```text
                                                35


                                           References

       Ádámkovics, M., de Pater, I., Hartung, M., Eisenhauer, F., Genzel, R. 2006. Titan’s bright

spots: multiband spectroscopic measurement of surface diversity and hazes. J. Geophys. Res.

111, E07S06.

       Brown, R.H., and 25 colleagues 2006. Observations in the Saturn system during approach

and orbital insertion, with Cassini’s visual and infrared mapping spectrometer (VIMS). Astron.

Astrophys. 446, 707-716.

       Caldwell, J.D., Smith, P.H., Tomasko, M.G., McKay, C.P. 1992. Titan: evidence of

seasonal change—A comparison of Voyager and Hubble Space Telescope images. Icarus 103,

1-9.

       Colina, L., Bohlin, R.C., Castelli, F. 1996. The 0.12-2.5 micron absolute flux distribution

of the Sun for comparison with solar analog stars. Astronom. J. 112, 307-315.

       Coustenis, A., Gendron, E., Lai, O., Véran, J.-P., Woillez, J., Combes, M., Vapillon, L.,

Fusco, T., Mugnier, L., Rannou, P. 2001. Images of Titan at 1.3 and 1.6 µm with adaptive optics

at the CFHT. Icarus 154, 501-515.

       Crespin, A., Lebonnois, S., Vinatier, S., Bézard, B., Coustenis, A., Teanby, N.A.,

Achterberg, R.K., Rannou, P., Hourdin, F. 2008. Diagnostics of Titan’s stratospheric dynamics

using Cassini/CIRS data and the two-dimensional IPSL circulation model. Icarus 197, 556-571.

       de Kok, R., Irwin, P.G.J., Teanby, N.A., Vinatier, S., Tosi, F., Negrão, A., Osprey, S.,

Ardiani, A., Moriconi, M.L., Coradini, A. 2010. A tropical haze band in Titan’s stratosphere.

Icarus 207, 485-490.

       Doose, L.R., Karkoschka, E., Tomasko, M. 2015. Vertical structure and optical properties
```

<!-- PDF_PAGE: 37 -->

## PDF page 37

```text
                                               36



of Titan’s aerosols from measurements made inside and outside the atmosphere. Submitted to

Icarus.

     Ely, J., Aloisi, A., Bohlin, R., Bostroem, A., Diaz, R., Dixon, V., Goudfrooij, P. Hodge, P.,

Lennon, D., Long, C., Niemi, S., Osten, R., Proffitt, C., Walborn, N., Wheeler, T., Wolfe, M.,

York, B., Zheng, W., 2011. STIS Instrument Handbook, Version 11.0, STScI, Baltimore.

     Goudfrooij, P., Kimble, R.A. 2002. Correcting STIS CCD photometry for CTE loss. 2002

HST calibration Workshop, Arribas, S., Koekemoer, A., Whitmore, B. eds., STScI, Baltimore.

     Goudfrooij, P., Maíz-Apellániz, J., Brown, T., Kimble, R.          2006.    Charge transfer

efficiency of the STIS CCD: The time dependence of charge loss and centroid shifts from

Internal Sparse Field data. Instrument Science Report STIS 2006-01, STScI, Baltimore.

     Hirtzig, M., Coustenis, A., Gendron, E., Drossart, P., Negrão, A., Combes, M., Lai, O.,

Rannou, P., Lebennois, S., Luz, D. 2006. Monitoring atmospheric phenomena on Titan. Astron.

Astrophys. 456, 761-774.

     Karkoschka, E. 1994. Spectrophotometry of the Jovian planets and Titan at 300- to 1000-

nm wavelength: the methane spectrum. Icarus 111, 174-192.

     Karkoschka, E. 1998. Methane, ammonia, and temperature measurements of the jovian

planets and Titan from CCD-spectrophotometry. Icarus 133, 134-146.

     Karkoschka, E., Lorenz, R.D. 1996. Latitudinal variation of aerosol sizes inferred from

Titan’s shadow. Icarus 125, 369-379.

     Karkoschka, E., Tomasko, M. 2009. The haze and methane distributions on Uranus from

HST-STIS spectroscopy. Icarus 202, 287-309.

     Karkoschka, E., Tomasko, M.G. 2010. Methane absorption coefficients for the jovian

planets from laboratory, Huygens, and HST data. Icarus 205, 674-694.
```

<!-- PDF_PAGE: 38 -->

## PDF page 38

```text
                                              37



      Krist, J. 2004. The Tiny Tim User’s Guide, Version 6.3. Space Telescope Science

Institute, Baltimore.

      Lavvas, P., Griffith, C.A., Yelle, R.V. 2011. Condensation in Titan’s atmosphere at the

Huygens landing site. Icarus 215, 732-750.

      Lockwood, G.W. 1977. Secular brightness increases of Titan, Uranus, and Neptune, 1972-

1976. Icarus 32, 413-430.

      Lockwood, G.W., Thompson, D.T. 2009. Seasonal photometric variability of Titan, 1972-

2006. Icarus 200, 616-626.

      Lora, J.M., Lunine, J.I., Russell, J.L. 2015. GCM simulations of Titan’s middle and lower

atmosphere and comparison to observations. Icarus 250, 516-528.

      Lorenz, R.D., Smith, P.H., Lemmon, M.T., Karkoschka, E., Lockwood, G.W., Caldwell, J.

1997. Titan’s north-south asymmetry from HST and Voyager imaging: comparison with models

and ground-based photometry. Icarus 127, 173-189.

      Lorenz, R.D., Lemmon, M.T., Smith, P.H., Lockwood, G.W. 1999. Seasonal change

observed on Titan’s with the Hubble Space Telescope WFPC-2. Icarus 142, 391-401.

      Lorenz, R.D., Young, E.F. Lemmon, M.T.         2001.   Titan’s smile and collar:    HST

observations of seasonal change 1994-2000. Geophys. Res. Lett. 28, 4453-4456.

      Lorenz, R.D., Smith, P.H., Lemmon, M.T. 2004. Seasonal change in Titan’s haze 1992-

2002 from Hubble Space Telescope observations. Geophys. Res. Lett. 31, L10702.

      Lorenz, R.D., Lemmon, M.T., Smith, P.H. 2006. Seasonal evolution of Titan’s dark polar

hood: midsummer disappearance observed by the Hubble Space telescope. Monthly Notices

Royal Astron. Soc. 369, 1683-1687.

      McGrath, M.A., Hodge, P., Baum, S. 1998. Calstis7: Two-dimensional rectification of
```

<!-- PDF_PAGE: 39 -->

## PDF page 39

```text
                                               38



spectroscopic data in the STIS calibration pipeline. Instrument Science Report STIS 98-13.

STScI, Baltimore.

     Penteado, P.F., Griffith, C.A., Tomasko, M.G., Engel, S., See, C., Doose, L., Baines, K.H.,

Brown, R.H., Buratti, B.J., Clark, R., Nicholson, P., Sotin, C. 2010. Latitudinal variations in

Titan’s methane and haze from Cassini VIMS observations. Icarus 206, 352-365.

     Porco, C.C., and 35 colleagues 2005. Imaging of Titan from the Cassini spacecraft.

Nature 434, 159-168.

     Quijano, K., Aloisi,A., Brown,T., Busko, I., Davies, J., Diaz-Millaer, A., Dressel, L.,

Goudfrooij, P., Hodge, P., Hofeltz, S., Kaiser, M.E., Apellániz, J.M., Mobasher, B., Proffitt, C.,

Sahu, K., Stys, D., Walborn, N.      2007.   STIS Instrument Handbook, Version 8.0.         Space

Telescope Science Institute, Baltimore, Maryland.

     Rannou, P., Hourdin, F., McKay, C.P., Luz, D. 2004. A coupled dynamics-microphysics

model of Titan’s atmosphere. Icarus 170, 443-462.

     Rannou, P., Cours, T., Le Mouélic, S., Rodriguez, S., Sotin, C., Drossart, P., Brown, R.

2010. Titan haze distribution and optical properties retrieved from recent observations. Icarus

208, 850-867.

     Roe, H.G., de Pater, I., Macintosh, B.A., Gibbard, S.G., Max, C.E., McKay, C.P.        2002.

Titan’s atmosphere in late southern spring observed with adaptive opticson the W. M. Keck II

10-meter telescope. Icarus 157, 254-258.

     Smith, B.A., Soderblom, L.A., Beebe, R., Boyce, J., Briggs, G., Bunker, A., Collins, S.A.,

Hansen, C.J., Johnson, T.V., Mitchell, J.L., Terrile, R.J., Carr, M., Cook II, A.F., Cuzzi, J.,

Pollack, J.B., Danielson, G.E., Ingersoll, A., Davies, M.E., Hunt, G.E., Masursky, H.,

Shoemaker, E., Morrison, D., Owen, T., Sagan, C., Veverka, J., Strom, R., Suomi, V. 1981.
```

<!-- PDF_PAGE: 40 -->

## PDF page 40

```text
                                               39



Encounter with Saturn: Voyager 1 imaging results. Science 212, 163-182.

       Smith, B.A., Soderblom, L.A., Batson, B., Bridges, P., Inoe, J., Masursky, H., Shoemaker,

E., Beebe, R., Boyce, J., Briggs, G., Bunker, A., Collins, S.A., Hansen, C.J., Johnson, T.V.,

Mitchell, J.L., Terrile, R.J., Carr, M., Cook II, A.F., Cuzzi, J., Pollack, J.B., Danielson, G.E.,

Ingersoll, A., Davies, M.E., Hunt, G.E., Morrison, D., Owen, T., Sagan, C., Veverka, J., Strom,

R., Suomi, V. 1982. A new look at the Saturn system: The Voyager 2 images. Science 215,

504-537.

       Sromovsky, L.A., Suomi, V.E., Pollack, J.B., Kraus, R.J., Limaye, S.S., Owen, T.,

Revercomb, H.E., Sagan, C. 1981. Implications of Titan’s north-south brightness asymmetry.

Nature 292, 698-702.

       Sromovsky, L.A., Lockwood, G.W., Thompson, D.T. 1986. Titan’s albedo variation:

Relative influence of solar UV and seasonal contrast mechanism. Bull. Am. Astron. Soc. 18,

809.

       Teanby, N.A., Irwin, P.G.J., de Kok, R., Nixon, C.A., Coustenis, A., Royer, E., Calcutt,

S.B., Bowles, N.E., Fletcher, L., Howett, C., Taylor, F.W. 2008. Global and temporal variations

in hydrocarbons and nitriles in Titan’s stratosphere for northern winter observed by

Cassini/CIRS. Icarus 193, 595-611.

       Teanby, N.A., Irwin, P.G.J., de Kok, R., Nixon, C.A. 2009. Dynamical implications of

seasonal and spatial variations in Titan’s stratospheric composition. Phil. Trans. Roy. Soc. 367,

697-711.

       Teanby, N.A., Irwin, P.G.J., Nixon, C.A., de Kok, R., Vinatier, S., Coustenis, A., Sefton-

Nash, E., Calcutt, S.B., Flasar, F.M. 2012. Active upper-atmosphere chemistry and dynamics

from polar circulation reversal on Titan. Nature 491, 732-735.
```

<!-- PDF_PAGE: 41 -->

## PDF page 41

```text
                                               40



     Tokano, T., Neubauer, F.M., Laube, M., McKay, C.P. 1999. Seasonal variation of Titans

atmospheric structuresimulated by a general circulation model. Planet. Space Sci. 47, 493-520.

     Tomasko, M.G., and 39 colleagues 2005. Rain, winds and haze during the Huygens

probe’s descent to Titan’s surface. Nature 438, 765-778.

     Tomasko, M.G., Doose, L., Engel, S., Dafoe, L.E., West, R., Lemmon, M., Karkoschka, E.,

See, C. 2008. A model of Titan’s aerosols based on measurements made inside the atmosphere.

Planet. Space Sci 56, 669-707.

     Vinatier, S., Bézard, B., de Kok, R., Anderson, C.M., Samuelson, R.E., Nixon, C.A.,

Mamoutkine, A., Carlson, R.C., Jennings, D.E., Guandique, E.A., Bjoraker, G.L., Flasar, F.M.,

Kunde, V.G. 2010. Analysis of Cassini/CIRS limb spectra acquired of Titan during the nominal

mission II: aerosol extinction profiles in the 600-1420 cm-1 spectral range. Icarus 210, 852-866.

     Vinatier, S. Bézard, B., Lebonnois, S., Teanby, N.A., Achterberg, A.K., Gorius, N.,

Mamoutkine, A., Guandique, E., Jolly, A., Jennings, D.E. Flasar, F.M. 2015. Seasonal variation

in Titan’s middle atmosphere during the northern spring derived from Cassini/CIRS observations.

Icarus 250, 95-115.
```

<!-- PDF_PAGE: 42 -->

## PDF page 42

```text
                                                         41


Table 1: Observational parameters
Date, UT              Ls       lg(earth) φ(earth) φ(sun)             diameter      phase angle # of exposures

                      (°)      (º)          (º)         (º)          (arc-sec)      (º)

-----------------------------------------------------------------------------------------------------------------

1997-11-03.44         204      306           -9.4       -10.8        0.837         2.7              3

1999-10-07.95         229      319          -20.7       -19.9        0.851         3.3             18

2000-11-23.76         244      267          -23.6       -23.8        0.873         0.6             19

2002-11-27.55         272      241          -26.4       -26.6        0.875         2.4              9

2004-01-03.42         287      300          -25.5       -25.4        0.881         0.3              7

------------------------------------------------------------------------------------------------------------------

Note: lg and φ are sub-earth/sub-solar longitudes and latitudes, respectively. Ls is the solar

longitude relative to the spring equinox.
```

<!-- PDF_PAGE: 43 -->

## PDF page 43

```text
                                                       42


     Table 2: Principal component amplitudes A1 and A2
     Wavelength       κ(CH4)            amplitude A1                amplitude A2
                                    ________________             ________________
     nm                km-am-1 µ=0.6               µ=0.8         µ=0.6          µ=0.8
     ------------------------------------------------------------------------------------
     543                0.03         -0.0008       -0.0001       -0.0036        -0.0070
     543               0.10         -0.0007        0.0004        -0.0031       -0.0063
     576               0.03          0.0003        0.0016        -0.0040       -0.0073
     619               0.03          0.0019        0.0043        -0.0036       -0.0063
     619               0.10          0.0024        0.0051        -0.0031       -0.0059
     619               0.32          0.0029        0.0064        -0.0016       -0.0037
     666               0.03          0.0038        0.0079        -0.0023       -0.0046
     666               0.10          0.0044        0.0089        -0.0016       -0.0037
     727               0.03          0.0064        0.0120        -0.0009       -0.0028
     727               0.10          0.0072        0.0129        -0.0002       -0.0020
     727               0.32          0.0077        0.0134         0.0008       -0.0010
     727               1.0           0.0070        0.0122         0.0028        0.0016
     727               3.2           0.0043        0.0079         0.0056       0.0050
     793               0.03          0.0086        0.0146         0.0006       -0.0014
     793               0.10          0.0097        0.0154         0.0009       -0.0010
     793               0.32          0.0104        0.0159         0.0020        0.0002
     793               1.0           0.0095        0.0146         0.0038        0.0024
     890               0.03          0.0108        0.0164         0.0012       -0.0003
     890               0.10          0.0119        0.0175         0.0020        0.0004
     890               0.32          0.0125        0.0171         0.0024        0.0008
     890               1.0           0.0111        0.0146         0.0036        0.0018
     890               3.2           0.0066        0.0090         0.0057        0.0040
     890              10             0.0023        0.0040         0.0069        0.0058
     890              32            -0.0002        0.0006         0.0068        0.0058
      990               10             0.0040        0.0044        0.0059        0.0044
      ------------------------------------------------------------------------------------
      Note: Bands are identified by the listed central wavelength and the methane absorption
coefficient κ(CH4).
```

<!-- PDF_PAGE: 44 -->

## PDF page 44

```text
                                                  43


Table 3: Principal component values B1
Latitude 1997           1999         2000        2002         2004

--------------------------------------------------------------------

-65°         3.04        2.99         2.11        0.65        0.01

-55°         2.83        2.70         1.87        0.63        -0.08

-45°         2.16        1.96         1.23        0.32        -0.15

-35°         1.12        1.03         0.53       -0.22        -0.30

-25°         0.32        0.29        -0.04       -0.57        -0.65

-15°         0.04       -0.28        -0.48       -0.71        -0.94

 -5°        -0.45       -0.58        -0.79       -0.68        -1.11

  5°        -0.88       -0.74        -1.04       -0.63        -1.04

15°         -1.36       -0.82        -1.10       -0.43        -0.75

25°         -1.54       -0.70        -1.05       -0.26        -0.49

--------------------------------------------------------------------
```

<!-- PDF_PAGE: 45 -->

## PDF page 45

```text
                                                  44


Table 4: Principal component values B2
Latitude 1997           1999         2000        2002         2004

---------------------------------------------------------------------

-65°         0.17        0.37        -0.81       -2.09        -2.45

-55°         0.32        0.31        -0.85       -2.01        -2.45

-45°         0.35        0.38        -0.62       -1.79        -1.87

-35°         0.51        0.46        -0.32       -1.35        -1.37

-25°         0.76        0.74         0.31       -0.74        -0.72

-15°         0.93        1.04         0.87       -0.01        0.17

 -5°         0.92        1.05         1.01        0.74        0.94

  5°         0.54        0.64         0.86        1.00        1.36

15°         -0.41       -0.21         0.32        0.90        1.49

25°         -1.54       -1.34        -0.40        0.61        1.37

---------------------------------------------------------------------
```

<!-- PDF_PAGE: 46 -->

## PDF page 46

```text
                                     45


Table 5: Haze opacity factors
B1, B2      factor      factor

            for B1      for B2

----------------------------------

-4.0        0.09        0.12

-3.5        0.23        0.22

-3.0        0.37        0.33

-2.5        0.53        0.43

-2.0        0.68        0.54

-1.5        0.84        0.65

-1.0        1.00        0.77

-0.5        1.17        0.88

0.0         1.34        1.00

0.5         1.51        1.12

1.0         1.69        1.25

1.5         1.88        1.38

2.0         2.08        1.51

2.5         2.28        1.65

3.0         2.49          -

3.5         2.71          -

4.0         2.94          -

----------------------------------
```

<!-- PDF_PAGE: 47 -->

## PDF page 47

```text
                                         46


Table 6: Our seasonal haze model
Haze parameters from Doose et al. (2015), except:

1)        altitudes 0-80 km: multiply haze opacities within this layer by the opacity factor

for B1 (cf. Table 5) where B1 is calculated by Eq. 3 from M1, D1, and E1, and these

parameters as function of latitude are taken from Fig. 14 (for a southern latitude, use the

symmetric northern one and change the sign of D1),

2)        altitudes 80-150 km remain unchanged,

3)        altitudes above 150 km: multiply haze opacity within this layer by the opacity

factor for B2 (cf. Table 5) where B2 is calculated by Eq. 3 from M2, D2, and E2, similar to

low altitudes.
```

<!-- PDF_PAGE: 48 -->

## PDF page 48

```text
                                               47


Fig. 1: The original data of the 10 exposures of 2002, from top to bottom.   Nine exposure were

on Titan, from limb to limb, and the last exposure was a lamp flatfield. The top and bottom

spectra probe the SW and NE limbs, respectively. The spatial direction (down-up) corresponds to

NW-SE on Titan. The spectral direction goes from left to right. Two solar lines and three main

methane bands are indicated.
```

<!-- PDF_PAGE: 49 -->

## PDF page 49

```text
                                               48


Fig. 2: The geometry for all slit positions on Titan’s disk, shown for each observing date. North

on Titan is oriented upwards in each image. The center of the disk (sub-Earth point) is marked

by a cross, the sub-solar point by a circle. The nine shown latitude circles go from 60º South to

20º North every 10º. They are the boundaries between the 10 latitude regions of this work.
```

<!-- PDF_PAGE: 50 -->

## PDF page 50

```text
                                                49


Fig. 3: Spectral calibration (top) and spatial navigation (bottom). The measurements are shown

by solid dots, averages between top and bottom data points as open circles. The first exposure of

1999 is not shown since it was positioned off the disk of Titan. Curves are adopted fits.
```

<!-- PDF_PAGE: 51 -->

## PDF page 51

```text
                                              50


Fig. 4: Comparison of absolute intensity calibration in the y passband. The solid line and

symbols come from different investigations as shown. Error bars from Lockwood and Thompson

(2009) are shown as dotted lines. The symbols indicate a smoother variation than the solid line,

but both are consistent with each other. The spacing of vertical tick marks corresponds to 0.9 %

in intensity.
```

<!-- PDF_PAGE: 52 -->

## PDF page 52

```text
                                               51


Fig. 5: The geometric albedo spectrum of Titan (bottom) and its limb darkening ratio (top), the

reflectivity ratio between μ = 0.8 and μ = 0.6. The solid spectrum is the average from this work,

the dotted spectrum from Karkoschka (1998). Vertical dotted lines separate the eight wavelength

regions with the central methane band wavelength (nm) listed on top. Symbols mark averages

within each wavelength region for the listed methane absorption coefficients in (km-am)-1 as

shown by the symbols. The dashed curve at the bottom shows the influence of Titan’s surface

albedo on its geometric albedo if Titan’s surface had the reflectivity of its brightest versus its

darkest spot.
```

<!-- PDF_PAGE: 53 -->

## PDF page 53

```text
                                                52


Fig. 6: Titan images for our 25 spectral bands, combinations of central wavelength and methane

absorption coefficient listed on top, and five observing sessions listed on the left side. To better

display small variations, an average intensity distribution shown in the top row was subtracted

before the difference was displayed with quadruple contrast. The coverage of Titan’s disk in

1997 and 2004 is partial (cf. Fig.2). The column at right shows Titan surface albedo features

measured by Cassini ISS. These images were created from a longitude-latitude map at higher

spatial resolution released 2011-10-26 by NASA/JPL-Caltech/Space Science Institute.             The

bottom four rows show contributions from the first two principal components to Titan’s disk

reflectivity in 1999 and 2002.
```

<!-- PDF_PAGE: 54 -->

## PDF page 54

```text
                                                53


Fig. 7:   Amplitudes of the principal component amplitudes A1 (top), A2 (center), and A3

(bottom), ordered first according to methane absorption coefficient (top of each panel in (km-

am)-1), and then according to wavelength (bottom scale). Solid dots and curves refer to µ = 0.8,

while open circles and dotted curves refer to µ = 0.6.
```

<!-- PDF_PAGE: 55 -->

## PDF page 55

```text
                                             54


Fig. 8: Models of increased haze opacity (curves) below 80 km altitude by 34 % (top panel) and

above 150 km by 25 % (bottom panel) compared to observed amplitudes for the first two

principal components (symbols). Solid dots and curves are for µ = 0.8, open circles and dotted

curves for µ = 0.6. Spectral bands are identified by the central wavelength (top) and methane

absorption coefficient (bottom scale).
```

<!-- PDF_PAGE: 56 -->

## PDF page 56

```text
                                                  55


Fig. 9:   Principal component values B1 (top), B2 (middle) and B3 (bottom), ordered first

according to latitude (top of each panel), and then according to observing date (bottom scale with

tick marks every year). Dots sizes indicate the number of original observations with a factor of

about 25 between the largest and smallest dots, corresponding to five-fold larger uncertainties for

the smallest dots compared to the largest ones.
```

<!-- PDF_PAGE: 57 -->

## PDF page 57

```text
                                               56


Fig. 10: Principal component values B1 (top) and B2 (bottom), ordered first according to

observing date (top of each panel), and then according to latitude (bottom scale). Curves are for

the seasonal model.
```

<!-- PDF_PAGE: 58 -->

## PDF page 58

```text
                                               57


Fig. 11: Models (curves) and observations (dots) as function of methane absorption coefficient

for the first (A1, top panels) and second principal component amplitudes (A2, bottom panels)

within the 890 nm spectral band. The altitude regions of increased haze opacity are listed in each

panel. Solid curves and dots are for the geometry µ = 0.8, dashed curves and open circles are for

the geometry µ = 0.6.
```

<!-- PDF_PAGE: 59 -->

## PDF page 59

```text
                                               58


Fig. 12: Models of backward scattering phase functions below 80 km altitude (top panel) and

above 150 km (bottom panel) compared to observed amplitudes of the first two principal

components. Symbols are the same as in Figs. 7, 8, and 11. Spectral bands are identified by the

central wavelength (top) and methane absorption coefficient (bottom scale).
```

<!-- PDF_PAGE: 60 -->

## PDF page 60

```text
                                               59


Fig. 13: Models of increased single scattering albedos below 80 km altitude (top panel) and

above 150 km (bottom panel) compared to observed amplitudes of the first two principal

components. Symbols are the same as in Figs. 7, 8, and 11. Spectral bands are identified by the

central wavelength (top) and methane absorption coefficient (bottom scale).
```

<!-- PDF_PAGE: 61 -->

## PDF page 61

```text
                                            60


Fig. 14: The three parameters of the seasonal model for the first (solid dot and curves) and

second principal component (open squares and dashed curves), each as function of latitude

according to Eq. 3. Southern latitudes are assumed symmetric to northern ones, except for a

change of sign for D1 and D2.
```

<!-- PDF_PAGE: 62 -->

## PDF page 62

```text
                                                61


Fig. 15: The variation of values B1 (top) and B2 (middle) as function of time over a whole Titan

year for 14 latitudes from -65° to 65° latitude, every 10°. The six latitudes within 30° of the

Equator are shown as solid curves, the ones further north and south as dotted and dashed,

respectively. The bottom panel shows the sub-solar latitude. The five dots indicate the dates for

our observations, the horizontal bar the planned duration of the Cassini mission at Saturn, and the

vertical dotted lines the equinoxes and solstices. The scales at the right show approximate haze

opacity factors for altitudes below 80 km (top panel) and above 150 km (middle panel).
```

<!-- PDF_PAGE: 63 -->

## PDF page 63

```text
                                               62




> HST-STIS image cubes in five years provide a unique probe of atmospheric variations.
> The first principal component describes haze opacity variations below 80 km altitude.
> The second one has similar variations above 150 km but its phase is 1-2 years ahead.
> The tropics have higher (>150 km) and lower (<80km) opacities than the global mean.
> The north-south asymmetry may reverse in 2016 (>150 km) and 2017-2018 (<80 km).
```
