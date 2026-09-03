---
citation_key: "clark2018vims"
title: "The VIMS Wavelength and Radiometric Calibration 19, Final Report"
source_pdf: "data/papers/clark2018vims.pdf"
source_pdf_sha256: "19883c0b24bb0bf711d012dc0afa9652716cf24b0aa865cd0b6cec51cb0e7e22"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
        The VIMS Wavelength and Radiometric Calibration 19, Final Report

                Roger N. Clark1, Robert H. Brown2, Dyer M. Lytle2, Matt Hedman3
                        1
                          Planetary Science Institute, Lakewood, CO, 80227
           2
             Department of Planetary Sciences, University of Arizona, Tucson, AZ, 85721
                                3
                                  University of Idaho, Moscow, Idaho


                                  NASA Planetary Data System
                                         Version 2.0
                                         01/25/2018




Clark, R. N., R. H. Brown, D. M. Lytle, Hedman, M., 2018, The VIMS Wavelength and Radiometric
Calibration 19, Final Report, NASA Planetary Data System, The Planetary Atmospheres Node, 30p.
http://atmos.nmsu.edu/data_and_services/atmospheres_data/Cassini/vims.html
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
Abstract

We derive the final radiometric calibration of the mission for the Visual and Infrared Mapping
Spectrometer (VIMS) on Cassini. The VIMS instrument has undergone shifts in the wavelength
calibration of the spectrometer after launch, with large shifts during the Jupiter fly-by in 2000, a period
of stability until Saturn orbit insertion, then small shifts throughout the orbital tour until the end of
mission in September, 2017. We adopt a “zero point” shift as the quite period from 2002.9 to 2005.0,
which we call the standard wavelengths. The maximum shift during the Saturn orbital tour was 9.8 nm
(0.59 channels). Before the Jupiter encounter, the wavelengths relative to the standard were at -25.8
nm (-1.55 channels) with large shifts through the Jupiter flyby. The wavelength shifts require a time-
dependent radiometric calibration technique to be deployed to preserve radiometric accuracy. Herein
we quantify the time-dependent wavelength shift, and describe a compensatory scheme that provides an
accurate calibration for both specific intensity and I/F for VIMS measurements made during the Cassini
Mission. We also discuss unresolved issues.
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
Introduction

The radiometric responsivity of a spectrometer or imaging instrument is complex, depending upon
many factors, some of the most important being the aperture collection area, spectral transmission of
the optical system, and the response function of the detector(s). The VIMS is an imaging spectrometer
(Miller et al., 1996, Brown et al., 2004) whose spectral responsivity varies with angular position in the
field of view and with wavelength. The IR channel primary optical design incorporates Ritchey Cretien
foreoptics, a reverse-Dahl-Kirkham collimator with a large central obscuration, a triply blazed
reflection grating, Cassegrain camera optics and a linear, 256-element array of InSb detectors overlain
with 4 order-sorting/thermal-blocking filter segments. The VIS channel uses an off-axis Shafer
telescope with an Offner design spectrometer. The flat-field response of the instrument varies with
wavelength, and a flat-field image cube with 352 wavelengths is thus employed to correct the system
response over the field of view relative to the instrument’s boresight pixel. The spectral bandpass of the
instrument also varies slightly across the full range of VIMS’ spectral response, adding to the
complexity of the calibration. Additional issues arise due to ringing in the instrument response function
driven primarily by the blocking/order-sorting filters on the VIMS focal plane (in some cases ±10% or
more [Figure 1]). Because of the ongoing wavelength shifts in the instrument, relative to when the
instrument response function was measured on the ground and after the Cassini Jupiter flyby in late
2000, artifacts in radiometrically calibrated spectra result because the response function at any later
time is actually sampled at different wavelengths than the wavelength set used in the ground
measurements. The resulting artifacts are primarily caused by the use of a response function in
multiplicative or divisive operations sampled using the original, ground wavelength set. When used to
calibrate to specific intensity or I/F instrument data obtained at a later time would be done with a
different set of wavelengths than derived from the ground calibration. Below we describe our
methodology to track the shifting wavelengths, and then apply a time-dependent, radiometric
calibration that compensates for all the known effects.
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
Figure 1. VIMS order-sorting/blocking filter transmission for the 1.6-3.0 micron segment. The
ringing in the transmission propagates into the radiometric calibration in a complex way because a shift
in wavelength samples different places in the response function. (From Al-Jumaily, 1991).
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
VIMS Wavelengths Versus Time

The VIMS calibration pipeline and PDS-delivered data used a wavelength set and FWHM set measured
during thermal vacuum testing at JPL before launch (Brown et al., 2004), but shifted by 1.3 channels as
observed in Galilean Satellite data obtained during the Jupiter flyby (McCord et al., 2001). The
calibration also incorporated a small, additional set of corrections to a few channels near the
wavelength of the 4.25 micron CO2 absorption (Cruikshank et al., 2010). Here we employ terminology
to describe the various corrections to the VIMS Radiometric Calibrations by using a numbering scheme
RCnn (Radiometric Calibration nn). RC17 was the primary calibration used until June, 2016 (Clark et
al., 2012), and has been used in all data processed through the VIMS calibration pipeline, and all
calibration files delivered to the PDS prior to the calibration RC19 described herein. Calibration RC19
was put in place in the VIMS pipeline in June 2016 with projections to the end of mission (Clark et al.,
2016). In this paper we use data through the end of mission and include for the first time analysis of
the wavelength shifts back to before the Jupiter encounter.

The wavelength shift of the instrument since Cassini’s Saturn Orbit Insertion (SOI) is the is the reason
why we employ a time-dependent radiometric calibration. It has been intensively studied since about
2013. Two methods for determining the VIMS wavelengths have been employed. (1) Monitoring, as a
function of time, the reflectivity in 3 windows of the Titan spectrum (1.6, 2.0, 2.8 microns) by fitting
Gaussian profiles to the window peaks. (2) Using the VIMS internal, wavelength-calibration, laser
diode (central wavelength ~0.979 microns), and employing a fit to the Gaussian intensity profile of the
calibration diode convolved with the response of 8 VIMS channels centered near peak intensity of the
calibration diode. The result is the derivation of a time-dependent shift of the current wavelengths
relative to the set extant at the time of Cassini’s SOI. Figures 2a, 2b, and 2c show the results from both
methods.

To determine the absolute wavelength calibration of the VIMS infrared instrument before Saturn orbit
insertion back to the Jupiter flyby, a combination of calibration-diode measurements, and spectra of
Jupiter before, during and after the Jupiter flyby were used. Spectra of Jupiter were used to determine
relative wavelength shifts during the cruise phase that encompassed the Jupiter flyby. Those relative
shifts were subsequently put on an absolute scale by using spectra of Jupiter obtained in 2016, day 262
while in Saturn orbit, observation VIMS_242JU_FULLDISK001. The absolute wavelength position of
the averaged VIMS spectra of Jupiter from the 2016 observations, calibrated with contemporary results
from the VIMS calibration diode, were then compared to the relative wavelength shifts derived from
spectra from the Jupiter flyby using a least-squares technique to shift the spectra to best match the
absorption bands in Jupiter’s spectrum. Results are illustrated in Figure 3.
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
Figure 2a. VIMS wavelength shift versus time for three methods: Titan atmospheric windows (red
points), Jupiter atmospheric windows (green points) and VIMS calibration diode (dark blue points).
The lone green point near 2017 is the Jupiter reference spectrum, measured while in Saturn orbit, and
was used to fit Jupiter VIMS data to determine absolute wavelengths during the Jupiter flyby.
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
Figure 2b. VIMS wavelength shift versus time. The large deviations toward shorter wavelengths in
the calibration diode measurements are due to transient thermal fluctuations in the VIMS IR
spectrometer optics driven by changes in spacecraft orientation during particular observations in the
orbital tour. Similar fluctuations show in the shifts derived using Titan data, but they are of much
smaller magnitude because the Titan data integrate over larger time spans where the VIMS
spectrometer temperature is closer to nominal. It is also the case that spacecraft orientation during most
Titan flybys where the optical remote sensing instruments were prime was less prone to heating the
VIMS spectrometer optics. These data are the same data as in Figure 2a, showing the shifts during the
Saturn orbital tour.
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
Figure 2c. VIMS wavelength shift as measured by the calibration diode. The central, solid line is a
linear regression, and the two dotted lines are ± 1 sigma as measured relative to the linear fit. The
outlier points indicate real short term variations, most likely caused by temperature cycling of the
VIMS spectrometer optics.




Figure 3. Illustration of alignment agreement of the Jupiter fly-by data after applying the shifts in
Figure 2a. The horizontal axis is VIMS channel number.
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
We specify the shift as a function of time in Table 1, and generate a new wavelength set with the
associated calibration files when the change in wavelength relative to the previous period would be 0.5
nm (about 1/30th of one channel). The shifts are relative to the new “standard 2004.0” wavelengths
given in Table 2.


Calibration Equations

The basic equation we employ for calibrated, specific intensity, I, at one wavelength is:



   (1)


where RDN is the instrument raw data number in a given exposure, Dark is the instrument measured
dark current and thermal background (a dark/thermal background measurement is made after every
VIMS scan line using the same exposure time with the shutter closed), and flat is the VIMS flat-field
response at the given wavelength and position within the field of view normalized to the response of
the boresight pixel (i.e., the relative response of the boresight pixel in the flat-field cube is 1 at every
wavelength). The units of equation 1 are: energy/time/area/bandpass/solid angle.

C is defined as the calibration-multiplier vector which includes system transmission, grating efficiency,
detector response, and divided by the exposure time (see equations 4a and 4b, below).

   (2)                 ,

where
λ = wavelength,
Δλ = full width, half maximum, FWHM,
Ω = solid angle = 2.5x10-7 steradian,
A = aperture area IR: A = 96.1 cm2, = 0.00961 m2 , VIS: A = 15.88 cm2, A = 0.001588 m2,
c = speed of light = 2.998x1010 cm/s = 2.998x108 m/s, and
h = Planck constant = 6.626x10-27 erg-s = 6.626x10-34 J-s (= W s2).

The calibration to apparent reflectance, I/F, referred to that of a Lambert disk (a perfect diffuse
reflector) is (note that the factor of π comes from the result that I/F for a Lambert disk is π-1):



   (3)                             ,
where t is the exposure time – 0.004 second for the IR, or just the exposure time in seconds for the VIS
channel, and S is the solar spectrum from Thompson et al. (2015), corrected to the flux extant at the
heliocentric distance of the object in question. The 4 millisecond IR exposure time correction is due to
settling time after each mirror movement of the scanning secondary and was derived by McCord et al.,
(2004).

In the VIMS instrument, A and Ω are more complex (see Brown et al., 2005). In standard resolution
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
mode, the VIS channel bins 3x3 pixels to make a 0.5 x 0.5 mradian IFOV, and the IR IFOV is actually
0.5 x 0.25 mradian which gets moved 0.25 mradian half way through the integration to make a 0.5 x
0.5 mradian square pixel. In high resolution mode, the VIS channel is 0.17 x 0.17 and the IR is 0.5 x
0.25 mradian. The full telescope aperture, a 22.9 cm diameter Ritchey Cretien telescope, 800-mm focal
length, has obscuration in both the Ritchey-Cretien secondary, and in the inverse Dahl-Kirkham
collimator and Cassegrain spectrometer optics. We use different constants to scale the relative
multiplier (equation 4, below) to compensate for these factors to derive the value of C for each VIS and
IR channel and each wavelength.


Derived Spectral Properties

The wavelength calibration and FWHM for VIMS were measured on the ground in a thermal vacuum
chamber before launch (Brown et al., 2005). At the Jupiter fly-by it was determined that the IR
wavelengths had shifted 1.3 channels from the ground calibration (McCord et al., 2001). That 1.3-
channel shift from the ground calibration defined the VIMS standard wavelengths for post launch to
RC17. We now know that the 1.3-channel shift was more complex (Figure 2a).

Using the measured wavelength shifts from Table 1 we modeled the effect of the shift on solar port data
from 2005 to 2012 (Figure 4). The model does not include order-sorting/blocking-filter response.
Unfortunately, as the data in Figure 1 show, the analog plot of filter transmission made over 20 years
ago is not precise enough for tracking the effects of the wavelength shifts.
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
Figure 4. Solar port data from 2005 divided by data from 2012 (blue line) shows larger than predicted
changes than just due to solar spectral structure.

Investigation of the causes of the discrepancy shown in Figure 4 revealed that, besides ringing in the
order sorting filter transmission, fine structure in the original measured wavelengths of each vims
channel and the derived FWHM display periodic structure which we traced to a periodic error in the
grating position of the test monochromater used in the ground-based, thermal-vacuum testing of VIMS.
Figures 5 and 6 show the trends.
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
Figure 5. VIMS IR wavelengths divided by a linear fit to the end channels of the IR spectrometer (blue
line). A cubic-spline fit (black line) is our new derivation: the new RC19 2004.0 “standard” wavelength
set (values in Table 2). The blue line is a plot of the wavelengths derived in thermal vacuum testing
before launch and shifted 1.3 channels (derived that the Jupiter fly-by pre RC19) with a correction for
the wavelengths around the 4.25 micron CO2 atmospheric absorption (Cruikshank et al. 2010).
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
Figure 6. VIMS IR-channel bandpasses (FWHM) in microns (blue line) derived during thermal
vacuum testing before launch. A cubic-spline fit (black line) defines the new RC19 FWHM set (values
in Table 2).


Calibration Multiplier

The calibration multiplier vector, C, from equation 1 is the major factor that remains to be determined.
Ideally it could be derived from measurements of a known, standard reflector illuminated with a source
of known intensity, such as a calibration target on the spacecraft illuminated by the Sun, but such a
device was descoped from VIMS in early Cassini budget cuts, thus we are left with only more
approximate methods. Standard stars (that is, stars whose spectral specific intensities are known with
the required precision) could be used, but because they are sub-pixel to VIMS, and because the VIMS
pixel response function is variable within a pixel, unresolved objects like stars cannot be used with
sufficient accuracy. Therefore, the only practical avenue that remains involves the use of known spectra
of Saturn’s icy satellites and rings, Saturn itself, Jupiter and Titan. Using a cross correlation of the
spectra of the various objects, we were able to ascertain in general which “bumps and wiggles” in the
response function were instrument related, and which were real features in the spectra of the various
objects.

Furthermore, because the reflection spectra of solar-systems objects can be strong functions of their
illumination geometry, the spectra of some objects were analyzed as a function of phase angle. For
example, high-phase observations of Saturn's D and F-rings are expected to be dominated by
diffraction, with muted spectral signatures of water ice, except near the water-ice vibrational
fundamentals, where the real index of refraction becomes close to 1.0. The derived, relative-multiplier
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
vector for 2004.0 is shown in Figure 7.

VIMS has order-sorting filter gaps near 1.65, 3, and 3.9 microns (the upward spike in Figure 7), and
they do not shift with time because they are fixed with respect to the detector array, whereas the
spectrum of an object projected onto the detector array changes with time, and thus all other structure
in spectral measurements does change in the IR channel of VIMS. The sampled wavelengths of the
visible channel do not shift, thus its ground calibration is sufficient for calibration of the visible-
channel data. The VIMS-IR multiplier vector (excluding the filter gap transmission) from Figure 7 was
cubic-spline interpolated to the wavelength set derived for each year and given in Table 1. We then
applied the effects of the filter-gap transmission.




Figure 7. The relative calibration multiplier set for year 2004.0, used in Equation 4. The vector is
resampled with time as described in the text.

The results of the new calibration are illustrated in Figure 8. The RC17 calibration, which does not take
into account the effects of shifting wavelengths shows increasing ringing (compare The D-ring 2006
RC 17 with F-ring 2015 RC17 spectra). The new RC19 calibration, which tracks the effects of the
shifting wavelength sets, eliminates the ringing problems.
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
Figure 8. Example VIMS spectra using RC17 which was constant with time, and the new RC19
method described here.
```

<!-- PDF_PAGE: 16 -->

## PDF page 16

```text
Application/implementation of the time-dependent transfer functions to calibration of data

Calibration of VIMS uses equation 3, above. The value of B from equation 2 is computed by selecting
the wavelengths and FWHM for the time of the observation, and MKS units as follows.
Ω = 2.5x10-7 steradian.
c = 2.998x108 m/s.
A = 0.00961 m2 (VIS), 0.001588 m2 (IR).
h = 6.626x10-34 J-s (= W s2).
λ = wavelength in meters (ascii listings are in microns. So divide by 1000.0).
Δλ = FWHM in microns.

ASCII listing of the value of B for each VIMS channel is given in the Appendix.

In equation 3,

  (4a) CVIS = 29554. * calibration multiplier set from the appendix / (t * gain),
  (4b) CIR = 8112. * calibration multiplier set from the appendix / (t * gain),


the calibration multiplier set is one of the files with a name of
   vims.calibration.multiplier.RC19-20NN.N.txt
where 20NN.N is the year. The constants in 4a and 4b apply to both standard and high resolution
modes. Almost all VIMS observations have been carried out with gain = 1.

The Thompson et al. (2015) solar spectra convolved to VIMS are also given in the Appendix for each
wavelength and calibration multiplier set.

The resulting units from equation 3 are dimensionless (I/F). To get to radiance, multiply by the solar
spectrum. The supplied VIMS-convolved solar spectra are in units of Watts / m2 / micron.

Remaining Issues

No further analysis will be carried out by the VIMS Team after after this report. There are differences
in the calibration that remain unresolved between RC19 and the earlier RC17. Remaining issues and
uncertainties include the following.


1) The dip and rise near 2.6-2.9 microns may or may not be real (Figures 9, 10). The structure in this
spectral region for the D-ring high phase observation is compared to spectra of Titan. The difference in
the RC17 versus RC19 changes the intensities in the Titan windows in the 2.7 and 2.8 micron regions
by about 5%. Which version is correct is not certain, though models suggest the RC17 is more correct
at these wavelengths. However, the RC19 calibration may be showing a structure from another
compound.

2) The feature near 3 microns is on the edge of the order sorting filter gap and may or may not be real.
If real the feature is most likely due to an N-H stretch (e.g. ammonia or ammonium compound). Other
space-based spectra through this spectral region are needed to resolve the issue.
```

<!-- PDF_PAGE: 17 -->

## PDF page 17

```text
3) The filter gap transmission around 1.65 and 3.9 microns may need adjustment.

4) The dip seen in Figure 9, top few spectra, between 4.5 and 5.1 microns may not be real. Or perhaps
there is a rise between ~4.2 and ~4.7 microns that may not be real. This is less than a 5% effect.




Figure 9. VIMS spectra of a region in the middle A ring derived from HIPHASE observations in Revs
263, 264 and 271. Each curve gives the average spectrum from all pixels within 25 cubes which
observed the rings within a range of 0.1 degrees of phase angle and 100 km of 135,400 km. Note the
decrease in H2O ice band depths with increasing phase angles. The feature at 2.85 microns may be real
and due to the minimum in the index of refraction in ice at that wavelength. The spectral structure short
of that minimum is in question. The spectral structure in the visible around 0.5 microns shows
similarity to the index of refraction of metallic iron.
```

<!-- PDF_PAGE: 18 -->

## PDF page 18

```text
Figure 10. VIMS spectra of Titan and Dring at high phase are compared. Differences in the RC19
versus RC17 calibration changes the ratio of the intensities in the Titan 2.7 to 2.8-micron windows by
about 5%. This ratio is sensitive to Titan’s surface composition. The spectral structure from about 2.62
to 2.82 microns is uncertain at about the 5% level. The 2.95 to 3.0-micron spectral structure is located
at a VIMS order-sorting filter change so has some uncertainty. If real it could indicate the presence of
NH3. The 2.85-micron spectral feature in the D-ring spectra is real and is due to the minimum in the
H2O ice index of refraction.


Appendices

A tar file, vims-calibration+wavelengths-vs-time-files-rc19-2018.tar, contains ascii listings of the
wavelengths, calibration multiplier, the computed B values (equation 2), and the VIMS-convolved
solar spectrum as a function of time. The time increment is 1 year centered mid year. The VIMS
wavelength shift is small enough that this granularity is sufficient for most applications, as the shift
between these calibrations is less than 0.05 channel.

Conclusions

The VIMS wavelengths are shifting small amounts per year (less than 0.1 channel) during the Saturn
orbital tour and larger amounts during the Jupiter fly-by, necessitating the need for a time-dependent
calibration. We have derived a new methodology that tracks this change and produces significantly
better spectra with lower noise and lower artifacts. This new calibration should enable greater
```

<!-- PDF_PAGE: 19 -->

## PDF page 19

```text
confidence in observed spectral features, leading to detection of lower abundance components and
therefore new and better science.
```

<!-- PDF_PAGE: 20 -->

## PDF page 20

```text
                        Table 1
                   Add to               Change in
Time       Shift   standard              interval
           (nm)    Wavelength
                   set:
                   (microns)              - to +
                                          (nm)
1999.6     -25.8 -0.0258                     0.0
2000.064   -25.8 -0.0258          0.0        0.5
2000.350   -24.8 -0.0248          0.5        0.5
2000.630   -23.8 -0.0238          0.5        0.5
2000.695   -22.8 -0.0228          0.5        0.5
2000.743   -21.8 -0.0218          0.5        0.5
2000.776   -20.8 -0.0208          0.5        0.5
2000.805   -19.8 -0.0198          0.5        0.5
2000.824   -18.8 -0.0188          0.5        0.5
2000.842   -17.8 -0.0178          0.5        0.5
2000.856   -16.8 -0.0168          0.5        0.5
2000.868   -15.8 -0.0158          0.5        0.5
2000.878   -14.8 -0.0148          0.5        0.5
2000.884   -13.8 -0.0138          0.5        0.5
2000.894   -12.8 -0.0128          0.5        0.5
2000.903   -11.8 -0.0118          0.5        0.5
2000.911   -10.8 -0.0108          0.5        0.5
2000.918   -9.8    -0.0098        0.5        0.5
2000.925   -8.8    -0.0088        0.5        0.5
2000.931   -7.8    -0.0078        0.5        0.5
2000.936   -6.8    -0.0068        0.5        0.5
2000.942   -5.8    -0.0058        0.5        0.5
2000.947   -4.8    -0.0048        0.5        0.5
2000.952   -3.8    -0.0038        0.5        0.5
2000.956   -2.8    -0.0028        0.5        0.5
2000.961   -1.8    -0.0018        0.5        0.5
2001.12    -0.8    -0.0008        0.5        0.5
2001.30    0.2     0.0002         0.3        0.5
2001.50    1.2     0.0012         0.5        0.6
```

<!-- PDF_PAGE: 21 -->

## PDF page 21

```text
2002.0-2005.0 0.0   0.0000   0.6   0.2   "Standard wavelengths"
2005.5        0.4   0.0004   0.2   0.4
2006.0        1.1   0.0011   0.4   0.4
2006.5        1.9   0.0019   0.4   0.4
2007.0        2.6   0.0026   0.4   0.4
2007.5        3.4   0.0034   0.4   0.4
2008.0        4.1   0.0041   0.4   0.4
2008.5        4.9   0.0049   0.4   0.4
2009.0        5.6   0.0056   0.4   0.2
2009.5        6.0   0.0060   0.2   0.0
2010.0        6.0   0.0060   0.0   0.0
2010.5        6.0   0.0060   0.0   0.0
2011.0        6.0   0.0060   0.0   0.0
2011.5        7.0   0.0070   0.5   0.2
2012.0        7.3   0.0073   0.2   0.2
2012.5        7.7   0.0077   0.2   0.2
2013.0        8.0   0.0080   0.2   0.2
2013.5        8.4   0.0084   0.2   0.2
2014.0        8.7   0.0087   0.2   0.2
2014.3        9.1   0.0091   0.2   0.0   change from 2016 RC19 in 4/20/2017 starts
                                         here
2014.5        9.1   0.0091   0.0   0.1
2015.0        9.2   0.0092   0.1   0.1
2015.5        9.3   0.0093   0.1   0.1
2016.0        9.4   0.0094   0.1   0.1
2016.5        9.5   0.0095   0.1   0.1
2017.0        9.6   0.0096   0.1   0.1
2017.5        9.7   0.0097   0.1   0.1
2017.8        9.8   0.0098   0.1
```

<!-- PDF_PAGE: 22 -->

## PDF page 22

```text
Table 2
VIMS Standard Wavelengths (2004.0)

       Standard
       wavelength     FWHM
       (microns)     (microns)

  1       0.350540     0.007368
  2       0.358950     0.007368
  3       0.366290     0.007368
  4       0.373220     0.007368
  5       0.379490     0.007368
  6       0.387900     0.007368
  7       0.395180     0.007368
  8       0.402520     0.007368
  9       0.409550     0.007368
 10       0.417310     0.007368
 11       0.424360     0.007368
 12       0.431840     0.007368
 13       0.439190     0.007368
 14       0.446520     0.007368
 15       0.453720     0.007368
 16       0.461630     0.007368
 17       0.468410     0.007368
 18       0.476220     0.007368
 19       0.486290     0.007368
 20       0.489670     0.007368
 21       0.497770     0.007368
 22       0.506280     0.007368
 23       0.512220     0.007368
 24       0.519630     0.007368
 25       0.527660     0.007368
 26       0.534160     0.007368
 27       0.541560     0.007368
 28       0.549540     0.007368
 29       0.556140     0.007368
 30       0.563530     0.007368
 31       0.571310     0.007368
 32       0.578100     0.007368
 33       0.585480     0.007368
 34       0.593120     0.007368
 35       0.599380     0.007368
 36       0.607570     0.007368
 37       0.615050     0.007368
 38       0.622070     0.007368
 39       0.629400     0.007368
 40       0.637040     0.007368
 41       0.644080     0.007368
 42       0.651420     0.007368
```

<!-- PDF_PAGE: 23 -->

## PDF page 23

```text
43   0.659100   0.007368
44   0.666090   0.007368
45   0.673420   0.007368
46   0.681020   0.007368
47   0.688030   0.007368
48   0.695350   0.007368
49   0.702880   0.007368
50   0.710000   0.007368
51   0.717330   0.007368
52   0.724840   0.007368
53   0.731980   0.007368
54   0.739300   0.007368
55   0.746760   0.007368
56   0.753960   0.007368
57   0.761280   0.007368
58   0.768740   0.007368
59   0.775950   0.007368
60   0.783280   0.007368
61   0.790720   0.007368
62   0.797930   0.007368
63   0.805220   0.007368
64   0.812620   0.007368
65   0.819890   0.007368
66   0.827210   0.007368
67   0.834630   0.007368
68   0.841900   0.007368
69   0.849220   0.007368
70   0.856630   0.007368
71   0.863910   0.007368
72   0.871220   0.007368
73   0.878630   0.007368
74   0.885890   0.007368
75   0.893860   0.007368
76   0.900320   0.007368
77   0.907870   0.007368
78   0.915180   0.007368
79   0.922540   0.007368
80   0.929830   0.007368
81   0.937130   0.007368
82   0.944450   0.007368
83   0.951770   0.007368
84   0.959070   0.007368
85   0.966380   0.007368
86   0.973820   0.007368
87   0.981000   0.007368
88   0.988830   0.007368
89   0.995880   0.007368
90   1.002950   0.007368
91   1.010050   0.007368
```

<!-- PDF_PAGE: 24 -->

## PDF page 24

```text
 92   1.016950   0.007368
 93   1.024710   0.007368
 94   1.031950   0.007368
 95   1.038650   0.007368
 96   1.045980   0.012480 End of VIS channel
 97   0.884210   0.012878
 98   0.900753   0.012767
 99   0.916924   0.012507
100   0.933078   0.013169
101   0.949803   0.012869
102   0.965683   0.012728
103   0.982262   0.013370
104   0.998820   0.012790
105   1.014790   0.012748
106   1.031320   0.013186
107   1.047550   0.012847
108   1.065410   0.013136
109   1.081830   0.013063
110   1.098060   0.012686
111   1.113960   0.012828
112   1.130240   0.013111
113   1.146950   0.013322
114   1.163700   0.013266
115   1.179960   0.012968
116   1.196220   0.013018
117   1.212460   0.012921
118   1.228590   0.013031
119   1.244920   0.013401
120   1.261660   0.013631
121   1.278130   0.013372
122   1.294820   0.013121
123   1.310910   0.013101
124   1.326950   0.013146
125   1.343240   0.013389
126   1.359520   0.013663
127   1.376950   0.013366
128   1.393260   0.012821
129   1.409400   0.013147
130   1.425570   0.013137
131   1.441840   0.013216
132   1.458410   0.013480
133   1.475140   0.013610
134   1.491690   0.013066
135   1.507940   0.013063
136   1.524210   0.012992
137   1.540350   0.013059
138   1.556740   0.013388
139   1.573610   0.014011
140   1.590180   0.013901
```

<!-- PDF_PAGE: 25 -->

## PDF page 25

```text
141   1.602280   0.008457 order-sorting filter change
142   1.625230   0.032000 order-sorting filter change
143   1.641600   0.009862 order-sorting filter change
144   1.655670   0.013304
145   1.672380   0.013532
146   1.689010   0.013253
147   1.705360   0.013300
148   1.721750   0.013068
149   1.738020   0.013084
150   1.754360   0.013155
151   1.771050   0.013455
152   1.787710   0.013080
153   1.804010   0.013090
154   1.820040   0.012902
155   1.836160   0.012985
156   1.852880   0.013531
157   1.869330   0.012939
158   1.886790   0.012600
159   1.902610   0.013058
160   1.919160   0.013059
161   1.935450   0.013127
162   1.951910   0.013498
163   1.968710   0.013615
164   1.985310   0.013293
165   2.001670   0.013209
166   2.017810   0.013294
167   2.034240   0.013415
168   2.050910   0.013889
169   2.067570   0.013472
170   2.084000   0.013579
171   2.100340   0.013428
172   2.116670   0.013719
173   2.133370   0.013943
174   2.150180   0.013787
175   2.166520   0.013547
176   2.182880   0.013600
177   2.199200   0.013571
178   2.215910   0.014009
179   2.232820   0.013918
180   2.249520   0.013700
181   2.266220   0.013600
182   2.282380   0.014012
183   2.299210   0.013974
184   2.316120   0.014211
185   2.333250   0.014287
186   2.350430   0.014407
187   2.367650   0.014286
188   2.384720   0.014294
189   2.401560   0.014079
```

<!-- PDF_PAGE: 26 -->

## PDF page 26

```text
190   2.418200   0.013921
191   2.434710   0.013829
192   2.450970   0.013748
193   2.467230   0.013784
194   2.483600   0.014044
195   2.500020   0.014293
196   2.516590   0.014306
197   2.532920   0.013704
198   2.549160   0.013918
199   2.564370   0.011963
200   2.581760   0.013610
201   2.598070   0.014726
202   2.615080   0.012722
203   2.630000   0.011283
204   2.646500   0.013711
205   2.661460   0.012674
206   2.680850   0.016119
207   2.696200   0.014697
208   2.712050   0.015964
209   2.732700   0.012786
210   2.747700   0.018701
211   2.763050   0.016296
212   2.781180   0.013689
213   2.798890   0.014400
214   2.816060   0.015083
215   2.832470   0.014680
216   2.849540   0.014842
217   2.866090   0.015667
218   2.882420   0.015534
219   2.898780   0.015325
220   2.915400   0.015088
221   2.931430   0.015720
222   2.947260   0.015350
223   2.963270   0.015716 order-sorting filter change
224   2.977200   0.015512 order-sorting filter change (~30% transmission)
225   3.000720   0.012919 order-sorting filter change
226   3.013820   0.015570
227   3.029700   0.015398
228   3.048060   0.014922
229   3.064460   0.015466
230   3.080360   0.015882
231   3.096890   0.015851
232   3.112130   0.015753
233   3.129620   0.016580
234   3.146670   0.015851
235   3.163040   0.016127
236   3.179740   0.016115
237   3.197080   0.015685
238   3.213640   0.015830
```

<!-- PDF_PAGE: 27 -->

## PDF page 27

```text
239   3.231500   0.016740
240   3.248060   0.017771
241   3.265610   0.016161
242   3.282980   0.016285
243   3.299460   0.016286
244   3.316190   0.015816
245   3.333380   0.015203
246   3.349810   0.016500
247   3.365640   0.015590
248   3.381830   0.015717
249   3.398720   0.016471
250   3.415460   0.016457
251   3.431780   0.016343
252   3.448740   0.015852
253   3.464750   0.015634
254   3.481370   0.015608
255   3.497950   0.015779
256   3.512840   0.016141
257   3.530150   0.015057
258   3.546640   0.016643
259   3.562740   0.016735
260   3.580340   0.016474
261   3.596100   0.017033
262   3.613870   0.020159
263   3.630850   0.018293
264   3.648530   0.017622
265   3.665220   0.018895
266   3.682830   0.018505
267   3.699530   0.019496
268   3.717430   0.018635
269   3.734390   0.019045
270   3.751030   0.019296
271   3.767630   0.017966
272   3.784440   0.019006
273   3.800830   0.018599
274   3.817420   0.018210
275   3.834720   0.019856
276   3.851410   0.018125
277   3.861840   0.015574 order-sorting filter change
278   3.881670   0.023959 order-sorting filter change
279   3.898590   0.020270
280   3.914780   0.021217
281   3.930690   0.020631
282   3.947620   0.019721
283   3.963750   0.020799
284   3.980150   0.021142
285   3.996720   0.021846
286   4.012800   0.021142
287   4.029440   0.021531
```

<!-- PDF_PAGE: 28 -->

## PDF page 28

```text
288   4.047300   0.021598
289   4.062950   0.022566
290   4.080861   0.021479
291   4.097430   0.022433
292   4.114500   0.022013
293   4.131830   0.022290
294   4.148830   0.022294
295   4.166440   0.020424
296   4.183199   0.021180
297   4.200100   0.019057
298   4.217000   0.017383
299   4.233700   0.022866
300   4.250500   0.021600
301   4.267300   0.021600
302   4.284000   0.021600
303   4.300600   0.021600
304   4.317101   0.021600
305   4.333600   0.021600
306   4.350200   0.019916
307   4.366500   0.021335
308   4.382900   0.020260
309   4.397930   0.019563
310   4.415370   0.021034
311   4.431720   0.019802
312   4.447720   0.019867
313   4.465730   0.019735
314   4.482400   0.019931
315   4.499511   0.021189
316   4.515910   0.020529
317   4.533790   0.019303
318   4.551870   0.019921
319   4.567970   0.020376
320   4.585560   0.020113
321   4.602900   0.019105
322   4.620100   0.020267
323   4.636150   0.017409
324   4.654160   0.019612
325   4.670340   0.018281
326   4.687210   0.019814
327   4.702900   0.017902
328   4.719561   0.020080
329   4.737060   0.018831
330   4.753510   0.018017
331   4.770310   0.015955 hot pixel
332   4.786730   0.018821
333   4.803490   0.017274
334   4.819521   0.018179
335   4.835771   0.018545
336   4.852920   0.018106
```

<!-- PDF_PAGE: 29 -->

## PDF page 29

```text
337   4.869401   0.018799
338   4.885530   0.019556
339   4.902650   0.018114
340   4.919831   0.018570
341   4.936851   0.017740 slightly noisier (1.5x at low signal)
342   4.953890   0.018779
343   4.971780   0.018266
344   4.988960   0.020001
345   5.005760   0.018402
346   5.022400   0.018621
347   5.040781   0.016783
348   5.057340   0.019510
349   5.074020   0.017953
350   5.091060   0.020883 slightly noisier (2x at low signal)
351   5.106800   0.015704
352   5.122500   0.016
```

<!-- PDF_PAGE: 30 -->

## PDF page 30

```text
References

Al-Jumaily, G.A., E. Miller, and N. Raof, 1991, C/C VIMS Filter requirements, JPL VIMS Design File
Memorandum, 6-25-1991.

Brown, R.H., Baines, K.H., Bellucci, J.P., Bibring, B.J., Buratti, E., Bussoletti, F., Capaccioni, P.,
Cerroni, P., Clark, R.N., Coradini, D.P., Cruikshank, P., Drossart, Y., Formisano, R., Jaumann, Y.,
Langevin, D.J., Matson, T.R., 2005, The Cassini visual and infrared mapping spectrometer
investigation: Space Science Reviews, 115 (1-4), 111-168.

Clark, R. N., D. P. Cruikshank, R. Jaumann, R. H. Brown, K. Stephan, C. M. Dalle Ore, K. E. Livo, N.
Pearson, J. M. Curchin, T. M. Hoefen, B. J. Buratti, G. Filacchione, K. H. Baines, and P. D. Nicholson,
2012, The Composition of Iapetus: Mapping Results from Cassini VIMS, Icarus, 218, 831-860.

Clark, R. N., R. H. Brown, and D. M. Lytle, 2016, The VIMS Wavelength and Radiometric Calibration,
NASA Planetary Data System, The Planetary Atmospheres Node, 23p.

Cruikshank, D. P., A. W. Meyerb and, R. H. Brownc and, R. N. Clark, R. Jaumann, K. Stephan, C. A.
Hibbitts, S. A. Sandford, R. M.E. Mastrapa, G. Filacchione, C. M. Dalle Ore, P. D. Nicholson, B. J.
Buratti, T. B. McCord, R. M. Nelson, J. B. Dalton, K. H. B., and D. L. Matson, 2010, Carbon dioxide
on the satellites of Saturn: Results from the Cassini VIMS investigation and revisions to the VIMS
wavelength scale, Icarus, 206, 561–572, doi:10.1016/j.icarus.2009.07.012 .

McCord, T. B., Brown, R., Baines, K., Bellucci, G., Bibring, J.-P., Buratti, B., Capaccioni, F., Cerroni,
P., Clark, R.N., Coradini, A., Cruikshank, D., Drossar, P., Formisano, V., Jaumann, R., Langevin, Y.,
Matson, D., Mennella, V., Nelson, R., Nicholson, P., Sicardy, B., Sotin, C., Hansen, G., Hibbitts, C.,
2004, Cassini VIMS observations of the Galilean satellites, including the VIMS calibration procedure:
Icarus, Vol. 172, p. 104-126.

Miller, Edward A., Gail Klein, David W. Juergens, Kenneth Mehaffey, Jeffrey M. Oseas, Ramon A.
Garcia, Anthony Giandomenico, Robert E. Irigoyen, Roger Hickok, David Rosing, Harold R. Sobel,
Carl F. Bruce, Jr., Enrico Flamini, Romeo DeVidi, Francis M. Reininger, Michele Dami, Alain
Soufflot, Yves Langevin, Gerard Huntzinger, 1996, The Visual and Infrared Mapping Spectrometer for
Cassini , Proc. SPIE 2803, Cassini/Huygens: A Mission to the Saturnian Systems, 206-220 (October 7,
1996), doi: 10.1117/12.253421 .

Thompson, D. R., F. C. Seidel, B. C. Gao, M. M. Gierach, R. O. Green, R. M. Kudela, and P.
Mouroulis , 2015, Optimizing irradiance estimates for coastal and inland water imaging spectroscopy,
Geophys. Res. Lett., 42, 4116–4123, doi:10.1002/2015GL063287.
```
