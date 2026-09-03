---
citation_key: "nichols2021tracking"
title: "Tracking short-term variations in the haze distribution of Titan’s atmosphere with SINFONI VLT"
source_pdf: "data/papers/nichols2021tracking.pdf"
source_pdf_sha256: "c64437261fc152c9500d54bd75f7be2d0b14c73ec539da00fbef39b624089600"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                   https://doi.org/10.3847/PSJ/abffd7
© 2021. The Author(s). Published by the American Astronomical Society.




  Tracking Short-term Variations in the Haze Distribution of Titan’s Atmosphere with
                                    SINFONI VLT
Fiona Nichols-Fleming1                  , Paul Corlies2,3 , Alexander G. Hayes2, Máté Ádámkovics4 , Patricio Rojo5, Sebastien Rodriguez6                     ,
                    1
                                             Elizabeth P. Turtle7, Juan M. Lora8 , and Jason M. Soderblom3
                        Department of Earth, Environmental and Planetary Sciences, Brown University, Providence, RI 02912, USA; pcorlies@mit.edu
                                                  2
                                                    Department of Astronomy, Cornell University, Ithaca, NY 14853, USA
                    3
                        Department of Earth, Atmospheric, and Planetary Science, Massachusetts Institute of Technology, Cambridge, MA 02139, USA
                                               4
                                                 Advanced Technology Center, Lockheed Martin, Palo Alto, CA 94304, USA
                                                    5
                                                      Departamento de Astronomia, Universidad de Chile, Santiago, Chile
                                             6
                                               Université de Paris, Institut de Physique du Globe de Paris, CNRS, Paris, France
                                                       7
                                                         Johns Hopkins Applied Physics Lab, Laurel, MD 20723, USA
                                       8
                                         Department of Earth and Planetary Sciences, Yale University, New Haven, CT 06520, USA
                                    Received 2020 August 3; revised 2021 May 3; accepted 2021 May 6; published 2021 September 6

                                                                               Abstract
             While it has long been known that Titan’s haze and atmosphere are dynamic on seasonal timescales, recent results
             have revealed that they also exhibit signiﬁcant subseasonal variations. Here, we report on observations of Titan
             acquired over an eight-month period between 2014 April and 2015 March with the Spectrograph for Integral Field
             Observations in the Near Infrared instrument on the Very Large Telescope using adaptive optics. These
             observations have an average ﬁve-day cadence, permitting interrogation of the short-period variability of Titan’s
             atmosphere. Disk-resolved spectra in the H and K bands (1.4–2.4 μm) were analyzed with the PyDISORT radiative
             transfer model to determine the spatial distribution and variation of stratospheric haze opacity over subseasonal
             timescales. We observed a uniform decrease in haze opacity at 20°N and an increase in haze opacity at 250–300°E
             and ∼40°N over the span of our observations. Globally, we found variations on the order of 5%–10% on
             timescales of weeks, as well as a steady, global increase in the amount of haze over timescales of months. The
             observed variations in haze opacity over the short timescales of our observations were of similar magnitude to
             long-period variations attributed to seasonal variation, suggesting rapid dynamical processes that may take part in
             the distribution of hazes in Titan’s atmosphere.
             Uniﬁed Astronomy Thesaurus concepts: Ground-based astronomy (686); Atmospheric variability (2119);
             Saturnian satellites (1427)
             Supporting material: data behind ﬁgures


                                    1. Introduction                                         Voyager observations showed a hemispheric asymmetry in
                                                                                         the haze opacity, which was later linked to vertical and
   Titan’s dense atmosphere and photochemically produced
                                                                                         meridional motions above 300 km (Toon et al. 1992; Rannou
hydrocarbon haze have been prominent features of study, as the
                                                                                         et al. 2002; West et al. 2018). This circulation, as well as varying
ﬁrst resolved images of the moon were acquired by the
                                                                                         rates of haze production, were also invoked to reconcile the
Voyager missions four decades ago (Hanel et al. 1981; Smith
                                                                                         discrepancy between predicted and observed geometric albedo
et al. 1981, 1982). The haze distribution has been used to trace
                                                                                         and brightness variations (Hutzell et al. 1993, 1996). After the
the global circulation of the atmosphere (Smith et al. 1981;
                                                                                         descent of the Huygens probe, a coupled photochemical–
Lorenz et al. 2001; Rannou et al. 2004; Larson et al. 2015;
                                                                                         microphysical model of haze, incorporating various chemical
West et al. 2018), while the properties of sunlight scattered by
                                                                                         pathways of haze formation (Lavvas et al. 2008b), self-
the haze particles have been used to constrain their shape, size,
                                                                                         consistently reproduced the local distribution implied by
and abundance distributions (Rages et al. 1983; McKay et al.
                                                                                         measurements from the Descent Imager/Spectral Radiometer
1989; Tomasko et al. 2008; Mishchenko et al. 2016), which
                                                                                         (DISR) instrument on board the probe (Tomasko et al.
vary spatially and with altitude in Titan’s atmosphere (Rages &
                                                                                         2005, 2008; Lavvas et al. 2008b, 2011). Over the years,
Pollack 1983; Lorenz et al. 2001; Tomasko et al. 2005, 2008;
                                                                                         observations of atmospheric haze have identiﬁed signiﬁcant
Anderson et al. 2008; Doose et al. 2016). Thus, the study of the
                                                                                         structure and variability (Teanby et al. 2009; Penteado et al.
physicochemical properties of Titan’s haze and its evolution,
                                                                                         2010; Vinatier et al. 2015, 2020; Karkoschka 2016; Ádámkovics
along with trace gaseous species, are fundamental for better
                                                                                         & de Pater 2017; Seignovert et al. 2017; West et al. 2018) and a
comprehension of the atmosphere’s photochemistry, structure,
and dynamics (Strobel 1974; Yung et al. 1984; McKay et al.                               wide variety of atmospheric models have been developed to
1989, 2001; Rannou et al. 2002; Wilson & Atreya 2004;                                    investigate its formation, evolution, and interaction with the
Lavvas et al. 2008a; Tomasko et al. 2008; Krasnopolsky 2009).                            atmosphere’s circulation (Toon et al. 1992; Rannou et al. 2004;
                                                                                         Lavvas et al. 2008a; Lebonnois et al. 2012; Larson et al. 2015;
                                                                                         Lora et al. 2015). However, the impacts of seasonal changes and
                 Original content from this work may be used under the terms
                 of the Creative Commons Attribution 4.0 licence. Any further            the large-scale dynamics on the global distribution and seasonal
distribution of this work must maintain attribution to the author(s) and the title       variations of haze remain relatively unconstrained, so informa-
of the work, journal citation and DOI.                                                   tion about the temporal evolution of the haze on a variety of

                                                                                     1
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                               Nichols-Fleming et al.

timescales is helpful for better understanding the dominant              44 observations, acquired from 2014 April through 2015
processes in Titan’s atmosphere.                                         March, have higher temporal, spectral, and spatial resolution
   Although the in situ measurements from the Huygens probe              than previous ground-based near-infrared work, and cover a
are the most detailed available, the haze is neither uniform             different season. We interpret these new haze evolution
spatially nor static temporally, as documented by a variety of           observations with a radiative transfer model, as presented in
observations. For example, Hubble Space Telescope (HST)                  Section 3. We present the results of ﬁtting the haze variations in
data show that aerosol size varies with altitude (Karkoschka &           Section 4, and discuss the results and conclude in Section 5.
Lorenz 1997). Subsequent HST observations, acquired two
years later on the opposite hemisphere, found that these
measurements differed signiﬁcantly (Lorenz et al. 1999).                                         2. Observations
Additional variations in haze brightness were observed on                   Titan was observed from 2014 April to 2015 March using the
seasonal timescales as well as with latitude (especially near the        Spectrograph for INtegral Field Observations in the Near Infrared
pole) (Lorenz et al. 2001, 2004). Young et al. (2002) compiled           (SINFONI) on the Very Large Telescope (VLT) at the Paranal
a three-dimensional map of Titan’s haze variations in altitude           Observatory. SINFONI is an integral ﬁeld spectrograph that uses
using six HST ﬁlters and showed that the atmospheric aerosols            adaptive optics to allow for diffraction-limited spatial resolution
above 16 km had a peak opacity near the equator and the                  of Titan’s disk. As an integral ﬁeld spectrometer, each pixel is
atmospheric aerosols below 16 km had peak optical depths in              ﬁber fed to a spectrograph, which has a spectral resolution of
both north and south mid-latitudes. Meier et al. (2000)                  Δλ ≈ 1 nm and a corresponding resolving power of R = λ/
tentatively identiﬁed banded haze structures near the surface            Δλ ≈ 1500 (Eisenhauer et al. 2003; Bonnet et al. 2004).
with HST observations from 1997 and 1998, and Karkoschka                    Throughout the campaign, the H and K bands which
(2016) analyzed the 1997 through 2004 HST observations to                correspond to the wavelength range 1.45–1.7 μm and
show that there are two separately varying haze opacity                  1.9–2.25 μm, respectively, were used simultaneously for each
components found above and below 100 km.                                 observation. These observations used a 0 8 × 0 8 ﬁeld of view
   Similarly, analysis of Cassini Visual and Infrared Mapping            with a pixel scale of 25 mas per pixel, corresponding to 180 km
Spectrometer (VIMS; Brown et al. 2004) acquired between                  at Titan’s equator. The integration time for the standard star
2004 and 2008 revealed both gradual gradients and disconti-              was four seconds in all of the observations and for Titan was
nuities in haze distributions (Penteado et al. 2010; Rannou et al.       ﬁfteen seconds initially, increasing to a few minutes for later
2010). VIMS also observed features such as the north polar               observations. Four exposures were coadded to cover the entire
hood (Rannou et al. 2012; Hirtzig et al. 2013), and a                    disk and increase signal to noise; observations were then
combination of VIMS, Cassini Imaging Science Subsystem                   reduced with the SINFONI data reduction pipeline.
(ISS), and Cassini Composite Infrared Spectrometer (CIRS)                   The average time between observations was ﬁve days.
data were used to identify and study a tropical haze band (de            Frequently, target of opportunity observations allowed for
Kok et al. 2010).                                                        multiple observations of Titan on a given night. The longest
   Haze asymmetries and evolution can also be observed from              gap in the observations occurred between 2014 October and
ground-based telescopes in the near-infrared (Gibbard et al.             2015 February, when Saturn was in conjunction and therefore
1999; Roe et al. 2002; Ádámkovics et al. 2004, 2006). Gibbard            not observable from Earth.
et al. (2004) measured the 2.0 μm haze opacity with speckle and
adaptive optics imaging at the W. M. Keck Observatory and                                           3. Methods
found that the haze opacity in the southern hemisphere decreased
by a factor of two between 1996 and 2004, while the equatorial                            3.1. Radiative Transfer Model
haze opacity stayed approximately constant. Ádámkovics & de                 To analyze the reﬂected sunlight through Titan’s atmos-
Pater (2017) used observations from the OH Suppressing                   phere, we use the PyDISORT radiative transfer model, a
Infrared Imaging Spectrograph (OSIRIS) instrument at the Keck            Python implementation of the widely used DISORT radiative
observatory acquired between 2006 and 2015 to investigate                transfer code (Stamnes et al. 1988), which has been developed
seasonal and meridional variations in haze opacity, and found            speciﬁcally for Titan (Ádámkovics et al. 2016; Ádámkovics &
that the haze above 20 km showed signiﬁcant nonlinear                    de Pater 2017). The Huygens Gas Chromatograph Mass
meridional variations, while the haze below 20 km varied                 Spectrometer (GCMS) and Atmospheric Structure Instrument
nonmonotonically over seasonal timescales.                               (HASI) data provided the methane mole fraction, altitude,
   While it has been demonstrated that seasonally driven                 pressure, and temperature from around 1400 km to the surface
dynamics can redistribute hazes on global and seasonal                   (Fulchignoni et al. 2005; Niemann et al. 2010) and the DISR
timescales, little is known about the distribution of hazes at           data provide the aerosol opacity structure, single-scattering
smaller spatial scales and short temporal intervals. Recent work         albedos, and phase functions (Tomasko et al. 2008).
with VIMS and ISS data has shown variations in the haze on                  PyDISORT utilizes a plane parallel discrete ordinate routine
timescales of hours to days (Carrasco et al. 2018; Rodriguez             to model the radiative transfer of Titan’s atmosphere. It
et al. 2018; West et al. 2018; Seignovert et al. 2021) during            includes contributions from gaseous absorption, collision
Cassini ﬂybys of Titan, and ground-based telescopes can                  induced absorptions, aerosol (multiple) scattering, and reﬂec-
complement these observations by offering similar observation            tions from Titan’s surface. Gaseous absorption is modeled
cadence over months-long temporal baselines.                             using the correlated k-coefﬁcients that are interpolated on a
   Here, we present ground-based observations of haze opacity            temperature and pressure grid to relevant Titan-like conditions
variations in the near-infrared that are complementary to the            as a function of altitude. Likewise, collision induced absorp-
recent works of Ádámkovics & de Pater (2017), West et al.                tions are also modeled for Titan-like temperature and pressures
(2018), Carrasco et al. (2018) and Rodriguez et al. (2018). Our          (McKellar 1989; Lafferty et al. 1996; Hartmann et al. 2017).

                                                                     2
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                         Nichols-Fleming et al.

   To account for scattering, we use the haze scattering phase                                            Table 1
function as derived in Tomasko et al. (2008), extrapolating                  Summary of the Nodes Used to Generate a Look-up Table for Fitting
                                                                                       Observations to the Radiative Transfer Model
the phase function for altitudes >80 km to the surface, as
prescribed in Campargue et al. (2012). Though DISR provided               Parameter                                      Nodes
altitude-dependent values for the single-scattering albedo, these         Haze scaling factor             0.3, 0.5, 0.7, 0.9, 1.1, 1.3 1.5, 1.7
only apply to wavelengths <1.6 μm, and so we utilize a
constant single-scattering albedo in both altitude and spectral           Incident angle                       0, 10, 20, 30, 40, 50, 60
region for each of our two regions of interest. These values are          Emission angle                       0, 10, 20, 30, 40, 50, 60
0.85 for 1.65–1.72 μm and 0.75 for 2.15–2.22 μm, as derived
in Hirtzig et al. (2013). Finally, we use a constant surface              Azimuth angle         0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
albedo of 0.10, although the surface albedo does not affect our                                                          75, 80
results, given as we are not ﬁtting surface-sensitive channels.
   Over the course of the Cassini mission, variations in
temperature of a few degrees on the surface (Jennings et al.              methane opacity limits altitude sensitivity, we do not ﬁt the full
2019) and 20°–30° in the stratosphere (Schinder et al. 2020)              redward wings but rather ﬁt 1.65–1.72 and 2.15–2.22 μm.
have been observed. CIRS was able to measure vertical
temperature proﬁles of Titan’s atmosphere over the course of                               3.3. Data Reduction and Calibration
the Cassini mission (e.g., Vinatier et al. 2015; Teanby et al.               To remove telluric features from our data, we utilize the
2017, 2019; Sylvestre et al. 2018; Coustenis et al. 2020; Mathé           standard processing technique of observing a standard star to
et al. 2020). Here we use seasonally minimum/maximum                      spectrally calibrate the observed ﬂux from Titan. First, a sky
proﬁles as measured by CIRS to measure the inﬂuences of                   template was created using the ESO SkyCalc tool for the time
temperature in the model. To determine if these variation                 and pointing of the observation. Telluric features were then
contribute signiﬁcantly to our analysis, we model reasonable              removed from the observed data by dividing the data by the sky
ranges of the observed temperature proﬁle as measured with                template. Finally a photometric calibration was performed by
CIRS over the latitudes/season of our observations. While                 measuring the observed ﬂux of the standard star, following the
temperature variations in the stratosphere may vary on the scale          process described in Ádámkovics et al. (2016), applying this
of ≈20 K, we do not see variations in the modeled spectra                 correction independently to the H and K bands.
above the noise of the observations and conclude that the single             A small wavelength shift (1–2 spectral channels) was also
Huygens temperature proﬁle is sufﬁcient for this analysis (see            found to result from the VLT calibration pipeline. To account
Appendix A).                                                              for this shift we ﬁt our observations (to get an approximate
                                                                          model spectrum), measured the offset between our observation
                                                                          and our model ﬁt, shifted the observation the correct number of
                  3.2. Altitude Sensitivity Tests                         wavelength bins, and then ﬁnally reﬁt the corrected observation
   Ádámkovics & de Pater (2017) used two scaling factors to               to derive the best-ﬁt haze scaling factor.
constrain the temporal variations in the haze opacity, one for
altitudes <20 km and one for altitudes >20 km. The altitude of                                    3.4. Spectrum Fitting
20 km was chosen as an ad hoc critical altitude at which to split            The process of ﬁtting spectra with PyDISORT is computa-
the model atmosphere. They found relatively stable retrievals             tionally expensive. Each generation of a model spectrum takes
for altitudes >20 km, but generally poorer ﬁts (i.e., large spatial       about two minutes on a standard workstation and, as a result,
or rapid temporal variations) for altitudes <20 km, impacting             the full minimization process requires ∼20 minutes to ﬁnd a
the interpretation of these data.                                         best-ﬁt spectrum, assuming only 10 spectral generations are
   To simplify this analysis, and to more cleanly deﬁne the               required. To reduce the computation time of our ﬁtting routine,
critical altitudes at the tropopause, we adopt regions of interest        we create a ﬁve dimensional look-up table over our spectral
for this work that are only sensitive to vertical opacity of              regions (1D), viewing geometry (3D), and haze scaling factor
Titan’s hazes for altitudes >40 km. We limit ourselves to these           (1D); which we then use for the rapid inversion of the best-ﬁt
altitudes to eliminate sources of error and degeneracies related          haze scaling parameter. Fitting our data in this way takes only
to surface variations and any variability in tropospheric                 seconds, dramatically improving computation time. For this
methane abundance (Lora & Ádámkovics 2017). To determine                  analysis, the haze scaling factor is applied as a constant scaling
which wavelengths would be sensitive to changes in the haze               in haze opacity relative to the DISR proﬁle above 20 km and
opacity at altitudes above 40 km, a sensitivity test similar to           the wavelengths ﬁt constrain this scaling further to the
the one conducted in Ádámkovics & de Pater (2017) was                     stratosphere as described in Section 3.2. The nodes of our
performed. This new test accounts for potential differences in            look-up table are listed in Table 1. To validate this procedure,
sensitivity resulting from the differing spectral resolutions             several ﬁts were also performed using the full minimization
between OSIRIS and SINFONI as well as determines the                      process. The results for these test cases were identical to within
wavelengths of interest for the 1.6 μm window. Figures 8 and 9            error (1σ).
in Appendix B plot the results of this analysis for the 1.6 and              As mentioned previously, we only focus on the stratospheric
2.0 μm windows, respectively. As expected, we ﬁnd that                    haze layers, and therefore we ﬁt the regions of the spectra
wavelength regions with higher I/F are sensitive to lower                 sensitive only to the stratosphere. These regions are 1.65–1.72
altitudes and vice versa, and that the regions of the spectra             and 2.15–2.22 μm and consist of 282 spectral channels.
sensitive to the stratosphere are the far redward wings of the               To ﬁnd best-ﬁt parameters for our observations, we deﬁned a
two methane windows which correspond to 1.65–1.8 and                      function that will return a spectrum from our linearly
2.15–2.30 μm. To prevent ﬁtting to regions in which strong                interpolated look-up table for a given set of observational

                                                                      3
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                         Nichols-Fleming et al.




Figure 1. Observed and best-ﬁt spectra for one latitudinal band of the observations selected for Figure 5. The pixels of interest are shown in yellow on the inset image
of the disk averaged between 1.56 and 1.61 μm, from which averaged spectra are shown in black. The best-ﬁt model spectrum, determined from our minimization
routine, is shown in green. The optimal haze scaling factor and latitude of each ﬁt are listed on the associated panels along with the subobserver latitude and longitude,
and date of the observation.

parameters. We then run a Levenberg–Marquardt (LM) mini-                                each pixel is inside of the disk and that the cosine of the
mization on this function with a constant error in the spectrum of                      average incidence angle and the cosine of the average emission
0.002 and an initial value for the haze scaling factor of 1.0. The                      angle were both greater than 0.80. This requirement removed
error is based on the spread of a histogram of the I/F values of                        any nonphysical variations caused by the degradation of our
all the spectra used in this work in the wavelength range from                          model at the extreme viewing geometries toward the edge of
2.30 to 2.45 μm, following Ádámkovics et al. (2010). This                               the disk. Accounting for these restrictions we ﬁnd a spatial
wavelength region is commonly considered a region of stable                             coverage of ∼70° in latitude (from ∼10°S to ∼60°N) and
signal, so it can be used to model the residual error in the                            ∼100° in longitude. To normalize for inter-epoch variations,
observations. We ﬁt the histogram with a Gaussian, and the                              each observation is scaled to a common value in a region of
square root of the variance of the best-ﬁt histogram was used as                        high methane opacity, and therefore minimal spectral varia-
our spectral error. Our ﬁtting routine returned errors on the                           bility, taken to be over the 2.20–2.35 μm region.
derived parameters based on the gradient of the steepest descent
in the minimization routine. For a more detailed explanation of                                                           4. Results
our consideration of errors see Appendix C.                                                Figure 1 shows the observed and best-ﬁt spectra for a
   To investigate the variations in Titan’s haze, we look at                            selection of the 44 observations covering a variety of
averaged meridional variations across the center of Titan as                            subobserver longitudes and temporal separations between
well as co-varying spatial variations across the entire disk. To                        observations. Overall, the model agrees well with the
increase the signal to noise in our analysis, averages of several                       observations, with the exception of a slight overprediction of
pixels were used for these analyses. In the case of latitudinal                         absorption in the ∼1.7 μm region, suggesting an overestima-
variations, we average 10 pixels centered on the horizontal                             tion of the methane opacity in this spectral region.
center of the disk for that observation in each vertical row.                              Figure 2 shows the latitudinal distribution of the derived
When looking for two-dimensional spatial distributions, we                              haze scaling factor fH with three sigma error bars for the same
averaged pixels in 3 × 3 squares. In all cases, we require that                         selection of observations as displayed in Figure 1. Early in the

                                                                                    4
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                     Nichols-Fleming et al.




Figure 2. Observed and best-ﬁt spectra for the ∼44°N latitude bin on 2015 March 8. The pixels of interest are shown in yellow on the inset image of the disk averaged
between 1.56 and 1.61 μm, from which averaged spectra are shown in black. The best-ﬁt model spectrum, determined from our minimization routine, is shown in
green. The optimal haze scaling factor and latitude of the ﬁt are listed on the ﬁgure along with the subobserver latitude and longitude, and date of the observation.

campaign, a slight latitudinal gradient was observed from 0 to                        minimum at ∼20°N for all longitudes, the evolution of which is
20°N, with a remaining relatively ﬂat scaling factor poleward of                      plotted in Figure 3. The individual two-dimensional distribu-
20°N. In late 2014 May, we observed a sudden increase in the                          tions of haze scaling factor for each observation are provided in
haze opacity at all latitudes, and the emergence of the 0–20°N                        as the data behind Figure 4.
enhancement. The following two months after this global
enhancement show relatively stable haze distributions.
   We ﬁnd approximately ﬂat distributions of the haze scaling                                            5. Discussion and Conclusions
factor in latitude for each observation, with the exception of a                         Based on these 44 observations of Titan we observe global
few interesting features; however, we found that the overall                          changes in the haze on timescales of months that are similar in
haze scaling factor could vary by up to 10% on timescales of                          magnitude to the previously observed seasonal changes
weeks. One feature in particular that we saw in all of the                            (Karkoschka 2016; Ádámkovics & de Pater 2017), as well as
observations, spanning a range of 12 months, was a dip in haze                        smaller-scale variations in the spatial distribution of the haze on
scaling factor at around 20°N latitude. Later observations                            weekly timescales.
showed an enhancement between this dip and the equator as                                Most notable in the observations is a 10% increase in the
well as an occasional swelling between 20 and 60°N latitude                           haze scaling factor over the ﬁrst few months of observations
that ﬂattens over time (see Figure 3).                                                (see Figure 5). Three-dimensional models that attempt to
   The global variations from epoch-to-epoch suggest varia-                           simulate the distribution of Titan’s haze suggest that an
tions in the overall haze opacity on short timescales are similar                     increase in haze in the atmosphere above 200 km is expected
in magnitude to the variations in haze opacities observed on                          around the time of these observations, resulting from the
seasonal timescales (Karkoschka 2016; Ádámkovics & de                                 overturning circulation after northern vernal equinox (Rannou
Pater 2017). This suggests rapid dynamical variations in Titan’s                      et al. 2002, 2004; Lebonnois et al. 2012; Larson et al. 2015).
stratosphere; however, exactly where in the atmospheric                               However, the speed of the enhancement is particularly stark;
column these variations are dominant is not constrained.                              while most models predict an increase in the haze during this
   In addition to ﬁtting constant latitude bins, we compiled a                        time, variations driven by circulation typically happen more
two-dimensional map of haze scaling factor in latitude and                            slowly, on the scale of two to three Earth years based on
longitude using all of our observations, shown in Figure 4. This                      changes to the altitude of the detached haze (see Figure 10 in
plot shows the median haze scaling factor for each latitude and                       Larson et al. 2015). Thus, the observed rapid evolution of the
longitude over the course of our observations, with a resolution                      stratospheric haze over a period of a few months suggests that
of one degree per pixel. The square areas that correspond to the                      rapid perturbations can be as important as seasonal variation in
regions ﬁt by our minimization routine were projected onto                            determining the opacity of Titan’s hazes (Rannou et al.
latitude and longitude maps providing haze scaling factors for                        2002, 2004; Larson et al. 2015).
those regions. There appears to be large-scale structure to the                          At latitudes greater than 50°N, we ﬁnd a general decrease in
spatial variation of the haze, including a local maximum in the                       the haze opacity for all of our observations. While the limb
haze scaling factor at ∼250–300°E and ∼30–45°N, which                                 viewing geometry could produce similar effects, the asymmetry
is present in all observations of the campaign, as well as a                          observed between the equatorial and high northern latitudes in

                                                                                 5
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                      Nichols-Fleming et al.




Figure 3. Latitudinal distribution of haze scaling factor with three sigma error bars for a selection of observations spanning a variety of subobserver longitudes and
temporal separations. The date and subobserver longitude of each observation is indicated in the top right of each panel.

all our observations indicates that this decrease is real (see                        years after northern vernal equinox (2009 August), but before
Figure 3) and is consistent with a Hadley cell circulation                            northern summer solstice (2017 May). Finer constraints on the
decreasing the abundance of hazes over the summer pole                                time, duration, and structure of the transition between seasons
(Rannou et al. 2004; Lebonnois et al. 2012; Larson et al. 2015).                      will require similarly cadenced observations as those presented
It is believed that as Titan changes seasons, the mean                                here, but over a longer temporal baseline.
meridional circulation clears the pole that is moving into                               Another interesting feature is the evolution of hazes at
summer of the hazes, driving them to the pole that is moving                          equatorial to mid-latitudes. The campaign begins showing a
into winter (West et al. 2018). While we cannot observe                               general decrease in haze opacities at these latitudes, but the
southern latitudes, the proﬁles in the northern hemisphere are                        observations subsequently evolve to form a local enhancement in
consistent with this interpretation. Furthermore, these observa-                      the hazes from 0 to 20°N for much of the campaign (see
tions provide a constraint on the timescale of the seasonal shifts                    Figure 3). Comparison with observations from Ádámkovics &
of the overturning circulation of Titan’s atmosphere, suggesting                      de Pater (2017) in 2015 July, which show no such enhancement,
the meridional motions of hazes is well underway by several                           suggests that this feature may have dissipated on the timescale of

                                                                                  6
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                          Nichols-Fleming et al.




Figure 4. Spatial distribution of the median haze scaling factor for all of our observations. Fits are projected onto a latitude and longitude map with a resolution of one
degree per pixel. The medians of each pixel on the maps are compiled to create an overall spatial distribution and then averaged in either dimension to produce
equatorial and meridional variations over the full length of our observations. The error bars on the equatorial and meridional variations represent the standard deviation
of the values in the associated latitudinal or longitudinal strip. The data behind this ﬁgure is available in a numpy array in the .tar.gz package.
(The data used to create this ﬁgure are available.)




Figure 5. Temporal variability of Titan’s haze scaling factor from our observations. Disk-averaged values of haze scaling factor are shown with 1σ error bars for each
observation representing the propagation of errors for each spectra ﬁt across the disk. Our observations show a monotonic global increase for the ﬁrst few months of
the campaign along with some short-term variability on the timescale of weeks. The data behind this ﬁgure is available in a numpy array and machine readable format.
(The data used to create this ﬁgure are available.)


a few months, capping this local enhancement to approximately                            equatorial latitudes between mid 2014 to early 2015. Although
12 months in duration. We cannot determine whether the                                   the reappearance of the detached haze layer was not observed
variations in hazes in our observations are caused by a source or                        until late 2015 or early 2016 (West et al. 2018; Seignovert et al.
sink of the stratospheric hazes and more 3D modeling of the                              2021), the haze extinction in the atmosphere above 350 km
photochemical haze production and dynamics are required.                                 varied signiﬁcantly at the time of our observations (see Figure 2
   One possibility could be the result of rising motion in the                           of West et al. 2018; Figure 15 of Seignovert et al. 2021). This
stratosphere resulting from the local insolation maximum at these                        upper atmosphere variability is consistent with our results,
latitudes for this season (Lora et al. 2015). Indeed, upwelling in                       suggesting the high variability observed in the upper atmosphere
the equatorial region during this time period is predicted to                            by ISS is also observed in the stratosphere. Variations in haze
explain the independent measurements of Bézard et al. (2018)                             opacity of this magnitude were also observed in the atmosphere
and Vinatier et al. (2020) and 3D GCMs predict an enhancement                            above 70 km by VIMS between 2009 May and 2010 July (see
in hazes above 200 km at equatorial latitudes (see Figure 5 of                           Supplementary Information of Rodriguez et al. 2018).
Lebonnois et al. 2012, Figure 10 of Larson et al. 2015). The                                Further, the local enhancement results in the apparent
results presented here also favor the prediction of Larson et al.                        appearance of a dark band at ∼20°N (see Figure 4), which
(2015) of the reappearance of the detached haze layer at                                 could also be a similar feature to the dark bands observed by

                                                                                    7
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                 Nichols-Fleming et al.

the Voyager 1 mission at northern mid-latitudes (Smith et al.              Seignovert et al. 2021), but this work shows that they can be
1981). Our observations are approximately half-way between                 observed from the ground as well and that further ground-based
Titan’s spring equinox and summer solstice, and the Voyager 1              monitoring campaigns can provide valuable insight into the
encounter occurred around Titan’s spring equinox, suggesting               types of variations and features described in this work.
the possibility of the seasonal recurrence of this feature.
Combined with the local enhancement at equatorial latitudes                   This work was based on observations collected at the
described previously, this would suggest a dynamic origin for              European Southern Observatory under ESO programs 093.C-
this feature.                                                              0557 and 094.C-0422. Fiona Nichols-Fleming gratefully
   If diurnal variations in the haze exist (previously seen in             acknowledges the summer 2017 REU program in the Cornell
Coustenis et al. 2001; Hirtzig et al. 2006), we do not see strong          Center for Astrophysics and Planetary Science at Cornell
variations between 9 am and 3 pm local time in our                         University and the NSF for ﬁnancial support under NSF award
observations. This lack of visible variation, therefore, places            AST-1659264. Paul Corlies gratefully acknowledges NASA
a limit on the times and magnitudes over which the diurnal                 for ﬁnancial support under NASA Earth and Space Science
differences in Titan’s haze may exist.                                     Fellowship NNX14AO31H S03. We thank Peter Gierasch and
   Finally, we compare our observations to previous work,                  Maryame El Moutamid for providing us with CIRS temper-
which has shown a long-term variability in Titan’s hazes                   ature data to constrain the model sensitivity to variations in
(Karkoschka 2016; Ádámkovics & de Pater 2017). Our work                    temperature. Finally, we thank two anonymous reviewers for
complements these longer baseline studies by observing at a                the careful and considerate reviews, which have helped to
higher temporal cadence. From this comparison we ﬁnd short                 signiﬁcantly improve the ﬁnal manuscript.
timescale variations in the haze of the same magnitude in
addition to a consistent increase in global haze over the ﬁrst                                   Appendix A
few months of the observing campaign (see Figure 5). The data                   The Effect of Temperature Variations on Spectra
sets from this work as well as Karkoschka (2016) and
                                                                              CIRS was able to measure vertical temperature proﬁles of
Ádámkovics & de Pater (2017) were each constructed using
                                                                           Titan’s atmosphere over the course of the Cassini mission (e.g.,
a unique model for observations from a different instrument
                                                                           Vinatier et al. 2015; Teanby et al. 2017, 2019; Sylvestre et al.
(SINFONI, HST, and OSIRIS, respectively), therefore we do
                                                                           2018; Coustenis et al. 2020; Mathé et al. 2020). To quantify the
not mean to compare the absolute agreement of the three
                                                                           importance of temperature variations on the derived haze
models, but rather compare the results to show context for the             scaling factor, we used three temperature proﬁles measured by
magnitude of the variations seen in this work. Combined, these             CIRS for northern, southern, and equatorial latitudes as well as
three sets of observations suggest variations in the hazes at              the temperature proﬁle measured by Huygens (see Figure 6) to
multiple timescales, thus requiring high temporal cadence to               produce four different modeled spectra for the same viewing
fully understand the dynamical properties of Titan’s hazes, and            geometry. There are slight variations between these spectra, as
suggests that long-term trends must also be considered in the              shown in Figure 7, but these variations are well within our error
context of short-term variability to understand and interpret the          of 0.002 indicating that there is not signiﬁcant error introduced
complex dynamics of Titan’s atmosphere from haze circula-                  by using the Huygens temperature proﬁle at all latitudes. The
tion. A more thorough analysis using multiple telescopes could             atmospheric temperatures can also vary throughout Titan’s
better inform the relative contributions of short-term and                 seasons, which may introduce other errors as our observations
seasonal variability of Titan’s hazes.                                     are from a different season than when the Huygens temperature
   These observed changes in haze opacity, both globally and               proﬁle was measured, but these variations should be on the
on smaller scales, indicate that there is either more variability in       same order as those in Figure 6, and therefore it should be the
the production of stratospheric haze, or more variability in the           case that these effects are well within our error of 0.002 as well.
redistribution of hazes into the stratosphere, than previously
appreciated. While we have proposed some explanations for
                                                                                                    Appendix B
observed features, more detailed work on GCMs is necessary
                                                                                            Altitude Sensitivity Analysis
beyond the scope of this work. These short timescale variations
have been seen previously in the detached haze and atmosphere                To determine which SINFONI wavelengths are sensitive to
from Cassini data (Rodriguez et al. 2018; West et al. 2018;                which altitudes, we use a methodology similar to that of




                                                                       8
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                 Nichols-Fleming et al.




Figure 6. Measured temperature proﬁles from Huygens and three additional latitudinal regions from CIRS. There are minimal temperature variations observed for the
surface with larger variations up to about 40 K in the upper part of the stratosphere.




Figure 7. Modeled spectra for the same viewing geometry using the four temperature proﬁles shown in Figure 6. Uniform error bars of 0.002 applied to the modeled
spectrum using the Huygens proﬁle are shown by the gray envelope. Modeled spectra using the temperature proﬁles from the northern, southern, and equatorial
latitudes are shown in red, blue, and green, respectively. Differences between the modeled spectra become more pronounced for the longer wavelengths.




                                                                               9
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                       Nichols-Fleming et al.

Ádámkovics & de Pater (2017), in which we construct a                                  each SINFONI channel is most sensitive to, similar to a
high-resolution model that sampled Titan’s atmosphere every                            contribution function. Figures 8 and 9 display the result of
∼2.5 km over the entire spectral range of our observations.                            this analysis. In all, we ﬁnd wavelengths in the methane
For each altitude, we insert a small, homogeneous addition of                          windows, and therefore weaker methane absorption, to be
haze opacity (δτH = 0.1) for that layer and generate a new                             sensitive all the way to Titan’s surface. Conversely,
model for comparison to the un-altered reference. As the                               wavelengths in the methane band, and therefore stronger
hazes are efﬁcient scatterers, increasing their opacity acts to                        methane absorption, are only sensitive to the highest
increase Titan’s reﬂectivity, with the strongest variations                            altitudes. For our ﬁnal selection, we chose a combination of
corresponding the to altitude at which the wavelength is most                          wavelengths most sensitive to Titan’s stratosphere from
sensitive. In this way, we empirically determine the altitude                          approximately 40 to 300 km.




Figure 8. Figure showing the sensitivity of wavelength channels in the 1.6 μm window to changes in haze opacity at particular altitudes. The channels between 1.65
and 1.8 μm are sensitive to changes in the stratosphere (above 40 km). The black line in the upper panel is the original I/F without the addition of any haze variation.




Figure 9. Figure showing the sensitivity of wavelength channels in the 2 μm window to changes in haze opacity at particular altitudes. The channels between 2.15 and
2.3 μm are sensitive to changes in the stratosphere (above 40 km). The black line in the upper panel is the original I/F without the addition of any haze variation.


                                                                                  10
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                       Nichols-Fleming et al.




Figure 10. Histograms of derived haze scaling factors for three MCMCs with the solid black line indicating the best-ﬁt scaling factor. Each spectrum was offset
uniformly by a random value sampled from a Gaussian distribution centered on zero with a standard deviation of one, two, or three times 0.002, the error of our
observations, before being ﬁt by our minimization routine.




Figure 11. Histograms of derived haze scaling factors for an MCMC with the solid black line indicating the best-ﬁt scaling factor. Each wavelength of the spectrum was
offset by a random value sampled from a Gaussian distribution centered on zero with a standard deviation of 0.002, the error of our observations, before being ﬁt by our
minimization routine. A best-ﬁt Gaussian distribution with a mean of 1.043 and a standard deviation of 0.0127 is shown in blue as well.




                                                                                  11
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
The Planetary Science Journal, 2:180 (12pp), 2021 October                                                                                       Nichols-Fleming et al.

                              Appendix C                                                 Doose, L. R., Karkoschka, E., Tomasko, M. G., & Anderson, C. M. 2016, Icar,
                         Justiﬁcation of Errors                                             270, 355
                                                                                         Eisenhauer, F., Abuter, R., Bickert, K., et al. 2003, Proc. SPIE, 4841, 1548
   To get a sense for an upper bound on the errors associated                            Fulchignoni, M., Ferri, F., Angrilli, F., et al. 2005, Natur, 438, 785
with our derived haze scaling factors, we ran three MCMCs in                             Gibbard, S. G., de Pater, I., Macintosh, B. A., et al. 2004, GeoRL, 31, L17S02
                                                                                         Gibbard, S. G., Macintosh, B., Gavel, D., et al. 1999, Icar, 139, 189
which a chosen spectrum is uniformly offset by a value                                   Hanel, R., Conrath, B., Flasar, F. M., et al. 1981, Sci, 212, 192
randomly sampled from a normal distribution centered at zero                             Hartmann, J.-M., Boulet, C., & Toon, G. C. 2017, JGRD, 122, 2419
with a standard deviation of one, two, or three times the error in                       Hirtzig, M., Bézard, B., Lellouch, E., et al. 2013, Icar, 226, 470
the observations of 0.002 as used in the main text. Figure 10                            Hirtzig, M., Coustenis, A., Gendron, E., et al. 2006, A&A, 456, 761
contains the histograms of the results from these MCMCs                                  Hutzell, W. T., McKay, C. P., & Toon, O. B. 1993, Icar, 105, 162
                                                                                         Hutzell, W. T., McKay, C. P., Toon, O. B., & Hourdin, F. 1996, Icar, 119, 112
where the solid black line in each is the best-ﬁt scaling factor                         Jennings, D. E., Tokano, T., Cottini, V., et al. 2019, ApJL, 877, L8
for the original, unchanged spectrum. As expected, with larger                           Karkoschka, E. 2016, Icar, 270, 339
offsets allowed, the distribution of scaling factors becomes                             Karkoschka, E., & Lorenz, R. D. 1997, Icar, 125, 369
wider. A ﬁt of the histogram in Figure 10(a) provides an                                 Krasnopolsky, V. A. 2009, Icar, 201, 226
                                                                                         Lafferty, W. J., Solodov, A. M., Weber, A., Olson, W. B., & Hartmann, J.-M.
estimate of the error for a systematic offset in the data of                                1996, ApOpt, 35, 5911
approximately 20%. This is likely an overestimate of the true                            Larson, E. J. L., Toon, O. B., West, R. A., & Friedson, A. J. 2015, Icar,
errors as it would represent a systematic error source.                                     254, 122
   To get a more accurate representation of errors, we ran                               Lavvas, P., Grifﬁth, C. A., & Yelle, R. V. 2011, Icar, 215, 732
another MCMC where each channel of our selected spectrum                                 Lavvas, P. P., Coustenis, A., & Vardavas, I. M. 2008a, P&SS, 56, 67
                                                                                         Lavvas, P. P., Coustenis, A., & Vardavas, I. M. 2008b, P&SS, 56, 27
was given a unique offset randomly sampled from the Gaussian                             Lebonnois, S., Burgalat, J., Rannou, P., & Charnay, B. 2012, Icar, 218, 707
distribution centered on zero with a standard deviation of                               Lora, J. M., & Ádámkovics, M. 2017, Icar, 286, 270
0.002. This produced the histogram shown in Figure 11 where                              Lora, J. M., Lunine, J. I., & Russell, J. L. 2015, Icar, 250, 516
the solid black line represents the best ﬁt for the data without an                      Lorenz, R. D., Lemmon, M. T., Smith, P. H., & Lockwood, G. W. 1999, Icar,
                                                                                            142, 391
offset. When this distribution is ﬁt with a Gaussian (shown in
                                                                                         Lorenz, R. D., Smith, P. H., & Lemmon, M. T. 2004, GeoRL, 31, L10702
blue on Figure 11), the estimate of the errors is approximately                          Lorenz, R. D., Young, E. F., & Lemmon, M. T. 2001, GeoRL, 28, 4453
1% which is on the same order as the errors reported by our                              Mathé, C., Vinatier, S., Bézard, B., et al. 2020, Icar, 344, 113547
minimization routine justifying our use of the minimization                              McKay, C. P., Coustenis, A., Samuelson, R. E., et al. 2001, P&SS, 49, 79
derived errors as our reported errors.                                                   McKay, C. P., Pollack, J. B., & Courtin, R. 1989, Icar, 80, 23
                                                                                         McKellar, A. R. W. 1989, Icar, 80, 361
                                                                                         Meier, R., Smith, B. A., Owen, T. C., & Terrile, R. J. 2000, Icar, 145, 462
                                ORCID iDs                                                Mishchenko, M. I., Dlugach, J. M., Yurkin, M. A., et al. 2016, PhR, 632, 1
                                                                                         Niemann, H. B., Atreya, S. K., Demick, J. E., et al. 2010, JGRE, 115, E12006
Fiona Nichols-Fleming https://orcid.org/0000-0002-                                       Penteado, P. F., Grifﬁth, C. A., Tomasko, M. G., et al. 2010, Icar, 206, 352
7700-5139                                                                                Rages, K., & Pollack, J. B. 1983, Icar, 55, 50
                                                                                         Rages, K., Pollack, J. B., & Smith, P. H. 1983, JGR, 88, 8721
Paul Corlies https://orcid.org/0000-0002-6417-9316                                       Rannou, P., Cours, T., Le Mouélic, S., et al. 2010, Icar, 208, 850
Máté Ádámkovics https://orcid.org/0000-0003-1869-0938                                    Rannou, P., Hourdin, F., & McKay, C. P. 2002, Natur, 418, 853
Sebastien Rodriguez https://orcid.org/0000-0003-                                         Rannou, P., Hourdin, F., McKay, C. P., & Luz, D. 2004, Icar, 170, 443
1219-0641                                                                                Rannou, P., Le Mouélic, S., Sotin, C., & Brown, R. H. 2012, ApJ, 748, 4
Juan M. Lora https://orcid.org/0000-0001-9925-1050                                       Rodriguez, S., Le Mouélic, S., Barnes, J. W., et al. 2018, NatGe, 11, 727
                                                                                         Roe, H. G., de Pater, I., Macintosh, B. A., et al. 2002, Icar, 157, 254
Jason M. Soderblom https://orcid.org/0000-0003-                                          Schinder, P. J., Flasar, F. M., Marouf, E. A., et al. 2020, Icar, 345, 113720
3715-6407                                                                                Seignovert, B., Rannou, P., Lavvas, P., Cours, T., & West, R. A. 2017, Icar,
                                                                                            292, 13
                                                                                         Seignovert, B., Rannou, P., West, R. A., & Vinatier, S. 2021, ApJ, 907, 36
                                 References                                              Smith, B. A., Soderblom, L., Batson, R. M., et al. 1982, Sci, 215, 504
                                                                                         Smith, B. A., Soderblom, L., Beebe, R. F., et al. 1981, Sci, 212, 163
Ádámkovics, M., Barnes, J. W., Hartung, M., & de Pater, I. 2010, Icar,                   Stamnes, K., Tsay, S. C., Jayaweera, K., & Wiscombe, W. 1988, ApOpt,
   208, 868                                                                                 27, 2502
Ádámkovics, M., & de Pater, I. 2017, Icar, 290, 134                                      Strobel, D. F. 1974, Icar, 21, 466
Ádámkovics, M., de Pater, I., Hartung, M., et al. 2006, JGRE, 111, E07S06                Sylvestre, M., Teanby, N. A., Vinatier, S., Lebonnois, S., & Irwin, P. G. J.
Ádámkovics, M., de Pater, I., Roe, H. G., Gibbard, S. G., & Grifﬁth, C. A.                  2018, A&A, 609, A64
   2004, GeoRL, 31, L17S05                                                               Teanby, N. A., Bézard, B., Vinatier, S., et al. 2017, NatCo, 8, 1586
Ádámkovics, M., Mitchell, J. L., Hayes, A. G., et al. 2016, Icar, 270, 376               Teanby, N. A., de Kok, R., & Irwin, P. G. J. 2009, Icar, 204, 645
Anderson, C. M., Young, E. F., Chanover, N. J., & McKay, C. P. 2008, Icar,               Teanby, N. A., Sylvestre, M., Sharkey, J., et al. 2019, GeoRL, 46, 3079
   194, 721                                                                              Tomasko, M. G., Archinal, B., Becker, T., et al. 2005, Natur, 438, 765
Bézard, B., Vinatier, S., & Achterberg, R. K. 2018, Icar, 302, 437                       Tomasko, M. G., Doose, L., Engel, S., et al. 2008, P&SS, 56, 669
Bonnet, H., Conzelmann, R., Delabre, B., et al. 2004, Proc. SPIE, 5490, 130              Toon, O. B., McKay, C. P., Grifﬁth, C. A., & Turco, R. P. 1992, Icar, 95, 24
Brown, R. H., Baines, K. H., Bellucci, G., et al. 2004, SSRv, 115, 111                   Vinatier, S., Bézard, B., Lebonnois, S., et al. 2015, Icar, 250, 95
Campargue, A., Leshchishina, O., Wang, L., et al. 2012, JQSRT, 113, 1855                 Vinatier, S., Mathé, C., Bézard, B., et al. 2020, A&A, 641, A116
Carrasco, N., Tigrine, S., Gavilan, L., Nahon, L., & Gudipati, M. S. 2018,               West, R. A., Seignovert, B., Rannou, P., et al. 2018, NatAs, 2, 495
   NatAs, 2, 489                                                                         Wilson, E. H., & Atreya, S. K. 2004, JGRE, 109, E06002
Coustenis, A., Gendron, E., Lai, O., et al. 2001, Icar, 154, 501                         Young, E. F., Rannou, P., McKay, C. P., Grifﬁth, C. A., & Noll, K. 2002, AJ,
Coustenis, A., Jennings, D. E., Achterberg, R. K., et al. 2020, Icar, 344, 113413           123, 3473
de Kok, R., Irwin, P. G. J., Teanby, N. A., et al. 2010, Icar, 207, 485                  Yung, Y. L., Allen, M., & Pinto, J. P. 1984, ApJS, 55, 465




                                                                                    12
```
