---
citation_key: "neilson2018limb"
title: "Limb darkening and planetary transits II: Intensity profile bias factors for a grid of model stellar atmospheres"
source_pdf: "data/papers/neilson2018limb.pdf"
source_pdf_sha256: "efdead1e24016212e73871a5b797c1c1918c5eb9d35bd4e26cb6c0feef7934b2"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
A&A 662, A38 (2022)
https://doi.org/10.1051/0004-6361/201833635                                                                             Astronomy
                                                                                                                         &
© ESO 2022
                                                                                                                        Astrophysics


                               Limb darkening and planetary transits
          II. Intensity profile bias factors for a grid of model stellar atmospheres
                                     Hilding R. Neilson1 , John B. Lester1,2 , and Fabien Baron3

      1
        Department of Astronomy & Astrophysics, University of Toronto, 50 St. George Street, Toronto, ON, M5S 3H4, Canada
        e-mail: neilson@astro.utoronto.ca
      2
        Department of Chemical & Physical Sciences, University of Toronto Mississauga, Mississauga, ON L5L 1C6, Canada
      3
        Center for High Angular Resolution Astronomy, Department of Physics and Astronomy, Georgia State University, PO Box 5060,
        Atlanta, GA 30302-5060, USA
     Received 13 June 2018 / Accepted 15 November 2021


                                                                    ABSTRACT

     The ability to observe extrasolar planets transiting their stars has profoundly changed our understanding of these planetary systems.
     However, these measurements depend on how well we understand the properties of the host star, such as radius, luminosity, and
     limb darkening. Traditionally, limb darkening is treated as a parameterization in the analysis, but these simple parameterizations are
     not accurate representations of actual center-to-limb intensity variations (CLIV) to the precision needed for interpreting these transit
     observations. This effect leads to systematic errors for the measured planetary radii and corresponding measured spectral features. We
     computed synthetic planetary transit light curves using model stellar atmosphere CLIV and their corresponding best-fit limb-darkening
     laws for a grid of spherically symmetric model stellar atmospheres. From these light curves, we measured the differences in flux as a
     function of the star’s effective temperature, gravity, mass, and the inclination of the planet’s orbit. We find that the ratio of the planet
     radius to the radius of the star may have errors up to about 13% depending on stellar type, wavelength, and inclination of the orbit.
     Key words. planets and satellites: fundamental parameters – stars: atmospheres


1. Introduction                                                             for a sample of Kepler transit observations were inconsistent
                                                                            with predictions, raising questions about the physics of stellar
There are currently more than three thousand confirmed extra-               atmospheres along with our understanding of planetary tran-
solar planets, many of which were discovered using the Kepler               sits. Howarth (2011) argued that the differences were the result
space telescope via the transit method. This method has revolu-             of the planet’s orbit being inclined with respect to our line of
tionized our view of planets and the potential for discovering life         sight. In that case, the measured limb-darkening parameters dif-
in the Universe.                                                            fered, because the transit observations probed only part of the
    Planet transit observations are now so precise that it is pos-          CLIV, whereas the LDLs from model stellar atmospheres are
sible to characterize the composition and structure of extrasolar           constructed from the entire CLIV. Howarth (2011) was able to
planets (Seager & Deming 2010). As more powerful telescopes                 resolve those errors for some stars of that sample by fitting
and satellites become available and surveys are launched, it is             limb-darkening coefficients over only part of the CLIV.
expected that more Earth-like planets will be discovered in the                  In addition to the degeneracy created by the transit inclina-
next decade and that we will potentially detect the presence of             tion, the representation of the CLIV also impacts attempts to
biomarkers from their the atmospheres (Rauer et al. 2014; Ricker            extract information about the transiting planet’s spectrum and
et al. 2015).                                                               composition from the light curve. For example, there have been
    Even with all of the progress made in the past decade,                  conflicting claims regarding the composition of the atmosphere
there remain a number of challenges. One such challenge is that             of GJ 1214 from transit spectral observations (Croll et al. 2011).
analyzing planetary transit light curves requires understanding             Using near-infrared transit spectra, Croll et al. (2011); Gillon
stellar limb darkening, which is also called the center-to-limb             et al. (2014) and Cáceres et al. (2014) determined that the planet’s
intensity variation (CLIV). The CLIV is the observed change of              atmosphere must have a small mean-molecular weight, but that
intensity from the center to the edge of the stellar disk. Mandel           result is contested by other observations (Bean et al. 2011; Berta
& Agol (2002) developed an analytic model of a planetary transit            et al. 2012).
assuming a simplified parameterization of stellar CLIV, typically                Similarly, Hirano et al. (2016); Fukui et al. (2016) and oth-
as either a quadratic limb-darkening law or a four-parameter law            ers report precisions of the order of 1% for measuring Rp /R∗ for
(Claret 2000).                                                              planets orbiting F-type stars. Almenara et al. (2015) reported pre-
    Representing the CLIV with a simplified limb-darkening law              cisions better than 1% for planets orbiting an evolved metal-poor
(LDL) has been a reasonable approach for understanding most                 F star. These results are very precise, yet they depend on their
planetary transit observations, but there are a number of exam-             assumptions of stellar limb darkening. As such, can we be sure
ples where the measured limb darkening disagreed with that                  these best-fit parameters are accurate?
predicted from model stellar atmospheres. Kipping & Bakos                        It is becoming increasingly apparent that the current two-,
(2011a,b) found that limb-darkening parameterizations measured              three- and four-parameter limb-darkening laws are simply
                                                        Article published by EDP Sciences                                           A38, page 1 of 10
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                          A&A 662, A38 (2022)

inadequate for high-precision planetary transit models. We             any fit, as well as determining their impact on additional phe-
showed (Neilson et al. 2017, hereafter Paper I) that synthetic         nomena such as spectral features and oblateness. In the next
planetary transit light curves computed directly from model stel-      section, we describe our models and how we measure the differ-
lar atmosphere CLIV differ from light curves computed from             ences between synthetic planetary transit light curves computed
best-fit limb-darkening laws for a solar-like star, where the only     directly from the model CLIV and from limb-darkening laws.
difference is the shape of the intensity profile employed. This        In Sect. 3, we consider the definition of the stellar radius and its
shows that fitting errors in planetary transit observations do not     impact on our analysis. We present the biases computed from our
come only from errors in the limb-darkening parameters, but            model stellar atmosphere grids in Sect. 4 as a function of stel-
also from the assumption of a specific type of limb-darkening          lar properties, and we present our results in Sects. 5 and 6. We
law. These errors range from about 100 to a few hundred                discuss the impact of these results in terms of the atmospheric
parts-per-million and vary as a function of wavelength. Simi-          extension of a star, that is, the size of the atmosphere relative to
lar results were found independently by Morello et al. (2017).         the stellar radius, in Sect. 7.
Hence, assuming a simple limb-darkening law contaminates
measurements of extrasolar planet spectra, oblateness, and other       2. Model stellar atmospheres
phenomena.
     A number of previous works have compared the differences          Our analysis used the spherically symmetric model stellar atmo-
between planet transit light curves. Mandel & Agol (2002)              spheres from Neilson & Lester (2013b), which were computed
showed that the transit depth changes if one uses the Claret           using the SATLAS codes (Lester & Neilson 2008). These mod-
(2000) four-parameter law instead of a quadratic law of the order      els were computed for stellar masses spanning the range from
of tens of parts-per-million. However, that result is based on         M∗ = 0.2 to 1.4 M in steps of ∆M∗ = 0.3 M , effective tem-
limb-darkening coefficients computed from plane-parallel model         peratures T eff = 3500 to 8000 K in steps of 100 K and surface
stellar atmospheres. Sing (2010) computed Kepler- and CoRot-           gravities log g = 4 to 4.75 in steps of 0.25 dex. This is equivalent
band limb-darkening coefficients for plane-parallel Atlas models       to a range of luminosities from about 0.01 to 15 L and radii from
as well finding small flux errors.                                     0.3 to 2 R . For each model, the stellar CLIV was computed at
     However, limb-darkening laws and plane-parallel model             329 wavelengths for one thousand points of µ, where µ is the
atmospheres are not accurate representations of stellar atmo-          cosine of the angle formed by a point on the stellar disk and the
sphere CLIV, particularly near the limb of the star. Neilson &         disk center, or the angle formed between the line of sight and
Lester (2011) found that, for spherically symmetric model stellar      the line that is normal to the surface at a point on the stellar disk.
atmospheres, currently favored quadratic limb-darkening laws fit       The model atmosphere employed in Paper I is part of this grid of
the model CLIV poorly. The Claret (2000) four-parameter law            models.
provides a more precise fit, but it is still of limited accuracy           The model CLIVs, integrated over the BVRI JK, and Transt-
near the limb of the star. This result was confirmed for giant and     ing Exoplanet Survey Satellite, TESS-wavebands (Ricker et al.
supergiant stars (log g ≤ 3) (Neilson & Lester 2013a) as well as       2016), were used to compute the corresponding best-fit limb-
for dwarf stars (Neilson & Lester 2013b). Specifically, these laws     darkening coefficients for the quadratic limb-darkening law. We
fail for two reasons: the first is the more complex structure of the   used these CLIVs, calculated using the methods described in
CLIV that prevents simple limb-darkening laws from fitting the         Paper I, and the corresponding best-fit coefficients to com-
intensity near the limb of the star, and the second is the inabil-     pute synthetic planetary transit light curves using the analytic
ity for best-fit limb-darkening laws to accurately reproduce the       prescription developed by Mandel & Agol (2002) for the small-
stellar flux; hence, they are not necessarily related to the stellar   planet assumption, represented by ρ, which is defined as
atmosphere itself.                                                           Rp
     These two differences between model CLIV and best-fit             ρ≡       ≤ 0.1.                                                    (1)
                                                                             R∗
limb-darkening laws cause the differences between synthetic
planetary transit light curves found in Paper I. Because the               While the small-planet assumption is not perfect, we have
errors in best-fit limb-darkening are a function of stellar prop-      shown that the difference between light curves follows the
erties, it is likely that the errors introduced by assuming a simple   same behavior regardless of planet radius. Furthermore, all we
limb-darkening parameterization are also a function of stellar         are truly modeling is the difference between CLIV and limb-
properties. In this work, we present computed biases as functions      darkening as a function of µ. We also note that Morello et al.
of stellar properties, and wavebands for dwarf stars for transit       (2017) found similar results using a different prescription for
light curves computed directly from model CLIVs and from best          modeling planetary transits.
fit limb-darkening laws fit to those model CLIV. This work is              Using the synthetic planetary transit light curves computed
done for the idealized cases where we are exploring only the bias      for each model atmosphere using both the CLIV and limb-
introduced by assuming a quadratic limb-darkening law instead          darkening coefficients, we compute the average difference and
of a more realistic, spherically symmetric model CLIV. We did          the greatest difference for each waveband and model stel-
not test synthetic observations as this is a parameter study ask-      lar atmosphere for ρ = 0.1. The computed average difference
ing what the error is in a perfect situation. We showed in Paper I     between light curves acts as a measure of the systematic error
that varying limb-darkening coefficients does not improve the          of the fit for properties, such as relative planet radius, limb-
bias significantly along with varying other planet-transit fitting     darkening coefficients, and, potentially, secondary quantities
parameters except the ratio of the planet and stellar radius. As       such as planetary oblateness and star spots.
such, any bias due to assuming a type of limb-darkening law will           The computed flux differences are functions of ρ, as defined
exist in the relative radius in any standard Markov-chain Monte        in Eq. (1). To the first order, the difference can be written as
Carlo (MCMC) analysis fit. The bias will not disappear nor be          ∆ f = (ICLIV − ILDL ) × ρ2 ,                                      (2)
“spread out” in such a fit.
     These biases can be applied to planetary transit observa-         where ICLIV and ILDL are the intensities from the CLIV and limb-
tions for the purpose of defining the systematic uncertainties of      darkening law, respectively. Because of the definition of ρ, the
A38, page 2 of 10
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                         H. R. Neilson et al.: Limb-darkening and planetary transits

average difference scales as the ratio of the surface areas of the           The second option is to construct a planetary transit code
planet and the star. For example, if one measures ρ = 0.05 and          that forces the edge of the stellar disk to be RRoss, such that µ = 0
our model assumes ρ = 0.1, then the measured average error will         corresponds to the point RLD . However, this method also requires
be (0.05/0.1)2 = 0.25× the difference measured in this paper for        that we know the ratio between RRoss and RLD a priori, so the first
the same stellar properties. We also computed the root-mean-            option is preferred as it is simpler for computation.
square (RMS) flux error as a measure of how well the assumption              The third option, which we reject, is to clip the CLIV so
of a quadratic limb-darkening law fits our more realistic CLIV          that the contribution to the CLIV from the extended part of the
planetary transit light curve.                                          atmosphere is removed, and then to rescale the CLIV so that
                                                                        µ = 0 corresponds to RRoss (Claret & Hauschildt 2003; Espinoza
                                                                        & Jordán 2016; Claret 2017). This clipping can be done by find-
3. Definition of the stellar radius                                     ing where the values RRoss and RLD are in the model that will
In a model stellar atmosphere, there is no “edge” that marks the        be clipped or by assuming that the point in the CLIV where
radius of the star and the transition to empty space. There are sev-    the derivative of the intensity with respect to µ is greatest.
eral ways to define the stellar radius (Baschek et al. 1991), and we    Aufdenberg et al. (2005) showed that this is approximately the
have chosen to use the Rosseland stellar radius, RRoss , defined as     point corresponding to RRoss .
the radius where the Rosseland optical depth, τRoss , has a value of         However, we reject this option because it removes informa-
two-thirds, because at that radius in the atmosphere the light has      tion about the stellar atmosphere and its radiation properties.
≈0.5 chance of escaping to space without being absorbed. How-           When we clip the CLIV, we remove information about atmo-
ever, there is still some radiation emitted by the star from above      spheric extension and make the CLIV more plane-parallel-like.
this level, and the structure of these levels and the radiation they    Furthermore, clipping the CLIV and rescaling the intensity pro-
emit are different for our spherical models compared to plane-          file will increase the moments of the intensity, in particular the
parallel models. Also, there are other definitions of the stellar       stellar flux. If the stellar flux is increased in a planetary transit
radius that are commonly used. One is the limb-darkened radius,         fit then the corresponding value of ρ will be smaller. As such,
RLD , derived from where the disk visibility observed using opti-       when one clips the CLIV to get a better fit, one creates both an
cal interferometry goes to zero (Wittkowski et al. 2004), though        inconsistency in the stellar models and biases the fit to smaller
it should be noted that interferometric visibilities are unreliable     values of ρ.
for visibilities less than 10−4 (Baron et al. 2014). To be clear,            Regardless of the method used to incorporate spherically
RLD is greater than RRoss . In the analysis to follow, we show          symmetric model stellar atmospheres into fits of transit light
that the exact definition of R is inconsequential because we are        curves, the results remain the same. One can either use model
comparing results found using the CLIV directly with results            knowledge of RRoss /RLD to improve the analysis, or one can
found using an LDL representation of the same CLIV, and the             continue to use geometrically-unrealistic models or models with
definition of R is essentially canceled out.                            inconsistent fluxes due to clipping that will bias any analysis. For
     In the next section, we explore how the representations of the     the sake of this work, the issue is not of consequence since we
CLIV differ as a function of stellar properties and inclination. As     show that the analysis is a relative comparison.
in Paper I, we define the inclination in terms of µ. The conven-
tional definition of the orbital inclination angle, i, is the angle     4. Measuring fitted limb-darkening law errors
between the orbit plane and the plane of the sky, so that i = 90◦ is
an orbit observed edge-on and i = 0◦ is an orbit observed face-on.      Neilson & Lester (2013a,b) found that the errors produced by fit-
     We define a new orbital inclination parameter:                     ting limb-darkening laws to spherically symmetric model stellar
                                                                        atmosphere CLIV varied as a function of atmospheric extension.
θ0 ≡ 90◦ − i,                                                    (3)    The extension can be represented as

and scaling
                                                                                                                       p
                                                                        Hp /R∗ ∝ T eff R∗ /M∗ = T eff /(gR∗ ) = T eff / gM∗          (5)
       a cos θ0                                                         (Baschek et al. 1991; Bessell et al. 1991; Neilson et al. 2016),
µ0 ≡            ,                                                (4)    where H p is the local pressure scale height. This extension, also
         RLD
                                                                        referred to as the stellar mass index (SMI) by Neilson et al.
where a/RLD is the normalized separation between the star and           (2016), is important because it indicates how the structure of
the planet. The purpose for these definitions is to allow a more        the CLIV changes near the edge of the stellar disk. Because the
direct connection between light curves as a function of inclina-        errors for fitting limb-darkening increase with increasing exten-
tion with CLIV and limb-darkening laws computed as a function           sion, we expect the average difference between synthetic light
of µ.                                                                   curves to also increase as a function of atmospheric extension.
     With the definition of ρ given in Eq. (1), we need to return           Before we explore the dependence of the limb darkening on
to the definition of the star’s radius. In particular, how do we        the parameterization of the atmospheric extension, we first con-
use the spherically symmetric model stellar atmosphere CLIV to          sider, for the case of edge-on inclination, i = 90◦ , how the average
fit planetary transit observations and measure the planet radius        differences change independently as functions of effective tem-
itself? We suggest two possibilities and reject a third.                perature, gravity, and stellar mass. Under these assumptions, we
     The first possible solution follows if one uses the spheri-        plotted the errors for the TESS- and K-bands, although we also
cal model CLIV or a limb-darkening law derived from fitting             computed these differences for BVRIH-bands. The results for
the spherical model CLIV. In either case, the approach is               these wavebands are similar to results for the TESS- and K-
to fit the observations and then multiply the measured value            bands. In Fig. 1, we plot the average flux difference between the
of ρ = Rp /RLD by the factor RLD /RRoss to transform ρ to the           CLIV and the best-fit quadratic limb-darkening law for an entire
Rosseland radius. Using the CLIV from the models makes                  transit and the greatest difference during the transit as a func-
RLD /RRoss readily available.                                           tion of effective temperature. It is notable that these differences
                                                                                                                           A38, page 3 of 10
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                                              A&A 662, A38 (2022)

                              0
                                                                             TESS                                       400
                             -50
                                                                                                                        300
                            -100




                                                                                            RMS (fCLIV - fLDL) (ppm)
                                                                                                                        200



    <fCLIV - fLDL> (ppm)
                            -150

                            -200                                                                                        100                                             TESS
                              0                                                                                         600
                                                                                   K                                    500
                            -100
                                                                                                                        400
                            -200                                                                                        300
                                                                                                                        200
                            -300
                                                                                                                        100                                                    K
                            -400                                                                                         0
                                   3.5   4   4.5   5   5.5 6       6.5   7   7.5       8                                      3.5   4   4.5   5   5.5 6       6.5   7    7.5       8
                                                       Teff (kK)                                                                                  Teff (kK)

Fig. 1. A comparison of predicted planetary transit light curves as a function of stellar effective temperature. Left: average differences between
synthetic planetary transit light curves computed using model stellar atmosphere CLIV and using best-fit quadratric limb-darkening laws as a
function of effective temperature for the TESS band (top) and K band (bottom). Right: same as the left panels, but for the RMS difference of the
light curves. For each effective temperature bin, there are models computed for four different gravities and five different masses.

                             -50
                                                                                                                        400                                             TESS
                            -100
                                                                                                                        300




                                                                                             RMS (fCLIV - fLDL) (ppm)
                            -150
                                                                                                                        200



     <fCLIV - fLDL> (ppm)
                            -200                                                                                        100
                                                                             TESS
                               0                                                                                        600
                                                                                                                        500                                                K
                            -100
                                                                                                                        400
                            -200                                                                                        300
                                                                                                                        200
                            -300
                                                                               K                                        100
                            -400                                                                                          0
                                         4         4.25            4.5             4.75                                             4         4.25            4.5              4.75
                                                          Log g                                                                                      Log g

Fig. 2. The differences between synthetic planetary transit light curve as a function of stellar gravity. Left: average differences between synthetic
planetary transit light curves computed using model stellar atmosphere CLIV and best-fit quadratric limb-darkening laws as a function of stellar
gravity for the TESS band (top) and K band (bottom). Right: same as the left panels, but for the RMS difference of the light curves. For each gravity
bin, there are models computed for 46 effective temperatures and five different masses.


tend toward greater values with increasing effective tempera-                              is disappointing, because the gravity-jitter relation (Bastien et al.
tures. Hence, hotter stars with transiting planets will have greater                       2013, 2014) would provide a quick and simple connection to the
systematic uncertainties, up to 400 ppm for the TESS band and                              errors if they were more sensitive to the surface gravity. Figure 3
600 ppm for the K band. This error in flux, ∆ f = fCLIV − fLDL , is                        plots the errors as a function of stellar mass, which is a compo-
also an error in the surface area of the planet relative to the star,                      nent of the surface gravity, showing that there is more of a trend,
which, for the small planet approximation is ρ2 = 0.01; hence,                             with the greatest differences occurring for the smallest stellar
the errors reach about 4 and 6% in the TESS and K bands,                                   masses.
respectively.                                                                                  The results of the three plots imply that the predicted errors
    The errors plotted in Fig. 1 do have a weak trend with effec-                          tend toward greater absolute values for hotter effective temper-
tive temperature, but there is an even more significant spread in                          atures and smaller masses, and they potentially depend on the
the errors, by as much as 200 ppm, at every effective temper-                              stellar gravity. To test this, we used the definition of atmospheric
ature. Because of this spread, we plot the errors as a function                            extension given in Eq. (5) expressed in solar units. In Fig. 4, we
of log g in Fig. 2. The errors essentially show no dependence                              plot the errors versus the atmospheric extension and find there
on surface gravity, with just a very slight increase for lower                             is a trend, though the range of atmospheric extensions is rela-
gravity model atmospheres. This weak dependence on gravity                                 tively small for these dwarf stars. Neilson et al. (2016) computed

A38, page 4 of 10
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                              H. R. Neilson et al.: Limb-darkening and planetary transits

                             -50
                                                                                                                              400                                              TESS
                            -100
                                                                                                                              300




                                                                                                   RMS (fCLIV - fLDL) (ppm)
                            -150
                                                                                                                              200



     <fCLIV - fLDL> (ppm)
                            -200                                              TESS                                            100

                               0                                                                                              600
                                                                                                                              500                                              K
                            -100
                                                                                                                              400
                            -200                                                                                              300
                                                                                                                              200
                            -300
                                                                              K                                               100
                            -400                                                                                                0
                                       0.2   0.4    0.6     0.8    1      1.2         1.4                                               0.2   0.4    0.6     0.8    1     1.2          1.4
                                                    Stellar Mass (M⊙)                                                                                Stellar Mass (M⊙)

Fig. 3. The differences between synthetic planetary transit light curve as a function of stellar mass. Left: average differences between synthetic
planetary transit light curves computed using model stellar atmosphere CLIV and using best-fit quadratric limb-darkening laws as a function of
stellar mass for the TESS band (top) and K band (bottom). Right: same as the left panels, but for the RMS difference of the light curves. For each
mass bin, there are models computed for four different gravities and 46 different effective temperatures.

                             -50
                                                                          TESS                                                400
                            -100
                                                                                                                              300




                                                                                                 RMS (fCLIV - fLDL) (ppm)
                            -150
                                                                                                                              200



   <fCLIV - fLDL> (ppm)
                            -200                                                                                              100                                          TESS
                              0                                                                                               600
                                                                                  K                                           500
                            -100
                                                                                                                              400
                            -200                                                                                              300
                                                                                                                              200
                            -300
                                                                                                                              100                                                  K
                            -400                                                                                               0
                                   0     1    2     3      4     5    6   7       8         9                                       0     1    2     3      4     5    6   7       8         9
                                              (Teff/Teff,⊙)(R✭/(R⊙)(/M⊙/M✭)                                                                    (Teff/Teff,⊙)(R✭//R⊙)(/M⊙/M✭)

Fig. 4. The differences between synthetic planetary transit light curve as a function of stellar atmospheric extension. Left: average differences
between synthetic planetary transit light curves computed using model stellar atmosphere CLIV and best-fit quadratric limb-darkening laws as a
function of atmospheric extension for the TESS band (top) and K band (bottom). Right: same as the left panels, but featuring the RMS difference
of the light curves. Points denoted by black circles are for model stellar atmospheres with T eff ≤ 3700 K. Given the model parameters, there are
twenty chains for the atmospheric extension due to the combination of four values of gravity and five values of mass.


atmospheric extensions for red giant and supergiant model stel-                                 greatest extension. The difference between planetary transit light
lar atmospheres that reach a few hundred R /M . In Fig. 4,                                      curves computed using model CLIV and those computed using
there appear to be two trends: one group has larger variability                                 best-fit limb-darkening coefficients is tracing the quality of the
and contains most of the models in the sample, and a second                                     fit of such coefficients.
group has few models and the smallest errors. That latter group                                      The greatest differences correspond to the greatest atmo-
corresponds to effective temperatures ≤3700 K, which likely cor-                                spheric extensions, hence the hottest and most evolved stars in
responds to a shift in the dominant opacities in the model stellar                              our sample with T eff → 8000 K and log g → 4.0. That is, the
atmospheres.                                                                                    greatest differences correspond to evolved main sequence F-type
    The key result from Fig. 4 is that the errors between the                                   stars. There have been numerous planet transit detections around
atmosphere’s actual CLIV and the limb-darkening law represen-                                   F-type stars (Gandolfi et al. 2012; Smalley et al. 2012; Bayliss
tation of this CLIV grows as a function of atmospheric extension.                               et al. 2013; Huang et al. 2015; Fukui et al. 2016), and many of
This result is consistent with the predictions of Neilson & Lester                              those exoplanets appear to be “bloated” hot Jupiters. Understand-
(2013b) that the best-fit limb-darkening coefficients fit the CLIV                              ing the biases introduced by assuming simple limb-darkening
of a model atmosphere most poorly when the models have the                                      laws could contribute to some of this bloating, especially since

                                                                                                                                                                          A38, page 5 of 10
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                                             A&A 662, A38 (2022)

                             0
                                                                            TESS                                     700
                           -100                                                                                                    TESS
                                                                                                                     600
                           -200
                                                                                                                     500
                           -300                                                                                      400




    <fCLIV - fLDL> (ppm)                                                                  RMS (fCLIV - fLDL) (ppm)
                           -400                                                                                      300
                           -500                                                                                      200
                           -600                                                                                      100
                              0
                                                                                  K                                  1000          K
                           -200                                                                                      800
                           -400                                                                                      600

                           -600                                                                                      400
                                                                                                                     200
                           -800
                                                                                                                         0
                                  3.5   4   4.5   5   5.5 6       6.5   7   7.5       8                                      3.5       4   4.5   5   5.5 6       6.5   7   7.5   8
                                                      Teff (kK)                                                                                      Teff (kK)

Fig. 5. The differences between synthetic planetary transit light curve as a function of stellar effective temperature for different transit inclinations.
Left: average differences between synthetic planetary transit light curves computed using model stellar atmosphere CLIV or best-fit quadratric
limb-darkening laws as a function of effective temperature for the TESS band (top) and K band (bottom). Right: same as the left panels, but for the
RMS of the light curves. The red crosses represent transits with µ0 = 1, blue stars those with µ0 = 0.7, and black open squares those with µ0 = 0.3.
The spread of the average differences and RMS values for each µ0 arises from variations in stellar mass and gravity for a given stellar effective
temperature.

we found in Paper I that those differences increase when we                                    In Fig. 6, we show the effect of atmospheric extension on the
consider orbits that are inclined edge on.                                                difference between the CLIV and the quadratic limb-darkening
                                                                                          law. The results are surprising. In Fig. 4, we see that the average
                                                                                          and maximum differences between synthetic planet transit light
5. Limb-darkening law errors as a function of
                                                                                          curves grow as a function of atmospheric extension. However, we
   orbital inclination                                                                    see that for more inclined orbits the average differences increase
In this section, we explore how the differences between synthetic                         rapidly as a function of atmospheric extension. When µ0 = 0.3,
planetary transit light curves computed using model stellar atmo-                         we find that the average differences reach almost −600 ppm and
sphere CLIV and those computed using best-fit limb-darkening                              −800 ppm in the TESS and K bands, respectively, with an atmo-
coefficients change as a function of orbital inclination. We repre-                       spheric extension of ≈2 R /M (as scaled relatively to the solar
sent the inclination using µ0 , defined in Eq. (4), with an edge-on                       effective temperature). This suggests that all stars with planets
orbit with µ0 = 1 and a face-on orbit with µ0 = 0. In Paper I, we                         orbiting in inclined orbits will have significant errors for even
found that the differences between light curves can increase with                         smaller atmospheric extensions.
increasing inclination until µ0 ≈ 0.3, which corresponds to θ0 ≈                               These results place distinct challenges on our understand-
70◦ , i ≈ 20◦ and impact parameter b ≈ 0.95, b ≡ (a/R∗ ) cos i,                           ing of planet transits and secondary effects such as oblateness,
where a/R∗ is the orbital separation relative to the radius of the                        rotation, and spots. For instance, for the case of KIC 8462852
star. As a result, for most orbits a change in inclination will lead                      (Boyajian et al. 2016) the transits are explained by large fam-
to greater errors, and the maximum differences between light                              ilies of orbiting comets (or dust clouds; Bodman et al. 2017).
curves depend on the inclination even more.                                               Because that analysis ignores limb darkening in fitting the fam-
    We test the role of orbital inclination by first plotting the                         ily of comets, if any of the orbits are inclined, the sizes required
errors as a function of effective temperature in Fig. 5. The errors                       for the comets will be significantly inaccurate. As such, our
due to assuming limb-darkening laws increase as a function of                             results show that we must treat limb darkening in planetary
orbital inclination. In the TESS band, the average difference                             transits with greater care and move from assuming the simple
between light curves shifts from about −300 ppm for µ0 = 1, up                            parameterizations to using more realistic models of CLIV.
to −400 ppm for µ0 = 0.7, and up to −600 ppm for µ0 = 0.3. In
the K band, the effect is even more significant, with differences                         6. Correcting for limb-darkening law bias in the
up to −700 ppm and −900 ppm for the hottest stars. As such,                                  planetary radius
we are seeing how connected and dependent the light curve and
the assumption of limb-darkening laws are to and on each other                            While the average flux difference offers one measure of the error
(Howarth 2011).                                                                           created by assuming a simple limb-darkening law, it does not
    The maximum differences also grow as a function of orbital                            offer a significant measurement of biases in the predicted plan-
inclination. As µ0 decreases from unity to zero, the maximum                              etary radius. To address this, we started by defining the χ2 from
difference reaches almost 1700 ppm and 2600 ppm in the TESS                               the transit model as
and K bands, respectively. Those differences correspond to about                                      fCLIV (ρ, z) − fLDL (ρ, z) 2 .
                                                                                                X
                                                                                          χ2 ≡                                                           (6)
                                                                                                                                
17 and 26% of the surface area of the assumed planet, and hence                                                      z
about 8.5 and 13% of the planet radius for δA = 2δRp . For more
inclined orbits, we find errors that are a significant fraction of the                    In Eq. (6), z is the projected separation between the center
relative planet size.                                                                     of the planet and the center of the star normalized by the
A38, page 6 of 10
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                                H. R. Neilson et al.: Limb-darkening and planetary transits

                              0
                                                                             TESS                                         700        TESS
                            -100
                                                                                                                          600
                            -200
                                                                                                                          500
                            -300                                                                                          400




     <fCLIV - fLDL> (ppm)                                                                      RMS (fCLIV - fLDL) (ppm)
                            -400                                                                                          300
                            -500                                                                                          200
                            -600                                                                                          100
                               0
                                                                                   K                                      1000       K
                            -200
                                                                                                                          800
                            -400                                                                                          600

                            -600                                                                                          400
                                                                                                                          200
                            -800
                                                                                                                            0
                                   0       1    2      3     4     5   6    7     8     9                                        0       1   2      3     4     5   6    7    8    9
                                                 (Teff/Teff,⊙)(R✭/R⊙)(M⊙/M✭)                                                                  (Teff/Teff,⊙)(R✭/R⊙)(M⊙/M✭)

Fig. 6. The differences between synthetic planetary transit light curve as a function of stellar atmospheric extension for different transit inclinations.
Left: average differences between synthetic planetary transit light curves computed using model stellar atmosphere CLIV or using best-fit quadratic
limb-darkening laws as a function of atmospheric extension for the TESS band (top) and K band (bottom). Right: same as the left panels, but for
the RMS difference of the light curves. The red crosses represent transits with µ0 = 1, blue stars those with µ0 = 0.3, and black open squares those
with µ0 = 0.7. For each inclination, we show twenty chains of atmospheric extension.

stellar radius. At the edge of the stellar disk, z = 1. This defini-                               We then minimized the χ2 -function, Eq. (7), with respect to
tion offers potential challenges for working with CLIV computed                                the radius, obtaining
for stellar models with atmospheric extension, which we discuss
in Sect. 7. We note this is assuming the small-planet approxima-                               dχ2    X                                  d fLDL
tion for ease, which will differ slightly from more exact methods.                                 =2    fCLIV (ρ, z) − fLDL (ρ + δρ, z)          = 0.                                 (10)
                                                                                               dρ                                            dρ
However, this analysis allowed us to probe the order of mag-                                          z
nitude of the predicted errors. Because both the CLIV and the
                                                                                               Again ignoring changes in I ∗ as a function of ρ, the derivative of
best-fit limb-darkening coefficients use the same planet radius
                                                                                               Eq. (8) gives
and inclination, this χ2 should ideally be a minimum. How-
ever, it is possible to ensure improvement by varying some of                                  d fLDL        I∗    2
the parameters. For instance, varying the limb-darkening coeffi-                                      = −2ρ     = − 1 − fLDL (ρ, z) .                                                  (11)
                                                                                                                                   
cients changes the predicted stellar flux, which will change the                                 dρ         4Ω     ρ
transit depth and, hence, will lead to a different value of the plan-
                                                                                               Using Eqs. (9) and (11) in Eq. (10) gives
etary radius. Yet, this change in limb-darkening coefficients will
produce compound errors in the way we understand the host star.                                X ("                                δρ
                                                                                                                                                     #
Similarly, varying the inclination creates biases in the measured                                    fCLIV (ρ, z) − fLDL (ρ, z) + 2 (1 − fLDL (ρ, z))
limb-darkening coefficients that alter a fit in the same direc-                                  z
                                                                                                                                   ρ
tion. For simplicity, we minimized the χ2 function using just the                              "
                                                                                                   2
                                                                                                                     #)
variation of the planetary radius.                                                               − (1 − fLDL (ρ, z)) = 0.
                                                                                                   ρ
    We started by perturbing the radius in the LDL light curve of
Eq. (6):                                                                                       Rearranging and solving for δρ/ρ leads to
            fCLIV (ρ, z) − fLDL (ρ + δρ, z) 2 .
      X
χ2 ∝                                                              (7)
                                           
                                                                                                      z ( fLDL − fCLIV )(1 − fLDL )
                                                                                                    P                             
                                                                                               δρ
                     z                                                                            =                                  .                                                 (12)
                                                                                                ρ           2 z (1 − fLDL )2
                                                                                                             P
Next, we assumed the small-planet approximation,
                                        I∗                                                     This shows again that the average difference in flux offers a
f (ρ, z) = 1 − ρ2                          ,                                            (8)    rough measure of the error of the fit that affects the predicted
                                       4Ω
                                                                                               depth of the transit and hence the measured planet radius.
is valid for both the CLIV and LDL light curves. In Eq. (8), 4Ω is                                  We note that Eq. (12) appears to be an explicit function of ρ
the stellar flux (and should not be confused with solid angle), and                            since fLDL and fCLIV themselves also depend on ρ. However, if
I ∗ is the mean flux blocked by the planet as it transits (Mandel                              we insert Eq. (8) into Eq. (12), then we find that δρ/ρ is a func-
& Agol 2002). For the purpose of this perturbation, we ignore                                  tion of I ∗ /4Ω only. The amount of flux blocked by the planet,
changes in I ∗ as a function of planet radius, as well as second-                              however, is implicitly dependent on the size of the planet, but to
order changes in δρ. Therefore,                                                                first order Eq. (12) is independent of planet radius. While one
                                                 I∗      δρ I ∗                                could measure these differences using fitting codes, this analysis
fLDL (ρ + δρ, z) = 1 − ρ2                            − 2 ρ2                                    illustrates how the relative planet radius depends on understand-
                                                4Ω        ρ 4Ω                                 ing the limb darkening. Furthermore, we note that this relation
                                                         δρ                                   implies that the result is independent of stellar radius in that we
                                       = fLDL (ρ, z) − 2     1 − fLDL (ρ, z) .          (9)
                                                                            
                                                          ρ                                    can replace δρ/ρ = δrp /rp .
                                                                                                                                                                         A38, page 7 of 10
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                                A&A 662, A38 (2022)

               5000                                                                         5000
               4500                                                                         4500
               4000                                                                         4000
               3500                                                                         3500
               3000                                                                         3000
               2500                                                                         2500



  δρ/ρ (ppm)                                                                   δρ/ρ (ppm)
               2000                                             TESS                        2000                                                TESS
               1500                                                                         1500
               7000                                                                         7000
               6000                                                                         6000
               5000                                                                         5000
               4000                                                                         4000
               3000                                                                         3000
               2000                                                   K                     2000                                                     K
               1000                                                                         1000
                      3.5   4   4.5   5   5.5 6       6.5   7   7.5       8                        0       1       2      3      4    5    6    7    8       9
                                          Teff (kK)                                                                 (Teff/Teff,⊙)(R✭//R⊙(/M⊙/M✭)

Fig. 7. Predicted overestimated value of planetary radius relative to the actual planet size as a function of (left) effective temperature and (right)
atmospheric extension for the TESS and K bands for an edge-on orbit. Points denoted by black circles in the right plot are for model stellar
atmospheres with T eff ≤ 3700 K. The difference shown here is solely caused by assuming a quadratic limb-darkening law. For the left plot, the
spread is due to differences in mass and gravity for a given effective temperature, and for the right-hand plot there are twenty chains caused by the
combination of models having four values of gravity and five values of mass.


    Figure 7 shows the relative biases to the planet’s radius                               140000
                                                                                                           TESS
assuming the quadratic limb-darkening law. This bias is the                                 120000
expected overestimation of the planet radius by fitting methods                             100000
that assume this limb-darkening law. We plot the difference of                               80000
the planet’s radius as a function of effective temperature and                               60000
atmospheric extension. These differences scale with approxi-                                 40000
mately the same behavior as the plots of RMS( fCLIV − fLDL ).

                                                                              δρ/ρ (ppm)
                                                                                             20000
Furthermore, the differences in the planet’s radius are signifi-
cantly greater than indicated by the average flux difference by                             160000
almost a factor of twenty in the TESS band and by more than a                               140000         K
factor of twenty in the K band. We see again that the bias of the                           120000
planet’s radius is also greatest for stars with the greatest effective                      100000
                                                                                             80000
temperature and atmospheric extensions.
                                                                                             60000
    As a result, we find that planetary radii can be overestimated
                                                                                             40000
by up to 7000 ppm in the near-IR and 5000 ppm in the optical                                 20000
and near-IR TESS bands. This overestimation is small relatively                                  0
to the assumed size of the planet, especially when the bias is                                         0       1     2      3     4     5   6    7       8       9
relative to the measured planet size. Assuming the small planet                                                       (Teff/Teff,⊙)(R✭/R⊙)(M⊙/M✭)
approximation, ρ = 0.1, and then δρ = 500 ppm and 700 ppm in
the optical and near-IR, respectively.                                        Fig. 8. Predicted overestimated value of planetary radius relative to
    The bias for the planetary radius increases with the incli-               the actual planet size as a function of atmospheric extension for dif-
nation of the orbit, similarly to that seen for the average flux              ferent inclinations in the TESS and K bands. The red crosses represent
difference. For instance, Fig. 8 shows that the bias increases                transits with µ0 = 1, blue stars represent those with µ0 = 0.7, and black
by an order of magnitude as µ0 → 0. For the most inclined                     open squares represent those with µ0 = 0.3. For each inclination, there
orbit, and assuming the best-fit limb-darkening coefficients for              are twenty chains of atmospheric extension that resulting from models
a quadratic limb-darkening law for each model, the radius bias                being computed for four values of gravity and five values of mass.
is about 10% of the actual planet radius for model stellar atmo-
spheres with the greatest atmospheric extension. This bias will               more free parameters to precisely measure limb darkening and its
be even greater if we assume even simpler limb-darkening laws,                impact on the exoplanet radius. However, our analysis and past
such as a linear law or a uniform-disk model (i.e., no limb                   works highlights the fact that the stellar CLIV cannot be accu-
darkening). Therefore, care must be taken to choose the appropri-             rately represented by simple functions (Neilson & Lester 2011,
ate limb-darkening parameterization or model when measuring                   2013a,b).
precision values of extrasolar planet radius.
    Out of curiosity, we also computed the error in the plane-                7. Atmospheric extension, stellar radius, and
tary radius if one assumes a star with no limb darkening, that
                                                                                 transits
is, a uniform-disk model. If one uses this method, the planet
radius will be underestimated instead of overestimated by about               The issue of understanding the stellar radius is important not
2 – 5%. This check is consistent with the need for more and                   just for planetary transit fits using spherical model atmospheres,

A38, page 8 of 10
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                         H. R. Neilson et al.: Limb-darkening and planetary transits

but also for fits using plane-parallel models and limb-darkening        between the CLIV and LDL transit light curves. These dif-
laws. The geometry of plane-parallel models contains no infor-          ferences were computed as a function of fundamental stellar
mation about the stellar radius. As such, it is merely assumed          parameters: effective temperature, gravity, and stellar mass,
that measurements of stellar radii from asteroseismology or stel-       along with the inclination of the orbit, which we parameterized
lar evolution models correspond to the stellar radii in planetary       as µ0 = cos(90◦ − i).
transit fits. Similarly, best-fit limb-darkening laws that are based         The results are striking. Before considering the role of incli-
on plane-parallel models or fit directly to observations make           nation, we found that the average differences between CLIV
the same assumption, and it is unclear whether this is true.            and limb-darkened transit light curves increased as a func-
Spherically symmetric model stellar atmospheres explicitly con-         tion of atmospheric extension, which implies that the average
tain information about the stellar radius, and therefore about          differences are greatest for more evolved F-type stars. When
the extension of the atmosphere. Because of the atmospheric             inclination is included, the differences increase significantly and
extension, the edge of the star is at a physical radius that is         depend on the atmospheric extension, indicating the errors are
not the Rosseland radius. As such, plane-parallel and spheri-           roughly similar for most atmospheric extensions. These negative
cal model stellar atmospheres should not be expected to give            errors tell us that the relative planetary radii are being overesti-
the same results when fit to planetary transit or interferometric       mated, especially for the F-stars, by as much as 5%, and by at
observations because they make different assumptions about the          least 1% for an edge-on orbit in the TESS band. Hirano et al.
structure of the photosphere. The challenges for measuring limb         (2016); Fukui et al. (2016) and others report precisions on the
darkening and stellar radii (or angular diameters) have been dis-       order of 1% for measuring Rp /R∗ for planets orbiting F-type
cussed in detail by numerous authors (Wittkowski et al. 2004;           stars. Almenara et al. (2015) reported precisions better than 1%
Neilson & Lester 2008; Baron et al. 2014; Kervella et al. 2017).        for planets orbiting an evolved metal-poor F-star. Given that our
     Just as the differences between plane-parallel and spherically     models show that Rp /R∗ are overestimated, these measurements
symmetric model stellar atmospheres lead to different measure-          have a systematic error of at least 1% that is not accounted for in
ments of stellar radii, they are also fit with varying precision        the fits.
by various limb-darkening laws. Neilson & Lester (2013a,b)                   We note that these errors are for the ideal situation where one
showed that six different commonly used limb-darkening laws             knows the inclination and where the limb-darkening coefficients
fit plane-parallel models with much better precision than spheri-       are the most accurately determined. Our analysis does not con-
cally symmetric models with the same fundamental parameters.            sider the cases where inclination, limb-darkening coefficients,
The source of this difference is the point of inflection in spher-      and relative radii are fit simultaneously. In those cases, limb-
ically symmetric CLIV that is a result of including physics of          darkening coefficient measurements can deviate significantly
atmospheric extension. Therefore, current limb-darkening laws           from those of model stellar atmospheres (Kipping & Bakos
do not fit the effects of atmospheric extension. This result was        2011a,b), implying a strong dependence of the limb darkening
found by other works such as Claret & Hauschildt (2003) and             on other fitting parameters. As the limb-darkening coefficients
Espinoza & Jordán (2016). However, these works avoid the chal-          deviate, so will the errors. This may not change the errors much,
lenge of fitting atmospheric extension by clipping the spherically      but it is something that must be explored in greater detail.
symmetric model CLIV to remove all information about the                     One key conclusion of our work is that we need to mea-
extension.                                                              sure stellar CLIV both precisely and directly. It is becoming
     Ligi et al. (2016) found uncertainties in measuring angu-          clear that our current assumptions of simple limb-darkening laws
lar diameters of exoplanet host stars to be about 1.9%. This            are just not good enough for understanding the planetary tran-
is much greater than the atmospheric extension of these stars.          sit observations. Interferometry is proving to be one method for
For instance, the Sun has an extension on the order of 0.1%             directly inferring stellar CLIV (Baron et al. 2014; Armstrong
based on the ratio of the pressure scale height in the atmo-            et al. 2016; Kervella et al. 2017). We recently showed that we
sphere and the solar radius. Therefore, these issues around the         can use interferometric measurements in combination with spec-
definition of stellar radius will not be readily apparent for direct    troscopy and spherically symmetric model stellar atmospheres to
measurements. On the other hand, Mann et al. (2018) measured            measure stellar fundamental parameters including stellar masses.
the relative planet radii for three exoplanets to a precision of        That result is based on measurements of atmospheric extension
δρ/ρ ≈ 2−4%, while Murgas et al. (2017) reported precisions             in stars. We suggest that method will be more robust if com-
on the order of 1% and better. Furthermore, the next genera-            bined with planetary transit observations as part of a global fit
tion of interferometric observations promises to measure angular        of stellar and planetary parameters. That work and this is part
diameters to about 0.5% precision (Zhao et al. 2011). At these          of an ongoing research project to test limb-darkening and stel-
uncertainties, the biases introduced by assuming the unphysical         lar radii measurements from interferometric observations against
limb-darkening laws are becoming significant, especially as we          state-of-the-art model stellar atmospheres. However, the results
attempt to measure spectral properties of exoplanets.                   of our current work clearly show that we are reaching the limits
                                                                        of plane-parallel model CLIV and arbitrary limb-darkening laws
                                                                        that have no physics basis.
8. Summary                                                                   From this analysis, we predicted biases of the relative radius
                                                                        of an exoplanet δρ/ρ for a grid of stellar atmosphere models
In this work, we took the CLIVs from the Neilson & Lester               for the wavebands BVRIHK and the TESS bands that are pub-
(2013b) grid of spherically symmetric model stellar atmospheres         licly available. While it is preferable to fit the model CLIV to
and the corresponding best-fit limb-darkening coefficients for          transit light curves and to shift from measuring limb-darkening
the quadratic limb-darkening law and computed the differences           coefficients to measuring stellar properties, these biases can help
between synthetic planetary transit light curves using the pre-         improve the precision of planetary transit fits of transit spectra.
scription described by Neilson et al. (2017). We evaluated the
error resulting from the use of the limb-darkening laws by com-         Acknowledgements. J.B.L. is grateful for funding from NSERC discovery grants.
puting both the average difference and the greatest difference          F.B. acknowledges funding from NSF awards AST-1445935 and AST-1616483.

                                                                                                                                 A38, page 9 of 10
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                                      A&A 662, A38 (2022)

H.R.N. and J.B.L. acknowledge that the University of Toronto operates on tra-        Gillon, M., Demory, B.-O., Madhusudhan, N., et al. 2014, A&A, 563, A21
ditional land of the Wendat, the Seneca, and most recently, the Mississaugas of      Hirano, T., Fukui, A., Mann, A. W., et al. 2016, ApJ, 820, 41
the Credit River. The authors are grateful to have the opportunity to work on this   Howarth, I. D. 2011, MNRAS, 418, 1165
land.                                                                                Huang, C. X., Hartman, J. D., Bakos, G. Á., et al. 2015, AJ, 150, 85
                                                                                     Kervella, P., Bigot, L., Gallenne, A., & Thévenin, F. 2017, A&A, 597, A137
                                                                                     Kipping, D., & Bakos, G. 2011a, ApJ, 730, 50
References                                                                           Kipping, D., & Bakos, G. 2011b, ApJ, 733, 36
                                                                                     Lester, J. B., & Neilson, H. R. 2008, A&A, 491, 633
Almenara, J. M., Díaz, R. F., Mardling, R., et al. 2015, MNRAS, 453, 2644            Ligi, R., Creevey, O., Mourard, D., et al. 2016, A&A, 586, A94
Armstrong, J. T., Baines, E. K., Schmitt, H. R., et al. 2016, Proc. SPIE, 9907,      Mandel, K., & Agol, E. 2002, ApJ, 580, L171
   990702                                                                            Mann, A. W., Vanderburg, A., Rizzuto, A. C., et al. 2018, AJ, 155, 4
Aufdenberg, J. P., Ludwig, H.-G., & Kervella, P. 2005, ApJ, 633, 424                 Morello, G., Tsiaras, A., Howarth, I. D., & Homeier, D. 2017, AJ, 154, 111
Baron, F., Monnier, J. D., Kiss, L. L., et al. 2014, ApJ, 785, 46                    Murgas, F., Pallé, E., Parviainen, H., et al. 2017, A&A, 605, A114
Baschek, B., Scholz, M., & Wehrse, R. 1991, A&A, 246, 374                            Neilson, H. R., & Lester, J. B. 2008, A&A, 490, 807
Bastien, F. A., Stassun, K. G., Basri, G., & Pepper, J. 2013, Nature, 500, 427       Neilson, H. R., & Lester, J. B. 2011, A&A, 530, A65
Bastien, F. A., Stassun, K. G., & Pepper, J. 2014, ApJ, 788, L9                      Neilson, H. R., & Lester, J. B. 2013a, A&A, 554, A98
Bayliss, D., Zhou, G., Penev, K., et al. 2013, AJ, 146, 113                          Neilson, H. R., & Lester, J. B. 2013b, A&A, 556, A86
Bean, J. L., Désert, J.-M., Kabath, P., et al. 2011, ApJ, 743, 92                    Neilson, H. R., Baron, F., Norris, R., Kloppenborg, B., & Lester, J. B. 2016, ApJ,
Berta, Z. K., Charbonneau, D., Désert, J.-M., et al. 2012, ApJ, 747, 35                 830, 103
Bessell, M. S., Brett, J. M., Scholz, M., & Wood, P. R. 1991, A&AS, 89, 335          Neilson, H. R., McNeil, J. T., Ignace, R., & Lester, J. B. 2017, ApJ, 845, 65
Bodman, E. H. L., Quillen, A. C., Ansdell, M., et al. 2017, MNRAS, 470, 202          Rauer, H., Catala, C., Aerts, C., et al. 2014, Exp. Astron., 38, 249
Boyajian, T. S., LaCourse, D. M., Rappaport, S. A., et al. 2016, MNRAS, 457,         Ricker, G. R., Winn, J. N., Vanderspek, R., et al. 2015, J. Astron. Teles. Instrum.,
   3988                                                                                 Syst., 1, 014003
Cáceres, C., Kabath, P., Hoyer, S., et al. 2014, A&A, 565, A7                        Ricker, G. R., Vanderspek, R., Winn, J., et al. 2016, SPIE Conf. Ser., 9904,
Claret, A. 2000, A&A, 363, 1081                                                         99042B
Claret, A. 2017, A&A, 600, A30                                                       Seager, S., & Deming, D. 2010, ARA&A, 48, 631
Claret, A., & Hauschildt, P. H. 2003, A&A, 412, 241                                  Sing, D. K. 2010, A&A, 510, A21
Croll, B., Albert, L., Jayawardhana, R., et al. 2011, ApJ, 736, 78                   Smalley, B., Anderson, D. R., Collier-Cameron, A., et al. 2012, A&A, 547,
Espinoza, N., & Jordán, A. 2016, MNRAS, 457, 3573                                       A61
Fukui, A., Narita, N., Kawashima, Y., et al. 2016, ApJ, 819, 27                      Wittkowski, M., Aufdenberg, J. P., & Kervella, P. 2004, A&A, 413, 711
Gandolfi, D., Collier Cameron, A., Endl, M., et al. 2012, A&A, 543, L5               Zhao, M., Monnier, J. D., Che, X., et al. 2011, PASP, 123, 964




A38, page 10 of 10
```
