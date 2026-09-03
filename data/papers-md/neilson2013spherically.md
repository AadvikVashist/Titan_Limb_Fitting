---
citation_key: "neilson2013spherically"
title: "Spherically-symmetric model stellar atmospheres and limb darkening-I. Limb-darkening laws, gravity-darkening coefficients and angular diameter corrections for red giant stars"
source_pdf: "data/papers/neilson2013spherically.pdf"
source_pdf_sha256: "b714a772578b5a741dcb4ffff735220056d20844c8211d3f5a0de1d32e01c5d4"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
A&A 554, A98 (2013)                                                                                                  Astronomy
DOI: 10.1051/0004-6361/201321502                                                                                      &
 ESO 2013
c                                                                                                                    Astrophysics


                   Spherically-symmetric model stellar atmospheres
                                  and limb darkening
    I. Limb-darkening laws, gravity-darkening coefficients and angular diameter
                         corrections for red giant stars
                                                     H. R. Neilson1 and J. B. Lester2,3

      1
        Department of Physics & Astronomy, East Tennessee State University, Box 70652, Johnson City, TN 37614, USA
        e-mail: neilsonh@etsu.edu
      2
        Department of Chemical and Physical Sciences, University of Toronto Mississauga, ON L5L 1C6, Canada
      3
        Department of Astronomy & Astrophysics, University of Toronto, ON M55, 3H4, Canada
        e-mail: lester@astro.utoronto.ca
      Received 18 March 2013 / Accepted 30 April 2013
                                                                   ABSTRACT

      Model stellar atmospheres are fundamental tools for understanding stellar observations from interferometry, microlensing, eclipsing
      binaries and planetary transits. However, the calculations also include assumptions, such as the geometry of the model. We use
      intensity profiles computed for both plane-parallel and spherically symmetric model atmospheres to determine fitting coeﬃcients
      in the BVRIHK, CoRot and Kepler wavebands for limb darkening using several diﬀerent fitting laws, for gravity-darkening and for
      interferometric angular diameter corrections. Comparing predicted variables for each geometry, we find that the spherically symmetric
      model geometry leads to diﬀerent predictions for surface gravities log g < 3. In particular, the most commonly used limb-darkening
      laws produce poor fits to the intensity profiles of spherically symmetric model atmospheres, which indicates the need for more
      sophisticated laws. Angular diameter corrections for spherically symmetric models range from 0.67 to 1, compared to the much
      smaller range from 0.95 to 1 for plane-parallel models.
      Key words. stars: atmospheres – stars: late-type – binaries: eclipsing – stars: evolution – techniques: interferometric


1. Introduction                                                                 Microlensing observations, like interferometry, also probe
                                                                           stellar limb darkening, but unlike interferometry, which tar-
Stellar limb darkening is an important tool for interpreting in-           gets specific nearby stars, microlensing observations are ran-
terferometric, microlensing and eclipsing binary observations of           dom. An et al. (2002) and Fields et al. (2003) constrained
red giant and supergiant stars. It also provides critical infor-           non-linear limb-darkening relations from microlensing obser-
mation about the temperature structure of a stellar atmosphere             vations of a K3 giant and compared them to model stellar at-
(Schwarzschild 1906) as well as a measure of the radial exten-             mospheres. They found significant disagreement between the
sion of an atmosphere (Neilson & Lester 2012).                             observed and predicted limb-darkening relation. More recently,
    Interferometric observations measure the angular diameter              however, microlensing observations have only constrained linear
of a star as well as the intensity variation across the stellar sur-       limb-darkening relations for red giant stars (Fouqué et al. 2010;
face. Some of the first interferometric observations measured              Zub et al. 2011).
only uniform-disk angular diameters, that is the angular diam-                  Eclipsing binaries and planetary transits provide yet another
eter for a star assumed to have a constant surface brightness              avenue for measuring stellar limb darkening. In terms of red
(Hanbury Brown et al. 1974). Wittkowski et al. (2004) presented            giant stars, there are a number of known eclipsing binary sys-
K-band interferometric observations of the M3 giant ψ Phoenicis            tems, specifically the ζ Aurigae systems that have a K4-5 red
with measurements of the first and second lobes of the visibility          giant primary and a main-sequence B-type companion. Eaton
curve, which constrain limb darkening. Unfortunately, these ob-            et al. (2008) fit the orbits for several of these systems assum-
servations were not precise enough to distinguish between dif-             ing a simple linear limb-darkening law. There is also the poten-
ferent model stellar atmospheres. Advances in interferometric              tial of observing planets transiting red giant stars, which would
observations have allowed for observations of convective cells             provide powerful constraints of theories of planetary evolution.
in Betelgeuse (Haubois et al. 2009) and measurements of grav-              Currently, extrasolar planets have been observed orbiting dwarf
ity darkening in Altair (van Belle et al. 2001). In terms of model         and subgiant stars (Howell et al. 2012), but not giant stars; future
stellar atmospheres, Aufdenberg et al. (2005) constrained three-           missions such as PLATO may remedy this (Catala et al. 2010).
dimensional models using observations of Procyon.                               These three types of observations are ideal tools for prob-
                                                                           ing stellar atmospheres and constraining the physics employed in
  
    Tables 2–17, and the model intensity profiles are only available at    numerical models. Likewise, predictions from model stellar at-
the CDS via anonymous ftp to                                               mospheres help constrain these types of observations. Recently,
cdsarc.u-strasbg.fr (130.79.128.5) or via                                  Sing (2010), Howarth (2011a) and Claret & Bloemen (2011)
http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/554/A98                      presented limb-darkening laws fit to plane-parallel model stellar
                                           Article published by EDP Sciences                                                    A98, page 1 of 10
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                         A&A 554, A98 (2013)

atmosphere intensity profiles. Even more recently, Claret et al.      model at 1000 equally spaced values of μ = cos θ, where θ is
(2012, 2013) fit limb-darkening laws to spherically-symmetric         the angle between the vertical direction and the direction to-
PHOENIX model stellar atmospheres of cool brown dwarf stars.          ward a distant observer. Limb-darkening profiles are computed
In this work, we study how the assumed geometry of the model          for Johnson-Cousins BVRIHK-wavebands (Johnson & Morgan
stellar atmosphere, plane parallel versus spherically symmetric,      1953; Bessell 2005) along with the CoRot (Auvergne et al. 2009)
aﬀects predictions of stellar limb darkening, gravity darkening       and Kepler (Koch et al. 2004) wavebands. Angular diameter cor-
and interferometric angular diameter corrections. We examine          rections for interferometric observations, gravity-darkening co-
model atmospheres spanning the eﬀective temperature and grav-         eﬃcients and various limb-darkening relations are computed us-
ity range consistent with yellow and red giant and supergiant         ing these wavelength-integrated intensity profiles.
stars. Tables of limb-darkening and gravity-darkening coeﬃ-
cients, as well as new angular diameter corrections are presented
as more physically based tools for understanding these bright         3. Limb-darkening laws
stars.                                                                An understanding of stellar limb darkening is required to
    In Sect. 2 we describe the stellar atmosphere code used in        model the properties of interferometric, eclipsing binary-star,
this work, as well as the model atmosphere grids computed             microlensing, and planetary-transit observations. As these ob-
for both plane-parallel and spherically symmetric geometries.         servations become more precise and more accurate, models of
In Sect. 3 limb-darkening coeﬃcients are presented for several        stellar limb darkening must also improve. Limb darkening is typ-
commonly used limb-darkening relations. We compute gravity-           ically treated as a simple parametrization as a function of θ (e.g.
darkening coeﬃcients in Sect. 4 and angular diameter correc-          Fouqué et al. 2010; Croll et al. 2011), which makes fitting the
tions in Sect. 5. Computations in these three sections provide        stellar intensity profile much simpler and reduces the number
insight into how intensity profiles depend on the assumed model       of free parameters. The most common parametrizations are lin-
geometry that can be directly compared to observations.               ear and quadratic relations (Al-Naimiy 1978; van Hamme 1993;
                                                                      Diaz-Cordoves et al. 1995), but other suggested relations include
2. Model stellar atmospheres                                          a four-parameter relation (Claret 2000a), a square-root relation
                                                                      (Wade & Rucinski 1985) as well as exponential and logarithmic
Model stellar atmospheres form a key foundation of our un-            relations (Claret 2000a; Claret & Hauschildt 2003).
derstanding of stars, arguably a great success of computational
astrophysics. However, the early success of model atmosphere
codes transformed them into standard tools, and only in the past      3.1. Best-fit limb-darkening laws
decade have these codes moved beyond simple plane-parallel,           We fit the following limb-darkening relations to the grids
local-thermodynamic-equilibrium (LTE) models to full three-           of plane-parallel and spherically symmetric model stellar
dimensional, statistical-equilibrium codes that can model non-        atmospheres:
LTE physics as well as stellar convection. Unfortunately, com-
puting power is still limited for calculating large-scale model         I(μ)
atmosphere grids varying stellar gravity, eﬀective temperature,                = 1 − u(1 − μ)                             Linear,    (1)
                                                                      I(μ = 1)
stellar mass and composition.
                                                                        I(μ)
     A step toward more realistic geometry is achieved by shifting             = 1 − a(1 − μ) − b(1 − μ)2             Quadratic,     (2)
from one-dimensional plane-parallel model stellar atmosphere          I(μ = 1)
codes to one-dimensional spherically symmetric codes, which             I(μ)                               √
                                                                               = 1 − c(1 − μ) − d(1 − μ)           Square-Root,      (3)
can be used to compute large grids of models atmospheres that         I(μ = 1)
include physics that is more appropriate to stars where the depth                     4
of the stellar photosphere is a significant fraction of the stellar     I(μ)
                                                                               =1−        f j (1 − μ j/2 )          4-Parameter,     (4)
radius, such as evolved giant and supergiant stars and pre-main       I(μ = 1)        j=1
sequence stars. One such code for modeling atmospheres assum-
ing spherically symmetric geometry is the SAtlas code (Lester           I(μ)
                                                                               = 1 − g(1 − μ) − h
                                                                                                    1
                                                                                                                    Exponential,     (5)
& Neilson 2008). This code is based on the ATLAS code devel-          I(μ = 1)                    1 − eμ
oped by Kurucz (1979), and continues its assumption of local            I(μ)
thermodynamic and hydrostatic equilibrium. However, the ra-                    = 1 − m(1 − μ) − nμ ln μ             Logarithmic.     (6)
                                                                      I(μ = 1)
diative transfer is computed assuming spherical geometry using
the Rybicki (1971) version of the Feautrier (1964) ray-tracing        We derive the best-fit coeﬃcients for each of the limb-darkening
method, while radiative and convecting equilibrium is enforced        laws using a general least-squares algorithm. This was done
using an updated version of the Avrett & Krook (1963) tem-            using the computed surface intensities for the BVRIHK- and
perature correction method. Models computed using this code           CoRot- and Kepler-wavebands. Figure 1 shows the Kepler-band
have been compared to spherically-symmetric Phoenix and               intensity profile and corresponding best-fit limb-darkening laws
MARCS models (Hauschildt et al. 1999; Gustafsson et al. 2008)         for both spherical and plane-parallel model atmospheres with the
and shown to produce similar results (Lester & Neilson 2008;          properties T eﬀ = 5000 K, log g = 2 and M = 10 M (mass is de-
Neilson & Lester 2008).                                               fined for the spherical model only). The chosen limb-darkening
     In this work we use the grid of spherical model atmospheres      laws all fit the plane-parallel model intensity profiles well. This
from Neilson & Lester (2011), extended in mass up to M =              is not surprising because plane-parallel model atmosphere inten-
20 M . The grid assumes solar composition and spans the grav-        sity profiles do not deviate significantly from being linear, and a
ities from log g = −1 to log g = 3 in steps of 0.25, eﬀective tem-    linear term is included in all of the chosen limb-darkening laws.
peratures from T eﬀ = 3000 to 8000 K and masses from M = 2.5          However, spherically-symmetric model stellar atmospheres have
to 20 M in steps of 2.5 M and includes models with masses           intensity profiles that are significantly non-linear, and the best-
M = 0.5 and 1 M . Surface intensities are computed for each          fit limb-darkening relations for these intensity profiles match
A98, page 2 of 10
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                  H. R. Neilson and J. B. Lester: Limb darkening in red giant stars

                                   1                                                                     1
                                 0.8                                                                   0.8



              Kepler I(μ)/I(1)                                                      Kepler I(μ)/I(1)
                                 0.6                                                                   0.6
                                 0.4                                                                   0.4
                                 0.2                                                                   0.2
                                   0                                                                     0
                                 0.1                                                                   0.1

              Δ                    0                                               Δ                     0

                                 -0.1                                                                  -0.1
                                        1   0.8   0.6       0.4     0.2       0                               1   0.8   0.6       0.4   0.2       0
                                                        μ                                                                     μ
Fig. 1. Kepler-band model intensity profiles (black-solid) predicted for both plane-parallel (left) and spherically symmetric (right) model stellar
atmospheres with T eﬀ = 5000 K, log g = 2 and M = 10 M . Along with the intensity profiles, best-fit linear (green-dashed), quadratic (orange-
short-dashed), square-root (blue-dotted), four-parameter (violet-long-dash-dotted), logarithmic (brown-short-dash-dotted), and exponential (grey-
double-dash) limb-darkening relations are plotted. Bottom panels show the diﬀerence, Δ ≡ Imodel − Ilaw , between model intensities and best-fit
limb-darkening laws.


                    1.4                                                             eﬀective temperatures and gravities vary from u = 0.6 to 1.4.
                                                                                    The coeﬃcients predicted for almost all limb-darkening laws
                    1.2                                                             examined here show the same behavior as the limb-darkening
                      1                                                             coeﬃcients predicted for a flux-conserving linear+square-root
    uKepler         0.8                                                             law (Neilson & Lester 2011, 2012). This uniform dependence
                                                                                    of the coeﬃcients on T eﬀ is surprising and suggests all of these
                    0.6                                                             laws carry essentially the same information regarding the mo-
                    0.4                                                             ments of the intensity and the atmospheric extension about the
                                                                                    stellar atmosphere in question. The one exception is the Claret
                    0.2                                                             (2000a) four-parameter limb-darkening law, for which the co-
                                                                                    eﬃcients appear to vary much more as a function of eﬀective
                                 3000 4000 5000 6000 7000 8000
                                                                                    temperature.
                                            Teff (K)                                    To explore the interdependence of the coeﬃcients, we plot
                                                                                    in Fig. 6 the Kepler-band b-coeﬃcient from the quadratic law as
Fig. 2. Limb-darkening coeﬃcient u, used in Eq. (1), applied to the                 a function of the a-coeﬃcient. This plot is typical of all the two-
Kepler photometric band. Red crosses are the plane-parallel model stel-             parameter limb-darkening laws considered in this work as well
lar atmospheres, and the blue squares are the spherical models.                     as the limb-darkening law employed by Neilson & Lester (2011,
                                                                                    2012), including the apparent hook in the correlation between
                                                                                    coeﬃcients. Figure 6 also plots the values of f2 + f4 as a func-
less well than for the plane-parallel models because of this non-                   tion of f1 + f3 for the four-parameter law, again for the Kepler
linearity. For the model shown in Fig. 1, limb-darkening laws                       photometric band. The correlation for both plane-parallel and
predict intensities that vary by Δ ≡ Imodel − Ilaw = 0.15 for the                   spherical models is readily apparent. A best-fit linear relation to
spherical model while Δ < 0.04 for the plane-parallel model.                        the coeﬃcients for spherical models is
Although limb-darkening laws fit the intensities of plane-parallel
model atmospheres better than spherically symmetric models,                          f2,Kepler + f4,Kepler = −0.989( f1,Kepler + f3,Kepler ) + 1.051.      (7)
the spherical models are more physically realistic, making them                     The correlation is diﬀerent for plane-parallel models for which
the more appropriate choice to use in modeling observations. We                     the slope is −0.978 and the intercept is 0.493.
explore the uncertainty of the limb-darkening fits later.                               These correlations are caused by the limb-darkening coeﬃ-
    We present in Figs. 2–5 the coeﬃcients derived by least-                        cients being linear combinations of various angular moments of
squares fitting for the limb-darkening laws given by Eqs. (1)–(6)                   the intensity. For instance,
                                                                                                                in plane-parallel
                                                                                                                                  model atmospheres
respectively for the Kepler photometric band as a function of ef-                   the moments J ≡ I(μ)dμ and K ≡ I(μ)μ2 dμ are related such
fective temperature for both plane-parallel and spherically sym-                    that J = 3K (Mihalas 1978). In spherical symmetry, this ratio
metric model stellar atmospheres. It is clear that more realistic                   varies, causing the moments of the intensity to diﬀer in spheri-
spherically symmetric model stellar atmospheres predict limb-                       cal symmetry from those predicted moments for plane-parallel
darkening coeﬃcients that vary much more as a function of ef-                       model stellar atmospheres. This diﬀerence in geometry is re-
fective temperature than those for plane-parallel models. For the                   flected in the diﬀerence between the zero-points of the relation
simplest case of the linear limb-darkening law, the u-coeﬃcient                     Eq. (7) for spherical models and that for plane-parallel models.
determined from plane-parallel models in the Kepler-band vary                       One can potentially use this diﬀerence to test observations and
from u = 0.2 to 0.5, whereas spherical models with the same                         test model geometry.
                                                                                                                                              A98, page 3 of 10
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                              A&A 554, A98 (2013)

                        3                                                                 4
                      2.5                                                                 3
                        2
                                                                                          2
          aKepler                                                              cKepler
                      1.5
                        1                                                                 1
                      0.5
                                                                                          0
                        0
                     -0.5                                                                -1
                      1.5
                                                                                          2
                        1
                      0.5                                                                 1

          bKepler                                                              dKepler
                        0                                                                 0
                     -0.5                                                                -1
                       -1                                                                -2
                     -1.5                                                                -3
                       -2                                                                -4
                            3000   4000   5000 6000       7000     8000                       3000     4000   5000 6000    7000    8000
                                            Teff (K)                                                            Teff (K)

Fig. 3. Limb-darkening coeﬃcients a and b used in Eq. (2) (left panel), and the coeﬃcients c and d used in Eq. (3) (right panel), all applied to the
Kepler photometric band. The symbols have the same meanings as in Fig. 2.


                      1.6                                                                 1.4
                      1.4                                                                 1.2
                      1.2                                                                   1

          gKepler                                                           mKepler
                        1                                                                 0.8
                      0.8                                                                 0.6
                      0.6                                                                 0.4
                      0.4
                                                                                          0.2
                      0.2
                                                                                            0
                    0.001                                                                   1
                                                                                          0.5

          hKepler                                                           nKepler
                                                                                            0
                       0
                                                                                         -0.5
                                                                                           -1
                -0.001                                                                   -1.5
                                                                                           -2
                            3000   4000   5000 6000      7000    8000                           3000   4000   5000 6000    7000    8000
                                            Teff (K)                                                            Teff (K)

Fig. 4. Limb-darkening coeﬃcients g and h used in Eq. (5) (left panel), and the coeﬃcients m and n used in Eq. (6) (right panel), all applied to the
Kepler photometric band. The symbols have the same meanings as in Fig. 2.


3.2. Error analysis                                                        which quantifies the deviation of the best-fit limb-darkening law
                                                                           from the surface intensities of the model atmosphere. We com-
Various limb-darkening laws, such as those given in Eqs. (1)–(6),          pute the relative error for each bandpass as a function of the fun-
are fit to the surface intensities computed with model stellar at-         damental stellar parameters for both plane-parallel and spherical
mospheres, and it is important to understand how well these laws           geometries, and show in Fig. 7 the relative errors as a function
represent the actual intensities. For instance, Diaz-Cordoves              of eﬀective temperature for fits in the Kepler-band. The relative
et al. (1995) argued that a square-root law fit intensity profiles for     error of the fits for spherical models is greater than the error
hotter stars (T eﬀ > 9000 K) better than a quadratic law, whereas          for plane-parallel fits for all the limb-darkening laws. The errors
no limb-darkening law is preferred for cooler stars. We compute            are similar only for T eﬀ ∼ 3500 K, where the spherical model
the relative error of the limb-darkening fit, Δ, using the relation        atmospheres predict intensity profiles that are closest to being
                                                                           linear, with the error of the linear limb-darkening law approach-
                                                                         ing a minimum value. This result appears to suggest that these
       [Imodel (μ) − Ifit (μ)]2                                            limb-darkening laws are inappropriate for fitting light curves and
Δλ ≡                           ,                                   (8)    interferometric observations, but this is not true.
             [Ifit (μ)]2
A98, page 4 of 10
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                  H. R. Neilson and J. B. Lester: Limb darkening in red giant stars

                              8                                                                   40

                              4                                                                   20

                f1, Kepler    0                                                      f3, Kepler    0

                              -4                                                                  -20

                              -8                                                                  -40
                                                                                                   20
                             20
                             10                                                                   10

           f2, Kepler                                                                f4, Kepler
                              0
                                                                                                   0
                             -10
                             -20                                                                  -10
                             -30
                                                                                                  -20
                                   3000   4000   5000 6000       7000     8000                          3000   4000   5000 6000    7000    8000
                                                   Teff (K)                                                             Teff (K)

Fig. 5. Limb-darkening coeﬃcients f1 , f2 , f3 and f4 used in the Claret (2000a) four-parameter law, Eq. (4), applied to the Kepler photometric band.
The symbols are the same as in Fig. 2.

                               3                                                                   40
                             2.5                                                                   30
                                                                                                   20
                               2
                                                                                                   10
                                                                                   f2 + f4
                             1.5                                                                    0
         a
                               1                                                                  -10
                                                                                                  -20
                             0.5                                                                  -30
                               0                                                                  -40
                         -0.5                                                                     -50
                                   -2 -1.5 -1 -0.5       0   0.5     1    1.5                        -40 -30 -20 -10 0 10 20 30 40 50
                                                     b                                                               f1 + f3
Fig. 6. Correlation between the Kepler-band limb-darkening coeﬃcients for the quadratic law (left) and for the four-parameter law, f2 + f4 as a
function of f1 + f3 (right). Red crosses represent coeﬃcients computed from plane-parallel models and blue squares spherical models.


    There are a number of issues with how the relative error is                          intensity profile. For example, the linear limb-darkening co-
computed and what the error tells us, such as how the limb-                              eﬃcient from Eq. (1) is a function of the mean intensity, J,
darkening laws are defined, how they are fit to the surface in-                          and the stellar flux, H ≡ I(μ)μdμ, and both of these quan-
tensities and the eﬀect of sampling.                                                     tities are more sensitive to the central intensity than to the
                                                                                         much smaller intensity near the limb. As with the defini-
 – Defining limb-darkening laws: The intensity profiles com-                             tion of the limb-darkening laws, using a least square fit-
   puted using the plane-parallel and spherical model atmo-                              ting algorithm fits the central part of the intensity structure
   spheres employed in this work are normalized with respect                             better. Similarly, one might fit limb-darkening coeﬃcients
   to the central intensity so that I(μ = 1) ≡ 1. Furthermore, all                       by enforcing flux conservation, but because the flux is the
   limb-darkening laws, except the exponential law, are defined                          μ-weighted integral of the intensity, any flux-conserving fit
   so that the I(μ = 1) ≡ 1, regardless of the values of the best-                       is constrained weakly by the intensity at the stellar limb rel-
   fit coeﬃcients. As a result, every fit to an intensity profile is                     ative to the intensity near the center of the stellar disk.
   anchored to the center of the stellar disk before representing                      – Sampling issues: Sampling is the most important of the
   the remainder of the intensity profile. This definition alone                         three issues aﬀecting the computed error of the fit of the
   results in a perfect fit to the center of the stellar disk and a                      intensity profile. For instance, Wade & Rucinski (1985)
   deteriorating fit as μ → 0 as the intensity profile deviates                          and Heyrovský (2007) noted that fitting an intensity pro-
   from the assumed structure of a particular limb-darkening                             file that is uniformly sampled in μ has a larger error than
   law.                                                                                  fitting the same profile that is uniformly sampled in r =
 – Fitting limb-darkening laws: Limb-darkening laws are typi-                            sin θ = sin(cos−1 μ). Uniform r-spacing emphasizes the in-
   cally fit to intensity profiles using a least-square algorithm.                       tensity profile near the center of the disk while a uniform
   Neilson & Lester (2011) showed that the best-fit coeﬃcients                           μ-spacing emphasizes the limb. Adopting any of the limb-
   for a given law are functions of weighted integrals of the                            darkening laws presented here, that law will fit the central

                                                                                                                                          A98, page 5 of 10
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                              A&A 554, A98 (2013)


                        0.4                   Linear                                       0.4                   Quadratic

                        0.3                                                                0.3

                        0.2                                                                0.2

                        0.1                                                                0.1

                         0                                                                  0
                        0.4                   Square-Root                                  0.4                   Logarithmic



        Kepler-Band Δ                                                      Kepler-Band Δ
                        0.3                                                                0.3

                        0.2                                                                0.2

                        0.1                                                                0.1

                         0                                                                  0
                        0.4                   Exponential                                  0.4                Four-parameter

                        0.3                                                                0.3

                        0.2                                                                0.2

                        0.1                                                                0.1

                         0                                                                  0
                              3000 4000 5000 6000 7000 8000                                      3000 4000 5000 6000 7000 8000
                                          Teff (K)                                                           Teff (K)
Fig. 7. Error of the best-fit limb-darkening relation, defined by Eq. (8), for every model atmosphere (red crosses represent plane-parallel models,
blue squares spherical models) for each of the six limb-darkening laws at Kepler-band wavelengths.


    part of the stellar surface more precisely than the limb be-           from spherical models. The only exception is the Claret (2000a)
    cause of the normalization at the center of the disk. If, in           four-parameter law, which fits the laws best, but appears to have
    addition, the surface intensity is sampled uniformly in r, that        unique properties.
    will give added weight to the central region. These two fac-
    tor combine to make the computed error of the fit smaller.
    Similarly, Howarth (2011b) found that limb-darkening coef-
                                                                           4. Gravity darkening coefficients
    ficients derived from planetary transits with large impact fac-        Rapid rotation distorts the shape of a star, making it aspheric,
    tors do not agree with model stellar atmosphere predictions.           with flattened poles and a bulged equator. As shown first by von
    This is because the planet passes across only the limb of the          Zeipel (1924), the gravity and eﬀective temperature vary in a co-
    star and not the center, therefore probing only part of the in-        ordinated way across the stellar surface such that at any point
    tensity profiles. Claret (2008, 2009) also found disagreement          the eﬀective temperature is proportional to the eﬀective gravity,
    between theoretical limb-darkening coeﬃcients and empiri-                         β1 /4
                                                                           T eﬀ ∼ geﬀ       , where β1 = 1 for radiative stars. However, this
    cal coeﬃcients measured from eclipsing binary light curves             value of β1 is valid only for bolometric radiation, and Kopal
    and comparisons to the planetary system HD 209458. Limb-               (1959) later derived monochromatic gravity-darkening correc-
    darkening coeﬃcients from stellar atmosphere models fit the            tions, y(λ). Claret (2000a), Claret & Hauschildt (2003) and
    whole profile yielding diﬀerent results.                               Claret & Bloemen (2011) have computed waveband-dependent
The combination of these three factors lead to calculated errors           gravity-darkening corrections as a a function of the central in-
that are relative and not an absolute measure of the quality of the        tensity of the star, as well as the gravity, eﬀective temperature
fit. In this work, diﬀerences in the error between fits to plane-          and the variable, β1 from plane-parallel models. Bloemen et al.
parallel and spherically symmetric model stellar atmosphere in-            (2011) derived
                                                                                                                        
tensity profiles computed with the same properties are due solely                    ∂ ln I(λ)          d ln T eﬀ ∂ ln I(λ)
to diﬀerences in the intensity profile near the limb where the             y(λ) =                   +                           ,         (9)
                                                                                       ∂ ln g T eﬀ       d ln g     ∂ ln T eﬀ g
spherical models provide more realistic predictions. Therefore,
the error analysis suggests that the various limb-darkening laws           and noted that (d ln T eﬀ /d ln g) = β1 /4. The variable β1 is a func-
lack the necessary complexity to precisely fit intensity profiles          tion of eﬀective temperature, but for the purpose of this analysis
A98, page 6 of 10
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                             H. R. Neilson and J. B. Lester: Limb darkening in red giant stars

                                             70                                                                                    70
                                             60                                                                                    60



                    (∂ ln(IV)/∂ ln Teff)g                                                                 (∂ ln(IV)/∂ ln Teff)g
                                             50                                                                                    50
                                             40                                                                                    40
                                             30                                                                                    30
                                             20                                                                                    20
                                             10                                                                                    10
                                               0                                                                                     0
                                               2                                                                                     2
                                             1.5                                                                                   1.5
                      eff                                                                                   eff
                                               1                                                                                     1


          (∂ ln(IV)/∂ ln g)T                                                                    (∂ ln(IV)/∂ ln g)T
                                             0.5                                                                                   0.5
                                               0                                                                                     0
                                            -0.5                                                                                  -0.5
                                              -1                                                                                    -1
                                            -1.5                                                                                  -1.5
                                              -2                                                                                    -2
                                            -2.5                                                                                  -2.5
                                             3.5                                                                                   3.5
                                               3                                                                                     3
                                             2.5                                                                                   2.5


            V-band y(λ)                                                                           V-band y(λ)
                                               2                                                                                     2
                                             1.5                                                                                   1.5
                                               1                                                                                     1
                                             0.5                                                                                   0.5
                                               0                                                                                     0
                                            -0.5                                                                                  -0.5
                                                   3000 4000 5000 6000 7000 8000                                                         -2   -1    0     1      2        3
                                                               Teff (K)                                                                            Log g (cgs)
Fig. 8. V-band central intensity derivatives and gravity-darkening coeﬃcients as function of eﬀective temperature (left) and gravity (right) com-
puted from plane-parallel (red crosses) and spherically symmetric (blue squares) model stellar atmospheres.


we assume β1 = 0.2 for T eﬀ < 7500 K and β1 = 1 for hotter                                         While the most significant diﬀerences between spherical and
stars. However, the value of β1 based on von Zeipel’s theorem                                  planar model predictions of gravity-darkening coeﬃcients are
is not strictly valid for radiative or convective stellar envelopes                            at lower temperatures, the gravity-darkening coeﬃcients com-
(Claret 2000b, 2012; Espinosa Lara & Rieutord 2011).                                           puted from spherically symmetric models are greater than those
    In Fig. 8, we plot the V-band values of each intensity deriva-                             of plane-parallel models for every eﬀective temperature. For ex-
tive for each model stellar atmosphere in Eq. (9), as well as                                  ample, a spherically symmetric model with T eﬀ = 8000 K has a
y(λ) computed for the assumed values of β1 . We find that plane-                               V-band gravity-darkening coeﬃcient of yV ≃ 0.165 while the
parallel and spherically symmetric model stellar atmospheres                                   plane-parallel model with the same eﬀective temperature has
predict similar gravity-darkening coeﬃcients for T eﬀ > 4000 K,                                yV ≃ 0.14. The diﬀerence is small but systematic.
but there are significant diﬀerences for cooler stars. We interpret
these diﬀerences for the cooler stars as consequences of both                                  5. Angular diameter corrections
surface convection and the shift from the negative hydrogen ion
to titanium oxide as the dominant opacity source. Both plane-                                  Interferometric observations measure the angular diameter of
parallel and spherical model intensities show greater variation                                a star along with its limb-darkening profile, but, unfortu-
at these cool eﬀective temperatures, but the intensity profiles of                             nately, the measured angular diameter and limb-darkening pro-
spherically symmetric model atmospheres vary more than that                                    files are not independent quantities. This is especially true
of plane-parallel model atmospheres.                                                           when the measured visibilities do not probe the second lobe.
                                                                                                                                                                     A98, page 7 of 10
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                            A&A 554, A98 (2013)

                            1                                                                1
                          0.95                                                             0.95




         V-Band θUD/θLD                                                   V-Band θUD/θLD
                           0.9                                                              0.9
                          0.85                                                             0.85
                           0.8                                                              0.8
                          0.75                                                             0.75
                           0.7                                                              0.7
                          0.65                                                             0.65
                             1                                                                1
                          0.95                                                             0.95




         K-band θUD/θLD                                                   K-band θUD/θLD
                           0.9                                                              0.9
                          0.85                                                             0.85
                           0.8                                                              0.8
                          0.75                                                             0.75
                           0.7                                                              0.7
                          0.65                                                             0.65
                                 3000 4000 5000 6000 7000 8000                                    -2   -1     0     1     2       3
                                             Teff (K)                                                       Log g (cgs)
Fig. 9. Interferometric angular diameter correction computed in V-band (top) and K-band as functions of eﬀective temperature (left) and gravity
(right). Corrections computed from plane-parallel model atmospheres are denoted with red x’s and spherically symmetric models blue squares.

Davis et al. (2000) measured stellar angular diameters from in-              Fits to spherically-symmetric model atmospheres suggest
terferometric observations by assuming that the stellar intensity        significantly diﬀerent angular diameter corrections as functions
profile is uniform, i.e. the intensity at any point on a stellar disk    of both eﬀective temperature and gravity. The V-band correc-
is equal to the central intensity. In that case, the uniform-disk        tions from spherical models, denoted ks , range from ks = 0.67
angular diameter can be directly fit to the observed visibilities        to 0.95, with no overlap with the plane-parallel model predic-
and then converted to a limb-darkened angular diameter using             tions. The K-band corrections show similar behaviors except that
model stellar atmospheres. Davis et al. (2000) computed cor-             spherical and planar corrections overlap somewhat. These results
rections using plane-parallel ATLAS models (Kurucz 1993) and             suggest that using plane-parallel model atmosphere corrections
found k ≡ θUD /θLD = 0.91 to 0.98 in the wavelength range                systematically underestimates the stellar angular diameter. For
λ = 400–800 nm. These limb-darkening corrections have been               instance, Mozurkewich et al. (2003) presented uniform-disk an-
applied to observations of Cepheids (Gallenne et al. 2012) and           gular diameters for a sample of 85 stars, along with limb-
Sirius (Davis et al. 2011) for example.                                  darkened angular diameters corrected using limb-darkening co-
    We compute angular diameter corrections using the recipe             eﬃcients from Claret et al. (1995) and Diaz-Cordoves et al.
described by Marengo et al. (2004), where we assume a limb-              (1995). Their angular diameter corrections vary from k = 0.89
darkened angular diameter of θLD = 1 mas to compute interfer-            to ≈1, consistent with the values found here for plane-parallel
ometric visibilities from a model atmosphere intensity profile.          model atmospheres.
That synthetic visibility is then fit by a uniform-disk angular di-          Of particular interest are the results of Mozurkewich et al.
ameter. The best-fit uniform-disk angular diameter is then equiv-        (2003) for α Persei (F5 Ib), for which they measured T eﬀ =
alent to the theoretical angular diameter correction. We compute         6750 K, and for Geminorum (G8 Ib), which was measured to
angular diameter corrections for the Johnson-Cousins BVRIHK              have T eﬀ = 4485 K. Mozurkewich et al. (2003) measured the
wavebands and show the corrections for the V- and K-bands in             uniform-disk angular diameters at 550 nm to be 2.986 ± 0.042
Fig. 9 as a function of eﬀective temperature for plane-parallel          for α Per and 4.467 ± 0.115 mas for Gem. Using these they
and spherically symmetric models. Corrections from spherical             computed limb-darkened angular diameters of 3.188 ± 0.035
models clearly diﬀer from corrections from plane-parallel model          and 4.703 ± 0.047 mas, respectively. Our spherically-symmetric
atmospheres. Intensity profiles from plane-parallel model stel-          models with log g = 1.5 and M = 10 M yield V-band angular-
lar atmospheres predict corrections in the narrow range from             diameter corrections of 0.929 for α Per and 0.916 for Gem.
k = 0.97–0.99 in V-band and approaches unity for longer wave-            Applying these to the uniform disk measurements gives larger
lengths. We show in Fig. 9 the V- and K-band angular diam-               limb-darkened angular diameters: θLD = 3.214 mas for α Per
eter corrections as function of eﬀective temperature and grav-           and θLD = 4.877 mas for Gem. The spherical correction
ity for plane-parallel and spherically symmetric model stellar           for α Per yields a value for θLD that is marginally consistent
atmospheres.                                                             with the angular diameter found using plane-parallel correction,
A98, page 8 of 10
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                       H. R. Neilson and J. B. Lester: Limb darkening in red giant stars

             0.98                                          K             coeﬃcients depend on the definition of the laws, all of which
                                                                         anchor the fit to I(μ = 1) = 1, making the fit sensitive to the
             0.96                                          H             sampling of the intensity profile as well as to the method for
                                                                         fitting the data. Because intensity profiles for spherical models

   θUD/θLD
             0.94                                          I             are more complex, the fitting error is greater than the error for
             0.92                                          R             simpler plane-parallel model intensity profiles. However, spher-
                                                                         ically symmetric model atmospheres are a more realistic repre-
              0.9                                          V             sentation of actual stellar atmospheres, meaning they are better
                                                           B             suited for limb darkening studies.
             0.88                                                             Fits to the four-parameter limb-darkening law also show cor-
                                                                         relations between the limb-darkening coeﬃcients; we find that
                    0 2 4 6 8 10 12 14 16 18 20                          the linear combination of the four coeﬃcients are approximately
                         Stellar Mass (M)                               constant, with that constant being a function of the atmosphere’s
                                                                         geometry. This result suggests that the linear combination of the
Fig. 10. Interferometric angular diameter corrections as a function of   observed coeﬃcients for the four-parameter law provides a sim-
waveband and stellar mass for spherically symmetric model stellar at-    ple test of whether the observations are probing the edge of the
mospheres with log g = 2 and T eﬀ = 3500 K.                              stellar disk, i.e. sphericity.
                                                                              We also predict wavelength-dependent gravity-darkening co-
                                                                         eﬃcients based on the Claret & Bloemen (2011) prescription.
                                                                         Unlike the limb-darkening coeﬃcients, the gravity-darkening
whereas the limb-darkened angular diameter of Gem measured               coeﬃcients are less dependent on model atmosphere geometry.
by Mozurkewich et al. (2003) is almost 4% smaller than what              This is because the gravity-darkening coeﬃcients depend on the
would be predicted by applying spherical model corrections.              change of the central intensity with respect to eﬀective temper-
This diﬀerence may appear to be small but this underestimate             ature and gravity, hence the diﬀerence between atmospheres for
is systematic.                                                           the same geometry. Gravity-darkening is also a function of the
    As a test, we check how the angular diameter corrections             central intensity, which is insensitive to model geometry. The
vary as function of stellar mass. Because models with low eﬀec-          spherically symmetric gravity-darkening coeﬃcients are simi-
tive temperature but relatively high gravity appear to predict the       lar to plane-parallel coeﬃcients for T eﬀ > 5000 K and begin
smallest corrections, we hold T eﬀ = 3500 K and log g = 2. The           to diverge for cooler stellar atmosphere models. Only at the
angular diameter corrections are shown in Fig. 10 as a function          coolest eﬀective temperatures, 3000 K ≤ T eﬀ ≤ 4000 K, is the
of stellar mass for the six Johnson-Cousins wavebands consid-            geometry of the model atmosphere important, with the spher-
ered in this work. The figure suggests that the corrections are          ically symmetric coeﬃcients being approximately an order-
insensitive to the mass of the stellar model except for low-mass         of-magnitude greater than those predicted from plane-parallel
(M ≤ 1 M ) models. This is reassuring and suggests that when            model atmospheres.
applying these corrections, one can ignore the stellar mass. The              Unlike the gravity darkening coeﬃcients, the interferometric
diﬀerence between limb-darkening profiles and angular diame-             angular-diameter corrections do depend on geometry. For plane-
ter corrections is small and consistent with previous results by         parallel model atmospheres the angular-diameter corrections
Lester et al. (2013).                                                    vary from about 0.95–1, whereas the corrections for spherically
                                                                         symmetric model atmospheres vary from 0.67–1. Previous anal-
6. Summary                                                               yses had assumed that corrections from plane-parallel models
                                                                         are applicable to all stars, but this is not true. At low gravity,
In this work, we present model atmosphere intensity profiles for         log g < 3, spherically symmetric corrections deviate signifi-
the BVRIHK, CoRot and Kepler passbands from both plane-                  cantly from plane-parallel model predictions. The diﬀerence be-
parallel and spherically symmetric geometries based on mod-              tween spherical and plane-parallel models is a function of both
els computed by Neilson & Lester (2011, 2012). We fit a num-             gravity and eﬀective temperature and also appears to vary as a
ber of limb-darkening laws to these intensity profiles, as well          function of stellar mass.
as compute gravity-darkening coeﬃcients and angular diameter                  The angular-diameter corrections, limb-darkening and
corrections for interferometry. We test how these fits vary as a         gravity-darkening coeﬃcients are publicly available as online ta-
function of model atmosphere geometry and compile tables of              bles. Each table has the format T eﬀ (K), log g, and M (M ) and
limb-darkening coeﬃcients, gravity-darkening coeﬃcients and              then the appropriate variables for each waveband, such as linear
angular diameter corrections that can be applied to observations.        limb-darkening coeﬃcients. Tables of gravity-darkening coeﬃ-
    We consider six limb-darkening laws in this work: linear,            cients also contain values of the intensity derivatives with respect
quadratic, square-root, four-parameter, exponential and logarith-        to gravity and eﬀective temperature. For plane-parallel models,
mic. These laws fit the intensity profiles from plane-parallel           only T eﬀ and log g are given in the tables. We list the properties
model atmospheres well, but not the intensity profiles of the            of these tables in Table 1, that are archived in electronic form at
spherical models based on computed relative errors. The one ex-          the CDS. Model atmosphere intensity profiles are also archived
ception is the Claret (2000a) four-parameter law, for which the          at the CDS
diﬀerence between the spherical model intensity profiles and the              Techniques such as optical interferometry, microlensing
predictions of the fitting law is small enough to still be applicable    observations, planetary transit and eclipsing binary observa-
to observations, although the law still fits the spherical profiles      tions are continuously improving the measurements of stel-
more poorly than the plane-parallel intensities.                         lar limb darkening needed to test model stellar atmospheres
    While those predicted errors are useful for comparing fits to        and the physics assumed in their calculation. At lower grav-
planar and spherical model intensity profiles, they are not ideal        ities, these observations require the more physically realistic
for studies of actual limb darkening. Best-fit limb-darkening            spherically symmetric models to constrain stellar properties.
                                                                                                                           A98, page 9 of 10
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                                      A&A 554, A98 (2013)

Table 1. Summary of limb-darkening coeﬃcient, gravity-darkening co-                  Claret, A. 2012, A&A, 538, A3
eﬃcient and interferometric angular diameter correction tables found                 Claret, A., & Bloemen, S. 2011, A&A, 529, A75
online.                                                                              Claret, A., & Hauschildt, P. H. 2003, A&A, 412, 241
                                                                                     Claret, A., Diaz-Cordoves, J., & Gimenez, A. 1995, A&AS, 114, 247
                                                                                     Claret, A., Hauschildt, P. H., & Witte, S. 2012, A&A, 546, A14
    Name         Geometry       Type                                                 Claret, A., Hauschildt, P. H., & Witte, S. 2013, A&A, 552, A16
    Table2       Spherical      Linear limb darkening Eq. (1)                        Croll, B., Albert, L., Jayawardhana, R., et al. 2011, ApJ, 736, 78
    Table3       Spherical      Quadratic limb darkening Eq. (2)                     Davis, J., Tango, W. J., & Booth, A. J. 2000, MNRAS, 318, 387
                                                                                     Davis, J., Ireland, M. J., North, J. R., et al. 2011, PASA, 28, 58
    Table4       Spherical      Square root limb darkening Eq. (3)                   Diaz-Cordoves, J., Claret, A., & Gimenez, A. 1995, A&AS, 110, 329
    Table5       Spherical      Four-parameter limb darkening Eq. (4)                Eaton, J. A., Henry, G. W., & Odell, A. P. 2008, ApJ, 679, 1490
    Table6       Spherical      Exponential limb darkening Eq. (5)                   Espinosa Lara, F., & Rieutord, M. 2011, A&A, 533, A43
    Table7       Spherical      Logarithmic limb darkening Eq. (6)                   Feautrier, P. 1964, Comptes Rendus Academie des Sciences (série non spécifiée),
    Table8       Planar         Linear limb darkening Eq. (1)                           258, 3189
    Table9       Planar         Quadratic limb darkening Eq. (2)                     Fields, D. L., Albrow, M. D., An, J., et al. 2003, ApJ, 596, 1305
    Table10      Planar         Square root limb darkening Eq. (3)                   Fouqué, P., Heyrovský, D., Dong, S., et al. 2010, A&A, 518, A51
    Table11      Planar         Four-parameter limb darkening Eq. (4)                Gallenne, A., Kervella, P., Mérand, A., et al. 2012, A&A, 541, A87
    Table12      Planar         Exponential limb darkening Eq. (5)                   Gustafsson, B., Edvardsson, B., Eriksson, K., et al. 2008, A&A, 486, 951
                                                                                     Hanbury Brown, R., Davis, J., Lake, R. J. W., & Thompson, R. J. 1974, MNRAS,
    Table13      Planar         Logarithmic limb darkening Eq. (6)                      167, 475
    Table14      Spherical      Gravity darkening                                    Haubois, X., Perrin, G., Lacour, S., et al. 2009, A&A, 508, 923
    Table15      Planar         Gravity darkening                                    Hauschildt, P. H., Allard, F., Ferguson, J., Baron, E., & Alexander, D. R. 1999,
    Table16      Spherical      Angular diameter corrections                            ApJ, 525, 871
    Table17      Planar         Angular diameter corrections                         Heyrovský, D. 2007, ApJ, 656, 483
                                                                                     Howarth, I. D. 2011a, MNRAS, 413, 1515
Notes. Tables listed here can be retrieved electronically from the CDS.              Howarth, I. D. 2011b, MNRAS, 418, 1165
                                                                                     Howell, S. B., Rowe, J. F., Bryson, S. T., et al. 2012, ApJ, 746, 123
                                                                                     Johnson, H. L., & Morgan, W. W. 1953, ApJ, 117, 313
                                                                                     Koch, D. G., Borucki, W., Dunham, E., et al. 2004, in SPIE Conf. Ser. 5487,
The predicted limb-darkening coeﬃcients, gravity-darkening                              ed. J. C. Mather, 1491
coeﬃcients and angular diameter corrections from spherically                         Kopal, Z. 1959, Close binary systems (London: Chapman & Hall)
symmetric SAtlas models are new tools that for aiding analy-                         Kurucz, R. L. 1979, ApJS, 40, 1
ses of these observations.                                                           Kurucz, R. L. 1993, in IAU Colloq. 138: Peculiar versus Normal Phenomena in
                                                                                        A-type and Related Stars, eds. M. M. Dworetsky, F. Castelli, & R. Faraggiana,
Acknowledgements. The authors acknowledge support from a research grant                 ASP Conf. Ser., 44, 87
from the Natural Sciences and Engineering Research Council of Canada,                Lester, J. B., & Neilson, H. R. 2008, A&A, 491, 633
the Alexander von Humboldt Foundation and National Science Foundation                Lester, J. B., Dinshaw, R., & Neilson, H. R. 2013, PASP, 125, 335
(AST-0807664).                                                                       Marengo, M., Karovska, M., Sasselov, D. D., & Sanchez, M. 2004, ApJ, 603,
                                                                                        285
                                                                                     Mihalas, D. 1978, Stellar atmospheres, 2nd edition (San Francisco:
                                                                                        W. H. Freeman and Co.)
References                                                                           Mozurkewich, D., Armstrong, J. T., Hindsley, R. B., et al. 2003, AJ, 126, 2502
Al-Naimiy, H. M. 1978, Ap&SS, 53, 181                                                Neilson, H. R., & Lester, J. B. 2008, A&A, 490, 807
An, J. H., Albrow, M. D., Beaulieu, J.-P., et al. 2002, ApJ, 572, 521                Neilson, H. R., & Lester, J. B. 2011, A&A, 530, A65
Aufdenberg, J. P., Ludwig, H.-G., & Kervella, P. 2005, ApJ, 633, 424                 Neilson, H. R., & Lester, J. B. 2012, A&A, 544, A117
Auvergne, M., Bodin, P., Boisnard, L., et al. 2009, A&A, 506, 411                    Rybicki, G. B. 1971, J. Quant. Spec. Radiat. Transf., 11, 589
Avrett, E. H., & Krook, M. 1963, ApJ, 137, 874                                       Schwarzschild, K. 1906, Nachrichten von der Königlichen Gesellschaft der
Bessell, M. S. 2005, ARA&A, 43, 293                                                     Wissenschaften zu Göttingen, Mathematisch-physikalische Klasse, 41
Bloemen, S., Marsh, T. R., Østensen, R. H., et al. 2011, MNRAS, 410, 1787            Sing, D. K. 2010, A&A, 510, A21
Catala, C., Arentoft, T., Fridlund, M., et al. 2010, in Pathways Towards Habitable   van Belle, G. T., Ciardi, D. R., Thompson, R. R., Akeson, R. L., & Lada, E. A.
   Planets, eds. V. Coudé Du Foresto, D. M. Gelino, & I. Ribas, ASP Conf. Ser.,         2001, ApJ, 559, 1155
   430, 260                                                                          van Hamme, W. 1993, AJ, 106, 2096
Claret, A. 2000a, A&A, 363, 1081                                                     von Zeipel, H. 1924, MNRAS, 84, 665
Claret, A. 2000b, A&A, 359, 289                                                      Wade, R. A., & Rucinski, S. M. 1985, A&AS, 60, 471
Claret, A. 2008, A&A, 482, 259                                                       Wittkowski, M., Aufdenberg, J. P., & Kervella, P. 2004, A&A, 413, 711
Claret, A. 2009, A&A, 506, 1335                                                      Zub, M., Cassan, A., Heyrovský, D., et al. 2011, A&A, 525, A15




A98, page 10 of 10
```
