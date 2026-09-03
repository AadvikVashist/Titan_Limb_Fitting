---
citation_key: "neilson2011comparison"
title: "Comparison of Limb-Darkening Laws from Plane-Parallel and Spherically-Symmetric Model Stellar Atmospheres"
source_pdf: "data/papers/neilson2011comparison.pdf"
source_pdf_sha256: "8c9909eb6f8dfbf5a936d0b829fb8bd3eb3ce48a56471c3dab66520e0aafd2e0"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
From Interacting Binaries to Exoplanets: Essential Modeling Tools
Proceedings IAU Symposium No. 282, 2011                     International Astronomical Union 2012
                                                            c
Mercedes T. Richards & Ivan Hubeny, eds.                            doi:10.1017/S174392131102744X



  Comparison of Limb-Darkening Laws from
  Plane-Parallel and Spherically-Symmetric
         Model Stellar Atmospheres
                                     Hilding R. Neilson
                     Argelander-Institut für Astronomie, Bonn Universität,
                         Auf Dem Hügel 71, Bonn, D-53121, Germany
                             email: hneilson@astro.uni-bonn.de

Abstract. Limb-darkening is a fundamental constraint for modeling eclipsing binary and plan-
etary transit light curves. As observations, for example from Kepler, CoRot, and Most, be-
come more precise then a greater understanding of limb-darkening is necessary. However, limb-
darkening is typically modeled as simple parameterizations ﬁt to plane-parallel model stellar at-
mospheres that ignores stellar atmospheric extension. In this work, I compute linear, quadratic
and four-parameter limb-darkening laws from grids of plane-parallel and spherically-symmetric
model stellar atmospheres in a temperature and gravity range representing stars evolving on the
Red Giant branch. The limb-darkening relations for each geometry are compared and are found
to ﬁt plane-parallel models much better than the spherically-symmetric models. Assuming that
limb-darkening from spherically-symmetry model atmospheres are more physically representa-
tive of actual stellar limb-darkening than plane-parallel models, then these limb-darkening laws
will not ﬁt the limb of a stellar disk leading to errors in a light curve ﬁt. This error will increase
with a star’s atmospheric extension.
Keywords. stars: atmospheres, stars: fundamental parameters, (stars:) supergiants



1. Introduction
   Stellar limb-darkening is the observed change of intensity from the center of the stellar
disk to the observable edge, where the intensity decrease is due to the geometric pro-
jection of the line-of-sight relative to the radius of the star. This eﬀect is an important
challenge for the interpretation of observations of binary stars (e.g. Claret 2008), and
planetary transits (e.g. Knutson et al. 2007, Croll et al. 2011), as well as interferometric
(e.g. Wittkowski et al. 2004) and microlensing (e.g. Zub et al. 2011) observations. Typ-
ically, limb-darkening is treated as a parameterization or relation as a function of the
cosine of the angle formed by the radius and line-of-sight, called µ to simplify the analysis.
   Stellar atmosphere models and binary/transit observations are complementary tools for
understanding limb-darkening and stellar astrophysics in general because observed limb-
darkening can help constrain models. There are numerous articles describing diﬀerent
limb-darkening relations (Al-Naimiy 1978, Wade & Rucinski 1985, Claret et al. 1995,
Claret 2000), limb-darkening coeﬃcients from predicted intensity proﬁles for a number
of stellar atmosphere codes such as Atlas and Phoenix (e.g. Howarth 2011, Sing 2010,
Claret & Hauschildt 2003), and diﬀerent ﬁtting methods (Wade 1985, Heyrovsky 2003,
2007, Claret 2008). In this work, I focus on a small number of limb-darkening laws and
compare predicted ﬁts for intensity proﬁles from plane-parallel and spherically symmetric
model stellar atmospheres. In the next section, I describe the stellar atmosphere code
and three limb-darkening laws of interest: a linear, quadratic, and four-parameter (Claret
2000). In Sect. 3, I present results of the ﬁtting of the limb-darkening laws using model
                                                 243
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
244                                       H. Neilson
atmospheres, and how the errors of the ﬁt depend on assumed geometry. I summarize
this work in Sect. 4.

2. Stellar atmosphere code and Limb-darkening Laws
   For this analysis, I use a new Fortran 90 version of the Kurucz Atlas code (Lester
& Neilson 2008). The code computes opacities using opacity distribution functions, and
atmospheres are assumed to be in local thermodynamic equilibrium and hydrostatic equi-
librium. Each atmosphere model outputs intensity proﬁles as a function of wavelength, for
an equal spacing of µ for 1000 points. Typical calculations for the Atlas code compute
intensity proﬁles for 10 - 17 µ-points. The program computes models for either plane-
parallel or spherically-symmetric geometries, where the plane-parallel model is described
by two fundamental parameters such as Teﬀ and log g, while spherical models require an
additional parameter such as stellar mass. Neilson & Lester (2008) ﬁt model intensity
proﬁles to interferometric observations from Wittkowski et al. (2004) and predicted sim-
ilar fundamental parameters as those authors. Also, Neilson & Lester (2011) predicted
limb-darkening coeﬃcients for a speciﬁc limb-darkening law from spherical models, com-
pared them to results for microlensing observations from Fields et al. (2003), and found
better agreement than the authors did using plane-parallel models and spherical models
that had intensity proﬁles clipped to remove the extended limb.
   I have computed approximately 2000 model stellar atmospheres in spherical symmetry
for the parameter range Teﬀ = 3000 - 8000 K in steps of 100 K, log g = −1 - + 3 in steps
of 0.25 in cgs units, and M = 2.5 - 10 M in steps of 2.5 M . Plane-parallel models are
computed for the same values of Teﬀ and log g.
   For this work, I compute least-squared ﬁts to three laws:
               I(µ)
                      = 1 − a(1 − µ)                     Linear,                     (2.1)
               I(1)
               I(µ)
                      = 1 − b(1 − µ) − c(1 − µ)2         Quadratic,                  (2.2)
               I(1)
               I(µ)           
                              4
                      = 1−          di (1 − µi/2 )       Four Parameter,             (2.3)
               I(1)           i=1

where intensities are computed in the Kepler white light passband. All ﬁts are computed
using least-square ﬁtting of the limb-darkening coeﬃcients. The quality of the ﬁt may be
measured in a number of ways; here, I test the quality of the ﬁt of limb-darkening laws
by checking how well they conserve stellar ﬂux, ∆F/F = (FM o del − FLaw )/FM o del .

3. Results & Summary
  In Fig. 1, I show the predicted intensity proﬁles for a Teﬀ = 4000 K and log g = 2
model atmosphere for both geometries. There is a signiﬁcant diﬀerence between the
model intensity proﬁles such that the intensity near the limb of the spherically-symmetric
model atmosphere is much smaller than the plane-parallel model atmosphere. The plane-
parallel model does not appear to go to zero, though the equation of transfer suggests
that as the intensity in the limit µ → 0 then I(µ) → 0 (Mihalas 1978). This may be
an issue with the resolution of µ. The plane-parallel model is also much better ﬁt by
the limb-darkening laws than the spherically-symmetric model atmosphere because the
spherically-symmetric intensity proﬁle is more complex.
  The relative diﬀerence between the model intensity proﬁles and stellar ﬂuxes predicted
by the limb darkening laws are shown in Fig. 2. The ﬁts to plane-parallel model atmo-
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                Model Atmospheres and Stellar Limb-Darkening                                                                                                    245
                       1                                                                                                      1
                                                                                                                                                                     Spherical Model
                                                                                                                                                                               Linear
                                                                                                                            0.8                                            Quadratic
                     0.8
                                                                                                                                                                        4-parameter
                                                                                                                            0.6


     Kepler I/I(1)                                                                                          Kepler I/I(1)
                     0.6
                                                                                                                            0.4
                     0.4
                                                                                                                            0.2
                                                  Plane-Parallel Model
                     0.2                                         Linear                                                       0
                                                            Quadratic
                                                         4-parameter
                       0                                                                                                    -0.2
                            1      0.8          0.6                   0.4           0.2            0                               1            0.8           0.6       0.4           0.2         0
                                                                µ                                                                                                   µ

  Figure 1. Intensity proﬁles for a plane-parallel (left) and spherically-symmetric (right) model
    atmosphere with Te ﬀ = 4000 K and log g = 2, along with the best-ﬁt limb-darkening laws.

                     0.35                                                                                                    0.4
                                                                        Spherical                                                                                            Spherical
                      0.3                                                 Planar                                            0.35                                               Planar

                                                                                                                             0.3
                     0.25



Kepler ∆F/F                                                                                              Kepler ∆F/F
                                                                                                                            0.25
                      0.2
                                                                                                                             0.2
                     0.15
                                                                                                                            0.15
                      0.1
                                                                                                                             0.1
                     0.05                                                                                                   0.05

                       0                                                                                                       0
                       2000     3000     4000   5000 6000                   7000     8000     9000                             2000           3000     4000     5000 6000      7000      8000   9000
                                                  Teff (K)                                                                                                        Teff (K)


                                                                     0.1
                                                                                                                                   Spherical
                                                                    0.08                                                             Planar

                                                                    0.06



                                                  Kepler ∆F/F
                                                                    0.04

                                                                    0.02

                                                                       0

                                                                    -0.02

                                                                    -0.04

                                                                        2000       3000     4000       5000 6000                       7000     8000     9000
                                                                                                         Teff (K)

Figure 2. Relative diﬀerence of between model and ﬁt stellar ﬂuxes for the three limb-darken-
ing laws in the Kepler passband, linear (upper-left), quadratic (upper-right), and Claret (2000)
four-parameter (bottom) laws for plane-parallel and spherically-symmetric model stellar atmo-
spheres.

spheres appear to be better; the average error is < 5% for the linear law, and is much
smaller for the other laws. The quality of the ﬁt for plane-parallel model atmospheres
is also apparently independent of eﬀective temperature. The results for the spherically-
symmetric model atmospheres are strikingly diﬀerent. The ﬂux errors are much larger,
5-10% for the linear law, 0-20% for the quadratic law, and 0-5% for the four parame-
ter limb-darkening law. The diﬀerence in ﬁts due to model atmosphere geometry sug-
gests a signiﬁcant problem for understanding stellar limb-darkening. It is reasonable to
assume that a spherical geometry is a more physical representation of an actual star
than a plane-parallel stellar atmosphere, hence people should be hesitant when using
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
246                                      H. Neilson
limb-darkening coeﬃcients generated from plane-parallel model atmospheres. It also sug-
gests that these are not ideal limb-darkening laws to use and it may be necessary to
develop new limb-darkening relations for the future.
References
Al-Naimiy, H. M. 1978, Ap&SS, 53, 181
Claret, A. 2000, A&A, 363, 1081
—. 2008, A&A, 482, 259
Claret, A., Diaz-Cordoves, J., & Gimenez, A. 1995, A&AS, 114, 247
Claret, A. & Hauschildt, P. H. 2003, A&A, 412, 241
Croll, B., Albert, L., Jayawardhana, R., Miller-Ricci Kempton, E., Fortney, J. J., Murray, N.,
     & Neilson, H. 2011, ApJ, 736, 78
Fields, D. L., Albrow, M. D., & An, J., et al. 2003, ApJ, 596, 1305
Gustafsson, B., Edvardsson, B., Eriksson, K., Jørgensen, U. G., Nordlund, Å., & Plez, B. 2008,
     A&A, 486, 951
Hauschildt, P. H., Allard, F., Ferguson, J., Baron, E., & Alexander, D. R. 1999, ApJ, 525, 871
Heyrovský, D. 2003, ApJ, 594, 464
—. 2007, ApJ, 656, 483
Howarth, I. D. 2011, MNRAS, 413, 1515
Knutson, H. A., Charbonneau, D., Noyes, R. W., Brown, T. M., & Gilliland, R. L. 2007, ApJ,
     655, 564
Lester, J. B. & Neilson, H. R. 2008, A&A, 491, 633
Mihalas, D. 1978, Stellar atmospheres /2nd edition/, ed. Hevelius, J.
Neilson, H. R. & Lester, J. B. 2008, A&A, 490, 807
—. 2011, A&A, 530, A65
Sing, D. K. 2010, A&A, 510, A21
Wade, R. A. & Rucinski, S. M. 1985, A&AS, 60, 471
Wittkowski, M., Aufdenberg, J. P., & Kervella, P. 2004, A&A, 413, 711
Zub, M. & Cassan, A., Heyrovský et al., 2011, A&A, 525, A15
Discussion
R. Wilson: Your starting explanation of why the intensity goes to zero at the limb for
a plane parallel case is not correct. Actually, the intensity does not go to zero at the
limb. This result comes from neglect of emission along the line of sight; it is not just
an attenuation problem, but has both a source factor and an attenuation factor in the
intensity integral. If one looks into an inﬁnite uniform region, the received intensity is
not zero, but is the intensity characteristic of the region’s temperature.

I. Hubeny: Response to Bob Wilson’s comment: The intensity does not indeed have to
go to zero at the limb, but such a case is not covered by a 1-D plane-parallel treatment of
the transfer equation anyway because, in this case, the medium is inﬁnite with no natural
boundary condition. Comment on the talk: The Eddington factor (the ratio of the K-
moment and the mean intensity) is not necessarily equal to 1/3 in the plane-parallel case.
Such a quantity is usually called a variable Eddington factor, and depends on depth and
frequency; it goes to 1/3 only deep in the atmosphere.

A. Prša: I understand why one would use an analytic model for a Mandel-Agol type
approach, but perhaps the systematic error from the simple ﬁt may be avoided simply
by linearly interpolating (or looking up) I(µ).

P. Stee: I did not understand why you used the ﬁrst lobe of the visibility function to
ﬁt the LD instead of the second lobe, especially since you may ﬁt the ﬁrst lobe with a
uniform disk?
```
