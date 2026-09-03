---
citation_key: "bazzon2014HST"
title: "HST observations of the limb polarization of Titan"
source_pdf: "data/papers/bazzon2014HST.pdf"
source_pdf_sha256: "e8c8566d296299481fddcfed5230f9351153e5741ee2afb450501c931c669706"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
A&A 572, A6 (2014)                                                                                                  Astronomy
DOI: 10.1051/0004-6361/201323139                                                                                     &
 ESO 2014
c                                                                                                                   Astrophysics


                  HST observations of the limb polarization of Titan
                                             A. Bazzon1 , H. M. Schmid1 , and E. Buenzli2

      1
        ETH Zurich, Institute of Astronomy, Wolfgang-Pauli-Str. 27, 8093 Zurich, Switzerland
        e-mail: bazzon@astro.phys.ethz.ch
      2
        Max-Planck-Institut für Astronomie, Königstuhl 17, 69117 Heidelberg, Germany
      Received 27 November 2013 / Accepted 5 September 2014

                                                                  ABSTRACT

      Context. Titan is an excellent test case for detailed studies of the scattering polarization from thick hazy atmospheres. Accurate
      scattering and polarization parameters have been provided by the in situ measurements of the Cassini-Huygens landing probe. For
      Earth-bound observations Titan can only be observed at a backscattering situation, where the disk-integrated polarization is close to
      zero. However, with resolved imaging polarimetry a second order polarization signal along the entire limb of Titan can be measured.
      Aims. We present the first limb polarization measurements of Titan, which are compared as a test to our limb polarization models.
      Methods. Previously unpublished imaging polarimetry from the HST archive is presented, which resolves the disk of Titan. We
      determine flux-weighted averages of the limb polarization and radial limb polarization profiles, and investigate the degradation and
      cancelation eﬀects in the polarization signal due to the limited spatial resolution of our observations. Taking this into account we
      derive corrected values for the limb polarization in Titan. The results are compared with limb polarization models, using atmosphere
      and haze scattering parameters from the literature.
      Results. In the wavelength bands between 250 nm and 2 μm a strong limb polarization of about 2−7% is detected with a position
      angle perpendicular to the limb. The fractional polarization is highest around 1 μm. As a first approximation, the polarization seems
      to be equally strong along the entire limb. The comparison of our data with model calculations and the literature shows that the
      detected polarization is compatible with expectations from previous polarimetric observations taken with Voyager 2, Pioneer 11, and
      the Huygens probe.
      Conclusions. Our results indicate that ground-based monitoring measurements of the limb-polarization of Titan could be useful for
      investigating local haze properties and the impact of short-term and seasonal variations of the hazy atmosphere of Titan. Planets with
      hazy atmospheres similar to Titan are particularly good candidates for detection with the polarimetric mode of the upcoming planet
      finder instrument at the VLT. Therefore, a good knowledge of the polarization properties of Titan is also important for the search and
      investigation of extra-solar planets.
      Key words. polarization – planets and satellites: atmospheres – scattering – radiative transfer – instrumentation: polarimeters



1. Introduction                                                           polarimetry is available. In this work we present previously un-
                                                                          published imaging polarimetry from the HST archive which re-
Solar light reflected from planets, moons, and smaller objects is         solves the disk of Titan and clearly shows a strong limb polariza-
polarized. This basic property of light reflection provides a pow-        tion eﬀect. Together with literature values for the albedo Ag and
erful diagnostic tool for the remote investigation of the scattering      the quadrature polarization p(90◦) of Titan, our limb polariza-
particles in atmospheres and the reflecting surfaces of solar sys-        tion measurements can now be used to test polarization models
tem bodies.                                                               for a haze scattering atmosphere, and we can make predictions
    Polarization studies of Titan are particularly well suited to         for the detection and characterization of reflected light of extra-
studying the scattering polarization from a hazy atmosphere.              solar planets.
The hazy atmosphere of Titan produces a very strong polariza-                  The limb polarization is a well-known second order scat-
tion signal over a wide wavelength range. At quadrature phase             tering eﬀect of reflecting atmospheres with predominantly
α ≈ 90◦ , the fractional polarization of Titan from the UV to the         Rayleigh-type scattering processes (e.g., van de Hulst 1980).
R band is p ≈ 50%, as measured by the Pioneer 11 (Tomasko                 In general, single backscattering with scattering angles ∼180◦
& Smith 1982) and Voyager 2 (West et al. 1983) spacecrafts.               would produce a very small polarization signal or no signal at
Furthermore, thanks to the joint NASA-ESA Cassini-Huygens                 all. Thus, the polarization measured at the limb arises from sec-
satellite mission, Titan’s surface and atmospheric structure are          ond order and also higher order scatterings by light that is scat-
known in great detail. In particular, the measurements made in-           tered sideways, i.e., more or less parallel to the limb, and then
side Titan’s atmosphere, made available by the Huygens landing            scattered back to the observer. The polarization angle induced
probe, have provided accurate scattering and polarization param-          by Rayleigh scattering, i.e. single dipole-type scattering, is per-
eters for the haze particles (e.g., Brown et al. 2010; Tomasko            pendicular to the propagation direction of the incoming photon.
et al. 2008).                                                             Hence, the position angle of polarization is perpendicular to the
    For Earth-bound instruments the atmosphere of Titan can               limb everywhere.
only be observed at very limited phase angles α  5◦ , where the               Over the last 20 years, Titan’s thick and hazy atmosphere has
disk-integrated polarization is close to zero, and the polarimet-         been monitored and intensively studied by spectral HST obser-
ric properties can only be investigated if disk-resolved imaging          vations, revealing strong local albedo variations mainly caused
                                           Article published by EDP Sciences                                                   A6, page 1 of 13
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                          A&A 572, A6 (2014)

by seasonal migration of haze from one hemisphere to another
(Lorenz et al. 2004, and references therein). The most prominent
features are a varying north-south asymmetry, a dark polar hood
that is most prominent in the UV, and a detached haze layer lying
only ∼200 km above the optical limb of Titan (e.g., Lorenz et al.
2004, 2006). Observations of Titan from the ground, e.g., us-
ing the upcoming SPHERE instrument at the VLT (Beuzit et al.
2008), consisting of state-of-the-art imagers and polarimeters,
will have the advantage of a much higher spatial resolution and
a broader wavelength coverage. Therefore, a consistent monitor-
ing program of Titan’s atmosphere from the ground could po-
tentially be useful for investigating the temporal changes in the
polarization structure along the limb, and for constraining local
haze properties of Titan.
    After the description of the observational data and the basic
data reduction in Sects. 2 and 3 respectively, we discuss the in-
tensity images in Sect. 4 and compare them with the literature. In
Sect. 5 we derive the Stokes images for all our wavelength bands,
which are then converted into radial limb polarization images in
Sect. 6. There, we discuss the radial polarization profiles and the
advantage of disk-integrated radial polarization measurements,
and we explain how we correct our images for the polarization
degradation caused by a PSF smearing eﬀect. In Sect. 7 we de-
scribe our radiative transfer model for Titan, and we compare         Fig. 1. Upper panel: geometric albedo of Titan (full line) and eﬀec-
the model with our limb polarization results and literature val-      tive albedos Ag,eﬀ and wavelengths λeﬀ for the HST filter polarimetry
ues for the geometric albedo Ag and the quadrature polarization       (crosses). The dashed line illustrates the solar spectrum. Lower panel:
p(90◦) of Titan. The last section gives a summary and discusses       normalized filter eﬃciency curves and the normalized wavelength dis-
the prospects for a polarimetric monitoring program of Titan and      tribution of the registered photons.
for the detection and investigation of extra-solar haze planets.
                                                                      median wavelengths of the individual photon distributions as ef-
                                                                      fective filter wavelengths λeﬀ , and the reflected-flux weighted
2. Observations                                                       values for the eﬀective albedos Ag,eﬀ for each filter. The cor-
                                                                      responding values are indicated in Fig. 1 and Table 3 lists the
We reduced and analyzed imaging polarimetry of Titan from             derived values and RT (λeﬀ ).
the HST archive1 for which only the intensity images have
                                                                          We note that when looking specifically at the limb the
been published (Lorenz et al. 2006) but not the polarization
                                                                      methane bands are weaker and center-to-limb diﬀerences in the
data. The data were recorded with the ACS HRC and the
                                                                      albedo spectrum are present (e.g., Smith et al. 1996; Lorenz et al.
NICMOS instruments in seven filters covering wavelengths
                                                                      2004, 2006). However, we mainly focus on the derivation of
0.25 μm−2 μm. Polarimetry is achieved with three subsequent
                                                                      disk-integrated albedo and limb polarization values (Sect. 6.3).
measurements, using three polarizers with diﬀerent orientations.
                                                                          The eﬀective filter wavelengths λeﬀ are shifted to longer
Titan was observed in 2002 during two visits on November 27
                                                                      wavelength for the UV/blue filters because of the steep pho-
and December 2, i.e., shortly after southern summer solstice that
                                                                      ton spectrum. The eﬀective albedos Ag,eﬀ are relevant for the fil-
occurred in late October 2002, and with the north pole of Titan
                                                                      ters covering spectral regions with strong CH4 absorption bands.
on the hidden hemisphere. Table 1 gives an overview of the ob-
                                                                      They are about 20% higher for the NIC1 and 40% higher for
servational parameters, the used instruments and corresponding
                                                                      the NIC2 passbands than the simple mean. For the F250W and
filters, the total exposure times, and the plate scales.
                                                                      the F330W filters the diﬀerence is <5%, and for the other fil-
     For broad-band observations the nominal filter wavelength
                                                                      ters the diﬀerence is ∼10% or less (steep flux gradient combined
may diﬀer significantly from the average wavelength of the pho-
                                                                      with systematic albedo gradient).
tons registered in the polarization map. Similarly the evalua-
tion of the reflected-flux weighted albedo Ag,eﬀ for a given filter
considers a weighting with the eﬀective spectral distribution of      3. Basic data reduction
the registered photons, taking into account the wavelength de-
pendence of the instrument, the solar photon spectrum, and the        The data provided by the HST data reduction pipeline are al-
albedo of Titan.                                                      ready corrected for bias, dark, flatfield, and image distortion.
     Figure 1 illustrates the spectral dependence of the full disk    However, in the case of the ACS data the pipeline drizzled, com-
albedo of Titan from the mid-UV to the near-IR (McGrath et al.        bined images showed a strange stripe pattern which was caused
1998 for the UV, Karkoschka 1998 for the visual, Negrão et al.        by incorrectly set bits in the data quality mask that are used
2006 for the near-IR) assuming Titan’s optical radius varying         for identifying pixels flagged as cosmic rays by MultiDrizzle2 .
with λ according to the Toon et al. (1992) radius RT . The solar      Using the instructions provided by the stsdas3 helpdesk we reset
irradiance photon spectrum derived from Thuillier et al. (2004)       the dq_bits and re-run multidrizzle using the standard settings.
is given by the dashed line. The normalized instrument eﬃcien-            Both the ACS and the NICMOS instrument contain sets
cies in the diﬀerent HST filters and the calculated spectral dis-     of three linear polarizer filters with their relative polarization
tribution of the registered photons are also indicated. We use the
                                                                      2
                                                                          The MultiDrizzle Handbook, Chap. 5.4.6.3 “Final Products”.
1                                                                     3
    HST proposal ID 9385.                                                 Space Telescope Science Data Analysis System.

A6, page 2 of 13
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                        A. Bazzon et al.: HST observations of the limb polarization of Titan

Table 1. Observational parameters and summary of the HST polarime-           Table 2. Calibration parameters for ACS polarimetry.
try used in this work.
                                                                                  Spectral filter      c0       c60        c120       T      T⊥
                            ACS HRC            NICMOS 1 and 2                        F250W∗∗         1.840     1.625      1.801      0.28    0.02
   Obs. parametersa,b                                                                F330W          1.7302    1.5302     1.6451     0.475    0.05
     Date                  2002-12-02              2002-11-27                        F435W          1.6378    1.4113     1.4762     0.525    0.02
     Diameter                 0.88                  0.88                         F625W∗         1.0443    0.9788     0.9797     0.500    0.00
     Phase angle               1.8◦                    2.4◦                          F775W          1.0867    1.0106     1.0442     0.650    0.00
     NP distance             –0.39                 –0.39
     Angle NP-Ncel           −5.3◦                   −5.3◦                   Notes. (∗) Calibration not scaled for Stokes I. According to HST hand-
   Exposures                                                                 book there is also some evidence of a polarization pathology. (∗∗) No cal-
     F250W                3 × 4 × 365 s                  –                   ibration parameters available. See text for the derivation of these values.
     F330W                3 × 4 × 200 s                  –
     F435W                 3 × 4 × 45 s                  –
     F625W                 3 × 2 × 12 s                  –                   parameters indicated in Table 2. Especially for the integrated ra-
     F775W                  3×2×9s                       –                   dial polarization described in Sect. 6 these calibration parame-
     POL S                      –                 3 × 2 × 20 s               ters produce very reasonable results which are consistent with
     POL L                      –                 3 × 2 × 12 s               the other wavelengths.
   Plate scale             0.025 /pix     0.043 /pix & 0.075 /pix

Notes. The southern summer solstice of Titan occurred in late October        3.2. Calculation of Stokes parameters for NICMOS
2002. NP distance gives the angular distance of the north pole from
the center of the disk, whereas the negative distance indicates that         In case of NICMOS the HST handbook provides the user with
the north pole is on the hidden hemisphere. (a) US Naval Observatory         two coeﬃcient matrices M1;2 to calculate the Stokes parameters
& Royal Greenwich Observatory (2000). (b) http://ssd.jpl.nasa.               of NIC1 and NIC2 respectively:
gov/?horizons.                                                               ⎛ ⎞                    ⎛          ⎞
                                                                             ⎜⎜⎜ I ⎟⎟⎟              ⎜ i0 ⎟
                                                                               ⎜⎜⎜ Q ⎟⎟⎟ = M −1 ⎜⎜⎜⎜⎜ i120 ⎟⎟⎟⎟⎟
directions oriented according to 0◦ , 60◦ , 120◦ and 0◦ , 120◦ , 240◦            ⎝ ⎠        1;2 ⎝              ⎠
                                                                                   U                  i240
respectively. In the first data reduction step the basic pipeline
processed images were cut out and aligned to subpixel accuracy               with the matrices
of ±0.1 pixel. Then the images corresponding to the Stokes pa-                          ⎛                           ⎞
rameters I, Q, and U were calculated and corrected for instru-                          ⎜⎜⎜ 0.3936 0.3820 0.0189 ⎟⎟⎟
mental polarization.                                                         MNIC1 = ⎜⎜⎜⎝ 0.3959 −0.1118 −0.1463 ⎟⎟⎟⎠ ,
                                                                                            0.3902 −0.2768 0.1150
3.1. Calculation of Stokes parameters for ACS
                                                                                        ⎛                           ⎞
                                                                                        ⎜⎜⎜ 0.5094 0.3550 0.1131 ⎟⎟⎟
In case of ACS the Stokes parameters are calculated accord-                  MNIC2 = ⎜⎜⎜⎝ 0.5139 −0.0403 −0.3206 ⎟⎟⎟⎠ .
ing to:                                                                                     0.5159 −0.3262 0.3111
      
       2
 I=       [i0 · c0 (λ) + i60 · c60 (λ) + i120 · c120 (λ)] ,                  For NIC1 the coeﬃcient matrix calibration is not perfect and a
       3                                                                     residual instrumental polarization at a level of pinst. ≈ 1.2−1.5%
                                                                    
       2                                                       T + T⊥       was reported (see Batcheldor et al. 2009). Furthermore, for
Q=        [2i0 · c0 (λ) − i60 · c60 (λ) − i120 · c120 (λ)]               ,
       3                                                       T − T⊥       bright targets ghost images are present in two NIC1 polarization
                                                                         filters (i0◦ , i240◦ ). In case of NIC2 the instrumental eﬀects are
        2                                       T + T⊥
U = √ [i60 · c60 (λ) − i120 · c120 (λ)]                      ,               very well calibrated, and uncertainties as low as pinst. ≈ 0.2%
         3                                      T − T⊥                      should be achievable with bright objects. Both for NIC1 and
whereas i∗ indicate the intensity images corresponding to the                NIC2, this is in good agreement with our findings for the Titan
three polarizer orientations, c∗ are corresponding correction fac-           polarization in the disk center in Sect. 5.
tors calibrating the polarization zero-point, and the T ∗ param-
eters correct for polarization cross-talks caused by leakages of             4. Intensity images
the polarizing filters. The calibration parameters are given in the
HST calibration handbook (Fig. 5.3, Table 6.3) which we sum-                 The used HST-dataset was mainly taken to study seasonal ef-
marize in Table 2. According to the handbook for the ACS cam-                fects of the stratospheric haze on Titan, and an analysis of the
era the residual instrumental polarization uncertainty should be             spectro-photometric data is given by Lorenz et al. (2004, 2006).
at the one-part-in-ten level for highly polarized sources and at             In particular, they describe and explain in detail the varying dark
the 1% level for weakly polarized targets.                                   polar hood and the north-south asymmetry measured in diﬀerent
    For the F250W filter no calibration parameters are provided              narrow-band filters.
by the HST handbook. However, because of symmetry reasons                        The dark polar hood was first seen in 1980 by Voyager 1
one can assume that to first order the fractional polarization at            around the north, and disappeared from the south pole in
the center of the apparent disk of Titan should be zero:                     2002−2003 (Lorenz et al. 2006). Most probably, the polar hood
                                                                             is associated to a downwelling during the long polar night, redis-
(Q/I)center ≈ (U/I)center ≈ 0.
                                                                             tributing haze from the summer hemisphere towards the winter
Therefore, by using extrapolated values for c∗ as starting points,           pole (Rannou et al. 2002; Lorenz et al. 2006). This process then
and by minimizing the fractional polarization around the cen-                also creates the detached haze layer as the haze is horizontally
ter of Titan, we determined estimates for the F250W calibration              drawn from beneath the formation zone.
                                                                                                                                      A6, page 3 of 13
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                               A&A 572, A6 (2014)




                                                                            Fig. 3. Stokes Q (left) and U (right) images in the F775W band. The
                                                                            south pole and the equator are indicated. The gray scale is normalized
                                                                            to the central intensity of Stokes I by ±0.02 Icenter .


                                                                                The butterfly polarization pattern is typical for a centro-
                                                                            symmetric scattering geometry. For example, the Rayleigh-
                                                                            scattering atmospheres of Uranus and Neptune show this pat-
                                                                            tern of radial limb polarization (Schmid et al. 2006b). For all our
                                                                            data the pattern is highly symmetric, indicating that the limb po-
                                                                            larization has similar strength along the entire limb of Titan. The
Fig. 2. Intensity cuts/profiles through the disk center for the F250W
                                                                            strength of the limb polarization increases with wavelength until
band (bottom), the F435W band (middle), and the NIC1 band (top). The
solid line indicates the north-south profile through the planetary poles,   it peaks in the 1 μm band measurement after which it decreases
and the dashed line is the east-west profile perpendicular to the polar     again.
axis respectively.                                                              We do not see a significant imprint of the north-south asym-
                                                                            metry of Titan (Sect. 4) in the polarization images. To study the
                                                                            polarization along the north-south (and east-west) direction the
     Lorenz et al. (2004) find that in the narrow-band filters the          polarization pattern was aligned with respect to the polar axis of
north-south asymmetry is reversed between the blue (439 nm)                 Titan by
and the red (889 nm), and almost absent at 619 nm. In the blue
Titan is brighter in the south than in the north and vice versa in          QNS = Q cos(2θNP ) − U sin(2θNP ),
the red methane bands. In the near-IR, the variation of the asym-
metry with wavelength is dramatic, and diﬀerent narrow-band                 where θNP is the angle between the polarizer reference axis and
filters may see diﬀerent reversions of the north-south asymme-              the polar axis of Titan.
try as they are probing diﬀerent altitude regions.                              For all filters we calculate disk-integrated Stokes fluxes ΣI,
     The north-south asymmetry is also visible in our broad-band            ΣQ, and ΣU by summing up all counts within the integration
images ranging from 0.3−2 μm of the same HST visit as the                   radius Rint = 0.75 from the apparent disk center of the I, Q,
narrow-band data of Lorenz et al. (2004), and it is in qualitative          and U images respectively. Rint includes the halo of the planet
agreement with the spectro-photometric analysis by Lorenz et al.            which is greater than the nominal limb at RTitan = 0.44. We
(2004, 2006). Figure 2 shows N-S and E-W profiles of Titan for              then calculate disk-integrated fractional polarization parameters
the F330W band, the F435W band, and the NIC1 band. Intensity                for Stokes Q (and similarly U):
images for the bands F330W, F435W, F625W, NIC1, and NIC2
                                                                             Q/I m (Rint ) = ΣQ/ΣI;                                           (2)
are given in the left panel of Fig. 5. The nominal optical radius
of Titan, the south pole, and the equator are also indicated. The            Q/I m and U/I m are equivalent to a measurement with aper-
images are normalized such that                                             ture polarimetry, where the aperture is larger than the planet. The
     Rint       2π                                                          results for all filters are given in Table 3. The disk-integrated po-
                     Ir drdφ = Ag,eﬀ · πR2int ,                      (1)    larization of Titan is essentially zero (p < 0.2%) for all filters in
 0          0                                                               agreement with Veverka (1973) and Zellner (1973), except for
                                                                            F330W and for NIC1; these results are not real and can be ex-
where Ag,eﬀ indicates the eﬀective albedo given in Table 3 and              plained by instrumental eﬀects.
Fig. 1, and Rint = 0.75 is the integration radius, which is greater           As explained in Sect. 3.2 NIC1 is not well calibrated for in-
than the nominal radius of Titan RTitan = 0.44 .                          strumental polarization and between pinst. ≈ 1.2%−1.5% resid-
                                                                            ual polarization is expected. We are measuring Q/I m = 1.5%
5. Stokes Q and U images for Titan                                          despite the expectation that Titan has zero net polarization. The
                                                                            same instrument oﬀset is also seen at the center of the disk which
In Fig. 3 Stokes Q and U images in the F775W band are shown.                should be zero because of symmetry reasons. Similarly for the
The same highly symmetric quadrant pattern is present in all of             F330W measurement the center of the Stokes Q and U images is
our data. This butterfly pattern is real and it is practically impos-       not zero indicating residual instrumental polarization at the level
sible to artificially create such a pattern by misalignments of the         of pinst. ≈ 0.8% in Q and pinst. ≈ 0.2% in U respectively.
three polarization images i∗ or other spurious eﬀects. The gray                 The absence of a net polarization in the disk averages in-
scale is normalized to the central intensity Icenter of the planetary       dicates that the limb polarization has a similar strength along
disk and goes from −0.02 Icenter (black) to +0.02 Icenter (white).          the entire limb for all observed bands. We note that the disk-
At the center Q and U are essentially zero (Q/I, U/I  ±0.2%).              integrated parameters Q/I m and U/I m can hardly be aﬀected
A6, page 4 of 13
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                           A. Bazzon et al.: HST observations of the limb polarization of Titan

Table 3. Polarization results for Titan.

                                                       Integrated polarization             Integrated radial polarization           Max. radial pol.
  Filter        λc       λeﬀ     Ag,eﬀ         RT        Q/I m       U/I m           Qr /I m     Ur /I m      CPSF        Qr /I   (Qr /I)max
                                                                                                                                         m   (Qr /I)max
              [nm]      [nm]                  [km]        [%]         [%]             [%]         [%]                     [%]       [%]         [%]
  F250W        273       299    0.042         2901        0.13        0.00           0.99         0.03        0.84        1.18     1.82         4.18
  F330W        336       341    0.051         2892       –0.81       –0.36           1.10        –0.05        0.85        1.30     2.37         4.40
  F435W        432       448    0.097         2871        0.06        0.08           1.71        –0.09        0.86        1.98     3.44         5.31
  F625W        632       636    0.219         2835        0.00        0.19           2.51        –0.09        0.85        2.95     4.93         6.86
  F775W        768       762    0.211         2813        0.09        0.03           2.97        –0.14        0.82        3.63     5.19         8.70
  NIC1        1071      1018    0.155         2771       –1.49       –1.40           4.03         0.04        0.74        5.45     6.90        10.49
  NIC2        2002      2022    0.094         2651       –0.30       –0.09           1.99         0.00        0.58        3.42     3.44          –∗

Notes. The columns give the filter, the central wavelength λc , the eﬀective wavelength λeﬀ , the eﬀective albedo Ag,eﬀ for the photons registered
in the broad-band filters, and the Toon et al. (1992) radius RT for λeﬀ . Q/I m , U/I m , Qr /I m , and Ur /I m are the measured disk-integrated
polarization parameters. The parameter CPSF describes the degradation of the polarization measurement due to the PSF smearing eﬀect (Sect. 6.4),
and the corrected radial polarization value is given by Qr /I = Qr /I m /CPSF . (Qr /I)max m is the measured maximum radial polarization, whereas
(Qr /I)max is the modeled value for infinite resolution. The statistical 1σ error for the disk-integrated polarization is estimated to be Δp ≤ ± 0.1%
and ΔCPSF = ±0.01. In addition, a systematic uncertainty of (Δp)syst. ≈ ±0.2% is estimated. (∗) wavelength range of model: 200 nm < λ < 1600 nm.


by inaccuracies in the image centering procedure or other spu-
rious eﬀects due to the data reduction. The statistical 1σ mea-
suring error is Δp < 0.1% as estimated for the measuring error
of the integrated radial polarization Ur /I m in Sect. 6 which is
independent of any residual instrumental oﬀset.

6. The radial polarization
The polarization flux of an object is given by p × I = Q2 + U 2 .
However, because of the squares in this formula, large system-
atic bias errors are introduced if the absolute value of one or
both measured signals |Q| and |U| is not significantly higher                    Fig. 4. Radial Stokes Qr (left) and Ur (right) images in the F775W band.
                                                                                 The south pole and the equator are indicated. The gray scale is normal-
than the measuring noise ΔQ and ΔU. In our Titan data there                      ized to the central intensity of Stokes I by ±0.02 Icenter .
is ΔQ ≈ |Q| and ΔU ≈ |U| in the middle of the planetary disk,
and between positive and negative quadrants in the butterfly pat-
tern. Therefore one should not use the polarized flux p × I or
the normalized polarization p as measuring parameter. We adopt
radial Stokes parameters, which are particularly well-suited for                 which is expected to be unpolarized. Figure 4 shows Qr and Ur
characterizing centro-symmetric polarization patterns of planets                 for Titan in the F775W filter. In all of our data the limb polariza-
(e.g., Schmid et al. 2006b).                                                     tion is clearly visible as a bright ring with positive Qr polariza-
    The radial Stokes parameters Qr and Ur describe the polar-                   tion and essentially zero Ur polarization. Except for the F250W
ization in radial and tangential direction on the disk of Titan.                 filter, the level of the Ur polarization is typically about 10 times
They are given by                                                                lower than the positive Qr signal along the limb. For the F250W
                                                                                 filter the Ur polarization is about 3 times lower than Qr . Thus Qr
Qr = +Q cos 2φ + U sin 2φ,                                              (3)      dominates in all filters. In Fig. 5 we show images of Qr as well
Ur = −Q sin 2φ + U cos 2φ,                                              (4)      as corresponding Stokes I for the rest of our data. The gray scale
                                                                                 of the radial polarization images in Figs. 4 and 5 is scaled to the
where φ is the polar angle of a given position (x, y) on the ap-                 central intensity Icenter by ±0.02 · Icenter , and the intensity images
parent planetary disk (disk center (x0 , y0 )) with respect to the               are normalized to the eﬀective albedo Ag,eﬀ according to Eq. (1).
polarizer reference direction:
             x − x0                                                              6.2. Polarization as function of radius
φ = arctan          ·
             y − y0
                                                                                 The observed polarization of Titan is essentially centro-
Qr > 0 is equivalent to a radial polarization or a polarization                  symmetric, and contrary to the strong north-south albedo asym-
perpendicular to the limb, while Qr < 0 indicates a tangential                   metry, the corresponding imprint in the radial polarization flux
polarization component. Ur describes the polarization in the di-                 is either absent or much weaker. In the fractional radial polariza-
rections ±45◦ with respect to the radial direction.                              tion images Qr /I we see marginal north-south diﬀerences with
                                                                                 higher polarization in the north for the bands shorter than 1 μm,
                                                                                 the opposite eﬀect in the NIC1 band, and about equal polariza-
6.1. Radial stokes Qr and Ur images for Titan
                                                                                 tion for the NIC2 band. However, these results are not signifi-
For all observed bands we calculated radial polarization images                  cant and on the order of our systematic uncertainties Δp/p ≈
Qr and Ur . In case of the F330W and NIC1 observations the                       0.05−0.1.
Stokes Q and U images were first corrected for residual instru-                       In the infrared there is a similar weak indication for an
mental polarization (see Sect. 5), derived from the disk center                  east-west asymmetry of Qr /I, whereas for the NIC1 band the
                                                                                                                                        A6, page 5 of 13
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                               A&A 572, A6 (2014)




                                                                            Fig. 6. Titan radial profiles for intensity I (upper panel, thick lines) and
                                                                            radial polarization Qr (upper panel, thin lines) in the F435W (dash-dot),
                                                                            F775W (dash) and NIC1 (solid) filters. Both I and Qr are normalized
                                                                            to the peak flux at r = 0. The lower panel shows the corresponding
                                                                            normalized radial polarization Qr /I.



                                                                            the poles. However, the quality of our data is not good enough
                                                                            to draw firm conclusions, and at least part of the polarization
                                                                            diﬀerences mentioned above could be due to systematics in the
                                                                            data reduction, e.g., slight misalignments of corresponding po-
                                                                            larization images coupled with strong intensity gradients at the
                                                                            limb.
                                                                                The detection of structure in the limb polarization would be
                                                                            very interesting for investigating local haze properties, e.g., such
                                                                            as particle size diﬀerences of the photochemical haze between
                                                                            the morning and the evening limb of Titan. Observations with
                                                                            higher polarimetric sensitivity and higher spatial resolution are
                                                                            required for such studies.
                                                                                For the moment, assuming a rotational symmetry for the po-
                                                                            larization structure seems to be a reasonable first approximation,
                                                                            and we construct rotationally averaged, radial profiles for the
                                                                            polarization, the normalized polarization, and the intensity. The
                                                                            results for the F435W, F775W, and NIC2 filters are shown in
                                                                            Fig. 6.
Fig. 5. Intensity (left) and radial polarization images Qr (right) in the
F330W, F435W, F625W, NIC1, and NIC2 band (top to bottom). The
                                                                                In all filters the radial profiles look very similar. The polar-
                                         R 2π                               ization Qr in the disk center at R = 0 is essentially zero. The nor-
intensity images are normalized to 0 0 Irdrdφ = Ag,eﬀ · πR2 and
                                                                            malized radial polarization Qr /I increases steadily with radius
the gray scale of the polarization images is scaled to ±2% of the central
intensity Icenter .                                                         until it peaks at around RTitan = 0.44 after which it decreases
                                                                            again until the photon noise starts dominating the measurements.
                                                                            Similarly, the radial polarization flux Qr also increases with ra-
                                                                            dius but only up to a radius slightly smaller than RTitan , and then
polarization seems to be higher in the east, which is inverted              it decreases farther out to zero in step with the intensity profile.
for the NIC2 band. It also seems that for wavelengths shorter                   Both Qr /I and Qr increase with wavelength until they peak
than 1 μm the maximum polarization is higher at the eastern                 in the NIC1 band at 1 μm. Then, in the NIC2 band at 2 μm the po-
and western limbs than in the north and south, whereas for the              larization has again significantly dropped. Using tentative fits for
NIC1 band the polarization in east-west and north-south is of               the radial profiles, we measure maximum fractional radial polar-
about equal strength, and for NIC2 the polarization is higher at            ization values Qr /I of 1.8% for the F250W band, up to 6.9% for
A6, page 6 of 13
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                     A. Bazzon et al.: HST observations of the limb polarization of Titan

the NIC1 band, and 3.4% for the NIC2 band. The values for all
filters are given in Table 3. However we note that these values
are not yet corrected for the PSF smearing eﬀect described in
Sect. 6.4.

6.3. Disk-integrated radial polarization
Similar to the calculation of disk-integrated Stokes parameters
in Sect. 5 disk-integrated radial Stokes parameters Qr /I m and
 Ur /I m are calculated for all filters. Because of the intrinsic ro-
tational symmetry of Eqs. (2)−(4), Qr /I m and Ur /I m have the
additional advantage that any instrumental polarization oﬀset or
gradient cancels out. Therefore, we do not need to correct the
instrumental oﬀset described in Sect. 5 for the F330W and NIC1
filter to calculate corresponding disk-integrated radial polariza-
tion values. This was also verified by calculating Qr /I m and
 Ur /I m for the NIC1 data, both with and without correction of
the instrumental oﬀset, which showed that the diﬀerence is less         Fig. 7. Encircled energies for HST PSFs (thick lines) and Gaussian
than Δp = 0.01%.                                                        PSFs (thin lines): ACS F625W (solid), NIC1 POL0S (dashed), and
     A strong positive signal is obtained for the disk-integrated       NIC2 POL0L (dash-dot).
radial polarization Qr /I m , while Ur /I m is essentially zero
(Table 3). The 1σ measuring error is Δp < 0.1% as estimated
from the assumption U/I m = 0. Both Qr /I m and Ur /I m                     Because of these extended PSF wings, the opposite polar-
are only very weakly aﬀected by small asymmetries caused by             ization components +Q and −Q overlap and cause a reduction
inaccuracies in the image centering procedure or other spuri-           in the resulting net polarization. In the most extreme case of
ous eﬀects due to the data reduction. However, strong asymmet-          an unresolved centro-symmetric planetary disk, the polarization
ric perturbations such as strong ghosts could probably bias the         cancelation would be perfect and only a zero net polarization
result. Anyway, except for NIC1 we do not see any ghosts in             level could be measured. The compensation eﬀect is stronger for
our Qr and Ur images (Sect. 6.1). In case for NIC1 it is known          longer wavelengths, where the diﬀraction limited spatial resolu-
that weak ghosting is present for bright sources but we estimate        tion of HST is not as good, and at the same time the PSF wings
that the impact on the integrated radial polarization is less than      are stronger.
Δp/p = 0.01 (see also Sect. 7.3).                                           For an estimate of the polarization cancelation, we adopted
     The integrated radial polarization Qr /I is a good param-          our haze scattering model for the expected polarization pattern
eter for characterizing the overall limb polarization of a planet.      (see Sect. 7.2). From the model we constructed two-dimensional
Since Qr is either positive or close to zero everywhere on the disk     intensity images for i0 , i90 , i45 , and i135 , from which the cor-
no polarization compensation eﬀect is present. Furthermore,             responding Stokes Q and U images can be calculated by Q =
there is Qr /I  Ur /I ≈ 0, so that we can approximate                  i0 − i90 and U = i45 − i135 . Similarly, we constructed smeared
                                                                        Stokes images Qs and Us by folding the i∗ images with the sim-
    pr =   Qr /I 2 + Ur /I 2 ≈ Qr /I .                                  ulated HST PSFs. Figure 8 illustrates the cancelation eﬀect in the
                                                                        Stokes Q image due to the HST PSF of ACS at F625W, NIC1 at
                                                                        1 μm, and NIC2 at 2 μm.
6.4. Correction for the PSF smearing effect                                 The Q and U images can then be converted into radial po-
                                                                        larization images Qr and Ur , as for the observations. From Qr
The point spread functions (PSFs) of HST ACS and NICMOS                 the integrated radial polarizations Qr /I and Qr /I s are calcu-
have a finite width given by the telescope diﬀraction ∼λ/D, and         lated for the diﬀerent filters in the same way as for the observa-
they are aﬀected by optical aberrations, geometric distortions,         tions. The ratio between the clean and the smeared polarization
and in case of ACS a long-wavelength halo produced by the de-           then yields the factor for the expected degradation of the disk-
tector itself. This leads to extended PSF wings which limit the         integrated radial polarization
resolution of the HST observations. Therefore, the measured in-
tegrated limb polarizations Qr /I m and Ur /I m need to be cor-                   Qr /I s
rected by the inverse of a degradation factor CPSF , accounting for     CPSF =            ·                                             (5)
                                                                                  Qr /I
a polarization cancelation due to PSF smearing.
    We produced simulated ACS and NICMOS PSFs for all                   The corresponding values of CPSF and the corrected limb polar-
wavelengths, using the Tiny Tim4 PSF simulation software pack-          ization Qr /I for all filters are given in Table 3. Especially for
age for the HST (see Krist et al. 2011). Figure 7 compares the en-      NIC2 the degradation factor CPSF = 0.58 is low, while for the
circled energy of these PSFs to the encircled energy of Gaussian        ACS bands and for NIC1 the degradation factor is about CPSF =
PSFs with FWHM = λ/D. One can see that for the Gaussian                 0.85 and CPSF = 0.74 respectively. The statistical 1σ error of the
PSFs essentially all the energy is contained within half the ra-        degradation factor is estimated to be about ΔCPSF = ±0.01.
dius of Titan. However, in the NIC2 band up to 45% of the en-               The polarization cancelation depends not on the strength, but
ergy is smeared over an area larger than half of the radius of          on the geometric structure of the polarization pattern. Rayleigh
Titan. For the ACS filters and NIC1 the fraction originating from       scattering models indicate that the polarization pattern is very
R > 0.5 RTitan is considerably smaller but still about 15% and          similar for the diﬀerent model parameters (see Schmid et al.
20% respectively.                                                       2006b). Thus the degradation depends not significantly on the
                                                                        exact haze scattering parameters of the planet. A sanity check
4
     http://tinytim.stsci.edu                                           by recalculating CPSF using a Rayleigh scattering model indeed
                                                                                                                           A6, page 7 of 13
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                                A&A 572, A6 (2014)




                                                                             Fig. 9. Vertical optical depth τ for λ = 445 nm (dash-dot), λ = 775 nm
                                                                             (dash), λ = 940 nm (dot), λ = 1000 nm (methane absorption band,
                                                                             solid), and for λ = 1580 nm.
Fig. 8. Modeling of the degradation of the Titan polarization Q due
to the cancelation of opposite polarization components +Q and −Q
caused by the limited resolution; a) unlimited resolution; b) HST PSF        around 500 km altitude (Porco et al. 2005). Above 700−800 km
at 630 nm (F625W); c) HST PSF at 1 μm (NIC1); and d) HST PSF at              most of the methane is destroyed by photolysis.
2 μm (NIC2). The gray scale spans for all panels the range from −2%              For simplification, we assume the atmospheric composition
(black) to +2% (white) of the peak intensity of the initial perfect inten-   to be solely made out of methane and nitrogen. The methane
sity image.                                                                  mole fraction fCH4 (h) between h = 0−144 km altitude was taken
                                                                             from Brown et al. (2010), whereas above 140 km fCH4 (h) was
shows that within our 1σ error bars the results for CPSF were                assumed to linearly drop to zero at an altitude of 600 km. We
identical.                                                                   use the temperature T (h) and density ρ(h) profiles provided by
                                                                             the Huygens Atmospheric Structure Instrument (HASI)5 , and the
                                                                             total molecular number densities and the Rayleigh scattering op-
7. Comparison with limb polarization models                                  tical depth τRay (h, λ) are then calculated from the ideal gas law
                                                                             and the gas column density (see Appendix).
Titan is an excellent test case for detailed studies of the scatter-
                                                                                 For the methane absorption coeﬃcients κ(T, λ) we use the
ing polarization from a hazy atmosphere, and accurate scattering             formula given by Karkoschka & Tomasko (2010). We note that
and polarization parameters are available from the in situ mea-              below λ = 1 μm these coeﬃcients are generally close to those
surements of the Huygens landing probe (e.g., Brown et al. 2010;
                                                                             by Karkoschka (1998).
Tomasko et al. 2008). In the next section we describe the ba-                    A detailed model of the aerosol properties of Titan is given
sic atmospheric structure that we used for our radiative transfer
                                                                             by Tomasko et al. (2008), based on measurements of the DISR
model, which is described in Sect. 7.2, and in Sect. 7.3 we com-
                                                                             instrument on the Huygens landing probe (Tomasko et al. 2002).
pare the model with our limb polarization measurements and lit-              The haze optical depth per unit path length τhaze (h, λ) is derived
erature values for the geometric albedo Ag and the quadrature
                                                                             for three altitude regions, i.e., above 80 km, between 30−80 km,
polarization p(90◦ ) of Titan.
                                                                             and below 30 km. On the one hand, Tomasko et al. (2008) de-
                                                                             scribe the wavelength dependence by three diﬀerent power laws,
7.1. Atmospheric parameters                                                  corresponding to the three altitude regions. Then on the other
                                                                             hand, the cumulative optical depth increases with decreasing al-
For our model we assume an atmosphere ranging from                           titude. Between 0−30 km and 30−80 km the increase is linear
0−1300 km. This includes Titan’s troposphere with its well                   but with two diﬀerent slopes, and above 80 km the increase is
defined tropopause at ∼44 km (112 mbar), the stratosphere                    exponential with a scale height of 65 km.
with the stratopause located at ∼260−310 km (0.22−0.08 mbar;                     The vertical optical depth is shown in Fig. 9 for five diﬀerent
Fulchignoni et al. 2005; Vinatier et al. 2007), the mesosphere               wavelengths between λ = 445 nm and λ = 1580 nm, including
with the mesopause at ∼494 km (0.002 mbar; Fulchignoni et al.                the methane absorption band at λ = 1 μm. For wavelengths λ <
2005), and the thermosphere ranging up to ∼1300 km (1.4 ×                    1 μm most of the light is scattered in the stratosphere between
10−8 mbar).                                                                  ∼100−300 km altitude where τ ≈ 1, whereas for λ = 1.6 μm the
     The atmospheric composition is predominantly N2 , with                  light penetrates down to about 30 km altitude. The impact of the
CH4 and H2 the second and third most abundant molecules re-                  methane absorption is particularly strong in the troposphere but
spectively. Near the surface CH4 has an abundance of ∼5%,                    has almost no eﬀect for higher altitudes.
falling to ∼1.4% in the stratosphere (Brown et al. 2010). Because                Similar to τ, the single scattering albedo ωhaze (h, λ) can
of its long chemical lifetime H2 is essentially uniformly mixed              be split into wavelength and altitude dependent parts, whereas
throughout the atmosphere with a mixing ratio ∼0.1% (Courtin
et al. 2008). A thick haze layer is located in the stratosphere, and         5
                                                                               http://atmos.nmsu.edu/PDS/data/hphasi_0001/DATA/
a second detached haze layer lies just above the mesopause at                PROFILES/

A6, page 8 of 13
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                     A. Bazzon et al.: HST observations of the limb polarization of Titan

typically ωhaze (h, λ) ≈ 0.8−1. We adopt the altitude model of
Tomasko et al. (2008). Between 0−30 km, 30−80 km, and above
144 km respectively, we assume ωhaze (h, λ) to be constant with
altitude, using the values given in Table 2 of Tomasko et al.
(2008), whereas between 80−144 km we linearly interpolate be-
tween the adjacent regions. For the region between 80 km to
144 km Tomasko et al. (2008) suggest that new material is in-
corporated in the aerosols as they fall, and that the aerosols grow
in size. In the few kilometers above the surface they also see
some weak evidence of a decrease of ωhaze (h, λ), i.e., reversing
the general trend with altitude. The wavelength dependence of
ωhaze for the region above 144 km and 30−80 km is given in
Fig. 48 of Tomasko et al. (2008), and the same dependence is
assumed for the region 80−144 km. For the region below 30 km
we adopt a three-dimensional polynomial fit to the values given
in Table 2 of Tomasko et al. (2008).                                    Fig. 10. Wavelength dependence of the single scattering polarization
     For the surface we assume a diﬀusely scattering surface with       parameter pm defined by Eq. (7). The fit values to the Tomasko et al.
constant albedo As = 0.2 for all wavelengths. This is a strong          (2008) curves for the blue and the red channel are also indicated.
simplification and according to the literature the surface albedo
of Titan varies between As = 0.1−0.3, depending on wavelength
and also the season of Titan (e.g., McKay et al. 1989; Tomasko          where θ is the scattering angle. The code does not consider cir-
et al. 1997, 2008). Anyway, we tested diﬀerent models using a           cular polarization because these eﬀects are expected to be very
range of As = 0.1−0.3, and the variation from p(0◦ )|As = 0.2 is less   small and negligible for a simple scattering model.
than Δp(0◦ )/p(0◦) ≈ 0.1.                                                   We used the aerosol scattering phase functions F11 (θ) given
     The wavelength range of our model is restricted by the             in tabulated form by Tomasko et al. (2008) for two altitude re-
Tomasko et al. (2008) values for τhaze (h, λ) and ωhaze (h, λ),         gions, above 80 km and below 80 km, and for diﬀerent wave-
which are only given for a wavelength range of 400−1600 nm.             lengths ranging between 355 nm to 5166 nm. Typically a phase
On the one hand, we extended this range towards the UV by ex-           function is given about every 100 nm for λ < 1 μm and about
trapolating the parameters down to 200 nm. To first order this          every 200 nm for 1 μm < λ < 1.5 μm, and for our model calcu-
is valid because between 200−400 nm we do not expect strong             lations we linearly interpolate between these values. Anyway, it
spectral or altitudinal features in τhaze (h, λ) and ωhaze (h, λ). On   turned out that for the polarization results the dependence on λ
the other hand, we did not extrapolate the parameter range to-          is very weak. This was tested by running diﬀerent model im-
wards the red end because above λ ≈ 1 μm the spectral variation         plementations using only the tabulated phase functions with the
in ωhaze is more complex, and because the extrapolation interval        closest match to λ, and within our observational error bars the
to include the NIC2 band at 2.2 μm was too large. Therefore, the        calculated limb polarization was the same.
final spectral range of the model covers 200−1600 nm, which                 Tomasko et al. (2008) also derive a single scattering polar-
includes all our polarization data except the NIC2 polarimetry.         ization fraction (−F12 /F11 ) for the blue (470−530 nm) and the
                                                                        red (880−970 nm). To first order, we find a very tight fit to their
7.2. Radiative transfer code                                            model using a scaled Rayleigh-like single scattering polarization
                                                                        dependence according to
We use an extended version of the Monte Carlo scattering code
described by Buenzli & Schmid (2009). Basically, the code cal-           F12 (θ)     cos2 (θ) − 1
culates random walk histories of many photons entering the at-                   = pm 2           ,                                      (7)
                                                                         F11 (θ)     cos (θ) + 1
mosphere, and follows their direction and polarization change
until they are absorbed or they escape. The intensity and polar-        with pm = 0.920 for the blue channel and pm = 0.975 for the red
ization spectra of the planet can then be established for diﬀer-        channel. For our model we use pm (λ) with a linear slope, going
ent lines of sight, and in the case for backscattering (α = 0◦ )        through the values of the red and blue channel, and pm (λ) = 1
as a function of radial distance from the disk center. For the          above λ ≈ 1 μm as shown in Fig. 10.
calculation, the spherical model atmosphere is assumed to be
                                                                            For F33 (θ) we use the same dependence as for Rayleigh scat-
rotationally homogeneous, consisting of multiple locally plane
                                                                        tering according to
parallel layers. The incident radiation is a parallel beam of
unpolarized photons, whereas despite multiple scattering, the
                                                                         F33 (θ)   2 cos(θ)
photons emerge at the same point where they entered into the                     =            ·                                          (8)
atmosphere.                                                              F11 (θ) cos2 (θ) + 1
    The scattering processes are described by probability den-
sity functions, derived from the appropriate phase matrices of          The atmospheric parameters used for our model are described in
the scattering particles (see also Schmid 1992). For scattering         the previous section. Our calculations include Rayleigh scatter-
on haze particles, the code allows for scattering matrices of the       ing, aerosol scattering, and methane absorption but we neglect
form                                                                    Raman scattering, which has only a very small eﬀect on the re-
                                                                        flectivity and polarization (e.g., Sromovsky 2005). The model
          ⎛                                  ⎞                          atmosphere consists of 47 diﬀerent layers above a diﬀusely scat-
          ⎜⎜⎜ F11 (θ) F12 (θ)     0     0⎟⎟
                                             ⎟
            ⎜⎜⎜ F12 (θ) F11 (θ)   0     0⎟⎟⎟⎟                           tering surface, and the models are run with 109 −1010 photons
F(θ) = ⎜⎜⎜⎜                                    ,
                                F33 (θ) 0⎟⎟⎟⎟⎠
                                                              (6)       depending on wavelength, such that the statistical error of the
              ⎜⎝ 0        0
                  0       0       0     0                               fractional polarization is Δp ≤ ±0.1%.
                                                                                                                            A6, page 9 of 13
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                             A&A 572, A6 (2014)




Fig. 11. Comparison of our model results for the limb polarization with
literature values. Top panel: geometric albedo Ag from the literature
(dashed, see also Fig. 1) and from our model (solid). Bottom panel:
integrated limb polarization p(0◦ ) of our model for the complete wave-
length range (solid), and the HST filter pass bands (+), as well as our
HST polarimetry observations (×) given in Table 3.

                                                                          Fig. 12. Comparison of our model results at quadrature phase α = 90◦
7.3. Results                                                              with literature values. Top panel: reflectivity f (90◦ ). Middle panel: in-
                                                                          tegrated quadrature polarization p(90◦ ) of our model for the complete
Our model calculates the geometric albedo Ag , the quadrature             wavelength range (solid), and the results from the Pioneer 11 () and
polarization p(90◦ ) = Q/I (90◦ ), and the limb polarization              Voyager 2 () spacecrafts. The model results in the Pioneer 11 pass
p(0◦ ) = Qr /I . Figure 11 compares in the top panel the cal-             bands are also indicated (+). Bottom panel: integrated polarization flux
culated geometric albedo with observational data (see Sect. 2).           p × f (90◦ ).
We find a good qualitative agreement for the complete wave-
length range. Above 500 nm the agreement outside strong ab-
sorption bands is better than ΔAg /Ag = 0.2, whereas for the              of the PSF smearing could explain a Δp/p of a few percent but
strong absorption around 1.15 μm and 1.45 μm the agreement                certainly not the full discrepancy between model and data.
is ΔAg /Ag ≈ 0.3. Below 500 nm the agreement gets worse with                   Concerning the model parameters for the NIC1 wavelength
decreasing wavelength. At 400 nm it is ΔAg /Ag ≈ 0.4 and at               range and their impact on Qr /I , we tested the eﬀect of the
300 nm it is only ΔAg /Ag ≈ 0.75. We note that the disagreement           surface albedo As , the single scattering polarization parame-
below 400 nm is not alarming because there the albedo is very             ter pm , and the methane fraction as functions of altitude. Using
low. Furthermore, it could also origin from our questionable ex-          a range of As = 0.1−0.3 showed that the impact of As is less
trapolation of the Tomasko et al. (2008) haze parameters from             than Δp/p ≈ 0.1. Similarly, setting pm = 1 everywhere cannot
400 nm to 200 nm.                                                         explain the discrepancy and the eﬀect is less than Δp/p ≈ 0.05.
     The limb polarization results for the model and our mea-             Finally, we set the methane mole fraction fCH4 = 0 for h > 80 km
surements are shown in the bottom panel of Fig. 11. Between               to check the impact of the methane absorption in the stratosphere
400−900 nm the agreement is good with Δp/p ≈ 0.1 for                      and mesosphere. This basically gets rid of the absorption dips
the F625W pass band, and Δp/p ≤ 0.05 for the F435W and                    in Qr /I but only has a minor impact outside the absorption
F775 pass bands. Below 400 nm the model seems to system-                  bands. Since Qr /I is flux weighted, we conclude that the over-
atically overestimate the polarization, yielding a discrepancy of         all impact of the methane absorption on the limb polarization is
Δp/p ≈ 0.3. Qualitatively this UV-oﬀset agrees with our result            much less than Δp/p ≈ 0.1. Therefore, the cause of the disagree-
for the geometric albedo Ag , which also seems to be systemati-           ment between the model and the NIC1 measurement remains
cally too high in the UV. Above 900 nm our NIC1 polarization is           uncertain.
much higher than the model result and Δp/p ≈ 0.25. We could                    Titan full disk phase curves for intensity and polarization
not conclusively determine whether this discrepancy is caused             have been obtained by the Pioneer 11 (Tomasko & Smith 1982)
by an issue of the measurement, the modeling, or both.                    and the Voyager 2 (West et al. 1983) spacecrafts. The Pioneer 11
     It is known that the polarimetric calibration of NIC1 has            data were obtained in the B and R bands, covering phase angles
some deficiencies such as residual instrumental polarization of           between α = 28◦ −96◦ , whereas the Voyager 2 data were taken
pinst. ∼ 1.5% and weak ghosting (see Sect. 3.2). However, be-             in the near UV at 264 nm and the near IR at 750 nm, covering
cause of the intrinsic symmetry of the radial Stokes parame-              phase angles α = 2.7◦ −154◦. The second panel of Fig. 12 com-
ters Qr and Ur the residual instrument polarization has no ef-            pares our model with the quadrature polarization at α = 90◦
fect on Qr /I (see Sect. 6.3). The ghosting on the other hand is          measured by Pioneer 11 and Voyager 2. For three bands we
asymmetric and thus could have an impact on Qr /I . However,              find a good agreement better than Δp/p ≈ 0.1, whereas for the
in our data we only see very weak ghosts which we do not be-              Pioneer 11 R band the agreement is not as tight but still at the
lieve to have an eﬀect at the percent level. Finally, it could also       level Δp/p ≈ 0.25.
be that the real PSF is diﬀerent from the adopted PSF used for                 We note that the quadrature polarization p(90◦ ) is generally
the eﬃciency correction CPSF (Sect. 6.4). An overcompensation             higher in the absorption bands because multiple scattering is
A6, page 10 of 13
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                                    A. Bazzon et al.: HST observations of the limb polarization of Titan

strongly reduced, and the reflection is dominated by strongly
polarized single scattering at ∼90◦. For the limb polarization
p(0◦ ) this Ag − p correlation is reversed. Multiple scattering and
a low single scattering albedo ω are required for producing a
strong polarization signal since the total polarization is mainly
produced by second order but also higher order scatterings.

8. Discussion and conclusions
We present disk resolved imaging polarimetry of Titan, mea-
sured with the HST for the UV to the near-IR spectral region.
From these observations, we derive the disk-integrated radial
limb polarization Qr /I (Table 3) for various filter pass bands,
and we compare our results with the polarization of a model at-
mosphere. For the model we use a radiative transfer code pre-
sented in Buenzli & Schmid (2009), adopting Titan atmosphere
parameters from the literature, which were mainly derived from
                                                                       Fig. 13. Measured radial polarization profile for Titan in the F775W
the Huygens landing probe (e.g., Tomasko et al. 2008).                 band (dashed) versus the modeled profile without PSF smearing (solid).
    The geometric albedo Ag and the quadrature polarization
p(90◦) are important reference quantities for characterizing the
scattering properties of a reflecting atmosphere. Therefore, we
derive the reflected-flux weighted geometric albedo Ag,eﬀ for the           The observed maximum limb polarization is averaged down
used filter pass bands (Table 3), using the spectrophotometry of       by the limited resolution and the PSF structure of HST. This has
McGrath et al. (1998), Karkoschka (1998), and Negrão et al.            been taken into account in our analysis of the measured disk-
(2006); and we compare our model results for p(90◦ ) with mea-         integrated radial polarization Qr /I m . Using synthetic PSF pro-
surements obtained by the Pioneer 11 (Tomasko & Smith 1982)            files for HST we have modeled the resolution eﬀect on the po-
and the Voyager 2 (West et al. 1983) spacecrafts.                      larization, and we derive corrected values for the intrinsic Qr /I
    A comparison between the model and our observations for            of Titan (Table 3). We find Qr /I ≈ 1.2% in the UV, increasing
 Qr /I (λ), as well as a comparison between our model and liter-       to about 5.5% at 1 μm, and then decreasing again to about 3.4%
ature values for the geometric albedo Ag (λ) and the quadrature        at 2 μm. For comparison, a semi-infinite, conservative (ω = 1)
polarization p(90◦ , λ) are given in Figs. 11 and 12.                  Rayleigh-scattering atmosphere produces a disk-integrated limb
                                                                       polarization Qr /I ≈ 2.75 % (Buenzli & Schmid 2009).
8.1. Detection of the limb polarization.                                    Using our radiative transfer model for Titan we predict the
                                                                       maximum limb polarization without PSF smearing, e.g., for ob-
For all observed filter bands, our data show a strong limb po-         servations with larger earth-bound telescopes or from space-
larization of several percent, as expected from previous polar-        crafts close to Titan. We find (Qr /I)max ≈ 4.2% for the F250W
ization measurements of Titan taken at larger phase angles. To         band, up to (Qr /I)max ≈ 8.7% for the F775W and (Qr /I)max ≈
our knowledge, this is the first time that the limb polarization       10.5% for the NIC1 band, which is even larger than ≈8%
of Titan has been measured, and we did not find any previous           for a semi-infinite conservative Rayleigh-scattering atmosphere
earth-bound imaging polarimetry which resolved Titan.                  (Buenzli & Schmid 2009). The modeled (Qr /I)max for all filters
    Within the resolution limits of the observations, the mea-         are also given in Table 3, and Fig. 13 compares the measured
sured limb polarization for Titan is centro-symmetric. This is         F775W radial polarization profile with the model result. In the
similar to observations of Uranus and Neptune (Schmid et al.           figure one can see both the polarization dilution, as well as the
2006b) but for Titan the polarization is much stronger. On the         decreased resolution provided by the HST.
other hand, similar polarization strength can be found in obser-
vations of Jupiter but there the limb polarization is essentially
only present at the poles (e.g., Schmid et al. 2011), indicating       8.2. Comparison with model calculations
thick polar haze layers and non-polarizing reflection from the
clouds along the equator.                                              Our model calculates the intensity and the polarization spectra
    The polarimetric sensitivity and the resolution of our data        of the planet at diﬀerent phase angles α = 0◦ −180◦ , and in case
are not suﬃcient to detect variations of the polarization signal       of backscattering (α = 0◦ ), the limb polarization as a function of
along the limb of Titan, as it is seen in albedo observations. In      the radial distance from the disk center.
Sect. 6.2 we pointed to some tentative limb polarization struc-            Overall, we find a good agreement at the level of (ΔAg )/Ag ≈
ture, which could be present in our data. Additional observations      0.1−0.2 or (Δp)/p ≈ 0.1 respectively between model, litera-
of Titan with increased polarimetric sensitivity and resolution        ture values, and our observations. In one filter (NIC1 at 1 μm)
are required to progress in this direction.                            we find a discrepancy of Δp/p = 0.25 between the measured
    Assuming rotational symmetry, we derive center-to-limb             limb polarization Qr /I and our model, for which the cause
profiles for the radial Stokes parameter Qr (r)/I(r). Because of       is still uncertain. Despite this outlier, our analysis shows that
the scattering symmetry, Qr (r)/I(r) is essentially zero in the disk   limb-polarization measurements potentially oﬀer an additional
center, whereas the polarization increases for larger radii, reach-    diagnostic tool for investigating the properties of scattering par-
ing a maximum in the seeing halo at Rmax > RTitan . Depending          ticles in Titan with earth-bound observations. Since our model
on wavelength, we measured maximum fractional polarization             assumes a rotationally homogeneous atmosphere, our results fur-
values in the range of ∼2−7% with the highest value obtained in        ther show that the locally derived haze and atmosphere parame-
the NIC1 band at 1 μm.                                                 ters from the Huygens probe are indeed representative for Titan.
                                                                                                                           A6, page 11 of 13
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                            A&A 572, A6 (2014)

    Additional observations of Titan with increased resolution           reduced by Δp/p ≈ 0.4 to about p(90◦) ≈ 25%. The polariza-
and a high polarimetric sensitivity better than 0.1%, e.g., us-          tion signal of a pure Rayleigh scattering atmosphere would also
ing the upcoming SPHERE instrument at the VLT (Beuzit                    be diﬀerent with a high signal in shorter wavelength bands be-
et al. 2008), might reveal structure along the limb and tempo-           tween 550-700 nm, and a significantly lower signal for the longer
ral changes in the polarization. Such limb polarization measure-         filter bands between 700−800 nm (Buenzli & Schmid 2009).
ments could be useful for investigating local haze properties of         Therefore, a good knowledge of the polarization properties of
Titan, and the impact of short-term and seasonal variations. In          the hazy atmosphere of Titan is useful for the search and inves-
particular, the limb polarization is very sensitive to the maxi-         tigation of the polarimetric signal of extra-solar planets.
mum haze single scattering polarization pm , which is strongly
depending on the monomer size of the small haze aggregates               Acknowledgements. We thank the referee for very thoughtful comments and
(e.g., Tomasko et al. 2008). Changing pm in the F775W band               suggestions which lead to a much improved revised version of the paper. Part
from, e.g., pm = 0.96 to pm = 0.94, i.e., corresponding to an            of this work was supported by the FINES research fund by a grant through the
increase of the monomer radius of about 15% (rough estimate              Swiss National Science Foundation (SNF).
based on Fig. 18 by Tomasko et al. 2008), reduces the limb po-
larization by Δp/p ≈ 0.1, while the albedo remains essentially
unchanged.                                                               Appendix A: Titan scattering model parameters
    For other atmospheric parameters like τhaze , ωhaze , or the         A.1. Number densities and column density
CH4 fraction the geometric albedo Ag and the limb polarization
Qr /I will change together (ωhaze ) or in opposite direction (τhaze ).   For the calculation of the methane number density nCH4 and the
Thus no structure in the limb polarization is expected if there is       scale height ZCH4 the following formulas are used:
no strong albedo feature, like a north-south asymmetry.
                                                                                   ρ [g/cm3 ] × 6.02 × 1023
                                                                         nCH4 =                                                               (A.1)
                                                                                         28/ fCH4 − 12
8.3. Prospects for extra-solar planets                                                     fN           1 − fCH4
                                                                             nN2 = nCH4 · 2 = nCH4 ·             ·                            (A.2)
Diﬀerential polarimetric imaging is a particularly promising                              fCH4            fCH4
technique for the detection and characterization of extra-solar
planets. With sensitive polarimetry the measurable contrast be-          Derivation:
tween star and planet can be enhanced by searching for the polar-
                                                                             – ntot = ρ [g/cmμ ]·NA = nCH4 + nN2
                                                                                               3

ized signal of the scattered light from the planet within the halo
of the unpolarized light from the star (e.g., Schmid et al. 2006b).          – μ = 28 · (1 − fCH4 ) + 16 · fCH4
                                                                                                      f 4 ρ [g/cm3 ]·NA
The upcoming planet finder instruments SPHERE (Beuzit et al.                 – nCH4 = fCH4 ntot = 28CH (1− fCH4 )+16 fCH4 ·
2008) and GPI (Macintosh et al. 2012) will both provide im-
proved performance for substantial progress in this direction.           Column density in km-am6 :
    The measurable polarization contrast can be described by

Cp (α, λ) = p(α, λ) f (α, λ)(Rp /dp )2 ,                          (9)    Z [1/km2] = 1015            n [1/cm3] dh = 2.687 × 1034 × Z [km-am].

where α is the phase angle, Rp the radius of the planet, dp its sep-                                                                          (A.3)
aration to the star, f (α, λ) is the phase-dependent reflectivity, and
p(α, λ) is the integrated fractional polarization. Therefore, the in-    A.2. Rayleigh scattering optical depth
vestigation of p(α, λ) and the polarization flux p(α, λ) × f (α, λ)
of Titan is important for planning future observing projects on          This is from PDS7 :
extrasolar planetary systems and interpreting observational data.
    On the one hand, Titan shows that atmospheres with thick             τray,sc = τ1 (H2 ) · (10.1509 · ZCH4 + 4.6035 · ZN2 ),               (A.4)
layers of small aggregate haze particles produce a very strong
                                                                         with Zi [km-am] and the optical depth per km-am of hydrogen
polarization signal of p(α ≈ 90◦ ) ≈ 50% over a wide wave-
length range from the UV at 300 nm to the near-IR at 2 μm.               given as (λ [Å]):
The bottom panel of Fig. 12 gives the expected polarized flux                                                                    
                                                                                              8.14 × 1011 1.28 × 1018 1.61 × 1024
p × f of Titan if we would observe the object at quadrature              τ1 (H2 ) = 2.687 ·              +           +              ·
phase α = 90◦ . In the optical p × f (90◦ ) is increasing with                                    λ4          λ6          λ8
wavelength with p × f (90◦ ) ≈ 0.02 at λ = 450 nm, up to                                                                        (A.5)
p × f (90◦ )  0.03 at λ = 850 nm. Above 850 nm the polar-
ized flux is strongly decreased in the methane absorption bands.         A.3. Methane absorption
Therefore, planets with hazy atmospheres and aerosol properties
similar to Titan could be particularly good candidates for detec-        Methane absorption from Karkoschka & Tomasko (2010).
tion with ZIMPOL/SPHERE (Beuzit et al. 2008; Schmid et al.               Below 1 μm these methane absorption coeﬃcients are generally
2006a) because of their large polarization over the entire wave-         close to those by Karkoschka (1998),
length range of the instrument (520−900 nm).                                                                        
    On the other hand, the polarization signal could be strongly         log κ(T ) = 0.5z(z − 1) log κ(100) + 1 − z2 log κ(198)
reduced for a planet with a Titan-like atmosphere but consisting                         + 0.5z(z + 1) log κ(296),                            (A.6)
of larger aggregates. If the monomers were about a factor of two
larger then the single scattering polarization in the F775W band         6
                                                                           1 km-am = 2.687 × 1024 molecules cm−2 .
is reduced by about Δpm /pm ≈ 0.2 (rough estimate based on               7
                                                                           http://pds-atmospheres.nmsu.edu/education_and_
Tomasko et al. 2008), and the quadrature polarization will be            outreach/

A6, page 12 of 13
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
                                        A. Bazzon et al.: HST observations of the limb polarization of Titan

with z = (T − 198)/98; and κ(100), κ(198), and κ(296) are                  A.5. Effective single scattering albedo
given in Table 4 of the supplementary material of Karkoschka
& Tomasko (2010). Therefore, we get                                        τtot = τray,sc + τCH4 + τhaze                                            (A.18)
τCH4 = κCH4 · ZCH4 .                                              (A.7)    ωray = τray,sc /(τray,sc + τCH4 )                                        (A.19)
                                                                           ωray,eﬀ = τray,sc /τtot                                                  (A.20)
A.4. Haze optical depth and single scattering albedo                       ωhaze,eﬀ = (ωhaze · τhaze )/τtot .                                       (A.21)
The haze optical depth τh and single scattering albedo ωhaze are
taken from Tomasko et al. (2008). τh is given for three altitude           References
regions, above 80 km, 30−80 km, and below 30 km (see Figs. 47,
50). Above 80 km, the cumulative optical depth increases with              Batcheldor, D., Schneider, G., Hines, D. C., et al. 2009, PASP, 121, 153
                                                                           Beuzit, J.-L., Feldt, M., Dohlen, K., et al. 2008, in SPIE Conf. Ser., 7014, 18
decreasing altitude with a scale height of 65 km. Between 80 and           Brown, R. H., Lebreton, J.-P., & Waite, J. H. 2010, Titan from Cassini-Huygens
30 km, and below 30 km the variation is linear with two diﬀerent              (Springer Science)
slopes. The wavelength dependence for the three regions is taken           Buenzli, E., & Schmid, H. M. 2009, A&A, 504, 259
from Fig. 47 of Tomasko et al. (2008):                                     Courtin, R. D., Sim, C., Kim, S., Gautier, D., & Jennings, D. E. 2008, BAAS,
                                                                              40, 446
τ80 (λ) = 1.012 × 107 × λ−2.339                                   (A.8)    Fulchignoni, M., Ferri, F., Angrilli, F., et al. 2005, Nature, 438, 785
                                                                           Karkoschka, E. 1998, Icarus, 133, 134
                              −1.409                                       Karkoschka, E., & Tomasko, M. G. 2010, Icarus, 205, 674
τ30 (λ) = 2.029 × 10 × λ
                       4
                                                                  (A.9)    Krist, J. E., Hook, R. N., & Stoehr, F. 2011, in SPIE Conf. Ser., 8127
                                                                           Lorenz, R. D., Smith, P. H., & Lemmon, M. T. 2004, Geophys. Res. Lett., 31,
τ0 (λ) = 6.270 × 102 × λ−0.9706                                 (A.10)        10702
                                                                           Lorenz, R. D., Lemmon, M. T., & Smith, P. H. 2006, MNRAS, 369, 1683
τ>80 (h, λ) = τ80 (λ) · e− 65 km
                           h−80 km
                                                                (A.11)     Macintosh, B. A., Anthony, A., Atwood, J., et al. 2012, in SPIE Conf. Ser., 8446
                                                                         McGrath, M. A., Courtin, R., Smith, T. E., Feldman, P. D., & Strobel, D. F. 1998,
                                       h − 30 km                              Icarus, 131, 382
τ30−80 (h, λ) = τ80 (λ) + τ30 (λ) 1 −                           (A.12)     McKay, C. P., Pollack, J. B., & Courtin, R. 1989, Icarus, 80, 23
                                          50 km                            Negrão, A., Coustenis, A., Lellouch, E., et al. 2006, Planet. Space Sci., 54, 1225
                                                                         Porco, C. C., Baker, E., Barbara, J., et al. 2005, Nature, 434, 159
                                                h
τ<30 (h, λ) = τ80 (λ) + τ30 (λ) + τ0 (λ) 1 −          ·         (A.13)     Rannou, P., Hourdin, F., & McKay, C. P. 2002, Nature, 418, 853
                                              30 km                        Schmid, H. M. 1992, A&A, 254, 224
                                                                           Schmid, H. M., Beuzit, J.-L., Feldt, M., et al. 2006a, in Direct Imaging of
The single scattering albedo ωhaze is given in Tomasko et al.                 Exoplanets: Science & Techniques, eds. C. Aime, & F. Vakili, IAU Colloq.
(2008) for three altitude regions (Fig. 48, Table 2), above                   200, 165
                                                                           Schmid, H. M., Joos, F., & Tschan, D. 2006b, A&A, 452, 657
144 km, 30−80 km, and below 30 km. The wavelength depen-                   Schmid, H. M., Joos, F., Buenzli, E., & Gisler, D. 2011, Icarus, 212, 701
dence of ωhaze for the region above 144 km and 30−80 km is                 Smith, P. H., Lemmon, M. T., Lorenz, R. D., et al. 1996, Icarus, 119, 336
given in Fig. 48 of Tomasko et al. (2008) whereas for the region           Sromovsky, L. A. 2005, Icarus, 173, 254
below 30 km we adopt a three-dimensional polynomial fit to the             Thuillier, G., Floyd, L., Woods, T. N., et al. 2004, Adv. Space Res., 34, 256
values given in Table 2 of Tomasko et al. (2008). For the regions          Tomasko, M. G., & Smith, P. H. 1982, Icarus, 51, 65
                                                                           Tomasko, M. G., Lemmon, M., Doose, L. R., et al. 1997, in Huygens: Science,
below 30 km, 30−80 km, and above 144 km, we assume ωhaze to                   Payload and Mission, ed. A. Wilson, ESA SP, 1177, 345
be constant with altitude, whereas for the region 80−144 km we             Tomasko, M. G., Buchhauser, D., Bushroe, M., et al. 2002, Space Sci. Rev., 104,
linearly interpolate between the values at 80 km and 144 km:                  469
                                                                           Tomasko, M. G., Doose, L., Engel, S., et al. 2008, Planet. Space Sci., 56, 669
ωhaze,>144 (h, λ) = ωhaze,144 (λ)                               (A.14)     Toon, O. B., McKay, C. P., Griﬃth, C. A., & Turco, R. P. 1992, Icarus, 95, 24
                                                                           US Naval Observatory & Royal Greenwich Observatory 2000, The Astronomical
ωhaze,80−144 (h, λ) = ωhaze,30−80 (λ)                                         Almanac for the year 2002 (Washington, USA, London, UK: US Government
                                                                              Printing Oﬃce (USGPO) and The Stationary Oﬃce)
                       ωhaze,144 − ωhaze,30−80                             van de Hulst, H. C. 1980, Multiple light scattering, 2 (New York, NY: Academic
                      +                        · (h − 80 km) (A.15)           Press)
                                64 km
                                                                           Veverka, J. 1973, Icarus, 18, 657
ωhaze,30−80 (h, λ) = ωhaze,30−80 (λ)                         (A.16)        Vinatier, S., Bézard, B., Fouchet, T., et al. 2007, Icarus, 188, 120
                                                                           West, R. A., Hart, H., Simmons, K. E., et al. 1983, J. Geophys. Res., 88, 8699
ωhaze,<30 (h, λ) = ωhaze,<30 (λ).                               (A.17)     Zellner, B. 1973, Icarus, 18, 661




                                                                                                                                        A6, page 13 of 13
```
