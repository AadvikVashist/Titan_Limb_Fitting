---
citation_key: "robinson2014titan"
title: "Titan solar occultation observations reveal transit spectra of a hazy world"
source_pdf: "data/papers/robinson2014titan.pdf"
source_pdf_sha256: "eaba8dca47ad51f3fc3683ed7bfa3e4178f4acdb1d5ec683d79860a1c5196f86"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
Titan solar occultation observations reveal transit
spectra of a hazy world
Tyler D. Robinsona,b,1, Luca Maltagliatic, Mark S. Marleya, and Jonathan J. Fortneyd
a
  National Aeronautics and Space Administration (NASA) Ames Research Center, Moffett Field, CA 94035; bVirtual Planetary Laboratory, Astrobiology
Institute, NASA, Seattle, WA 98195; cLaboratoire d’Études Spatiales et d’Instrumentation en Astrophysique, Observatoire de Paris, Centre National de la
Recherche Scientifique, Université Pierre et Marie Curie, Meudon Cedex 92195, France; and dDepartment of Astronomy and Astrophysics, University of
California, Santa Cruz, CA 95064

Edited* by Jonathan I. Lunine, Cornell University, Ithaca, NY, and approved April 24, 2014 (received for review February 24, 2014)

High-altitude clouds and hazes are integral to understanding exopla-              Here, we turn to the archetypal hazy world—Titan—to shed
net observations, and are proposed to explain observed featureless             light on how high-altitude clouds and hazes can influence transit
transit spectra. However, it is difficult to make inferences from these        observations, thus building a bridge between Titan studies and
data because of the need to disentangle effects of gas absorption              exoplanetary science, where Titan analog worlds are currently being
from haze extinction. Here, we turn to the quintessential hazy world,          modeled (17–19) and may prove to be a very common class of
Titan, to clarify how high-altitude hazes influence transit spectra. We        planet in the universe (20). Titan is ideally suited to this task, as it
use solar occultation observations of Titan’s atmosphere from the              possesses a haze that extends to pressures approaching 10−6 bar (21,
Visual and Infrared Mapping Spectrometer aboard National Aeronau-              22), and, unlike exoplanets, is extremely well-studied, including in
tics and Space Administration’s (NASA) Cassini spacecraft to generate          situ observations (23). We link Titan to exoplanet transit observa-
transit spectra. Data span 0.88–5 μm at a resolution of 12–18 nm, with         tions using solar occultation observations, which have an analogous
uncertainties typically smaller than 1%. Our approach exploits sym-            geometry to exoplanet transits, and have a long history of providing
metry between occultations and transits, producing transit radius              detailed information on the atmospheric composition and structure
spectra that inherently include the effects of haze multiple scattering,       of solar system worlds (24–26).
refraction, and gas absorption. We use a simple model of haze ex-                 To date, the similarities between exoplanet transits and solar
tinction to explore how Titan’s haze affects its transit spectrum. Our         or stellar occultations by solar system worlds have not been
spectra show strong methane-absorption features, and weaker fea-               exploited as a means of bridging these two research fields. Al-
tures due to other gases. Most importantly, the data demonstrate               though Earth’s atmospheric transmission has been measured
that high-altitude hazes can severely limit the atmospheric depths             during lunar eclipse and interpreted in terms of exoplanet
probed by transit spectra, bounding observations to pressures smaller          observations (27, 28), these data only probe a limited range of
than 0.1–10 mbar, depending on wavelength. Unlike the usual as-                altitudes (∼ 10 km, depending on the solar-elevation angle), and
sumption made when modeling and interpreting transit observations              require corrections for telluric absorption, solar lines, and the
of potentially hazy worlds, the slope set by haze in our spectra is not        lunar albedo. Furthermore, it is clear that, with regards to exo-
flat, and creates a variation in transit height whose magnitude is             planetary science, what is needed is a better understanding of how
comparable to those from the strongest gaseous-absorption features.            high-altitude hazes influence transmission spectra, and Earth does
These findings have important consequences for interpreting future             not have a particularly hazy upper atmosphere.
exoplanet observations, including those from NASA’s James Webb                    In this work, we use observations from the National Aeronautics
Space Telescope.                                                               and Space Administration (NASA) Cassini mission of solar occul-
                                                                               tations by Titan’s atmosphere to, for the first time (to our knowl-
transit spectroscopy   | extrasolar planet                                     edge), produce transit radius spectra of a hazy, well-characterized
                                                                               world. Because of the symmetry in the geometry of occultations

C   louds and hazes are ubiquitous in the atmospheres of solar
    system worlds (1). Furthermore, it is now becoming apparent
that high-altitude hazes strongly influence observed spectra of
                                                                                   Significance

exoplanets (2–6). These hazes can limit our ability to study the                   Hazes dramatically influence exoplanet observations by ob-
underlying atmosphere, especially in transit spectroscopy, where                   scuring deeper atmospheric layers. This effect is especially
the opacity of an exoplanet’s atmosphere is studied by observing                   pronounced in transit spectroscopy, which probes an exopla-
the wavelength-dependent dimming of the host star as the planet                    net’s atmosphere as it crosses the disk of its host star. How-
crosses the stellar disk (7, 8). Here, long pathlengths through the                ever, exoplanet observations are typically noisy, which hinders
atmosphere mean that even relatively tenuous haze layers can                       our ability to disentangle haze effects from other processes.
become optically thick (9). Depending on the cloud or haze                         Here, we turn to Titan, an extremely well-studied world with
properties, the result can be a flat or smoothly varying spectrum                  a hazy atmosphere, to better understand how high-altitude
that contains little information about the composition of the bulk                 hazes can impact exoplanet transit observations. We use data
of the exoplanet’s atmosphere.                                                     from National Aeronautics and Space Administration’s Cassini
   A major obstacle to interpreting observations of potentially                    mission, which observed occultations of the Sun by Titan’s at-
hazy exoplanet atmospheres is a lack of understanding of how                       mosphere, to effectively view Titan in transit. These new data
                                                                                   challenge our understanding of how hazes influence exoplanet
aerosols influence transit spectra. A number of key physical
                                                                                   transit observations, and provide a means of testing proposed
processes are at play—gas absorption, atmospheric refraction,
                                                                                   approaches for exoplanet characterization.
Rayleigh scattering, and multiple scattering by cloud and haze
particles (10, 11). Although models of atmospheric transmission                Author contributions: T.D.R. designed research; T.D.R. and L.M. performed research; T.D.R.
effects on a transit exist (12–16), the complexity and computa-                and L.M. contributed new reagents/analytic tools; T.D.R., L.M., M.S.M., and J.J.F. analyzed
tional cost of implementing all of the aforementioned processes                data; and T.D.R., L.M., M.S.M., and J.J.F. wrote the paper.

forces simplification of the problem. As a result, models com-                 The authors declare no conflict of interest.
monly treat clouds and hazes as an opaque, gray absorbing layer                *This Direct Submission article had a prearranged editor.
that prevents light from probing deeper levels.                                1
                                                                                To whom correspondence should be addressed. E-mail: tyler.d.robinson@nasa.gov.



9042–9047 | PNAS | June 24, 2014 | vol. 111 | no. 25                                                                www.pnas.org/cgi/doi/10.1073/pnas.1403473111
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                                                 Table 1. Parameters for Titan solar occultation measurements
                                                                                 Date        Cassini flyby    Season    Latitude D (km) Resolution (km)

                                                                                 Jan. 2006        T10        N winter     70°S      8,300           15
                                                                                 Apr. 2009        T53        Equinox      1°N       6,300            7
                                                                                 Sep. 2011        T78        N spring     40°N      9,700           10
                                                                                 Sep. 2011        T78        N spring     27°N      8,400           10

                                                                                   N, northern.



                                                                                    The uncertainty on the transmission values are given by the
                                                                                 standard deviation over the average of the solar spectrum out-
                                                                                 side the atmosphere, which is stable except for random noise.
                                                                                 Additional details on the data-treatment process are described in
                                                                                 Maltagliati et al. (30). We note that these results are in good
                                                                                 agreement with the analysis of the 70°S occultation dataset by
                                                                                 Bellucci et al. (22), who used different data-processing methods.
Fig. 1. Geometry and parameters relevant to occultation and transit. In
                                                                                    Note that the angular diameter of the Sun at Saturn’s orbital
occultation, rays enter from the right, with impact parameter b, are bent by     distance, θ⊙ , is about 1 mrad, so that its image actually subtends
atmospheric refraction through an angle ω, and have a distance of closest        a range of altitudes given by θ⊙ D, or about 6–10 km. Thus, each
approach rmin . In transit, rays follow the opposite trajectory. Note also the   individual transmission spectrum contains information from a
planetary radius, Rp , the radial coordinate, r, the corresponding vertical      small range of altitudes. Fortunately this range is smaller than
height coordinate, z = r − Rp , and the polar angle, ϕ.                          both the vertical resolution of the corresponding datasets (shown
                                                                                 in Table 1) and the atmospheric scale height (∼ 40 km, implying
                                                                                 that atmospheric properties should not change dramatically over
and transits, these data inherently include the effects of refraction            the 6–10 km range). Nevertheless, future applications of the tech-
and aerosol multiple scattering. Our observations provide an es-                 niques described here may need to account for this “smearing” ef-
sential and much-needed means of validating exoplanet transit                    fect, possibly by performing an analysis using a resolved portion of
models against solar system data, and can be used to test proposed               the solar disk [where, then, the relevant angular size is determined
approaches for deciphering transit spectra. To better understand                 by the pixel or instrument field of view; see, e.g., (31, 32)].
how Titan’s high-altitude haze affects the transit spectra, we de-
velop an analytic model of haze extinction. Finally, we interpret

                                                                                                                                                                  ASTRONOMY
                                                                                 Refraction Effects. Refraction has two key effects on occultation
our spectra within the context of exoplanet observations, yielding               observations. The first, and most familiar, is the bending of a light
insights into the effects of hazes on transit observations.                      ray as it passes through the atmosphere. This effect is characterized
                                                                                 by the refraction angle, ω, which is the angle between the original
Observations and Data Processing                                                 ray path and the exit path. Generally, the refraction angle is a
Fig. 1 shows the geometry and relevant variables of occultation                  function of wavelength (due to the wavelength-dependent index of
and transit, which are analogous to one another. In transit, rays                refraction of the atmosphere), and causes a distinction between
leave the stellar disk at the left of the diagram, are refracted and             a ray’s impact parameter, b, and its distance of closest approach to
attenuated, and exit the atmosphere to travel to the observer,                   the planet, rmin . These parameters are all shown in Fig. 1. Re-
who is effectively an infinite distance away. In occultation, rays               fraction is most pronounced for rays that pass near the surface,
from the occulted star come from the right of the diagram. For
a distant star, these rays are parallel, and for solar occultations
rays can be nonparallel, depending on the angular size of the
Sun. These rays are attenuated and refracted by the atmosphere
before exiting in the direction of the observer (a relatively short
distance, D, away). Thus, occultation measurements can be readily
converted into transit radius spectra.

Occultation Spectra. The Visual and Infrared Mapping Spec-
trometer (VIMS) (29) aboard NASA’s Cassini orbiter has ob-
served 10 solar occultations through Titan’s atmosphere since
the beginning of the mission. Spectra are acquired through a
special solar port, which attenuates the intensity of sunlight on
the detector, span 0.88–5 μm, and have a spectral resolution be-
tween 12 and 18 nm, increasing with wavelength. Because of tech-
nical problems related to pointing stability and parasitic light, only
four occultations out of ten could be analyzed. Table 1 summarizes
the main parameters of the four datasets.
   The atmospheric transmission, tλ , along the line of sight is ob-
tained by taking the ratio of every spectrum to the average spectrum
outside the atmosphere (i.e., the reference solar spectrum). This
                                                                                 Fig. 2. Wavelength-dependent transmission through Titan’s atmosphere
is a self-calibrating method—instrumental effects and systematic
                                                                                 from the 27° N occultation. The vertical axis is the ray’s altitude of closest
errors are removed with the ratio, provided that the occultation is              approach, where an altitude of 0 corresponds to the planetary surface, at
stable and the intensity variations are only due to the atmosphere.              a radius of Rp = 2,575 km. Darker shades indicate lower transmission, and
Fig. 2 shows the altitude-dependent transmission spectra for the                 noise can be seen at transmission values very near to 1 and at wavelengths
27°N occultation.                                                                beyond about 4.5 μm.


Robinson et al.                                                                                           PNAS | June 24, 2014 | vol. 111 | no. 25 | 9043
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                                                profiles of these parameters as a function of their altitude of
                                                                                closest approach, and demonstrates that, for our purposes, re-
                                                                                fraction effects are only important in the lowest 100 km of the
                                                                                atmosphere.
                                                                                   We note that refraction can also influence exoplanet transit
                                                                                observations under conditions where atmospheric opacity does
                                                                                not preclude light rays from reaching the deeper regions of an
                                                                                atmosphere. Here, the finite size of the host star paired with the
                                                                                geometry of refraction may prevent rays from probing altitudes
                                                                                below some critical height in the lower atmosphere (38). In ad-
                                                                                dition, the transit signal may increase or decrease slightly due to
                                                                                the competing effects of refraction bending rays perpendicular to
                                                                                the limb while also focusing rays from within the planet’s shadow
                                                                                toward the observer (39–41).

                                                                                Computing Transit Spectra. We define the transmission corrected
                                                                                for refractive losses at an impact parameter bi as t′λ;i = tλ;i =fref;i
                                                                                (where a subscript ‘i’ references the vertical gridding of the ob-
Fig. 3. Profiles of the impact parameter, b, refraction angle, ω, and the       served transmission spectra). These can be converted into a transit-
refractive loss factor, fref , from our Titan ray-tracing model. The vertical
                                                                                radius spectrum by considering the attenuation produced by con-
coordinate is the ray’s altitude of closest approach, zmin . Parameters are
shown for a wavelength of 5 μm, where Titan’s atmospheric haze is least
                                                                                centric annuli above Titan’s surface. An annulus has thickness
opaque, and we use a distance, D, from the spacecraft to Titan of 8,400 km,     πðb2i + 1 − b2i Þ, and we can define an effective transit radius as (42):
appropriate for the 27° N occultation, when computing the refractive loss.
                                                                                                                  N                
The dotted line plots along the diagonal, and shows that the impact pa-                                          X   t′λ;i+1 + t′λ;i  2            
rameter is always larger than the distance of closest approach.                              R2eff;λ = R2top −                        bλ;i+1 − b2λ;i ;        [2]
                                                                                                                 i=1
                                                                                                                           2

where molecular number densities are large. Note, however, that                 where Rtop = Rp + zatm is the radial distance to the top of the
for Titan, strong attenuation by atmospheric haze particles at                  atmosphere, whose altitude (zatm ) is large enough that atmo-
visible and near-infrared wavelengths largely limits sensitivity to             spheric extinction and refraction are assumed to be negligible.
the deep portions of the atmosphere where refractive bending of                 We also define an effective transit height as zeff;λ = Reff;λ − Rp ,
light rays is most significant.                                                 which is useful for identifying where in the atmosphere a given
   The second key refractive effect is an apparent brightness loss,             wavelength is probing. Finally, note that the transit depth is pro-
which is present even in the absence of molecular and aerosol                   portional to R2eff;λ , or, equivalently, ðzeff;λ + Rp Þ2 .
attenuation (24, 26). This loss can be thought of as an apparent
shrinking of the solar/stellar disk in the vertical direction or,               Transit Radius Spectra
equivalently, a spreading of rays from the source (33). Here,                   Fig. 4 shows the effective transit height, zeff;λ , for all four oc-
brightness is diminished by a wavelength-dependent factor, fref ,               cultation datasets. Error bars (1 − σ) are shown where the errors
which is given by                                                               are larger than 1% of the transit height, and key absorption
                                                                                features are identified. In general, errors tend to be large beyond
                                         1                                      about 4 μm, where the solar flux is relatively weak.
                          fref =                 :                       [1]      The most obvious features are the methane bands at 1.2, 1.4,
                                   1 + Ddω=drmin
                                                                                1.7, 2.3, and 3.3 μm. Weak absorption due to acetylene (C2H2)
   To model these two effects, we use a ray tracing scheme de-
scribed by van der Werf (34), which concisely outlines an accurate,
fourth-order Runge–Kutta integration algorithm for tracking rays
through an atmosphere. The primary inputs to this model are
profiles of atmospheric density and composition, as well as the
refractive indexes of the major atmospheric constituents (which
are, generally, wavelength dependent). For Titan, we elect to use
standard model profiles of atmospheric molecular number den-
sity and composition (35), as localized structure in measured pro-
files can lead to spurious features in our refraction calculations. Our
refraction models only include molecular nitrogen and methane
in our computations, as these are the only major atmospheric
constituents. Finally, we use a measured, wavelength-dependent
refractivity for molecular nitrogen (36) and a refractive index for
methane of 1.0004478 (37), although our calculations are largely
insensitive to this value due to the low mixing ratio of methane in
the atmosphere.
   By tracing rays on a fine grid of impact parameters (1-km
vertical resolution from 0 to 1,500 km), we determine the rela-
tions between the impact parameter, altitudes of closest ap-
proach (zmin = rmin − Rp ), refraction angle, and the refractive loss           Fig. 4. Spectra of effective transit height, zeff,λ = Reff,λ − Rp , for all four
factor fref . Our computed values are only weak functions of                    Cassini/VIMS occultation datasets. Key absorption features are labeled, and
wavelength, as the refractivity of molecular nitrogen changes by                error bars are shown only where the 1 − σ uncertainty is larger than 1%. Our
less than 1% over the wavelength range of interest. Fig. 3 shows                best-fit haze model for the 70° S dataset is shown (dashed line).


9044 | www.pnas.org/cgi/doi/10.1073/pnas.1403473111                                                                                                 Robinson et al.
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                                                  haze continuum levels. At wavelengths dominated by haze opacity,
                                                                                  although, the deviation is much larger. Clearly future occultation
                                                                                  measurements could help to better understand latitudinal and
                                                                                  seasonal effects on our transit spectra, thus improving our char-
                                                                                  acteristic spectrum.

                                                                                  A Simple Haze Extinction Model
                                                                                  To investigate the source and behavior of the continuum in our
                                                                                  transit height spectra, we derived an analytic model of extinction by
                                                                                  an opacity source that is distributed vertically in the atmosphere
                                                                                  with scale height Ha , and whose absorption cross-section, σ λ varies
                                                                                  according to a power law in wavelength, with σ λ ∝ λβ . Ignoring re-
                                                                                  fraction effects, which are negligible at most altitudes probed by our
                                                                                  spectra, the wavelength-dependent optical depth through the at-
                                                                                  mosphere for a given impact parameter is (Appendix)
                                                                                                             β       
                                                                                                             λ   b     b ðRp +z0 Þ=Ha
                                                                                                 τλ = 2τ0           K1    e           ;             [3]
                                                                                                             λ 0 Ha    Ha
Fig. 5. Characteristic transit spectrum for Titan showing both the signal for
Titan transiting the Sun (left y axis) and the effective transit height (right    where τ0 is a reference optical depth at altitude z0 , and Kn ðxÞ is
y axis), assembled as a weighted mean of the four spectra in Fig. 4. The shaded   a modified Bessel function of the second kind. With this model,
region indicates uncertainty in our averaging, and is due to deviations from      a transit spectrum can be generated by finding the value of the
the mean in the four individual transit spectra. A best-fit haze model is
                                                                                  impact parameter where τλ ≈ 1, which requires solving a transcen-
shown (dashed).
                                                                                  dental expression.
                                                                                     We fit our analytic model to the continua in the 70°S spectrum
can be seen near 3.1 μm. The 3.3-μm methane band is blended                       (selected since this dataset has been previously analyzed), and in
with other features, including the C-H stretching mode of ali-                    the characteristic spectrum, which are shown in Figs. 4 and 5,
phatic hydrocarbon chains appears near 3.4 μm (22, 30). An                        respectively. The free parameters in this fit are the haze-scale
absorption feature of carbon monoxide, which forms from oxy-                      height, Ha , the reference optical depth, τ0 , and the exponent in
gen ions that precipitate into Titan’s upper atmosphere and react                 the cross-section power law, β. For the 70°S dataset, we find
with hydrocarbon species (43), appears near 4.6 μm, although                      Ha = 58 ± 7 km, τ0 = 0:9 ± 0:2 (at λ0 = 0:5 μm and z0 = 200 km,


                                                                                                                                                           ASTRONOMY
data are particularly noisy here. Finally, additional absorption                  which we will use hereafter), and β = −2:2 ± 0:2. For the charac-
has been noted in the 2.3- and 3.3-μm methane bands (22, 30),                     teristic spectrum, we find Ha = 55 ± 8 km, τ0 = 0:8 ± 0:4, and
which is due to other yet unknown species.                                        β = −1:9 ± 0:2. Note that the slope of our power law is not due to
   What is possibly the most interesting aspect of these spectra is               pure Rayleigh scattering, which would have β = −4. Instead it is
the wavelength-dependent slope of the continuum between the                       due to the complexities of haze particle scattering between the
methane bands. When observed across the full wavelength range,                    limits of pure Rayleigh scattering and geometric optics.
this slope produces a transit height variation that is comparable                    Our parameters are in excellent agreement with in situ mea-
to, or larger than, the gaseous-absorption features. Assuming                     surements reported by Tomasko et al. (47), who found Ha = 65 km
that the continuum is set by haze extinction, which shall be ar-                  (with an uncertainty of 20 km), τ0 = 0:76, and β = −2:33 above
gued later, then the differences between the continuum levels for                 80-km altitude . For further comparison, Bellucci et al. (22), in
the four different datasets are related to different haze dis-                    their analysis of the 70°S occultation, found Ha = 55  79 km,
tributions (both vertically and in particle size) at the different                β = −1:7  2:2 between 120- and 300-km altitude, and τ0 ∼ 0:6.
latitudes/times of observation. This is consistent with Titan’s                   Finally, Hubbard et al. (48), in their analysis of stellar occulta-
known hemispherical asymmetry (44), which may be caused by                        tions by Titan’s atmosphere, found β = −1:7 ± 0:2. These compar-
seasonally varying atmospheric circulation patterns (45). Note                    isons strongly support our conclusion that the continuum level in
that methane clouds in Titan’s atmosphere are found below                         our transit spectra is set by Titan’s high-altitude haze.
about 30-km altitude (46), and do not affect our transit spectra,
which probe much higher altitudes.                                                Implications
   Finally, Fig. 5 demonstrates an “average” transit spectrum for                 The transit spectra shown in Figs. 4 and 5 demonstrate that high-
Titan, shown as effective transit height and, as an example, as the               altitude hazes could have complex and important effects on
transit-depth signal for Titan crossing the solar disk. To obtain                 exoplanet observations. Note that our data span wavelengths
this result, we performed a weighted average of the four in-                      that are nearly identical to (or larger than) the spectral coverage
dividual spectra in Fig. 4. The weights were determined from the                  of the Near InfraRed Camera, Near InfraRed Spectrograph, and
latitude distribution of the individual spectra, assuming that the                the Near InfraRed Imager and Slitless Spectrograph instruments
70°S spectrum is representative of latitudes between the south                    that will launch aboard NASA’s James Webb Space Telescope.
pole and midway to the 1°N spectrum, that the 1°N spectrum is                     Thus, the spectra presented here indicate the types of haze
representative of latitudes midway between the 70°S spectrum                      effects that this mission may observe for transiting exoplanets.
and the 27°N spectrum, and so on. Using these weights, we                            For Titan, the haze continuum slope is strongly wavelength
combine the spectra in R2eff;λ , which is proportional to the transit             dependent, and is certainly not flat. This is contrary to what is
depth signal. Although this weighted averaging is somewhat                        commonly assumed in simple transit spectra models. Clearly this
crude, as it ignores variations in longitude and time, the goal is                continuum slope is of first-order importance, as the magnitude of
only to produce a characteristic spectrum.                                        the transit height variations caused by the haze continuum is just
   SDs computed by comparing our characteristic spectrum to                       as large as the observed gaseous absorption features.
the individual spectra are shown as a shaded swath in Fig. 5. The                    Our transit spectra also show that haze opacity obscures
deviations are small near the peaks of methane bands, which probe                 information from the deep atmosphere, limiting the pres-
higher in the atmosphere and are less sensitive to variations in the              sures probed to above ∼ 0:1 mbar at the shortest wavelengths,

Robinson et al.                                                                                         PNAS | June 24, 2014 | vol. 111 | no. 25 | 9045
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
and ∼ 10 mbar at the longest wavelengths. Even at the longest              spectra, limiting sensitivity to pressures smaller than 0.1–10 mbar,
wavelengths, the altitudes probed are still 2–3 pressure-scale             depending on wavelength. Extinction from the haze imparts a
heights above the surface. Furthermore, at most continuum                  distinct slope on the transit radius spectra, the magnitude of which
wavelengths in our spectra, haze limits sensitivity to pressures           is comparable to that of the strongest gaseous absorption bands.
lower than (i.e., altitudes above) the ∼ 1 mbar level, with this           Thus, haze substantially impacts the amount of information that
effect becoming more severe at shorter wavelengths. Thus, it is            can be gleaned from transit spectra.
empirically possible for high-altitude hazes to strongly limit the            We note that the techniques used here apply equally well to
planetary characteristics that can be inferred from transit spec-          occultation observations taken from orbit around any world. Thus,
tra, despite what others have claimed (49). To further clarify this        there are opportunities to empirically study the tenuous, dusty
issue, it would be a very useful exercise to challenge current             atmosphere of Mars (32) and the atmosphere of Saturn (60) in the
exoplanet retrieval models (50–53) with our Titan transit spectra,         context of exoplanet transit spectroscopy. Of course, numerous
with the goal of improving our ability to understand and in-
                                                                           occultation observations exist for Earth (31), which could be used
terpret transit observations of hazy exoplanets.
                                                                           to derive a transit spectrum of the only known habitable planet.
   Looking to wavelengths beyond those analyzed here, we note
that haze opacity effects in transit will become negligible in the         Finally, our understanding of how hazes influence transit spectra
midinfrared, where refraction and gas absorption will then play            of Titan could be greatly improved by acquiring additional oc-
a key role in limiting sensitivity to the lower atmosphere. How-           cultation observations in a Cassini extended mission.
ever, haze extinction will have much more dramatic effects
                                                                           Appendix
at UV and visible wavelengths, where Titan’s haze particles
are strongly absorbing (47). This will make Rayleigh scattering            Given the extinction coefficient, αλ = na σ λ , where na is the absorber
effects undetectable—a ray passing through the atmosphere with             number density and σ λ is the wavelength-dependent absorption
a tangent height of ∼ 300 km (which is optimistic, as this is ap-          cross-section, the optical depth is determined by the integral
propriate for the shortest wavelengths discussed here, not UV/                                            Z
visible wavelengths) will encounter 1020 molecules per square                                         τλ = na σ λ ds;                           [4]
centimeter, which is not optically thick to Rayleigh scattering by
molecular nitrogen except at extreme UV wavelengths (∼ 40 nm)
and shorter. Thus, Rayleigh scattering slopes in transit spectra,          where integration proceeds along a ray’s path shown in Fig. 1.
which have been proposed for constraining partial pressures due            Ignoring refraction, we have ds = ðRp + zÞdϕ=sinðϕÞ and Rp + z =
to spectrally inactive gases (52), may not be accessible in hazy           b=sinðϕÞ, so that
atmospheres.
   Recently, the 6-Earth-mass transiting planet GJ 1214b (54)                                                Zπ=2
                                                                                                                       b
has been the target of many observational campaigns to char-                                        τλ = 2                  na σ λ dϕ;                [5]
acterize the nature of its atmosphere (5, 55, 56). This is the                                                      sin ðϕÞ
                                                                                                                       2
                                                                                                             0
smallest planet with transit spectra observations, which appear to
be flat at the 30-ppm level from 1.1 to 1.7 μm (57), and this trend        where we have exploited the symmetry about ϕ = π=2. If the
may extend to 5 μm (58). A Titan-like haze has been proposed as            absorber is distributed with a scale height Ha , with na = na0
a viable explanation for these observations (17–19), and the data          exp½−ðz − z0 Þ=Ha , where na0 is the number density at the altitude
constrain this haze to be above the 10−1  10−2 mbar level, de-            z0 , and assuming that the absorption cross-section is a power law
pending on atmospheric composition (57). Although the methane              in wavelength, σ λ = σ λ0 ðλ=λ0 Þβ , where λ is wavelength, σ λ0 is the
concentration in the atmosphere of GJ 1214b is unknown, the                fiducial value at λ0 , and β defines the slope of the power law, we
high-altitude haze interpretation is not entirely consistent with          then have
the observations presented here. Extending Titan’s haze to the
aforementioned low pressures would mask the methane features,                                       β          Zπ=2 −b=Ha sin ϕ
but still would not produce a flat spectrum due to the wavelength-                                  λ   b RpH+z0     e
dependent haze opacity. Thus, a Titan-like haze on GJ 1214b would                         τλ = 2τ0         e a                    dϕ;                 [6]
                                                                                                    λ 0 Ha             sin2 ðϕÞ
need to contain a continuum of effective particle radii that extends                                                           0
to sizes larger than is seen for Titan [the haze particles of which
have a characteristic size of about 1–2 μm (47), and are aggregates        where τ0 = na0 σ λ0 Ha is a reference vertical optical depth. Making
of smaller-sized monomers], as larger particles would tend to pro-         the substitution coshðyÞ = 1=sinðϕÞ, we have
duce a flatter spectrum. However, these larger-sized particles may
be rather difficult to keep aloft at such low pressures (59), especially                            β          Z∞
                                                                                                    λ   b RpH+z0             b coshðyÞ
given that the gravitational acceleration for GJ 1214b is nearly an                     τλ = 2τ0           e a      coshðyÞe− Ha dy;                  [7]
                                                                                                    λ 0 Ha
order of magnitude larger than that in Titan’s upper atmosphere.                                                           0

Conclusions                                                                which has the analytic solution given pﬃﬃﬃﬃﬃﬃﬃby Eq. 3. Note that, for
                                                                                                                                     pﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
We developed a technique for adapting occultation measure-                 large b=Ha , we have K1 ðb=Ha Þ ∼ π=2 expð−b=Ha Þ= b=Ha , so
ments of solar system worlds into transit radius spectra suitable          that Eq. 3 gives
for model validation and comparison with exoplanet observations.
We applied this technique to Titan, deriving realistic spectra that                                   β sﬃﬃﬃﬃﬃﬃﬃ
                                                                                                      λ    2πb −ðb−Rp −z0 Þ=Ha
inherently include effects due to gas absorption, refraction, and                           τλ ∼ τ 0               e             ;               [8]
haze scattering, and used these spectra to better understand the                                      λ0    Ha
effects of high-altitude hazes on transit observations. Absorption
features due to methane are clearly visible, and weaker features           which is in agreement with Fortney (9).
due to acetylene, carbon monoxide, and a C-H stretching mode of
aliphatic hydrocarbon chains.                                              ACKNOWLEDGMENTS. We thank W. B. Hubbard, P. Muirhead, and an
                                                                           anonymous referee for friendly and constructive feedback on earlier versions of
   The continuum level in our spectra is set by Titan’s extensive          this work. T.D.R. acknowledges support from an appointment to the NASA
haze, and is well reproduced by an analytic haze extinction                Postdoctoral Program at NASA Ames Research Center, administered by Oak
model derived here. Haze has a dramatic effect on the transit              Ridge Affiliated Universities. L.M. thanks the Agence Nationale de la Recherche


9046 | www.pnas.org/cgi/doi/10.1073/pnas.1403473111                                                                                        Robinson et al.
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
(ANR Project “Analysis of Photometric Observations for the Study of Titan                      NASA’s Planetary Atmospheres program. J.J.F. also acknowledges support
Climate” 11BS56002, France). M.S.M. and J.J.F. acknowledge support from                        from the National Science Foundation.


 1. Sánchez-Lavega A, Pérez-Hoyos S, Hueso R (2004) Clouds in planetary atmospheres: A         31. Gunson MR, et al. (1996) The atmospheric trace molecule spectroscopy (atmos) ex-
    useful application of the Clausius–Clapeyron equation. Am J Phys 72(6):767–774.                periment: Deployment on the atlas space shuttle missions. Geophys Res Lett 23(17):
 2. Pont F, Knutson H, Gilliland R, Moutou C, Charbonneau D (2008) Detection of at-                2333–2336.
    mospheric haze on an extrasolar planet: The 0.55–1.05 μm transmission spectrum of          32. Maltagliati L, et al. (2013) Annual survey of water vapor vertical distribution and
    HD 189733b with the Hubble space telescope. Mon Not R Astron Soc 385(1):109–118.               water–aerosol coupling in the Martian atmosphere observed by SPICAM/MEx solar
 3. Lecavelier Des Etangs A, Pont F, Vidal-Madjar A, Sing D (2008) Rayleigh scattering in          occultations. Icarus 223(2):942–962.
    the transit spectrum of HD 189733b. Astron Astrophys 481(2):L83–L86.                       33. Baum WA, Code AD (1953) A photometric observation of the occultation of σ ARIETIS
 4. Sing D, et al. (2009) Transit spectrophotometry of the exoplanet HD 189733b. i.                by Jupiter. Astron J 58:108–112.
    Searching for water but finding haze with HST NICMOS. Astron Astrophys 505(2):             34. van der Werf SY (2008) Comment on “Improved ray tracing air mass numbers model”.
    891–899.                                                                                       Appl Opt 47(2):153–156.
 5. Bean JL, Kempton EMR, Homeier D (2010) A ground-based transmission spectrum of             35. Waite J, Bell J, Lorenz R, Achterberg R, Flasar F (2013) A model of variability in Titan’s
    the super-Earth exoplanet GJ 1214b. Nature 468(7324):669–672.                                  atmospheric structure. Planet Space Sci 86:45–56.
 6. Gibson NP, Pont F, Aigrain S (2011) A new look at NICMOS transmission spectroscopy         36. Washburn EW (1930) International Critical Tables of Numerical Data: Physics, Chem-
    of HD 189733, GJ-436 and xo-1: No conclusive evidence for molecular features. Mon              istry and Technology (McGraw-Hill, New York), Vol 7.
    Not R Astron Soc 411(4):2199–2213.                                                         37. Weber M (2002) Handbook of Optical Materials, Laser & Optical Science & Technology
 7. Seager S, Sasselov D (2000) Theoretical transmission spectra during extrasolar giant           (CRC Press, New York).
    planet transits. Astrophys J 537(2):916–921.                                               38. Betremieux Y, Kaltenegger L (2013) Impact of atmospheric refraction: How deeply
 8. Charbonneau D, Brown TM, Noyes RW, Gilliland RL (2002) Detection of an extrasolar              can we probe exo-Earth’s atmospheres during primary eclipse observations? arXiv:
    planet atmosphere. Astrophys J 568(1):377–384.                                                 1312.6625.
 9. Fortney JJ (2005) The effect of condensates on the characterization of transiting          39. French RG (1977) On the Theory and Analysis of Occultation Light Curves. PhD thesis
    planet atmospheres with transmission spectroscopy. Mon Not R Astron Soc 364(2):                (Cornell University, Ithaca, NY).
    649–653.                                                                                   40. Hubbard WB (1977) Wave optics of the central spot in planetary occultations. Nature
10. Brown TM (2001) Transmission spectra as diagnostics of extrasolar giant planet at-             268(5615):34–35.
    mospheres. Astrophys J 553(2):1006–1026.                                                   41. Hui L, Seager S (2002) Atmospheric lensing and oblateness effects during an extra-
11. Hubbard W, et al. (2001) Theory of extrasolar giant planet transits. Astrophys J 560(1):
                                                                                                   solar planetary transit. Astrophys J 572(1):540–555.
    413–419.
                                                                                               42. Bétrémieux Y, Kaltenegger L (2013) Transmission spectrum of earth as a transiting
12. Fortney J, et al. (2003) On the indirect detection of sodium in the atmosphere of the
                                                                                                   exoplanet from the ultraviolet to the near-infrared. Astrophys J 772:L31.
    planetary companion to HD 209458. Astrophys J 589(1):615–622.
                                                                                               43. Hörst SM, Vuitton V, Yelle RV (2008) Origin of oxygen species in Titan’s atmosphere.
13. Barman T (2007) Identification of absorption features in an extrasolar planet atmo-
                                                                                                   J Geophys Res Planets 113(E10):E10006.
    sphere. Astrophys J 661(2):L191–L194.
                                                                                               44. Sromovsky LA, et al. (1981) Implications of Titan’s north–south brightness asymmetry.
14. Miller-Ricci E, Seager S, Sasselov D (2009) The atmospheric signatures of super-Earths:
                                                                                                   Nature 292(5825):698–702.
    How to distinguish between hydrogen-rich and hydrogen-poor atmospheres. As-
                                                                                               45. Rannou P, Hourdin F, McKay CP (2002) A wind origin for Titan’s haze structure. Na-
    trophys J 690(2):1056–1067.
                                                                                                   ture 418(6900):853–856.
15. Kaltenegger L, Traub W (2009) Transits of earth-like planets. Astrophys J 698(1):
                                                                                               46. Rannou P, Montmessin F, Hourdin F, Lebonnois S (2006) The latitudinal distribution of
    519–527.
                                                                                                   clouds on Titan. Science 311(5758):201–205.
16. De Kok R, Stam D (2012) The influence of forward-scattered light in transmission



                                                                                                                                                                                                ASTRONOMY
                                                                                               47. Tomasko M, et al. (2008) A model of Titan’s aerosols based on measurements made
    measurements of (exo) planetary atmospheres. Icarus 221(2):517–524.
                                                                                                   inside the atmosphere. Planet Space Sci 56(5):669–707.
17. Kempton EMR, Zahnle K, Fortney JJ (2012) The atmospheric chemistry of GJ 1214b:
                                                                                               48. Hubbard W, et al. (1993) The occultation of 28 sgr by Titan. Astron Astrophys 269:541–563.
    Photochemistry and clouds. Astrophys J 745(1):3.
                                                                                               49. de Wit J, Seager S (2013) Constraining exoplanet mass from transmission spectros-
18. Howe AR, Burrows AS (2012) Theoretical transit spectra for GJ 1214b and other
                                                                                                   copy. Science 342(6165):1473–1477.
    “Super Earths”. Astrophys J 756(2):176.
                                                                                               50. Madhusudhan N, Seager S (2009) A temperature and abundance retrieval method for
19. Morley CV, et al. (2013) Quantitatively assessing the role of clouds in the transmission
                                                                                                   exoplanet atmospheres. Astrophys J 707(1):24–39.
    spectrum of GJ 1214b. Astrophys J 775(1):33.
                                                                                               51. Lee JM, Fletcher LN, Irwin PGJ (2012) Optimal estimation retrievals of the atmospheric
20. Lunine JI (2010) Titan and habitable planets around M-dwarfs. Faraday Discuss 147:
    405–418, discussion 527–552.                                                                   structure and composition of HD 189733b from secondary eclipse spectroscopy. Mon
21. Porco CC, et al. (2005) Imaging of Titan from the Cassini spacecraft. Nature 434(7030):        Not R Astron Soc 420(1):170–182.
    159–168.                                                                                   52. Benneke B, Seager S (2012) Atmospheric retrieval for super-earths: Uniquely con-
22. Bellucci A, et al. (2009) Titan solar occultation observed by Cassini/VIMS: Gas ab-            straining the atmospheric composition with transmission spectroscopy. Astrophys J
    sorption and constraints on aerosol composition. Icarus 201(1):198–216.                        753(2):100.
23. Brown RH, Lebreton JP, Waite JH (2009) Titan from Cassini-Huygens (Springer, New           53. Line MR, et al. (2013) A systematic retrieval analysis of secondary eclipse spectra. I.
    York).                                                                                         A comparison of atmospheric retrieval techniques. Astrophys J 775(2):137.
24. Elliot J, Olkin C (1996) Probing planetary atmospheres with stellar occultations. Annu     54. Charbonneau D, et al. (2009) A super-Earth transiting a nearby low-mass star. Nature
    Rev Earth Planet Sci 24(1):89–123.                                                             462(7275):891–894.
25. Broadfoot AL, et al. (1979) Extreme ultraviolet observations from Voyager 1 en-            55. Désert JM, et al. (2011) Observational evidence for a metal-rich atmosphere on the
    counter with Jupiter. Science 204(4396):979–982.                                               super-earth GJ1214b. Astrophys J 731(2):L40.
26. Hubbard W, Hunten D, Dieters S, Hill K, Watson R (1988) Occultation evidence for an        56. Berta ZK, et al. (2012) The flat transmission spectrum of the super-earth GJ1214b from
    atmosphere on Pluto. Nature 336(6198):452–454.                                                 wide field camera 3 on the Hubble space telescope. Astrophys J 747(1):35.
27. Pallé E, Osorio MRZ, Barrena R, Montañés-Rodríguez P, Martín EL (2009) Earth’s             57. Kreidberg L, et al. (2014) Clouds in the atmosphere of the super-Earth exoplanet GJ
    transmission spectrum from lunar eclipse observations. Nature 459(7248):814–816.               1214b. Nature 505(7481):69–72.
28. Vidal-Madjar A, et al. (2010) The earth as an extrasolar transiting planet. Earth’s at-    58. Fraine JD, et al. (2013) Spitzer transits of the super-earth GJ1214b and implications for
    mospheric composition and thickness revealed by lunar eclipse observations. Astron             its atmosphere. Astrophys J 765(2):127.
    Astrophys 523:57.                                                                          59. Spiegel DS, Silverio K, Burrows A (2009) Can Tio explain thermal inversions in the
29. Brown R, et al. (2004) The Cassini visual and infrared mapping spectrometer in-                upper atmospheres of irradiated giant planets? Astrophys J 699(2):1487–1500.
    vestigation. The Cassini-Huygens Mission, ed Russell CT (Springer, New York), pp           60. Banfield D, Gierasch P, Conrath B, Nicholson P, Hedman M (2011) Saturn’s He and
    111–168.                                                                                       CH4 abundances from Cassini VIMS occultations & CIRS limb spectra. EPSC-DPS
30. Maltagliati L, et al. (2014) Titan’s atmosphere as observed by VIMS/Cassini solar oc-          Joint Meeting 2011:1548. Available at http://meetingorganizer.copernicus.org/
    cultations: CH4, CO and evidence for C2H6 absorption. arXiv:1405.6324.                         EPSC-DPS2011/EPSC-DPS2011-1548-2.pdf. Accessed May 19, 2014.




Robinson et al.                                                                                                              PNAS | June 24, 2014 | vol. 111 | no. 25 | 9047
```
