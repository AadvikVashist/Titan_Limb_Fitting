---
citation_key: "brown2004cassini"
title: "The Cassini visual and infrared mapping spectrometer (VIMS) investigation"
source_pdf: "data/papers/brown2004cassini.pdf"
source_pdf_sha256: "33c5264a546ce7f8e4da0916620ffe02370cd62aeb6db49d329a8f4ac97d2018"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
 THE CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER
                   (VIMS) INVESTIGATION


    R. H. BROWN∗ , K. H. BAINES, G. BELLUCCI, J.-P. BIBRING, B. J. BURATTI,
 F. CAPACCIONI, P. CERRONI, R. N. CLARK, A. CORADINI, D. P. CRUIKSHANK,
  P. DROSSART, V. FORMISANO, R. JAUMANN, Y. LANGEVIN, D. L. MATSON,
 T. B. MCCORD, V. MENNELLA, E. MILLER, R. M. NELSON, P. D. NICHOLSON,
                           B. SICARDY AND C. SOTIN
   Department of Planetary Sciences, 1629 University Boulevard, University of Arizona, Tucson,
                                       AZ 85721, U.S.A.
              (∗ Author for correspondence: E-mail address: rhb@lpl.arizona.edu)

                  (Received 5 April 2002; Accepted in final form 2 January 2004)




Abstract. The Cassini visual and infrared mapping spectrometer (VIMS) investigation is a multidis-
ciplinary study of the Saturnian system. Visual and near-infrared imaging spectroscopy and high-speed
spectrophotometry are the observational techniques. The scope of the investigation includes the rings,
the surfaces of the icy satellites and Titan, and the atmospheres of Saturn and Titan. In this paper, we
will elucidate the major scientific and measurement goals of the investigation, the major characteris-
tics of the Cassini VIMS instrument, the instrument calibration, and operation, and the results of the
recent Cassini flybys of Venus and the Earth–Moon system.

Keywords: Cassini, Saturn, infrared mapping spectrometer



                                         1. Introduction

The visual and infrared mapping spectrometer (VIMS) is a state-of-the-art, imaging
spectrometer that spans the 0.3–5.1 µm wavelength range. It obtains observations
of targets in the Saturnian system. These data are used by the team members to carry
out many, different, multidisciplinary investigations. These studies, for example,
seek to increase our knowledge about atmospheric processes, the nature of the rings,
and the mineralogical composition of surfaces.
    The VIMS strategy is to measure scattered and emitted light from surfaces and
atmospheres. While the spectral domain is emphasized, spatial resolution is also
important particularly when it allows the correlation of the spectral characteristics
with surface or atmospheric features. Since the spatial domain is important for
understanding and interpreting the data, close collaboration is maintained with
other, primarily imaging, investigations that are aboard Cassini. The addition of data
of relatively high-spatial resolution, but of modest spectral resolution, to a data set of
high spectral resolution and modest spatial resolution results in synergistic scientific
advances well beyond those which could be accomplished using either data set
separately. Conversely, very-high spectral-resolution instruments (e.g. CIRS) on
      Space Science Reviews 115: 111–168, 2004.
       2004 Kluwer Academic Publishers. Printed in the Netherlands.
      C
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
112                               R.H. BROWN ET AL.

the Cassini spacecraft offer a different set of synergistic interactions for VIMS.
Thus, the Cassini VIMS investigation will perform much of its work with an eye
toward the synergy it has with other experiments.

1.1. T ECHNICAL H ERITAGE OF THE VIMS I NSTRUMENT

The Cassini VIMS instrument has its roots in a long line of proposed imaging
spectrometers, starting with a rudimentary instrument proposed for the Lunar Polar
Orbiter in 1974. A more refined concept was proposed for the Mariner Jupiter
Uranus (MJU) mission in 1975. This design was accepted for the Jupiter Orbiter
Probe (JOP, later named Galileo) Mission in 1977. After selection, this design
rapidly evolved into that of the NIMS instrument. Further consideration of these
concepts for terrestrial applications led the Jet Propulsion Laboratory to propose to
design and develop the airborne visible/infrared spectrometer (AVIRIS) in 1983.
In 1987 this system measured its first spectral images (Green et al., 1998). Also, in
the early 1980’s, an instrument development program for future planetary missions
was setup at JPL and headed by Larry Soderblom as principal investigator.
    This collaboration resulted in several similar instruments optimized for different
planetary targets, most notably the OMEGA instrument, the Mars Observer VIMS,
and the CRAF VIMS. Mars Observer VIMS and CRAF VIMS are Cassini VIMS’
most immediate forbearers, but changes made in the VIMS instrument design in
response to NASA’s demands to reduce the cost of the strawman Cassini VIMS
resulted in a substantial heritage from Galileo NIMS. Despite that heritage, which
goes so far as to include parts from the Galileo NIMS engineering model, Cassini
VIMS is a substantial step beyond NIMS in the evolution of visual and infrared
imaging spectrometers.
    This program developed imaging spectrometer designs for the Comet Ren-
dezvous Asteroid Flyby (CRAF), the Mars Observer (MO), and the Cassini mis-
sions. While these spectrometers were selected as facility instruments on all three
missions, the CRAF mission was cancelled and the MO instrument was removed
from the payload in favor of the remaining payload elements that never reached
Mars. Only the Cassini instrument survived and flew in space.
    Cassini VIMS has inherited much from the NIMS instrument, including me-
chanical and optical parts from the original NIMS engineering model. VIMS nev-
ertheless represents a substantial improvement over the state of the art as marked
by the NIMS instrument. In particular, where the NIMS instrument incorporates a
moving grating that scans a multiple-order spectrum across a linear array of dis-
crete detectors, the VIMS instrument includes a fixed, multiply blazed grating and
array detectors in both its visual and near-infrared channels. Another important
improvement is the way the visual and infrared channels scan spatial targets. The
VIMS-VIS (visual channel) uses a two-dimensional array detector and a scanned
slit (literally scanning a two-dimensional scene across the spectrometer slit) to pro-
vide two-dimensional spatial coverage. The infrared channel has only a linear array
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                    113

of detectors and therefore must use a two-dimensional scanning secondary mirror
to obtain the same spatial coverage. The NIMS instrument by contrast scans a scene
in only one direction while using the relative motion between the target and the
spacecraft to scan the other spatial dimension.
    The major differences between the Cassini VIMS and the Galileo NIMS lie in
the incorporation of a separate visual channel using a frame-transfer, silicon CCD
detector with separate foreoptics and analog/control electronics, the inclusion of
a radically improved infrared detector with improved order sorting and thermal
background rejection, an improved infrared focal plane cooler design, an improved
main electronics design, a two-dimensional, voice-coil-actuator-driven, scanning
secondary mirror in the infrared foreoptics, a fixed, triply blazed grating in the
infrared, a redundant 16-megabit buffer, and a redundant, lossless, hardware data
compressor using a unique compression algorithm developed by Yves Langevin.
These improvements result in an instrument with substantially greater capability
for planetary imaging spectroscopy—so much so that Cassini VIMS is the most
capable and complex imaging spectrometer presently flying on a NASA planetary
spacecraft.


1.2. I NTERNATIONAL C OOPERATION

The VIMS investigation is truly an international enterprise. The instrument resulted
from the best thinking and skill of scientists and engineers in the United States, Italy,
France, and Germany. The bonds of cooperation and friendship formed during the
design, construction, and testing of this pioneering instrument have now resulted in
the closely knit team that is using this instrument to obtain new knowledge about
the Saturnian system.

1.2.1. The VIMS Engineering Team
The Cassini VIMS Development Project began with the Cassini instrument selec-
tions in November 1990 and ended 30 days after the Cassini launch on October 15,
1997. Overall management responsibility was awarded to the NASA Jet Propul-
sion Laboratory, but soon after that selection, agencies of Italy and France were
invited to join the instrument engineering team. In the resulting division of respon-
sibilities, VIMS-VIS (visual spectral region component) was designed and built by
Officine Galileo in Florence Italy for the Agenzia Spaziale Italiana (ASI); the VIMS
data compressors and data buffers were supplied by the Institut d’Astrophysique
Spatiale (IAS) in Orsay, France for the Centre National de la Recherche Sci-
entifique (CNRS); and VIMS-IR (infrared spectral region component) was de-
signed and built by the NASA Jet Propulsion Laboratory (JPL) in Pasadena,
California. JPL also conducted the integration, test, and calibration of the instru-
ment with involvement by the science team and members of all the engineering
teams.
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
114                                      R.H. BROWN ET AL.

   Note that throughout this document we denote the visible spectral region com-
ponent (channel) of VIMS as VIMS-VIS, and the infrared component (channel) as
VIMS-IR.
   The original VIMS project manager at JPL was Robert F. Lockhart. Mr. Lockhart
was later asked to lead the effort to build the entire suite of four Cassini facility
instruments, including VIMS, and Dr. Gail Klein, then the deputy project manager
for VIMS, assumed the VIMS project manager role. David Juergens continued in
the role of instrument manager and Edward Miller continued as the senior VIMS
system engineer. In Italy, Enrico Flamini of the Agenzia Spaziale Italiana directed
the VIMS-VIS Visual Channel (VIMS-VIS) project, with Romeo DeVidi as the
VIMS-VIS project manager at Officine Galileo. Francis Reininger served as the
Visual Channel system engineer. In France, Alain Soufflot and Yves Langevin led
the development of the VIMS data compressor and buffer hardware at IAS.
   In total, over 50 engineers in France, Italy, and the US collaborated to design
and build the VIMS instrument. The instrument has to date operated flawlessly en
route to Saturn, a testament to the cohesion and dedication of the engineering team.


1.2.2. The VIMS Investigation (Science) Team
The VIMS Science Team came into existence on November 5, 1990 when the
US National Aeronautics and Space Administration announced the selections of
the various instrument and investigation teams for the Cassini Orbiter. The origi-
nal VIMS science team consisted of: Robert H. Brown (Team Leader), Kevin H.
Baines, Jean-Pierre Bibring, Andrea Carusi, Roger N. Clark, Michael Combes,
Angioletta Coradini, Dale P. Cruikshank, Pierre Drossart, Vittorio Formisano, Ralf
Jaumann, Yves Langevin, Dennis L. Matson, Robert M. Nelson, Bruno Sicardy,
and Christophe Sotin.1
   There have been some changes since the original selection. Added to team
membership were: Alberto Adriani, Giancarlo Bellucci, Bonnie J. Buratti, Ezio
1 Institutional affiliations are as follows (in alphabetical order by last name): Alberto Adriani (CNR

Istituto di Astrofisica Spaziale, Italy), Kevin H. Baines (NASA Jet Propulsion Laboratory), Jean-
Pierre Bibring (Universite de Paris Sud-Orsay, France), Giancarlo Bellucci (CNR Istituto Fisica
Spazio Interplanetario, Italy), Bonnie J. Buratti (NASA Jet Propulsion Laboratory), Robert H. Brown
(University of Arizona, Team Leader), Ezio Bussoletti (Instituto Universario Navale, Italy), Fabrizio
Capaccioni (CNR Istituto di Astrofisica Spaziale, Italy), Andrea Carusi (CNR Istituto di Astrofisica
Spaziale, Italy), Priscilla Cerroni (CNR Istituto di Astrofisica Spaziale, Italy), Roger N. Clark (U.S.
Geological Survey, Denver), Michael Combes (Observatoire de Paris-Meudon, France), Angioletta
Coradini (CNR Istituto di Astrofisica Spaziale, Italy), Dale P. Cruikshank (NASA Ames Research
Center), Pierre Drossart (Observatoire de Paris-Meudon, France), Vittorio Formisano (CNR Istituto
Fisica Spazio Interplanetario, Italy), Ralf Jaumann (Institute for Planetary Exploration, DLR, Berlin,
Germany), Yves Langevin (Universite de Paris Sud-Orsay, France), Thomas B. McCord (University
of Hawaii, Honolulu), Dennis L. Matson, (NASA Jet Propulsion Laboratory), Vito Mennella (Osser-
vatorio Astronomico di Capodimonte, Italy), Phillip D. Nicholson (Cornell University), Robert M.
Nelson (NASA Jet Propulsion Laboratory), Bruno Sicardy (Observatoire de Paris-Meudon, France)
Christophe Sotin (Universite de Nantes, Nantes France).
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   115

Bussoletti, Fabrizio Capaccioni, Priscilla Cerroni, Thomas B. McCord, Vito Men-
nella, and Phillip D. Nicholson.1 Resignations were received from Andrea Carusi,
Alberto Adriani, and Ezio Bussoletti.
   The VIMS science team members’ scientific interests range over three major
areas: planetary surfaces, atmospheres and rings. The combined scientific goals of
the VIMS scientific team are discussed later in this paper.

                  2. Scientific Goals of the Vims Investigation

At the time of this writing, the Cassini spacecraft has flown by Venus twice, the
Earth once, and Jupiter once. Later in this paper we describe some of the obser-
vations made for the purposes of instrument test-and-calibration during the Venus
encounters and the Earth flyby. Scientific results derived from the flybys of Venus
and Jupiter are published elsewhere (Baines et al., 2000; Brown et al., 2003; Mc-
Cord et al., 2004). Now we discuss the scientific goals addressed during the cruise
phase and orbital tour of Saturn.

2.1. V ENUS

The specific scientific objectives for Venus arose from VIMS unique ability to
peer into Venus’ turbulent middle atmosphere as well as the cloud-covered upper
atmosphere. Unfortunately, the fact that VIMS-IR was not operational at Venus,
combined with the restricted attitude of the spacecraft dictated by the need to protect
the Huygens probe from excessive insolation, made it impossible to obtain a data
set at Venus which would allow us to address the list of scientific goals originally
envisioned for Venus. Nevertheless, useful data were obtained with VIMS-VIS. In
particular, VIMS-VIS saw the surface of Venus for the first time at multiple sub-
micrometer wavelengths, proving that sub-micrometer spectroscopy of the Venus
surface is possible. In Section 6 we detail the results obtained during the second
Venus flyby (see also Baines et al., 2000).

2.2. J UPITER

The VIMS scientific objectives for Jupiter were built upon the results obtained by
the Galileo investigations (up until the late fall of 2000). VIMS extended this cov-
erage in time and illumination/emission geometries, as well as provided contiguous
spectral coverage into the visual and ultraviolet. Overall, the unique circumstances
of the several-month-long Cassini fly-by enabled full global maps of Jupiter over
both, a large and continuous range of wavelengths spanning nearly the entire so-
lar spectrum, and over a large range of phase angles. Specific VIMS scientific
objectives at Jupiter were:
1. Determine the vertical aerosol distributions and microphysical(optical aerosol
   properties of Jupiter’s atmosphere as a function of latitude and longitude.
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
116                               R.H. BROWN ET AL.

2. Determine the temporal and spatial variations in constituent gases, including
   condensables (ammonia, water), and disequilibrium gases (phosphine and or-
   tho/para hydrogen).
3. Investigate the temporal and spatial distribution of lightning, and measure the
   emission spectrum of lightning from the UV to the near infrared.
4. Determine the vertical and horizontal solar flux deposition in Jupiter’s atmo-
   sphere to constrain the role of solar energy in Jupiter’s meteorology.
5. Investigate auroral and polar haze phenomena, particularly the polar aerosol
   burden and the abundance of H+   3 Z.
6. Study the temporal evolution of the surfaces of the satellites of Jupiter, par-
   ticularly that of Io, using VIMS data compared to those of the Galileo NIMS
   instrument.
7. Investigate time variations in the Io torus.
8. Measure the spectrum of Himalia.
9. Measure the spectrum of Jupiter’s Ring.

2.3. THE O RBITAL TOUR OF THE SATURNIAN SYSTEM

The overall objectives of VIMS for Saturn and Titan are to investigate ongo-
ing chemical and dynamical processes in a diverse range of planetary and satel-
lite atmospheres, and determine constraints on planetary formation processes
and evolutionary histories. The multi-dimensional characteristics of VIMS data
acquisition—namely, the ability to acquire two-dimensional spectral maps in 352
wavelengths from the ultraviolet to the thermal infrared allows the instrument to ob-
tain three-dimensional views of atmospheric thermal, aerosol, and chemical struc-
tures. Furthermore, these will be over a wide variety of illuminations, and over
many emission angles. Meaningful parameters and relationships can be rendered
into two-dimensional maps. These then become the tools that enable diverse inves-
tigations of chemical, dynamical, and geophysical phenomena. Important targets
include the surface and atmosphere of Titan, the cloud-rich atmosphere of the planet
itself, its rings, and the plethora of icy moons. Observations of both the day and
night sides of these objects shall lead to increased insights into various phenomena
involving both reflection and emission of radiation. Occultations of the Sun and
stars by these objects should provide new insights into the nature of tenuous strato-
spheric hazes on Saturn and Titan, the structure of faint rings, and atmospheric
composition. Thus, the Saturnian system provides VIMS with many opportunities.

2.3.1. Titan
VIMS data will constrain gas and aerosol distributions in Titan’s atmosphere as well
as their variability with time, and the operative chemical and dynamical processes
(Baines et al., 1992). In addition, it is now known that VIMS will be able to see
the surface of Titan through atmospheric windows in the infrared (e.g. Tomasko
et al., 1989; Griffith and Owen, 1992; Stammes, 1992; Baines et al., 1992; Meier
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                  117

et al., 2000). Therefore, VIMS studies of Titan will include both the surface and
atmosphere and will help in the study of the surface-atmosphere interactions. VIMS
objectives for Titan include:

 1. Determination of Titan’s surface properties using VIMS data obtained at the
    wavelengths of near-infrared atmospheric windows in Titan’s atmosphere (e.g.
    0.95, 1.1, 1.3, 1.6, 2.0, 2.7 µm).
 2. Measurement of the vertical distribution of condensable and chemically active
    gas species, including methane, ethane, and acetylene, along with their spatial
    and temporal variability. In particular, studies of distributions at microbar and
    millibar levels using stellar occultation measurements.
 3. Determination of stratospheric temperature profiles from multispectral occul-
    tation measurements.
 4. Determination of equipotential surfaces near the 1 mbar level, and inference of
    geostrophic winds.
 5. Determination of vertical aerosol distributions and associated microphysical
    and optical properties, over latitude/longitude and time. Studies of stratospheric
    aerosol distributions from stellar occultation measurements.
 6. Determination of the bolometric Bond albedo for comparison to CIRS-
    determined temperature measurements.
 7. Determination of the three-dimensional distribution of solar flux deposition
    rates to constrain the role of solar energy in powering dynamics at various
    levels, and to constrain the surface solar energy flux over latitude, longitude,
    and time.
 8. Determination of wind fields as revealed by movements of spatially varying
    clouds and hazes.
 9. Determine the composition of Titan’s surface, and map that composition as a
    function of longitude and latitude.
10. Studies of the geology of Titan’s surface and any correlation between geomor-
    phology and composition.
11. Searches for signs of active volcanism and tectonism on Titan’s surface.

2.3.2. Saturn
VIMS will take advantage of the unique geometry of the Cassini orbital tour (en-
compassing both near-polar and equatorial Saturnian orbits) and the 4-year time
window to obtain detailed four-dimensional views (i.e. over latitude, longitude,
altitude, and time) of Saturn’s atmospheric phenomena. Specific VIMS scientific
objectives for Saturn include:

 1. Determination of vertical aerosol distributions, and associated microphysical
    and optical properties, over latitude, longitude and time. In addition, provide
    enhanced accuracy in the determination of stratospheric aerosol properties us-
    ing stellar occultation measurements.
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
118                                R.H. BROWN ET AL.

 2. Detection and characterization of spectrally identifiable ammonia cloud fea-
    tures. In particular, determine their spatial distribution and vertical and micro-
    physical properties. Such localized clouds have been identified on Jupiter by
    Galileo NIMS (c.f., Baines et al., 2002). Using similar techniques, their pres-
    ence on Saturn should be readily observed by Cassini VIMS. Also, the detection
    and determination of water clouds (recently identified in infrared observations
    by Simon-Miller et al., 2000) will be attempted.
 3. Determination of the vertical distributions of variable gas species, including
    condensables (ammonia and perhaps water) and disequilibrium species (phos-
    phine and ortho/para hydrogen), over latitude, longitude and time. Follow the
    lead of Galileo NIMS (e.g. Roos-Serote et al., 1999, 2000) in such analyses. In
    addition, study hydrocarbon and other gas distributions at microbar and millibar
    levels using stellar occultation measurements.
 4. Determination of stratospheric temperature profiles from multispectral occul-
    tation measurements.
 5. Determination of equipotential surfaces near the 1 mbar level, and inference of
    geostrophic winds.
 6. Determination, over time, of wind fields near the ammonia cloud tops and,
    perhaps, lower-lying clouds as probed by near-infrared wavelengths. Constrain
    vertical wind components from cloud top altitude variations.
 7. Improve the determination of the bolometric bond albedo of Saturn, and use in
    constraining the magnitude of Saturn’s internal heat source.
 8. Investigation of the three-dimensional distribution of lightning, and measure-
    ment of the emission spectrum of lightning from the UV to the near infrared.
 9. Place constraints on chemical/dynamical mechanisms from spatial/temporal
    variability in atmospheric phenomena. Specifically, investigate links between
    disequilibrium and condensable gas variations, as well as variations in cloud
    opacities and vertical distributions.
10. Determination of the three-dimensional solar flux deposition rate to constrain
    the role of solar energy in powering dynamics at various levels.
11. Investigation of links between polar aerosol burden and composition, and mag-
    netospheric phenomena utilizing charged particle flux estimates from Cassini
    plasma investigations.

2.3.3. Icy Satellites
Objectives for the icy satellites are:

1. Determine or constrain the mineralogical compositions of the satellite surfaces
   at maximum spatial resolution.
2. Determine the composition of the Iapetus dark side material and constrain the
   origin and evolution of Iapetus and its surface.
3. Relate the compositions of Saturn’s icy satellites to that of icy satellites of other
   planets (i.e. Jupiter, Uranus, Neptune, Pluto)
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   119

4. Constrain the insolation absorbed by the satellites of Saturn and thereby constrain
   the photometric and thermal properties of their regoliths.
5. For the smaller Saturnian satellites, obtain spectral data for comparison with
   the larger satellites, the ring system, and other small bodies of the outer Solar
   System.
6. Determine the composition of any dark material on Saturn’s satellites, espe-
   cially Iapetus, Hyperion and Phoebe, and relate the composition of this primi-
   tive/organic material to that on other solar system bodies, and to groundbased
   measurements of objects thought to have primitive material on their surfaces as
   well as to relevant laboratory data.

2.3.4. Rings
Ring objectives are:

1. Determine or constrain the mineralogical composition and grain-size distribution
   of small (micrometer-sized) as well as large (millimeter- and larger-sized) parti-
   cles in the rings of Saturn using absorption features found in reflected solar radia-
   tion. For rings that are optically thin, constrain, map variations in, and determine
   the size distribution of small (micrometer-sized) particles via multi-wavelength
   optical-depth measurements obtained from stellar and solar occultations.
2. Map the radial, azimuthal, and vertical distribution of ring material using stellar
   occultations, at radial resolutions of 0.1–1 km.
3. Probe dynamical phenomena responsible for ring structure and evolution, includ-
   ing density and bending waves, wakes due to embedded satellites, and resonantly
   forced ring edges and narrow ringlets, using multiple stellar occultation profiles.
4. Characterize the size distribution of ring particles by imaging the diffraction
   aureole during stellar occultations.
5. Establish the surface properties (grain size, degree of crystallinity, packing, etc.)
   of the bodies composing the rings using spectral reflectance measurements.
6. Set bounds on the insolation absorbed by the rings, and thereby constrain their
   photometric and thermal properties.
7. Determine the composition of any dark material in Saturn’s rings and relate its
   composition to that of other primitive/organic material elsewhere in the solar
   system and in the laboratory.


                              3. Measurement Goals

3.1. S URFACES

3.1.1. Spectral Imaging
The measurements required to meet the scientific objectives for surfaces can be
divided into four broad classes:
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
120                               R.H. BROWN ET AL.


Type S1: High-spectral- and low-spatial-resolution data acquired at great distances
  from the target. These data will provide orbital phase spectral information and
  will permit longitudinal mapping of the object at scales greater than 5◦ spherical
  lunes. Such observations, performed shortly after arrival at Saturn will provide
  a preliminary database for defining the spectral regions to be studied during
  subsequent targeted encounters.
Type S2: Observations with high spectral and modest spatial resolution on approach
  to a targeted satellite encounter until the object fills one VIMS field of view, or
  until mosaicing is required to obtain full areal coverage. The spatial resolution in
  this case ranges from ∼10 to ∼20 km. This will provide the best global coverage
  at highest spectral resolution.
Type S3: Observations with the highest spatial and spectral resolution allowing
  coverage of the entire illuminated hemisphere of the object in a limited time. In
  essence, this amounts to constructing the highest spatial resolution map at full
  spectral resolution in the shortest possible time consistent with the constraints of
  a given flyby. These types of observations will usually be made during targeted
  icy satellite flybys far enough ahead of closest approach that full coverage of the
  object can be made. The spectral ranges and resolutions and targeting will be the
  product of tradeoffs determined during observations of Types S1 and S2 above.
Type S4: High spatial and spectral resolution observations with limited coverage.
  This type of observation will occur near closest approach during a “close” flyby of
  an object. This observation type concentrates on obtaining high spatial resolution
  (at the expense of areal coverage). In practice, we envision riding along with the
  imaging science subsystem (ISS) and producing a complete VIMS image for each
  narrow-angle camera frame taken by the ISS. In terms of specific observations of
  objects in the Jupiter and Saturnian systems, the following discussion of essential
  observations is presented as a minimal requirement to fulfill the goals of this
  investigation. As the requirements of the total mission develop and are addressed,
  this list of observations will be augmented, particularly through obvious resource
  economies such as joint observations with other instruments.

3.1.2. The Galilean Satellites
Although the spatial resolution attained by VIMS during the Cassini Jupiter flyby
was limited to at best 1 pixel at closest approach, VIMS was able to provide in-
formation in the visual and near IR spectral range that was not obtained by either
Voyager or the Galileo NIMS spectrometer. During the Jupiter flyby, a full set of
Type S1 observations of the Galilean satellites were obtained. The reader is referred
to Brown et al. (2003) for a discussion of the results.

3.1.3. The Saturnian Satellites
The Phoebe flyby, prior to the start of the Cassini orbital tour, will permit observa-
tions of Types S1 through S3. Observations will commence in Type S1 and end up
in Type S3 depending upon the closest approach distance.
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   121

    The Cassini orbital tour will provide a variety of opportunities for Type S1
observations of the icy Saturnian satellites at solar phase angles from near zero to the
maximum solar cone angle constraint (∼160–170◦ ). The minimum requirement is
that these observations be made on both hemispheres of all objects at 5◦ increments
in solar phase angle and orbital phase angle. The close satellite encounters will
permit Types S2, S3 and S4 observations of each object. This will yield the near
complete spectral mapping of each satellite. Significant support imagery from the
ISS will be required for the success of this investigation, much of which will occur
as a matter of course as a result of these data being taken simultaneously with the
imaging data (VIMS-imaging ride-along mode).
    Many Type S1 observations will be made of Saturn’s satellites in order to con-
struct full solar and orbital phase curves. This can be done at times of low spacecraft
activity, reducing competition for spacecraft resources.
    The typical, targeted satellite flyby will be so fast that, in the time required
for a complete VIMS frame to be taken, large changes in phase angle will occur.
In addition, the size of the target will change by a large factor. The high spatial
resolution data obtained will be a singular event for best definition for subsequent
compositional mapping done in concert with ISS NA images. During the period
of such an encounter, continuous images will be taken beginning with sub-pixel
spatial resolution through the time of closest approach.

3.1.4. Titan
During each flyby of Titan, VIMS observations can commence several hours before
closest approach. Titan’s surface can be mapped at the wavelengths of near infrared
windows in Titan’s atmosphere. Type S4 observations will be performed in order
to obtain increasing spatial resolution as Cassini gets closer to Titan. Mosaics
composed of four images taken approximately 4 h before closest approach will
allow complete coverage of Titan’s hemisphere. Different integration times will
be used to measure the opacity of Titan’s atmosphere at the wavelengths of near-
infrared windows in Titan’s atmospheric spectrum. A resolution of about 10 km per
pixel can be achieved at 0.5 h before closest approach. A high-resolution mosaic of
Titan’s surface can be obtained using image cubes obtained on successive flybys.
This information will complement the observations of Titan’s surface by other
Cassini instruments and will allow the creation of a geological map of Titan’s
surface based on all Cassini observations.
    Observations obtained during the first Titan flyby will help characterize the
landing site of the Huygens probe. Also, at wavelengths longward of 3 µm, it
will be possible to map temperature variations expected to result from any ice
diapirs exposed at Titan’s surface. It may also be possible to sense the release
of volatiles from the interior of the satellite. Lenticulae on Europa are features
attributed to ice diapirs and have diameters around 10 km. Such a resolution can
be obtained by VIMS about half an hour before closest approach. Therefore, such
observations during selected flybys will be carried out so as to characterize such
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
122                                R.H. BROWN ET AL.

areas. In addition to observations dedicated for VIMS, we will acquire observations
while other imaging instruments have control of the spacecraft pointing.

3.2. ATMOSPHERES OF SATURN AND T ITAN AND ATMOSPHERE M ODES

For atmospheres there are two major types of measurement objectives with re-
spect to Saturn and Titan: spectral imaging and stellar occultation. A third type,
that of measuring solar occultations, may also be periodically undertaken. Spec-
tral imaging will be used to map and study trace gas and aerosol distributions,
microphysical properties of aerosols, aurorae, lightning production and associated
spectral/chemical characteristics, and, for Titan, surface mineralogy and topogra-
phy. Stellar occultations by the atmospheres of Saturn and Titan and observed by
VIMS will be used to study the vertical distribution of trace constituents (e.g. CH4 ,
C2 H2 CO, and CO2 ) and aerosols, particularly in Titan’s stratosphere. Some thermal
profile information may also be forthcoming from spectral regions where refraction
is the primary extinction mechanism.

3.2.1. Spectral Imaging—Spatial Distributions
Measurements for atmospheres largely follow those for satellite surfaces, in that
various combinations of spectral and spatial resolutions will be utilized, depending
on the target-spacecraft distance. The actual combinations however are different,
owing to the greater spectral activity in gases as compared to solids, the highest
spectral resolution modes will be used most frequently.
    Type A1: High-spectral-, high-spatial-resolution data sets acquired within 25 ob-
ject radii. The full spectral/spatial resolution and spectral coverage of VIMS-VIS
is utilized, as is the full (i.e. standard) spatial/spectral resolutions of VIMS-IR. The
data may be limited to specific targeted regions (such as particular regions of belts
and zones, storm systems, etc.). Multiple observations of these regions as a func-
tion of illumination and viewing geometry (i.e. center-to-limb observations as the
feature rotates with respect to the spacecraft, and observations acquired at various
phase angles) will enable the full three-dimensional character of the atmospheric
region to be determined. For Saturn, such detailed sets of observations are referred
to as Feature Tracks and Feature Campaigns. Specifically, a Feature Track is a set
of center-to-limb observations acquired at one phase angle; A Feature Campaign
comprises a set of Feature Tracks, obtained over the full range of available phase
angles during one orbit. It is expected that at least one Feature Campaign will be
acquired per orbit, comprised of at least five sets of Feature Tracks, with each Fea-
ture Track comprised of between three and seven individual Type A1 observations,
depending on the phase angle. In addition, a number of other Feature Tracks will
be conducted per orbit, scrutinizing various other features for temporal variability
and contextual relationship with primary Feature Campaign objects.
    Some Feature Tracks are obtained at the same time that data for Cylindrical
Maps are recorded. At these times large swaths of longitudes are mapped over a
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   123

relatively narrow range of latitudes. Other Feature Tracks are obtained in specific
remote-sensing Feature Track observations in cooperation with other remote sens-
ing instruments. These are typically accomplished at ranges less than seven Saturn
radii (R S ), in order to provide the necessary spatial resolution needed by the full
complement of remote-sensing instrumentation onboard Cassini. Still others are
accomplished during VIMS-specific feature tracks.
    In practice, it is also expected that many of the Feature Track observations will be
acquired in cooperative “ride-along” mode with other remote sensing instruments.
VIMS’s ability to image a 32 × 32 mrad field of view allows coverage of a wide
range of latitudes and longitudes, particularly when outside of 15 R S , thus enabling
many features to be captured within its field-of-view, even when spacecraft pointing
is being directed at other places on Saturn by other instruments.
    Feature Campaigns as done for Saturn will not be feasible for Titan, whose
atmosphere is expected to rotate much more slowly than the 10-hr Saturnian period.
The time when the spacecraft is within 25 Titan radii (RT ) is relatively short (i.e.
typically 2.5–5.5 h), but Feature Tracks will still be possible due to the rapidly
varying spacecraft-target geometry as Cassini flies over Titan. Approximately a
dozen such Feature Tracks are expected per Titan encounter.
    Type A2: High-spectral- and moderate-spatial-resolution data acquired outside
of 25 object radii, utilizing the highest spectral/spatial resolutions of VIMS. At this
range, global mosaics will be acquired wherein the target is covered pole-to-pole.
For Saturn, these mosaics may be built up from full-disk imaging or from a set
of pole-to-pole images acquired over a limited range of solar-illumination/viewing
geometry, taking advantage of the relatively rapid rotation rate of Saturn to enable
the acquisition of global mosaics under nearly uniform lighting conditions. For
Titan, full-disk imaging will be the norm. At least one mosaic of Saturn will be
acquired per orbit, lasting ten hours with a duty cycle of some 20%.
    Type A3: VIMS-IR only observations for nighttime thermal measurements.
These will be used particularly on Saturn to acquire high spatial-resolution maps
when periapsis occurs on the night side, primarily to map out tropospheric distri-
butions of trace constituents (e.g. PH3 , NH3 ). In particular, Cylindrical Maps will
be obtained wherein nadir views of Saturn’s deep atmosphere will be acquired,
peering through relative clearings in the clouds to deep-level “hot spots”. Ten hours
of continuous observations may be needed, depending on periapsis distance.
    Type A4: VIMS-IR long-integration observations for weak emissions. Methane
fluorescence will be observed on the dayside, and H+     3 Z emissions will be observed
on the night side using this technique. Observations of high-altitude emissions are
conducted within strong bands of atmospheric methane absorption. Integrations up
to 10 s may be used.
    Type A5: VIMS high-speed photometry mode for detection and characterization
of lightning emissions. Under nighttime conditions, while relatively close to the
planet (less than 10 R S ), VIMS stares at one location at a time, searching for light-
ning. When lightning strikes, the full spectrum is obtained, and acquiring data at
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
124                                 R.H. BROWN ET AL.

high rates (about 20 ms integrations), the chronological order of multiple lightning
strikes in a single locale can, in principle, be ascertained. From analysis of the light-
ning emission spectrum in and out of methane absorption bands, it is hoped that the
altitude of emission can be determined as well as the total energy for each lightning
strike.
3.2.2. Atmospheric Stellar Occultations
Stellar occultations will be attempted periodically during Titan encounters, for a
period of 5–10 min each. Integration times are typically 80 and 13 ms, respectively,
for VIMS-VIS and VIMS-IR. These high sampling rates thus enable VIMS-VIS
vertical resolutions of between 30 and 50 m for tangential spacecraft velocity com-
ponents, relative to Titan, of 3–5 km/s. Observations last 3–6 min on average.
Further details on particular occultation modes are discussed later.

3.3. R INGS

There are two major types of measurement objectives for the VIMS investiga-
tion with respect to Saturn’s ring system: Spectral imaging and stellar occultation.
Spectral imaging will be used to map and study the composition and mineralogy
of Saturn’s rings. Usually this will use image, line, or point instrument modes (see
Section 5.2.1 for definitions). Stellar occultations by the rings and observed by
VIMS will be used to study the dynamical structure of the rings. The occultation
modes are used for these (see Section 5.2.2 where these modes are defined).
3.3.1. Spectral Imaging of Rings
The purpose of ring spectral imaging with VIMS is to determine the spatial distri-
bution of various chemical species in Saturn’s rings by mapping absorption bands
in the visual and in the near infrared spectral regions. Radial mineralogical differ-
entiation of the ring may be a key issue, related to ongoing transport phenomena in
the rings, and related as well to the initial conditions through which the rings were
formed (like a satellite breakup or the remnants of the protoplanetary nebula).
    Another specific goal of VIMS is to characterize small-particle properties
through the observation of the rings at various phase angles. Micrometer-sized
particles are an important tracer of the ring dynamics because they are abundantly
released during collisions, and also because they are sensitive to non-gravitational
forces like radiation pressure or the Lorentz force. The visual and infrared channels
of VIMS have access to the very wavelengths from which Mie scattering phase
functions may be derived for these particles. Such functions yield some impor-
tant physical properties of the particles (real and imaginary refractive index, size
distribution).
    Spectral identifications will require mapping the Saturnian ring system at the
highest spectral and spatial resolution consistent with mission constraints. The best
opportunity for spectral imaging of the rings occurs at Saturn Orbit Insertion when
the spacecraft is closer to the rings than it will be for the rest of the Saturn Orbital
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   125

Tour. The present scenario calls for VIMS to be operated in line mode, using the
shortest possible integration time consistent with useful signal precision (signal-
to-noise ratio, or S/N). We note that VIMS-IR 2-D scan mirror is particularly
convenient for that purpose since it allows us to choose the region of the ring which
will be mapped, within the VIMS-IR 32 mrad field of view. During the rest of
the orbital tour, ring regions to be mapped will be selected according to the orbit
geometry, so as avoid redundancy and cover as much as possible of the ring system
in the initial phases. In later phases, when interesting compositional and dynamical
regions are discovered, they will be re-observed using a strategy specific to the
phenomena being studied.
    Some regions of dynamical interest will be mapped at the highest possible
spatial resolution. In particular, spectral signatures near sharp edges, wave features,
or narrow ringlets can provide evidence for the release of fresh material through
enhanced collision rate. The radial width of the waves features is typically 100
km, i.e. barely resolvable by the VIMS instrument. Nevertheless, mineralogical
differentiation near these regions can be detected, in particular during the close
approach of the entrance orbit. On the other hand, the typical width of sharp edges is
a few meters only. However, the streamline distortions near an edge are significant
over radial distances of several tens of km. Thus, VIMS may detect and map
mineralogical differentiation near edges also. Transient phenomena like clumps or
braids in the narrow F ring will also be mapped by VIMS.
    Depending on the precise orbital tour, VIMS observations of the rings will cover
a complete range of phase angles, allowing the size range of small particles to be
strongly constrained. The range of phase angles to be covered is 0–165◦ , the latter
limit dictated by the need to keep direct sunlight off the VIMS focal plane.
    Long exposures will also be made (consistent with the ∼1.2 s limit on the
integration time in VIMS-IR) to map the structure and composition of faint rings
like E ring, whose optical depth lies in the range 10−4 –10−5 . In addition, the
comparison of the spectrum of Saturn’s E ring to that of Enceladus will help to
determine if Enceladus is indeed the source of the E-ring material as is presently
thought.

3.3.2. Stellar Occultations by Rings
The purpose of ring-stellar-occultation observations with VIMS is to obtain
high-S/N optical depth profiles of the rings, at a variety of longitudes, ring
incidence angles, wavelengths and times during the Cassini orbital tour. The radial
resolution is limited by the VIMS sampling time (13–100 ms, corresponding to
100 m–1 km typically), by Fresnel diffraction (typically 40 m, at 3-µm wavelength
and a distance of 5 R S ), and by the diameter subtended by the stellar disk at
the rings (10–100 m). The S/N achieved will depend on the sampling time, the
stellar brightness, and the background flux of sunlight reflected by the rings into
the VIMS aperture. Simultaneous observations at two or more wavelengths will
permit this background signal to be subtracted from the stellar signal, once the
```

<!-- PDF_PAGE: 16 -->

## PDF page 16

```text
126                                R.H. BROWN ET AL.

appropriate calibration data are obtained. Typical occultation durations will be 30
min–5 h, with 2 h a good average for a complete ring occultation near pericenter.
A similar experiment may be done using the Sun as a source using the VIMS-IR
solar port, but in this case the primary purpose is to quantify the ring extinction as
a function of wavelength between 1 and 5 µm, at 300–1000 km spatial resolution.
    During stellar occultations, VIMS will also obtain information on the ring parti-
cle size distribution by observing the ’aureole’ of forward-scattered light surround-
ing the direct stellar line-of-sight. Utilizing the full VIMS wavelength range of
0.3–5 µm, and obtaining 32 × 32 mrad ’images’, particle sizes between 2 µm and
3 mm can be probed.
    Because the angular size of the effective VIMS pixel (0.5×0.5 mrad) is compara-
ble to the pointing command and control errors expected for the Cassini spacecraft,
an active method is necessary in order to keep the instrument bore-sighted on the
star during a ring occultation. This will be achieved by using the VIMS-IR 2-D
scanning capability to periodically perform small (4 × 4) raster scans, identify the
location corresponding to the maximum stellar signal, and then reset the mirror po-
sition accordingly. Because portions of the rings are likely to be effectively opaque
(notably the middle B Ring) this ’auto-guiding’ function can be turned off for short
periods according to a specified schedule.
    In order to correct the ring occultation data for the presence of reflected ring
light, a spatial map of the ring reflectivity spectrum is required. For this purpose,
1-D drift scans across the rings will be conducted at a range of phase angles and
lighting conditions (lit side/unlit side, as well as various incidence angles), using
the identical data mode used to take ring occultation data. For these scans, standard
spacecraft attitude control will be sufficient. In fact, these scans may be identical
to regular 1-D spectral maps of the rings acquired for other scientific purposes.
    Full IR spectral observations of the occultation stars prior to the actual occulta-
tion observations will be necessary for calibration purposes, and for final selection
of suitable occultation candidates. Some stellar observations must therefore be done
during the interplanetary cruise phase, prior to SOI.
    VIMS packets containing occultation data also contain absolute timing signals
derived from the ultra-stable oscillator (USO) in the radio science subsystem. The
exact relation between the photon-arrival time of the first data point in the packet
and the recorded clock signal will be known to ±1 VIMS sample, or 13 ms. The
absolute precision of the timing signals will be maintained to the same level, or
better, and the VIMS sampling rate will be controlled and known to an accuracy
such that the maximum cumulative timing error over one packet is appreciably
less than 13 ms. The overall goal is that the photon arrival time for any specified
occultation datum can be reconstructed to ± 13 ms, after the data is received on the
ground.
    Ring occultations can only be carried out on inclined orbits, and are thus con-
centrated within certain limited periods of the baseline orbital tour. Only the ingress
of the star behind the rings is observed, so as to permit acquisition of the star via
```

<!-- PDF_PAGE: 17 -->

## PDF page 17

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   127

the star tracker prior to the occultation. Similar opportunities exist on most inclined
(i > 10◦ ) orbits, and a great many more on the near-polar orbits in the last year
of the baseline tour. A total of 50–100 stellar occultations by the rings may be
observable during the nominal mission.


                            4. Instrument Description

This section presents a broad overview of the technical aspects of the VIMS in-
strument. For a detailed technical description, the reader is referred to the literature
(Miller et al., 1996; Reininger et al., 1994) and Cassini project documents that
describe the technical aspects of the VIMS instrument in detail. For a complete
discussion of the operational aspects of the Cassini VIMS the reader is directed to
the VIMS Users Manual (JPL document D-14200).

4.1. VIMS-VIS THE V ISUAL C HANNEL

VIMS-VIS is a multispectral imager covering the spectral range from 0.30–1.05
µm. VIMS-VIS is equipped with a frame transfer CCD matrix detector on which
spatial and spectral information is simultaneously stored. The CCD is passively
cooled in the range (−20 to +40 ◦ C. Radiation collected from VIMS-VIS tele-
scope is focused onto the spectrometer slit. The slit image is spectrally dispersed
by a diffraction grating and then imaged on the CCD: thus, on each CCD column
a monochromatic image of the slit is recorded. On-chip summing of pixels al-
lows implementation of a large number of operating modes for different observing
conditions.
   The maximum capabilities of VIMS-VIS are a spectral resolution of 1.46 nm
and a spatial resolution of 167 µrad while in the high-resolution mode of operation.
On-chip summing of five spectral × three spatial pixels enables the normal mode
of operation whereby VIMS-VIS achieves a spectral resolution of 7.3 nm and a
spatial resolution of 500 µrad. The total field of view of the instrument is 2.4◦ ×
2.4◦ , although matching with VIMS-IR imposes the use of a 1.8◦ × 1.8◦ FOV. The
main characteristics of the instrument are listed in Table I.
   As seen in the functional block diagram (Figure 1), VIMS-VIS is composed of
two modules, the optical head and the electronic assemblies, housed in separate
boxes. The VIMS-VIS optical head, shown in Figure 2, consists of two units: a
scanning telescope and a grating spectrometer, joined at the telescope focal plane
where the spectrometer entrance slit is located. The telescope mirrors are mounted
on an optical bench that also holds the spectrometer. In fact, the optical bench is
the reference plane for the whole instrument.
   The telescope primary mirror is mounted on a scan unit that accomplishes two
specific tasks: (a) pointing and (b) scanning. The scanning capability enables the
500 µrad IFOV of the nominal mode of operation by a two-step motion of the
```

<!-- PDF_PAGE: 18 -->

## PDF page 18

```text
128                                    R.H. BROWN ET AL.

                                      TABLE I
               VIMS-VIS System Characteristics (After Reininger et al., 1994).

      Spectral coverage                           350–1050 nm

      Spectral sampling (VIMS-VIS capability)     1.46 nm
      Spectral sampling (Cassini requirements)    7.3 nm (5 pixel summing)
      IFOV                                        0.17 × 0.17 mrad
      Effective IFOV                              0.5 × 0.5 mrad (3 × 3 pixel summing)
      FOV (Cassini requirements)                  1.83◦ × 1.83◦
      FOV (VIMS-VIS capability)                   2.4◦ × 2.4◦


primary mirror during each integration time; images are produced by scanning
across the object target in the down track direction (push-broom technique). The
pointing capability is used to image selected target regions in a range about 1.8◦
around the optical axis, and to observe the Sun, through the solar port, during
in-flight radiometric calibration.
    An in-flight calibration unit is located at the entrance of the telescope. The unit
consists of: (a) two LEDs for a two-point calibration of the spectral dispersion
and (b) a solar port for direct solar imaging and hence for radiometric calibration.
Because the remote sensing pallet is body-mounted to the spacecraft, to image the
Sun the spacecraft is reoriented to form a 20◦ angle with the boresight direction of
the instrument and the instrument scan mirror needs to be moved to an angle of 4.8◦
from boresight. Under these conditions light from the Sun passes through a cutout




                          Figure 1. VIMS-VIS functional block diagram.
```

<!-- PDF_PAGE: 19 -->

## PDF page 19

```text
               CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                  129




                         Figure 2. VIMS-VIS cutaway diagram.



in the instrument baffle and then through a prism that attenuates the solar radiation
and redirects it towards the telescope primary mirror. Due to budget constraints,
the high-resolution mode was been implemented and tested, but not calibrated.
Calibration will be accomplished during the cruise phase to Saturn.


4.2. VIMS-IR THE I NFRARED C HANNEL

In Figure 3 is a wire-frame drawing of VIMS-IR showing the major optical com-
ponents along with the light paths. At the top of Figure 3 can be seen the visual
channel with its light path and foreoptics, and near the bottom of Figure 3 is the
infrared channel with its light path and foreoptics. Also shown at the right is the
cover for the VIMS-IR optics that was successfully deployed as of August 15, 1999,
some 50 h before the time of closest approach during the Cassini Earth Swingby.
VIMS-VIS and VIMS-IR are mounted to a palette that holds them in optical align-
ment and helps to thermally isolate the instrument from the Cassini spacecraft.
Thermal isolation is particularly important for the infrared channel because ther-
mal background radiation from the IR spectrometer, combined with shot noise from
leakage current in the InSb photodiodes of the 256 element linear array detector
of VIMS-IR are the chief sources of noise in measurements obtained with the IR
channel. The nominal operating temperature for the VIMS-IR focal plane is 55–60
K and for the IR foreoptics and spectrometer optics 120 K.
```

<!-- PDF_PAGE: 20 -->

## PDF page 20

```text
130                               R.H. BROWN ET AL.




                         Figure 3. VIMS IR wire-frame drawing.



    The VIMS instrument is mounted on the Cassini spacecraft by means of a palette
(called the remote sensing palette or RSP) on which are also mounted the Cassini
imaging science subsystem (ISS), the composite infrared spectrometer (CIRS),
and the ultraviolet imaging spectrometer (UVIS). Mounting of all the Cassini
remote sensing instruments on a common palette allows relatively precise boresight
alignment of the four instruments, enhancing synergy between the four instruments.
    Figures 4 and 5 are computer-aided design (CAD) models of the Cassini space-
craft with all of the major spacecraft components and their spatial relationship to
the remote sensing palette indicated. Figure 4 specifically shows the mounting po-
sition and orientation of the VIMS instrument on Cassini. Figure 6 is a picture of
the infrared channel very near the time that the VIMS instrument began its thermal
vacuum and ground calibration tests prior to integration on the Cassini spacecraft.
Also in Figure 6 are some of the nefarious characters that helped give birth to VIMS.

4.3. F UNCTIONAL D IAGRAMS

Figure 7 is a functional block diagram of the entire VIMS instrument, with VIMS-
VIS and VIMS-IR integrated on an optical pallet assembly (OPA).

4.4. P ERFORMANCE AND OPTICAL SPECIFICATIONS

4.4.1. VIMS-VIS
The optical specifications of both components of the VIMS instrument are given in
Table II. The VIMS-VIS telescope is a f/3.2 system which uses a Shafer design to
```

<!-- PDF_PAGE: 21 -->

## PDF page 21

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                 131




                Figure 4. CAD model of Cassini orbiter with VIMS shown.




             Figure 5. CAD model of Cassini with various subsystems indicated.



couple an inverted Burch telescope to an Offner relay system. The Burch telescope
consists of a concave scanning primary mirror, a planar folding mirror, and a spher-
ical convex secondary mirror. The Burch telescope produces a curved, anastigmatic
virtual image which is fed to the Offner relay whose purpose is to produce a flat,
real image on the telescope focal plane without loosing the anastigmatic quality
```

<!-- PDF_PAGE: 22 -->

## PDF page 22

```text
132                                R.H. BROWN ET AL.




                      Figure 6. VIMS-IR and some of the engineers.




                   Figure 7. Integrated VIMS functional block diagram.



of the Burch telescope (See F. Reininger et al., 1994). The telescope is effectively
diffraction limited at 0◦ scan angle over the full field, the encircled energy at 1.2◦
off-axis (maximum VIMS-VIS capability) is 97.8% for a 24 µm square (size of
the CCD pixel); moreover, the geometrical spot sizes were under 12 µm at 0◦ scan
angle over the full field.
```

<!-- PDF_PAGE: 23 -->

## PDF page 23

```text
                                                               TABLE II
                                             VIMS-VIS and VIMS-IR optical system characteristics.

                                     VIMS-VIS VIMS-IR                              Total system

Spectral coverage                    0.35–1.0 µm (0.3–1.0 µm by special command) 0.85–5.1 µm                       0.35–5.1 µm or 0.3–5.1 µm
Spectral sampling                    7.3 nm/spectal (96 bands) (1 × 5 sum)       16.6 nm/spectal (256 bands)
Spatial characteristics
Instantaneous field of view (IFOV)   0.17 × 0.17 mrad                              0.25 × 0.5 mrad
Effective IFOV                       0.5 × 0.5 mrad (3 × 3 sum)                    0.5 (0.5 mrad (1 × 2 sum)       0.5 × 0.5 mrad
Total field of view                  64 × 64 pixels (32 × 32 mrad)                 64 × 64 pixels (32 × 32 mrad)   32 × 32 mrad
Swath width                          576 IFOVs (3 × 3 × (64)                       128 IFOV (2 × 64)               32 mrad
Image size modes (fast scan)         1–64 pixels                                   1–64 pixels                     1–64 pixels
Image size modes (slow scan)         1–64 pixels                                   1–64 pixels                     1–64 pixels
                                                                                                                                               CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER




                                                                                                                                               133
```

<!-- PDF_PAGE: 24 -->

## PDF page 24

```text
134                               R.H. BROWN ET AL.

    The focal plane of the telescope is the entrance object plane of the spectrom-
eter. The spectrometer is an Offner relay matched to the telescope relay (f/(3.2)
that utilizes a holographically recorded convex spherical diffraction grating as the
secondary mirror of the Offner relay. The grooves of the grating have a rectangular
profile. In addition, two different groove depths are used in adjacent sections of
the grating to enhance the grating efficiency spectrum in the UV and IR regions.
This compensates for the spectral response function of the CCD detector (quantum
efficiency) and for the solar radiance (the peaks occur at about 675 and 475 nm,
respectively). In 67.5% of the grating surface, there is a groove depth of 300 nm
while the remaining 32.5% is covered with grooves of depth 440 nm. The spec-
trometer spot diagrams show that for a 0◦ scan angle all the energy falls inside a
CCD pixel over the full angular and spectral fields.
    The CCD detector is a 512 × 512 Loral frame-transfer, front-side-illuminated
device. The pixel size is 24 µm × 24 µm. Half of the detector is used for the
frame transfer, and the effective sensitive area corresponds to a 256 spatial × 480
spectral pixels. To minimize second- and third-order light from the grating, two
long-pass filters were deposited on a window used to seal the detector. The junction
between the two filters (located at 601 nm) induces a dead zone on the detector
of less than 40 µm (a nominal spectral pixel is 120 µm wide). To improve CCD
detector responsivity in the UV region of the spectrum a Lumigen coating has been
deposited on the CCD sensitive surface in the region 300–490 nm.
    The performance of the instrument can be evaluated by its S/N for a given
configuration of the instrument and for a given target scenario. VIMS-VIS has
been designed to guarantee a S/N > 100 under all observing conditions for the
nominal mode of operation.

4.4.2. VIMS-IR
The VIMS infrared channel optical design follows that of the Galileo NIMS in-
strument, but incorporates several improvements that result in much better per-
formance than that of its predecessor. The IR foreoptics is a 23-cm diameter
Ritchey Cretien telescope operating at f /3.5. It is equipped with a secondary mir-
ror that can be scanned in two orthogonal directions, resulting in the scanning of
64 × 64 mrad scene across a 0.2 × 0.4 mm entrance slit. The entrance slit is cou-
pled to a classical grating spectrometer using a f/3.5 Dahl-Kirkham collimator. The
spectrometer incorporates a triply-blazed grating whose blaze angles have been
adjusted so as to compensate for the steep drop in intensity of the solar spectrum
toward long wavelengths, thus resulting in a more uniform signal-to-noise ratio
across the 0.85–5.0 µm wavelength region when viewing spectrally neutral objects
in sunlight. The dispersed light from the grating is imaged onto a 1 × 256 array
of InSb detectors using a f /1.8, all reflective, flat-field camera. The IR detector
employs four filters, used both for order sorting and for blocking excess thermal
radiation from the spectrometer optics. The secondary mirror of the VIMS fore-
optics is articulated using a two-dimensional, voice-coil actuator capable of 0.25
```

<!-- PDF_PAGE: 25 -->

## PDF page 25

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                  135

mrad steps in the fast-scan direction and 0.5 mrad steps in the slow-scan direction.
Because the optical performance and design of NIMS have been described in detail
elsewhere (Aptaker, 1982; Macenka, 1982), we direct the interested reader to those
discussions. Below we will focus in more detail on the optical improvements to the
NIMS design.
    The major areas in which the VIMS design improves on the NIMS design are:
a two-axis, voice coil actuator for articulation of the telescope secondary mirror;
a fixed, three-blaze grating; an improved shutter to replace the NIMS focal-plane
chopper; on-board solar and spectral calibration; and an improved IR detector.
    The two-axis scan mechanism employs four voice coil actuators for articulation
and two linear variable displacement transformers for position sensing. The sec-
ondary mirror is mounted on a monolithic gimble ring supported by four flexures.
The flexure life is predicted to be greater than 20 million cycles. The range of mo-
tion is 128 steps in the fast-scan direction and 64 steps in the slow-scan direction,
both covering approximately 64 mrad in angular displacement. The step transition
time is a maximum of 5 ms.
    The grating is blazed in three separate zones designed to more evenly dis-
tribute the performance of the spectrometer over the large 0.85–5.1 µm spec-
tral region covered by the VIMS instrument. The groove spacing is 27.661/mm,
and the first, second and third blazes cover 20, 40 and 40%, respectively of the
area of the grating. The three zones are blazed at wavelengths of 1.3, 3.25 and
4.25 µm, respectively.
    The shutter mechanism of VIMS is a blade located just in front of the spectrom-
eter slit. The shutter can be commanded to block the light from the foreoptics, thus
allowing a specific measurement of the thermal background radiation and detector
dark current that collects during a given integration period. In normal operation
this is accomplished at the end of every fast scan, and the result is both subtracted
from the open-shutter measurements, and downlinked with the data. A LED-photo-
sensor pair is employed to detect blade position. Lifetime tests indicate a minimum
life of 7 × 106 cycles at the normal operating temperatures of 120–140 K.
    The IR focal plane assembly is a linear array of 256 InSb photo detectors read
by two FET multiplexers. Each detector is 200 × 103 µm in size and positioned on
123 µm centers. Each multiplexer handles 128 detectors, and the two multiplex-
ers are interdigitated; thus, each reads every other detector along the array. This
arrangement was chosen to prevent the loss of half of the spectral range of the in-
strument if one of the multiplexers fails. The detector is housed in a Kovar package,
specially constructed to mimic the dimensions of the NIMS detector, and to allow
the employment of most of the original design of the NIMS passive cooler. To keep
the detector dark current within the range necessary for observations at Saturn, the
detector must be cooled to 60 K or less, and becomes non-operational for science
observations at 77 K. Mounted in the Kovar package and over the detector array are
four order-sorting filters that also reduce the amount of stray thermal radiation from
the spectrometer optics incident on the detectors. The filters are arranged in four
```

<!-- PDF_PAGE: 26 -->

## PDF page 26

```text
136                                R.H. BROWN ET AL.

contiguous segments, the lengths of which were carefully chosen so as to provide
efficient order sorting and to place the slight gaps between the filters in places that
would not have an adverse impact on science observations in the Saturn system.
For more details on the filters (or on any of the above topics) the reader is referred
to Miller et al. (1996).
    The final two enhancements to the NIMS design are the addition of a port
through which the Sun can be observed and a laser diode that is used for onboard
wavelength calibration. Detailed descriptions of the solar port and the on-board
spectral-calibration capability of VIMS appear later in this document.

4.5. DATA H ANDLING, COMPRESSION, AND BUFFERING

Reversible data compression for the VIMS instrument is performed by a dedi-
cated module based on a digital signal processor working at 6 MHz that inter-
faces with the VIMS command and data processing unit through an input first-
in-first-out buffer and an output first-in-first-out buffer. The data compression
algorithm consists of a preprocessing routine and a Rice entropy encoder (Rice
et al., 1991). The data unit for compression is a series of 64 spectra (a slice) ob-
tained during one scan (i.e, in nominal 64 × 64 image mode or 64 × 1 line mode)
or a series of scans. This provides 64 pixels each with 352 spectels. Each slice is
divided into 11 sub-slices of 32 spectels (three for the visual channel and eight for
the IR channel), and each sub-slice is compressed independently. This restricts the
impact of possible bit errors either in random access memory (single event upsets)
or during transmission.
    The preprocessing routine for the 64 pixel × 32 spectel sub-slice is based on
the large entropy content of variations in lighting and albedo when compared to
spectral signatures (more than 5 to 1). Four evenly distributed channels are summed
and divided by four so as to obtain a representative “brightness line” for this spectral
range. This line and the brightest spectrum are differenced, then Rice encoded into a
header. The matrix product of the brightest spectrum by the brightness line provides
a model rectangle where every pixel has the same brightness as the actual data and
the same spectrum as the brightest pixel. The model rectangle is then subtracted from
the actual data. The entropy of the resulting “spectral rectangle” is directly related
to the variations in relative spectral characteristics. It is then differenced along the
spatial direction and the 63 × 32 differences are Rice encoded. Reconstitution
on the ground first recovers the brightness line and brightest spectrum from the
header, then the rectangle of 63 × 32 differences, and finally the original data
taking advantage of the fact that model and data are identical for the brightest
spectrum. The compression ratio for fully reversible compression ranges between
2 and 3 depending on the actual entropy of the data.
    Reversible compression is highly sensitive to over sampling of the noise.
Each doubling of the data number (DN) level of the noise adds 1 bit per
data element to the compressed string. After calibration, the shot noise can
```

<!-- PDF_PAGE: 27 -->

## PDF page 27

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   137

be accurately predicted from the data itself. The algorithm provides the pos-
sibility of defining two DN limits, Lim2 and Lim4, above which the total
noise is expected to exceed 2 (4) DN. Each datum in the spectral rectangle
is divided by 2 if the model value exceeds Lim1 (Lim2) (4). The quantiza-
tion noise after reconstitution is then 2 × 0.28 DN (4 × 0.28 DN) instead of
0.28 DN hence the compression algorithm is able to automatically divide the actual
gain by 2 (4) for DN levels associated with large shot noise. Setting Lim1 and Lim2
to 4096 (larger than any 12 bit data) provides fully reversible compression.
   A data buffer of 4 Mb is used for buffering data between the VIMS processor
and the spacecraft. Its full memory is divided into two redundant pages of 2 Mb,
each consisting in 32 sectors organized as a matrix of four lines by eight columns.
Each sector contains two 32K × 8 RAM chips, connected in parallel to achieve a
16-bit architecture. Each sector can be specifically selected. The transfer is made
via a direct memory access (DMA), with write and read maximum times of 240
and 140 ns, respectively. Each column includes an “anti-latchup” capability and
over-current protection, which switches off the power autonomously in case of
over-threshold current (100 mA in 11 µs or 200 mA in 4.4 µs).
   Dassault Electronique of France developed the VIMS data buffer. It was qualified
for radiation levels up to 20 krad, storage temperatures of (−55 ◦ C, +125 ◦ C, and
operating (full functional) temperatures of (−40 ◦ C, +80 ◦ C. It consists of a single
board, with one page on each side, 162 × 129 × 17 mm3 in size, and 625 g in mass.
With a 5 ± 0.25 V supply voltage, its power consumption is 1.5 W in stand-by
mode, and 2 W during operations.

                             5. Instrument Operation

5.1. GENERAL O PERATIONAL C ONSIDERATIONS

5.1.1. VIMS-VIS
VIMS-VIS operates in different operating states and in different modes. The CCD
data acquisition mechanism combines pixels by summing them on chip, both in
the spatial direction (along slit) and in spectral direction. The use of a scanning
secondary mirror allows summation of pixels parallel to the slit. Thus square spatial
pixels are generated in lower spatial resolution modes. There are five operating
states of the instrument as a whole: science status, internal calibration, uploading,
downloading, and off. The scientific data and the internal calibration data are in turn
defined by different operating parameters that translate the scientific requirements
for the data collection into the physical, electrical and optical characteristics of the
instrument.
    When VIMS-VIS is switched on it goes into the science state in the nominal
mode. The commands to change the operating state are sent to the VIMS-VIS
electronics together with the set of parameters relevant to define a new state. The
scientific data set can be obtained in different “operative modes” according to the
```

<!-- PDF_PAGE: 28 -->

## PDF page 28

```text
138                               R.H. BROWN ET AL.

different observation scenarios. For each mode the operating parameters define
the characteristics of the data to be collected, such as spatial resolution, spectral
resolution, and extent and location of the scene to be observed in the instrument
FOVs. When the instrument is switched on, the combination of different parameters
that define the different observation modes are loaded. The input parameters of the
scientific data acquisition modes are:
(a) Summing parameters,
(b) Swath width,
(c) Swath offset,
(d) Spectral offset,
(e) Swath length,
 (f) Exposure time,
(g) Exposure time delay,
(h) Mirror,
 (i) Gain,
 (j) Anti-blooming.
    Different combinations of these parameters lead to several different operative
modes.

5.1.2. VIMS-IR
The two main considerations in the design of the IR component of VIMS are
flexibility and compatibility with the other Cassini remote sensing instruments. A
range of programmable parameters, including integration time, swath width and
length, and mirror position enable the instrument to adapt to a variety of observing
modes dictated by the scientific objectives of each phase of the Cassini mission, and
by synergistic observations with other Cassini instruments. Software enhancements
uploaded after launch provide more levels of flexibility, including spectral editing,
additional image sizes, increased spectral resolution in the visual channel, and a
doubling of the instrument’s spatial resolution.

5.2. S PECIFIC O PERATIONAL MODES

Image, line, point, and occultation are instrumental modes of operation. An oper-
ational mode corresponds to specific instrumental parameter settings, as was illus-
trated in Section 5.1.1. The image, line, and point modes are the most frequently
used. Observation types (Sections 3.1.1. and 3.2.1) are use oriented variations on
single modes.
5.2.1. Image, line and point mode descriptions
The majority of scientific observations by VIMS will be in the image, line, or point
modes. Several modes were tested for a variety of integration times and instrument
gain states. For image sizes smaller than 64 × 64, the scan mirror can be offset from
the boresight in both the x and z directions. In the image mode, VIMS is a framing
```

<!-- PDF_PAGE: 29 -->

## PDF page 29

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                    139

instrument, producing a spectral cube of 352 images in the specified size. Three rou-
tinely used sizes for the images are a full 64 × 64 IFOVs, 12 × 12 IFOVs and a single
(1 × 1) IFOV. The 12 × 12 image size corresponds to the field of view of the im-
age science subsystem’s narrow angle camera (ISS NAC). This mode will be used
to efficiently gather data simultaneously with ISS. The 1 × 1 image size corre-
sponds to point mode, which will be used primarily for stellar occultations or for
the accumulation of spectra during a rapid flyby of a target for which motion com-
pensation is not possible. Rectangular combinations of 12, 32, and 64 are also in
the group of tested modes: these will be used during periods when the data rates or
volume is constrained. It is possible to offset the position of the scan mirror so that
“sub images” are gathered in a position offset from the center of the field of view.
Compositional mapping and photometric studies of Saturn, the surfaces of the icy
satellites, Titan, and the rings, and dynamical studies of Saturn and Titan will be
performed primarily in the image modes.
    The VIMS instrument was also designed to obtain data in single lines of either
12 or 64 IFOVs parallel to either the spacecraft’s x- or z-direction. The line mode
parallel to the z-direction will be used in synergistic studies such as atmospheric
profiles of Saturn and Titan with CIRS and UVIS, both of which have slits parallel
to this axis. Line mode can also be operated such that the spacecraft’s motion scans
a second spatial dimension. These modes of operation are useful, for example, in a
rapid flyby where motion compensation is not possible.

5.2.2. Occultation Modes
VIMS is the only instrument on the Cassini payload capable of observing stellar
occultations in the visual and near IR portion of the spectrum. Stellar occultations by
the atmospheres of Titan and Saturn will be used to derive the vertical distribution
and microphysical properties of aerosols, and the stratospheric temperature profiles
of these two bodies. Stellar occultations of the rings of Saturn will provide a sensitive
probe of their radial structure, optical depth, composition and physical properties.
    Occultations will be executed in the point mode. Before each occultation a 12
× 12 image of the star will be obtained. On-board software will seek the pixel
with the maximum brightness. The instrument’s mirror will be offset so that data is
gathered in this pixel. If necessary, additional images can be gathered during long
occultations to check that the star is still centered in the selected pixel. If the star
has drifted, the mirror can be repositioned so that data are gathered from the pixel
with the maximum brightness.
    Three different modes of ring occultation observation are currently envisaged,
with different data rates and spectral coverages:
    Occultation mode 1: In this mode, which is applicable to stellar occultations
by Saturn, Titan, and the rings, complete IR spectra are recorded, compressed, and
either stored in the VIMS internal buffer or in the spacecraft on-board solid state
recorder (SSR). The scan mirror oscillates so as to synthesize an effective 0.5-mrad
square pixel for each spectrum. The sampling rate is adjusted, up to a maximum of
```

<!-- PDF_PAGE: 30 -->

## PDF page 30

```text
140                                R.H. BROWN ET AL.

∼10 Hz (one complete spectrum every 0.1 s). The maximum data rate (2:1 com-
pressed) is 0.5 × (256 channels × 12 bit (10 Hz) = 15.4 kb/s. Typical data volumes
are 5 Mb for a 300-s atmospheric occultation, and 110 Mb for a 2-h ring occultation.
A modification of this mode, employed for the aureole observations, is to interrupt
the regular occultation data stream with a full 64 × 64 pixel spatial/spectral map
of the starlight scattered by the rings into the instrument FOV. Such a map would
require 8.6 Mb of data, compressed, and take ∼450 s at 100 ms sampling. Since
each pixel subtends at least 150 km on the rings at 5 R S , the full image covers a
patch about 9600 km square, about twice the distance the star travels across the
rings in 450 s, and comparable to the largest homogeneous regions in the A, B and
C rings.
    Occultation Mode O2: In this mode, to be used primarily for ring occultations
(but possibly also for icy satellite occultations), edited spectral averages are
returned at the maximum VIMS sample rate of 77 Hz. Using spectral editing
mode, a small number of spectral editing masks will be chosen, but a typical mask
would require the co-addition of 15–30 contiguous spectral channels centered
at four different wavelengths. A typical example might be 30 co-added channels
centered at 2.00, 2.25, 2.90, and 3.50 µm. As in mode O1, the internal mirror
oscillates every 6.5 ms to synthesize a 0.5 mrad square pixel. The four co-added
signals are stored or transmitted, uncompressed, every 13 ms, for a data rate of
four channels × 12 bit × 77 Hz = 3.7 kb/s, plus timing and housekeeping data.
Total data volume is 27 Mb for a 2-h occultation.
    Occultation Mode O3: Solar occultations. During a solar occultation experiment
VIMS stares at the Sun through its solar calibration port, monitoring the extinction
of sunlight by Saturn’s rings, Saturn’s atmosphere or Titan’s atmosphere. Because
the Sun has a large angular diameter even at Saturn’s distance, (0.9 mrad at 10 AU),
the maximum spatial resolution under these conditions is 300 km, and the necessary
sampling time about 15 s or longer. Typical observations of the rings, for example,
would result in full IR spectra every 15 s, for one spatial pixel, yielding a data rate
(compressed) of 102 bps and a data volume of 740 kb for a 2-h occultation.


5.2.3. Solar Port
The solar port of VIMS was added late in the instrument design stage to recover
some of the capability lost by the deletion of the Cassini spacecraft’s calibration
target. As a bonus, the solar port is capable of performing observations of solar
occultations for atmospheric and ring studies.
    The solar port strongly attenuates the solar spectrum that, once it passes through
the solar port itself, follows the same path through the instrument as photons incident
on the boresight. Its chief purpose is to acquire a solar spectrum that can be used
to remove the effects of solar illumination from the VIMS data to obtain accurate
scales of relative and absolute reflectance. The port’s boresight is offset from VIMS
main optical axis by 20 ◦ , to coincide with the optical axis of the UVIS instrument.
```

<!-- PDF_PAGE: 31 -->

## PDF page 31

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                     141

This configuration facilitates compatible operations between the two instruments
for solar occultations of the atmospheres of Titan, Saturn, and of the rings.
    The large attenuation factor is achieved by the small aperture of the solar port in
comparison to the main beam, and by a series of one 70 ◦ and 90 ◦ reflections from
right angle prisms made of ZnSe. Most of the incident flux is reflected back out of
the entrance aperture by internal reflection in the prisms. The beam exiting the cal
port is first focused by the primary and secondary telescope mirrors onto the VIMS-
IR entrance slit, and then it enters the spectrometer. The incident solar radiation will
illuminate only a small portion of the diffraction grating, because the cal port aper-
ture samples only a portion of the main instrument’s aperture. The optical design
ensures that this region overlaps the short and medium-wavelength blaze regions
on the grating, in the ratio of 1:3. The long-wavelength blaze is not illuminated. The
predicted attenuation factor is 1.35 × 10−4 at 0.85 µm, 1.09 × 10−4 at 3.0 µm, and
1.04 × 10−4 at 5.0 µm. The actual attenuation factor in flight, measured by observ-
ing the Earth’s moon in both the occultation port and the main aperture, is about
2.5 × 10−6 .

5.2.4. IR and VIS Timing and Coordination
Although the infrared and visual channels have separate optics and detectors, the
two data streams are combined in the instrument’s electronics so that a single
spectrum of the same physical area on the target is produced. The visual channel
utilizes an area array so it is able to gather an entire line of data during a single
integration. The same line synch pulse that initiates the IR scan synchronizes the
acquisition of the visual line. After the first visual integration period is complete, the
entire frame of data is transferred and stored within the CCD for initial processing.
The line of visual data is then sequentially read out and digitized by the visual
channel electronics. Three lines of visual data are accumulated during the time of
the forward scan of the IR channel. These individual visual pixel elements are then
summed three by five (both spatially and spectrally) to yield a line of square pixel
elements of ∼0.5 mrad and with spectral channel widths of ∼7 nm. This line of
visual data is transferred to the instrument’s main electronics over the global bus in
exactly the same fashion as the IR data. The integrated visual and infrared spectra
are then stored in the control processor’s RAM.

             6. Instrument Calibration and Cruise Data Analysis

6.1. S YNOPSIS OF THE G ROUND CALIBRATION

Spectral calibration of VIMS occurred in three separate steps. As above, we denote
the visible spectral region component (channel) of VIMS as VIMS-VIS, and the
infrared component (channel) as VIMS-IR. VIMS-VIS was calibrated separately
in Italy prior to integration with VIMS-IR at JPL. VIMS-IR was calibrated in
the thermal vacuum tests at JPL January 28–February 5, 1996. The integrated
```

<!-- PDF_PAGE: 32 -->

## PDF page 32

```text
142                                R.H. BROWN ET AL.

instrument was further tested at the JPL thermal vacuum facility July 16–17, 1996.
This final test only included filter transmission and mineral target tests.
    The spectral calibration resulted in a specification of the wavelength position
and bandpass of each VIMS spectral channel across the instrument field of view
and as a function of temperature. The VIMS-IR spectral response is identical across
the full field of view to within about a nanometer (nm) over the temperature range
tested. The VIMS-IR sampling interval is about 16 nm in the IR, thus a 1-nm shift is
a small fraction (<7%) of the sampling interval. VIMS-VIS has a sampling interval
of ∼7 nm and tests show stability to better than about 0.3 nm.
    The detailed plan for the VIMS ground calibration evolved over a period of
approximately four years, being completed in mid-1995, roughly six months be-
fore the actual measurements were to commence. The time frame for the ground
calibration was driven by the planned delivery date of the fully integrated VIMS
instrument in the September–October period, 1996. As mentioned above, because
VIMS-VIS was not completed at the time of the main calibration, only VIMS-IR
was calibrated. As a result of the slip in the schedule for he delivery of VIMS-
VIS, a substantial recalibration of the integrated instrument has been undertaken
while Cassini is in route to Saturn. The details of the efforts to calibrate the VIMS
instrument in flight appear later in this document.
    The main ground calibration of the IR channel was carried out in six separate
areas: radiometric/flat field response, geometric, polarimetric, spectral, and solar
port response. In the early phases of the genesis of the VIMS ground calibration
plan, measurement of VIMS stray light rejection performance was also envisioned,
but practical difficulties in performing those measurements under vacuum and at
the operational temperatures required necessitated elimination of the groundbased
measurements in favor of measurements in flight. Those measurements are dis-
cussed later in this document.
    The actual measurements were carried out in the JPL thermal vacuum testing
facility in the largest thermal vacuum tank available. The in-flight thermal envi-
ronment for VIMS was simulated by cooling the interior surfaces of the thermal
vacuum tank with liquid N2 , and providing a cold target that filled the fields of view
of the passive coolers of both VIMS-VIS and VIMS-IR with a high-emissivity
surface cooled to 4 K. In retrospect, the thermal environment of the JPL thermal
vacuum tank and the cold target was quite accurate because the temperatures of the
instruments optics and focal planes in flight are within a few K of those measured
while the instrument was in the test environment.

6.2. RADIOMETRIC CALIBRATION

6.2.1. VIMS-IR
The radiometric response of VIMS-IR was carried out before launch in the ther-
mal vacuum facility at the Jet Propulsion Laboratory. A team of scientists, sup-
ported by the instrument engineering team at JPL, designed and carried out these
```

<!-- PDF_PAGE: 33 -->

## PDF page 33

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                     143

measurements. The team members responsible are Thomas B. McCord (lead),
Robert Brown, Angioletta Coradini, Vittorio Formisano, and Ralf Jaumann. Also
contributing were Giancarlo Bellucci, Bonnie Buratti, Frank Trauthan, Charles Hi-
bbitts and Gary Hansen. Two sessions were conducted, one in January and the other
in July of 1996. In-flight calibration efforts were conducted during the Venus, Earth
and Jupiter flybys, and for two star observations. A workshop was held in Hawaii in
February 2001 to review the information obtained. Additional people contributing
post launch were Kevin Baines, Roger Clark, and Robert Nelson.
    The equipment facility during the ground calibration included a JPL thermal
vacuum chamber cooled by liquid nitrogen and containing the instrument, which
viewed the outside through a window with known optical transmission. A reflecting
collimator fed light from several sources to the instrument. The calibrated sources
were a glow bar and a tungsten lamp; their energy delivered to the instrument
was controlled by adjustable iris diaphragms at the exit of the lamps. The light
sources and delivery system were covered to eliminate outside light and, during
the first session, the tent was purged with dry nitrogen to reduce the effects of the
atmosphere gas absorptions (mainly CO2 and H2O). Measurements were made at
several instrument and focal plane temperatures, but most measurements were made
with the focal plane temperature in the range 60.7–61.69 K. Data were acquired at
several light levels and integration times, including zero.
    The characteristics of the instrument that were explored were dark current (de-
tector thermal carrier generation and electronic off-sets), background signal (mostly
thermal radiation from the chamber window and from outside), linearity of response,
ratio of responses at the two gain states, performance of the detector for two dif-
ferent bias levels, and overall radiometric response over the spectral range. The
instrument was determined to be linear within the measurement error to detector
saturation. The dark current, background, gain ratios and behavior at different bias
are reasonable, stable and as expected. The overall spectral responsivity was most
difficult to calibrate due to several factors, including an unexpected and unknown
(at the time) change in a light source during the calibration effort. This effort has
been enhanced during in-flight calibration efforts and the responsivity calibration
is converging.
    In-flight calibrations were conducted at Venus only for the visual channel be-
cause the cooler cover was not yet removed from the IR channel radiator. The entire
instrument calibration was tested at the Earth–Moon (for the Moon only) and the
Jupiter fly-bys and for two star observations. The data are still being analyzed at this
writing for the in-flight calibrations, but the general result so far is to further refine
the calibrations and to gain better understanding of the instrument performance,
which remains as expected.
    One interesting characteristic experienced is the difficulty of achieving precise
and stable calibration for sources smaller than the spectrometer slit width (sub-pixel
sources). This is because the effective spectral resolution of the spectrometer and
the exact location of the source image on the detectors depend on the size of the
```

<!-- PDF_PAGE: 34 -->

## PDF page 34

```text
144                                     R.H. BROWN ET AL.




Figure 8. Spectro-radiometric response of the VIMS-IR channel as a function of wavelength in
photons per data number. This plot represents the status of the calibration as of the writing of this
article. Improvements and changes are expected with time and further analysis.


source and its exact location in the focal plane. Thus, it will be difficult to precisely
predict radiometric performance for sub-pixel sources. Nevertheless, for sources
that fill the slit, the instrument behavior is normal and as expected.
    The spectro-radiometric response function as currently known is given in Figure
8. Improvements and changes are expected with time and more analysis. Thus, the
reader is referred to the VIMS planetary data system (PDS) archive for appropriate
calibrations to be used with flight data.

6.2.2. VIMS-VIS
VIMS-VIS was constructed and calibrated in Italy, at Officine Galileo. It was cali-
brated in two phases: (1) at Officine Galileo prior to the integration with VIMS-IR,
and (2) at JPL after the integration on the remote sensing palette. The activity
carried out at JPL was mainly devoted to geometric measurements, that is, co-
alignment of the two channels and measurement of the relative radiometric response.
```

<!-- PDF_PAGE: 35 -->

## PDF page 35

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   145

Furthermore, the instrument spatial response (measurement of the image quality
through the instrument modulation transfer function and point spread function) was
evaluated as part of the full functional tests performed at Officine Galileo prior to
the calibration activity.
    VIMS-VIS was placed inside a vacuum chamber equipped with a thermally
stabilized radiator connected to the CCD and capable of keeping the CCD at a
temperature in the range −20 to 40 ◦ C under a residual pressure <10−4 mbar. The
chamber has a window made of TVC, with transparency better than 0.9 throughout
the entire spectral range. VIMS-VIS was mounted on two computer-controlled
rotating tables for fine positioning at a range of azimuthal and elevation angles.
Two lamps were used to cover the full spectral range: a xenon lamp for the range
0.3–0.4 µm, and a tungsten lamp between 0.4 and 1.035 µm. The lamp with
its housing, which includes a condenser and a diffusing screen to improve light
uniformity, was positioned at the input slit of a Jobin-Yvon HR640 monochromator
capable of better than 0.05-nm resolution (band width at half height) over the VIMS-
VIS spectral range. The monochromator output was then used to illuminate a slit,
pinhole or test targets (corresponding to a specific type of test) placed at the focus
of an off-axis collimator. The collimated beam was fed to the instrument inside
the vacuum chamber. Unfortunately, the collimated beam had an unknown spectral
irradiance; thus, we had to devise a method to measure it. This was achieved using
a beam splitter of known optical properties, placed in the optical path at 45 ◦ in
front of the chamber window. The reflected portion of the beam was collected by
a calibrated photodiode to monitor the irradiance output of the light source. An
additional calibrated photodiode was placed every 50 nm (or 50 monochromator
steps) directly in front of the collimator to have direct calibration at the collimator
aperture.
    Only one-sixth of the full VIMS-VIS FOV could be instantaneously illuminated
with the available collimator, thus a time consuming procedure was implemented
to repeat a full spectral sweep (0.3–1.05 µm in 1 nm steps) to cover the field of
view of the instrument.
    The radiometric calibration of VIMS-VIS was as follows. The unit response
(UR) of the instrument is defined as the output in data numbers (DN) when the
instrument entrance pupil is fed with a light beam of 1 W cm−2 , and this beam is
collected entirely into a single spectel and into a unit solid angle, for an integration
time of 1 s. This quantity was directly measured along with its dependence on wave-
length. The spectral calibration was performed in order to evaluate the spectrally
weighted center of each channel as well as the spectral width. The spectral width of
each spectel is 7.33 nm. For a detailed discussion on the calibration see Capaccioni
et al. (1998).
    We note that the radiometric transfer function obtained at Officine Galileo during
on ground calibration of VIMS-VIS when applied to the Moon and Venus illumi-
nated side is insufficient to remove instrumental effects. Moreover two problems
are apparent: a shift in wavelengths of about two nominal pixels and an inadequate
```

<!-- PDF_PAGE: 36 -->

## PDF page 36

```text
146                                     R.H. BROWN ET AL.




Figure 9. The VIMS-VIS field of view on the Moon during the flyby of August 18, 1999. Key: 1 Mare
Crisium, 2 Mare Fecunditatis, 3 Mare Tranquillitatis, 4 Mare Serenitatis, 5 Langrenus, 6 Petavius, A
Apollo 16 landing site.



removal of instrumental effects, particularly at short wavelengths. So we measured
a new unit response function using the Venus and Moon data in the following way.
   We use a telescopic reflectance spectrum (McCord and Adams, 1973) of the
Apollo 16 landing site, which is a bright area in the highlands of the Moon’s
surface where the albedo is particularly high.
   At this point on the Moon the signal is given by:

    FMoon (x ∗ , λ) = RMoon (x ∗ , λ) × SMoon (λ) · UR(x ∗ , λ),                                (1)

where RMoon is the lunar spectral reflectance computed at a point x ∗ in the lunar
highlands (see Figure 9), SMoon is the solar radiance at the mean Moon–Sun distance
(1 A.U.) and UR is the unknown transfer function at the point x ∗ . Accordingly, the
Unit Response transfer function is:
                       FMoon (x ∗ , λ)
   UR(x ∗ , λ) =                                                                                (2)
                    RMoon (λ) × SMoon (λ)

    This is the transfer function for the point on the Moon x∗ located in the slit
center.
    In order to obtain a target-independent UR function we have to consider a flat
field. Assuming the uniformity of Venus’ dayside atmosphere we define a flat field
using the ratio between the signal FVenus (x, λ) at some point x in the frame and
FVenus (x ∗ = 32, λ) in the center of the slit.
                             FVenus (x, λ)
   [FVenus (x, λ)]Norm =                                                                        (3)
                             FVenus (x ∗ , λ)
```

<!-- PDF_PAGE: 37 -->

## PDF page 37

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                147

   The signal on Venus is given by an expression analogous to (1) for the Moon:
     FVenus (x, λ) = UR(x, λ) × RVenus (λ) × SVenus (λ)                          (4)
   FVenus (x ∗ , λ) = UR(x ∗ , λ) × RVenus (λ) × SVenus (λ)                      (5)
where SVenus is the solar radiance at the mean Venus-Sun distance (0.73 A.U.).
Putting (4) and (5) into Equation (3) we obtain the normalized signal on Venus:
                                      
                           UR(x, λ)
   [FVenus (x, λ)]Norm =                                                      (6)
                           UR(x ∗ , λ)
   Using this result it is possible to define the UR as the product of the transfer
function computed for the Moon at point x ∗ (45, 33) and the transfer function
normalized to the center x ∗ of the VIMS-VIS slit on Venus:
   UR(x, λ) = UR(x ∗ , λ)Moon × [FVenus (x, λ)]Norm                              (7)
   Finally, we obtain the general UR:
                                              
                     ∗              UR(x, λ)
   UR(x, λ) = UR(x , λ)Moon ×                                                    (8)
                                    UR(x ∗ , λ) Venus
   Figure 10 shows the final UR transfer function for all the wavelengths: note that
the UR is a function of the pixel position. The UR function permits conversion of
DN to physical units.

6.2.3. Conclusions
Additional radiometric calibration improvements are expected as the Jupiter en-
counter data are analyzed and compared with the ground calibration results (see
McCord et al., 2004). During Cassini’s Saturn tour, there will be many measure-
ments for a variety of objects that will help identify and allow removal of any
remaining artifacts in the spectral radiometric calibration. Users of VIMS data
should consult later papers by the investigation team, as well as the VIMS website,
for the most current versions of the calibration of the instrument.

6.3. GEOMETRIC C ALIBRATION

For VIMS, the primary data product is, for each resolved pixel of a given target,
the determination of both the pixel position and the full spectrum from 0.35 to 5.1
µm. To build such spectral images, for each picture element of any given 64 ×
64 frame, the viewing direction of each of its 352 contiguous spectral elements
(spectels) must be known. Thus, the prime goal of the geometric calibration of
VIMS was to determine the relative viewing directions of all 96 VIMS-VIS and
256 VIMS-IR spectels within the full VIMS field of view (FOV), in a frame to be
referenced to the Cassini spacecraft.
    VIMS-VIS and VIMS-IR form images in two distinct ways. VIMS-VIS operates
like a push-broom, acquiring an entire cross-track line simultaneously spread over
```

<!-- PDF_PAGE: 38 -->

## PDF page 38

```text
148                                R.H. BROWN ET AL.




                      Figure 10. The computed UR transfer function.



its spectral dimension (from 0.35 to 1.0 µm) along the second dimension of its
CCD detector. The second spatial dimension is acquired either using the spacecraft
drift or the scanning of the VIMS-VIS telescope secondary mirror. VIMS-IR uses
only a linear array detector, thus it acquires one pixel only spread over its spectral
dimension (from 0.85 to 5.1 µm). The cross-track spatial dimension is acquired by
scanning the telescope secondary mirror (whiskbroom mode), while the along track
dimension is acquired (as for VIMS-VIS) using either spacecraft drift or scanning
the IR telescope’s secondary mirror in the second dimension. With two distinct
scanning mechanisms, telescopes, and spectrometers, the boresight alignment and
the Instantaneous Fields Of View (IFOV) within the full FOV are not identical by
design, and, in principle, are wavelength dependent. The geometric calibration is
thus intimately coupled to the determination of these spectral registration effects.
    Due to the late delivery of VIMS-VIS, the ground geometric calibration was
performed in steps: the in-depth geometric calibration of VIMS-IR was performed
first, prior to the integration with VIMS-VIS, followed by the geometric calibration
of VIMS-VIS after its integration. After integration, a large misalignment was
observed, requiring a global mechanical realignment of both the IR and V channels
(out of the calibration chamber), followed by a few final control measurements (back
to the calibration chamber) prior to integration with the Cassini spacecraft. Some in-
flight calibrations were performed to assess the actual geometrical characteristics of
```

<!-- PDF_PAGE: 39 -->

## PDF page 39

```text
               CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                  149

both channels, when the instrument had reached its proper in-flight thermal regime,
late during the cruise towards Jupiter.
    During the ground calibration, VIMS was mounted on a fixed platform, ther-
mally controlled and under vacuum. A collimator and optical bench assembly was
constructed outside of the thermal-vacuum chamber, and viewed through a large
window. For the geometric calibration, we chose to image with VIMS, two types of
targets, both mounted on an X–Y stage to allow coverage of the entire VIMS FOV.
In the target projector plane, one VIMS pixel (0.5 mrad) corresponded to ∼1 mm.
To cover the entire VIMS FOV, the target was 64 mm in size, and the stage was
moved by steps of 0.1 mm (1/10 VIMS pixel). The first type of target consisted
of linear blades to measure potential spectral registration effects by analyzing for
each spectel the pixel response while the blade was moved across the pixel. The
other target placed in the projector focal plane consisted of an opaque metal plate
with a grid of sub-pixel sized pinholes (0.1 mm in diameter) evenly spaced, and
back-illuminated by a tungsten lamp. The complete calibration data products are
in the form of tables, one for each spectral channel, giving the viewing direction of
each VIMS pixel. Below we summarize some of the main results.
    The IFOVs of both channels were accurately measured. Averaged over all wave-
lengths within a given channel, IFOVIR = 0.495 ± 0.003 mrad, and IFOVV I S =
0.506 ± 0.003 mrad. Although the differences appear small (∼2%), they result in
a misalignment of VIMS-VIS relative to VIMS-IR of more than one pixel over the
64-pixel FOV. This imposes the need for a thorough geometric re-sampling of all
image cubes larger than 32 pixels. Prior to launch, the last geometric measurement
showed a boresight alignment between VIMS-VIS and VIMS-IR of better than 0.3
pixels at all wavelengths. Images coincide independent of wavelength to within
<0.5 pixel in frames up to 12 × 12 pixels.
    The first measurements in flight were performed using both channels to observe
the Moon during the Cassini Earth-Moon fly-by (August 18, 1999). Those showed
a boresight misalignment of VIMS-IR with respect to VIMS-VIS (X, Y) = (−1
pixels, (+2 pixels). Because the VIMS-IR was warmer than its proper operating
temperature at the time of the Earth–Moon flyby, it was concluded that at least part
of misalignment resulted from thermal gradients in the IR channel that would lessen
as the Cassini spacecraft moved farther from the Sun and the IR channel cooled.
A more recent calibration using the Pleiades cluster and the star Fomalhaut (late
March, 2001) showed boresight offsets of (X,Y) = (−1,0) pixels. The Pleiades
observation (Figure 11) demonstrates the misalignment variation within the FOV,
showing the various stars in the visual (blue pixels, averaged over the first 30 VIS
spectels) and the near IR (red pixels, averaged over the first 40 spectels).
    The ground calibration yielded the VIMS-IR FOV for most IR spectels, lead-
ing to spectral geometric registration maps. For images up to 32 × 32 pixels, all
spectral images coincide to better than 1/3 pixel. Larger discrepancies appear in
the bottom right and top left corners of the frame, where spectral misalignments
up to half a pixel are present. This is illustrated in Figure 12, scaled in mrad, for
```

<!-- PDF_PAGE: 40 -->

## PDF page 40

```text
150                                   R.H. BROWN ET AL.




Figure 11. VIMS image of the Pleiades, showing the stars in the VIMS-VIS (blue pixels) and the
VIMS-IR (red).

three IR wavelengths (spectels 103, 150 and 206, at 0.98 µm, 1.75 µm and 2.68
µm, in red, blue and green respectively), enlarging the four corners and the center
of the FOV geometric spectral responses. Finally, the spectral registration effects
measured for the IR spectels were demonstrated to be very low, as illustrated in
the figure: the very high sensitivity of the measurements allows detection of effects
at a scale smaller than 1/10 pixel. With this resolution, the larger effects we see
between contiguous spectels actually amount to 1/10 pixel, and in fact result from
atmospheric contributions during the calibration (variation of H2 O and CO2 fea-
tures). The only large-scale effect present (black curve) has a very low frequency (at
the scale of the entire spectral range), and likely results optical aberrations within
the IR spectrometer. It is both smooth and small enough to minimize the risk of
misinterpreting potential large optical contrasts in the observed scene in terms of
false spectral signatures.

6.4. S PECTRAL C ALIBRATION

6.4.1. Summary of Tests
The goals of the spectral calibration of VIMS were to measure the spectral response
of each VIMS spectral channel to determine the central wavelength and spectral
profile of each detector, and its spectral stability as a function of temperature and
spatial position within the field of view of the instrument. To achieve this, the tests
included: (1) scanning a nearly monochromatic line (using a calibrated grating
monochrometer) over the VIMS wavelength range to map the spectral profile of each
VIMS detector, (2) transmission spectra of materials with sharp absorption bands,
and (3) measuring reflectance spectra of minerals and other targets with VIMS.
   The monochrometer scans were useful for a single position in the full VIMS field
of view because the relatively large field of view of the VIMS could not be covered
```

<!-- PDF_PAGE: 41 -->

## PDF page 41

```text
                   CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                               151




Figure 12. Spectral registration maps for five positions in the VIMS field of view. The position of
each panel in the instrument focal plane can be seen from the coordinates in each square frame. The
vertical and horizontal directions (relative to the upper left) correspond to the +Z and +X axes of the
spacecraft coordinate system.



with the narrow exit slit of the monochrometer, and small shifts in wavelength were
observed in the monochrometer as a function of position along the slit. This results
because a change in direction along the slit results in a slight change in the field
position in VIMS that corresponds to an angular movement. Such changes require
a change in the light path through the monochrometer with a corresponding change
in the effective output wavelength of the monochrometer. Thus the monochrometer
tests were done on-axis only. The monochrometer was typically scanned at 1 nm
increments to map out the profile of each VIMS spectral bandpass.
```

<!-- PDF_PAGE: 42 -->

## PDF page 42

```text
152                                       R.H. BROWN ET AL.




Figure 13. Plot of the deviation (calculated versus actual) for the pointing direction (in the direction
of the boresight) as a function of wavelength (shown as spectel, or channel number). As examples, in
the direction of the boresight, the IFOV of spectel number 1 deviates by +0.006 pixel, while spectel
number 115 deviates by −0.08 pixel.


    Calibrated transmission filters consisting of a mylar sheet displaying sharp ab-
sorption bands in the 1–3.5 µm region, with broader features at longer wavelengths,
and Corning glass filters containing rare-earth elements which give sharp absorption
features in the visual and near-infrared wavelength regions, and broader absorptions
at longer wavelengths were used to cover the full VIMS field of view. Because of
their uniformity and stability, the filters were used to determine the stability of the
VIMS wavelength response as a function of spatial position, temperature, and time
(because two tests were done over a period of about six months).
    To measure the response of VIMS to real targets with spectral features, a set of
minerals and other materials were assembled into a spectral target and measured
in reflectance. Because the calibration geometry was optimized for other tests, the
setup for reflectance measurements was not ideal, so these tests were qualitative. The
light source used to illuminate the samples was set up at relatively large angles to the
normal to the surface of the highly scattering surface of the spectral target, but the ex-
act viewing geometry was difficult to control. Furthermore, only samples mounted
vertically could be measured, so it was not possible to measure the reflectance of
unconstrained, loose particulate samples. Measurements were made of rocks and
solid samples that typically had large grains, thus the absorption bands were deep
and saturated, but the spectra of these rocks are quite identifiable with VIMS.
    The spectral calibration tests showed that VIMS is a remarkable instrument,
producing spectra comparable in quality to specialized laboratory spectrometers.
```

<!-- PDF_PAGE: 43 -->

## PDF page 43

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   153

The large spectral range of the VIMS includes the region of increasing thermal
emission at room temperature λ ≥ 3µm) making the light reflected from the sample
difficult to separate from the background thermal emission. The spectral range of
VIMS is greater than the spectral output any single convenient laboratory light
source, so tests were often done multiple times with different light sources.

6.4.2. Spectral Profiles
The VIMS-IR spectral profiles are very close to Gaussian in shape, with small,
Gaussian shaped side lobes (Figures 14a and 14b). The small side lobes are possibly
caused by reflections in the order sorting filters. The side lobes are stronger on the
short wavelength side at the short-wavelength end of the IR array, and are stronger
on the long wavelength side on the long-wavelength end of the array. The side lobes
are typically 2% or less of the strength of the central profile and can probably be
ignored except for the most rigorous work. A set of spectral profiles for each spectral
channel will be maintained on the Cassini VIMS team home page. Spectral profiles
are somewhat distorted near the VIMS order-sorting filter gaps. These gaps occur
at VIMS-IR channels 45–46, 128–129, and 183–183 (1.59, 2.96, and 3.86 µm). At
or near the filter gaps, the central portion of the spectral profile can be absorbed,
leaving the main signal coming from the side lobes (Figure 15). Except for the side
lobes, no out-of band signals were detected in the monochrometer scans.

6.4.3. Cruise Spectral Calibration
During cruise, high signal-to-noise spectra were obtained on the Moon, stars,
Jupiter, and the Galilean satellites. Immediately apparent in these cruise data is
that the wavelengths of VIMS-IR shifted after launch. A best fit that shows over-
lap consistency between VIMS-VIS and VIMS-IR using deep absorption bands in
the spectra of Jupiter shows a 1.3 channel (∼21 nm) shift in the IR wavelengths
(Figure 16). The positions of absorption features in sub-pixel images of stars and
the Galilean Satellites showed variability by up to 1/3 channel depending upon
where the object was located in the slit. This adds difficulty in determining the
wavelength calibration post launch to better than about 5 nm. Despite this diffi-
culty, comparison with the Galileo NIMS data on the Galilean satellites confirms
the 1.3 channel shift at the positions of the 4.25-µm CO2 feature and water–ice ab-
sorptions seen in spectra of Ganymede and Callisto. Analysis of absorption features
throughout the VIMS-IR spectral range shows that the shift is consistent across the
entire wavelength range of VIMS-IR. Calibrations using stars and other objects
will be performed frequently throughout cruise and the orbital tour.

6.5. P OLARIMETRIC CALIBRATION

The VIMS instrument contains no specific polarimetric capability, such as filters
or grids, but the design of the grating spectrometer makes it inevitable that the
instrument’s response will be linearly polarized to some extent. The purpose of the
```

<!-- PDF_PAGE: 44 -->

## PDF page 44

```text
154                                    R.H. BROWN ET AL.




Figure 14. (a) VIMS-IR spectral profile at several wavelengths. (b) Enlargement of one profile to
show the small side lobes.
```

<!-- PDF_PAGE: 45 -->

## PDF page 45

```text
                  CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                           155




Figure 15. Monochromatic spectral image profiles for VIMS-IR near the filter gaps, as noted in the
text.




Figure 16. Spectra of Jupiter with VIMS-VIS and VIMS-IR. The inset shows an enlargement of the
region of spectral overlap.

polarimetric calibrations was to characterize this sensitivity, so that its effect on
the accuracy of spectra obtained for both polarized and non-polarized targets in
the Saturnian system can be assessed. In general, spacecraft pointing constraints
will determine the roll attitude of the optical remote sensing instruments during
```

<!-- PDF_PAGE: 46 -->

## PDF page 46

```text
156                               R.H. BROWN ET AL.

targeted observations, leaving little opportunity for observations at multiple roll
orientations.
    Measurements for the polarization characterization of VIMS-IR were carried
out in the 10-feet thermal-vacuum tank in Building 144 at JPL, during January 17–
20, 1996. A target projector was set up with a 26-mm diameter IR linear polarizing
filter at the focal plane of a collimator. The target was illuminated by the 1-inch
output aperture of an integrating sphere, producing a polarized image of the exit
aperture of the sphere at the VIMS focal plane. The polarizer consisted of a ZnSe
substrate with a deposited aluminum pattern of parallel stripes, and its polarization
efficiency is documented from 2.5 µm to beyond 10 µm. The average single-
polarizer throughput is 38%, while the maximum cross-polarized throughput over
the 2.5- to 5.0-µm range is 1.5%. No data on the filter transmission were available
below 2.5 µm. A calibrated tungsten source with a Teflon-coated sphere provided
adequate SNR out to about 2.5 µm, while a glowbar source with a gold integrating
sphere provided adequate SNR at all wavelengths beyond 1.5 µm.
    Measurements were made at five spatial positions of the source in the target
plane, on boresight, and near the four corners of the VIMS field. For each source
position, measurements were made at seven settings of the polarizer, at 30 ◦ intervals
between (+90 ◦ and −90 ◦ , followed by a sequence of background measurements
with the shutter on the light source closed. Only VIMS-IR was available for these
measurements.
    Three cubes per measurement were co-added and background-subtraction spec-
tra were extracted both for a central spot in the image of the source and for a larger
region which included the full output aperture of the sphere. This was accomplished
at each orientation of the polarizer and for each source position.
    Figure 17 shows 3-D surface plots of the normalized spectra for both light
sources, measured on the boresight. At each wavelength, the measured intensity at
a particular polarizer angle was divided by the average over all seven positions. From
this figure, and similar ones constructed for other source positions, it is apparent
that the maximum signal invariably occurs at an orientation of either 0 ◦ (polarizer
parallel to the rulings on the grating) or 90 ◦ , but that the sense and magnitude
of the polarized response vary smoothly with wavelength. We thus calculated the
fractional polarization of VIMS-IR from the expression:

               2I (λ, 0◦ ) − I (λ, −90◦ ) − I (λ, +90◦ )
   P(λ, %) =                                             × 100
               2I (λ, 0◦ ) + I (λ, −90◦ ) + I (λ, +90◦ )

   Figure 18 shows the polarization determined from central-spot and full-aperture
data from both sources, at two locations in the VIMS FOV. The results were found
to be the same to within 1% or better for all source positions and aperture sizes,
and essentially identical for both tungsten and glowbar sources in their region of
overlap (1.2–2.5 µm).
```

<!-- PDF_PAGE: 47 -->

## PDF page 47

```text
               CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                157




              Figure 17. 3D surface plots of VIMS polarization measurements.


   In summary, the calibration data showed the following:


1. The maximum polarization sensitivity is either parallel to (positive) or perpen-
   dicular to (negative) the rulings on the diffraction grating.
2. The polarization is positive for wavelengths of 1.1–2.1 µm, and beyond 2.4 µm,
   but negative below 1.1 µm. Between 2.1 and 2.4 µm the polarization is negligible
   (less than 1%). A peak in polarization occurs at 1.6 µm (+3%). Beyond 2.5 µm,
   the polarization increases linearly with wavelength, reaching a maximum of
   about 11% at 5.0 µm.
```

<!-- PDF_PAGE: 48 -->

## PDF page 48

```text
158                                R.H. BROWN ET AL.




                     Figure 18. Results of polarization measurements.

3. Except for some artifacts at the segment boundaries of the blocking filter at
   1.6 and 3.9 µm, the level of polarization is continuous, suggesting that the
   polarization is not related to the filter.
4. The lack of a significant dependence of polarization on the particular light source
   used indicates that the origin of the polarization is not in the sources, although
   it could conceivably be in the window of the thermal-vacuum chamber.
5. The lack of variation of polarization with target position suggests that the polar-
   ization does not arise from the VIMS scan mirror.
    These results suggest that the origin of the measured polarization is within the
spectrometer, and most likely resides in the diffraction grating. Measurements of the
ZnSe chamber window indicate no significant polarization, at least at wavelengths
below 1.7 µm.
    To assess the implications for spectral observations, consider a source with
a fractional linear polarization f in a direction è with respect to the VIMS “zero
angle”, as defined above. The parallel and perpendicular components of the incident
intensity are then:
   Ipar = [0.5 + f cos2 θ ]I0 and Iperp = [0.5 + f sin2 θ]I0
The measured intensity of the source by VIMS is:
   I (θ ) = (1 + P)Ipar + (1 − P)Iperp = [1.0 + f + P f cos(2θ)]I0
```

<!-- PDF_PAGE: 49 -->

## PDF page 49

```text
                CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   159

The RMS fractional error in a single measurement of the intensity at an arbitrary
orientation is:
    σ (I )        Pf
           =√
    <I >        2(1 + f )
For the most severe case of P = 11% (Figure 18), the RMS error is <1% for f < 0.15,
which probably includes any natural source polarization likely to be encountered.
At wavelengths below 3.3 µm, the error will reach 1% only for f = 0.9, and is thus
almost certainly negligible.

6.6. SOLAR PORT C ALIBRATION

6.6.1. Ground Calibration
The solar calibration port (henceforth “cal port”) was installed within the VIMS-IR
telescope to provide a strongly attenuated spectrum of the Sun during the Cassini
mission, which facilitates reduction of VIMS spectra to an accurate scale of absolute
reflectance. The cal port axis is offset by 20 ◦ from the telescope boresight in the −Z
direction, aligned with the UVIS solar occultation port. Attenuation of the incident
solar beam by a factor of ∼2.5 − 107 is achieved by (i) the small aperture of the cal
port, compared to the main beam and (ii) a series of one 70 ◦ and five 90 ◦ reflections
from right-angle prisms made of ZnSe. Most of the incident flux is directed back
through the entrance aperture by internal reflection in the prisms. The beam exiting
the cal port is focused by the telescope optics onto the VIMS-IR entrance slit,
and then enters the spectrometer in the same way. But because the cal port aperture
samples only a portion (∼0.3%) of the full instrument aperture, the collimated beam
illuminates only a small part of the diffraction grating. The optical design ensures
that this region overlaps the short- and medium-wavelength blaze regions on the
grating, in the ratio 1:3. The cal port does not illuminate the long-wavelength blaze.
    The predicted throughput of the stack of prisms varies smoothly from 1.35 ×
10−4 at 0.85 µm to 1.09 × 10−4 at 3.0 µm and 1.04 × 10−4 at 5.0 µm, for radiation
linearly polarized in a plane perpendicular to the plane containing the incident and
exit beams and to the rulings on the VIMS diffraction grating. The transmission of
the orthogonal polarization is less than 1.0 × 10−7 . The output spectrum of a target
seen through the cal port differs from that produced by the same source seen on the
instrument boresight. This arises through a combination of polarization and partial
illumination of the grating. Partial illumination of the grating results in a different
set of contributions from the different blazes than occurs when the grating is fully
illuminated. The purpose of the cal port calibrations is to establish the ratio of the
spectrum of an unpolarized broadband source as seen through the cal port to that
of the same source on the instrument boresight.
    The simulated solar source used for the calibration runs was a high-power xenon
lamp, illuminating a large, white, teflon-coated integrating sphere. The output of
the sphere illuminated a 1-mrad diameter circular aperture in the focal plane of
```

<!-- PDF_PAGE: 50 -->

## PDF page 50

```text
160                               R.H. BROWN ET AL.

the calibration projector (equal to the angular size of the sun at 9 AU), after a 90 ◦
reflection from an aluminum plate. One side of this plate was polished and oriented
for specular reflection of the high-intensity beam onto the target for the cal port
runs, while the other was sand-blasted to provide diffuse reflected illumination of
the target for the boresight runs.
    All measurements were made on July 12, 1996, in the 10-feet thermal-vacuum
chamber at JPL. Inside the tank, a deployable periscope-like arrangement with two
aluminum mirrors was used to redirect the input beam into the cal port’s entrance
aperture. The input signal to VIMS thus experienced identical atmospheric paths,
with the same number of reflections, for both cal port and boresight runs. For each
observation multiple 8 × 8 image cubes were taken, centered manually on the image
of the target. Only VIMS-IR was available for these runs; VIS-VIS was calibrated
separately at Officine Galileo.
    A total of 100 separate cal port image cubes were background subtracted and
averaged in sets of ten, and all pixels co-added to produce spectra. Twenty boresight
cubes were processed in identical fashion. Figure 19 shows the average boresight
and solar port spectra, at the DN levels originally recorded. The resulting ratio
spectrum is displayed in Figure 20, normalized to an average value of unity. Data
for nine channels at the three filter segment boundaries have been interpolated in
both plots.
    The ratio spectrum shows that the cal port sensitivity is relatively low shortward
of VIMS-IR channel 50 (1.7 µm), increases rapidly between channels 50 and 70 to
a peak at channel 80 (2.2 µm), and then declines smoothly at longer wavelengths.
Oscillations near 2.7 and 4.3 µm are due to imperfect cancellation of CO2 absorption
features in the raw spectra. The high-frequency structure beyond of 4.0 µm is an
artifact of the array readout.
    The decline in sensitivity of the cal port relative to the main aperture at longer
wavelengths is attributable to the lack of illumination of the long-wavelength grating
blaze, and to the increasing polarization sensitivity of the VIMS-IR beyond 3 µm.
The steep drop shortward of 2 µm is, on the other hand, unexplained. Neither the
ZnSe prisms nor the several aluminum reflections seem to be capable of causing
this effect. Additional experiments showed no indication of a misalignment in the
projector illumination pattern. Previous calibration runs on June 26, 1996, which
did not use the external Al plate, did not achieve sufficient SNR at the longer
wavelengths. Nevertheless, these earlier results are not consistent with the final
runs at short wavelengths, for reasons that are presently unknown. Thus, the cal
port ratio spectrum in Figure 20 must be considered provisional, until in-flight
experience is available.
    Viewed through the VIMS-IR cal port, images of a circular target are noticeably
elongated. FWHM dimensions are ∼2.0 pixels in X, but increase steadily from 2.8
pixels in the Z direction at 1 µm to 4.1 pixels at 5 µm. This may be compared with
FWHM sizes for the boresight images of 1.5 × 2.2 pixels for the same target. The
elongation of the solar port images is due to diffraction at the elongated rectangular
```

<!-- PDF_PAGE: 51 -->

## PDF page 51

```text
        CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                          161




Figure 19. Average instrumental response through the boresight and solar port.




  Figure 20. Ratio of response, solar calibration port (cal port) to boresight.
```

<!-- PDF_PAGE: 52 -->

## PDF page 52

```text
162                                R.H. BROWN ET AL.

aperture of the cal port (30 mm × 5 mm). Image centroids are fairly stable, varying
in X by at most 0.3 pixels and in Z by at most 0.2 pixels. Because of the variable size
and slightly wavelength-dependent position of the solar image, it is recommended
that solar cal port images be co-added over at least an 8 × 8 pixel region in order
to obtain a stable, well-defined spectrum. In-flight observations show that the solar
calibration ports in VIMS-VIS and VIMS-IR are co-aligned to within 1 pixel, but
offset by 10 mrad from the nominal pointing. This is well within the capability of the
VIMS scan mirrors to correct, and will thus permit simultaneous solar occultation
measurements with the UVIS instrument.

6.6.2. Results from Earth–Moon Flyby
Due to the large attenuation factors incorporated in the design of the VIMS-VIS
and VIMS-IR solar calibration ports, the only targets suitable for calibration ob-
servations after launch were Venus and the Moon. Prior to the Earth–Moon flyby
and the deployment of the VIMS-IR covers, only the VIMS-VIS was operational.
Although a short series of observations of Venus through the VIMS-VIS solar port
was attempted during the V2 flyby on 24 June 1999, a software error prevented
their successful execution. It appears, however, that light from Venus inadvertently
entered the solar port during pre-encounter background calibration frames.
    Approximately 95 min after Earth’s close approach on 18 August 1999, the
crescent Moon passed through the field of view of the VIMS solar port. Because
full pointing control of the spacecraft was not possible at this time, actual data
collection was limited to the period of ∼5 min. during which the Moon crossed the
VIMS look direction. The observations were made in image mode, with the slits
oriented perpendicular to the direction of the Moon’s apparent motion across the
focal plane (at 1.32 mrad/min), and with a VIS integration time of 20/sline and an
IR integration time of 320 ms/pixel. A total of four full 64 × 64 pixel image cubes
were obtained over a period of 91 min, centered on the predicted observation time.
Because the VIMS slow-scan direction is towards Z, and the Moon was drifting
towards −Z, the Moon was expected to cross the spectrometer entrance slits in 2.8
min., with the lit crescent visible for perhaps one min.
    The VIMS-VIS solar calibration port did indeed apparently detect the Moon in
two successive 20 s. integrations acquired at approx. 5:13 UT, at an average signal
level of 22 ± 4 DN above the local sky background. At a distance of 460,000 km,
the lunar diameter subtended 7.6 mrad, or 15 VIMS pixels, and the lit crescent
completely filled the width of the 0.5 mrad entrance slit. Nevertheless, the design
of the cal port prism diffuses the light uniformly along the slit, so no actual image
was obtained. Figure 21 shows the resulting spectrum, averaged over all 64 spatial
pixels along the slit and over both integrations. Plotted with the solar port spectrum
is a spectrum of the bright lunar limb acquired by VIMS-VIS through its main
aperture, suitably scaled. Although the illumination direction is the same for both
spectra, it should be noted that the cal port observations were obtained at a phase
angle of 110 ◦ , whereas the boresight spectrum was obtained at 90 ◦ .
```

<!-- PDF_PAGE: 53 -->

## PDF page 53

```text
               CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                  163




                 Figure 21. Solar port calibration at the Earth-Moon flyby.


    The lower panel of Figure 21 shows that the ratio of lunar spectra obtained
through the cal port and on boresight varies relatively smoothly with wavelength
(within the SNR limitations of the data), and is highest in the UV. A rough estimate
of the overall average attenuation by the VIMS-VIS cal port, taking into account
the different integration times and instrument gain states, is,
                              
            22DN         0.32s       1
    R=                           ×        = 4.5 × 10−5
           1000DN         20s        8
at an effective wavelength of ∼650 nm. Unfortunately, all of the IR data were
saturated because both the detector array and the entire spectrometer were too
warm.
   Although the data obtained at the Moon are of insufficient SNR to use for
calibration purposes, they are valuable in (i) demonstrating the functionality of the
VIMS-VIS cal port, (ii) confirming the validity of the ground calibration data in,
and (iii) providing a basis for estimating solar exposure times at Saturn.
```

<!-- PDF_PAGE: 54 -->

## PDF page 54

```text
164                                 R.H. BROWN ET AL.




                     Figure 22. Scattered light measurements at Venus.



6.7. S CATTERED LIGHT MEASUREMENTS FOR THE V ISUAL C HANNEL

The Venus 2 flyby on June 24, 1999 provided an excellent opportunity for measure-
ments of scattered light from a very bright off-axis source. Twenty seven integrations
were executed by VIMS-VIS shortly after Venus close-approach, and immediately
after the VIMS boresight moved off the bright limb of the planet onto dark sky.
Operating in line mode with an integration time of 20 s, these observations spanned
the period 20:32:29–20:42 UT and commenced with the VIMS boresight pointed
4.9 ◦ away from the nearest point on Venus’ limb.
    Data from the first three integrations, within 10 ◦ of the limb, are saturated
in all 96 spectral channels and all 64 spatial pixels along the slit. Subsequent
integrations show scattered light rapidly declining, as shown in Figure 22, until the
sky background level was reached at around 20:39 UT. The signal had fallen by
a factor of 10 by 20:35, when the VIMS boresight was pointed 23.2 ◦ off the limb.
    Throughout the scattered light observations, the recorded spectrum remained
essentially unchanged except for the absolute level, as shown in Figure 23. This
spectrum, however, is virtually flat and does not show the characteristic spectral
response of VIMS-VIS to broadband solar radiation, such as was observed at Venus
and the Moon. The feature at channel 34 (∼600 nm) is due to partial obscuration
of the focal plane array at the segment boundary between the short- and long-
wavelength blocking filters.
    Further analysis strongly suggests that the measured signal is due to off-axis
light from Venus which entered the entrance slit of the spectrometer enclosure
directly (passing just above the fold mirror M2), without reflecting off the scanning
```

<!-- PDF_PAGE: 55 -->

## PDF page 55

```text
                 CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                         165




                             Figure 23. Non dispersed stray light.




Figure 24. VIMS-VIS spectrum of the night side of Venus, compared to a synthetic spectrum (nor-
malized at 1.01 µm). From Baines et al., 2000.


mirror, and then was reflected by mirror M3 and the collimator directly to the
FPA, bypassing the diffraction grating. At the time of the observations, Venus’ disk
subtended a radius of 55–59 ◦ , and the nearest point on the limb was located within a
few degrees of the +Z direction from the VIMS boresight. 3-D ray-tracing indicates
that undispersed scattered light may reach the FPA from an off-axis direction of
∼17 ◦ towards +Z and ±10 ◦ in X, a location which would have fallen on the
illuminated disk of Venus over a period which matches that of the observed signal
in Figure 22.
```

<!-- PDF_PAGE: 56 -->

## PDF page 56

```text
166                                R.H. BROWN ET AL.

   A series of similar observations were made just as the VIMS boresight crossed
the opposite limb of Venus to probe the planet’s night side spectrum. These spectra,
obtained when the illuminated disk of Venus was located between −Z and +X,
relative to the boresight, and with the terminator 10–15 ◦ off axis, show no evidence
of a significant component of scattered light similar to that in Figure 23.
   Although very evident in long integrations, the level of scattered light is in
fact fairly modest: at 13 ◦ from Venus’ limb, and accounting for integration time
and instrument gain, the measured signal was ∼0.1% of the average flux received
directly from Venus. Similar situations are likely to be encountered during some
Titan flybys, and perhaps for Saturn observations near periapse, and may warrant
corrections for scattered light.

6.8. V ENUS SURFACE AND ATMOSPHERE

The power of VIMS to make new discoveries was amply demonstrated by the very
first planetary spectrum acquired by the instrument (Baines et al., 2000). On June
24, 1999, during the Cassini flyby of Venus, the visual channel obtained a single
long-duration (10 s) exposure of the planet’s night side. This observation was cen-
tered near 20 ◦ N. latitude and 60 ◦ E. longitude, and for the first time quantitatively
measured Venusian surface emissions at sub-micrometer wavelengths. As shown in
Figure 24, the emission profiles of two spectral features predicted by Lecacheux et
al. (1993) at 0.85 and 0.90 µm were measured. As with other surface emissions pre-
viously observed near 1.01, 1.10, and 1.18 µm by ground-based observers and the
near-infrared mapping spectrometer (NIMS) onboard the Galileo spacecraft (e.g.
Carlson et al., 1991, 1993a,b; Crisp et al., 1991; Lecacheux et al., 1993; Meadows
and Crisp, 1996), these features are located in spectral windows devoid of atmo-
spheric absorption through which radiation emitted from the hot (∼740 K) surface
passes through the ∼90 bars of CO2 -laden atmosphere into space. This spectrum
is the first detection of the 0.90-µm feature. The shorter wavelength feature at 0.85
µm was provisionally detected in a single wavelength channel just above the detec-
tion limit by Galileo NIMS (Carlson et al., 1991). In contrast, the VIMS spectral
profile of this feature spans a wide range of wavelength at high signal-to-noise.
    Adding to the previous work of NIMS and groundbased observers, VIMS thus
demonstrated that the surface of Venus could be observed in five distinct windows
between 0.85 and 1.18 µm. As shown by Baines et al. (2000), compositional
maps of the surface of Venus may be obtainable from future spacecraft that take
advantage of this novel technique to observe the glowing surface of Venus under
nighttime conditions. Future observations should take simultaneous measurements
of the flux generated at several purely non-gas-absorbing atmospheric wavelengths
(e.g. 1.28, 1.74, 2.29 µm) as well as within the five surface-detecting windows,
so that the surface spectrum can be corrected for the highly-spatially-variable
extinction due to overlying clouds. As previously demonstrated (e.g. Carlson et al.,
1993a,b), due to the spherical shape and non-absorbing nature of Venus’s sulfuric
```

<!-- PDF_PAGE: 57 -->

## PDF page 57

```text
                 CASSINI VISUAL AND INFRARED MAPPING SPECTROMETER                   167

acid cloud particles, the wavelength-dependent extinction of Venus aerosols can
be estimated over a large wavelength range from observations made at relatively
long wavelengths (NIMS successfully used 1.73 and 2.4 µm fluxes to determine
cloud extinction at 1.18 µm). In the case of VIMS, observations longward of 1.05
µm were unobtainable since the optics cover of VIMS-IR had not been opened as
of the Venus encounter, as noted above. Thus, while the relative shape of various
surface emission features could be observed by VIMS-VIS, absolute surface
fluxes could not be determined. Nevertheless, the distinct advantage VIMS has
over previous spectral instruments to the outer planets—in particular, its ability
to simultaneously acquire all wavelengths of a spectrum, and its ability to take
long exposures (e.g. 10 s here as opposed to enabled VIMS at Venus to make new
discoveries with its very first observation, and portends well the making of new
discoveries at the mission’s major target, the Saturn system.


                                      7. Conclusions

All indications to the writing of this document are that the VIMS instrument in flight
has met or exceeded all of its preflight specifications. Measurements have been, and
will continue to be, made of the instrument’s performance throughout the roughly
6.5 years of Cassini’s cruise to Saturn in preparation for the orbital tour. There is
every reason to expect VIMS performance will meet or exceed expectations during
the Saturn orbital tour, and that the science results produced from VIMS data will be
both rich and exciting. It is indeed a privilege for all of the authors of this document
to have participated in the VIMS adventure thus far, and we look forward to the
exciting discoveries yet to be made at Saturn.


                                   Acknowledgements

The number of people who are responsible for the design, construction, test, launch
and operation of the Cassini/Huygens Mission in general, and the VIMS instru-
ment in particular, is truly enormous, and the brilliance of their contributions may
sometimes get lost in the glare of the science discoveries that we all hope will result
at Saturn. We cannot list them all here, but they know who they are, and the VIMS
Science Team especially wants to acknowledge the enabling contributions of all
of those bright and dedicated people who created VIMS, and those who are now
helping us to operate it. Without them, none of what is possible with VIMS would
be possible. Many thanks to all of you and we hope to see you at Saturn.


                                         References

Aptaker, I.: 1982, SPIE Instrumentation in Astronomy IV Proceedings, pp. 182–196.
```

<!-- PDF_PAGE: 58 -->

## PDF page 58

```text
168                                       R.H. BROWN ET AL.

Baines, K. H., Brown, R. H., Matson, D. L., Nelson, R. M., Buratti, B. J., Bibring, J. P., et al.: 1992,
    Symposium on Titan, ESA SP-338, pp. 215–219.
Baines, K. H., Bellucci, G., Bibring, J.-P., Brown, R. H., Buratti, B. J., Bussoletti, E., et al.: 2000,
    Icarus 148, 307–311.
Baines, K. H., Carlson, R. W., and Kamp, L. W.: 2002, Icarus 159, 74–94.
Brown, R. H., Baines, K. H., Bellucci, G., Bibring, J.-P., Buratti, B. J., Capaccioni, F., et al.: 2003,
    Icarus 164, 461–470.
Carlson, R. W., Baines, K. H., Kamp, L. W., Weissman, P. R., Smythe, W. D., Ocampo, A. C., et al.:
    1991, Science 253, 1541–1548.
Carlson, R. W., Baines, K. H., Girard, M., Kamp, L. W., Drossart, P., Encrenaz, T., et al.: 1993a,
    Proceedings of XXIV Lunar Planetary Science Conference, 253.
Carlson, R. W., Kamp, L. W., Baines, K. H., Pollack, J. B., Grinspoon, D. H., Encrenaz, T., et al.:
    1993b, Planet. Space Sci. 41, 477–485.
Crisp, D., Allen, D. A., Grinspoon, D. H., and Pollack, J. B.: 1991, Science 253, 1263–1266.
Green, R. O., Eastwood, M. L., Sarture, C. M., Chrien, T. G., Aronsson, M., Chippendale, B. J., et
    al.: 1998, Remote Sensing Environ. 65, 227–248.
Griffith, C. A. and Owen, T.: 1992, Symposium on Titan, ESA SP-338, 199–204.
Lecacheux, J., Drossart, P., Laques, P., Deladerriere, F., and Colas, F.: 1993, Planet. Space. Sci. 41,
    543–549.
McCord, T. B. and Adams, J. B.: 1973, Moon 7, 453–474.
McCord, T. B. and the VIMS Team: in press, Icarus.
Meadows, V. S. and Crisp, D.: 1996, J. Geophys. Res. 101, 4595–4622.
Miller, E, Klein, G., Juergens, D., Mehaffey, K., Oseas, J., Garcia, R., et al.: 1996, SPIE 2803, 206.
Macenka, S.: 1982, SPIE 43, 46.
Meier, R., Smith, B. A., Owen, T. C., and Terrile, R. J.: 2000, Icarus 145, 362–473.
Pollack, J. B., Dalton, J. B., Grinspoon, D., Watson, R. B., Freedman, R., Crisp, D., Allen, D. A., et
    al.: 1993, Icarus 103, 1–42.
Reininger, F., Dami, M., Paolinetti, R., Pieri, S., and Falugiani, S.: 1994, SPIE 2198, 239–250.
Rice, R., Yeh, P., and Miller, W.: 1991, Algorithms for a Very-High-Speed, Noiseless Encoding Module.
    JPL Publication 91-1.
Roos-Serote, M., Drossart, P., Encrenaz, T., Lellouch, E., Carlson, R. W., Baines, K. H., et al.: 1999,
    J. Geophys. Res. Planets 103, 23023–23042.
Roos-Serote, M., Vasavada, A. R., Kamp, L., Drossart, P., Irwin, P., Nixon, C., et al.: 2000, Nature
    405, 158–160.
Simon-Miller, A. A., Conrath, B., Gierasch, P., and Beebe, R.: 2000, Icarus 45, 454–461.
Stammes, P.: 1992, Symposium on Titan, ESA SP-338, pp. 205–210.
Tomasko, M., Pope, S., Kerola, D., Smith, P., and Giver, L.: 1989, Bull. Amer. Astron. Soc. 21, 961–962.
```
