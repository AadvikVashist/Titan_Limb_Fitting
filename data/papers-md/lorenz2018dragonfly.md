---
citation_key: "lorenz2018dragonfly"
title: "Dragonfly: A rotorcraft lander concept for scientific exploration at Titan"
source_pdf: "data/papers/lorenz2018dragonfly.pdf"
source_pdf_sha256: "7e31a509bbd8dfb7d7b8d83ae98d6583c5962e762273e90f88f364531de01ed9"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
     Dragonfly: A Rotorcraft Lander Concept for
     R. D. Lorenz et al.




     Scientific Exploration at Titan

                      Ralph D. Lorenz, Elizabeth P. Turtle, Jason W. Barnes, Melissa G. Trainer,
                           Douglas S. Adams, Kenneth E. Hibbard, Colin Z. Sheldon, Kris Zacny,
               Patrick N. Peplowski, David J. Lawrence, Michael A. Ravine, Timothy G. McGee,
                    Kristin S. Sotzen, Shannon M. MacKenzie, Jack W. Langelaan, Sven Schmitz,
                                                     Lawrence S. Wolfarth, and Peter D. Bedini




     ABSTRACT
     The major post-Cassini knowledge gap concerning Saturn’s icy moon Titan is in the composition
     of its diverse surface, and in particular how far its rich organics may have ascended up the ”ladder
     of life.” The NASA New Frontiers 4 solicitation sought mission concepts addressing Titan’s habit-
     ability and methane cycle. A team led by the Johns Hopkins University Applied Physics Laboratory
     (APL) proposed a revolutionary lander that uses rotors to land in Titan’s thick atmosphere and
     low gravity and can repeatedly transit to new sites, multiplying the mission’s science value from its
     capable instrument payload.




     INTRODUCTION
         Saturn’s moon Titan is in many ways the most Earth-          Titan is an “ocean world” that is rich in both carbon and
     like body in the solar system.1–3 This strange world is          nitrogen.4,5 See Table 1 for data on Titan’s environment.
     larger than the planet Mercury and has a thick nitrogen
     atmosphere laden with organic smog, which partly hides
     its surface from view. Since cold Titan is far from the          FORMULATION OF THE DRAGONFLY CONCEPT
     Sun, on Titan methane plays the active role that water               The NASA community announcement in Janu-
     plays on Earth, serving as a condensable greenhouse gas,         ary 2016 identifying Titan as a possible target for the
     forming clouds and rain, and pooling on the surface as           fourth New Frontiers mission opened new possibilities
     lakes and seas. Titan’s carbon-rich surface is shaped not        in Titan exploration (Box 1). Although the exploration
     only by impact craters and by winds that sculpt drifts           of Titan’s seas had previously been considered, notably
     of aromatic organics into long linear dunes but also by          by the APL-led Titan Mare Explorer (TiME) Discovery
     methane rivers and possible eruptions of liquid water            concept,6,7 the timing mandated by the announcement
     (“cryovolcanism”).                                               of opportunity precluded such a mission. Specifically,
         While living things are ~70% water, and finding water        with launch specified prior to the end of 2025, Titan
     has been a convenient initial focus for astrobiological          arrival would be in the mid-2030s, during northern
     investigations in the solar system, the chemical processes       winter. This means the seas, near Titan’s north pole, are
     that conspire to lead to life rely on functions exerted by       in darkness and direct-to-Earth (DTE) communication
     compounds of carbon, nitrogen, oxygen and hydrogen,              is impossible.8 Even with the higher budget threshold of
     with traces of sulfur and phosphorus (CHNOPS). In con-           New Frontiers 4 ($850 million plus launch and opera-
     trast to Europa (abundant in water, and perhaps sulfur),         tions costs) compared with Discovery (~$450 million


374­­­­                                                     Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                                                              Dragonfly: A Rotorcraft Lander Concept for Scientific Exploration at Titan


 Table 1. Titan’s Environment
                  Property                                                                      Surface Valuea
Diameter                                        5150 km (larger than Mercury)
Surface gravity                                 1.35 m/s2 (1/7 Earth)
Distance from Saturn                            1.2 million km (20 Saturn radii)
Rotation period (Titan day or Tsolb)            15.945 days (same as orbit period around Saturn)
Atmospheric pressure                            1.47 bar (note: Earth surface pressure = 1.01 bar)
Atmospheric temperature                         94 K
Atmospheric density                             5.4 kg/m3 (4× Earth sea level air)
Atmosphere composition                          95% nitrogen, 5% methane, 0.1% hydrogen, many trace organics
Speed of sound                                  195 m/s
Atmospheric viscosity                           6 × 10 –6 Pa-s (~3× smaller than Earth air)
Obliquity                                       26° to Sun (equatorial plane is ~ Saturn ring plane)
Surface illumination                            ~1000× less than Earth (or ~1000× full moonlight) predominantly in red and near-IR light;
                                                visibility near surface ~10 km
a Atmospheric properties vary with altitude; surface values shown here.

b Tsol, Titan solar day.



plus radioisotope power source and launch costs), it                                 proposed some 17 years ago.11,12 At that time, the vehi-
would be challenging indeed to provide a relay space-                                cle was imagined to be a helicopter, a vehicle that is used
craft and a sea probe.                                                               on Earth for near-guaranteed access to a wide range of
   A lander with DTE communication would be possible                                 terrain, for personnel delivery, and for search and rescue.
at lower latitudes, however. The only detailed study of                              However, helicopters are mechanically complex (one
such a mission (see Box 2) was the 2007 Titan Explorer
NASA Flagship Mission Study,9,10 led by the Johns Hop-
kins University Applied Physics Laboratory (APL). This                                      BOX 1. OCEAN WORLDS
study advocated the science that could be obtained from                                     Although the list of candidate New Frontiers mis-
three platforms, an orbiter, a hot-air (Montgolfière) bal-                                  sions described in the 2013 Planetary Science Decadal
loon, and a lander. The lander (designed before Titan’s                                     Survey did not include a mission to Titan, the survey
seas had been discovered) was intended to be delivered to                                   did recognize the scientific value of Titan exploration,
Titan’s Belet sand sea, a large—and thus easily targeted—                                   advocating technology development toward a flagship
dune field expected to be free of rock and gully hazards.                                   mission. Further, the 2008 New Opportunities in Solar
                                                                                            System Exploration (NOSSE): An Evaluation of the New
After the lander’s parachute descent and landing on
                                                                                            Frontiers Announcement of Opportunity report advo-
Pathfinder-like airbags (wherein if it landed on top of a                                   cated that New Frontiers missions should be responsive
dune, it would just roll down to the bottom), petals would                                  to scientific discoveries. In January 2016, NASA intro-
unfold and science would begin, with cameras, a chemi-                                      duced an “Ocean Worlds” target (Titan and/or Encela-
cal analysis suite, a seismometer, and a meteorology pack-                                  dus) into the community notice regarding the upcom-
age. Much of the science definition in the Titan Explorer                                   ing New Frontiers 4 announcement of opportunity, the
Study was useful in formulating the Dragonfly proposal.                                     final version of which was released in December 2016.
                                                                                            That announcement defines the overarching scientific
   A scientific limitation of a single lander, however, is
                                                                                            objectives as follows:
that it explores only a single location. This limitation
can be mitigated slightly at “grab-bag” landing sites                                          The Ocean Worlds mission theme is focused on
where geological processes have gathered samples from a                                        the search for signs of extant life and/or charac-
range of areas (in Mars Pathfinder’s case, a flood deposit                                     terizing the potential habitability of Titan and/
                                                                                               or Enceladus.
of rocks; dune sands may similarly have material from a
range of source locations). However, a lander with some                                        For Titan, the science objectives (listed without
kind of mobility, or augmented by some mobile element                                          priority) of the Ocean Worlds mission theme are:
(e.g., a “fetch” rover), would help address the challenge                                      • Understand the organic and methanogenic
of acquiring samples from sites more interesting than the                                        cycle on Titan, especially as it relates to prebi-
landing point, a site that would be most likely selected                                         otic chemistry; and
for safety rather than for scientific interest.                                                • Investigate the subsurface ocean and/or liquid
   The concept of a rotorcraft lander on Titan trickle-                                          reservoirs, particularly their evolution and pos-
charging a battery for brief atmospheric flights by using                                        sible interaction with the surface.
the power from a radioisotope power source had been


Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest                                                                            375­­­­
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
     R. D. Lorenz et al.




     Figure 1. Dragonfly mission concept. After delivery from space in an aeroshell and parachute descent, the vehicle lands under rotor
     power and deploys a high-gain antenna for DTE communication. Powered by a radioisotope power supply that provides heat and
     trickle-charges a large battery, the vehicle can operate nearly indefinitely as a conventional lander but can also make periodic brief
     battery-powered rotor flights to new locations.

     reason that this concept was considered only briefly in                     indicated that a vehicle of representative size and power
     the Flagship Study).                                                        could in fact achieve unparalleled regional mobility on
         However, technology developments in the last two                        Titan, and the Dragonfly concept was born. Initially it
     decades, notably the revolution in availability of multi-                   was imagined that the vehicle might have a flotation
     rotor dronesa made possible by modern compact sensors                       ring, to permit landing on one of Titan’s lakes, but a
     and autopilots as well as the development of sensing and                    more conventional box-with-skids layout soon emerged
     control capabilities for autonomous landing and site                        once it was decided that operations on dry land would be
     evaluation for planetary landers, made a quadcopter or a                    the focus of the mission. A constraint in this application
     similar vehicle a much more feasible prospect in 2016. In                   that is somewhat unusual for rotorcraft is the necessity
     contrast to helicopter flight, multi-rotor flight with dif-                 to be packaged in a hypersonic aeroshell. The geomet-
     ferential throttling effected purely electrically by motor                  ric trade of unblocked rotor disk area versus number of
     speed control is mechanically simple and therefore lends                    rotors14 with such a constraint suggests that, in fact, four
     itself to planetary application.                                            is optimal.
         A brief evaluation using a para-
     metric rotorcraft power model13
                                                                                      Backshell
     a It is interesting to recall that the first
     practical helicopter to fly in the United
     States, in 1924, was a multi-rotor vehi-
     cle, the “flying octopus” (see https://
     en.wikipedia.org/wiki/De_Bothezat_
     helicopter). Although this vehicle flew
     over 100 times with as many as four pas-
     sengers and broke many records, the pilot
     workload to achieve control by differen-
     tial thrust on four rotors each with vari-
     able pitch was formidable. Although the
     same capabilities were not achieved for
     another 20 years, the Army Air Service
     scrapped the project. It is also interest-                                       3.7-m-diameter heatshield
     ing to note that while hovering drones on
     Earth have been enabled by high-power-
     density battery technology, specifically
     the 21st-century emergence of lithium-ion      Figure 2. Although the challenges of delivering a vehicle into the Titan atmosphere are not
     and lithium-polymer cells, in Titan’s low      the subject of this article, the design of the cruise stage and entry system demanded signifi-
     gravity and thick atmosphere, compa-
     rable vehicles (if kept warm) would not        cant effort. The rotorcraft is launched “upside-down” with the stowed skids and the forward
     need such high power or energy densities.      face of the aeroshell upward on the launch vehicle.



376­­­­                                                                Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                                                            Dragonfly: A Rotorcraft Lander Concept for Scientific Exploration at Titan



     BOX 2. PROMINENT POST-CASSINI MISSION STUDIES AND PROPOSALS
     While many smaller studies are described in conference                              tion team (SDT). The SDT assigned a higher scientific
     papers or similar (see Ref. 46 for a review), the following list                    priority to the lander than to the Montgolfière—surface
     identifies major efforts. The first suggestion of Titan heli-                       chemistry and internal structure were considered more
     copters (at least the first mention of which we are aware)                          important goals.
     falls into this former category, a passing mention of small                     • 2009 Titan Saturn System Mission (TSSM). This
     fetch vehicles to return surface samples to a rather improb-                        JPL-led study50 built on Titan Explorer, but with a
     able 8-metric-ton nuclear-thermal reactor-powered space-                            headquarters-mandated architecture including Euro-
     plane, described by Zubrin in a 1990 conference paper.47                            pean Space Agency-provided in situ elements (a Mont-
     • 1999 Prebiotic Material in the Outer Solar System                                 golfière and a short-lived battery-powered lake lander),
       Campaign Science Working Group (CSWG). Various                                    requiring Enceladus as well as Titan science, and prohib-
       discipline-oriented CSWGs were a predecessor of the                               iting aerocapture. A related architecture was explored
       Planetary Science Decadal Survey, the first of which                              in a very preliminary way in the European-led TandEM
       convened in 2003, before Cassini’s arrival informed                               (Titan and Enceladus Mission) proposal.51
       future priorities. Nonetheless, the CSWG recognized48                         • 2010 AVIATR. This concept52 was for an airplane
       the potential for aerial mobility at Titan and the impor-                         at Titan, powered by Advanced Stirling Radioisotope
       tance of Titan’s surface chemistry. The first thinking                            Generators (ASRGs) to fly continuously to perform
       about heavier-than-air exploration, and rotorcraft in                             an aerial survey with DTE communication. Although
       particular, took place in this period.                                            stimulated by the 2010 Discovery solicitation, this idea
     • 2006 TiPEx—Titan Prebiotic Explorer.49 TiPEx was a                                proved incompatible with the Discovery budget.
       Jet Propulsion Laboratory (JPL) concept, not externally                       • 2010 TiME (Titan Mare Explorer). This APL–
       funded, for a Montgolfière (hot-air) balloon and orbiter.                         Lockheed Martin proposal6,7 was selected for a Phase A
       Surface chemistry was to be addressed by dropping a                               study in the 2010 Discovery solicitation. It was a capsule
       harpoon sampler to be winched back up to the balloon                              that would float in Ligeia Mare, Titan’s second-largest
       gondola. Earlier JPL studies had considered a more com-                           sea, using ASRGs for power and DTE communication
       plex dirigible balloon (airship).                                                 and would perform (liquid) composition measurements,
     • 2007 Titan Explorer Flagship. This APL-led NASA                                   imaging and sonar surveys, and meteorological
       study10,11 advocated a lander, Montgolfière, and aero-                            observations.
       captured orbiter to address the widest range of scientific                    It is evident that Dragonfly responds to long-standing sci-
       disciplines and spatial scales. The lander would address                      entific priorities and ideas. Remarkably, the combination of
       surface chemistry, relieving the Montgolfière of the risks                    long-term landed science and occasional aerial flight offers
       of near-surface operations and sampling. This was the                         in a single platform most of the combined capabilities of both
       first study to feature a NASA-appointed science defini-                       the lander and balloon elements of the 2007 Flagship Study.



    Although there is a small aerodynamic penalty in                                 NASA-appointed science definition team for the 2007
the “over–under” quad octocopter layout (with a top                                  Flagship Study lander, embracing geophysical, imaging,
and bottom pair of motors/rotors at each corner of the                               and meteorological studies, as well as the centerpiece
vehicle) compared with a “pure” quad, the octocopter                                 science of surface chemistry. Novel elements include
configuration is more resilient, being able to tolerate the                          measurement of atmospheric hydrogen as a possible
loss of at least one rotor or motor.                                                 biomarker16 and the capability of making rapid elemen-
    The architecture of the sample acquisition system, to                            tal composition measurements via neutron-activated
be provided by Honeybee Robotics, was another major                                  gamma-ray methods17 without requiring sample
trade: a sampling arm like those used on Viking, Phoe-                               ingestion—a particularly powerful capability for a relo-
nix, or the Mars Science Laboratory, was considered, but                             catable lander. Particular sites of interest deserving closer
it would be expensive and heavy and presented a single-                              investigation with ingested samples include those where
point failure. Instead, two sample acquisition drills, one                           liquid water (e.g., from impact melt) has interacted with
on each landing skid, with simple 1-degree-of-freedom                                Titan’s organic haze deposits to produce18,19 pyrimidines
actuators were selected. These provide a sample choice                               (bases used to encode information in DNA) and amino
and redundancy. Titan’s dense atmosphere permits the                                 acids, the building blocks of proteins. In addition to mul-
sample (whether sand, icy drill cuttings, or other mate-                             tiplying the surface chemistry science value by visiting
rial) to be conveyed pneumatically15 by a blower—the                                 multiple sites, Dragonfly’s capabilities for meteorological
material is sucked up through a hose and is extracted in a                           measurements and imaging during flight are comparable
cyclone separator (much like in a Dyson vacuum cleaner)                              with those of a balloon—the revolutionary single-ele-
for delivery to the mass spectrometer instrument.                                    ment Dragonfly concept affordably fulfils most of the
    The scientific payload (Box 3) for Dragonfly is in                               science objectives met by two of the elements (lander
many respects a (large) subset of that identified by a                               plus Montgolfière) in flagship architectures.


Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest                                                                          377­­­­
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
     R. D. Lorenz et al.

                                                                                                      nas, for example]. A mission
                                                                                                      following on from Huygens should
                                                                                                      logically do better than Huygens.
                                                                                                      The Huygens probe returned
                                                                                                      about 100 MB of data (~3.5 h of
                                                                                                      an S-band link at 8 kbps, relayed
                                                                                                      to Earth by the Cassini orbiter23).
                                                                                                      To do, say, 100 times better,
                                                                                                      10 GB, would therefore require
                                                                                                      at 10 AU about 0.5 GJ of energy
                                                                                                      (140,000 Wh, far beyond the capa-
                                                                                                      bility of practical stored energy
                                                                                                      systems like primary batteries) and
                                                                                                      necessitates radioisotope power.
                                                                                                          The free parameter in the
                                                                                                      system design is the mission dura-
                                                                                                      tion. For the steady output from a
                                                                                                      radioisotope power source, the mis-
                                                                                                      sion energy, and thus data return,
                                                                                                      scales directly with duration. One
                                                                                                      year of (say) 100 W output corre-
     Figure 3. The Dragonfly configuration for atmospheric flight (with the gray circular HGA         sponds to 3 GJ of energy.
     stowed flat). Note the aerodynamic fairing in front of the HGA gimbal. The cylinder at rear is       The New Frontiers 4 announce-
     the Multi-Mission Radioisotope Thermoelectric Generator (MMRTG). A sampling drill mech-          ment   of opportunity permits the
     anism is visible in the nearside skid leg, and forward-looking cameras are recessed into the     use of up to three MMRTGs. Since
     tan insulating foam forming the rounded nose of the vehicle. The rotor wing section and          these are relatively heavy, and the
     planform are designed for the Titan atmosphere.                                                  waste heat (some 2 kW) requires
                                                                                                      careful management (although
     ENERGY IS EVERYTHING                                                   some   heat is  in fact essential for this application), it was
                                                                            obvious that only a single unit should be used.
        It was recognized, in the same study12 that articulated                Slow degradation of the thermoelectric converter,
     the trickle-charged helicopter idea, that energy is the                in addition to the decay of the plutonium heat source,
     fundamental limitation in Titan surface exploration. In                means the electrical power output at Titan is consider-
     that environment, solar power is impracticable (sunlight               ably lower than at launch, 9 years earlier. Furthermore,
     at Titan’s surface is ~100× weaker than at Earth, due to               uncertainty in that degradation (known only from
     Titan’s distance from the Sun, and is further diminished               ground tests and from the ~5 years of operation of the
     by a factor of ~10 by Titan’s hazy atmosphere20), and the              MMRTG on Curiosity24) requires healthy margins on
     strong cooling provided by Titan’s dense 94-K atmo-                    the power budget. An electrical power output of about
     sphere requires sustained heat for thermal management.                 70 W from a single MMRTG is anticipated at Titan.
        The vehicle body, like the Huygens probe, has thick                 While this is indeed low, it may be recalled that both
     insulation around its main electronics box, and “waste”                Viking landers operated for years on this power level.
     heat from the Multi-Mission Radioisotope Thermoelec-                   The key is that landed operations are undemanding (no
     tric Generator (MMRTG) is tapped to maintain this                      propulsion or attitude control) and flexible.
     interior (and most particularly, the battery) at benign                   Although sample acquisition and chemical analysis
     temperatures. On the other hand, the sensitive gamma-                  are somewhat power-hungry activities, they require only
     ray detector of the DraGNS instrument (see Box 3) is                   a few hours of activity. Science activities that require
     mounted outside this warm box, exploiting the dense                    continuous monitoring, namely meteorological and seis-
     cold atmosphere to attain low operating temperatures                   mological measurements, although of low power, actually
     without needing a mechanical cryocooler.                               dominate the payload energy budget. Indeed, for these
        Missions with high-gain antennas (HGAs) empiri-                     extended periods, the lander avionics are powered down
     cally require about 5 mJ per bit per astronomical unit21               and data acquisition is performed only by the instru-
     to acquire and send science data to Earth [the linear                  ment, to maximize the rate of recharge of the battery.
     distance dependence is an interestingly emergent “allo-                   Except during polar summer or winter, operations of
     metric” correlation (see also Ref. 22) that results from               a lander on Titan with DTE communication are paced
     engineering efforts to defeat the inverse square law—                  by the Titan diurnal cycle. A Titan solar day (Tsol) is
     spacecraft at greater distances tend to have larger anten-             384 h long (16 Earth days). Seen from Titan, Earth in


378­­­­                                                           Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                                                            Dragonfly: A Rotorcraft Lander Concept for Scientific Exploration at Titan

the sky is always within 6° of the Sun. Interaction with                             higher conversion efficiencies than the MMRTG, would
Earth, and logically any operations requiring real-time                              permit an even higher data return or rate of flight.
observation (such as atmospheric flight), occur during
the day, and nighttime activities are generally mini-
mal and power can be devoted to recharging the bat-                                  ATMOSPHERIC FLIGHT PERFORMANCE AND
tery. Thus, a logical maximum size of the battery is that                            AERODYNAMIC DESIGN
which completely captures MMRTG power during the
Titan night, or 75*192 = 14 kWh. Such a battery—about                                   Titan’s atmosphere is both denser (4.4×) and colder
a quarter of the size of the battery in a Tesla electric                             (94 K) than Earth’s. The composition is predominantly
car—would be rather massive (140 kg), assuming a rep-                                (95%) nitrogen, and the low temperature means molecu-
resentative specific energy metric for space-qualified bat-                          lar viscosity is rather lower than for our air. The com-
teries of 100 Wh/kg. In practice, a smaller battery may                              bination of higher density and lower viscosity means
be chosen, sacrificing some energy-harvesting efficiency                             that an airfoil of given size and speed is operating at a
for lower mass and cost. It should be emphasized that                                Reynolds number that is several times higher than on
while the mission has been designed to function with                                 Earth. To a first order, then, the ~1 m rotors of Dragon-
the MMRTG, other comparable radioisotope power                                       fly should resemble rotors of much-larger-scale systems
systems,25 such as the Advanced Stirling Radioisotope                                on Earth—in fact, a blade section more typically used
Generator (ASRG) or an enhanced MMRTG with                                           in terrestrial wind turbines has been adopted. Not only


     BOX 3. DRAGONFLY SCIENCE PAYLOAD                                                  thermal anemometers (similar to those flown on several
     The Dragonfly science payload includes the following                              Mars missions) placed outboard of each rotor hub, so that
     instruments:                                                                      at least one senses wind upstream of the lander body,
                                                                                       minimizing flow perturbations due to obstruction and by
     • DraMS—Dragonfly Mass Spectrometer (Goddard                                      the thermal plume from the MMRTG. Methane abun-
       Space Flight Center). A central element of the pay-                             dance (humidity) is sensed by differential near-IR absorp-
       load is a highly capable mass spectrometer instrument,                          tion, using components identified in the TiME Phase A
       with front-end sample processing able to handle high-                           study. Electrodes on the landing skids are used to sense
       molecular-weight materials and samples of prebiotic                             electric fields (and in particular the AC field associated
       interest. The system has elements from the highly suc-                          with the Schumann resonance, which probes the depth
       cessful SAM (Sample Analysis at Mars) instrument on                             to Titan’s interior liquid water ocean) as well as to mea-
       Curiosity, which has pyrolysis and gas chromatographic                          sure the dielectric constant of the ground. The thermal
       analysis capabilities, and also draws on developments for                       properties of the ground are sensed with a heated tem-
       the ExoMars/MOMA (Mars Organic Material Analyser).                              perature sensor to assess porosity and dampness. Finally,
     • DraGNS—Dragonfly Gamma-Ray and Neutron                                          seismic instrumentation assesses regolith properties (e.g.,
       Spectrometer (APL/Goddard Space Flight Center).                                 via sensing drill noise) and searches for tectonic activity
       This instrument allows the elemental composition of                             and possibly infers Titan’s interior structure.
       the ground immediately under the lander to be deter-                          • DragonCam—Dragonfly Camera Suite (Malin Space
       mined without requiring any sampling operations. Note                           Science Systems). A set of cameras, driven by a common
       that because Titan’s thick and extended atmosphere                              electronics unit, provides for forward and downward
       shields the surface from cosmic rays that excite gamma-                         imaging (landed and in flight), and a microscopic imager
       rays on Mars and airless bodies, the instrument includes                        can examine surface material down to sand-grain scale.
       a pulsed neutron generator to excite the gamma-ray                              Panoramic cameras can survey sites in detail after land-
       signature, as also advocated for Venus missions. The                            ing: in many respects, the imaging system is similar to
       abundances of carbon, nitrogen, hydrogen, and oxygen                            that on Mars landers, although the optical design takes
       allow a rapid classification of the surface material (for                       the weaker illumination at Titan (known from Huygens
       example, ammonia-rich water ice, pure ice, and carbon-                          data) into account. LED illuminators permit color imag-
       rich dune sands). This instrument also permits the                              ing at night, and a UV source permits the detection of
       detection of minor inorganic elements such as sodium or                         certain organics (notably polycyclic aromatic hydrocar-
       sulfur. This quick chemical reconnaissance at each new                          bons) via fluorescence.
       site can inform the science team as to which types of
                                                                                     • Engineering systems. Data from the inertial measure-
       sampling (if any) and detailed chemical analysis should
                                                                                       ment unit (IMU) may be used to recover an atmospheric
       be performed.
                                                                                       density profile via the deceleration history during entry.
     • DraGMet—Dragonfly Geophysics and Meteorology                                    IMU and other navigation data may provide constraints
       Package (APL). This instrument is a suite of simple sen-                        on winds during rotorcraft flight. Additionally, the radio
       sors with low-power data handling electronics. Atmo-                            link via Doppler and/or ranging measurements may shed
       spheric pressure and temperature are sensed with COTS                           light on Titan’s rotation state, which, in turn, is influ-
       sensors. Wind speed and direction are determined with                           enced by its internal structure.




Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest                                                                          379­­­­
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
     R. D. Lorenz et al.

           10,000                                                                                   a possible seasonal PBL,29 and
                           Induced
         8,000             Profile                                                                  it is this quantity that appar-
                           Body                                                                     ently controls the spacing of

    Power (w)
         6,000             Total aerodynamic
                           Net draw                                                                 dunes on Earth and Titan.28
         4,000                                                                                      Although vertical ascent
         2,000                                                                                      is possible, vertical descent
                                                                                                    is not (except at very low
              00     2        4      6        8       10       12     14       16      18      20   speeds, as for landing) since
                                                Airspeed (m/s)                                      the vortex ring state, wherein
                                                                                                    the vehicle falls through its
     Figure 4. Rotorcraft power curve for a representative vehicle mass of 420 kg on Titan. The     own downwash, creating an
     induced power required for rotor thrust falls toward higher speed, whereas the body drag       unstable condition, must be
     increases quadratically and eventually dominates. These competing factors define the maxi-     avoided. Descending verti-
     mum endurance speed (the minimum in the curve ~8 m/s) and the maximum-range speed
                                                                                                    cally at very low speeds would
     (where the tangent to the curve passes through the origin, corresponding to ~10 m/s). Titan’s
                                                                                                    also be very energy inefficient.
     dense atmosphere and low gravity means that the flight power for a given mass is a factor of
                                                                                                    Nominally, then, profiling
     about 40 times lower than on Earth.
                                                                                                    flights30,31 would be performed
                                                                                                    with normal forward motion,
     is this section aerodynamically efficient, it is also very          ascending or descending at about 20° to the horizon-
     tolerant of surface roughening (typically, in the case of           tal. These flights could be performed during traverses to
     wind turbines, due to insect impingement), making it a              new locations, or if a local vertical profile with minimal
     robust choice for Titan.                                            horizontal displacement were desired, a spiral ascent and
         The low temperature also means that the speed of                descent could be executed with return to the original
     sound26 in Titan’s atmosphere is low (~194 m/s versus               landing site.
     340 m/s on Earth). This could be a factor for large or fast-            Titan’s near-surface winds are predicted by global cir-
     rotating propellers in that severe performance loss occurs          culation models (GCMs) to be only 1–2 m/s maximum31
     as the tip Mach number approaches unity. In practice, a             (about the same as those measured by Doppler tracking
     tip Mach number of 0.4 is not a strong design factor.               of the Huygens probe), and, thus, the 10-m/s flight tran-
         An informal guide to determining the vehicle capa-              sit speed means that wind effects on range are minor.
     bility in early development was the specification that it
     should offer revolutionary science mobility to access a
     variety of geological terrains, being able to fly, in one           SCIENCE MISSION PROFILE
     hop, farther than any Mars rover has driven in a decade                 Titan’s thick, extended atmosphere in fact allows a
     (i.e., about 40 km). Flight performance analysis14 sug-             rather wide corridor of entry flight-path angle (Huygens
     gested that the maximum-range speed (Fig. 4) would be               entered at –65°), making a rather wide annulus of target
     about 10 m/s, and that flight power for a representative            possibilities, depending on the direction of arrival. Aero-
     420-kg vehicle at this speed would be a little over 2 kW.           thermodynamic considerations weakly favor arrival on
     A 30-kg battery at 100 Wh/kg could theoretically permit             Titan’s trailing side (Titan is tidally locked to Saturn)
     flight for 2 h and achieve some 60 km in range. In prac-            to minimize the entry speed and, thus, heat loads and
     tice, battery performance would be heavily margined for             deceleration.
     safety and performance would be lower. Flight power                     Arrival at Titan in the mid-2030s with DTE com-
     scales roughly as mass^1.5, so a more massive vehicle               munication suggests a low-latitude landing site. This
     would have lower endurance or would require a larger                requirement means a similar location and season to
     battery. Although the vehicle configuration is designed             the Huygens descent in 2005, so the wind profile and
     overall as a planetary lander with a somewhat boxy                  turbulence characteristics measured by the Huygens
     appearance, some streamlining is implemented (e.g.,                 probe32,33 are directly relevant. Furthermore, the sand
     a rounded nose and fairings around the skid-leg drill               seas34 that girdle Titan’s equator are both scientifically
     mechanisms) to minimize aerodynamic drag in flight.                 attractive and favorable in terms of terrain characteris-
     For obvious reasons, the HGA is stowed during flight.               tics for landing safety—indeed, it was for these reasons
         In addition to horizontal mobility, there is science            that the 2007 Flagship Study identified these dune fields
     value in achieving altitude. Of particular interest is              as the preferred initial target landing area.
     the possibility of profiling the planetary boundary layer               The radar characteristics of Titan’s dune fields35 are
     (PBL) via ascent to 500 m to 4 km altitude. The diur-               such that there is relatively little small-scale roughness.
     nal PBL thickness was measured during the Huygens                   Various methods to recover large-scale topography (altim-
     descent to be ~300 m high,27 although a possible fea-               etry, stereo imaging, and radarclinometry) suggest that
           28
     ture at ~3 km has been identified and attributed to                 Titan’s dunes may be up to 150 m high with area-averaged


380­­­­                                                        Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                                                            Dragonfly: A Rotorcraft Lander Concept for Scientific Exploration at Titan




Figure 5. Initial descent. After release from the entry system and parachute, the vehicle can traverse many kilometers at low altitude
using sensors to identify the safest landing site. The schematic is shown against an aerial image of the Namib sand sea, a geomorpho-
logical analog of the Titan landing site, with ~100-m-high dunes spaced by several kilometers.

slopes of about 5°.36 Terrestrial analogs, for example the                           the performance of various sensors—for example, an
Namib sand sea in southern Africa,37 have linear dunes                               initial hop may be made using inertial guidance alone,
of the same morphology and spacing (3–4 km) and height                               whereas later flights use optical navigation only after the
with flat inter-dune areas: analysis of digital elevation                            quality of in-flight imaging and the abundance of suit-
models [e.g., the Advanced Spaceborne Thermal Emis-                                  able landmarks on Titan have been verified.
sion and Reflection Radiometer (ASTER) Global Digi-                                     If the Titan terrain is as benign as the Namib analog
tal Elevation Model (GDEM), with 30-m postings] shows                                suggests, safe landing zones can be more or less guaran-
that at this scale some 50% of the area has a slope of 1° or                         teed between the dunes, and the full flight range of the
less, and 95% has a slope less than 6°. For a vehicle able                           vehicle can be exploited. However, a more conservative
to tolerate modest slopes (e.g., 10°), there are certain to                          posture is as follows, based on a one-way flight range R
be ample locations that permit safe landing. In contrast                             (which itself will be a healthy margin beneath the actual
to conventional planetary landers with rocket propulsion,                            vehicle capability):
which have limited divert capability, on Titan a rotor-                              1. A second landing zone (B) is identified by ground
craft lander on initial descent has sufficient endurance                                analysis of reconnaissance imaging, a distance R/3
to scan a swath of many kilometers of terrain and then                                  or less away from the initial landing site A.
backtrack to the most favorable location.
    Once safe landing on arrival is achieved, the rotorcraft                         2. The vehicle makes a sortie over this zone using its
mobility capabilities can be exercised progressively—                                   sensors (lidar for terrain roughness, imaging, etc.)
for example, first making a brief hop for a few seconds                                 and returns to the original landing site (A).
within the immediate vicinity of the landing site where
the terrain will be known from panoramic and/or                                      3. Analysis on the ground of the sensor data confirms
descent imaging. Depending on the heterogeneity of the                                  one or more safe sites within zone B (or if no satisfac-
surface (e.g., patches of sand), a small displacement of a                              tory site is found, return to step 1).
few meters or tens of meters may enable the sampling of                              4. A candidate third landing zone (C) is identified in
different materials.                                                                    reconnaissance imaging, a distance 2R/3 away from A.
    Then, flights of progressively increasing duration,
range, and/or height can be made, returning to the origi-                            5. The vehicle makes a sensing sortie over (C) but
nal, known-safe, landing site. These flights can assess                                 lands at (B).


Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest                                                                          381­­­­
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
     R. D. Lorenz et al.


                     Cruise at 500 m above takeoff                                                                       tery is large enough to cap-
                                                                                                                           Long-range
            500                                                                                                          ture the full MMRTG output)
                                                                                                                      visual reconnaissance
                                                                                                                         excess energy is available.
                                                                                                                         Other nighttime scientific
      Altitude (m)
                                                                                                                         activities include seismological
                    Energy-optimal climb                  Landing at           Survey prospective
                                                                                                                         and meteorological monitoring
     100      300
                            at 20º                    pre-surveyed site        future landing site
                                                                                  C                                      and local (e.g., microscopic)
                                                                          Representative  dune   topography              imaging using LED illumina-
                A      Vertical takeoff         B
                                                                                                                         tors as flown on Phoenix and
            –100           to 50 m        Regional Cassini SAR topo data
                0                       5                     10                     15                        20
                                                                                                                         Curiosity (e.g., Ref. 38). These
                                                       Distance (km)                                                     illuminators would permit
                                                                                                                         better color discrimination of
     Figure 6. ”Leapfrog” reconnaissance and survey strategy enables potential landing sites to                          Titan surface materials (since
     be fully validated with sensor data and ground analysis before being committed to. Distance                         the daytime illumination, fil-
     shown is example only—actual performance may be much better.                                                        tered by the thick atmospheric
                                                                                                                         haze, is predominantly of red
         In this way, the mission need not commit to landing                         light) and could use UV illumination to help iden-
     sites that have not first been assessed to be safe. This                        tify surface organic material via fluorescence,39 which
     conservative approach, while taking longer to achieve                           is common in the polycyclic aromatic hydrocarbons
     a given multi-hop traverse range, enables the contem-                           expected in the dune sands.
     plation of much rougher terrains that may be associated                             If a site proves to be of interest, the vehicle (better
     with more appealing scientific targets (e.g., cryovolcanic                      thought of as a relocatable lander than an aircraft) can
     features or impact melt sheets where liquid water may                           remain at a given location for as long as desired, per-
     have interacted with organics on Titan).                                        haps performing more extensive imaging studies with
         At each new landing site, the HGA is unstowed and                           its panoramic cameras or sampling at different depths.
     downlink begins. Priority data might include flight per-                        It could also “shuffle” distances of a few meters to repo-
     formance information and aerial imaging of the land-                            sition the skids/drills or to obtain a different camera
     ing site to confirm its exact location in maps made                             view. Observing the methane humidity over one or
     from prior reconnaissance. A quick-look site assessment                         more Titan diurnal periods would inform the extent to
     would use thermal measurements on the landing skids to                          which methane moisture is exchanged with the surface
     estimate the surface texture (e.g., solid versus granular,                      (an analysis analogous to that performed by Curiosity
     damp versus dry); dielectric constant obtained by mea-                          for water vapor on Mars40). Note that although Drag-
     suring the mutual impedance between electrodes on the                           onfly lacks a robotic arm, it can nonetheless manip-
     skids would similarly constrain the physical character of                       ulate surface materials to understand their physical
     the surface material. These measurements would take                             character. One example is that the seismometer can
     only seconds to minutes. Over a
     period of a few hours, the neutron-




                                                     Battery charge (%, blue); Earth elevation (deg., red)
                                                         100
     activated gamma-ray spectrometer                                                       Flight of ~1 h (with communications                     Battery
                                                                                            before and after) uses significant                      recharged
     would determine the bulk ele-                                                          battery energy                                          for next
     mental composition of the land-                       80                                                                                       Tsol
     ing site, allowing identification
                                                                                                                                     Occasional
     among a number of basic expected                          Downlink
                                                           60 science
                                                                                                     Periodic
                                                                                                                                     nighttime
                                                                                                     science
     surface types (e.g., organic dune                         data                                  activities and
                                                                                                                                     science
     sand, solid water ice, and frozen                                                                                               activity
                                                                                                     downlinks
     ammonia-hydrate).                                     40
         Armed with this information,
     and with imaging to characterize                      20
     the geological setting, the science
     team on the ground might elect                                    Titan daytime—Earth and Sun                      Titan nighttime (~192 h)
                                                            0                 visible (~192 h)
     to acquire a surface sample with
                                                                 0                          5                           10                       15
     one or the other drills and ana-                                                                           Day
     lyze it with the mass spectrometer.
     Drilling and sample analysis are Figure 7. Energy management and communication concept of operations. MMRTG contin-
     relatively energy-intensive tasks, uously recharges the battery, but downlink and especially flight demand significant energy.
     which might be deferred into the Activities can be paced to match MMRTG in situ capability while maintaining healthy mar-
     Titan night when (unless the bat- gins on the battery state of charge.


382­­­­                                                                                                      Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                                                             Dragonfly: A Rotorcraft Lander Concept for Scientific Exploration at Titan

observe the noise transmitted through the ground                                     REFERENCES
during drilling, diagnostic of the mechanical proper-                                 1Lorenz, R. D., “The Exploration of Titan,” Johns Hopkins APL Tech.

ties of the regolith and possibly indicating near-surface                              Dig. 27(2), 133–144 (2006).
                                                                                      2Lorenz, R., and Sotin, C., “The Moon That Would Be a Planet,” Sci-
layering. Another example is that one or more rotors
                                                                                       entific American 302(3), 36–43 (2010).
can be spun (at progressively higher speeds) to induce                                3Lorenz, R. D., and Mitton, J., Titan Unveiled, Princeton University

a known downwash on the surface material, and the                                      Press, Princeton, NJ (2010).
                                                                                      4Chyba, C. F., and Hand, K. P., “Astrobiology: The Study of the Living
speed at which sand grains begin to move (indicated
                                                                                       Universe,” Annu. Rev. Astron. Astrophys. 43, 31–74 (2005).
either by imaging or electric field measurements) can                                 5Raulin, F., McKay, C., Lunine, J., and Owen, T., “Titan’s Astrobi-

thereby be determined. This “saltation threshold” is a                                 ology,” Titan from Cassini-Huygens, R. Brown, J. P. Lebreton, and
key parameter in interpreting the large-scale morphol-                                 J. Waite, (eds.), Netherlands, Springer, pp. 215–233 (2010).
                                                                                      6Stofan, E., Lorenz, R., Lunine, J., Bierhaus, E., Clark, B., et al.,
ogy and orientation of Titan’s dunes in global circu-                                  “TiME—The Titan Mare Explorer,” in Proc. IEEE Aerospace Conf.,
lation models.31,41 There are indications that, as on                                  Big Sky, MT, paper 2434 (2013).
                                                                                      7Lorenz, R., and Mann, J., “Seakeeping on Ligeia Mare: Dynamic
Earth, since large dunes take tens of thousands of years
                                                                                       Response of a Floating Capsule to Waves on the Hydrocarbon Seas
to form or reorient, the dune pattern carries a memory                                 of Saturn’s Moon Titan,” Johns Hopkins APL Tech. Dig. 33(2), 82–94
of past climate;42 models suggest that astronomical                                    (2015).
                                                                                      8Lorenz, R. D., and Newman, C. E., “Twilight on Ligeia: Implications
changes (Croll–Milankovitch cycles, similar to those
                                                                                       of Communications Geometry and Seasonal Winds for Exploring
on Earth and Mars) may alter Titan’s wind patterns                                     Titan’s Seas 2020–2040,” Adv. Space Res. 56(1), 190–204 (2015).
and indeed the geographical distribution of its surface                               9Lockwood, M. K., Leary, J. C., Lorenz, R., Waite, H., Reh, K., et al.,

liquids. Decoding the dune pattern, however, requires                                  “Titan Explorer,” in Proc. AIAA/AAS Astrodynamics Specialist Conf.,
                                                                                       Honolulu, HI, paper AIAA-2008-7071 (2008).
good knowledge of the saltation threshold (estimated                                 10Leary, J., Jones, C., Lorenz, R., Strain, R. D., and Waite, J. H., Titan
to be around 1 m/s,43 but laboratory measurements44                                    Explorer NASA Flagship Mission Study, JHU/APL, Laurel, MD
on Earth are limited in their capability to replicate                                  (Aug 2007).
                                                                                     11Lorenz, R. D., “Titan Here We Come,” New Scientist 2247, pp. 24–27
Titan conditions, to say nothing of our ignorance of                                   (15 July 2000).
the exact sand composition and the possible role of tri-                             12Lorenz, R. D., “Post-Cassini Exploration of Titan: Science Rationale

boelectric charging45).                                                                and Mission Concepts,” J. Br. Interplanet. Soc. 53, 218–234 (2000).
                                                                                     13Lorenz, R. D., “Flight Power Scaling of Airships, Airplanes and
    At any given landing site, then, there is scope for rich                           Helicopters: Application to Planetary Exploration,” J. Aircraft 38(2),
scientific investigation in a number of disciplines. This                              208–214 (2001).
                                                                                     14Langelaan, J., Schmitz, S., Palacios, J., and Lorenz, R., “Energetics of
scientific potential is multiplied by the dozens of possible
                                                                                       Rotary-Wing Exploration of Titan,” in Proc. IEEE Aerospace Conf.,
landing sites that could be visited in a mission lasting a                             Big Sky, MT, pp. 1–11 (2017).
couple of years or more. The output from an MMRTG                                    15Zacny, K., Betts, B., Hedlund, M., Long, P., Gramlich, M., et al.,

degrades slowly, and there are no major consumables on                                 “PlanetVac: Pneumatic Regolith Sampling System,” in Proc. IEEE
                                                                                       Aerospace Conf., Big Sky, MT, pp. 1–8 (2014).
the vehicle, so the surface mission duration is not heav-                            16McKay, C. P., and Smith, H. D., “Possibilities for Methanogenic Life
ily constrained.                                                                       in Liquid Methane on the Surface of Titan,” Icarus 178(1), 274–276
                                                                                       (2005).
                                                                                     17Lawrence, D., Burks, M. T., Do, D., Fix, S., Goldsten, J., et al., “The
                                                                                       GeMini Plus High-Purity Ge Gamma-Ray Spectrometer: Instrument
CONCLUSIONS                                                                            Overview and Science Applications,” in Proc. Lunar and Planetary
                                                                                       Science Conf., Houston, TX, abstract 2234 (2017).
   NASA is presently considering the Dragonfly con-                                  18Neish, C. D., Somogyi, A., Imanaka, H., Lunine, J. I., and
cept, among many other proposals for missions to Venus,                                Smith, M. A., “Rate Measurements of the Hydrolysis of Complex
Titan, Enceladus, comets, and other targets. The authors                               Organic Macromolecules in Cold Aqueous Solutions: Implications
                                                                                       for Prebiotic Chemistry on the Early Earth and Titan,” Astrobiol. 8(2),
hope it is selected in late 2017 for a Phase A study and                               273–287 (2008).
ultimately for flight. Regardless of the outcome of the                              19Neish, C. D., Somogyi, A., and Smith, M., “Titan’s Primordial Soup:

New Frontiers 4 solicitation, however, Dragonfly has                                   Formation of Amino Acids via Low-Temperature Hydrolysis of Tho-
                                                                                       lins,” Astrobiol. 10(3), 337–347 (2010).
introduced a revolutionary new paradigm in planetary                                 20McKay, C. P., Pollack, J. B., and Courtin, R., “The Greenhouse and
exploration by demonstrating a detailed implementation                                 Antigreenhouse Effects on Titan,” Science 253(5024), 1118–1121 (1991).
                                                                                     21Lorenz, R., “Energy Cost of Acquiring and Transmitting Science Data
proposal for unparalleled regional mobility. Having laid
                                                                                       on Deep Space Missions,” J. Spacecr. Rockets 52(6), 1691–1695 (2015).
out this concept, the authors predict that henceforth it                             22Calder, W. A., III, Size, Function, and Life History, Mineola, NY, Dover
may be difficult to imagine a Titan lander mission that                                Publications (1996).
                                                                                     23Lorenz, R. D., NASA/ESA/ASI Cassini-Huygens Owners Workshop
does not exploit this capability.
                                                                                       Manual, Haynes, Somerset, UK (Apr 2017).
                                                                                     24Lee, Y., and Bairstow, B., Radioisotope Power Systems Reference Book
ACKNOWLEDGMENTS: The development of the New Fron-                                      for Mission Designers and Planners, JPL Publication 15-6, JPL, Pasa-
tiers 4 Dragonfly mission from its initial conception                                  dena, CA (2015).
                                                                                     25Zakrajsek, J. F., Woerner, D. F., Cairns-Gallimore, D., Johnson, S. G.,
(in a dinner conversation between Jason Barnes and                                     and Qualls, L., “NASA’s Radioisotope Power Systems Planning and
Ralph Lorenz) into a detailed mission proposal in only                                 Potential Future Systems Overview,” in Proc. IEEE Aerospace Conf.,
15 months was only possible through the imaginative                                    Big Sky, MT, pp. 1–10 (2016).
                                                                                     26Hagermann, A., Rosenberg, P. D., Towner, M. C., Garry, J. R. C.,
and diligent efforts of dozens of talented specialists at                              Svedhem, H., et al., “Speed of Sound Measurements and the Methane
APL and its partner institutions.                                                      Abundance in Titan’s Atmosphere,” Icarus 189(2), 538–543 (2007).



Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest                                                                           383­­­­
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
     R. D. Lorenz et al.

     27Tokano,    T., Ferri, F., Colombatti, G., Mäkinen, T., and Fulchi-           40Savijärvi, H., Harri, A. M., and Kemppinen, O., “The Diurnal Water
       gnoni, M., “Titan’s Planetary Boundary Layer Structure at the Huy-             Cycle at Curiosity: Role of Exchange with the Regolith,” Icarus 265,
       gens Landing Site,” J. Geophys. Res. 111(E8), E08007 (2006).                   63–69 (2016).
     28Lorenz, R. D., Claudin, P., Radebaugh, J., Tokano, T., and                   41Lucas, A., Rodriguez, S., Narteau, C., Charnay, B., Pont, S. C., et
       Andreotti, B., “A 3km Boundary Layer on Titan Indicated by Dune                al., “Growth Mechanisms and Dune Orientation on Titan,” Geophys.
       Spacing and Huygens Data,” Icarus 205(2), 719–721 (2010).                      Res. Lett. 41(17), 6093–6100 (2014).
     29Charnay, B., and Lebonnois, S., “Two Boundary Layers in Titan’s              42Mitchell, J. L., and Lora, J. M., “The Climate of Titan,” Annu. Rev.
       Lower Troposphere Inferred from a Climate Model,” Nat. Geosci. 5(2),           Earth Planet. Sci. 44(1), 353–380 (2016).
       106–109 (2012).                                                              43Lorenz, R. D., “Physics of Saltation and Sand Transport on Titan: A
     30Chiba, O., Kobayashi, F., Naito, G. I., and Sassa, K., “Helicopter             Brief Review,” Icarus 230, 162–167 (2014).
       Observations of the Sea Breeze over a Coastal Area,” J. Appl. Meteo-         44Burr, D. M., Bridges, N. T., Marshall, J. R., Smith, J. K., White B. R.,
       rol. 38(4), 481–492 (1999).                                                    and Emery, J. P., “Higher-Than-Predicted Saltation Threshold Wind
     31Tokano, T., “Relevance of Fast Westerlies at Equinox for the Eastward          Speeds on Titan,” Nature 517(7532), 60–63 (2015).
       Elongation of Titan’s Dunes,” Aeolian Res. 2(2), 113–127 (2010).             45Mendez-Harper, J., McDonald, G. D., Dufek, J., Malaska, M. J.,
     32Folkner, W. M., Asmar, S. W., Border, J. S., Franklin, G. W.,                  Burr, D. M., et al., “Electrification of Sand on Titan and Its Influence
       Finley, S. G., et al., “Winds on Titan from Ground-Based Tracking of           on Sediment Transport,” Nat. Geosci. 10(4), 260–265 (2017).
       the Huygens Probe,” J. Geophys. Res. 111(E7), E07S02 (2006).                 46Lorenz, R. D., “A Review of Titan Mission Studies,” J. Br. Interplanet.
     33Karkoschka, E., “Titan’s Meridional Wind Profile and Huygens’ Ori-             Soc. 62, 162–174 (2009).
       entation and Swing Inferred from the Geometry of DISR Imaging,”              47Zubrin, R., “Missions to Mars and the Moons of Jupiter and Saturn
       Icarus 270, 326–338 (2016).                                                    Utilizing Nuclear Thermal Rockets with Indigenous Propellants,” in
     34Lorenz, R. D., Wall, S., Radebaugh, J., Boubin, G., Reffet, E., et al.,        Proc. 28th Aerospace Sciences Meeting, Reno, NV, paper AIAA-90-
       “The Sand Seas of Titan: Cassini RADAR Observations of Longitu-                0002 (Jan 1990).
       dinal Dunes,” Science 312(5774), 724–727 (2006).                             48Chyba, C., McKinnon, W. B., Coustenis, A., Johnson, R. E.,
     35Paillou, P., Bernard, D., Radebaugh, J., Lorenz, R., Le Gall, A., and          Kovach, R. L., et al., “Europa and Titan: Preliminary Recommenda-
       Farr, T., “Modeling the SAR Backscatter of Linear Dunes on Earth               tions of the Campaign Science Working Group on Prebiotic Chemis-
       and Titan,” Icarus 230, 208–214 (2014).                                        try in the Outer Solar System,” in Proc. 30th Annual Lunar and Plan-
     36Neish, C. D., Lorenz, R. D., Kirk, R. L., and Wye, L. C., “Radarcli-           etary Science Conf., Houston, TX, abstract 1537 (1999).
       nometry of the Sand Seas of Africa’s Namibia and Saturn’s Moon               49Elliott, J. O., Reh, K., and Spilker, T., “Titan Exploration Using a
       Titan,” Icarus 208, 385–394 (2010).                                            Radioisotopically-Heated Montgolfiere Balloon,” in AIP Conference
     37Radebaugh, J., Lorenz, R., Farr, T., Paillou, P., Savage, C., and Spen-        Proceedings, M. S. El-Genk (ed.), Vol. 880, No. 1, pp. 372–379, AIP,
       cer, C., “Linear Dunes on Titan and Earth: Initial Remote Sensing              College Park, MD (Jan 2007).
       Comparisons,” Geomorphol. 121(1), 122–132 (2010).                            50Reh, K. R., and Elliott, J., “Preparing for a Future in Situ Mission to
     38Goetz, W., Hecht, M. H., Hviid, S. F., Madsen, M. B., Pike, W. T.,             Titan,” in Proc. 2010 IEEE Aerospace Conf., Big Sky, MT, pp. 1–9 (2010).
       et al., “Search for Ultraviolet Luminescence of Soil Particles at the        51Coustenis, A., Atreya, S. K., Balint, T., Brown, R. H., Dough-
       Phoenix Landing Site, Mars,” Planet. Space Sci. 70(1), 134–147 (2012).         erty, M. K., et al., “TandEM: Titan and Enceladus Mission,” Exp.
     39Hodyss, R., McDonald, G., Sarker, N., Smith, M. A., Beau-                      Astron. 23(3), 893–946 (2009).
       champ, P. M., and Beauchamp, J. L., “Fluorescence Spectra of Titan           52Barnes, J., Lemke, L., Foch, R., McKay, C. P., Beyer, R. A., et al.,
       Tholins: In-Situ Detection of Astrobiologically Interesting Areas on           “AVIATR – Aerial Vehicle for In-Situ and Airborne Titan Recon-
       Titan’s Surface,” Icarus 171(2), 525–530 (2004).                               naissance,” Exp. Astron. 33(1), 55–127 (2012).




                               Ralph D. Lorenz, Space Exploration                                          Elizabeth P. Turtle, Space Exploration
                               Sector, Johns Hopkins University Applied                                    Sector, Johns Hopkins University Applied
                               Physics Laboratory, Laurel, MD                                              Physics Laboratory, Laurel, MD
                               Ralph D. Lorenz is a planetary scientist                                   Elizabeth P. Turtle, principal investigator
                               in APL’s Space Exploration Sector and a                                    of the Dragonfly mission, is a Principal
                               member of the Principal Professional Staff.                                Professional Staff member in APL’s Space
                               He is project scientist for the Dragonfly                                  Exploration Sector. She has a B.S. in phys-
                               mission. He has a B.Eng. in aerospace                                      ics from the Massachusetts Institute of
          systems engineering from the University of Southampton                    Technology and a Ph.D. in planetary sciences from the Univer-
          and a Ph.D. from the University of Kent. He has worked on                 sity of Arizona. She is the principal investigator of the Europa
          several missions, including roles as a co-investigator on the             Imaging System (EIS) on Europa Clipper, co-investigator for
          Huygens Surface Science Package (SSP); as a member of                     the Lunar Reconnaissance Orbiter Camera (LROC), and a
          teams for Cassini, the New Millennium DS-2, and the Mars                  member of several teams for Cassini. She was principal investi-
          Polar Lander; as a NASA participating scientist for JAXA                  gator on “Topographic and Reconnaissance Imaging for Europa
          Akatsuki (Venus Climate Orbiter); as a collaborator on the                Exploration” under the Instrument Concepts for Europa Explo-
          InSight SEIS seismometer investigation; and as project scien-             ration (ICEE) Program and also worked on Galileo. She has
          tist and physical properties instrument lead for the Titan Mare           been awarded several NASA Group Achievement Awards and
          Explorer (TiME) Phase-A study. He has been awarded six                    has been named to various NASA, NRC, American Astro-
          NASA Group Achievement Awards and served on the NRC                       nomical Society, and other groups and committees. Elizabeth
          Committee on Origins and Evolution of Life (COEL). He is                  has published many scholarly articles about planetary surface
          the co-author of several books and has published more than                features, subsurface structures, impact craters, and geologic
          250 articles in peer-reviewed journals. His e-mail address is             processes, as well as about planetary imaging and mapping. Her
          ralph.lorenz@jhuapl.edu.                                                  e-mail address is elizabeth.turtle@jhuapl.edu.




384­­­­                                                                   Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                                                            Dragonfly: A Rotorcraft Lander Concept for Scientific Exploration at Titan



                            Jason W. Barnes, Department of Physics,                                        Kenneth E. Hibbard, Space Exploration
                            University of Idaho, Moscow, ID                                                Sector, Johns Hopkins University Applied
                           Jason W. Barnes is deputy principal inves-                                      Physics Laboratory, Laurel, MD
                           tigator of the Dragonfly mission and an                                         Kenneth E. Hibbard is a Principal Profes-
                           associate professor at the University of                                        sional Staff member and group supervisor
                           Idaho. He has a B.S. in astronomy from                                          in APL’s Space Exploration Sector. He has
                           Caltech and a Ph.D. in planetary science                                        a B.S. in aerospace engineering from Penn
                           from the University of Arizona. He studies                                      State and an M.S. in systems engineer-
     the physics of planets and planetary systems and uses NASA                      ing from Johns Hopkins University. He has worked in vari-
     spacecraft data to study planets that orbit stars other than the                ous capacities on many missions and concept developments,
     Sun (extrasolar planets) and the composition and nature of the                  including Space Layer Experiment (SLX); Precision Tracking
     surface of Saturn’s moon Titan. Jason advocated heavier-than-                   Space System (PTSS); Titan Mare Explorer (TiME); Io Vol-
     air flight on Titan, leading the AVIATR concept study. He has                   cano Observer (IVO); Jupiter Europa Orbiter (JEO); MErcury
     published numerous articles in refereed journals. His e-mail                    Surface, Space ENvironment, GEochemistry, and Ranging
     address is jwbarnes@uidaho.edu.                                                 (MESSENGER); Solar & Heliospheric Observatory (SOHO);
                                                                                     Advanced Composition Explorer (ACE); and Swift. He has
                                                                                     received several NASA and International Academy of Astro-
                                                                                     nautics awards and has contributed numerous articles to the
                            Melissa G. Trainer, Planetary Environ-                   scientific literature. His e-mail address is kenneth.hibbard@
                            ments Laboratory, NASA Goddard Space                     jhuapl.edu.
                            Flight Center, Greenbelt, MD
                         Melissa G. Trainer, deputy principal investi-
                         gator of the Dragonfly mission, is a research
                         space scientist in the Planetary Environ-                   Colin Z. Sheldon, Space Exploration Sector, Johns Hopkins
                         ments Laboratory at NASA Goddard                            University Applied Physics Laboratory, Laurel, MD
                         Space Flight Center. She has a B.A. in                      Colin Z. Sheldon is a Senior Professional Staff member and tele-
     chemistry from Franklin and Marshall College and a Ph.D. in                     communications systems engineer and in APL’s Space Explora-
     chemistry from the University of Colorado. She is a science                     tion Sector. He has a B.S. in electrical engineering from Brown
     team member on the Sample Analysis at Mars (SAM) experi-                        University and an M.S. and a Ph.D. in electrical and computer
     ment aboard the Mars Science Laboratory Mission’s Curiosity                     engineering from the University of California, Santa Barbara.
     Rover; co-investigator and deputy instrument scientist for the                  He is the cognizant engineer and principal investigator for the
     cryogenic sampling inlet and Neutral Mass Spectrometer on                       Europa Lander GaN solid-state power amplifier technology
     the Titan Mare Explorer (TiME); and co-investigator on the                      development effort. He has presented at many technical con-
     Discovery Candidate Deep Atmosphere Venus Investigation of                      ferences and has contributed articles to peer-reviewed journals.
     Noble gases, Chemistry, and Imaging (DAVINCI) mission. She                      His e-mail address is colin.sheldon@jhuapl.edu.
     has been recognized with numerous NASA awards. She is a
     reviewer for several academic journals and NASA grant pro-
     grams, has organized sessions and conferences, and is a member
     of several professional societies. Melissa has published many                                         Kris Zacny, Vice President and Director of
     articles in refereed journals. Her e-mail address is melissa.                                         Exploration Technology, Honeybee Robot-
     trainer@nasa.gov.                                                                                     ics, Pasadena, CA
                                                                                                         Kris Zacny earned a B.Sc. in mechanical
                                                                                                         engineering from University of Cape Town
                                                                                                         and an M.E. in petroleum engineering and
                            Douglas S. Adams, Space Exploration                                          a Ph.D. in Mars drilling from the Univer-
                            Sector, Johns Hopkins University Applied                                     sity of California, Berkeley. He focuses on
                            Physics Laboratory, Laurel, MD                           robotic space drilling, sample acquisition, transfer and pro-
                        Douglas S. Adams is a Senior Professional                    cessing technologies, and geotechnical systems for mining
                        Staff member in APL’s Space Exploration                      applications. He has been a principal and co-investigator of
                        Sector. He has a B.S. in aeronautical and                    over 50 NASA- and Department of Defense-funded projects
                        astronautical engineering, an M.S. in aero-                  and participated in drilling expeditions in Antarctica, the
                        nautics and astronautics, and a Ph.D. in                     Atacama Desert, Mauna Kea, the Mojave Desert, and the
     aeronautics and astronautics, all from Purdue University. He                    Arctic. He has over 100 publications to his name, including
     has been an engineer for several instruments and missions,                      an edited book, Drilling in Extreme Environments: Penetration
     including the Low Density Supersonic Decelerator (LDSD) –                       and Sampling on Earth and Other Planets. Kris has earned more
     Parachute Deployment Device (PDD), the Soil Moisture                            than 15 NASA New Technology Records and three NASA
     Active Passive (SMAP), the Mars Science Laboratory (MSL),                       Group Achievement Awards. His e-mail address is zacny@
     the Phoenix Mars Scout, and the Mars Exploration Rover                          honeybeerobotics.com.
     (MER). Douglas has been awarded several team and NASA
     Achievement awards, and he has published several articles in
     peer-reviewed journals. His e-mail address is douglas.adams@
     jhuapl.edu.




Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest                                                                          385­­­­
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
     R. D. Lorenz et al.



                              Patrick N. Peplowski, Space Exploration                                  Timothy G. McGee, Argo AI, Pittsburgh,
                              Sector, Johns Hopkins University Applied                                 PA
                              Physics Laboratory, Laurel, MD                                         Timothy G. McGee is a senior engineer
                             Patrick N. Peplowski is a Senior Profes-                                with Argo AI. He has a B.S. in mechani-
                             sional Staff member, nuclear physicist, and                             cal engineering from the University of
                             planetary scientist in APL’s Space Explora-                             Illinois at Urbana-Champaign, an M.S. in
                             tion Sector. He has a B.S. in physics from                              mechanical engineering from the Univer-
                             the University of Washington and an M.S.                                sity of California, Berkeley, and a Ph.D. in
          and a Ph.D. in physics from Florida State University. He is a         mechanical engineering/control systems from the University
          co-investigator and instrument scientist for Psyche; a member         of California, Berkeley. He was the lead guidance, navigation,
          of the science team for Dawn at Vesta; an instrument scientist        and control engineer for the Robotic Lunar Lander Devel-
          for the Gamma-Ray and Neutron Spectrometer on MErcury                 opment Project and the Mighty Eagle Test-bed; the lead for
          Surface, Space ENvironment, GEochemistry, and Ranging                 onboard Terrain Relative Navigation (TRN) development for
          (MESSENGER); and principal investigator for NASA’s Mars               space applications at APL; and the lead engineer for the Solar
          Data Analysis, Discovery Data Analysis, and Planetary Mis-            Probe Plus (now Parker Solar Probe) solar array operations and
          sion Data Analysis Programs. He has received several achieve-         safing. He is the recipient of several NASA awards and various
          ment awards from NASA and APL and has published many                  fellowships. He has presented at many technical conferences
          peer-reviewed journal articles. His e-mail address is patrick.        and has contributed articles to peer-reviewed journals. His
          peplowski@jhuapl.edu.                                                 e-mail address is tmcgee@argo.ai.



                              David J. Lawrence, Space Exploration                                     Kristin S. Sotzen, Space Exploration
                              Sector, Johns Hopkins University Applied                                 Sector, Johns Hopkins University Applied
                              Physics Laboratory, Laurel, MD                                           Physics Laboratory, Laurel, MD
                               David J. Lawrence is a Principal Profes-                              Kristin S. Sotzen is a Senior Professional
                               sional Staff member in APL’s Space Explo-                             Staff member and project manager in APL’s
                               ration Sector. He has a B.S. in physics and                           Space Exploration Sector. She has a B.S.
                               mathematics from Texas Christian Uni-                                 in engineering physics from Embry-Riddle
                               versity and an M.A. and a Ph.D. in physics                            Aeronautical University and an M.S. in
          from Washington University in St. Louis. He is the investiga-         applied physics from Johns Hopkins University. She expects to
          tion lead for the Psyche Gamma-Ray and Neutron Spectrom-              earn a Ph.D. in earth and planetary sciences from Johns Hop-
          eter; a participating scientist and instrument scientist for the      kins University in 2020. Her recent roles include the Europa
          MErcury Surface, Space ENvironment, GEochemistry, and                 Clipper payload accommodation engineer and Titan Mare
          Ranging (MESSENGER) mission; a participating scientist for            Explorer Phase A instrument engineer. Kristin won an APL
          the Dawn mission; and principal investigator for the Depart-          Special Achievement Award for her work in 2015, and she has
          ment of Energy space-based treaty monitoring neutron sensors.         published several papers in the scientific literature. Her e-mail
          He has received awards from both NASA and APL and has                 address is kristin.sotzen@jhuapl.edu.
          published many articles in refereed journals. His e-mail address
          is david.lawrence@jhuapl.edu.

                                                                                                       Shannon M. MacKenzie, Space Explora-
                                                                                                       tion Sector, Johns Hopkins University
                              Michael A. Ravine, Advanced Projects                                     Applied Physics Laboratory, Laurel, MD
                              Manager, Malin Space Science Systems,                                Shannon M. MacKenzie is a postdoctoral
                              San Diego, CA                                                        researcher in APL’s Space Exploration
                              Michael A. Ravine is the advanced proj-                              Sector. She has a B.Sc. from the Univer-
                              ects manager at Malin Space Science                                  sity of Louisville, an M.Sc. in physics from
                              Systems. He has a B.S. in physics from                               the University of Idaho, and in 2017 was
                              Caltech, an M.Sc. in geology from Brown           awarded a Ph.D. in physics from the University of Idaho. She
                              University, and a Ph.D. in geophysics from        was a team collaborator on the Cassini mission and principal
          Scripps Institution of Oceanography. He has worked in vari-           investigator at the Jet Propulsion Laboratory Planetary Sci-
          ous capacities on numerous missions, including ExoMars Trace          ence Summer Seminar. She was a NASA Earth and Space Sci-
          Gas Orbiter Mars Atmospheric Global Imaging Experiment                ence Fellow, an Idaho Space Grant Consortium Fellow, and a
          (MAGIE); Juno Jupiter Orbiter; Mars Science Laboratory;               Goldwater Scholar. She has authored several papers published
          Semi-autonomous Rover Operations; Mars Phoenix Lander;                in peer-reviewed technical journals. Her e-mail address is
          Mars Reconnaissance Orbiter; L1: Return to Apollo; Freefall           shannon.mackenzie@jhuapl.edu.
          Interferometric Gravity Gradiometer; Mars Surveyor 2001
          and Mars Surveyor 1998; and Voyager. He has received sev-
          eral NASA Group Achievement Awards, a U.S. Navy Ant-
          arctica Service Medal, and an Exxon Teaching Fellowship. He
          has published many papers in technical journals. His e-mail
          address is ravine@msss.com.




386­­­­                                                               Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
                                                                                            Dragonfly: A Rotorcraft Lander Concept for Scientific Exploration at Titan



                            Jack W. Langelaan, Aerospace Engineer-                   Lawrence S. Wolfarth, Space Exploration Sector, Johns Hop-
                            ing Department, Penn State University,                   kins University Applied Physics Laboratory, Laurel, MD
                            University Park, PA                                      Larry Wolfarth is a parametric resource analyst in APL’s Space
                          Jack W. Langelaan is an associate profes-                  Exploration Sector. He has a B.S. and an M.A. in sociology
                          sor of aerospace engineering at Penn State                 from Indiana University Bloomington. He has more than
                          University. He has a B.S. in engineering                   30 years of experience supporting managers with the selection
                          physics from Queen’s University (Kings-                    and management of mission-critical hardware and software
                          ton, Canada), an M.S. in aeronautics and                   systems. Recent experience includes cost estimates for sev-
     astronautics from the University of Washington, and a Ph.D.                     eral NASA robotic missions as well as eight Decadal Survey
     in aeronautics and astronautics from Stanford University. He                    missions and cost, schedule, risk, and budgetary analyses for
     is the 2011 winner of the Green Flight Challenge, a NASA                        the Missile Defense Agency’s Precision Tracking and Surveil-
     Centennial Challenge focused on the problem of extreme                          lance System (PTSS). Prior to joining APL, Mr. Wolfarth led
     energy efficiency for general aviation aircraft. He has also been               investment, cost-benefit, risk, and schedule analyses for Federal
     awarded the Penn State Engineering Alumni Association Out-                      Aviation Administration (FAA) programs. His e-mail address
     standing Teaching Award, a European Space Agency award                          is lawrence.wolfarth@jhuapl.edu.
     outstanding contribution to the Huygens Probe, and a NASA
     Group Achievement Award for Cassini. He has published
     papers in refereed journals and presented at various technical
     conferences. His e-mail address is jlangelaan@psu.edu.                                                Peter D. Bedini, Space Exploration Sector,
                                                                                                           Johns Hopkins University Applied Physics
                                                                                                           Laboratory, Laurel, MD
                                                                                                         Peter Bedini is a program manager in APL’s
                            Sven Schmitz, Aerospace Engineering                                          Space Exploration Sector and a member of
                            Department, Penn State University, Uni-                                      the Principal Professional Staff. He has an
                            versity Park, PA                                                             A.B. in physics from Dartmouth College
                         Sven Schmitz is an associate professor of                                       and an M.S. in space physics from the Uni-
                         aerospace engineering at Penn State Uni-                    versity of Maryland. Peter was MErcury Surface, Space ENvi-
                         versity. He has a Dipl.Ing. in aerospace                    ronment, GEochemistry, and Ranging (MESSENGER) Project
                         engineering from RWTH Aachen (Ger-                          Manager from 2007 through the end of the first extended mis-
                         many) and a Ph.D. in mechanical and aero-                   sion in 2013. He also managed the development of the CRISM
     nautical engineering from the University of California, Davis.                  (Compact Reconnaissance Imaging Spectrometer for Mars)
     He leads graduate and postdoctoral researchers in fundamental                   instrument on NASA’s Mars Reconnaissance Orbiter and was
     computational and experimental investigations on rotary wing                    deputy project manager of the New Horizons mission to Pluto.
     aerodynamics, with a major focus on rotorcraft aeromechanics,                   At present, he is working on the Europa Lander mission study.
     rotor/blade design methodologies, rotor active control, rotor                   He was awarded the 2011 award for AIAA Engineering Man-
     hub flows, and large-eddy simulations of wind turbine wakes                     ager of the Year in the Mid-Atlantic Region; NASA Public
     and ship airwakes and their interaction with atmospheric tur-                   Service Group Achievement Awards for various missions; and
     bulence. He leads a team of researchers investigating the fun-                  several European Space Agency Achievement Awards for the
     damental fluid mechanics of rotor hub flows, supported by the                   Ulysses mission. His e-mail address is peter.bedini@jhuapl.edu.
     National Rotorcraft Technology Center. He is the recipient of
     a Penn State Engineering Alumni Association Outstanding
     Teaching Award and a Joseph L. Steger Fellowship Award. He
     has presented at several conferences and has published in vari-
     ous technical journals. His e-mail address is sus52@psu.edu.




Johns Hopkins APL Technical Digest, Volume 34, Number 3 (2018), www.jhuapl.edu/techdigest                                                                          387­­­­
```
