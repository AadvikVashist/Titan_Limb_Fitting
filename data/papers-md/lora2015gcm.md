---
citation_key: "lora2015gcm"
title: "GCM simulations of Titan’s middle and lower atmosphere and comparison to observations"
source_pdf: "data/papers/lora2015gcm.pdf"
source_pdf_sha256: "8f132d23f2b005709905f5345cf2697b99b73995161e1c3369bf77b19d27629e"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                                                                             Icarus 250 (2015) 516–528



                                                                  Contents lists available at ScienceDirect


                                                                                     Icarus
                                                   journal homepage: www.elsevier.com/locate/icarus




GCM simulations of Titan’s middle and lower atmosphere and
comparison to observations
Juan M. Lora a,⇑, Jonathan I. Lunine b, Joellen L. Russell a
a
    Department of Planetary Sciences, University of Arizona, Tucson, AZ 85721, United States
b
    Center for Radiophysics and Space Research, Cornell University, Ithaca, NY 14853, United States



a r t i c l e           i n f o                           a b s t r a c t

Article history:                                          Simulation results are presented from a new general circulation model (GCM) of Titan, the Titan Atmo-
Received 15 August 2014                                   spheric Model (TAM), which couples the Flexible Modeling System (FMS) spectral dynamical core to a
Revised 23 December 2014                                  suite of external/sub-grid-scale physics. These include a new non-gray radiative transfer module that
Accepted 23 December 2014
                                                          takes advantage of recent data from Cassini–Huygens, large-scale condensation and quasi-equilibrium
Available online 3 January 2015
                                                          moist convection schemes, a surface model with ‘‘bucket’’ hydrology, and boundary layer turbulent dif-
                                                          fusion. The model produces a realistic temperature structure from the surface to the lower mesosphere,
Keywords:
                                                          including a stratopause, as well as satisfactory superrotation. The latter is shown to depend on the
Titan, atmosphere
Titan, hydrology
                                                          dynamical core’s ability to build up angular momentum from surface torques. Simulated latitudinal tem-
Titan, clouds                                             perature contrasts are adequate, compared to observations, and polar temperature anomalies agree with
Atmospheres, dynamics                                     observations. In the lower atmosphere, the insolation distribution is shown to strongly impact turbulent
                                                          ﬂuxes, and surface heating is maximum at mid-latitudes. Surface liquids are unstable at mid- and low-
                                                          latitudes, and quickly migrate poleward. The simulated humidity proﬁle and distribution of surface tem-
                                                          peratures, compared to observations, corroborate the prevalence of dry conditions at low latitudes. Polar
                                                          cloud activity is well represented, though the observed mid-latitude clouds remain somewhat puzzling,
                                                          and some formation alternatives are suggested.
                                                                                                                           Ó 2014 Elsevier Inc. All rights reserved.




1. Introduction                                                                                  GCMs used to investigate Titan’s methane cycle in detail
                                                                                              (Mitchell et al., 2006; Schneider et al., 2012) have shown that the
   Observations of Titan since the time of the Voyager 1 ﬂyby have                            observed distribution of clouds (Rodriguez et al., 2009, 2011;
prompted the development of several general circulation models                                Brown et al., 2010; Turtle et al., 2011a) is a natural result of Titan’s
(GCMs) to study its atmosphere. The ﬁrst GCM of Titan (Hourdin                                changing seasons, and that the circulation efﬁciently transports
et al., 1995) studied the development of atmospheric superrota-                               methane poleward (Mitchell et al., 2006; Mitchell, 2012;
tion, showing relative agreement with then-current observations.                              Schneider et al., 2012), drying the equatorial regions (Mitchell,
Subsequent axisymmetric (two-dimensional) models provided a                                   2008). The use of gray radiative transfer in these models, though,
variety of additional insights into Titan’s climate processes, includ-                        results in unrealistic surface insolation distributions (Lora et al.,
ing the ﬁrst studies of the methane and ethane hydrological cycle                             2011), and precludes extension of the models to the stratosphere.
(Rannou et al., 2006), stratospheric gases (Hourdin et al., 2004),                            A variety of additional simpliﬁcations, such as prescribed sur-
and haze-dynamical feedbacks (Rannou et al., 2004; Crespin                                    face-level relative humidity and inﬁnite methane supply from
et al., 2008). Since the Cassini spacecraft’s present exploration of                          the surface, have also been employed (Rannou et al., 2006;
the Saturnian system, new GCMs developed to better take advan-                                Tokano, 2009; Mitchell et al., 2011; Mitchell, 2012), limiting those
tage of the increasing quality and quantity of data—in particular                             models’ ability to predict the distribution of liquids.
by returning to being three-dimensional—, have had success in                                    Furthermore, with the exception of the CAM Titan model
reproducing some of the observations, but have also been encum-                               (Friedson et al., 2009), which did not simulate the methane cycle
bered by a combination of numerical difﬁculties and unrealistic                               but produced a realistic temperature proﬁle, other Titan GCMs
assumptions.                                                                                  (Hourdin et al., 1995; Tokano et al., 1999; Richardson et al.,
                                                                                              2007), including those used to study stratospheric dynamics and
 ⇑ Corresponding author at: Department of Earth, Planetary, and Space Sciences,               haze (Rannou et al., 2002; Lebonnois et al., 2012a), have employed
University of California, Los Angeles, CA 90095, United States.                               versions of the radiative transfer model of McKay et al. (1989),

http://dx.doi.org/10.1016/j.icarus.2014.12.030
0019-1035/Ó 2014 Elsevier Inc. All rights reserved.
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                       J.M. Lora et al. / Icarus 250 (2015) 516–528                                               517


which works well for the troposphere and lower stratosphere, but                HITRAN line intensities. Included absorbers are CH4, C2H2, C2H4,
seems to produce unrealistic temperatures higher up including a                 C2H6, and HCN. The proﬁles of the stratospheric molecular species
cold stratosphere, a sharp increase in temperature in the upper-                are ﬁxed to observed values (Vinatier et al., 2007).
most regions, and a failure to obtain a stratopause (Lebonnois                     Shortwave haze optical parameters are those published by the
et al., 2012a).                                                                 DISR team (Tomasko et al., 2008b), thus assuming that the haze
    Separately, an important numerical problem, namely models’                  distribution is horizontally homogeneous. Values of optical depth
inability to properly attain atmospheric superrotation (Tokano                  for wavelengths larger than 1.6 lm (beyond those measured by
et al., 1999; Richardson et al., 2007; Friedson et al., 2009), has been         DISR) are extrapolated using power law ﬁts (Tomasko et al.,
widely studied (Newman et al., 2011; Lebonnois et al., 2012b) but               2008b); the haze is assumed to become more absorptive with
remains incompletely understood. Consensus on the mechanism of                  increasing wavelength beyond 1.6 lm. Haze in the thermal infra-
its maintenance and on numerical obstacles is not evident, as only              red is assumed to be perfectly absorbing, so that the single scatter-
some three-dimensional models simulate it, and only under special               ing albedo is always zero. Optical depths are calculated from
circumstances (Newman et al., 2011; Lebonnois et al., 2012a).                   volume extinction coefﬁcients determined from Cassini/CIRS data,
    In this paper, we present simulations from the Titan Atmo-                  available between wavenumbers 20–560 cm1 (Anderson and
spheric Model (TAM), a new three-dimensional Titan GCM devel-                   Samuelson, 2011) and 610–1500 cm1 (Vinatier et al., 2012). Val-
oped to alleviate some of these difﬁculties and to incorporate                  ues for wavenumbers larger than 1500 cm1 are interpolated
and study processes and phenomena being unveiled by the Cassini                 between these and the DISR results, using a power law ﬁt (note
mission in Titan’s middle and lower atmosphere. A previous lower-               that very little energy is transmitted at these wavenumbers). These
atmosphere version of this model was used to investigate Titan’s                volume extinction coefﬁcients are assumed constant between 0
recent paleoclimate (Lora et al., 2014). The model and methodol-                and 80 km, and decreasing with a scale height of 65 km above that.
ogy are described in Section 2. In Sections 3 and 4, simulations
from the middle and lower atmosphere are benchmarked against                    2.1.2. Moist processes
a variety of observational constraints (temperatures, winds,                       Methane saturation vapor pressure is calculated either over an
humidity, and cloud locations), as well as used to explore model                80/20 CH4/N2 liquid (Thompson et al., 1992) or pure methane ice
sensitivities. In Section 5, we discuss model limitations, providing            (Moses et al., 1992) depending on temperature, with the transition
groundwork for future development and studies. We summarize                     at 87 K where the vapor pressure curves intersect. The effects of
relationships between observed and modeled phenomena and con-                   ethane on the vapor pressure of methane are assumed negligible.
clude in Section 6.                                                                Two precipitation schemes are included: A large-scale conden-
                                                                                sation (LSC) scheme, which condenses any methane per grid box
                                                                                exceeding 100% relative humidity and allows it to re-evaporate
2. Model
                                                                                in underlying layers, and a quasi-equilibrium moist convection
                                                                                scheme (Frierson, 2007; O’Gorman and Schneider, 2008), where
2.1. Description of the GCM
                                                                                convectively unstable columns relax toward a moist pseudoadia-
                                                                                bat. In the latter, excess liquid falls immediately to the surface.
   The GCM, which makes use of the Geophysical Fluid Dynamics
                                                                                In both cases, whatever condensation occurs is assumed to be
Laboratory’s (GFDL) Flexible Modeling System (FMS) infrastruc-
                                                                                liquid, ignoring the 10% difference in latent heats between ice
ture, couples a physics package based on GFDL’s atmospheric com-
                                                                                and liquid, avoiding the need to model the ice–liquid transition
ponent models to the fully three-dimensional FMS spectral
                                                                                for energy balance. Furthermore, in all cases it is assumed that
dynamical core (Gordon and Stern, 1982). Here we describe the
                                                                                nucleation is always possible, ignoring detailed microphysics. The
component modules of the physics package.
                                                                                effects of clouds are neglected in the radiative transfer.

2.1.1. Radiation                                                                2.1.3. Surface
   The radiative transfer model is intended to compute accurate                    The GCM employs a soil model using 15 layers of variable thick-
radiative heating rates without approximations that compromise                  ness to 80 m depth, between which heat is transported by conduc-
their overall vertical or latitudinal distributions. Solar-wavelength           tion. The thermal properties of the soil are assumed to be those
(<4.5 lm) and thermal infrared (>4.5 lm) ﬂuxes are computed                     appropriate for the ‘‘porous icy regolith’’ of Tokano (2005). Neither
employing nongray, multiple scattering, plane-parallel two-stream               topography nor albedo variations are included. At the ground sur-
approximations from scaled extinction optical depths, single scat-              face, ﬂuxes of sensible and latent heat, radiation, and momentum
tering albedos, and asymmetry parameters (Toon et al., 1989;                    are calculated using bulk aerodynamic formulae, with drag coefﬁ-
Briegleb, 1992). Seasonal and diurnal cycles are included in the                cients from Monin–Obukhov similarity theory. Roughness length
computation of insolation.                                                      and gustiness parameters in this module are assumed to be
   Methane opacities at wavelengths short of 1.6 lm are calcu-                  0.5 cm and 0.1 m s1, respectively (Friedson et al., 2009;
lated with exponential sum ﬁts to transmissions, using DISR                     Schneider et al., 2012).
absorption coefﬁcients (Tomasko et al., 2008a) with varying col-                   A ‘‘bucket’’ model tracks the liquid content of the ground qg ,
umn abundance. The effects of methane opacity between 1.6 and
                                                                                @qg
4.5 lm are accounted for using correlated k coefﬁcients calculated                  ¼ P  E;                                                      ð1Þ
from HITRAN line intensities (Rothman et al., 2009). For the pur-               @t
poses of radiative transfer, the methane proﬁle is globally set to              where P is precipitation—resulting from moist processes—that accu-
that measured by Huygens (Niemann et al., 2005).                                mulates on the surface, and E is evaporation that removes methane
   Opacities due to CIA—which include combinations of N2, CH4                   from the surface reservoir. An availability factor parameterizes inﬁl-
and H2 pairs—are calculated from HITRAN data (Richard et al.,                   tration and sub-grid scale ponding, limiting evaporation when the
2011) with exponential sum ﬁts to pressure- and temperature-                    grid box has less than 100 kg m2 of methane and linearly decreas-
dependent transmissions. It should be noted that the mole fraction              ing to zero evaporation at zero methane. The thermal effects of
of H2 is assumed to be constant at 0.1% (Tomasko et al., 2008c).                liquid methane are not included, and surface liquid cannot move
Molecular absorption is treated with correlated k coefﬁcients from              laterally, even when multiple grid boxes are necessary to deﬁne a
temperature- and pressure-corrected (Rothman et al., 1996)                      ‘‘lake’’ or ‘‘sea.’’
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
518                                                                J.M. Lora et al. / Icarus 250 (2015) 516–528


2.1.4. Boundary layer                                                                       ﬁve. For the second, a global reservoir of four meters of methane
   Vertical diffusion in the boundary layer uses a standard K-pro-                          was imposed, with the addition of 100 m deep reservoirs at the
ﬁle scheme, wherein diffusivities of heat and moisture, K H , and                           approximate locations of Titan’s observed Ontario Lacus, Kraken
momentum, K m , are calculated as a function of height z within                             Mare, Ligeia Mare, and Punga Mare (henceforth lakes/seas simula-
the boundary layer h, and Monin–Obukhov stability functions                                 tion). This simulation was run for considerably longer (35 Titan
UH;m :                                                                                      years), as the surface reservoir only stabilized after the mid- and
          8                                                                                 low-latitude surfaces dried.
          < ðku z=UH;m Þ                           for z < hlow
K H;m ¼                                       2                                ð2Þ
          : ðku z=UH;m Þ            zhlow
                                  1  hh           for hlow 6 z < h;
                                         low                                                3. Middle atmosphere
where k is the von Karman constant, u the surface friction velocity
                                                                                               In this section, we discuss results from three simulations: the
calculated from Monin–Obukhov theory, and hlow the height of the
                                                                                            L50 case extending into the mesosphere, the L32 control case,
surface layer, assumed to be one-tenth of the boundary layer height.
                                                                                            and the L32 simulation using parameterized varying haze.
For stable or neutral conditions, the boundary layer height h is set
where the Richardson number, the ratio of potential to kinetic
energy and a measure of the importance of buoyancy, equals 1.0.                             3.1. Atmospheric temperatures
In the case of unstable conditions, h is set at the level of neutral
buoyancy for surface parcels.                                                                   Zonally averaged temperatures from the L50 simulation are
                                                                                            shown in Fig. 1. The times shown correspond to the seasonal
2.2. Methodology                                                                            extrema of the superrotation during northern fall and winter. An
                                                                                            immediately apparent feature is the existence of a clear stratopause
    A series of simulations was carried out to examine various                              at all latitudes, at pressures of around 0.03–0.1 mbar. Though this
aspects of the model atmosphere. All simulations presented here                             occurs at a somewhat lower pressure than the stratopause observed
used relatively low T21 resolution (roughly 5.6° horizontal resolu-                         by the Huygens probe (Fulchignoni et al., 2005), it is in excellent
tion) to minimize computational requirements, and were run with                             agreement with Cassini CIRS observations of the middle atmo-
eighth-order hyperdiffusion to dissipate enstrophy that builds up                           sphere (Flasar et al., 2005; Achterberg et al., 2008), in which the
at the model’s smallest resolved scales. An additional diffusive                            stratopause occurs roughly between 0.05 and 0.1 mbar.
(r2 ) ‘‘sponge’’ was applied to wind ﬁelds at the top-most layer                                The warmest temperatures occur directly over the winter polar
to reduce wave reﬂections and improve numerical stability. Most                             regions, as a result of adiabatic heating of descending air driven by
simulations used a 32-layer, hybrid-coordinate atmosphere                                   the meridional circulation. This warm region appears at pressures
extending from the surface to approximately 40 lbar, hereafter                              below those of the rest of the stratopause. As winter progresses
referred to as L32. One simulation used 50 layers extending to                              into spring, it then descends in altitude and cools, decreasing the
about 3 lbar (hereafter L50).                                                               contrast with the low-latitude stratopause. The initial altitude
    In order to test the model’s ability to naturally super-rotate, a                       and subsequent cooling agree well with thermal emission spectral
‘‘control,’’ L32 simulation was started from rest (zero wind speeds                         data (Achterberg et al., 2011). However, the timing of the simu-
globally) and allowed to run until equilibrium was reached in the                           lated process is different than those observations: A warm polar
atmospheric variables. The superrotation stabilized after about                             region of about 210 K was seen on Titan shortly after northern win-
70 Titan years of integration. This simulation was run for an addi-                         ter solstice, whereas in the model that feature is already dissipat-
tional ﬁve Titan years, and all other L32 simulations were initial-                         ing at the corresponding time. This discrepancy may be due to
ized with this spun-up atmosphere. A continuation of the control                            the modeled haze and radiatively active stratospheric gases being
simulation was used for any direct comparisons.                                             uncoupled to the dynamics and horizontally homogeneous, as this
    Once the model’s capacity for superrotation was established, a                          warm feature is a balance between adiabatic heating and radiative
much more computationally expensive L50 run was started from                                cooling.
a prescribed superrotating state (the timestep was reduced to six                               In the lower stratosphere, high-latitude winter regions are cold-
from 15 min). This simulation’s superrotation stabilized quickly,                           est, with low- and summer latitudes having relatively ﬂat iso-
within two Titan years. An additional two years were run for                                therms, also agreeing with thermal emission spectral data
analysis.                                                                                   (Achterberg et al., 2008). The cold winter polar regions in this part
    A L32 simulation to brieﬂy test the effects of a variable haze ver-                     of the atmosphere may be in part the result of radiative cooling
sus the control simulation was run with the following parameter-                            during the polar night (Titan’s effective obliquity is 26.7°). A rather
ization: After computation of the haze optical depth in the                                 drastic cooling of the atmosphere around 1–10 mbar seen particu-
radiative transfer module, each layer’s haze optical depth ds above                         larly in the left panel of Fig. 1 occurs as the stratopause above it
an altitude of 80 km was modiﬁed as                                                         warms. This is shown as vertical proﬁles in Fig. 2. At around the
                                                                                            same time that adiabatic heating begins to warm the polar strato-
ds ¼ dsð1  ðcosð2/Þ  1Þj sinðtÞjÞ;                                             ð3Þ
                                                                                            pause, a signiﬁcant cooling of the stratosphere around 0.3 mbar
where / and t are latitude and orbital time, respectively. Thus, a                          occurs, coinciding almost exactly with the onset of polar night.
seasonally oscillating enhancement of wintertime haze optical                               The cooling slowly propagates downward and equatorward. The
depth, more pronounced at the poles, represents a rough parame-                             resulting temperature oscillation slowly extends to higher pres-
terization of haze transport by the atmosphere.                                             sures over the course of the season, reaching approximately the
   Finally, two additional simulations were used to examine the                             5 mbar level around LS  230 . This feature is consistent with
methane cycle in the lower atmosphere (without parameterized                                high-latitude temperature proﬁles observed from radio occulta-
haze variability). In the ﬁrst, the surface methane was replaced                            tions (Schinder et al., 2012), which display a temperature inversion
with a deep reservoir of 100 m, which represents an inexhaustible                           in the middle stratosphere. Though the simulation does not
global surface methane reservoir akin to what has been used in                              develop a proper inversion, the upper stratospheric temperatures,
most previous studies (Mitchell et al., 2006; Rannou et al., 2006;                          the sharp change in temperature gradient, the pressure where the
Tokano, 2009; Mitchell, 2012). This simulation reached equilib-                             feature again joins the ‘‘background’’ temperature proﬁle, and the
rium in less than ﬁve Titan years, and was run for an additional                            variation with latitude are all remarkably similar. This feature in
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                                         J.M. Lora et al. / Icarus 250 (2015) 516–528                                                                                              519




Fig. 1. Zonal-mean temperatures (K) from (left) northern fall, LS  230 , and (right) northern winter, LS  315 , from L50 simulation. An approximate altitude scale is
included for reference.




                                                  −2
                                                 10
                                                                                                                                                                400


                                                  −1
                                                 10                                                                                                             300




                               Pressure (mbar)
                                                  0                                                                                                             200



                                                                                                                                                                      Altitude (km)
                                                 10



                                                  1
                                                 10                                                                                                             100



                                                  2
                                                 10                                80 N
                                                                                        °
                                                                                                                                                   80°N         50
                                                                                        °                                                            °
                                                                                   74 N                                                            74 N
                                                                                                                                                     °
                                                                                   53°N                                                            53 N
                                                  3
                                                 10
                                                      60   90   120     150       180       210 60       90     120                     150   180         210

                                                                Temperature (K)                                Temperature (K)

Fig. 2. Left: L50 simulated vertical temperature at three latitudes shortly before winter solstice. Right: Selected radio occultation temperature proﬁles from the highest
observed latitudes, during various times in late northern winter (Schinder et al., 2012). An approximate altitude scale is included for reference.



the model also seems to dissipate too early, and has practically dis-
                                                                                                                                   −2
appeared by the time corresponding to the actual observations. A                                                                  10
                                                                                                                                                                                             400
likely candidate for this too-fast warming may be the lack of
buildup of stratospheric gases in the winter vortex. The altitude                                                                  −1
                                                                                                                                  10                                                         300
drop of the stratopause in late winter, not apparent in observa-
tions, is probably closely related. Note also that in the L32 simula-



                                                                                                                Pressure (mbar)                                                                    Altitude (km)
tion, which does not develop a proper stratopause, this                                                                            0
                                                                                                                                  10                                                         200
temperature oscillation is almost non-existent, indicating a strong
connection between the two features, probably related to their
                                                                                                                                   1
radiative effects.                                                                                                                10                                                         100

    In the troposphere, temperatures at all latitudes are in excellent
quantitative agreement with observations. At mid and low lati-                                                                     2                                                         50
                                                                                                                                  10                                                  L50
tudes (between ±60°), the tropopause remains between 69.5 and                                                                                                                         L32
71 K year-round, while the high-latitude tropopause dips to                                                                                                                           HASI
64 K in winter and also occurs at lower pressures, as seen by                                                                     3
                                                                                                                                  10
radio occultations (Schinder et al., 2012). This tropopause temper-                                                                    60     90          120        150               180
ature is also a sharper minimum than at lower latitudes where the                                                                                   Temperature (K)
tropopause is bracketed by a region that is nearly isothermal. Both
of these features agree with observations.                                                           Fig. 3. Simulated and observed (Fulchignoni et al., 2005) vertical temperature
                                                                                                     proﬁle near the equator for the time of the Huygens descent. An approximate
    A comparison of the vertical temperature structure from both
                                                                                                     altitude scale is included for reference.
L50 and L32 control simulations, relevant to the time and season
of the Huygens probe’s descent, is shown in Fig. 3. In both cases,
the lowest levels of the stratosphere occur at too-high pressures,                                   occultations (Schinder et al., 2011); our results are in excellent
though in general the stratospheric temperatures are close to those                                  agreement with these observations, as discussed above, all the
observed, without qualitatively different structures appearing in                                    way through the model domain. Note that no tuning of radiative
the simulations. Though the Huygens observations display a                                           parameters was done to achieve these temperature proﬁles.
stratopause at around 0.3 mbar (Fulchignoni et al., 2005), the same                                     Between the low stratosphere and the surface, the agreement
is not true of Cassini CIRS data (Flasar et al., 2005) or radio                                      between the simulations is good. Toward the high stratosphere,
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
520                                                           J.M. Lora et al. / Icarus 250 (2015) 516–528


the L32 simulation is increasingly cold in comparison, probably                        (similar to what was attained by Hourdin et al. (1995)). Strong
due to the model top inhibiting the formation of a full stratopause.                   winter jets develop within the ﬁrst few Titan years of simulation,
Nevertheless, the two simulations are in adequate agreement, val-                      with maximum wind speeds over high mid-latitudes. During
idating the use of the L32 model for investigating the lower strato-                   spring/fall, the wind maximum travels across the equator, dissolv-
sphere and troposphere.                                                                ing the springtime polar jet and ramping up the opposite hemi-
                                                                                       sphere’s. During this time, the peak winds also shift to lower
3.2. Meridional circulation                                                            pressures, in agreement with observations, which suggest an
                                                                                       increase in windspeeds of several tens of m s1 at pressures below
    The zonal-mean meridional streamfunction of the L50 simula-                        0.1 mbar between 2005 and 2009 (Achterberg et al., 2011). The
tion is shown in Fig. 4 for southern summer solstice (LS  270 )                      maximum integrated angular momentum occurs shortly after
and northern vernal equinox (LS  0 ). In the former case, a pole-                    equinoxes, when wind speeds also peak.
to-pole circulation is apparent, particularly in the stratosphere,                        All latitudes in the stratosphere and upper troposphere contin-
with rising motion in the summer hemisphere and subsidence in                          uously support westerlies. In the middle troposphere, zonal winds
the winter hemisphere. This is consistent with previous models                         reach tens of meters per second, in agreement with inferred winds
(e.g., Friedson et al., 2009; Newman et al., 2011; Lebonnois et al.,                   from cloud observations (Grifﬁth et al., 2005; Porco et al., 2005).
2012a). A small tropospheric cell is also visible at high southern                     Close to the surface, easterlies dominate at low latitudes, with
latitudes, due to rising motion occurring over the location of max-                    mid-latitude winds oscillating between pro- and retro-grade with
imum surface heating; this is further discussed in the following                       season.
section. Previous three-dimensional models show similar struc-                            The behavior of winds in the L50 simulation is similar, though
tures at high latitudes (Newman et al., 2011; Lebonnois et al.,                        with slightly higher (more realistic) wind speeds. Superrotation
2012a) and Mitchell et al. (2009) also showed in an axisymmetric                       extrema lag in comparison to the L32 by about 15° of LS , with peak
model that latent heating limits the Hadley upwelling in the tropo-                    winds occurring during mid-fall (Fig. 5), and weakening through
sphere, similarly to our simulated circulation. During equinox, a                      winter before the hemispheric reversal. Some effects of the top-
more symmetric equator-to-pole circulation develops throughout                         layer sponge are apparent at the model top, and the highest few
the atmosphere, as the equivalent of an intertropical convergence                      layers cannot be considered reliable. Nevertheless, the simulated
zone (ITCZ), where rising motion dominates, crosses the equator.                       winds at pressures above 0.01 mbar are satisfactory.
    This meridional circulation (Fig. 4) is also representative of that                   Fig. 6 shows the simulated vertical proﬁle of zonal wind at the
from the L32 simulations, though in those cases the lower model                        season and approximate latitude of the Huygens descent, com-
top expectedly suppresses the circulation of the lowest pressure                       pared to observations. The agreement in the troposphere in the
levels.                                                                                control case is good, including the presence of weak easterlies
                                                                                       between the surface and 5 km altitude (though the observed
3.3. Zonal winds                                                                       weak surface westerlies are not present (Bird et al., 2005)). Just
                                                                                       above the tropopause, there is a decrease in the windspeed gradi-
    A primary aim of this model was to both reproduce the temper-                      ent with altitude, especially pronounced in the L50 model, but nei-
ature structure through the stratopause and also achieve atmo-                         ther simulation reproduces the observed stillness between 70 and
spheric superrotation, something that has proven difﬁcult for                          80 km. Lebonnois et al. (2012a) produce a modest decrease in zonal
three-dimensional models (e.g. Friedson et al., 2009; Newman                           winds in the vicinity of this region, suggesting its formation may be
et al., 2011; Lebonnois et al., 2012a; Tokano, 2013). In the L32 con-                  related to haze feedbacks (see below and Fig. 7). It is also worth
trol simulation spun up from rest, the atmosphere quickly becomes                      noting that the observed altitude of this drop-off coincides with
superrotating, though with zonal wind magnitudes lower than the                        that where the previously described polar temperature oscillation
observed 200 m s1 (Achterberg et al., 2008), of around 130 m s1                     ends.




Fig. 4. Mean meridional streamfunction (109 kg s1) corresponding to southern summer solstice (left) and northern spring equinox (right), showing the mean circulation
from the L50 simulation. Bottom panels show a zoomed view of the troposphere. Positive values indicate clockwise motion; negative values are dashed lines in shaded
regions. The contour magnitudes increase by factors of 4, with the lowest magnitude labeled. An approximate altitude scale is included for reference.
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                                  J.M. Lora et al. / Icarus 250 (2015) 516–528                                                        521




Fig. 5. Zonal-mean zonal winds (m s1) from northern mid-fall (left) and mid-winter (right) from the L50 simulation, corresponding to the temperature ﬁelds in Fig. 1. The
contour interval is 20 m s1; the 0 m s1 contour is in white. An approximate altitude scale is included for reference.


                                                                                           trace gas coupling, but should not signiﬁcantly affect the results
                                                                                           in the troposphere and lower stratosphere.
                           140
                                                                                               The annual-mean difference between zonal-mean zonal winds
                                                                                           from the variable haze and control simulations (both L32) is shown
                           120                                                             in Fig. 7. Two main features are immediately apparent: First, in the
                                                                                           high stratosphere, the varying haze acts to increase the wind
                                                                                           speeds, and more so at higher latitudes (with peak instantaneous
                           100                                                             differences >15 m s1 in the winter jets). Second, a decrease in
                                                                                           wind speed occurs around 20 mbar. This coincides with the alti-
                                                                                           tude of the observed wind speed minimum in the stratosphere.


           Altitude (km)
                           80
                                                                                           Though it is also exactly the altitude (80 km) of the chosen transi-
                                                                                           tion between varying and non-varying haze, tests with a lower
                           60                                                              transition (40 km; not shown) produced no difference in the alti-
                                                                                           tude or magnitude of this wind deceleration, and high-altitude
                                                                                           winds were additionally enhanced. This simple parameterization
                           40
                                                                                           of the seasonal variability of haze is insufﬁcient to accurately study
                                                                L50                        this phenomenon, but it appears plausible that this variability may
                           20                                   L32                        at least partially affect the apparent de-coupling of tropospheric
                                                                DWE                        and stratospheric winds, seen in the data (Bird et al., 2005). These
                                                                                           features illustrate the importance of the stratospheric haze on the
                            0                                                              zonal winds, which agrees with the conclusion from axisymmetric
                                 0   20      40    60      80     100
                                                                                           models that haze-dynamics coupling enhances, rather than sup-
                                          Wind speed (m/s)
                                                                                           presses, wind speeds in the stratosphere (Rannou et al., 2004). Fur-
Fig. 6. Simulated zonal-mean zonal wind and observed zonal wind (Bird et al.,              ther studies of the relationship between varying haze, polar
2005) proﬁle near the equator for the time of the Huygens descent.                         temperatures, and the paucity of winds in this region of the atmo-
                                                                                           sphere will be the subject of a future study. It should be noted,
                                                                                           however, that the difference in wind speeds between variable
                                                                                           and non-variable haze simulations is modest, and the impact of
                                                                                           other effects, such as resolution, also needs to be explored.


                                                                                           3.4. Superrotation

                                                                                               Despite the lower-than-observed wind speeds in the high
                                                                                           stratosphere (particularly in the L32 simulations where vertical
                                                                                           resolution is low and the circulation is affected by the top-most
                                                                                           layer sponge), the agreement between simulated and observed
                                                                                           zonal winds is good at least below approximately 1.0 mbar in both
                                                                                           simulations (all of Fig. 6), and the simulated atmosphere is clearly
Fig. 7. Difference between the L32 annual-mean variable haze and control                   adequately superrotating. This is a result of relative angular
simulations’ zonal-mean zonal winds (m s1). An approximate altitude scale is              momentum build-up, which derives from surface torques that
included for reference.
                                                                                           transfer net angular momentum from the solid body to the
                                                                                           atmosphere.
                                                                                               Fig. 8 shows the surface torque and rate of change of the atmo-
   Above 80 km, the wind proﬁles again agree satisfactorily up to                          spheric angular momentum (top), and the total atmospheric angu-
the altitude of the in situ measurements (Bird et al., 2005). As sta-                      lar momentum versus integrated surface torque (bottom), for the
ted above, the winds drop off too quickly in the upper stratosphere                        ﬁrst year of the L32 control simulation, started from rest. The top
above this, particularly in the L32 simulation, so the peak winds                          panel also shows the total numerical torque,  , which represents
observed by CIRS at 0.1 mbar (Achterberg et al., 2008) are not                            spurious torques due to conservation errors from the dynamical
attained. This difﬁculty is likely due to a variety of model                               core and hyperdiffusion, as well as the effect of the top layer
constraints, including the low model top and the lack of haze or                           sponge (see Lebonnois et al., 2012b). Though this numerical torque
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
522                                                                                J.M. Lora et al. / Icarus 250 (2015) 516–528


                                   8                                                                        4.1. Surface energy budget
                                           dMr/dt




        Torques (×1016 kg m s )
        −2
                                   6
        2                                  F                                                                    An important consequence of accurate shortwave radiative
                                   4       ε*                                                               transfer is that, because of the increased pathlength through the
                                                                                                            atmosphere at high latitudes due to curvature, the distribution of
                                   2
                                                                                                            insolation at the surface is not proportional to that at the top of
                                   0                                                                        the atmosphere (Lora et al., 2011). The top panel of Fig. 9 shows
                                                                                                            this surface distribution from the GCM. Insolation peaks at summer
                                  −2
                                                                                                            mid-latitudes, and the southern summer, which is slightly shorter
                                  −4                                                                        than its northern counterpart, experiences higher insolation, due to
                                       0        0.2      0.4           0.6   0.8            1
                                                                                                            Saturn’s orbital eccentricity and obliquity.
                                   8                                                                            The bottom panel of Fig. 9 shows the surface net turbulent
                                           Atmosphere AM                                                    ﬂuxes, the sum of sensible heat ﬂux to the atmosphere and evapo-
                                   6                                                                        rative (latent) energy ﬂux, from the global surface liquid simula-




        AM (×1024 kg m2 s−1)
                                           Integrated Surface Torque

                                   4
                                                                                                            tion. Results from the lakes/seas simulation are similar, though
                                                                                                            the partitioning between evaporation and sensible heat is entirely
                                   2                                                                        different. Where there is available surface methane, evaporation
                                                                                                            tends to dominate the surface ﬂux. The distribution of these turbu-
                                   0
                                                                                                            lent ﬂuxes is less neatly organized than that of the insolation, but
                                  −2                                                                        the overall pattern is still obvious, and clearly mimics the latter,
                                                                                                            with maxima at the summer mid-latitudes and minima over the
                                  −4
                                       0        0.2      0.4           0.6   0.8            1               winter poles. The magnitudes of these ﬂuxes are also remarkably
                                                       Time (Titan years)                                   similar, despite the turbulent ﬂuxes responding to the total surface
                                                                                                            radiative imbalance. (Thermal infrared ﬂuxes, which dominate the
Fig. 8. Top: The rate of change of the atmospheric relative angular momentum                                radiative ﬂux at the surface, are much less variable than the short-
dMr =dt, net friction torque from the surface F, and spurious numerical torques,  ,
for the ﬁrst year of the L32 control simulation. Bottom: The corresponding total
                                                                                                            wave.) The maximum heating of the surface, often cited as the
                                                                             R                              mechanism for cloud formation (e.g., Brown et al., 2002), does
atmospheric relative angular momentum (AM) and integrated surface torque, F dt.
Ideally, these two curves would be identical.                                                               not occur over the polar regions, and therefore neither does the
                                                                                                            maximum of destabilizing turbulent ﬂux. Polar surface tempera-
is not zero (the ideal case), it remains for the most part signiﬁ-                                          tures also never exceed those of the lower latitudes. Surface tem-
cantly smaller than the net friction torque from the surface, and                                           peratures are further discussed below.
therefore does not impede the development of the atmosphere’s
angular momentum: the curves in the bottom panel are very close,
                                                                                                            4.2. Surface temperatures
and are positive after a year of simulation. This is further validation
that the physical and numerical representation of Titan’s atmo-
                                                                                                               Thermal infrared measurements of Titan’s surface brightness
sphere in our model is robust. Note that, since these simulations
                                                                                                            temperatures (Jennings et al., 2011) are compared to two sets of
do not include topography, mountain torques are not simulated,
                                                                                                            simulated surface temperatures in Fig. 10. The temperatures from
though they may have an additional impact on the angular
momentum budget.
    In our development of this Titan GCM, we initially coupled the
physics package to the ﬁnite volume, cubed-sphere dynamical core
from the GFDL Atmosphere Model 3 (AM3; Donner et al., 2011).
However, we found that, with that dynamical core, the numerical
torques  compensate the net frictional torques F almost
exactly—similarly to what is shown in Fig. 3b/d of Lebonnois
et al. (2012b) for a simpliﬁed-physics Venus GCM with the CAM5
dynamical core—and thereby completely prevent the buildup of
atmospheric angular momentum. In our case, using a ‘‘full’’ as
opposed to simpliﬁed physics package, tests with basic topogra-
phy, as well as various amounts of divergence damping, did not
improve the situation, though these were by no means exhaustive.
It is possible that similar difﬁculties with the CAM dynamical core,
which is closely related to the GFDL core, used by Friedson et al.
(2009) are responsible for their failure to achieve any superrota-
tion. Though further tests with these dynamical cores and Titan-
like physics are clearly warranted, we opted to switch to GFDL’s
spectral core since our primary aim was a capable and realistic
Titan model.


4. Lower atmosphere and methane cycle

   In this section, we present results of the lower atmosphere from
two L32 simulations, one with an inexhaustible, global surface                                              Fig. 9. Top: Insolation distribution at the surface (W m2). Bottom: Distribution of
                                                                                                            turbulent ﬂuxes (sum of evaporative and sensible heat ﬂuxes; W m2) from L32
liquid reservoir and another initiated with a limited surface                                               global reservoir simulation. Note that the insolation is positive into the surface,
methane supply plus deeper reservoirs at the locations of Titan’s                                           while turbulent ﬂuxes are positive into the atmosphere. Vertical lines indicate the
largest lakes/seas.                                                                                         timing of solstices and northern vernal equinox.
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                                  J.M. Lora et al. / Icarus 250 (2015) 516–528                                                              523


                                                                                           temperatures are the result of global evaporative cooling, and
                                                                                           immediately highlight the implausibility of realistic simulations
                                                                                           assuming global surface methane coverage. Note, however, that
                                                                                           we have not varied the surface thermal properties between these
                                                                                           two simulations. Indeed, a global ‘‘ocean’’ of methane would in
                                                                                           reality have a larger thermal inertia, so the surface temperatures
                                                                                           would probably be even less variable.


                                                                                           4.3. Surface winds

                                                                                               Fig. 11 shows daily zonal and meridional wind speeds from four
Fig. 10. Surface temperatures from two sets of L32 simulations for two seasonal            latitudinal regions of the GCM’s lowest atmospheric layer
periods corresponding to the thermal emission analysis of Jennings et al. (2011).          (1439 mbar) for a Titan year of simulation with lakes/seas. Polar
Red and blue curves correspond to late northern winter (‘‘LNW;’’ September 2006–           zonal winds are consistently eastward and stronger than equato-
May 2008) and northern spring equinox (‘‘NSE;’’ November 2008–May 2010)                    rial winds, which are predominantly westward. In summertime,
periods, respectively. Solid lines are the lakes/seas simulation, dashed lines the
                                                                                           the former occasionally reach speeds approximately twice as fast
global methane simulation, and the light shaded regions are approximately
equivalent to the measurements with error bars of Jennings et al. (2011).                  as the wintertime average, which is also slightly higher in the
                                                                                           north than the south. Though the maximum speeds increase dur-
                                                                                           ing spring, there is considerable variability throughout the year.
the lakes/seas simulation (solid lines) agree reasonably well with                         Meridional winds in the polar regions are at least an order of mag-
the observations, especially at higher latitudes. The signiﬁcant                           nitude weaker, and vary signiﬁcantly more during their respective
decrease in temperatures poleward of ±70°, due to the prevalence                           hemisphere’s summer, reaching their maximum magnitudes.
of surface liquids and associated evaporative cooling, may also be                             Equatorial winds experience less variability, and the zonal com-
present in the measurements, especially in the south, and produces                         ponent displays the opposite trend in magnitude as at the poles:
a strong resemblance between simulations and measurements.                                 Faster (easterly) winds occur during wintertime in both hemi-
    Equatorward of these latitudes, the simulated temperatures are                         spheres, with summertime wind speeds averaging close to
higher than observed by 0.5–1.0 K, though the observations                                 0 m s1. On the other hand, meridional equatorial winds are of
roughly represent a zonal average that includes varying topogra-                           equivalent magnitude as zonal winds, and oscillate between south-
phy and different albedos and surface properties, none of which                            ward and northward through a Titan year, with transitional peri-
is currently included in the model. Nevertheless, the simulated sur-                       ods of close to zero wind speed near equinoxes. Also evident is
face temperatures follow the same overall trend and peak at the                            the fact that these meridional winds are cross-equatorial (ﬂowing
same approximate latitudes, roughly 10°S and 5°N for the two                               from winter to summer hemispheres), as the two curves vary
periods shown, respectively, with an equator-pole difference of                            together.
about 2 K in the south and 3 K in the north. Additionally, the                                 We brieﬂy discuss the implications of these wind results. In
observed and modeled northward warming trends are in general                               general, the seasonally reversing equatorial meridional winds are
agreement, even for a period roughly equivalent to only 10% of a                           in agreement with the results of Tokano (2010), and therefore with
Titan year.                                                                                that assessment of dune orientation. Though evidence of the fast
    These simulated temperatures are highest during this period,                           equinoctal westerlies discussed in that paper is absent here, we
roughly during late northern winter and vernal equinox. 180° of                            did not analyze instantaneous maximum and minimum wind
LS later, they are generally lower (not shown), because of reduced                         speeds, and therefore their signal may be lost to time averaging.
insolation due to the larger Sun–Saturn distance. Equator-pole sur-                        The variability of surface winds also suggests a connection to
face temperature gradients are approximately the same year-                                weather events, which could be the source of eastward gusts that
round, with the winter pole being coldest.                                                 may control dune orientation (Lucas et al., 2014). Regardless, per-
    On the other hand, the surface temperatures produced by the                            sistent equatorial westerlies, previously discussed as a candidate
global surface liquid simulation (dashed lines in Fig. 10) are too                         from the dune orientations (e.g. Radebaugh et al., 2008), are con-
latitudinally homogeneous, and signiﬁcantly lower than the                                 clusively inconsistent with our results (and indeed prevalent eas-
measurements. Equator-pole contrasts in this case are only                                 terlies are necessary for the ﬂux of angular momentum into the
0.2 K and 0.9 K in the south and north, respectively. These surface                       atmosphere from the surface).




Fig. 11. Zonal (left panel) and meridional (right panel) winds from the lowest atmospheric layer of the L32 lakes/seas simulation. Curves labeled polar correspond to the
average between 60° and 90° in each hemisphere, whereas equatorial winds (‘‘Eq.’’) refer to average winds of the two gridpoints closest to the equator, roughly between 0°
and 10°. The legend for all curves is in the right panel. Note the different scales. The shaded areas correspond to the range of threshold wind speeds for generating waves from
Hayes et al. (2013), scaled to the altitude of the simulated winds. Vertical lines indicate the timing of solstices and northern vernal equinox.
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
524                                                                         J.M. Lora et al. / Icarus 250 (2015) 516–528


   Separately, our simulated polar winds are considerably stronger                                   of CH4–N2 liquid, calculated from the observed methane proﬁle,
and somewhat more variable than those cited in Fig. 5 of Hayes                                       stops increasing linearly with altitude (Tokano et al., 2006), and
et al. (2013), and the transition from below to above the threshold                                  may mark the beginning of the transition from a liquid mixture
speed for generating waves on Titan’s seas is not evident. Our                                       to a pure methane ice. Regardless, the simulated increase of spe-
results imply yearly mean reference height (10 m) winds of                                           ciﬁc humidity toward the surface would occur, with slightly differ-
0.5 m s1, just around the threshold speeds suggested. However,                                     ent values, in either case, as long as a source of methane were
especially in the northern hemisphere, occasional winds exceeding                                    present on the surface; this behavior was not observed and is not
2 m s1 (equivalent to reference height winds 1.3 m s1) occur in                                   produced in the lakes/seas simulation.
summertime, and are well above the thresholds; thus, our results                                         In all cases, the simulated methane proﬁle above the tropo-
agree with the prediction of waves on Titan’s northern seas in sum-                                  pause (not shown) increases slightly to a speciﬁc humidity of about
mertime. Both zonal wind speeds and the slight increase after                                        0.01 at 10 mbar, and is constant at lower pressures. This is a slight
northern vernal equinox (roughly 0.55 of one Titan year) are in                                      over-estimate compared to the observations, and the increase rep-
agreement with the possible detection of waves on Punga Mare,                                        resents a too-large ﬂux of methane between troposphere and
and the inferred wind speed of 0.8 m s1 (Barnes et al., 2014).                                     stratosphere, despite the cold trap of the tropopause; the mecha-
Nevertheless, the onset of wave activity is not obvious from the                                     nism for this is not clear, but is potentially related to the lack of
simulations, and the detection of waves may depend on the timing                                     a sink for methane at the model top. Nevertheless, this discrepancy
of observations, as the wind speeds are not persistently high dur-                                   has a negligible effect on the methane cycle of the lower tropo-
ing the season. It is furthermore unclear that the model’s low res-                                  sphere and surface.
olution is capable of predicting the appropriate mesoscale                                               Distributions of precipitation versus time are shown in Fig. 13,
conditions that might be the dominant inﬂuence on local wave-                                        with some cloud observations overlain for comparison (with the
generating wind speeds.                                                                              assumption that model precipitation can be used as a proxy for
                                                                                                     clouds). Most of this is moist convective precipitation that imme-
                                                                                                     diately reaches the surface. Relatively light but sustained precipita-
4.4. Humidity and methane cycle
                                                                                                     tion is prevalent in the global surface reservoir simulation, and a
                                                                                                     clear relationship exists between the location of low and mid-lati-
   Modeled and observed (Niemann et al., 2005) equatorial tropo-
                                                                                                     tude precipitation and that of seasonally-controlled upwelling, in
spheric methane proﬁles are shown in Fig. 12. In both simulations,
                                                                                                     agreement with previous models (Mitchell et al., 2006, 2011). Also
the speciﬁc humidity at the surface is high, compared to that mea-
                                                                                                     present is summertime polar precipitation that decreases but does
sured, but the global methane simulation overestimates it signiﬁ-
                                                                                                     not cease in other seasons. Though this precipitation distribution
cantly more. Additionally, the lakes/seas simulation produces a
                                                                                                     appears to agree with the majority of cloud observations, it also
nearly-constant speciﬁc humidity at pressures above 1100 mbar,
                                                                                                     implies nearly-permanent cloud cover and continued activity at
as observed, whereas the proﬁle in the other case is distinctly dif-
ferent, with the speciﬁc humidity increasing almost to the surface.
This is a consequence of the availability of moisture from the sur-
face, which indicates that, within our assumptions, the observed
methane proﬁle at low latitudes is consistent with a dry surface.
   The turn-over in the lakes/seas simulation equatorial methane
proﬁle starting around 1160 mbar also corresponds to a transi-
tional region in the global methane simulation proﬁle. This is
because that level in the atmosphere marks the temperature
(87 K) chosen as the transition between the methane–nitrogen
liquid and methane ice in the computation of saturation vapor
pressure. This temperature is high compared to the standard
assumption that the condensate is liquid down to around 80 K,
but agrees quite well with the altitude where the relative humidity




                                                        Lakes/seas         40
                                                        Global CH
                                                                    4
                              200
                                                        Huygens




            Pressure (mbar)
                                                                           30



                                                                                Altitude (km)
                              500                                          20
                                                                                                     Fig. 13. Top: Precipitation (kg/m2) distribution for the L32 global surface reservoir
                                                                                                     simulation, averaged over ﬁve Titan years, in color contours. Bottom: The same for
                                                                                                     the L32 lakes/seas simulation in color. Additional gray contours show the
                                                                           10                        distribution of the frequency of large-scale condensation in the troposphere (see
                              1000
                                                                                                     text); the light and dark contour lines are 0.25 and 0.75, respectively. Observations
                                                                                                     of clouds are shown in both panels for comparison: Black squares are from Bouchez
                                                                                                     and Brown (2005); black diamonds are from Schaller et al. (2006); small light gray
                                0.005      0.015        0.025           0.035                        circles are observations from VIMS (S. Rodriguez, personal communication;
                                        Specific Humidity (kg/kg)                                    Rodriguez et al., 2009, 2011); black circles are clouds labeled ‘‘convective’’ from
                                                                                                     Turtle et al. (2011a); Light gray  are from Roe et al. (2005); and darker gray  are
Fig. 12. Modeled (zonally averaged) and observed methane speciﬁc humidity near                       clouds labeled other than ‘‘convective’’ from Turtle et al. (2011a). Note the different
the equator around the time of the Huygens probe descent. An approximate altitude                    scales for the color contours. Vertical lines indicate the timing of solstices and
scale is included for reference.                                                                     northern vernal equinox.
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                                    J.M. Lora et al. / Icarus 250 (2015) 516–528                                                525


the south pole during and after equinox, neither of which are                                Le Mouélic et al., 2012); however, these form by condensation of
observed. In addition, the observed preference for clouds around                             downwelling species from the stratosphere and are composed pri-
40° is only satisﬁed because of the full-hemisphere coverage of                              marily of ethane, so are not captured by the model.
precipitation for half of Titan’s year; the particular latitudes are                             The surface liquid coverage near the end of the lakes/seas sim-
not actually preferred.                                                                      ulation is shown in Fig. 14, along with the net change during the
    In the lakes/seas simulation, precipitation is by comparison                             last year of simulation. Methane accumulates at both poles in
much more sparse but at times up to an order of magnitude more                               agreement with previous results (Schneider et al., 2012), with a
intense. Summertime polar precipitation is robust, as are occa-                              slight enhancement in coverage visible in the north. Surface meth-
sional low-latitude outbursts. The latter are in agreement with                              ane in the model is highly unstable at latitudes <50°, and is quickly
some of the data, which correspond to large (observed by                                     transported poleward by the atmosphere, as in previous models
ground-based telescopes) events and clouds labeled ‘‘convective’’                            (Rannou et al., 2006; Mitchell et al., 2006, 2009; Mitchell, 2008,
by Turtle et al. (2011a). Indeed, pauses in activity occur after pre-                        2012; Schneider et al., 2012). (Note that there is no build-up of sur-
cipitation outbursts, in agreement with the suggestion that atmo-                            face methane at mid-latitudes as the equatorial surface dries, as in
spheric depletion inhibits subsequent convection (Schaller et al.,                           the results of Mitchell (2008).) The simulated atmosphere holds
2006). However, this precipitation distribution does not match                               roughly 5 m of precipitable methane, which agrees with previous
well with other observations of clouds, particularly in the mid-lat-                         models and observations (Schneider et al., 2012; Tokano et al.,
itudes. Those clouds display characteristics consistent with con-                            2006). Patchy surface liquid at low latitudes, which coincides with
vective systems (Grifﬁth et al., 2005), but also tend to exhibit                             the detection of equatorial lakes (Grifﬁth et al., 2012), is associated
different, elongated morphologies compared to the polar clouds                               with bursts of precipitation there. But, as shown by the bottom
that were prevalent shortly after solstice (Turtle et al., 2011a).                           panel of Fig. 14, these features are shallow and ephemeral. Inter-
    The bottom panel of Fig. 13 also shows the distribution of the                           estingly, there is increased activity at mid latitudes in the vicinity
frequency of large-scale condensation in the troposphere between                             of the large northern seas, but again no signiﬁcant buildup
the surface and approximately 500 mbar (gray contours). While                                remains.
the vast majority of this condensation does not produce precipita-                               While seasonal activity is clearly an important mechanism in
tion that reaches the ground, its distribution is similar to that of                         the development of precipitation (Fig. 13), the availability of sur-
precipitation in the global-methane simulation. Mid-latitude cloud                           face methane appears to be a prerequisite to rain, with the notable
observations fall within regions where condensation occurs fairly                            exception of the occasional low-latitude outbursts reminiscent of
frequently. Thus, this large-scale condensation provides a possible                          those observed around equinox (Schaller et al., 2009; Turtle
explanation for mid-latitude clouds, as well as for the optically-                           et al., 2011b), and previously explained as due to 3D wave activity
thin stratiform clouds tentatively detected from in situ data                                (Mitchell et al., 2011). Consistent precipitation at high latitudes is
(Tokano et al., 2006). Note that at pressures lower than 500 mbar                            clearly linked to the availability of methane at the surface, as well
(not shown), light large-scale condensation is frequent during                               as the insolation. Without invoking a physically implausible fast
polar winter as a result of decreasing temperatures at the tropo-                            sub-surface transport of liquid (i.e., Schneider et al., 2012), only
pause. This is a different mechanism associated with higher-alti-                            simulations with an inexhaustible, global surface reservoir produce
tude, non-convective cloud decks. Similar cloud decks have been                              any signiﬁcant precipitation at summer mid-latitudes. Also consid-
observed over the winter polar tropopause (Grifﬁth et al., 2006;                             ering the dearth of observed lakes/seas away from polar regions
                                                                                             and the observed prevalence of 40°S clouds before, during, and
                                                                                             after equinox, this suggests that mid-latitude cloud activity is
                                                                                             either non-convective and non-precipitating, or caused by a mech-
                                                                                             anism not currently included in the model that is only somewhat
                                                                                             related to the changing seasons. Some possible such mechanisms
                                                                                             might be topographical forcing (i.e., via orographic gravity waves),
                                                                                             a sub-surface source of methane (possibly cryovolcanism or seep-
                                                                                             age from a methane table), or another non-convective/non-precip-
                                                                                             itating form of cloud formation. It is worth noting that early studies
                                                                                             of these clouds suggested a longitudinal as well as latitudinal
                                                                                             dependence (Porco et al., 2005; Roe et al., 2005), though later anal-
                                                                                             yses disputed this (Grifﬁth et al., 2005; Brown et al., 2010;
                                                                                             Rodriguez et al., 2011).
                                                                                                 The mean meridional energy transport by the atmosphere can
                                                                                             be examined via ﬂuxes of moist static energy (MSE), which is the
                                                                                             sum of dry static energy (DSE; internal plus potential energy in
                                                                                             an air parcel) and latent energy due to moisture. Fig. 15 shows
                                                                                             the annual-mean ﬂuxes of vertically integrated moist static, dry
                                                                                             static, and latent energies as a function of latitude for the lakes/
                                                                                             seas simulation. The MSE ﬂux is dominated by the ﬂux of DSE at
                                                                                             all but the highest latitudes, and is divergent at the equator. On
                                                                                             the other hand, latent energy ﬂux, which dominates at high lati-
                                                                                             tudes and is asymmetric (with a net northward value), is conver-
                                                                                             gent at the equator; latent energy ﬂows opposite to the MSE ﬂux,
Fig. 14. Top: Map of surface liquid (m) at northern summer solstice in the last year         dominantly transported by near-surface air from winter to sum-
of L32 lakes/seas simulation. The four darkest features are the initialized lakes/seas,      mer hemispheres. The divergence of the latent energy ﬂux at
which have remained above 100 m in depth throughout the simulation. Bottom:                  mid-latitudes illustrates the atmosphere’s ability to transport
The net change in surface liquid (m) over the course of the last year of simulation,
between northern autumnal equinoxes. Note that several of the features in the
                                                                                             methane away from these regions.
bottom map are not visible in the top, indicating their ephemeral nature. Both maps              Grifﬁth et al. (2014) suggest that cold-trapped polar methane
are Mollweid equal area projections.                                                         (Schneider et al., 2012) may explain the observed equatorial
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
526                                                                    J.M. Lora et al. / Icarus 250 (2015) 516–528


                               10                                                               improve the accuracy and ﬁdelity of simulated tropospheric
                               7.5                                                              clouds.
                                5
                                                                                                   Furthermore, no microphysical considerations were included in




           Energy Flux (TW)
                                                                                                the simulation of cloud formation or precipitation. The moist con-
                               2.5
                                                                                                vective parameterization provides an improvement over exclu-
                                0                                                               sively including large-scale condensation, which, particularly at
                              −2.5                                                              this resolution, is probably inaccurate. However, quasi-equilibrium
                                                                  Latent
                               −5                                 DSE                           convection still produces precipitation that is fairly sporadic, and is
                              −7.5
                                                                  MSE                           only an idealized representation of the process. Given the discrep-
                                                                                                ancy between the distribution of precipitation and observed tropo-
                              −10
                                −90   −60   −30      0       30   60       90                   spheric clouds, and considering that several mechanisms may be
                                                  Latitude                                      involved in cloud formation, a more detailed condensates scheme
                                                                                                as implemented in other models (Rannou et al., 2006; Burgalat
Fig. 15. Annual-mean atmospheric energy ﬂuxes from the L32 lakes/seas simula-                   et al., 2014), may also be justiﬁed.
tion. ‘‘DSE’’ and ‘‘MSE’’ stand for dry static energy and moist static energy,
respectively. Positive values indicate northward ﬂux.
                                                                                                   The surface model, while showing that an inﬁnite source of
                                                                                                methane constantly available to the atmosphere is inconsistent
                                                                                                with the observations, could also beneﬁt from a considerable
humidity (50%), since the observed equator to pole surface tem-                                increase in complexity. An obvious improvement is the inclusion
perature gradient would imply 85% polar humidity with the same                                 of topography, which would allow for testing of the importance
methane content. This would agree with estimates, based on                                      of orographic forcing on cloud formation (Roe et al., 2005; Porco
energy arguments, of low advective transport of methane                                         et al., 2005). Including surface runoff might also prove useful, for
(Grifﬁth et al., 2008). However, Mitchell (2012) showed that Titan’s                            example in simulating and predicting the locations of small lakes
constant outgoing longwave radiation with latitude is evidence of                               and lake-effect or ‘‘marine’’ clouds (Brown et al., 2002), though
transport by the atmosphere, a large portion of which is done by                                perhaps only at higher resolutions. Surface thermal properties, as
latent energy ﬂuxes. Our results provide more realism by eliminat-                              well as albedos, should also be allowed to vary.
ing the global methane source assumed in Mitchell (2012), but
nevertheless show signiﬁcant transport of methane, via which                                    6. Conclusions
polar moisture humidiﬁes the equatorial atmosphere in agreement
with Grifﬁth et al. (2014).                                                                         We have presented results from simulations using TAM, a new,
                                                                                                fully three-dimensional GCM of Titan’s atmosphere with realistic
5. Discussion                                                                                   radiative transfer, as well as moist processes and a surface liquid
                                                                                                model. Benchmarked against the available observations, our work
   The simulations presented in this paper represent an effort to                               demonstrates that two of the most important factors for simulat-
model the circulation and climate of Titan’s atmosphere realisti-                               ing the key aspects of Titan’s atmosphere are accurate radiative
cally, eliminating several signiﬁcant simpliﬁcations from past                                  transfer and a dynamical core numerically capable of developing
models and succeeding in reproducing many important aspects                                     superrotation.
of the atmosphere. Nevertheless, the model still necessarily                                        Several aspects relating to the state of the surface-atmosphere
employs a variety of simpliﬁcations that are presently discussed.                               system are elucidated through the simulations:
   The relatively low wind speeds in the model’s high stratosphere
are not particularly surprising given the variety of factors that are                             The vertical temperature proﬁle through the stratopause, both
probably contributing to inhibiting the magnitude of superrota-                                    at the equator and at the poles, is reproduced satisfactorily
tion. These include (but are probably not limited to): The low                                     without the need to invoke complex interactions between haze,
model top, with a sponge layer to prevent spurious wave reﬂec-                                     trace gases, and dynamics (though these may further improve
tions, which artiﬁcially damps the winds of the top layer and prob-                                the results (Rannou et al., 2002, 2004; Crespin et al., 2008)).
ably signiﬁcantly affects the top several layers’ circulation; the lack                            CIRS measurements are particularly well reproduced. The polar
of haze-dynamics coupling, which is shown to depress wind-                                         structure observed in radio occultations (Schinder et al., 2012)
speeds at high altitudes (as in Rannou et al., 2002, 2004) and                                     is similar to what is produced by a temperature oscillation that
should increase latitudinal temperature contrasts (Rannou et al.,                                  originates in the high stratosphere at the onset of polar night
2004; Crespin et al., 2008); the assumption of hydrostatic balance                                 and propagates downward, though the simulated timing pre-
that ignores the effect of the wind-induced equatorial bulge                                       cedes the observations.
(Tokano, 2013); and the assumption of horizontally-homogeneous,                                   No additional physics or numerical techniques are necessary to
dynamically uncoupled stratospheric trace gases that are radi-                                     achieve proper atmospheric superrotation. The zonal wind pro-
atively active. Indeed, the zonal wind results at pressures higher                                 ﬁle through the lower stratosphere agrees well with observa-
than 1 mbar are excellent, and the model’s capability to build                                    tions, with the exception of the observed minimum around
up atmospheric angular momentum is quite promising. Further                                        75 km altitude; this structure may be related to both the polar
upgrades to alleviate the above restrictions are the subject of                                    temperature oscillation and/or the seasonal variation of haze
future work.                                                                                       and its radiative effects.
   Several aspects of the methane cycle should also be considered                                 Surface turbulent ﬂuxes respond to the surface insolation, and
simpliﬁcations that warrant further development. Importantly,                                      as a result their maxima occur over summer mid-latitudes,
only methane is included as a tracer, so the depressing effect of                                  not the polar regions, counter to what has been previously
ethane on evaporation rates, for instance, is neglected. Related to                                suggested.
this is the simpliﬁed calculation of vapor pressure, in which the                                 Surface liquids quickly migrate polewards from lower latitudes,
effects of dissolved nitrogen are only roughly accounted for. The                                  in agreement with prior studies (Rannou et al., 2006; Mitchell
inclusion of additional tracers for the methane cycle, such as meth-                               et al., 2006, 2008; Schneider et al., 2012; Lora et al., 2014); they
ane ice particles, and coupling to the radiative transfer, may also                                are unstable equatorward of approximately 60° on timescales of
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                                      J.M. Lora et al. / Icarus 250 (2015) 516–528                                                                   527


   order one Titan year. Both the surface temperature and the ver-                             Grifﬁth, C.A. et al., 2012. Possible tropical lakes on Titan from observations of dark
                                                                                                   terrain. Nature 486 (7402), 237–239.
   tical distribution of humidity at low latitudes are consistent
                                                                                               Grifﬁth, C.A., McKay, C.P., Ferri, F., 2008. Titan’s tropical storms in an evolving
   with this distribution of surface liquids. Atmospheric energy                                   atmosphere. Astrophys. J. 687, L41–L44.
   ﬂuxes are also consistent with this picture, and also indicate a                            Grifﬁth, C.A., Rafkin, S., Rannou, P., McKay, C.P., 2014. Storms, clouds, and weather.
   preference toward northward methane transport in the current                                    In: Müller-Wodarg, I., Grifﬁth, C.A., Lellouch, E., Cravens, T.E. (Eds.), Titan:
                                                                                                   Interior, Surface, Atmosphere, and Space Environment. Cambridge University
   epoch, as in Schneider et al. (2012).                                                           Press, Cambridge.
  Summer mid-latitude clouds, however, are difﬁcult to explain                                Hayes, A. et al., 2013. Wind driven capillary-gravity waves on Titan’s lakes: Hard to
   with the simulated precipitation distribution, except when glo-                                 detect or non-existent? Icarus 225, 403–412.
                                                                                               Hourdin, F., Talagrand, O., Sadourny, R., Courtin, R., Gautier, D., McKay, C.P., 1995.
   bal surface liquid is imposed, and then not very satisfactorily.                                Numerical simulations of the general circulation of the atmosphere of Titan.
   We suggest that these clouds, as observed on Titan, are either                                  Icarus 117, 358–374.
   non-precipitating, or are related to a process not currently cap-                           Hourdin, F., Lebonnois, S., Luz, D., Rannou, P., 2004. Titan’s stratospheric
                                                                                                   composition driven by condensation and dynamics. J. Geophys. Res. 109,
   tured in this GCM.                                                                              E12005.
                                                                                               Jennings, D. et al., 2011. Seasonal changes in Titan’s surface temperatures.
    Titan’s atmosphere and surface represent a complex system in                                   Astrophys. J. 737 (1), L15.
                                                                                               Lebonnois, S., Burgalat, J., Rannou, P., Charnay, B., 2012a. Titan global climate
which a variety of factors communicate. With this work, it is evi-                                 model: A new 3-dimensional version of the IPSL Titan GCM. Icarus 218, 707–
dent that realistic interactions between physical processes like                                   722.
radiation and variable surface moisture—seldom considered                                      Lebonnois, S. et al., 2012b. Angular momentum budget in general circulation
                                                                                                   models of superrotating atmospheres: A critical diagnostic. J. Geophys. Res. 117,
together in previous models—are critical to properly simulating
                                                                                                   E12004.
the climate of this unique world. Future studies, in particular those                          Le Mouélic, S. et al., 2012. Dissipation of Titan’s north polar cloud at northern spring
involving the methane cycle, must include proper radiative trans-                                  equinox. Planet. Space Sci. 60, 86–92.
fer and move beyond the assumption of a global ocean on Titan.                                 Lora, J.M., Goodman, P.J., Russell, J.L., Lunine, J.I., 2011. Insolation in Titan’s
                                                                                                   troposphere. Icarus 216, 116–119.
                                                                                               Lora, J.M., Lunine, J.I., Russell, J.L., Hayes, A.G., 2014. Simulations of Titan’s
Acknowledgments                                                                                    paleoclimate. Icarus 243, 264–273.
                                                                                               Lucas, A. et al., 2014. Growth mechanisms and dune orientation on Titan. Geophys.
                                                                                                   Res. Lett. 41, 6093–6100.
   The authors acknowledge support from NASA Earth and Space                                   McKay, C.P., Pollack, J.B., Courtin, R., 1989. The thermal structure of Titan’s
Science Fellowship NNX12AN79H, and the Cassini project. Simula-                                    atmosphere. Icarus 80, 23–53.
tions were carried out with an allocation of computing time on the                             Mitchell, J.L., 2008. The drying of Titan’s dunes: Titan’s methane hydrology and its
                                                                                                   impact on atmospheric circulation. J. Geophys. Res. 113, E08015.
High Performance Computing systems at the University of Arizona.                               Mitchell, J.L., 2012. Titan’s transport-driven methane cycle. Astrophys. J. 756, L26.
The authors would also like to thank S. Rodriguez and an anony-                                Mitchell, J.L., Pierrehumbert, R.T., Frierson, D.M.W., Caballero, R., 2006. The
mous reviewer for detailed comments to improve the manuscript,                                     dynamics behind Titan’s methane clouds. Proc. Natl. Acad. Sci. 103, 18421–
                                                                                                   18426.
and S. Rodriguez for providing the VIMS cloud observations.                                    Mitchell, J.L., Pierrehumbert, R.T., Frierson, D.M.W., Caballero, R., 2009. The impact
                                                                                                   of methane thermodynamics on seasonal convection and circulation in a model
References                                                                                         Titan atmosphere. Icarus 203, 250–264.
                                                                                               Mitchell, J.L., Ádámkovics, M., Caballero, R., Turtle, E.P., 2011. Locally enhanced
                                                                                                   precipitation organized by planetary-scale waves on Titan. Nat. Geophys. 4,
Achterberg, R.K., Conrath, B.J., Gierasch, P.J., Flasar, F.M., Nixon, C.A., 2008. Titan’s
                                                                                                   589–592.
    middle-atmospheric temperatures and dynamics observed by the Cassini
                                                                                               Moses, J.I., Allen, M., Yung, Y.L., 1992. Hydrocarbon nucleation and aerosol
    Composite Infrared Spectrometer. Icarus 194, 263–277.
                                                                                                   formation in Neptune’s atmosphere. Icarus 99, 318–346.
Achterberg, R.K., Gierasch, P.J., Conrath, B.J., Flasar, F.M., Nixon, C.A., 2011. Temporal
                                                                                               Newman, C.E., Lee, C., Lian, Y., Richardson, M.I., Toigo, A.D., 2011. Stratospheric
    variations of Titans middle-atmospheric temperatures from 2004 to 2009
                                                                                                   superrotation in the TitanWRF model. Icarus 213, 636–654.
    observed by Cassini/CIRS. Icarus 211, 686–698.
                                                                                               Niemann, H.B. et al., 2005. The abundances of constituents of Titan’s atmosphere
Anderson, C., Samuelson, R., 2011. Titan’s aerosol and stratospheric ice opacities
                                                                                                   from the GCMS instrument on the Huygens probe. Nature 438, 779–784.
    between 18 and 500 lm: Vertical and spectral characteristics from Cassini CIRS.
                                                                                               O’Gorman, P.A., Schneider, T., 2008. The hydrological cycle over a wide range of
    Icarus 212 (2), 762–778.
                                                                                                   climates simulated with an idealized GCM. J. Climate 21, 3815–3832.
Barnes, J.W. et al., 2014. Cassini/VIMS observes rough surfaces on Titan’s Punga
                                                                                               Porco, C. et al., 2005. Imaging of Titan from the Cassini spacecraft. Nature 434, 159–
    Mare in specular reﬂection. Planet. Sci. 3, 3.
                                                                                                   168.
Bird, M.K. et al., 2005. The vertical proﬁle of winds on Titan. Nature 438, 800–802.
                                                                                               Radebaugh, J. et al., 2008. Dunes on Titan observed by Cassini Radar. Icarus 194,
Bouchez, A.H., Brown, M.E., 2005. Statistics of Titan’s south polar tropospheric
                                                                                                   690–703.
    clouds. Astrophys. J. 618, L53–L56.
                                                                                               Rannou, P., Hourdin, F., McKay, C.P., 2002. A wind origin for Titan’s haze structure.
Briegleb, B.P., 1992. Delta-Eddington approximation for solar radiation in the NCAR
                                                                                                   Nature 418, 853–856.
    community climate model. J. Geophys. Res. 97, 7603–7612.
                                                                                               Rannou, P., Hourdin, F., McKay, C.P., Luz, D., 2004. A coupled dynamics-
Brown, M.E., Bouchez, A.H., Grifﬁth, C.A., 2002. Direct detection of variable
                                                                                                   microphysics model of Titan’s atmosphere. Icarus 170, 443–462.
    tropospheric clouds near Titan’s south pole. Nature 420, 795–797.
                                                                                               Rannou, P., Montmessin, F., Hourdin, F., Lebonnois, S., 2006. The latitudinal
Brown, M.E., Roberts, J.E., Schaller, E.L., 2010. Clouds on Titan during the Cassini
                                                                                                   distribution of clouds on Titan. Science 311, 201–205.
    prime mission: A complete analysis of the VIMS data. Icarus 205, 571–580.
                                                                                               Richard, C. et al., 2011. New section of the HITRAN database: Collision-induced
Burgalat, J., Rannou, P., Cours, T., Rivière, E.D., 2014. Modeling cloud microphysics
                                                                                                   absorption (CIA). J. Quant. Spectrosc. Radiat. Trans. 113, 1276–1285.
    using a two-moments hybrid bulk/bin scheme for use in Titan’s climate models:
                                                                                               Richardson, M.I., Toigo, A.D., Newman, C.E., 2007. PlanetWRF: A general purpose,
    Application to the annual and diurnal cycles. Icarus 231, 310–322.
                                                                                                   local to global numerical model for planetary atmospheric and climate
Crespin, A. et al., 2008. Diagnostics of Titan’s stratospheric dynamics using Cassini/
                                                                                                   dynamics. J. Geophys. Res. 112, E09001.
    CIRS data and the 2-dimensional IPSL circulation model. Icarus 197, 556–571.
                                                                                               Rodriguez, S. et al., 2009. Global circulation as the main source of cloud activity on
Donner, L. et al., 2011. The dynamical core, physical parameterizations, and basic
                                                                                                   Titan. Nature 459, 678–682.
    simulation characteristics of the atmospheric component AM3 of the GFDL
                                                                                               Rodriguez, S. et al., 2011. Titan’s cloud seasonal activity from winter to spring with
    global coupled model CM3. J. Climate 24 (13), 3484–3519.
                                                                                                   Cassini/VIMS. Icarus 216, 89–110.
Flasar, F.M. et al., 2005. Titan’s atmospheric temperatures, winds, and composition.
                                                                                               Roe, H.G., Brown, M.E., Schaller, E.L., Bouchez, A.H., Trujillo, C.A., 2005. Geographic
    Science 308, 975–978.
                                                                                                   control of Titan’s mid-latitude clouds. Science 310, 477–479.
Friedson, A.J., West, R.A., Wilson, E.H., Oyafuso, F., Orton, G.S., 2009. A global climate
                                                                                               Rothman, L. et al., 1996. The HITRAN molecular spectroscopic database and HAWKS
    model of Titan’s atmosphere and surface. Planet. Space Sci. 57, 1931–1949.
                                                                                                   (HITRAN atmospheric workstation): 1996 edition. J. Quant. Spectrosc. Radiat.
Frierson, D., 2007. The dynamics of idealized convection schemes and their effect on
                                                                                                   Trans. 60, 665–710.
    the zonally averaged tropical circulation. J. Atmos. Sci. 64 (6), 1959–1976.
                                                                                               Rothman, L. et al., 2009. The HITRAN 2008 molecular spectroscopic database. J.
Fulchignoni, M. et al., 2005. In situ measurements of the physical characteristics of
                                                                                                   Quant. Spectrosc. Radiat. Trans. 110 (9), 533–572.
    Titan’s environment. Nature 438, 785–791.
                                                                                               Schaller, E.L., Brown, M.E., Roe, H.G., Bouchez, A.H., Trujillo, C.A., 2006. Dissipation of
Gordon, C.T., Stern, W.F., 1982. A description of the GFDL global spectral model.
                                                                                                   Titan’s south polar clouds. Icarus 184, 517–523.
    Mon. Weather Rev. 110, 625–644.
                                                                                               Schaller, E.L., Roe, H.G., Schneider, T., Brown, M.E., 2009. Storms in the tropics of
Grifﬁth, C.A. et al., 2005. The evolution of Titan’s mid-latitude clouds. Science 310,
                                                                                                   Titan. Nature 460, 873–875.
    474–477.
                                                                                               Schinder, P.J. et al., 2011. The structure of Titan’s atmosphere from Cassini radio
Grifﬁth, C.A. et al., 2006. Evidence for a polar ethane cloud on Titan. Science 313,
                                                                                                   occultations. Icarus 215, 460–474.
    1620–1622.
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
528                                                              J.M. Lora et al. / Icarus 250 (2015) 516–528


Schinder, P.J. et al., 2012. The structure of Titan’s atmosphere from Cassini radio       Tomasko, M., Bézard, B., Doose, L., Engel, S., Karkoschka, E., 2008a. Measurements of
    occultations: Occultations from the Prime and Equinox missions. Icarus 221,               methane absorption by the descent imager/spectral radiometer (DISR) during
    1020–1031.                                                                                its descent through Titan’s atmosphere. Planet. Space Sci. 56 (5), 624–647.
Schneider, T., Graves, S.D.B., Schaller, E.L., Brown, M.E., 2012. Polar methane           Tomasko, M., Doose, L., Engel, S., Dafoe, L., West, R., Lemmon, M., Karkoschka, E., See,
    accumulation and rainstorms on Titan from simulations of the methane cycle.               C., 2008b. A model of Titan’s aerosols based on measurements made inside the
    Nature 481, 58–61.                                                                        atmosphere. Planet. Space Sci. 56 (5), 669–707.
Thompson, W.R., Zollweg, J.A., Gabis, D.H., 1992. Vapor–liquid equilibrium                Tomasko, M.G., Bézard, B., Doose, L., Engel, S., Karkoschka, E., Vinatier, S., 2008c.
    thermodynamics of N2 + CH4: Model and Titan applications. Icarus 97, 187–199.             Heat balance in Titan’s atmosphere. Planet. Space Sci. 56, 648–659.
Tokano, T., 2005. Meteorological assessment of the surface temperatures on Titan:         Toon, O., McKay, C., Ackerman, T., Santhanam, K., 1989. Rapid calculation of
    Constraints on the surface type. Icarus 173, 222–242.                                     radiative heating rates and photodissociation rates in inhomogeneous multiple
Tokano, T., 2009. Impact of seas/lakes on polar meteorology of Titan: Simulation by           scattering atmospheres. J. Geophys. Res. 94 (D13), 16287–16301.
    a coupled GCM-Sea model. Icarus 204, 619–636.                                         Turtle, E.P. et al., 2011a. Seasonal changes in Titan’s meteorology. Geophys. Res.
Tokano, T., 2010. Relevance of fast westerlies at equinox for the eastward                    Lett. 38, L03203.
    elongation of Titans dunes. Aeolian Res. 2, 113–127.                                  Turtle, E.P. et al., 2011b. Rapid and extensive surface changes near Titan’s equator:
Tokano, T., 2013. Wind-induced equatorial bulge in Venus and Titan general                    Evidence of April showers. Science 331, 1414–1417.
    circulation models: Implications for the simulation of superrotation. Geophys.        Vinatier, S. et al., 2007. Vertical abundance proﬁles of hydrocarbons in Titan’s
    Res. Lett. 40, 4538–4543.                                                                 atmosphere at 15°S and 80°N retrieved from Cassini/CIRS spectra. Icarus 188,
Tokano, T., Neubauer, F.M., Laube, M., McKay, C.P., 1999. Seasonal variation of               120–138.
    Titan’s atmospheric structure simulated by a general circulation model. Planet.       Vinatier, S., Rannou, P., Anderson, C., Bézard, B., De Kok, R., Samuelson, R., 2012.
    Space Sci. 47, 493–520.                                                                   Optical constants of Titan’s stratospheric aerosols in the 70–1500 cm1 spectral
Tokano, T. et al., 2006. Methane drizzle on Titan. Nature 442, 432–435.                       range constrained by Cassini/CIRS observations. Icarus 219 (1), 5–12.
```
