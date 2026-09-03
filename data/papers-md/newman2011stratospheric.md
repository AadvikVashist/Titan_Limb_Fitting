---
citation_key: "newman2011stratospheric"
title: "Stratospheric superrotation in the TitanWRF model"
source_pdf: "data/papers/newman2011stratospheric.pdf"
source_pdf_sha256: "aa551fb28148085b4676e3b836f8dae743e0948d111ba5bdaa9c2a84778174cf"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                                                                           Icarus 213 (2011) 636–654



                                                               Contents lists available at ScienceDirect


                                                                                  Icarus
                                                 journal homepage: www.elsevier.com/locate/icarus




Stratospheric superrotation in the TitanWRF model
Claire E. Newman a,⇑, Christopher Lee a, Yuan Lian a, Mark I. Richardson a, Anthony D. Toigo b
a
    Ashima Research, Suite 104, 600 South Lake Avenue, Pasadena, CA 91106, USA
b
    The Johns Hopkins University, Applied Physics Laboratory, 11100 Johns Hopkins Road, Laurel, MD 20723, USA



a r t i c l e          i n f o                          a b s t r a c t

Article history:                                        TitanWRF general circulation model simulations performed without sub-grid-scale horizontal diffusion of
Received 2 December 2010                                momentum produce roughly the observed amount of superrotation in Titan’s stratosphere. We compare
Revised 25 March 2011                                   these results to Cassini–Huygens measurements of Titan’s winds and temperatures, and predict tempera-
Accepted 25 March 2011
                                                        ture and winds at future seasons. We use angular momentum and transformed Eulerian mean diagnostics
Available online 5 April 2011
                                                        to show that equatorial superrotation is generated during episodic angular momentum ‘transfer events’
                                                        during model spin-up, and maintained by similar (yet shorter) events once the model has reached steady
Keywords:
                                                        state. We then use wave and barotropic instability analysis to suggest that these transfer events are pro-
Titan
Atmospheres, Dynamics
                                                        duced by barotropic waves, generated at low latitudes then propagating poleward through a critical layer,
                                                        thus accelerating low latitudes while decelerating the mid-to-high latitude jet in the late fall through early
                                                        spring hemisphere. Finally, we identify the dominant waves responsible for the transfers of angular
                                                        momentum close to northern winter solstice during spin-up and at steady state. Problems with our sim-
                                                        ulations include peak latitudinal temperature gradients and zonal winds occurring 60 km lower than
                                                        observed by Cassini CIRS, and no reduction in zonal wind speed around 80 km, as was observed by Huy-
                                                        gens. While the latter may have been due to transient effects (e.g. gravity waves), the former suggests that
                                                        our low (420 km) model top is adversely affecting the circulation near the jet peak, and/or that we require
                                                        active haze transport in order to correctly model heating rates and thus the circulation. Future work will
                                                        include running the model with a higher top, and including advection of a haze particle size distribution.
                                                                                                                           Ó 2011 Elsevier Inc. All rights reserved.




1. Introduction                                                                            circulation in which angular momentum is dispersed diffusively
                                                                                           (however weak the diffusion) cannot have an extremum of abso-
   A major feature of Titan’s stratosphere is the presence of strong                       lute angular momentum away from the boundaries, which in turn
superrotation (the atmosphere rotating many times faster than the                          means that zonal winds cannot exceed those implied by an angu-
surface) at both equatorial and higher latitudes. This has been in-                        lar-momentum-conserving circulation. For the equator, this tells
ferred from Voyager IRIS (e.g. Flasar et al., 1981, 2005) and Cassini                      us that westerly winds can only be produced if diffusion of angular
CIRS (e.g. Achterberg et al., 2008a, 2011) temperature observations,                       momentum is balanced by something other than the angular-
and from stellar occultation measurements (e.g. Hubbard et al.,                            momentum-conserving axisymmetric circulation, i.e., can only be
1993), as well as being measured directly from Earth using the                             produced by upgradient eddy ﬂuxes of angular momentum. Thus
Doppler shift of spectroscopic lines (e.g. Kostiuk et al., 2001) and                       equatorial stratospheric superrotation seems to require that angu-
the Doppler shift of the Huygens probe’s radio signal (e.g. Folkner                        lar momentum be transported either vertically from lower levels
et al., 2006). The presence of a westerly zonal jet in the winter                          or horizontally toward the equator by some type of eddies.
hemisphere is expected in an atmosphere with a single solsticial                           Gierasch (1975) and Rossow and Williams (1979) suggested a
Hadley cell (which can stretch almost from pole to pole due to Ti-                         plausible mechanism by which this might occur, with angular
tan’s slow rotation rate) as this mean meridional circulation trans-                       momentum gained from the low latitude surface, transferred into
ports angular momentum at upper levels into the winter                                     the stratosphere and redistributed to higher latitudes by the mean
hemisphere. However, the presence of strong superrotation at                               meridional circulation, then returned to low latitudes at upper
the equator requires a different explanation. Hide’s theorem (Hide,                        levels by poleward-propagating eddies produced in barotropical-
1969; Schneider, 1977) demonstrates that a steady axisymmetric                             ly-unstable regions at the (low latitude) edge of the zonal jets.
                                                                                               Attempts to realistically simulate this aspect of Titan’s strato-
    ⇑ Corresponding author.
                                                                                           sphere in atmospheric general circulation models (GCMs) have
                                                                                           met with mixed success. (Note that Titan’s equatorial troposphere
    E-mail addresses: claire@ashimaresearch.com (C.E. Newman), lee@ashimare-
search.com (C. Lee), lian@ashimaresearch.com (Y. Lian), mir@ashimaresearch.com             is also superrotating, but is not a focus of the present study.)
(M.I. Richardson), toigo@astro.cornell.edu (A.D. Toigo).                                   The ﬁrst 3D Titan GCM, that of the Laboratoire de Météorologie

0019-1035/$ - see front matter Ó 2011 Elsevier Inc. All rights reserved.
doi:10.1016/j.icarus.2011.03.025
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                    C.E. Newman et al. / Icarus 213 (2011) 636–654                                             637


Dynamique (LMD), produced signiﬁcant equatorial superrotation                domain, but could not be run with a global mother domain due to
(Hourdin et al., 1995). Their model atmosphere was overall very              the assumption of conformal grids. In a conformal grid, the map-
similar to that observed by Voyager and more recently by Cassini             ping from real world to model spacing at each gridpoint is the
with respectively the IRIS and CIRS instruments (Flasar et al.,              same in the x and y directions, which prevents it from extending
1981, 2005; Achterberg et al., 2008a, 2011). Unfortunately, the              all the way to the poles. Major modiﬁcations to produce a global
model that produced this result is no longer in use, and no in-depth         model thus included modifying the dynamical core to run with
analysis was ever published showing the mechanisms responsible               non-conformal grids (e.g. a simple cylindrical projection, lat–lon
for the strong equatorial superrotation. The LMD group currently             grid) and adding special treatment of polar boundary conditions
runs a 2D (latitude-height) model. This model produces realistic             and polar ﬁltering of high latitude regions to prevent instabilities.
amounts of superrotation by parameterizing the effects of eddies             Our original Titan GCM (TitanWRF v1, see Section 2.3) was based
based on results from their 3D GCM (e.g., Luz et al., 2003; Rannou           upon this version of global WRF.
et al., 2004). The many results from their 2D model and their                    Our modiﬁcations to the WRF model were then passed back to
impressive match to observations are not discussed further here,             the WRF developers at the National Center for Atmospheric Re-
however, as we are concerned primarily with the production of                search (NCAR), who over a period of time validated and improved
superrotation in 3D models by the eddies themselves, which 2D                on them, and they were subsequently included into a later WRF re-
models inherently cannot address.                                            lease (v3.0.1.1) so that the entire WRF community could beneﬁt.
    The inability of any group to reproduce the Hourdin et al. (1995)        This version of WRF also contained numerous other improvements
result became a common theme within the modeling community.                  and additions, as do all new releases.
The Köln (Cologne) Titan model (Tokano et al., 1999) suffers from                Signiﬁcantly, the NCAR team had identiﬁed some errors in our
weak superrotation (peak zonal winds of under 60 m/s, rather than            choice of map scale factors, predominantly relating to terms in
the observed 200 m/s), as does the TitanWRF v1 model (see Sec-              the geopotential equation and the calculation of meridional wind
tion 3.1.1) published in Richardson et al. (2007) and the Titan CAM          tendencies. Although representing a tiny fraction of the total
(Community Atmosphere Model) of Friedson et al. (2009). Another              changes made, these errors were sufﬁcient to produce overly-
model (Mingalev et al., 2006) achieves superrotation, but only by            strong wind and temperature gradients at high latitudes when
ﬁxing atmospheric temperatures, whereas in reality the circulation           the model spun up vigorously (see Section 3.1.1). TitanWRF v2 is
produced would act to swiftly disrupt their imposed temperature              therefore based on this improved version of global WRF produced
structure, making this model internally inconsistent and thus                by NCAR, and provides us with greatly improved results, as de-
unsatisfactory. Other published Titan models, while insightful for           scribed in Section 3.1.2.
tropospheric methane studies, are conﬁned to the troposphere
only and use highly simpliﬁed radiative transfer schemes (e.g.               2.3. The planetary WRF model
Mitchell et al., 2006; Mitchell, 2008). While various reasons for
failure had been suggested there remained a bias among modelers                  Once we had a global WRF base model, major modiﬁcations to
that the issue was fundamentally dynamical or numerical in nature            produce a planetary version (PlanetWRF) included removing hard-
(e.g. Friedson et al., 2009).                                                wired ‘Earth’ settings from parameterizations of physical processes
    This proved to be correct, in the TitanWRF model at least. As de-        (e.g., boundary layer mixing schemes) and adding planet-speciﬁc
scribed in Section 3.1.2, the most recent version of TitanWRF now            treatments of radiative transfer for Mars, Titan and Venus. Early re-
successfully simulates the observed magnitude of superrotation,              sults were shown in Richardson et al. (2007). The Titan version of
with the amount of imposed horizontal diffusion found to be the              PlanetWRF, TitanWRF, uses a two-stream radiative transfer model
critical factor involved in obtaining this result. Section 3.2 com-          to generate heating rates. Gas and haze optical properties are
pares TitanWRF results with available observations, while Sec-               found using a modiﬁed version of the scheme described by McKay
tion 3.3 describes the impact of resolution on results, and                  et al. (1989, personal communication 2004–2006). TitanWRF also
Section 3.4 shows TitanWRF predictions of the atmospheric circu-             uses a multi-layer subsurface heat diffusion solver, surface energy
lation in different seasons. Section 4 then provides details of the          balance solver, a non-local boundary layer diffusion solver, and
mechanisms that both generate and maintain TitanWRF’s equato-                horizontal subgrid-scale diffusion that can be prescribed (ﬁxed)
rial superrotation. In Section 5 we summarize our ﬁndings and dis-           or calculated from the resolved-wind deformation, though as dis-
cuss future work.                                                            cussed in Section 3.1 we ﬁnd optimum results when no horizontal
                                                                             diffusion is used. The model incorporates the impact of Saturn’s
2. PlanetWRF and the TitanWRF GCM                                            gravitational ﬁeld as Titan moves around Saturn in an eccentric or-
                                                                             bit (e.g. Tokano and Neubauer, 2002) by including tidal accelera-
2.1. Overview                                                                tions into the momentum equations. The model used in this
                                                                             work includes no spatial variations in topography, albedo, or ther-
   PlanetWRF is a multi-scale, planetary atmospheric model, devel-           mal inertia.
oped from the terrestrial, limited-area WRF (Weather, Research and               For this work, TitanWRF was run as a global, latitude–longitude,
Forecasting) model as described in Richardson et al. (2007). The             hydrostatic model with 54 r0 (modiﬁed-sigma) layers in the verti-
project began with the production of a global WRF model (see Sec-            cal from the surface to 420 km, where r0 = (P  Ptop)/(Psurf  Ptop),
tion 2.2), followed by the generalization of parameters and addition         with P = pressure. The standard horizontal resolution used was
of planetary physics to produce PlanetWRF and speciﬁcally the                5.625° in longitude and 5° in latitude, though results are also
TitanWRF model (see Section 2.3). Although all results shown here            shown using half this resolution in Section 3.3. At the standard
use TitanWRF as purely a global model (or GCM), TitanWRF may                 model resolution, one Titan year runs in 3.5 actual days using
also be run as a limited-area model, or as a multi-scale model with          24 processors on the Pleiades cluster at the NASA Ames High End
a global mother domain and embedded high-resolution ‘nests.’                 Computing center.
                                                                                 We typically begin new TitanWRF simulations with the atmo-
2.2. The global WRF model                                                    sphere at rest with respect to the surface of Titan, and with either
                                                                             a globally-uniform temperature proﬁle (based on the observed glo-
   When we began this project, the WRF model (v2.1.2 at that                 bal-mean proﬁle) or an isothermal atmosphere, the latter becom-
time) could be run with several nests embedded within the mother             ing very similar to the former in only tens of Titan days of
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
638                                                            C.E. Newman et al. / Icarus 213 (2011) 636–654


radiative forcing. The model is then run until the atmosphere is                        that only by setting the imposed horizontal diffusion to effectively
fully spun up, i.e., has reached a point at which the seasonal cycle                    zero could we obtain a winter hemisphere zonal jet with approxi-
approximately repeats from year to year. Exceptions are ‘restart’                       mately the observed location and magnitude.
simulations begun from the spun-up state of a previous run.                                This result suggests that the processes responsible for superro-
    For the remainder of this paper, ‘year,’ ‘day’ and ‘hour’ will refer                tation in Titan’s atmosphere are exceptionally sensitive, and are
to a Titan year, day and hour, as these have a direct relationship to                   adversely impacted by any horizontal smoothing of temperature
the timescales of the applied solar forcing on Titan whereas Earth                      and wind gradients that result from a diffusion scheme, far more
years, days and hours clearly do not. All units, such as ms1, will                     so than we see in Earth or Mars GCMs. This is most likely because
continue to be in SI, however.                                                          the radiative forcing on Titan is far weaker than on Earth or Mars,
                                                                                        thus any gradients produced are more easily destroyed.
                                                                                           The zonal-mean temperatures and zonal winds obtained after
3. Stratospheric superrotation in TitanWRF
                                                                                        nearly 5 years of using zero horizontal diffusion in TitanWRF v1
                                                                                        are shown in the bottom left of Figs. 1 and 2, respectively. They
3.1. The importance of low horizontal diffusion in producing signiﬁcant
                                                                                        show many similarities to the CIRS results shown in the top left
superrotation
                                                                                        plots. As already visible in these TitanWRF v1 results, however,
                                                                                        as jet speeds increased so too did unrealistically sharp gradients
3.1.1. Results using TitanWRF v1
                                                                                        near the poles, and the model became unstable soon after this
   We ran the TitanWRF v1 GCM for several years until the atmo-
                                                                                        point (Newman et al., 2008). These deﬁciencies had not shown
sphere reached a steady, ‘spun up’ state. However, as described in
                                                                                        up in our weakly superrotating Titan simulations or in simulations
Richardson et al. (2007), we originally produced about an order of
                                                                                        using the strongly radiative-forced MarsWRF model.
magnitude less superrotation than observed. Fig. 1 shows zonal-
mean temperatures, and Fig. 2 zonal-mean zonal winds, for the
period Ls  293°–323° as observed by Cassini CIRS (top left plots)                      3.1.2. Results using TitanWRF v2
and produced by three TitanWRF simulations. Our original Titan-                            During this period, the WRF development team at NCAR had
WRF v1 results are shown at the top right of Figs. 1 and 2, and re-                     incorporated our global WRF modiﬁcations into an upcoming
veal an atmosphere with very weak latitudinal temperature                               WRF release (which included numerous other desirable features,
gradients and a weak winter hemisphere zonal jet peaking at under                       such as greater parallelizability and thus shorter run times). In
30 m/s.                                                                                 doing so, they had identiﬁed and corrected a small number of er-
   We investigated possible reasons for this weaker-than-ob-                            rors in the map scale factors we introduced during our globaliza-
served superrotation, and were ﬁnally able to attribute it to the                       tion of the dynamical core (see Section 2.2). We thus integrated
amount of imposed (parameterized) horizontal diffusion inside                           our planetary modiﬁcations into this WRF release to produce
TitanWRF. Typically, GCMs are run with imposed horizontal diffu-                        TitanWRF v2, and again attempted to run the model with zero hor-
sion used to parameterize the effects of sub-grid scale eddy-mixing                     izontal diffusion until it reached a steady state.
of heat and momentum, though in practice these schemes also act                            TitanWRF v2 proved to be far more stable than TitanWRF v1 as
to prevent the growth of numerical instabilities. We experimented                       the high-latitude zonal jets develop: the model now spins up to
with different forms and magnitudes of diffusion (e.g., constant                        steady state without instabilities developing, with no sharp gradi-
coefﬁcient; Smagorinsky; hyper-diffusion), but ultimately found                         ents occurring near the poles, as shown in the bottom right plot of




Fig. 1. Zonal-mean temperatures (K) for Ls  293°–323°, retrieved from Cassini CIRS data (top left, provided by the Cassini dynamics team using the techniques of Achterberg
et al. (2008a)), and as modeled by a steady state year (year 12) of a TitanWRF v1 simulation with standard diffusion (top right), the ﬁnal year (year 4) of a TitanWRF v1
simulation with zero horizontal diffusion (bottom left), and a steady state year (year 75) of a TitanWRF v2 simulation with zero horizontal diffusion (bottom right). The
TitanWRF v1 simulation with zero horizontal diffusion became unstable and crashed during year 5.
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                                C.E. Newman et al. / Icarus 213 (2011) 636–654                                                              639




Fig. 2. As in Fig. 1, but showing zonal-mean zonal winds (m/s). The top left plot shows winds inferred from CIRS temperatures as in Flasar et al. (2005). The red line shows the
latitudinal limit of the gradient wind balance assumption, and winds are linearly interpolated across this region.



Fig. 2. A side-effect is a signiﬁcant increase in the time required to
spin-up the model from rest, 75 years, suggesting that the previous
errors were signiﬁcantly impacting our zero diffusion results. (Note
that we also performed TitanWRF v2 simulations using increased
horizontal diffusion, and veriﬁed that weak superrotation is still
produced, with results very similar to those produced using Titan-
WRF v1 with the same diffusion scheme and coefﬁcients.)
     Figs. 3 and 4 show the ‘superrotation index’ (the total angular
momentum of an atmospheric layer divided by the total angular
momentum of that layer at rest) for respectively years 1–15.5
and years 45–74 of a TitanWRF v2 simulation with zero imposed
horizontal diffusion. Note that plotting superrotation index gives
undue weight to low-density regions, compared with plotting
angular momentum itself (as in Fig. 11), thus emphasizing regions
with the largest zonal wind speeds rather than those with the larg-
est angular momentum. During the initial years of the simulation                           Fig. 4. As in Fig. 3, but showing years 45–74 only. The simulation reaches steady
there are long periods of rapid spin-up at pressures below 20 mbar,                        state after roughly 69 years.
i.e., net gains in atmospheric angular momentum at the expense of

                                                                                           the angular momentum of the solid surface (see Section 3.1). By
                                                                                           the ﬁnal years of the simulation, however, the spin-up is very grad-
                                                                                           ual, with net gains from year to year ﬁnally ceasing when the mod-
                                                                                           el reaches steady state at about 69 years in.
                                                                                               Superimposed on the overall pattern of spin-up is a bi-annual
                                                                                           oscillation, as seen more clearly in Fig. 5 showing results for a stea-
                                                                                           dy state year. In the uppermost 2 mbar of TitanWRF’s stratosphere,
                                                                                           superrotation peaks at 30° of Ls after equinox, and minima occur
                                                                                           30° after solstice; lower in the stratosphere, these timings are
                                                                                           shifted 30° later. This pattern is a result of the seasonal variation
                                                                                           in the mean meridional circulation, as shown in the mass stream-
                                                                                           function plots at the bottom of Figs. 8 and 9, which affects angular
                                                                                           momentum transfers within the atmosphere. Around equinox,
                                                                                           upwelling at low latitudes brings eastward angular momentum
                                                                                           (gained from the low latitude surface, see Section 3.1) up into
                                                                                           the upper stratosphere. Around solstice, upwelling (and the arrival
Fig. 3. Superrotation indices in four broad layers for the ﬁrst 15.5 years of a
standard horizontal resolution (5.625°  5°) simulation with zero horizontal
                                                                                           of eastward angular momentum) occurs at high summer latitudes,
diffusion and no top level damping. Layers are: 0.0087–2 mbar (solid), 2–20 mbar           but is more than compensated for by downwelling at the location
(dashed), 20–200 mbar (dotted) and 200 mbar-surface (dot-dashed).                          of the strong westerly jet (see Fig. 8 zonal wind plots) in the winter
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
640                                                                 C.E. Newman et al. / Icarus 213 (2011) 636–654


                                                                                             temperatures and zonal winds observed by Cassini CIRS (top left
                                                                                             of Figs. 1 and 2) in this season. Note that the CIRS results are com-
                                                                                             plicated by the need to deﬁne a lower boundary condition for their
                                                                                             wind retrievals. They set the wind at 10 mbar to four times the so-
                                                                                             lid body rotation rate, a rather large (albeit necessary) assumption,
                                                                                             and though this was chosen to be consistent with Huygens it
                                                                                             slightly underestimates the true wind speed there compared to
                                                                                             Huygens measurements (Folkner et al., 2006). As in the CIRS obser-
                                                                                             vations, peak TitanWRF v2 temperatures of over 200 K occur high
                                                                                             above the winter (north) pole, and a peak zonal wind jet with
                                                                                             winds well over 175 m/s is located between 75°N and 20°S, with
                                                                                             peak equatorial zonal winds of over 150 m/s. However, there are
                                                                                             several mismatches too, most obviously the difference in level be-
                                                                                             tween the observed and modeled temperature and wind maxima;
                    Fig. 5. As in Fig. 3, but showing year 75 only.
                                                                                             observed temperatures [winds] peak at 0.01 mbar [0.1 mbar],
                                                                                             while modeled temperatures [winds] peak at 0.08 mbar
                                                                                             [0.8 mbar], approximately 60 km (about one scale height) lower.
                                                                                             Another prominent difference is the simulated increase (rather
                                                                                             than the observed decrease) in temperature above 0.01 mbar. All
                                                                                             of the above may be a side-effect of the relatively low placement
                                                                                             of the model top with respect to the interesting atmospheric fea-
                                                                                             tures, or of the lack of active haze transport allowing radiative–
                                                                                             microphysical–dynamical feedbacks within the model atmosphere.
                                                                                             The northern high latitude heating (hence the associated wind jet)
                                                                                             in this season occurs during polar night and is a dynamical phe-
                                                                                             nomenon associated with adiabatic heating in the downward
                                                                                             branch of the solsticial Hadley cell, thus problems with this circu-
                                                                                             lation – due either to a low model top or to the lack of correct haze
                                                                                             forcing in the ‘driver’ region – may have led to the mismatch de-
                                                                                             scribed. Also, in reality the atmosphere of Titan is sufﬁciently ex-
Fig. 6. TitanWRF v2 zonal wind proﬁles from the surface up to 150 km altitude for            tended that signiﬁcant heating still occurs in the stratosphere at
the location and date of Huygens’s arrival (10°S, Ls = 300.5) for 64 different local
                                                                                             latitudes for which the surface is in polar night (e.g. Achterberg
times of day (solid lines) compared with Huygens Doppler wind measurements
(dotted line) taken from the PDS (Folkner et al., 2006).
                                                                                             et al., 2008a); however, we do not currently include this effect in
                                                                                             TitanWRF. These issues will be addressed in future work.
                                                                                                 Cassini CIRS observations also revealed a small (4°) displace-
                                                                                             ment of the axis of symmetry of the zonal mean ﬂow with respect
                                                                                             to the pole over a range of pressures (Achterberg et al., 2008b). We
                                                                                             ﬁnd negligible displacement of the pole in TitanWRF at the pres-
                                                                                             sures observed, and though we do produce a small but signiﬁcant
                                                                                             displacement of the axis of symmetry at higher altitudes it is un-
                                                                                             clear how or if this is related. However, given suggestions that
                                                                                             the observed displacement may increase with altitude (Achterberg
                                                                                             et al., 2008b) we may be partially capturing the effect, and this will
                                                                                             be investigated in future work.
                                                                                                 Fig. 6 shows TitanWRF v2 zonal wind proﬁles from the surface
                                                                                             up to 150 km altitude for the location and date of Huygens’s arrival
                                                                                             for 64 different local times of day, as well as the Huygens Doppler
                                                                                             wind measurements for comparison (Folkner et al., 2006). Well be-
                                                                                             low the large dip in measured zonal wind speed (centered at
                                                                                             80 km) we match observations reasonably well, with wind
                                                                                             speeds of 5 m/s at 20 km altitude, but by 60 km we predict wind
Fig. 7. As in Fig. 3, but for a simulation run at half the resolution (11.25°  10°). Just
                                                                                             speeds of 60 m/s rather than the 40 m/s observed, and fail to pre-
over the ﬁrst 28 Titan years are shown. The simulation reaches steady state after
roughly 24 years.                                                                            dict any dip in wind speed between 60 and 120 km. Well above
                                                                                             this dip, however, we again show a reasonable match to observa-
                                                                                             tions, predicting wind speeds of 115 m/s (only 10 m/s faster
hemisphere, which removes angular momentum from the upper                                    than observed) at 120 km.
stratosphere and returns it toward the surface. Hence there is a                                 Overall, high zonal winds appear to persist far lower in the
bi-annual peak in upper stratospheric angular momentum shortly                               model atmosphere than inferred by Cassini CIRS or measured di-
after both equinoxes, with a minimum shortly after both solstices.                           rectly by the Huygens probe, though it is possible that wave activ-
Angular momentum exchange and transport are discussed in more                                ity may have produced transient lower zonal winds (in particular
detail in Section 3.1.                                                                       the strong dip in wind speed centered at 80 km) during the Huy-
                                                                                             gens descent that are not representative of the background wind
3.2. Comparison between TitanWRF v2 results and observations                                 speed. Whether the wind speed dip is a persistent or transient fea-
                                                                                             ture, however, TitanWRF v2 does not reproduce it in simulations
   The steady state TitanWRF v2 results for Ls  293°–323°                                   for which we produce the observed strong stratospheric superrota-
(bottom right of Figs. 1 and 2) reveal many similarities to the                              tion, suggesting that we are either misrepresenting the distribution
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                               C.E. Newman et al. / Icarus 213 (2011) 636–654                                                            641




Fig. 8. Zonal-mean temperatures (in K, top row), zonal winds (in m/s, middle row) and mass streamfunction (in kg/s, bottom row) from a fully ‘spun up’ (steady state) year of
the standard TitanWRF v2 simulation (year 75), averaged over 12 Titan days around northern summer (Ls  90°, left column) and winter (Ls  270°, right column) solstice.
Positive streamfunction values indicate clockwise rotation.



of atmospheric heating or are missing some wave generation                               than 19). This suggests a dependence on horizontal resolution
mechanism. Gravity waves, which grow with height and break                               in terms of the generation and meridional propagation of waves re-
aloft, redistributing momentum, are estimated to be signiﬁcant                           quired to drive superrotation in TitanWRF (see Section 4).
in Titan’s atmosphere (Strobel, 2006). However, TitanWRF v2 only                             This result also motivates us to establish whether the peak
includes the effects of vertically-propagating waves that are re-                        superrotation achieved has converged by our standard resolution
solved at our standard resolution of 5° and generated over a                            of 5.625°  5°, or whether it would increase if resolution were in-
smooth surface; it does not currently include a parameterization                         creased further. We will investigate this in subsequent work. How-
of sub-grid-scale gravity waves or (in our standard model) any sur-                      ever, given the time requirements for our standard resolution
face topography.                                                                         simulation to reach steady state (8 months of real time), we will
    To summarize, it appears that explicit horizontal diffusion of                       most likely interpolate our steady state atmosphere from this sim-
heat and momentum, imposed within TitanWRF’s dynamical core                              ulation onto a higher-resolution (2.8125°  2.5°) grid, and proceed
to represent sub-grid scale mixing, was originally too high and                          from there, rather than begin with the atmosphere at rest.
was effectively dissipating the waves postulated by Gierasch
(1975) and Rossow and Williams (1979) to cause the equatorward                           3.4. Seasonal variations
angular momentum transport that accelerates the equatorial ﬂow.
Using zero horizontal diffusion, we now produce signiﬁcant latitu-                          Figs. 8 and 9 show TitanWRF predicted zonal-mean tempera-
dinal temperature gradients and superrotation, albeit with zonal                         tures, zonal winds and mass streamfunctions averaged over
winds and temperatures that peak roughly one scale height lower                          12 days surrounding the four cardinal seasons. Fig. 8 shows both
than observed in the real atmosphere.                                                    the northern summer (Ls  90°, left) and winter (Ls  270°, right)
                                                                                         solstices, and demonstrates the high degree of symmetry between
3.3. Impact of resolution on results                                                     these seasons, with temperatures and winds at Ls  270° resem-
                                                                                         bling the results already shown in Fig. 1 (which covers a slightly la-
   We also ran TitanWRF v2 with a horizontal resolution of                               ter time period). At solstice, the stratospheric circulation as shown
11.25°  10° (i.e., at half our standard resolution), and found the                      by the mass streamfunction consists of a nearly pole-to-pole Had-
impact on our results to be signiﬁcant. Fig. 7 shows the superrota-                      ley cell, with rising motion at high summer latitudes and descend-
tion index for the ﬁrst 32 years of this low-resolution simulation.                     ing motion above the winter pole. This descending motion
The model atmosphere now takes roughly 30 years to spin-up to a                          produces strong adiabatic heating in the winter hemisphere,
steady state, less than half the time required for our standard res-                     resulting in a temperature maximum high above the winter pole
olution simulation, however the peak superrotation index by this                         despite this portion of the model receiving no solar insolation at
point is slightly smaller (16 for the upper layer shown, rather                         this time of year. The peak temperatures produced aloft in the
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
642                                                      C.E. Newman et al. / Icarus 213 (2011) 636–654




                                Fig. 9. As in Fig. 8 but for northern spring (Ls = 0°, left) and fall (Ls = 180°, right) equinoxes.



northern hemisphere during local winter (top right plot) are stron-                 range shown (Hourdin et al.’s Fig. 8 is cut off at 0.3 mbar), the
ger than those produced during southern winter (top left plot),                     main difference being the greater wind magnitudes in TitanWRF
reﬂecting the stronger solar forcing during the former time period                  (a peak of 180 m/s versus their 120 m/s peak). The strong sim-
(which includes perihelion at Ls  278° (Hourdin et al., 1995)),                    ilarities, as well as the preliminary analysis of wave forcing that
which drives a stronger Hadley circulation (as shown by the                         they present, suggests that similar processes to those identiﬁed
streamfunction plots) and hence stronger downwelling and adia-                      in TitanWRF (see Section 4) may have been at work in their model
batic heating. However, this slight asymmetry does not appear to                    as well. Interestingly, Hourdin et al. (1995) state that parameters in
signiﬁcantly affect the overall latitudinal temperature gradient,                   their horizontal dissipation scheme were chosen ‘‘somewhat arbi-
or the zonal wind jets peaking at winter mid-latitudes that relate                  trarily,’’ suggesting that they may also have used quite low
to the temperature gradient via the thermal wind equation. Small                    amounts of horizontal diffusion without fully realizing the signiﬁ-
Ferrell cells also exist at high latitudes at northern summer, though               cance of this.
are far weaker or absent in southern summer.
   Fig. 9 shows both the northern spring (Ls  0°, left) and fall                   4. The mechanisms behind TitanWRF’s equatorial superrotation
(Ls  180°, right) equinoxes. (Cassini arrived in orbit around Saturn
after northern winter solstice at Ls  290°, so by late 2010 had just               4.1. Equatorward angular momentum transfer in the steady state and
covered the Ls  0° period shown.) At each equinox, two large Had-                  transient model atmosphere
ley cells dominate above 30 mbar, with rising motion at low lat-
itudes and descending motion over both poles. Below this,                               A major question for atmospheres with superrotating ﬂow at
however, the circulation is more complex, with the streamfunction                   the equator is how angular momentum builds up at low latitudes,
even reversing at some latitudes before transitioning to two smal-                  as described above. Having achieved our goal of simulating a real-
ler (less broad) Hadley cells that are less symmetric about the                     istically superrotating stratosphere, we can now analyze the model
equator in the troposphere. The bulk of the temperature and wind                    atmosphere to deduce the probable cause. Any analysis of Titan’s
ﬁeld is again quite symmetric in latitude (i.e., about the equator)                 atmosphere is complicated by Titan’s axial tilt (obliquity) which
within each season, though remants of the preceding solstice’s                      results in the seasonal variations in circulation shown in Figs. 8
high-latitude winter hemisphere temperature gradients and en-                       and 9. Unlike Venus, which has zero obliquity, the ‘steady state’
hanced zonal wind jets can be seen in their respective winter/                      (spun up) atmosphere is not approximately constant with time,
spring hemispheres.                                                                 but rather varies from season to season in an approximately
   The circulation produced by TitanWRF has many similarities to                    repeating fashion from year to year. Our initial analysis showed a
that produced by Hourdin et al. (1995) using the only other three-                  complex pattern of angular momentum variations, leading us to
dimensional GCM to have achieved signiﬁcant stratospheric super-                    use a ‘box model’ approach, that of splitting the atmosphere into
rotation. In particular, the shape of the zonal wind jets during                    six regions and considering angular momentum changes and trans-
northern winter and spring is extremely similar over the pressure                   ports within and between them. The six regions were produced by
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                               C.E. Newman et al. / Icarus 213 (2011) 636–654                                                           643


splitting the model domain vertically into two regions, and latitu-                     between this location and the penultimate model layer (at
dinally into three, as shown in Fig. 10. In the vertical we deﬁne the                   0.005 mbar). In the horizontal we deﬁne a southern region south-
‘lower atmosphere’ as the region between the surface and slightly                       ward of 22.5°S; an equatorial region as between 22.5°S and 22.5°N;
above the tropopause at 110 mbar, and the ‘upper atmosphere’ as                        and a northern region as northward of 22.5°N.




               Fig. 10. Schematic showing the ‘box model’ used to analyse the regional dependence of angular momentum variations in TitanWRF output.




Fig. 11. Total angular momentum, M in kg m2 s1, in the atmospheric regions deﬁned in Fig. 10, for a spin-up year of the standard TitanWRF v2 simulation (year 12). The
annual mean is subtracted so that all latitudinal regions can be clearly displayed on both plots (without the equatorial total angular momentum dominating). The upper plot
shows results summed over the upper atmosphere region for the southern (US, in blue), equatorial (UE, in green) and northern (UN, in red) latitude regions. The lower plot
shows results summed over the lower atmosphere region for the southern (LS, in blue), equatorial (LE, in green) and northern (LN, in red) latitude regions. Note that in the
print version of this article, the lines colored red, green and blue are shown respectively as light gray, dark gray and black.
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
644                                                                C.E. Newman et al. / Icarus 213 (2011) 636–654




Fig. 12. As Fig. 11, but showing the rate of change of total angular momentum, dM/dt in kg m2 s2, rather than M. Also shown in the bottom plot (as dotted lines) are the
surface torques (in kg m2 s2) exerted by the surface on the atmosphere, summed over the surface of each region.



    In Fig. 11 we plot the total angular momentum, M, summed                                where
over each region, with the mean subtracted for clarity. In Figs. 12–
14 we show the rate of change of total angular momentum, dM/dt,
                                                                                            t ¼ a cos /se ¼ a cos /qC d ðu2 þ v 2 Þ1=2 u                           ð3Þ
and the torque exerted by the surface on the atmosphere, T, again                           and where se is the surface zonal wind stress, u and v are respec-
summed over each atmospheric region (though only surface winds,                             tively the zonal and meridional wind speed at the surface, and Cd
etc. are required to calculate T). In each ﬁgure the upper atmo-                            is a drag coefﬁcient depending on wind speed and stability param-
sphere region is shown at the top and the lower atmosphere region                           eters in the planetary boundary layer (see e.g. Ponte and Rosen,
at the bottom, with the northern region in red1, the southern region                        1993). In practice, Cd was derived from TitanWRF’s surface layer
in blue, and the equatorial region in green. M and dM/dt are shown                          scheme (Hong and Pan, 1996).
with solid lines, while T are shown with dotted lines.
    The total angular momentum, M, in each model grid cell is given                         4.1.1. Angular momentum transfer during the spin-up period
by                                                                                              We are initially interested in how the fast equatorial superrota-
M ¼ Dm  ½u þ Xa cos /a cos /                                                              tion was achieved in the TitanWRF v2 model, so we begin by look-
                                                                                            ing at a year during rapid ‘spin-up’ of the model atmosphere, when
       a3 D/Dk
   ¼           ½u þ Xa cos / cos2 /jDPj                                            ð1Þ     the total angular momentum of the atmosphere is increasing sig-
           g
                                                                                            niﬁcantly each year (as angular momentum picked up at the sur-
where Dm is the mass of the grid cell in kg, u is the zonal wind                            face is distributed within the atmosphere). We choose year 12 of
velocity in m/s, X is Titan’s rotation rate in radians per second, a                        our standard TitanWRF v2 simulation, which took 69 Titan years
is Titan’s radius in m, / is latitude in radians, g is gravity in m/s2,                     to spin-up completely (reach a steady state) as shown in Figs. 3 and
k is latitude in radians, D/ and Dk are the size of the grid cell in                        4.
respectively latitude and longitude in radians, and DP is the pres-
sure thickness of the grid cell in Pa.                                                      4.1.1.1. The lower atmosphere. The bottom of Fig. 11 shows total
    The rate of change of total angular momentum, dM/dt, provides                           angular momentum, M, for the lower atmosphere regions. The ba-
a sense of the ‘acceleration’ and ‘deceleration’ of each region, and                        sic pattern for both the northern and southern regions is a gradual
dM/dt summed over the entire atmosphere should balance the                                  increase in M from mid-summer to spring equinox, followed by a
net torque exerted on the atmosphere by the surface, which is                               rapid decrease during spring. This pattern is partly due to local
determined by the pattern of surface zonal winds. The torque con-                           gains from / losses to the surface, as seen in the torques, T, shown
tribution from each surface grid cell, T, is given by                                       in Fig. 12 (dotted red and blue lines), which are in turn tied to the
                                                                                            pattern of surface zonal wind (not shown). Easterly winds result in
T ¼ ta2 cos /j D/Dk                                                                 ð2Þ
                                                                                            positive torques on the atmosphere, hence transfer of angular
                                                                                            momentum from surface to atmosphere occurs where surface eas-
 1
   For interpretation of color in Figs. 1, 2, 8–19, and 24, the reader is referred to the   terlies exist, and transfer from atmosphere to surface where there
web version of this article.                                                                are surface westerlies, with the torque increasing as latitude
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                            C.E. Newman et al. / Icarus 213 (2011) 636–654                                                645




                           Fig. 13. As Fig. 12, but for a steady state year of the same simulation (year 75).




Fig. 14. As Fig. 12, but for a second steady state year of the same simulation (year 76) for comparison with Fig. 13 (showing year 75).
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
646                                                  C.E. Newman et al. / Icarus 213 (2011) 636–654


decreases for a given zonal wind speed (see Eqs. (2) and (3)). The            4.1.2. Equatorial superrotation in the steady state atmosphere
northern and southern regions experience large negative torques,                  Fig. 13 shows the same as in Fig. 12, but now for year 75 of this
and thus lose signiﬁcant angular momentum to the surface, in                  TitanWRF v2 simulation. By this time the model atmosphere has
spring (when high-latitude surface westerlies peak) and summer                fully spun up (reached a steady state), thus we now focus on pro-
(when strong westerlies occur at mid-latitudes, exerting more tor-            cesses maintaining (rather than growing) the equatorial superrota-
que for a given wind speed), with very little net gain from the sur-          tion in the upper atmosphere. Note that, despite the variations
face during the remainder of the year. However, the lower                     shown, there is no net change in angular momentum of any region
atmosphere in the northern and southern regions (red and blue so-             over the course of several Titan years once the model has reached
lid lines, bottom of Fig. 11) also gains angular momentum from the            steady state. There are some notable differences between the spin-
equatorial lower atmosphere and from the stratosphere, producing              up and steady state years, the ﬁrst being the shorter duration of the
an increase in M (dM/dt > 0 in Fig. 12) from late summer through to           transfer events (apart from those occurring shortly before equinox)
early spring. The greatest gains occur during fall as the solsticial          in the upper atmosphere. It thus appears that, in its balanced state,
Hadley cell develops, transporting more angular momentum into                 the atmosphere experiences shorter periods of instability. Another
the fall hemisphere.                                                          notable difference is in the lower atmosphere, where in the spun-
    The equatorial lower atmosphere (green line, bottom of Fig. 11)           up year several large increases in angular momentum occur (at e.g.
shows a bi-annual oscillation in M, peaking 30° after equinox                Ls  110°, 140°, 265° and 340°), coinciding with angular momen-
with minima 30° after solstice. However, this is roughly anti-               tum decreases in both the northern and southern regions. Unlike
correlated with the torque exerted on this region of the atmo-                those in the upper atmosphere, these equatorial increases thus ap-
sphere by the equatorial surface (dotted green line, bottom of                pear to be due to equatorward angular momentum transfers from
Fig. 12). In other words, rather than accelerating the equatorial             both hemispheres, and will be investigated further in a subsequent
atmosphere, angular momentum gained from the surface at low                   paper devoted to the troposphere.
latitudes is rapidly transferred to higher latitudes in the fall/winter           It is interesting to compare the timings of these transfer events in
hemisphere or into the stratosphere. Surface easterlies thus con-             spin-up and steady state years (years 12 and 75, in Figs. 12 and 13,
tinue to dominate over most of the equatorial region, hence the               respectively) and also in two steady state years (years 75 and 76,
net torque on the atmosphere is always positive here. The torque              shown in Figs. 13 and 14, respectively). Although the seasonal pat-
peaks from late spring through late summer, when the solsticial               tern remains the same in all years, with transfer events always
Hadley cell drives strong cross-equatorial winds that are then                occurring between early fall and early spring in ﬁrst the southern
turned westward on their approach to the equator by the Coriolis              then northern hemisphere, the events do not generally occur at
effect. Note that observations of dune morphologies on Titan have             the same Ls, as might be expected if they were directly tied to sea-
been used to suggest the presence of surface westerlies at low lat-           sonal changes in the solar forcing. If, however, these events are ed-
itudes (e.g. Radebaugh et al., 2008), though we ﬁnd no evidence in            dies triggered by the gradual build-up of instabilities tied to a
TitanWRF of dominant low latitude surface westerlies at any time              slowly-varying circulation that has intrinsic interannual variability,
of year.                                                                      we would expect more variability in their timings, as demonstrated
                                                                              here. Interestingly, while years 75 and 76 differ considerably in the
4.1.1.2. The upper atmosphere. The upper atmosphere shows similar             timings of their southerly (from the south) transfer events – in fact,
seasonal trends in M and dM/dt (top of Figs. 11 and 12 respec-                year 76 is the only year shown with such an event well into spring –
tively), including the increase in angular momentum (dM/dt > 0)               the timings of their northerly transfers are remarkably similar until
through most of the fall to late winter period in the northern and            almost spring. However, this is a ﬂuke and does not occur in all years.
southern regions, producing peak M around or just after winter sol-               Now that we have identiﬁed when equatorward angular trans-
stice as the single, solsticial Hadley cell carries angular momentum          fer occurs, the next step is to establish how (Section 4.2) and why
into the winter hemisphere. However, these seasonal trends are                (Section 4.3) and ﬁnally to identify what waves are primarily
now interrupted by large perturbations in M, with increases in                responsible (Section 4.4).
equatorial M coinciding with decreases in M in one – but not both
– of the higher latitude regions (top of Fig. 11).                            4.2. Wave-driven angular moment transport in the TitanWRF model
    These perturbations are seen more clearly in dM/dt (top of
Fig. 12), with large maxima in equatorial dM/dt coinciding very                  We can demonstrate that the modeled up-gradient angular
cleanly with large minima in either the northern or southern re-              momentum transport toward the equator is due to eddies by
gion. We will henceforth refer to them as ‘transfer events,’ as they          examining the model output using the Transformed Eulerian Mean
clearly correspond to the transfer of angular momentum from one               (TEM) formulation of Andrews and McIntyre (1978). The zonal
hemisphere toward the equator. Which type of transfer event oc-               momentum equation can be rewritten using the TEM formulation
curs is completely controlled by the season, with equatorward                 to separate out the mean and eddy contribution, as given in e.g.
transport of angular momentum always coming from the fall/win-                Andrews et al. (1987). The equation becomes:
ter hemisphere, i.e., from the south (blue troughs and green peaks)                                                                 
for Ls  30°–210°, and from the north (red troughs and green                   d
                                                                                u                  1         d
                                                                                                              u cos /                       du 
                                                                                  ¼ v                                        f  w
peaks) for Ls  210°–30°. These major events episodically increase             dt            a   cos     /       d/                         dz
                                                                              |{z} |ﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ} |ﬄﬄ{zﬄﬄ}
the equatorial angular momentum in the upper atmosphere and,                   term1                       term2                        term3

despite other periods during which dM/dt becomes slightly nega-                                  1
tive, there is a net increase in equatorial angular momentum over                       þ                   rF þ |{z}X                             ð4Þ
                                                                                          q0 a cos /
the course of the year shown.                                                             |ﬄﬄﬄﬄﬄﬄﬄﬄﬄ {zﬄﬄﬄﬄﬄﬄﬄﬄﬄ } term5
                                                                                                   term4
    This result shows that, at least in TitanWRF v2, the processes
(presumably eddies) responsible for transporting angular momen-               where a is Titan’s radius, t is time, / is latitude, u is zonal wind, an
tum up-gradient back toward the equator – thus maintaining and                overbar indicates the zonal mean, f is the Coriolis parame-
growing the amount of equatorial superrotation – occur only spo-              ter = 2X sin /, where X = Titan’s rotation rate in radians/s, z = log
radically, and over relatively short periods of time (tens of Titan           pressure coordinate = ln(P0/P), where P0 is a reference pressure of
days or less), rather than being a continual process lasting through          1.44e5 Pa, X contains other processes (e.g., diffusion, damping), q0
a whole season or even the entire year.                                       is reference density = P/(R0 T0), where P = pressure, T0 = a reference
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                                                         C.E. Newman et al. / Icarus 213 (2011) 636–654                                                               647


pressure of 120 K and R0 is the gas constant for Titan’s atmosphere,                                                    saturated in plots b through d, the large degree of cancelation be-
and where F is the Eliassen-Palm ﬂux vector (see below). The resid-                                                     tween the different angular momentum transport processes (resid-
ual mean circulation components in the meridional and vertical                                                          ual mean meridional and vertical, and eddy driven) results in
directions are given by respectively                                                                                    relatively small net increases or decreases in zonal wind (plot a).
                     $        %                       $            %                                                    By comparing plots b–d with plot a, it is possible to attribute the
                1 d q0 v 0 h0                  1     d cos /v 0 h0                                                      acceleration of the mean ﬂow to each transport process.
v      v
     ¼                               
                                and w ¼ w þ
                                     
                q0 dz dh=dz                a cos / d/ dh=dz                                                               Fig. 15 shows the terms during the Ls 270° transfer event that
                                                                                                               ð5Þ      occurred during the spin-up year shown in Fig. 12. By comparing
                                                                                                                        plots a–d, we can identify the increase in equatorial zonal wind
           0                                                                                      
where indicates the perturbation from the zonal mean, indicates                                                         (term1, plot a) at 1 mbar as being due to eddy-driven angular
the residual circulation, v is meridional wind, w is vertical wind and                                                  momentum transport (plot d). The EP ﬂux divergence is positive
h is potential temperature. The Eliassen-Palm ﬂux divergence is gi-                                                     in this region, denoting acceleration of the mean ﬂow (see Eq.
ven by                                                                                                                  (4)), and negative at northern mid-to-high latitudes, denoting
                  1     d              dF z                                                                             deceleration of the mean ﬂow. This pressure level corresponds
rF ¼                     ½f/ cos / þ      ;                        where                                     ð6Þ      roughly to the base of the zonal jet peak at this point in this sim-
               a cos / d/              dz
                                                                                                                        ulation (not shown).
                      $                                          %                                                          By contrast, Fig. 16 shows the same terms during the ‘gap period’
                        v 0 h0 du=dz                                                                                   (beginning at Ls 278°) that followed this transfer event. Now there is
F / ¼ q0 a cos /                                     v   0 u0       and                                       ð7Þ
                          dh=dz                                                                                        no signiﬁcant equatorial acceleration of the zonal wind (plot a) and
                                                                                                                        no signiﬁcant eddy transport in this region (plot d). The low latitude
                      (                                        )
                                 1       cos /Þ v 0 h0
                                      dðu                                                                               acceleration produced just south of the equator at 0.1 mbar by the
F z ¼ q0 a cos /        f                                w0 u0                                               ð8Þ      residual mean vertical circulation (plot c) is more than canceled out
                              a cos /     d/      dh=dz
                                                                                                                        by the residual mean meridional ﬂow (plot b).
   Figs. 15 and 16 show the ﬁrst four terms in the zonal momen-                                                             Figs. 17 and 18 show the EP ﬂux vectors F ¼ ½F/; Fz (Eqs. (7) and
tum equation (Eq. (4)) averaged over respectively the Ls  270°                                                         (8)) plotted over term4 (Eq. (4)) and focusing on the upper atmo-
transfer event and following (‘gap’) period during spin-up year                                                         sphere region (which includes 1 mbar). Fig. 17 shows the EP ﬂuxes
12. In the absence of additional forces (such as damping and diffu-                                                     for the same transfer event examined in Fig. 15, while Fig. 18
sion), the zonal-mean acceleration of an atmospheric region (the                                                        shows them for the same gap period as in Fig. 16. During the trans-
LHS of the equation, term1) should be equal to the sum of acceler-                                                      fer event, the EP ﬂux vectors appear horizontal and show propaga-
ation due to meridional (term2) and vertical (term3) transport by                                                       tion of wave activity from equatorial and low southern latitudes
the residual mean circulation plus the acceleration due to non-                                                         toward mid-to-high northern latitudes. This, along with the direc-
angular-momentum conserving eddies (term4). Thus plot a in each                                                         tion of momentum transport (i.e., toward the equator), suggests
ﬁgure should be equal to the sum of the other three, with plot d                                                        that the waves transport westward angular momentum from the
representing wave-driven angular momentum transport. In both                                                            equator toward the northern hemisphere, thus accelerating the
Figs. 15 and 16, the strong Hadley circulation at this time of year,                                                    equatorial region and decelerating higher northern latitudes (e.g.,
with overall rising motion in the south and ﬂow from south to                                                           Schneider and Liu, 2009). By contrast, the EP ﬂuxes are much smal-
north aloft, is responsible for the pattern of plots b and c, which                                                     ler during the following gap period, implying little wave-driven
relate to respectively v   and w  . Although the contours are                                                     angular momentum transport.



                                                                                        a                                                             b
                                                           0.01                                                          0.01




                                   Pressure (mbar)
                                                           0.10                                                          0.10
                                                           1.00                                                          1.00
                                                       10.00                                                            10.00
                                                      100.00                                                           100.00
                                                     1000.00                                                          1000.00
                                                                       -60    -30       0        30       60                       -60      -30       0     30      60
                                                                                        c                                                             d
                                                           0.01                                                          0.01




                                   Pressure (mbar)
                                                           0.10                                                          0.10
                                                           1.00                                                          1.00
                                                       10.00                                                            10.00
                                                      100.00                                                           100.00
                                                     1000.00                                                          1000.00
                                                                       -60   -30     0    30              60                       -60     -30     0    30          60
                                                                             Latitude (deg N)                                              Latitude (deg N)

                                                                      -8.00e-07               -4.00e-07                0.00                4.00e-07              8.00e-07


                                                          -1.00e-06               -6.00e-07               -2.00e-07             2.00e-07              6.00e-07              1.00e-06

                                                                                                 2
Fig. 15. Terms in the TEM zonal momentum equation (Eq. (4)) in ms                                     averaged over 12 days during the Ls  270°, spin-up year 12 transfer event shown in Figs. 11 and 12.
See text for details. (a) term1; (b) term2; (c) term3; (d) term4.
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
648                                                                             C.E. Newman et al. / Icarus 213 (2011) 636–654


                                                                              a                                                             b
                                                 0.01                                                          0.01




                            Pressure (mbar)
                                                 0.10                                                          0.10
                                                 1.00                                                          1.00
                                                10.00                                                         10.00
                                               100.00                                                        100.00
                                              1000.00                                                       1000.00
                                                            -60     -30       0        30       60                       -60      -30       0     30      60
                                                                              c                                                             d
                                                 0.01                                                          0.01




                            Pressure (mbar)
                                                 0.10                                                          0.10
                                                 1.00                                                          1.00
                                                10.00                                                         10.00
                                               100.00                                                        100.00
                                              1000.00                                                       1000.00
                                                            -60    -30     0    30              60                       -60     -30     0    30          60
                                                                   Latitude (deg N)                                              Latitude (deg N)

                                                            -8.00e-07               -4.00e-07                0.00                4.00e-07              8.00e-07


                                                -1.00e-06               -6.00e-07               -2.00e-07             2.00e-07              6.00e-07              1.00e-06

      Fig. 16. As Fig. 15, but averaged over 12 Titan days in the ‘gap period’ (beginning at Ls  278°) following the Ls  270° transfer event shown in Figs. 11 and 12.




Fig. 17. EP ﬂux divergence term (term4 in Eq. (4) in ms2, shown as contours) and EP ﬂux vectors (in kg s2, shown in arrows) during spin-up year 12, averaged over 12 Titan
days during the Ls  270° transfer event shown in Fig. 15. Note that the pressure axis shows the correct pressure range, but is only approximate as it assumes a ﬁxed
logarithmic relationship between pressure and height over this region.



4.3. Critical layers and wave breaking                                                                        cases, variables were output 48 times per day. As in e.g. Randel and
                                                                                                              Held (1991), we ﬁrst ran a Fourier analysis on the eddy momentum
   The next step is to understand why the eddies produced in the                                              ﬂux (u0 v0 ) and calculated each wave’s phase speed (which is a func-
TitanWRF v2 atmosphere have the above-described impact on the                                                 tion of latitude and frequency). Then using bins 1/100th the size of
mean ﬂow. According to linear theory, eddies are expected to break                                            the peak zonal mean wind speed we found the total amplitude in
and deposit angular momentum into the mean ﬂow in ‘critical lay-                                              each phase speed bin (i.e., summing over all waves with phase
ers,’ which exist where the eddies’ zonal phase speed is close to                                             speeds in that range, and over all wavenumbers present). Also shown
the background zonal wind speed (e.g., Yamamoto and Takahashi,                                                in each plot is the zonal-mean zonal wind for the same pressure level
2004, 2006; Mitchell and Vallis, 2010). Fig. 19 shows the eddy                                                and time period. The wave amplitudes are clearly far weaker during
momentum ﬂux cospectra as a function of phase speed and latitude                                              the gap periods (plots b and d) than during the transfer events (plots
for the 0.8 mbar pressure level during the Ls  270° transfer events                                          a and c), with faster and stronger waves present during the steady
and subsequent gap periods in spin-up year 12 and steady state year                                           state transfer event (plot c) than during spin-up (plot a).
75. For year 12 these periods were longer, and 12 days of data were                                               During both transfer events shown in Fig. 19, the strong eddy
used, whereas in year 75 only 6 days were used in each case; in all                                           momentum ﬂux is convergent at the equator (u0 v0 < 0 in the
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
                                                                 C.E. Newman et al. / Icarus 213 (2011) 636–654                                                               649




                                                    Fig. 18. As Fig. 17, but for the Ls  278° ‘gap period’ shown in Fig. 16.




Fig. 19. Phase speed-latitude plot of eddy momentum ﬂux cospectra (in m2 s2, shown as contours) and zonal-mean zonal wind (in ms1, thick black line) at 0.8 mbar,
averaged over all zonal wavenumber for four different times. Top left: spin-up year 12, Ls  270° transfer event. Top right: spin-up year 12, Ls  278° ‘gap’ period. Bottom left:
steady state year 75, Ls  270° transfer event. Bottom right: steady state year 75, Ls  278° ‘gap’ period.


northern hemisphere, and far weaker but positive in the southern                            mean ﬂow at low latitudes. These waves then appear to cross the
hemisphere) and the waves are moving faster than the background                             critical layer (given by the zonal-mean zonal wind line) at 25°–
ﬂow, thus they deposit eastward momentum and accelerate the                                 30°N and remain coherent up to 50°–60°N. Once they have
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
650                                                            C.E. Newman et al. / Icarus 213 (2011) 636–654


crossed the critical layer the waves are traveling westward with re-                     16.3 per day. The higher frequency (hence phase speed) of the
spect to the mean ﬂow, so now deposit westward angular momen-                            dominant waves in the spun-up model is clearly linked to the fas-
tum into the mean ﬂow, decelerating it at these higher latitudes.                        ter superrotation of the atmosphere by this time. These waves are
This is entirely consistent with the transfers of eastward (positive)                    also completely absent in the following gap periods (not shown),
angular momentum from the northern to the equatorial region at                           thus are strong candidates for increasing superrotation during
this time of year, as shown in Figs. 12–14.                                              the transfer event.
                                                                                            Figs. 20 and 21 show the u, v and T amplitudes of the wavenum-
4.4. The nature of the eddies                                                            ber 1, 5.1 per Titan day and wavenumber 1, 16.3 per Titan day
                                                                                         waves during the spin-up and steady state transfer events, respec-
   In the above sections we focused on the impact of eddies on the                       tively. Noticeable in all plots is the concentration of peak ampli-
mean ﬂow. We now look at the eddies themselves, focusing on                              tudes around the 1 mbar level, with the waves mostly conﬁned
those most important for generating or maintaining superrotation                         between 0.2 and 3 mbar. Also interesting is that, while peak wave
during the transfer event periods shown in Fig. 19. A Fourier                            amplitudes in u and T occur above the equator, peak amplitudes in
analysis of the perturbation zonal wind, meridional wind and                             v occur at 30°–40°N.
temperature ﬁelds output 48 times per day shows large west-                                 Figs. 22 and 23 show the same waves in the eddy momentum
ward-propagating diurnal and semi-diurnal tides, which are ex-                           ﬂux (u0 v0 ), with peak amplitudes occurring between 0 and 50°N.
pected to decrease rather than increase the superrotation of the                         Again, their ‘pancaked’ appearance, now trapped between 0.5
equatorial region. The analysis also reveals several large-amplitude                     and 2 mbar in pressure, is to be expected for a strongly stratiﬁed
eastward-propagating waves, dominated by wavenumber 1, with                              atmosphere.
some wavenumber 2 but very few wavenumber 3 (or higher)                                     Aside from the structure and frequency of these global waves,
modes with any signiﬁcant spectral power.                                                another important question is how they are formed. Are they pro-
   During the spin-up period transfer event, when the background                         duced by barotropic instabilities, as suggested by e.g. Gierasch
zonal wind peaked at 70 ms1, the frequencies of the largest                            (1975) and Rossow and Williams (1979), or by some other mecha-
wavenumber 1 waves ranged from 5 to 7.5 per Titan day (rela-                           nism? Whatever the mechanism, it seems clear that it generates
tive to a ﬁxed longitude on the surface). We identiﬁed the wave                          waves which then affect the mean ﬂow (as demonstrated by e.g.
with phase speeds closest to the critical layer at the most latitudes                    the change in the zonal-mean zonal wind following a transfer
as having a frequency of 5.1 per Titan day. Similarly, for the steady                    event, Fig. 19) so as to remove or mitigate the conditions responsi-
state transfer event, when the background zonal wind peaked at al-                       ble for producing the instability.
most 190 ms1, the largest wave (which also had a phase speed                               The condition for quasi-geostrophic barotropic instability (Kuo,
close to the critical layer at many latitudes) had a frequency of                        1949, 1973) is that




Fig. 20. Perturbation zonal wind (u0 , left), meridional wind (v0 , middle) and temperature (T0 , right) wave amplitudes for the largest eastward-propagating wave with phase
speed close to the mean zonal wind speed during the Ls  270° transfer event in spin-up year 12. Wavenumber = 1, phase frequency = 5.08333 per day.
```

<!-- PDF_PAGE: 16 -->

## PDF page 16

```text
                                                                C.E. Newman et al. / Icarus 213 (2011) 636–654                                                               651




Fig. 21. As in Fig. 20 but for the Ls  270° transfer event in steady state year 75. Wavenumber = 1, phase frequency = 16.3 per day. Contour intervals are four times larger than
in Fig. 20, reﬂecting the larger wave amplitudes in the spun-up model.




                              Fig. 22. Eddy momentum ﬂux (u0 v0 ) amplitude in m2 s2 of the year 12 (spin-up period) wave shown in Fig. 20.




        2
@2 
   u=dy  b ¼ 0                                                                  ð9Þ       sphere, but it may provide some insight into how atmospheric
somewhere in the domain, where b ¼ df =dy ¼ 2X cos   /
                                                      . This is unsat-                     instabilities build over time. We thus deﬁne a ‘barotropic instability
                                                   a
isfactory for Titan, as the moon’s slow rotation rate means that the                       parameter,’ 1, as the left hand side of Eq. (9), and plot this parameter
geostrophic approximation in not valid through most of the atmo-                           in Fig. 24, averaged over the upper atmosphere region for spin-up
```

<!-- PDF_PAGE: 17 -->

## PDF page 17

```text
652                                                            C.E. Newman et al. / Icarus 213 (2011) 636–654




                                            Fig. 23. As Fig. 22 but for the year 75 (steady state period) wave shown in Fig. 21.




Fig. 24. Barotropic instability parameter, 1, averaged over the 24 model levels lying within the upper atmosphere region deﬁned in Fig. 10 (in m1 s1, shown in contours),
and rate of change of angular momentum, dM/dt, in the equatorial upper atmosphere (solid black line), for spin-up year 12. Note that the y-axis positioning of dM/dt has been
arbitrarily chosen to aid comparison with changes in 1.



year 12. Also shown for comparison of timings is dM/dt for the                           zonal wind jets and equatorial superrotation in TitanWRF’s strato-
equatorial upper atmosphere in the same year (previously shown                           sphere. The model takes 69 Titan years to spin-up from rest, i.e.,
as the green line in the top plot of Fig. 9).                                            reach a steady state in which the seasonal circulations approxi-
   The QG barotropic instability criterion is met (1 = 0) when the                       mately repeat from year to year. During spin-up, equatorial super-
contours change from pale green to yellow-green. Although the                            rotation is generated by episodic angular momentum transfer
signiﬁcance is unclear, it is interesting to note that the transfer                      events due to non-conservative waves which, as shown by e.g. EP
events (sharp increases in dM/dt) appear to occur immediately                            ﬂux diagnostics, transport westward angular momentum from
after such color changes occur, when ‘ﬁngers’ of high 1 extend into                      low latitudes into the late fall through early spring hemisphere,
otherwise ‘green’ (1 < 0) low-latitude regions, at 0°–10°S from                         accelerating the low latitudes and decelerating the fall/winter jet
Ls  30° to 210°, and from 0° to 10°N from Ls  210° to 30°. Fur-                       core. Once the model has reached steady state the transfer events
ther work is clearly required, however, to tie these waves deﬁni-                        are generally shorter in duration, suggesting that in its balanced
tively to a barotropic origin.                                                           state the atmosphere experiences shorter periods of instability.
                                                                                            Del Genio et al. (1993) performed simulations with a simpliﬁed
5. Discussion and further work                                                           Earth GCM run at Titan- and Venus-like rotation rates, and found a
                                                                                         weakly dissipative environment to be vital for the production of
   By removing any imposed sub-grid scale horizontal diffusion                           superrotation, though focused on suppression of vertical mixing
from the TitanWRF v2 general circulation model we are able to                            (below planetwide clouds) rather than on reduced horizontal mix-
simulate realistically strong latitudinal temperature gradients,                         ing. Del Genio and Zhou (1996) found the reduction of numerical
```

<!-- PDF_PAGE: 18 -->

## PDF page 18

```text
                                                      C.E. Newman et al. / Icarus 213 (2011) 636–654                                                                653


dissipation to be vital to producing superrotation in a Venus-like             we are naturally keen to demonstrate that our solution to the
model, though not for Titan (where results were more robust).                  superrotation problem – and our ﬁndings regarding the mecha-
However, as in Mitchell and Vallis (2010), all these experiments               nisms at work – are robust, rather than limited to one extant
were performed with the seasonal cycle removed, i.e., with solar               numerical model. For this reason we are developing a second Titan
forcing that did not vary with time. This is signiﬁcant, as we ﬁnd             GCM, based on the highly-efﬁcient cubed-sphere MITgcm (Adcroft
huge differences between the spin-up of TitanWRF with time-                    et al., 2004), and will repeat our experiments using this model in
invariant versus seasonally-varying forcing. Using time-invariant              the near future.
forcing similar to that of Del Genio et al. or Mitchell and Vallis,               An obvious validation approach is to compare our predicted
we ﬁnd that superrotation (peaking high above the equator) builds              wave types with those observed; however, the rather sparse nat-
rapidly, reaching over a hundred m/s in just a few Titan years, and            ure of Cassini-based observations makes it difﬁcult to look for
that this occurs even if signiﬁcant horizontal diffusion is present. It        short-lived episodic events in the data, particularly as a compar-
therefore seems that the mechanisms generating the initial super-              ison of steady state years 75 and 76 suggests that, aside from
rotation are quite different between these constant forcing models             their broad seasonal dependence, the precise timing of transfer
and the more realistic TitanWRF simulations presented here, in                 events varies signiﬁcantly from year to year, so we cannot predict
which similar mechanisms appear to be at work during the spin-                 exactly what to look for at a given Ls. In addition, CIRS data can
up and steady state periods.                                                   only be used to produce temperature maps separated by Earth
    The mechanisms maintaining superrotation once all models                   weeks, where 1 Earth week is roughly half a Titan day. This
have reached steady state appear to have more similarities, with               would thus allow the identiﬁcation of waves with longer periods,
barotropic instabilities and the generation and propagation of                 but effectively rules out the precise identiﬁcation of waves with
low wavenumber waves appearing to play key roles in both con-                  periods that are much shorter than a Titan day, as we predict will
stant forcing models (e.g., Del Genio and Zhou, 1996; Mitchell                 dominate.
and Vallis, 2010) and in models with seasonal forcing (Hourdin                    Remaining problems with our modeled atmosphere include the
et al., 1995; this paper). However, the model ﬁnal states look quite           mismatch between the observed and modeled height (and to a les-
different: for example, constant forcing models do not have sea-               ser extent magnitude) of the zonal jet peak, which is 60 km too
sons, so never produce jets that are asymmetric about the equator              low and 10 m/s too weak in TitanWRF compared with Cassini
(as shown in Fig. 8). Identifying similarities between the mecha-              CIRS data. These issues may be related to the relatively low model
nisms thus requires more investigation, and will be dealt with in              top, which may be impeding the upper atmosphere circulation
a follow-up paper.                                                             hence affecting adiabatic heating in the winter hemisphere and
    Looking in more detail at the eddies and processes involved in             thus temperature gradients and winds. They may also be due to
superrotation for TitanWRF, analysis of selected transfer events re-           the lack of active haze transport, which precludes feedbacks be-
veals the dominant waves at these times of year to be eastward                 tween haze microphysics, the circulation (via advection of haze
propagating, zonal wavenumber 1 waves, with frequencies of                     particles) and radiative transfer. Rannou et al. (2004) showed that
5–7.5 per Titan day and 16.3 waves per Titan day, respectively.              such coupling of haze microphysics and dynamics in their 2D mod-
These waves have convergent eddy momentum ﬂuxes at low lati-                   el was able to reinforce their meridional circulation, increasing the
tudes, and thus accelerate the mean ﬂow there before propagating               strength of superrotation and better reproducing observations.
polewards through a critical layer in which the mean ﬂow exceeds               This suggests that a similar result may be produced in a 3D GCM,
the phase speed of the waves. Linear theory predicts that at this              in which additional feedbacks will be present between haze and
point the waves should begin to break, depositing westward                     eddies. Future work will therefore include the upwards extension
momentum and decelerating the mean ﬂow in the jet core. To date                of the model domain and the inclusion of haze microphysics and
we have analyzed only a small fraction of our total dataset (which             advection.
includes dozens of transfer events per year, and tens of spin-up and
steady state years), and we will explore this dataset more rigor-
ously in the near future. However, our preliminary analysis sug-               Acknowledgments
gests that the relevant eddies responsible for superrotation
always form at low latitudes between 10 and 0.01 mbar, propa-                    We thank Christopher McKay for providing us with his updated
gate across the equator into the early fall through early spring               Titan radiative transfer scheme. We thank the WRF model develop-
hemisphere, decelerate the zonal ﬂow there and accelerate the zo-              ment teams at NCAR, particularly Jimy Dudhia, Joseph Klemp, John
nal ﬂow in their source region, and thus spin-up (or maintain                  Michalakes and William Skamarock, for their vital participation
superrotation of) the equatorial stratosphere.                                 and assistance in extending WRF to run as a global model. We also
    The transfer events appear to follow (or at least coincide with)           thank Cassini CIRS team members Peter Gierasch, Richard Achter-
low-latitude increases in barotropic instability, thus there is some           berg, Barney Conrath and Michael Flasar, for providing us with
suggestion that the waves responsible for generating and main-                 CIRS retrievals of temperature and inferred zonal wind, and for
taining superrotation are generated by this mechanism. However,                valuable discussions. We also thank our two anonymous reviewers
our analysis is not yet conclusive, and identifying the mechanism              for their very helpful and detailed comments. This work was
responsible for producing the waves, and hence superrotation, is               funded by grants from NASA’s Outer Planets Research program
the primary goal of future work and will be detailed in a follow-              and the NASA Astrobiology Institute, and simulations were per-
up paper. Other outstanding questions include: How do the domi-                formed on the CITerra cluster in the GPS division at Caltech and
nant wavenumbers and frequencies change as the model evolves                   on the Pleiades cluster at the High End Computing facility at NASA
during spin-up toward steady state? At steady state, what waves                Ames.
are responsible for equatorward momentum transport at different
seasons? How consistent are the dominant wavenumbers and fre-
quencies from year to year?                                                    References
    A major question here, as with all model-based results, is
whether the features and mechanisms identiﬁed in TitanWRF are                  Achterberg, R.K., Conrath, B.J., Gierasch, P.J., Flasar, F.M., Nixon, C.A., 2008a. Titan’s
                                                                                  middle-atmospheric temperatures and dynamics observed by the Cassini
those that actually occur on Titan itself. Given the initial difﬁculties          Composite Infrared Spectrometer. Icarus 194, 263–277. doi:10.1016/
experienced by ourselves, and by several other modeling groups,                   j.icarus.2007.09.029.
```

<!-- PDF_PAGE: 19 -->

## PDF page 19

```text
654                                                                  C.E. Newman et al. / Icarus 213 (2011) 636–654


Achterberg, R.K., Conrath, B.J., Gierasch, P.J., Flasar, F.M., Nixon, C.A., 2008b.            Luz, D., Hourdin, F., Rannou, P., Lebonnois, S., 2003. Latitudinal transport by
    Observation of a tilt of Titan’s middle-atmospheric superrotation. Icarus 197,                barotropic waves in Titan’s stratosphere. II. Results from a coupled dynamics–
    549–555. doi:10.1016/j.icarus.2008.05.014.                                                    microphysics–photochemistry GCM. Icarus 166, 343–358.
Achterberg, R.K., Gierasch, P.J., Conrath, B.J., Flasar, F.M., Nixon, C.A., 2011. Temporal    McKay, C.P., Pollack, J.B., Courtin, R., 1989. The thermal structure of Titan’s
    variations of Titan’s middle-atmospheric temperatures from 2004 to 2009                       atmosphere. Icarus 80, 23–53.
    observed by the Cassini/CIRS. Icarus 211, 686–698. doi:10.1016/                           Mingalev, I.V., Mingalev, V.S., Mingalev, O.V., Kazeminejad, B., Lammer, H., Biernat,
    j.icarus.2010.08.009.                                                                         H.K., Lichtenegger, H.I.M., Schwingenschuh, K., Rucker, H.O., 2006. First
Adcroft, A., Campin, J.-M., Hill, C., Marshall, J., 2004. Implementation of an                    simulation results of Titan’s atmosphere dynamics with a global 3-D non-
    atmosphere–ocean general circulation model on the expanded spherical cube.                    hydrostatic circulation model. Ann. Geophys. 24, 2115–2129.
    Mon. Weather Rev. 132 (12), 2845–2863.                                                    Mitchell, J.L., 2008. The drying of Titan’s dunes: Titan’s methane hydrology and its
Andrews, D.G., McIntyre, M.E., 1978. Generalized Eliassen-Palm and Charney-                       impact on atmospheric circulation. J. Geophys. Res. (Planets) 113, E12.
    Drazin theorems for waves on axisymmetric mean ﬂows in compressible                           doi:10.1029/2007JE003017.
    atmospheres. J. Atmos. Sci. 35, 175–218.                                                  Mitchell, J.L., Vallis, G.K., 2010. The transition to superrotation in terrestrial
Andrews, D.G., Holton, J.R., Leovy, C.B., 1987. Middle Atmosphere Dynamics.                       atmospheres. J. Geophys. Res. 115, E12008. doi:10.1029/2010JE003587.
    Academic Press, Orlando, FL.                                                              Mitchell, J.L., Pierrehumbert, R.T., Frierson, D.M.W., Caballero, R., 2006. The dynamics
Del Genio, A.D., Zhou, W., Eichler, T.P., 1993. Equatorial superrotation in a slowly              behind Titan’s methane clouds. Proc. Natl. Acad. Sci. USA 103 (49), 18421–18426.
    rotating GCM: Implications for Titan and Venus. Icarus 101, 1–17. doi:10.1006/            Newman, C.E., Richardson, M.I., Lee, C., Toigo, A.D., Ewald, S.P., 2008. The TitanWRF
    icar.1993.1001.                                                                               model at the end of the Cassini Prime Mission. Am. Geophys. Union, Fall
Del Genio, A.D., Zhou, W., 1996. Simulations of superrotation of slowly rotating                  Meeting. Abstract #P12A-02.
    planets: Sensitivity to rotation and initial condition. Icarus 120, 332–343.              Ponte, R.M., Rosen, R.D., 1993. Determining torques over the ocean and their role in
    doi:10.1006/icar.1996.0054.                                                                   the planetary momentum budget. J. Geophys. Res. 98 (D4), 7317–7325.
Flasar, F.M., Samuelson, R.E., Conrath, B.J., 1981. Titan’s atmosphere: Temperature               doi:10.1029/92JD02953.
    and dynamics. Nature 292 (5825), 693–698.                                                 Radebaugh, J. et al., 2008. Dunes on Titan observed by Cassini Radar. Icarus 194 (2),
Flasar, F.M. et al., 2005. Titan’s atmospheric temperatures, winds, and composition.              690–703.
    Science 308 (5724), 975–978.                                                              Randel, W.J., Held, I.M., 1991. Phase speed spectra of transient eddy ﬂuxes and
Folkner, W.M. et al., 2006. Winds on Titan from ground-based tracking of the                      critical layer absorption. J. Atmos. Sci. 48, 688–697.
    Huygens probe. J. Geophys. Res. 111, E07S02. doi:10.1029/2005JE002649.                    Rannou, P., Hourdin, F., McKay, C.P., Luz, D., 2004. A coupled dynamics–
Friedson, A.J., West, R.A., Wilson, E.H., Oyafuso, F., Orton, G.S., 2009. A global climate        microphysics model of Titan’s atmosphere. Icarus 170, 443–462.
    model of Titan’s atmosphere and surface. Planet. Space Sci. 57 (14–15), 1931–             Richardson, M.I., Toigo, A.D., Newman, C.E., 2007. PlanetWRF: A general purpose,
    1949.                                                                                         local to global numerical model for planetary atmospheric and climate
Gierasch, P.J., 1975. Meridional circulation and the maintenance of the Venus                     dynamics. J. Geophys. Res. (Planets) 112, E09001. doi:10.1029/2006JE002825.
    atmospheric rotation. J. Atmos. Sci. 32, 1038–1044.                                       Rossow, W.B., Williams, G.P., 1979. Large-scale motion in the Venus stratosphere. J.
Hide, R., 1969. Dynamics of the atmospheres of the major planets with an appendix                 Atmos. Sci. 36 (3), 377–389.
    on the viscous boundary layer at the rigid bounding surface of an electrically-           Schneider, E.K., 1977. Axially symmetric steady-state models of the basic state for
    conducting rotating ﬂuid in the presence of a magnetic ﬁeld. J. Atmos. Sci. 26,               instability and climate studies. Part II. Nonlinear calculations. J. Atmos. Sci. 34,
    841–853.                                                                                      280–296.
Hong, S.-Y., Pan, H.-L., 1996. Nonlocal boundary layer vertical diffusion in a                Schneider, T., Liu, J., 2009. Formation of jets and equatorial superrotation on Jupiter.
    medium-range forecast model. Mon. Weather. Rev. 124, 2322–2339.                               J. Atmos. Sci. 66 (3), 579–601. doi:10.1175/2008JAS2798.1.
Hourdin, F., Talagrand, O., Sadourny, R., Courtin, R., Gautier, D., McKay, C.P., 1995.        Strobel, D.F., 2006. Gravitational tidal waves in Titan’s upper atmosphere. Icarus
    Numerical simulation of the general circulation of the atmosphere of Titan.                   182 (1), 251–258. doi:10.1016/j.icarus.2005.12.015.
    Icarus 117, 358–374.                                                                      Tokano, T., Neubauer, F.M., 2002. Tidal winds on Titan caused by Saturn. Icarus 158,
Hubbard, W.B. et al., 1993. The occultation of 28 Sgr by Titan. Astron. Astrophys.                499–515.
    269, 541–563.                                                                             Tokano, T., Neubauer, F.M., Laube, M., McKay, C.P., 1999. Seasonal variation of
Kostiuk, T., Fast, K.E., Livengood, T.A., Hewagama, T., Goldstein, J.J., Espenak, F., Buhl,       Titan’s atmospheric structure simulated by a general circulation model. Planet.
    D., 2001. Direct measurement of winds on Titan. Geophys. Res. Lett. 28 (12),                  Space Sci. 47, 493–520.
    2361–2364. doi:10.1029/2000GL012617.                                                      Yamamoto, M., Takahashi, M., 2004. Dynamics of Venus’ superrotation: The eddy
Kuo, H.-L., 1949. Dynamic instability of two-dimensional nondivergent ﬂow in a                    momentum transport processes newly found in a GCM. Geophys. Res. Lett. 31,
    barotropic atmosphere. J. Meteorit. 6 (2), 105–122.                                           L09701. doi:10.1029/2004GL019518.
Kuo, H.-L., 1973. Dynamics of quasigeostrophic ﬂows and instability theory. Adv.              Yamamoto, M., Takahashi, M., 2006. Superrotation maintained by meridional
    Appl. Mech. 13, 247–300.                                                                      circulation and waves in a Venus-like AGCM. J. Atmos. Sci. 63, 3296–3314.
```
