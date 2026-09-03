---
citation_key: "tokano2007near"
title: "Near-surface winds at the Huygens site on Titan: interpretation by means of a general circulation model"
source_pdf: "data/papers/tokano2007near.pdf"
source_pdf_sha256: "ce23968c54ff4d7d000df02625d6bfd7fbd31eea481dc5182ab745198590c739"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                                                           ARTICLE IN PRESS



                                              Planetary and Space Science 55 (2007) 1990–2009
                                                                                                                            www.elsevier.com/locate/pss




    Near-surface winds at the Huygens site on Titan: Interpretation by
                  means of a general circulation model
                                                              Tetsuya Tokano
                      Institut für Geophysik und Meteorologie, Universität zu Köln, Albertus-Magnus-Platz, 50923 Köln, Germany
                                                               Accepted 13 April 2007
                                                            Available online 25 April 2007



Abstract

   This study aims at interpreting the zonal and meridional wind in Titan’s troposphere measured by the Huygens probe by means of a
general circulation model. The numerical simulation elucidates the relative importance of the seasonal variation in the Hadley circulation
and Saturn’s gravitational tide in affecting the actual wind proﬁle. The observed reversal of the zonal wind at two altitudes in the lower
troposphere can be reproduced with this model only if the near-surface temperature proﬁle is asymmetric about the equator and
substantial seasonal redistribution of angular momentum by the variable Hadley circulation takes place. The meridional wind near the
surface is mainly caused by the meridional pressure gradient and is thus a manifestation of the Hadley circulation. Southward meridional
wind in the PBL (planetary boundary layer) is consistent with the near-surface temperature at the equator being lower than at mid
southern latitudes. Even small changes in the radiative heating proﬁle in the troposphere can substantially affect the mean zonal and
meridional wind including their direction. Saturn’s gravitational tide is rather weak at the Huygens site due to the proximity to the
equator, and does not clearly manifest itself in the instantaneous vertical proﬁle of wind. Nevertheless, the simulated descent trajectory is
more consistent with the observation if the tide is present. Because of a different force balance in Titan’s atmosphere from terrestrial
conditions, PBL-speciﬁc wind systems like on Earth are unlikely to exist on Titan.
r 2007 Elsevier Ltd. All rights reserved.

Keywords: Titan; Meteorology; Wind; Huygens



1. Introduction                                                               m s1 in the troposphere. However, there are substantial
                                                                              differences in the predicted wind speeds among models
   For many years the research of the atmospheric                             (Hourdin et al., 1995; Tokano et al., 2001; Tokano and
dynamics of Saturn’s moon Titan has focussed on the                           Neubauer, 2002; Rannou et al., 2004; Tokano and
stratospheric superrotation and its formation mechanisms.                     Neubauer, 2005). Shortly before the arrival of Cassini/
On the other hand, our knowledge of the atmospheric                           Huygens at Titan, astronomical observations of tropo-
circulation in Titan’s troposphere was poor or nearly                         spheric clouds at high southern latitudes began providing
absent prior to the Cassini/Huygens mission. The relevance                    the ﬁrst data on the wind speed and direction at least at
of near-surface winds for Titan’s geology is evident from                     those locations where clouds appeared (Bouchez and
recently detected aeolian features such as longitudinal sand                  Brown, 2005; Roe et al., 2005; Schaller et al., 2006). In
dunes (Lorenz et al., 2006) and other putative wind streaks                   most cases weak eastward wind was retrieved, but some-
(Porco et al., 2005). Moreover, the near-surface wind may                     times also northward drift was observed (Roe et al., 2005).
be regarded as an important component in the atmospheric                      However, since convective clouds develop only at altitudes
angular momentum cycle (Tokano and Neubauer, 2005).                           higher than 10 km (Grifﬁth et al., 2005), the cloud drift
General circulation models (GCMs) of Titan’s atmosphere                       speed may not be representative of near-surface winds. The
predicted wind speeds of a few m s1 up to some tens of                       Huygens probe that descended into Titan’s atmosphere on
                                                                              14 January 2005 provided the ﬁrst in situ data of wind
  Tel.: +49 221 4704489; fax: +49 221 4705198.                               speed and direction in Titan’s troposphere, as described in
   E-mail address: tokano@geo.uni-koeln.de.                                   Section 2.

0032-0633/$ - see front matter r 2007 Elsevier Ltd. All rights reserved.
doi:10.1016/j.pss.2007.04.011
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                 ARTICLE IN PRESS
                                    T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                             1991


   The purpose of this study is twofold. First, it aims at         was southward from the upper troposphere down to
understanding the wind system in the lower troposphere of          15 km, northward between 15 km and 800 m and south-
Titan. Particularly it is investigated to what extent Earth-       ward in the lowest 800 m. Thus the wind vector performs
like meteorology can be expected on Titan. The second              nearly one complete albeit discontinuous rotation in the
purpose is to verify the GCM prediction concerning the             lower 7 km.
tropospheric wind data. Such a veriﬁcation was not                    The lower part of the troposphere comprises the
possible prior to the Huygens mission. Section 2 sum-              planetary boundary layer (PBL), which represents a
marises the wind data in Titan’s troposphere obtained from         transitional layer between the planetary surface and
Huygens. In Section 3 the observed wind proﬁle is                  atmosphere in which the atmospheric dynamics is sub-
analytically analysed, taking into account various char-           stantially affected by the surface. The characteristics of
acteristic wind systems known in the terrestrial atmo-             Titan’s PBL were described based on the thermal structure
spheric boundary layer. Section 4 presents a set of                retrieved by the Huygens Atmospheric Structure Instru-
numerical simulations under various assumptions in an              ment (HASI) (Tokano et al., 2006). The PBL at the
effort to reproduce and interpret the observed wind proﬁle.        Huygens site was weakly convective and had an approx-
The GCM is run under different assumptions concerning              imate depth of 300 m, with presumably negligible diurnal
the global atmospheric circulation such as presence or             variation. The mean wind speed in the surface layer within
absence of Saturn’s gravitational tide, i.e. the force that        the PBL comprising the lowest 10 m was estimated to be
arises on Titan due to the elliptical orbit of Titan (Tokano       0:04 m s1 or less based on the thermal proﬁle in the PBL.
and Neubauer, 2002), and seasonal effects (Tokano, 2005).          On the other hand, the wind proﬁle above 10 m could not
In Section 5 the descent trajectory of the Huygens probe is        be estimated from these data. Also the thermal behaviour
simulated with the wind proﬁle predicted by the GCM and            of the Huygens probe after the landing on Titan indicated
compared with observation. The overall results are                 that the near-surface winds were less than 0:2 m s1 , and
discussed in general context in Section 6.                         probably much less (Lorenz, 2006).
                                                                      While some tentative explanations for the observed wind
2. Near-surface wind data acquired by Huygens                      proﬁle near the surface have already been given (Bird et al.,
                                                                   2005; Tomasko et al., 2005; Folkner et al., 2006), we are
   The Doppler Wind Experiment (DWE) onboard the                   left with several questions that can be formulated as
Huygens probe performed a precise in situ measurement of           follows:
the wind speed in Titan’s atmosphere from an altitude of
146 km down to the surface (Bird et al., 2005; Folkner et           1. Why is there a multiple reversal of both the zonal and
al., 2006). The wind proﬁle in the lower troposphere is                meridional wind near the surface?
depicted in Fig. 7 of Folkner et al. (2006). The DWE wind           2. Why is the meridional wind near the surface faster than
is slightly eastward near the surface below 1 km, turns to             a few kilometres above? Is there evidence of Saturn’s
westward between 1 and 5 km and then returns to zero by                gravitational tide or Hadley circulation?
5 km. The wind proﬁle between 5 and 13 km is unknown,               3. What is the likely temporal and spatial variation of the
but is likely to be prograde and to increase to 3 m s1 at             wind proﬁle?
13 km, above which the wind speed smoothly increases                4. Is the Huygens wind proﬁle representative of whole
with altitude. In this data retrieval the meridional wind was          Titan?
assumed to be zero. Due to the ambiguity of the zonal and
meridional drift direction of Huygens the same Doppler
shift could in principle be generated by several combina-          3. Analytic interpretation of the observed wind proﬁle
tions of u and v. For instance eastward wind of 1 m s1 has
a projection in the direction to Earth of 0:505 m s1 and a         Three basic types of wind proﬁle are known to exist in
northward wind of 1 m s1 has a projection in the direction        the terrestrial PBL (e.g. Stull, 1988). The wind proﬁle
to Earth of 0:239 m s1 (Folkner et al., 2006). In other           characteristic of a PBL with neutral thermal stratiﬁcation is
words it is also possible to interpret the eastward wind of        the Ekman spiral although a pure Ekman spiral is rarely
1 m s1 as southward wind of 2:11 m s1 .                         observed. On the other hand, the wind speed in a
   Horizontal wind speed and direction were also retrieved         convective PBL (with an unstable stratiﬁcation) is
from ground tracking of the Huygens probe by the DISR              roughly uniform across the PBL outside the surface layer
(Descent Imager Spectral Radiometer) (Tomasko et al.,              by virtue of intense vertical mixing. The third type is the
2005; Karkoschka et al., 2007). The probe initially drifted        nocturnal, stable PBL characterised by a large vertical
eastward with a slight additional southward drift down to          variation in wind speed within the PBL, with a marked
an altitude of 7 km. It then quickly turned and drifted            peak (low-level jet) near the top of the PBL. Each of these
westnorthwestward with a speed of 1 m s1 down to                 characteristic wind proﬁles occurs under idealised (homo-
800 m at which a second left turn took place. Eventually           geneous, stationary, dry, etc.) conditions, so the wind can
the probe approached the surface with less than 0:3 m s1          further be affected by inhomogeneous, non-stationary
towards southeast or southsoutheast. The meridional wind           conditions.
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                                      ARTICLE IN PRESS
1992                                                  T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


3.1. Ekman spiral                                                                          where K ¼ 7:4  103 m2 s1 is the eddy diffusivity at the
                                                                                           Huygens site adopted from Tokano et al. (2006) and
   We ﬁrst consider whether the observed wind proﬁle may                                   f ¼ 1:63  106 s1 is the Coriolis parameter at the
represent an Ekman spiral under neutrally stratiﬁed                                        Huygens site.
condition. In an Ekman spiral the wind speed decreases                                        The major unknown in these equations is the geostrophic
from the geostrophic wind in the free atmosphere to zero at                                wind ug that cannot be calculated because the meridional
the surface and the wind vector describes a spiral because                                 pressure gradient at the Huygens site is unknown
of the variation in wind direction with altitude. This spiral                              from observations. Particularly it is unknown whether ug
is a result of the variation with height of the zonal and                                  is positive or negative. Therefore, we assume that
meridional Coriolis force within the PBL.                                                  ug ¼ 1 m s1 because this is close to the zonal wind
   Evident to the eyes is the fact that the shape of the                                   observed by Huygens near 300 m.
descent trajectory reconstructed by Karkoschka et al.                                         Fig. 1 shows the analytic vertical proﬁle of an Ekman
(2007) is not characteristic of an Ekman spiral in that the                                spiral calculated with Eqs. (1) and (2) for positive
total turn angle of 315 is by far too large to be consistent                             (eastward, prograde) and negative (westward, retrograde)
with an Ekman spiral and the wind turn occurs in two                                       values of ug . In either case the wind vector spirals anti-
sharp reversals rather than in a smooth loop.                                              clockwise with increasing altitude, as the Huygens site is
   A more precise assessment is possible if one constructs a                               located in the southern hemisphere. If the geostrophic wind
hypothetical Ekman spiral that is consistent with the PBL                                  is positive (þ1 m s1 , eastward), the zonal wind in the
parameters derived by Tokano et al. (2006). An analytic                                    Ekman spiral increases rapidly with altitude within the
solution of the zonal and meridional wind, u and v, in the                                 lowest 200 m, slightly exceeds the geostrophic speed near
Ekman spiral (e.g. Stull, 1988) can be written as                                          200 m and then stays nearly constant at 1 m s1 at higher
u ¼ ug ½1  expðz=DÞ cosðz=DÞ,                                                  (1)      altitudes. At the same time the meridional wind increases
                                                                                           from zero to 0:3 m s1 (northward) at 75 m and then returns
v ¼ ug expðz=DÞ sinðz=DÞ,                                                       (2)      to zero by 300 m, the top of the PBL. Meridional wind in
                                                                                           the free atmosphere is negligible. This wind proﬁle is
where ug is the geostrophic wind speed, z is the altitude and                              consistent with the DWE data only in the lowest 300 m, i.e.
D is the depth of an Ekman spiral. Here the negative sign                                  within the PBL. The reversal of the zonal wind direction
of the right-hand side of Eq. (2) stands for the southern                                  near 1 km is not consistent with the Ekman spiral and since
hemisphere in which the Huygens site is located (in the                                    the actual zonal wind substantially deviates from 1 m s1 ,
northern hemisphere the sign is positive).                                                 this cannot be regarded as geostrophic wind.
  Here,                                                                                       If we instead assume ug ¼ 1 m s1 , both the zonal and
     sﬃﬃﬃﬃﬃﬃ
                                                                                           meridional wind of the Ekman spiral exhibit a reversed
       2K
D¼           ,                                             (3)                             vertical proﬁle. Although this zonal wind agrees better with
        f                                                                                  the observed wind speed and direction above 1 km, there is

                                                            Zonal wind                                               Meridional wind
                                              2                                                          2
                                                                     DWE                                                          +1
                                                                      +1                                                          -1
                                                                       -1

                                             1.5                                                        1.5




                               Height [km]                                                Height [km]
                                              1                                                          1




                                             0.5                                                        0.5




                                              0                                                          0
                                               -1.5    -1   -0.5      0 0.5   1     1.5                       -0.4    -0.2      0    0.2   0.4
                                                                   u [m/s]                                                   v [m/s]

Fig. 1. Hypothetical Ekman spiral in Titan’s PBL at the Huygens site for different geostrophic winds (1 m s1 ). Positive zonal wind is eastward and
positive meridional wind is northward. Also displayed for comparison is the zonal wind measured by the Huygens DWE (Folkner et al., 2006) adopted
from the ESA-Huygens Data Archive. The thin horizontal line at 0.3 km marks the top of the PBL as inferred by Tokano et al. (2006).
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                     ARTICLE IN PRESS
                                        T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                          1993


a large discrepancy below 1 km in that the eastward wind                  In the case of Titan the Coriolis force is unlikely to be
near the surface is impossible in an Ekman spiral with a               unbalanced at any instance. Fig. 2 shows a comparison of
westward geostrophic wind. A similar statement can be                  several meridional forces at the Huygens site as a function
made concerning the meridional wind. Neither a prograde                of altitude. The meridional Coriolis force, fu, scales with
nor retrograde geostrophic wind can generate the observed              the wind speed measured by DWE (Folkner et al., 2006),
simultaneous presence of northerly and southerly wind in               and varies between 4  106 and 2  106 m s2 in the
the PBL.                                                               lower 5 km. The centrifugal force,  tan fu2 =a, where f is
   In this consideration it was assumed that, according to             the latitude and a is Titan’s radius, is northward, but
Bird et al. (2005) and Folkner et al. (2006), the meridional           negligible in this altitude region. The meridional compo-
component of the observed wind proﬁle is negligible.                   nent of Saturn’s tide calculated after Tokano and
  If we now hypothetically drop this assumption and                    Neubauer (2002) for the time and place of Huygens
consider the most extreme opposite assumption, i.e. that               landing is 2:3  106 m s2 , i.e. northward. Thus the tide
the DWE data arose solely from meridional wind and that                has a magnitude comparable with the Coriolis force. The
the zonal wind was negligible, this would require a peak               presence of a substantial force other than the Coriolis force
southward wind of about 2 m s1 near 300 m and an                     means that the most important prerequisite (unbalanced
increase of the northward wind speed above 1 km altitude.              Coriolis force) is not satisﬁed. Therefore, the local wind
Such a wind proﬁle is even more inconsistent with an                   maximum near 300 m should not be regarded as an Earth-
Ekman spiral, as can be readily seen from Fig. 1b for                  like low-level jet characteristic of a nocturnal PBL.
ug  1 m s1 .                                                            We can conclude that the near-surface wind proﬁle
   This means that an Ekman spiral cannot qualitatively                observed by Huygens cannot be simply understood by
reproduce the vertical proﬁle of zonal wind in the lowest
few kilometres observed by the Huygens DWE irrespective
                                                                                       20
of the sign of geostrophic wind. The steepness of the
                                                                                              Coriolis
Ekman spiral at a given site is sensitive to the eddy                                       centrifugal
diffusivity and a larger eddy diffusivity than retrieved by                                         tide
Tokano et al. (2006) would deepen the Ekman spiral, but
still the reversal of the zonal wind direction cannot be
generated in an Earth-like Ekman spiral.
                                                                                       15
3.2. Further simplified wind profiles

  Tokano et al. (2006) identiﬁed that the PBL at the
Huygens site is weakly convective and has a depth of
300 m. In a strongly convective PBL the wind speed in the

                                                                         Height [km]
outer layer of the PBL tends to be uniform by virtue of
                                                                                       10
vertical mixing (Stull, 1988). However, the observed wind
proﬁle is certainly not uniform within the PBL (and also
beyond the PBL). Instead the wind speed varies between 0
and 1 m s1 within the lowest kilometre (Folkner et al.,
2006). This inconsistency suggests that the vertical mixing
of momentum in the PBL is by far too weak to maintain a
wind speed which is constant with height, as is evident from                            5
a tiny eddy diffusivity of 7:4  103 m2 s1 (Tokano et al.,
2006).
  At a glance the local maximum of wind speed near 300 m
(Folkner et al., 2006) is also suggestive of a low-level jet
characteristic of a stably stratiﬁed, nocturnal PBL. How-
ever, there are several reasons to discard the likelihood of a
                                                                                       0
low-level jet mechanism. Low-level jets in a nocturnal PBL                             -4e-06      -2e-06   0     2e-06     4e-06   6e-06   8e-06
are usually caused by inertial oscillations. This develops if
                                                                                                            Acceleration [m s-2]
the PBL becomes decoupled from the surface because of
weak turbulence and if there is no horizontal pressure                 Fig. 2. Comparison of three major forces in meridional direction at the
gradient (Stull, 1988). Under this condition the Coriolis              Huygens site. The Coriolis and centrifugal force are calculated with the
                                                                       zonal wind after Folkner et al. (2006) and Saturn’s instantaneous
force is no longer balanced by the pressure gradient force
                                                                       gravitational tide is calculated after Tokano and Neubauer (2002). No
and turbulence, so the wind begins to accelerate and to                wind data could be retrieved for altitudes between 5 and 13 km by DWE
rotate with a period of 2p=f , where f is the Coriolis                 (Folkner et al., 2006), so the Coriolis and centrifugal force are linearly
parameter.                                                             interpolated. Positive acceleration is northward.
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                 ARTICLE IN PRESS
1994                                T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


PBL-speciﬁc wind systems known in terrestrial boundary             Table 1
layer meteorology under any thermal stratiﬁcation. This            Overview of the GCM versions run in this study
implies that the inﬂuence of horizontal inhomogeneity in           Simulation no.         Tide         Seasonality   Solar heating
the surface or atmospheric properties as well as large-scale
atmospheric dynamics have to be taken into account.                1                      Yes          Yes           Nominal
Therefore, we investigate in the next section the actual           2                      No           Yes           Nominal
                                                                   3                      Yes          No            Nominal
wind proﬁle at the Huygens site in the context of global-
                                                                   4                      Yes          Yes           Reduced
scale atmospheric dynamics.

4. Interpretation by means of a GCM
                                                                   4.1.1. Zonal wind
4.1. Baseline simulation (Simulation 1)                               In Tokano and Neubauer (2005) it was shown that the
                                                                   zonal wind in the lower troposphere undergoes substantial
   As a next step we make use of a three-dimensional Titan         seasonal variation as a result of latitudinal and vertical
GCM to understand the wind proﬁle measured by                      redistribution of atmospheric angular momentum by virtue
Huygens. The main purpose of such simulations is to put            of the Hadley circulation and surface friction.
the instantaneous single wind proﬁle into the context of              Fig. 3 shows the meridional–vertical cross-section of the
global-scale atmospheric circulation considering various           zonally and diurnally averaged zonal wind u at different
effects that may play a role such as Saturn’s gravitational        seasons. The seasonal variation in u can only be mean-
tide, Hadley circulation and seasonal variation. For this          ingfully understood in combination with the global
study the GCM of Tokano and Neubauer (2005) is applied             temperature ﬁeld depicted in Fig. 4. The main driver of
that is described in that paper and preceding papers of the        the changing wind ﬁeld is the insolation that heats up the
group (Tokano et al., 1999; Tokano and Neubauer, 2002;             surface. The solar radiation gives rise to a pole-to-pole
Tokano, 2005). The radiation code is that of McKay et al.          surface temperature gradient (Fig. 1 of Tokano, 2005).
(1989). The methane mixing ratio for the calculation of the        Subsequently, the PBL in the summer hemisphere is heated
radiative ﬂuxes is prescribed as a sole function of altitude       by virtue of sensible heat ﬂux from the warm surface,
after Lellouch et al. (1989). The methane cycle (condensa-         causing a warm summer pole and a cold winter pole, with a
tion and transport) is not predicted unlike in Tokano et al.       decreasing latitudinal temperature contrast with increasing
(2001) or Rannou et al. (2006). Similarly the vertical proﬁle      height (Fig. 4). The simulation also illustrates that the
of the haze and stratospheric gases is held ﬁxed after             latitudinal temperature gradient is predicted to reverse
McKay et al. (1989). The ground surface temperature is             quickly when the Sun crosses the equator at equinox.
predicted assuming thermal properties of porous icy                   The zonal wind ﬁeld is tightly correlated with the solar
regolith (surface type 1) as described in Tokano (2005).           forcing, as the Hadley circulation (thermally direct
The bulk soil density is r ¼ 800 kg m3 , the soil thermal         circulation) transports angular momentum vertically and
conductivity is k ¼ 0:1 W m1 K1 , the soil-speciﬁc heat          meridionally. Within the lower troposphere the global
capacity is c ¼ 1400 J K1 kg1 , resulting in a surface           proﬁle of the zonal wind resembles that in the terrestrial
thermal inertia of I ¼ 334:7 J m2 s1=2 K1 . The surface         stratosphere in that the summer hemisphere exhibits
albedo is A ¼ 0:38, the surface emissivity is  ¼ 0:86 and         easterlies (retrograde wind), while in a large part of the
the surface drag coefﬁcient is C D ¼ 0:002. Saturn’s               winter hemisphere the wind is eastward (prograde), with
gravitational tide is calculated after Tokano and Neubauer         the core of the prograde and retrograde jets being located
(2002).                                                            at 5 km altitude. The maximum wind speed of the jets is
   The assumed atmospheric and surface parameters are              4 m s1 . The zonal wind ﬁeld reverses after the equinox as
not updated with information gathered from the Huygens             with the temperature ﬁeld. A more or less hemispherically
mission since many relevant data have not yet been                 symmetric wind ﬁeld with mainly prograde wind is found
published and it makes little sense to update only those           only during a short period after the equinoxes.
parameters already published, such as surface albedo and              Fig. 5a shows the vertical proﬁle of the zonal wind at the
column optical depth of the haze (Tomasko et al., 2005) or         Huygens landing site (10 S) at different seasons. The
vertical proﬁle of methane mixing ratio (Niemann et al.,           annual amplitude of the zonal wind is smaller than at
2005). Also no attempt is made to include global                   higher latitudes. Seasonal variation is evident only in the
topography or spatial variation in surface albedo or               lower troposphere below 17 km. The reason for this is
thermal inertia since global maps of surface parameters            that under the predicted conditions the temperature
that can readily be incorporated into climate models do not        variation occurs only up to that altitude (Fig. 4). At higher
yet exist for Titan and this is a separate topic beyond the        altitudes in the troposphere the thermal wind is virtually
scope of this work.                                                independent of season.
   The baseline simulation (Simulation 1, see Table 1 for an          The vertical proﬁle of zonal wind behaves like a string
overview of GCM simulations) includes Saturn’s tide and            which is ﬁxed at the surface and upper troposphere. The
includes the seasonal cycle of insolation.                         bulge of the string oscillates with a period of half a Titan
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                                   ARTICLE IN PRESS
                                                     T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                     1995


                                                          Ls=30°                                                    Ls=225°
                                       20                                                         20

                                       15                                                         15


                         Height [km]                                                Height [km]
                                       10                                                         10

                                        5                                                          5

                                        0                                                          0
                                         −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                       Latitude [°]                                               Latitude [°]

                                                          Ls=90°                                                    Ls=270°
                                       20                                                         20

                                       15                                                         15


                         Height [km]                                                Height [km]
                                       10                                                         10

                                        5                                                          5

                                        0                                                          0
                                         −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                       Latitude [°]                                               Latitude [°]

                                                         Ls=135°                                                    Ls=300°
                                       20                                                         20

                                       15                                                         15


                         Height [km]                                                Height [km]
                                       10                                                         10

                                        5                                                          5

                                        0                                                          0
                                         −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                       Latitude [°]                                               Latitude [°]

                                                         Ls=180°                                                     Ls=0°
                                       20                                                         20

                                       15                                                         15


                         Height [km]                                                Height [km]
                                       10                                                         10

                                        5                                                          5

                                        0                                                          0
                                         −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                       Latitude [°]                                               Latitude [°]

Fig. 3. Meridional–vertical cross-section of zonally and diurnally averaged zonal wind speed u (in m s1 ) in the lower troposphere predicted by the GCM
(Simulation 1) at different seasons. Positive zonal wind is eastward (prograde) and is drawn with solid isotachs. LS is the solar longitude describing the
season (beginning with LS ¼ 0 at northern vernal equinox, see also Fig. 1 of Tokano et al., 1999). LS ¼ 300 is the season of Huygens descent.


year. The retrograde wind intensiﬁes after each equinox                                region, in which the passage of the interface of two Hadley
and becomes fastest after the solstice. When the season                                cells (intertropical convergence zone) occurs twice per
approaches the next equinox, the retrograde wind weakens                               Titan year near the equinoxes.
and the zonal wind near the surface turns to slightly                                    However, there is some difference between local summer
prograde. This behaviour is characteristic of the equatorial                           and winter at the Huygens site. As can be seen from Fig. 3
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                                 ARTICLE IN PRESS
1996                                               T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


                                                        Ls=30°                                                    Ls=225°
                                     20                                                         20

                                     15                                                         15


                       Height [km]                                                Height [km]
                                     10                                                         10

                                      5                                                          5

                                      0                                                          0
                                       −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                     Latitude [°]                                               Latitude [°]

                                                       Ls=90°                                                     Ls=270°
                                     20                                                         20

                                     15                                                         15


                       Height [km]                                                Height [km]
                                     10                                                         10

                                      5                                                          5

                                      0                                                          0
                                       −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                     Latitude [°]                                               Latitude [°]

                                                       Ls=135°                                                    Ls=300°
                                     20                                                         20

                                     15                                                         15


                       Height [km]                                                Height [km]
                                     10                                                         10

                                      5                                                          5

                                      0                                                          0
                                       −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                     Latitude [°]                                               Latitude [°]

                                                       Ls=180°                                                     Ls=0°
                                     20                                                         20

                                     15                                                         15


                       Height [km]                                                Height [km]
                                     10                                                         10

                                      5                                                          5

                                      0                                                          0
                                       −90   −60   −30    0      30   60    90                    −90   −60   −30    0      30   60   90
                                                     Latitude [°]                                               Latitude [°]

Fig. 4. Meridional–vertical cross-section of zonally and diurnally averaged temperature T (in K) in the lower troposphere predicted by the GCM
(Simulation 1) at different seasons.

the Huygens site is located close to the boundary of                                   Fig. 6a shows the instantaneous vertical proﬁle of zonal
easterlies and westerlies in southern winter (LS ¼ 90 ),                            wind predicted for the time and place of Huygens’ landing
while it is close to the core of the easterlies in southern                          along with the predictions of other scenarios discussed in
summer (LS ¼ 300 ), the season of the Huygens descent.                              the subsequent subsections. The zonal wind can readily be
Therefore, the retrograde wind becomes fastest near the                              compared with the wind proﬁle measured by the Huygens
Huygens season.                                                                      DWE (Folkner et al., 2006). The predicted zonal wind
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                                               ARTICLE IN PRESS
                                                           T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                             1997


                                                                 Zonal wind                                                      Meridional wind
                                       20                                                                        20

                                       18                                                                        18

                                       16                                                                        16

                                       14                                                                        14

                                       12                                                                        12


                         Height [km]                                                               Height [km]
                                       10                                      DWE                               10
                                                                                 0°
                                                                                30°
                                        8                                       90°                               8
                                                                               135°
                                                                               180°
                                        6                                      225°                               6
                                                                               270°
                                                                               300°
                                        4                                                                         4

                                        2                                                                         2

                                        0                                                                         0
                                            -3   -2    -1     0        1       2       3   4   5                   -0.4 -0.3 -0.2 -0.1   0   0.1 0.2 0.3 0.4
                                                                  u [m s-1]                                                          v [m s-1]

Fig. 5. Vertical proﬁle of the diurnally averaged zonal wind (a) and meridional wind (b) at the Huygens site predicted by the GCM (baseline simulation) at
different seasons (LS ). The DWE wind proﬁle is shown for comparison as well.



                                                                 Zonal wind                                                     Meridional wind
                                       20                                                                        20

                                       18                                                                        18

                                       16                                                                        16

                                       14                                                                        14

                                       12                                                                        12


                         Height [km]                                                               Height [km]
                                       10                                      DWE                               10
                                                                                 1
                                                                                 2
                                        8                                        3                                8
                                                                                 4
                                        6                                                                         6

                                        4                                                                         4

                                        2                                                                         2

                                        0                                                                         0
                                            -4   -3   -2    -1     0       1       2   3   4   5                   -0.4 -0.3 -0.2 -0.1   0   0.1 0.2 0.3 0.4
                                                                  u [m s-1]                                                         v [m s-1]

Fig. 6. Same as Fig. 5, but showing the instantaneous proﬁle at the time and place of Huygens’ landing. The numbers denote the simulation number
explained in Table 1.


roughly agrees with the observed wind at altitudes below                                                   in the retrieved DWE data between 5 and 13 km (Folkner
3 km, except that the prograde wind below 1 km may be                                                      et al., 2006) it is reasonable to assume that this reversal
slightly too weak. Particularly, the reversal of the wind                                                  takes place somewhere between 5 and 13 km, probably
direction at 800 m and an increase of the retrograde wind in                                               closer to 5 km. This quantitative discrepancy could indicate
the lowest 3 km are nicely reproduced. On the other hand,                                                  that in this model version the retrograde jet centred at
the predicted wind has a peak retrograde wind near 7 km                                                    southern mid latitudes is too strong. Possible cause of this
rather than at 3 km and the reversal from retrograde to                                                    is an excessive surface–atmosphere exchange of angular
prograde wind occurs near 15 km. Although there is a gap                                                   momentum, excessive seasonal variation in near-surface
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                                  ARTICLE IN PRESS
1998                                 T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


temperature or seasonal lag in the solar forcing in the             ture was 94.1 K, i.e. slightly warmer than the surface air
troposphere.                                                        (93.5 K) measured by HASI (Fulchignoni et al., 2005) and
   The predicted wind below 1 km is prograde although it            also warmer than predicted by the GCM for 300 m altitude
seems to be weaker than measured by Huygens. However,               (93.3 K). While the ground temperature undergoes a
as already mentioned in Section 2 there is some ambiguity           diurnal variation with an amplitude of 1 K, the air
between eastward and southward wind in the DWE data                 temperature at 300 m is nearly independent of the solar
(Folkner et al., 2006). A part of the missing eastward wind         local time, i.e. there is virtually no diurnal variation.
in the predicted proﬁle could be partly compensated by the          Regions with a higher thermal inertia than assumed in the
near-surface southward wind. This aspect will be discussed          model will have even less diurnal variation, so the predicted
in Section 5 in the context of the probe’s descent trajectory.      longitudinal uniformity in the air temperature can be
   The most obvious discrepancy between the GCM                     regarded as realistic. This is consistent with the conclusion
prediction and observation is the systematic underestima-           of the analysis of the PBL by Tokano et al. (2006). Hence,
tion of the prograde wind by about 3 m s1 at altitudes             unlike in the terrestrial PBL, it is unlikely that (thermally
above 10 km (Figs. 5 and 6). In no season and no                    forced) diurnal variations in the PBL affect the wind
simulation run does the predicted wind speed approach               proﬁle.
the observed 3–4 m s1 near 20 km. This bias may be                    Consequently, the global pattern of the surface pressure
correlated with the inability of this GCM in generating             (Fig. 8d) does not correlate with the solar local time or
strong stratospheric superrotation, as discussed in Tokano          pattern of the ground temperature. Instead the global
et al. (1999) or Tokano and Neubauer (2002), and is less            surface pressure map exhibits a wave 2 pattern unique to
likely to reﬂect inaccuracies in the predicted seasonal cycle.      the gravitational tide as described by Tokano and
While this topic is beyond the scope of this study, possible        Neubauer (2002). At the time of Huygens landing two
discrepancies between the calculated and real radiative             pressure maxima are found at the equator near 20 W and
heating proﬁle could be responsible for this discrepancy, as        160 E and two troughs are located in between, near 110 W
will be discussed in Section 4.4.                                   and 70 E. These high pressures are regions of convergence
                                                                    of the tidal acceleration ﬂow (Fig. 9a). In Tokano and
4.1.2. Meridional circulation and tide                              Neubauer (2002) the surface pressure map was symmetric
   One remarkable result of the Huygens mission was the             about the equator because in that model version season-
detection of meridional winds in the troposphere down to            ality was virtually absent owing to the temporally ﬁxed
the surface (Tomasko et al., 2005; Karkoschka et al., 2007).        ground surface temperature. In this simulation the long-
In this part of this study we elucidate the mechanism               itudinally averaged surface pressure is higher at the north
behind the observed meridional wind. Particularly, we               pole than at the south pole as a result of the pole-to-pole
investigate whether Saturn’s gravitational tide, the Hadley         temperature gradient and the associated Hadley circula-
circulation, transient eddies or something else can account         tion. It turns out that the Huygens site is located just
for the observed meridional wind.                                   southeast of the centre of a tidally induced high-pressure
   A persistent component of meridional circulation is the          region. Therefore, the pressure decreases from northwest to
thermally direct circulation (Hadley circulation), which is         southeast at the Huygens site.
present regardless of the gravitational tide. Fig. 7a depicts          As a consequence of the tide, the horizontal pressure
the mass streamfunction of the mean Hadley circulation in           gradient undergoes a similar diurnal oscillation with a
the Huygens season (LS ¼ 300 ) predicted by the GCM                magnitude comparable to the tide (Fig. 10b). The surface
baseline version. As is typical of solstice-type circulation        pressure varies by 1 hPa during a Titan day, and
one single cell extends from the south pole to the north            coincidentally attained a maximum at the time of Huygens
pole, except for a small opposite cell near the south pole.         descent (Fig. 10c). A comparison of Fig. 10a and b
The streamlines indicate that southward ﬂow exists below            illustrates that the tide and pressure gradient are anti-
2 km altitude at almost all latitudes, representing the lower       correlated, i.e. there is an approximate balance between the
branch of the Hadley cell. Upwelling occurs in the entire           tide and pressure gradient. This pressure gradient variation
southern hemisphere, while downwelling occurs in the                is clearly caused by the tide, and would disappear in its
northern hemisphere.                                                absence. The zonal pressure gradient has a larger
   Fig. 8 shows the instantaneous global map of tempera-            amplitude than the meridional one, as with the tidal
tures at different levels in the atmosphere and at the              acceleration, but the diurnal average is zero. On the other
surface. Also shown is the surface pressure at the time of          hand, the diurnal average of the meridional pressure
Huygens’ landing that is relevant in assessing the hor-             gradient is southward and is responsible for the cross-
izontal force balance as well as the relative role of tide and      equatorial Hadley circulation. Wiggles superposed on the
Hadley circulation. The subsolar point at the time of               periodical oscillation of the pressure gradient force are
landing is located at 22 S, 208 E (152 W) close to the           caused by the ﬁnite response time of the pressure and wind
boundary between Shangri-La and Xanadu. The ground                  ﬁeld to adjust themselves to the changing tidal ﬁeld.
temperature peaks near 22 S, 220 E, i.e. 12 east of the             However, it is also important to note that the tide at the
subsolar point. At the Huygens site the ground tempera-             Huygens site is weaker than at other locations. Fig. 10a
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                            ARTICLE IN PRESS
                                               T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                         1999


                                                                          Simulation 1
                                       20



                                       15




                         Height [km]
                                       10



                                        5



                                        0
                                         −90     −60           −30              0            30              60              90
                                                                      Latitude [°]

                                                                          Simulation 3
                                       20



                                       15




                         Height [km]
                                       10



                                        5



                                        0
                                         −90     −60           −30              0            30              60              90
                                                                           Latitude [°]
                                                                          Simulation 4
                                       20



                                       15




                         Height [km]
                                       10



                                        5



                                        0
                                         −90     −60           −30              0            30              60              90
                                                                           Latitude [°]

Fig. 7. Mass streamfunction (in 108 kg s1 ) of the Hadley circulation in the Huygens season predicted under three different conditions. The ﬂow is
clockwise/anti-clockwise along solid/dashed streamlines. The result for Simulation 2 is almost identical to that of Simulation 1, and thus is not shown.




illustrates that the meridional tide has an amplitude                              Fig. 9 shows the global map of the instantaneous
roughly three times weaker than the zonal tide. This                            horizontal wind vector at three levels in the troposphere
behaviour is caused by the vicinity of the Huygens site to                      (300, 4.5 and 20 km) along with the vector of Saturn’s tidal
the equator, which is a symmetry axis of the tide where the                     acceleration. In no way does the wind vector follow the
meridional component vanishes for geometrical reasons.                          instantaneous direction of Saturn’s gravitational tide. This
At the time of Huygens’ landing the tide had a slight                           is because the wind direction is a result of the horizontal
northward and a three times larger westward acceleration,                       force balance consisting, among others, of the pressure
so the tidal vector pointed westnorthwestward.                                  gradient force, tidal force and Coriolis force. The inﬂuence
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                                                                       ARTICLE IN PRESS
2000                                                    T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


                                                        With tide                                                       Without tide
                                                   Temperature at 4.5 km                                            Temperature at 4.5 km
                                        90                                                               90
                                        60                                                               60



                         Latitude [°]                                                     Latitude [°]
                                        30                                                               30
                                         0                                                                0
                                        −30                                                              −30
                                        −60                                                              −60
                                        −90                                                              −90
                                              0   60     120 180 240          300   360                        0   60     120 180 240 300      360
                                                         East longitude [°]                                               East longitude [°]

                                                       Temperature at 300 m                                             Temperature at 300 m
                                        90                                                               90
                                        60                                                               60



                         Latitude [°]                                                     Latitude [°]
                                        30                                                               30
                                         0                                                                0
                                        −30                                                              −30
                                        −60                                                              −60
                                        −90                                                              −90
                                              0   60     120 180 240 300            360                        0   60     120 180 240 300      360
                                                         East longitude [°]                                               East longitude [°]
                                                        Ground temperature                                               Ground temperature
                                        90                                                               90
                                        60                                                               60



                         Latitude [°]                                                     Latitude [°]
                                        30                                                               30
                                         0                                                                0
                                        −30                                                              −30
                                        −60                                                              −60
                                        −90                                                              −90
                                              0   60     120 180 240 300            360                        0   60     120 180 240 300      360
                                                         East longitude [°]                                               East longitude [°]

                                                         Surface pressure                                                 Surface pressure
                                        90                                                               90
                                        60                                                               60



                         Latitude [°]                                                     Latitude [°]
                                        30                                                               30
                                         0                                                                0
                                        −30                                                              −30
                                        −60                                                              −60
                                        −90                                                              −90
                                              0   60     120 180 240 300            360                        0   60     120 180 240 300      360
                                                         East longitude [°]                                               East longitude [°]

Fig. 8. Instantaneous global map of temperatures and surface pressure predicted by the GCM for the time of the Huygens descent. The left column shows
the results with Saturn’s tide (Simulation 1), the right column the results without Saturn’s tide (Simulation 2). The temperatures are in K, the surface
pressure in hPa. The Huygens site is marked as ‘H’.


of the tide is best seen in the wave 2 pattern superposed on                                    than at higher altitudes. At the Huygens site there is a
the main zonal ﬂow.                                                                             generally southward cross-equatorial ﬂow that is present at
  However, it can also be seen that the tidally induced                                         all longitudes. This cannot be ascribed to the tide, but to
longitudinal variation in the meridional wind is rather                                         the strong southward pressure decrease (Fig. 8d). Since the
weak near the equator including the Huygens site. The                                           Coriolis force disappears at the equator the southward
wind ﬁeld at the top of the PBL (300 m) is more complex                                         pressure gradient force cannot be balanced by the Coriolis
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                                    ARTICLE IN PRESS
                                                      T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                     2001


                                                          With tide                                                   Without tide
                                                      Tidal acceleration                                           Tidal acceleration
                                       90                                                           90
                                       60                                                           60



                        Latitude [°]                                                 Latitude [°]
                                       30                                                           30
                                        0                                                            0
                                       −30                                                          −30
                                       −60                                                          −60
                                       −90                                                          −90
                                             0   60   120 180 240 300          360                        0   60   120 180 240 300      360
                                                      East longitude [°]                                           East longitude [°]

                                                        Wind at 20 km                                               Wind at 20 km
                                       90                                                           90
                                       60                                                           60



                        Latitude [°]                                                 Latitude [°]
                                       30                                                           30
                                        0                                                            0
                                       −30                                                          −30
                                       −60                                                          −60
                                       −90                                                          −90
                                             0   60   120 180 240 300          360                        0   60   120 180 240 300      360
                                                      East longitude [°]                                           East longitude [°]
                                                       Wind at 4.5 km                                               Wind at 4.5 km
                                       90                                                           90
                                       60                                                           60



                        Latitude [°]                                                 Latitude [°]
                                       30                                                           30
                                        0                                                            0
                                       −30                                                          −30
                                       −60                                                          −60
                                       −90                                                          −90
                                             0   60   120 180 240 300          360                        0   60   120 180 240 300      360
                                                      East longitude [°]                                           East longitude [°]

                                                        Wind at 300 m                                               Wind at 300 m
                                       90                                                           90
                                       60                                                           60



                        Latitude [°]                                                 Latitude [°]
                                       30                                                           30
                                        0                                                            0
                                       −30                                                          −30
                                       −60                                                          −60
                                       −90                                                          −90
                                             0   60   120 180 240 300          360                        0   60   120 180 240 300      360
                                                      East longitude [°]                                           East longitude [°]

Fig. 9. Instantaneous global map of tidal acceleration and wind vector predicted by the GCM for the time of the Huygens descent. The tidal acceleration
is deﬁned in Tokano and Neubauer (2002). The wind vector length scales with the wind speed and one grid distance corresponds to 1 m s1 at 20 and
4.5 km altitude and 0:5 m s1 at 300 m.


force, as would be the case in a geostrophic balance, so the                               10 km (Fig. 5b). It is faster near the surface, where it
wind simply follows the pressure gradient. Thus it seems                                   exceeds 0:3 m s1 . v also exhibits a clear seasonal variation,
that the horizontal wind at the Huygens site is not                                        although the annual amplitude is about 1 order of
substantially affected by Saturn’s tide.                                                   magnitude smaller than that of u. At the Huygens
   The diurnal-mean meridional wind v at the Huygens site                                  site the meridional wind direction reverses near 800 m
is less than 0:01 m s1 in the upper troposphere above                                     altitude. Above 800 m northward wind is found in local
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
                                                                                      ARTICLE IN PRESS
2002                                                                     T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


                                                                                              Saturn’s tidal acceleration
                                                                10




                                   Acceleration [10-6 m s-2]
                                                                 8                                                                 zonal
                                                                 6                                                            meridional
                                                                 4
                                                                 2
                                                                 0
                                                                -2
                                                                -4
                                                                -6
                                                                -8
                                                               -10
                                                                     0                  4                   8                  12          H    16
                                                                                             Tidal phase past periapsis [days]
                                                                                            Pressure gradient acceleration
                                                                12




                                   Acceleration [10-6 m s-2]
                                                                10                                                                 zonal
                                                                 8                                                            meridional
                                                                 6
                                                                 4
                                                                 2
                                                                 0
                                                                -2
                                                                -4
                                                                -6
                                                                -8
                                                               -10
                                                               -12
                                                                     0                  4                   8                  12          H    16
                                                                                             Tidal phase past periapsis [days]
                                                                                                    Surface pressure
                                                          1468




                               ps [hPa]                   1467




                                                          1466
                                                                     0                  4                   8                  12      H       16
                                                                                             Tidal phase past periapsis [days]

Fig. 10. (a) Diurnal variation in the zonal and meridional component of Saturn’s gravitational tidal acceleration af the Huygens site (10 S, 168 E)
calculated after Tokano and Neubauer (2002). At the time of Huygens descent the tide was northwestward, with a zonal component of 4:1  106 m s2
and a meridional component of 2:3  106 m s2 . At this near-equatorial site the zonal tide has a larger amplitude than the meridional one. The
instantaneous global map of the tide is similar to that of day 14 in Fig. 2 of Tokano and Neubauer (2002). (b) Diurnal variation in the zonal and
meridional pressure gradient acceleration at the Huygens site predicted by the GCM. (c) Diurnal variation in the surface pressure at the Huygens site
predicted by the GCM. The time axis is expressed in terms of Titan’s tidal phase in days beginning from periapsis (day 0), via apoapsis (day 8), the
Huygens landing time (day 13.88) to the following periapsis (day 16) as deﬁned in Tokano and Neubauer (2002).



summer and autumn (from LS ¼ 270 to shortly before                                                           The instantaneous meridional wind at the Huygens
LS ¼ 90 ), and southward wind otherwise. Below 1 km this                                                  site (Fig. 6b) changes direction at several altitudes. South-
is exactly reversed. At the Huygens season v is southward                                                  ward wind is found from the surface up to 3 km and
below 800 m and the maximum speed occurs near the                                                          between 8 and 16 km, while northward wind occurs in
surface. Above 800 m v turns to northward, with a nearly                                                   between. Remarkably, the southward wind is strongest in
uniform speed of 0:05 m s1 up to 7 km. Above 7 km v                                                       the PBL, with 0:25 m s1 . This meridional wind reversal is
becomes negligible. This vertical proﬁle is very consistent                                                mainly a result of the near-surface Hadley circulation
with the wind proﬁle retrieved by Karkoschka et al. (2007).                                                whose core is located near 2 km (Fig. 7). The near-surface
Both the altitude of the reversal as well as the fact that the                                             southward ﬂow represents the lower branch of the Hadley
near-surface v is faster than further above are well                                                       cell beneath the core. The northward ﬂow between 3 and
reproduced by the GCM. This seasonal pattern is likely                                                     8 km is located in the upper branch of the Hadley cell,
to be related to the Hadley circulation and its seasonal                                                   with a ﬂow from the summer (south) to winter (north)
reversal (Fig. 7) and the altitude of the reversal of the                                                  hemisphere.
meridional wind direction corresponds to the core of the                                                      However, the southward ﬂow above 8 km cannot be
mass streamfunction.                                                                                       explained by the Hadley circulation alone, as the mass
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
                                                  ARTICLE IN PRESS
                                     T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                              2003


streamfunction indicate mean northward ﬂow all the way              remains. The wind ﬁeld in the free atmosphere is mostly
above 1 km (Fig. 7). Thus this is likely to reﬂect a transient      zonal and the meridional wind almost disappears. Also the
feature not visible in the mean meridional circulation              vertical proﬁle (Fig. 6b) reveals that the tide does not seem
pattern. At this altitude the horizontal temperature                to systematically and substantially affect the meridional
gradient is already quite small (Fig. 4), so the pressure           wind.
gradient associated with inhomogeneous temperature                     In the PBL, however, there are still substantial mer-
distribution is smaller than that caused by Saturn’s                idional winds although the global pattern does not agree
gravitational tide. As the meridional component of the              with that predicted in the presence of tide. Generally
tide is northward (Fig. 10a) the pressure gradient                  southward ﬂow in the equatorial regions exists even in the
force is southward (Fig. 10b) to balance the tide, which            absence of tide (Fig. 9h), reinforcing that this is caused by
is responsible for the southward wind in the upper                  the southward pressure decrease near the surface, i.e.
troposphere.                                                        by the lower branch of the cross-equatorial Hadley
                                                                    circulation. The slightly weaker southward wind in the
4.2. Influence of Saturn’s tide or the lack of it (Simulation       PBL (Fig. 6b) in comparison with Simulation 1 may be
2)                                                                  explained by the lack of tide.

   In this subsection the inﬂuence of Saturn’s gravitational        4.3. Influence of seasonality (Simulation 3)
tide on the actual zonal wind proﬁle at the Huygens site is
pursued, as the tide is predicted to affect both the zonal             The baseline simulation presented in Section 4.1
and meridional wind (Tokano and Neubauer, 2002). To                 was run under ordinary seasonal variation in the solar
investigate the impact of the tide or the lack of it on the         forcing. However, another series of Titan GCM
wind ﬁeld the GCM run (with seasonal forcing) is now                (Rannou et al., 2004, 2006) predicts more or less
repeated without tide, i.e. the tidal acceleration in the           hemispherically symmetric surface temperature throughout
momentum equation is artiﬁcially switched off (Simulation           the year. This could occur if there is a permanent
2). Fig. 6 shows a comparison of the instantaneous vertical         accumulation of haze particles in the polar region of either
proﬁle of zonal wind at the place and time of Huygens’              hemisphere. In this case the seasonality ceases in the lower
landing predicted by the GCM with (Simulation 1) and                troposphere in that the latitudinal distribution of insola-
without tide (Simulation 2). The zonal wind does not differ         tion becomes more or less symmetric about the equator
much from the simulation in which the tide is included. At          throughout the year.
least the inﬂuence of the tide on u is much smaller than the           To account for this hypothetical effect we repeat the
seasonal difference.                                                same Titan GCM simulation without seasonal variation in
   However, the zonal wind near the surface remains                 the insolation (Simulation 3). The solar declination is set to
retrograde, while in Simulation 1 it reverses to slightly           0 during the entire simulation, so the equator always
prograde near 500 m. In the absence of tide the instanta-           receives the largest amount of sunlight. The resulting
neous zonal wind is systematically shifted by   0:3 m s1         global ﬁeld of zonal wind and temperature is, not
in comparison with the simulation with tide. This                   surprisingly, symmetric about the equator (Fig. 11c and
small difference can only be ascribed to the absence of             d). In contrast to the simulation with seasonal variation the
tide. If the tide is included, the westward tidal acceleration      temperature monotonically decreases with latitude in either
(Fig. 10a) gives rise to an eastward pressure gradient force        hemisphere up to about 5 km and the zonal wind is almost
at the time of Huygens’ landing (Fig. 10b). Since the               everywhere prograde, with stationary tropospheric jets
Coriolis force is negligible in the PBL the presence of an          near 50 latitude and 5 km altitude. The only exception to
eastward pressure gradient force immediately causes east-           this is the PBL close to the surface. In other words regions
ward wind. If, on the other hand, the tide is absent, the           with easterlies (retrograde wind) virtually disappear in this
longitudinal pressure gradient disappears and the zonal             simulation. Most remarkably, the wind in the equatorial
wind in the PBL is primarily affected by the vertical               region is now prograde although the wind speed is very
transport of angular momentum. Thus the zonal wind in               low.
the PBL is something between zero and the retrograde                   The simultaneous formation and maintenance of wester-
wind outside the PBL.                                               lies in either hemisphere can be explained by the
   The right column of Figs. 8 and 9 shows the horizontal           surface–atmosphere transfer of angular momentum by
temperature and wind ﬁeld at selected levels predicted in           the easterlies near the surface and subsequent transport of
the absence of tide. The global temperature ﬁeld barely             it by the equator-to-pole equinox-type Hadley circulation.
differs from that in the presence of tide since the tide does       The decrease of the temperature with latitude implies
not directly affect the thermally forced Hadley circulation.        positive thermal wind, i.e. the westerly zonal wind increases
The major difference is found in the surface pressure ﬁeld          with altitude up to the level at which the latitudinal
that lacks a clear longitudinal variation. The marked wave          temperature gradient ceases (5 km). Above 5 km the zonal
2 pattern caused by the tide disappears, although some              wind diminishes since there is a slight increase of the
irregular ﬂuctuations caused by the turbulence in the PBL           temperature with latitude. At the Huygens site the
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
                                                                  ARTICLE IN PRESS
2004                                                 T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


                                                      u, Simulation 1                                           T, Simulation 1
                                       20                                                         20

                                       15                                                         15


                         Height [km]                                                Height [km]
                                       10                                                         10

                                        5                                                          5

                                        0                                                          0
                                         −90   −60   −30    0      30    60    90                   −90   −60   −30    0      30   60   90
                                                       Latitude [°]                                               Latitude [°]
                                                      u, Simulation 3                                           T, Simulation 3
                                       20                                                         20

                                       15                                                         15


                         Height [km]                                                Height [km]
                                       10                                                         10

                                        5                                                          5

                                        0                                                          0
                                         −90   −60   −30    0      30    60    90                   −90   −60   −30    0      30   60   90
                                                       Latitude [°]                                               Latitude [°]
                                                      u, Simulation 4                                           T, Simulation 4
                                       20                                                         20

                                       15                                                         15


                         Height [km]                                                Height [km]
                                       10                                                         10

                                        5                                                          5

                                        0                                                          0
                                         −90   −60   −30    0      30    60    90                   −90   −60   −30    0      30   60   90
                                                       Latitude [°]                                               Latitude [°]

Fig. 11. Meridional–vertical cross-section of zonally and diurnally averaged zonal wind in m s1 (left column) and temperature in K (right column) in the
Huygens season predicted under different conditions. The result of Simulation 2 is almost identical to that of Simulation 1, and thus is not shown.



predicted zonal wind is prograde in the entire troposphere,                              temperature at the solstice was asymmetric about the
with wind speeds of 1 m s1 and little variation with                                   equator.
height (Fig. 6). This proﬁle is qualitatively inconsistent                                  On the basis of this simulation and comparison with
with the observation in that the reversal of the wind                                    other GCMs we can conclude that on Titan easterlies
direction and the retrograde wind are absent.                                            preferentially develop in regions with a temperature
   Also Grieger et al. (2004) showed that in the absence of a                            increase with latitude down to the surface and the wind
hemispheric asymmetry in the tropospheric temperature                                    balance is closer to geostrophic than to cyclostrophic, as is
the near-surface easterlies substantially weaken in compar-                              the case in the lower troposphere. Seasonal variation in the
ison with a model version with a warmer summer pole. The                                 surface and near-surface temperature contributes a major
GCM of Luz et al. (2003) and Rannou et al. (2004) predict                                part in the generation of substantial easterlies in the lowest
that in the equatorial region the zonal wind is prograde                                 few kilometres. At the same time the presence of these
almost everywhere except in the lowest 100 m or so. In their                             easterlies is consistent with a warmer southern (summer)
model the near-surface temperature is symmetric about the                                hemisphere compared with the equator in the Huygens
equator even in the Huygens season. We note that, in                                     season.
contrast to their present GCM version, their early 3D                                       Fig. 7b shows the mass streamfunction of the Hadley
version of the GCM (Hourdin et al., 1995) did generate                                   circulation under the condition that no seasonality exists,
retrograde wind at the Huygens site from the surface up to                               i.e. under a permanent equinox condition. The tropo-
about 1000 hPa (6 km). In that version the near-surface                                  spheric Hadley circulation in this case is split into four
```

<!-- PDF_PAGE: 16 -->

## PDF page 16

```text
                                                  ARTICLE IN PRESS
                                     T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                        2005


cells. A thermally direct cell extends from the equator to                     20
40 latitude in each hemisphere. Another, thermally                                                                         HASI
                                                                                                                               1
indirect cell extends from there to the poles. In contrast                                                                     3
                                                                               18
to the baseline simulation there is no cross-equatorial mean                                                                   4
ﬂow any more and the mass ﬂux is smaller by a factor of 5.
A comparison of Fig. 7a and b clariﬁes that the mean                           16
meridional wind direction at the Huygens site (10 S) is
opposite to that predicted in the baseline simulation, i.e.
the near-surface ﬂow is northward (equatorward), while at                      14
higher altitudes the mean ﬂow is poleward (southward).
This meridional circulation pattern shows some resem-
                                                                               12
blance to that predicted by Rannou et al. (2006) in that
several cells are located side by side. However, oblique cells
                                                                      z [km]
as predicted by Rannou et al. (2006) are not predicted in                      10
the present model and this may be the inﬂuence of
tropospheric methane clouds not simulated here.
   These differences to the baseline simulation can also be                     8
recognised in the instantaneous vertical proﬁle of v at the
Huygens site (Fig. 6b). The meridional wind barely exceeds
                                                                                6
0:1 m s1 although the tide is taken into account in the
simulation. Persistent northward wind between 7 km and
800 m as well as the southward wind below that level                            4
observed by DISR are not reproduced at all. This reﬂects
the lack of a cross-equatorial temperature and pressure
gradient in this scenario.                                                      2


4.4. Sensitivity to the heating rate (Simulation 4)
                                                                                0
                                                                                    70   75         80           85          90          95
   Another parameter that requires attention in the context                                              T [K]
of this study is the strength of the Hadley circulation,
which not only represents the meridional wind, but also             Fig. 12. Instantaneous vertical proﬁle of temperature at the Huygens site
affects the zonal wind proﬁle by virtue of global                   at the time of landing predicted by the GCM under different conditions.
                                                                    The numbers denote the simulation number listed in Table 1. The
redistribution of angular momentum. Zhu and Strobel
                                                                    temperature measured by HASI (Fulchignoni et al., 2005) is shown for
(2005) found in a sensitivity experiment with a two-                comparison. The result of Simulation 2 is almost identical to that of
dimensional model of Titan’s stratosphere that the strength         Simulation 1, and thus is not shown.
of the meridional circulation directly scales with the
radiative forcing. If in their model the radiative heating
rate was increased by a factor of 10, the meridional                the temperature measured by HASI (Fulchignoni et al.,
circulation increased by the same factor, too. The GCM of           2005), but with increasing altitude the predicted tempera-
Rannou et al. (2004) is another example that shows how              ture shows a negative deviation. This indicates that
the radiative ﬂux affects the meridional circulation. In            either the solar heating rate is underestimated or the
comparison with previous simulations with a ﬁxed haze               cooling rate is overestimated in the upper troposphere
distribution the meridional circulation in the stratosphere         compared with the lower troposphere. However, it could
was intensiﬁed. However, since both Rannou et al. (2004)            also indicate that the vertical mixing of heat is excessive,
and Zhu and Strobel (2005) focussed on the stratosphere it          so the lapse rate is closer to adiabiatic than the measured
is not evident from their studies how the change in the             lapse rate.
radiative forcing affects the meridional and zonal circula-            The calculation of the radiative heating rate in this GCM
tion in the troposphere.                                            is based on the radiation model of McKay et al. (1989).
   The net heating rate is the sum of solar heating rate and        The column optical depth in the visible spectrum varies
thermal cooling rate, and is the driving force of the Hadley        between 2 and 2.5 and the troposphere was assumed to be
circulation. In the lower 20 km the solar heating rate at the       clear of haze. However, Huygens detected that there is
Huygens site predicted by the GCM is 2  108 K s1 or              signiﬁcant haze opacity at all altitudes down to the surface
less, and increases slowly with altitude. The amount of the         (Tomasko et al., 2005). The measured column optical
thermal cooling rate is less than 108 K s1 , thus smaller         depth of the haze is 4–5 at 531 nm and 2.5–3.5 at 329 nm.
than the solar heating rate, so the net heating rate is larger,     This indicates that a smaller amount of sunlight should
as expected for the equatorial region. Near the surface the         arrive at Titan’s surface and the solar heating rate in the
temperature at the Huygens site (Fig. 12) closely matches           lower troposphere may be smaller than predicted by the
```

<!-- PDF_PAGE: 17 -->

## PDF page 17

```text
                                                ARTICLE IN PRESS
2006                               T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


GCM, while a larger amount of solar radiation may be              shift the altitude of the zonal wind reversal to some
absorbed at higher altitudes.                                     extent.
   Given this fact, we reduce in Simulation 4 the solar
heating rate intentionally by a factor of 2 with respect to       5. Descent trajectory simulation
the baseline simulation. While this ad hoc assumption is
simplistic, future studies taking into account detailed              An alternative way of verifying the vertical proﬁle of
results of the Huygens DISR concerning the radiative              wind speed and direction with the Huygens data is to
budget in Titan’s atmosphere may constrain the radiative          compare the descent trajectory of the probe with that
forcing relevant in GCMs.                                         calculated with the predicted zonal and meridional wind.
   This modiﬁcation introduces several quantitative               The descent trajectory was reconstructed by the Huygens
changes in the lower troposphere. First the temperature           DISR by visual ground tracking of the probe (Tomasko
drops by up to 2 K compared with the baseline simulation          et al., 2005; Karkoschka et al., 2007), as also mentioned
(Fig. 12). The temperature decrease is most pronounced            in Section 2.
near the surface. The predicted temperature does not ﬁt the          The descent trajectory of the probe can be calculated as
HASI temperature proﬁle at any altitude. On the other
                                                                  Xðt þ DtÞ ¼ XðtÞ þ ðu; v; wÞðtÞDt,                          (4)
hand the predicted lapse rate, which is smaller than in the
baseline simulation, agrees much better with that observed        where X is the instantaneous three-dimensional cartesian
by HASI. This suggests that at least the convective heat          coordinate of the probe, Dt ¼ 5 s is the time step interval
transport may be more realistically reproduced in the             and ðu; v; wÞ is the zonal, meridional and vertical drift speed
model.                                                            of the probe. It is assumed that the probe movement
   Another consequence of the reduced solar heating is the        immediately responds to the instantaneous wind speed
weakening of the tropospheric Hadley circulation while the        since the response time of Huygens was merely 3–5 s in the
circulation pattern does not change (Fig. 7). Compared            lowest 10 km (Bird et al., 2005), thus comparable with the
with the baseline simulation the mass streamfunction              time step interval in this calculation. Here, the instanta-
reduces by up to a factor of 2 in the lower troposphere.          neous values of u and v predicted by the GCM for the
This also means that the amount of angular momentum               Huygens site as a function of altitude are used. w is the
transported from one hemisphere to another on seasonal            probe’s descent speed that was reconstructed from the
timescales becomes smaller and the hemispherical                  temperature and pressure proﬁle measured by HASI (Harri
asymmetry is somewhat less pronounced. While the zonal            et al., 2006). The simulation is started at an altitude of
wind in the lower troposphere still exhibits a pair of            20 km and for the sake of a convenient comparison with
prograde and retrograde jet (Fig. 11e), both of them are          the observed trajectory (Karkoschka et al., 2007), the
weaker. Also the retrograde wind of the summer hemi-              coordinate is centred at the respective predicted landing
sphere does not extend as high as in the baseline                 point. Here, the x- and y-axis correspond to Titan’s east
simulation.                                                       longitude and latitude.
   The instantaneous zonal wind at the Huygens site                  Figs. 13a and b depict the calculated descent trajectory
(Fig. 6a) in the lower 5 km closely follows that in the           projected on the surface using the predicted vertical proﬁle
baseline simulation. Maximum retrograde wind occurs               of zonal and meridional wind with a different zoom. In the
near 5 km and above this level the wind speed decreases           baseline simulation (Simulation 1) the probe mostly drifts
and turns to prograde wind by 12 km. In other words,              westward after an initial clockwise loop near 20 km. There
the wind reversal takes place at a lower altitude than            is an additional slight meridional drift, which is initially
in the baseline simulation although this is still higher          southward and then turns to northward. The initial
than observed. Also the predicted maximum retrograde              westward drift is clearly inconsistent with observation
wind near 5 km is much sharper and thus resembles that            (Karkoschka et al., 2007), conﬁrming the wrong predicted
observed by the Huygens DWE (Folkner et al., 2006).               altitude of the zonal wind reversal. Furthermore, in
This change can be ascribed to the different magnitudes           comparison with the observation, the predicted northward
of the seasonal angular momentum transport and it                 drift below 7 km is underestimated by a factor of about 2.
seems that an even slightly weaker interhemispherical             This may indicate that the predicted meridional ﬂow in the
Hadley circulation could generate the observed wind               upper branch of the Hadley cell is too low.
proﬁle. However, the underestimated wind speed in                    Remarkably, the model predicts a sharp left turn by
the upper troposphere can probably not be readily                 more than 90 near 800 m altitude. Eventually the probe
generated by a further simple modiﬁcation of the Hadley           approaches the surface towards southeast to south-
circulation pattern, as this bias is common to all                southeast. While this is dissimilar to an Earth-like Ekman
simulations presented here. Perhaps the radiative heating         spiral, the last portion of the trajectory almost exactly ﬁts
pattern at higher altitudes would have to be reeva-               the observational data (Fig. 13b). Therefore, the lower
luated based, e.g. on information from Huygens. Never-            branch of the Hadley cell and/or the near-surface tidal
theless, this sensitivity study indicates that a change in        wind seems to be realistically reproduced. The GCM does
the radiative heating proﬁle in the troposphere can               not resolve the PBL below 300 m, but since also the
```

<!-- PDF_PAGE: 18 -->

## PDF page 18

```text
                                                                                             ARTICLE IN PRESS
                                                                    T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                        2007


                          Descent trajectory in the lower 20 km                                              westward drift almost without additional meridional drift.
         300
                                                                                         DISR
                                                                                                             Hence Saturn’s tidal force indeed seems to enhance the
                                                                                            1                overall meridional wind. Below 800 m the probe turns to
         200                                                                                2
                                                                                            3                the left, as in Simulation 1, but the southeastward drift is
                                                                                            4
                                                                                                      20
                                                                                                        15   smaller than in the presence of tide and less consistent with
         100
                                                                    5                20                      observation. Since the only difference between Simulation
 y [m]      0       20 20 15        10        5                     5 5
                                                                                      15 10
                                                                                     10                15
                                                                                                             1 and 2 is the presence or absence of tide the slightly better
                                                                                                      20
                                                                                         10                  agreement of the trajectory of Simulation 1 with the DISR
         -100                                                                                                data may be interpreted as a result of Saturn’s gravitational
                                                         5
                                                        10
                                                                                                             tide.
         -200                            15                                                                     The descent trajectory in Simulation 3 (without season-
                                                                                                             ality) is more inconsistent with the Huygens data than
         -300                                                                                                Simulations 1 and 2 because the predicted drift of the probe
            -4000        -2000                     0                2000             4000             6000
                                                                                                             is almost only eastward. Exceptionally, the drift in the ﬁnal
                                                            x [m]
                                                                                                             portion of the descent below 1 km is northwestward, but
                       Zoomed descent trajectory near the surface
                                                                                                             this is also opposite to what was observed by DISR. This
         300
                                                                                         DISR                discrepancy indicates that the presence of an equinox-type
                                                                                            1
         200                                                                                2                Hadley cell with an equatorward ﬂow near the surface and
                                                                                            3
                                                                                            4                a poleward ﬂow above it is unlikely to have existed at the
         100                                                                             3
                                                                                                             time of Huygens’ landing.
                                                        2
                         11         1
                                                                        2                                       The result of Simulation 4 (reduced heating rate) is
 y [m]
                                1
                                                             2
            0   2          1
                                    2                                                                        rather similar to that of Simulation 1 below 10 km, but the
                                                                            3                                westward drift is smaller and thus more consistent with the
         -100                                                                        4
                                                                                                             DISR data. The sharp left turn near 800 m is predicted as
                                                                                                 5
                                                                                10           9    8          well, although the eastward component is slightly weaker
                               12                           11
         -200
                                                                                                             than in Simulation 1. The anti-clockwise spiral in the
         -300                                                                                                highest part of the descent goes in the opposite sense from
             -200          0                      200               400              600               800   Simulation 1, but this sense is more consistent with the
                                                            x [m]                                            DISR data than the clockwise turn of Simulation 1
                                                                                                             although the observed spiral is not as narrow as predicted.
Fig. 13. Simulated descent trajectories of the Huygens probe using the
                                                                                                             However, the predicted southward drift direction in the
predicted horizontal wind speed and the measured probe descent speed
starting from an altitude of 20 km. Each trajectory is centred at the                                        ﬁnal portion of the descent below 1 km deviates from the
predicted landing site, where the x- and y-axis correspond to Titan’s                                        observed one by 45 .
longitude and latitude, respectively. The descent trajectory observed by the                                    While the reduction of the diabatic heating rate certainly
Huygens DISR (Karkoschka et al., 2007) is also shown for comparison.                                         weakens the Hadley circulation, this does not simply cause
Panel (a) shows the entire domain, but the x- and y-axis are not in scale
                                                                                                             a slower meridional wind speed, but also changes the
considering the larger longitudinal drift compared with the meridional
drift. Panel (b) shows the zoom of the landing site and the x- and y-axis are                                vertical proﬁle of zonal wind and the actual force balance
in scale. The numbers on each trajectory mark the altitude and the                                           at each altitude, as described in the previous section, so the
numbers on the margin denote the simulation numbers listed in Table 1.                                       prediction of the expected change in the descent trajectory
                                                                                                             is not straightforward, as illustrated in this simulation.
                                                                                                                As a whole none of the simulated descent trajectory can
Huygens probe could not directly constrain the probe drift                                                   exactly ﬁt the observed descent trajectory, but the
below 250 m for technical reasons (Karkoschka et al., 2007)                                                  combination of the northwestward drift below 7 km and
we do not pursue the detailed wind proﬁle below 300 m.                                                       the southeastward drift below 800 m is better reproduced
   The error in the predicted trajectory arising from the                                                    by Simulation 1 than in other simulations. This means that
assumption of an instantaneous response of the probe drift                                                   a combination of seasonally varying Hadley circulation
to the change in wind speed is negligible in the lower part of                                               with a cross-equatorial southward temperature increases
the troposphere. With a descent speed of less than 7 m s1                                                   near the surface at the time of landing and Saturn’s
(Harri et al., 2006), the vertical distance traversed during                                                 gravitational tide may substantially contribute to the
the 3–5 s response time is about 30 m, and the predicted                                                     observed wind proﬁle.
and observed vertical shear of the wind speed in this                                                           It is evident that the descent trajectory is sensitive to
vertical interval is only 7:5  103 m s1 . If a response time                                              various factors in the troposphere such as tide, seasonal
of 5 s is taken into account in the descent trajectory                                                       variation or solar heating rate. Although not explicitly
simulation, the predicted landing point shifts only by a few                                                 simulated in this study it is also likely that other factors
metres, so it was not plotted in the ﬁgure.                                                                  such as surface properties including topography or clouds
   Simulation 2, which excludes Saturn’s tide, generates a                                                   would also affect the near-surface wind proﬁle. Therefore,
slightly different descent trajectory. The probe performs a                                                  it is a formidable task to simultaneously tune the
```

<!-- PDF_PAGE: 19 -->

## PDF page 19

```text
                                                  ARTICLE IN PRESS
2008                                 T. Tokano / Planetary and Space Science 55 (2007) 1990–2009


parameters such as to exactly ﬁt the zonal and meridional           force. Above 800 m the major force balance is that between
wind as well as temperature.                                        the pressure gradient force, Coriolis force and Saturn’s
                                                                    gravitational tide. The cyclostrophic balance relevant in the
6. Discussion and conclusions                                       stratosphere is negligible in the lower troposphere anyway.
                                                                       The inﬂuence of Saturn’s tide is rather small at the
   This study has shown in some detail how sensitively the          Huygens site given its proximity to the symmetry axis of
wind near the surface depends on various forces as well as          the tide, i.e. the equator. While the zonal and meridional
how the zonal and meridional wind depend on each other.             wind do not exhibit a clear signature of tidal wind, the
The observed wind proﬁle does not resemble characteristic           southeastward wind near the surface can be generated in
wind proﬁles of the terrestrial PBL such as an Ekman                the presence of tide.
spiral of a neutrally stratiﬁed PBL, a uniform wind proﬁle             However, it should be noted that the near-surface wind
characteristic of a strongly convective PBL or a low-level          direction is likely to be modiﬁed by local topography, as
jet typical for the nocturnal PBL. Instead the global-scale         pointed by Lorenz et al. (2006) after examining the
atmospheric dynamics seems to control the actual wind               orientation of sand dunes. For this reason a ﬁrm
proﬁle.                                                             conclusion about the signiﬁcance of the tidal wind on the
   Unlike in the stratosphere, which is characterised by            basis of the single measurement by Huygens is difﬁcult.
super-rotating prograde winds in either hemisphere (Bird               The result of this model study has several implications
et al., 2005; Flasar et al., 2005; Luz et al., 2005; Kostiuk et     for Titan’s tropospheric meteorology and geophysics. First,
al., 2006), this GCM suggests that the zonal wind direction         this study suggests the likelihood of a warmer southern
in the lower troposphere varies with season. This occurs            (summer) hemisphere in comparison with the equator. If
because the cyclostrophic wind balance breaks down near             so, this would favour the insolation hypothesis for the
Titan’s surface and thus the hemispheric asymmetry in the           generation of convective clouds (Brown et al., 2002;
temperature ﬁeld gives rise to a global Hadley circulation          Tokano, 2005; Schaller et al., 2006). Secondly, the seasonal
that transport angular momentum from one hemisphere to              reversal of the tropospheric Hadley circulation implies that
the other. As a consequence, prograde and retrograde wind           some exchange of angular momentum is likely to occur on
arise, depending on whether the temperature decreases or            seasonal timescales between the surface and atmosphere,
increases with latitude. The detected substantial retrograde        with a possible impact on a seasonal variation in Titan’s
winds in the lowest few kilometres of the Huygens site are          length-of-day (Tokano and Neubauer, 2005).
consistent with this mechanism, and may be contrasted to               The simulation also indicates that the temporal variation
the lower atmosphere of Venus that lacks a reversal of the          in both the zonal and meridional wind in the lower
zonal wind direction and substantial seasonality (Gierasch          troposphere largely depends on the strength of the seasonal
et al., 1998). The simulation shows that, if there were no          temperature variation and Hadley circulation. Thus, the
seasonal variation in the tropospheric temperature and              strength of the seasonality in Titan’s troposphere may
Hadley circulation pattern, the equatorial wind would have          depend on how much sunlight penetrates to Titan’s surface
been prograde down to very close to the surface. Thus one           and how asymmetric the surface temperature becomes at
major difference between Venus and Titan, both represent-           solstice. The incorrect prediction of the altitude of the
ing slowly rotating planetary bodies with a thick atmo-             zonal wind reversal is likely to reﬂect inaccuracies in the
sphere, is the impact of seasonal variation on the zonal            GCM in calculating the radiative ﬂuxes and thus of
wind.                                                               inaccurate reproduction of the Hadley circulation or
   The meridional wind at the Huygens site is mainly a              temperature. In situ data on the properties of atmospheric
manifestation of the cross-equatorial Hadley circulation.           opacity sources or radiative ﬂuxes at the Huygens site
While immediately above the surface the meridional ﬂow is           currently under analysis by the Huygens teams can
southward following the southward pressure gradient force           probably improve these details of the GCM in the near
and representing the lower branch of the Hadley circula-            future. In addition, possible future balloon missions to
tion, a reversed ﬂow is found above 1 km, representing the          Titan would further constrain the wind system in Titan’s
upper branch of the Hadley circulation. The altitude of the         lower troposphere (Tokano and Lorenz, 2006).
zonal and meridional wind reversal should depend on up to
which the altitude seasonal variation in temperature                Acknowledgements
induced by the sensible heat ﬂux from the surface can be
felt. The occurrence of the wind reversal between 7 and                The author received a grant from the DFG (Deutsche
10 km (Karkoschka et al., 2007) indicates that the                  Forschungsgemeinschaft) in the priority programme
horizontal temperature gradient may change sign a few               ‘‘Mars and the Terrestrial Planets’’. He is also grateful
km below this altitude.                                             for Björn Grieger and an anonymous reviewer for
   Another wind reversal closer to the surface (800 m)              constructive suggestions. The Java program Titan24
indicates a transition between two regimes with different           (available at http://www.giss.nasa.gov/tools/titan24) was
force balances. Below 800 m the major force balance is that         used to calculate the astronomical parameters used in this
between Saturn’s gravitational tide and pressure gradient           study.
```

<!-- PDF_PAGE: 20 -->

## PDF page 20

```text
                                                               ARTICLE IN PRESS
                                               T. Tokano / Planetary and Space Science 55 (2007) 1990–2009                                              2009


References                                                                        Lorenz, R.D., et al., 2006. The sand seas of Titan: Cassini RADAR
                                                                                     observations of longitudinal dunes. Science 312, 724–727.
Bird, M.K., et al., 2005. The vertical proﬁle of winds on Titan. Nature 43,       Luz, D., Hourdin, F., Rannou, P., Lebonnois, S., 2003. Latitudinal
    800–802.                                                                         transport of barotropic waves in Titan’s stratosphere. II. Results from
Bouchez, A.H., Brown, M.E., 2005. Statistics of Titan’s south polar                  a coupled dynamics–microphysics–photochemistry GCM. Icarus 166,
    tropospheric clouds. Astrophys. J. 618, L53–L56.                                 343–358.
Brown, M.E., Bouchez, A.H., Grifﬁth, C.A., 2002. Direct detection of              Luz, D., Civeit, T., Courtin, R., Lebreton, J.-P., Gautier, D., Rannou, P.,
    variable tropospheric clouds near Titan’s south pole. Nature 420,                Kaufer, A., Witasse, O., Lara, L., Ferri, F., 2005. Characterization of
    795–797.                                                                         zonal winds in the stratosphere of Titan with UVES. Icarus 179,
Flasar, F.M., et al., 2005. Titan’s atmospheric temperatures, winds, and             497–510.
    composition. Science 308, 975–978.                                            McKay, C.P., Pollack, J.B., Courtin, R., 1989. The thermal structure of
Folkner, W.M., Asmar, S.W., Border, J.S., Franklin, G.W., Finley, S.G.,              Titan’s atmosphere. Icarus 80, 23–53.
    Gorelik, J., Johnston, D.V., Kerzhanovich, V.V., Lowe, S.T., Preston,         Niemann, H.B., et al., 2005. The abundances of constituents of Titan’s
    R.A., Bird, M.K., Dutta-Roy, R., Allison, M., Atkinson, D.H., Edenhofer,         atmosphere from the GCMS instrument on the Huygens probe.
    P., Plettemeier, D., Tyler, G.L., 2006. Winds on Titan from ground-based         Nature 438, 779–784.
    tracking of the Huygens probe. J. Geophys. Res. 111, E07S02.                  Porco, C.C., et al., 2005. Imaging of Titan from the Cassini spacecraft.
Fulchignoni, M., et al., 2005. In situ measurements of the physical                  Nature 434, 159–168.
    characteristics of Titan’s environment. Nature 438, 785–791.                  Rannou, P., Hourdin, F., McKay, C.P., Luz, D., 2004. A coupled
Gierasch, P.J., et al., 1998. The general circulation of the Venus                   dynamics–microphysics model of Titan’s atmosphere. Icarus 170,
    atmosphere: an assessment. In: Bougher, S.W., Hunten, D.M.,                      443–462.
    Phillips, R.J., et al. (Eds.), Venus II. University of Arizona Press,         Rannou, P., Montmessin, F., Hourdin, F., Lebonnois, S., 2006. The
    Tucson, pp. 459–500.                                                             latitudinal distribution of clouds on Titan. Science 311, 201–205.
Grieger, B., Segschneider, J., Keller, H.U., Rodin, A.V., Lunkeit, F., Kirk,      Roe, H.G., Brown, M.E., Schaller, E.L., Bouchez, A.H., Trujillo, C.A.,
    E., Fraedrich, K., 2004. Simulating Titan’s tropospheric circulation             2005. Geographic control of Titan’s mid-latitude clouds. Science 310,
    with the portable university model of the atmosphere. Adv. Space Res.            477–479.
    34, 1650–1654.                                                                Schaller, E.L., Brown, M.E., Roe, H.G., Bouchez, A.H., 2006. A large
Grifﬁth, C.A., Penteado, P., Baines, K., Drossart, P., Barnes, J.,                   cloud outburst at Titan’s south pole. Icarus 182, 224–229.
    Bellucci, G., Bibring, J., Brown, R., Buratti, B., Capaccioni, F.,            Stull, R.B., 1988. Introduction to Boundary Layer Meteorology. Kluwer
    Cerroni, P., Clark, R., Combes, M., Coradini, A., Cruikshank, D.,                Academic Publishers, Dordrecht.
    Formisano, V., Jaumann, R., Langevin, Y., Matson, D., McCord, T.,             Tokano, T., 2005. Meteorological assessment of the surface temperatures
    Mennella, V., Nelson, R., Nicholson, P., Sicardy, B., Sotin, C.,                 on Titan: constraints on the surface type. Icarus 173, 222–242.
    Soderblom, L.A., Kursinksi, R., 2005. The evolution of Titan’s mid-           Tokano, T., Lorenz, R.D., 2006. GCM simulation of balloon trajectories
    latitude clouds. Science 310, 474–477.                                           on Titan. Planet. Space Sci. 54, 685–694.
Harri, A.-M., Mäkinen, T., Lehto, A., Kahanpää, H., Siili, T., 2006.           Tokano, T., Neubauer, F.M., 2002. Tidal winds on Titan caused by
    Vertical pressure proﬁle of Titan—observations of the PPI/HASI                   Saturn. Icarus 158, 499–515.
    instrument. Planet. Space Sci. 54, 1117–1123.                                 Tokano, T., Neubauer, F.M., 2005. Wind-induced seasonal angular
Hourdin, F., Talagrand, O., Sadourny, R., Courtin, R., Gautier, D.,                  momentum exchange at Titan’s surface and its inﬂuence on Titan’s
    McKay, C.P., 1995. Numerical simulation of the general circulation of            length-of-day. Geophys. Res. Lett. 32, L24203.
    the atmosphere of Titan. Icarus 117, 358–374.                                 Tokano, T., Neubauer, F.M., Laube, M., McKay, C.P., 1999. Seasonal
Karkoschka, E., Tomasko, M.G., Doose, L.R., Rizk, B., See, C.,                       variation of Titan’s atmospheric structure simulated by a general
    McFarlane, L., Schröder, S., 2007. DISR imaging and the geometry                circulation model. Planet. Space Sci. 47, 493–520.
    of the descent of the Huygens probe within Titan’s atmosphere. Planet.        Tokano, T., Ferri, F., Colombatti, G., Mäkinen, T., Fulchignoni, M.,
    Space Sci., in press, doi:10.1016/j.pss.2007.04.019.                             2006. Titan’s planetary boundary layer structure at the Huygens
Kostiuk, T., Livengood, T.A., Sonnabend, G., Fast, K.E., Hewagama, T.,               landing site. J. Geophys. Res. 111, E08007.
    Murakawa, K., Tokunaga, A.T., Annen, J., Buhl, D., Schmülling, F.,           Tokano, T., Neubauer, F.M., Laube, M., McKay, C.P., 2001. Three-
    Luz, D., Witasse, O., 2006. Stratospheric global winds on Titan at the           dimensional modeling of the tropospheric methane cycle on Titan.
    time of Huygens descent. J. Geophys. Res. 111, E07S03.                           Icarus 153, 130–147.
Lellouch, E., Coustenis, A., Gautier, D., Raulin, F., Dubouloz, N., Frère, C.,   Tomasko, M.G., et al., 2005. Rain, winds and haze during the Huygens
    1989. Titan’s atmosphere and hypothesized ocean: a reanalysis of the             probe’s descent to Titan’s surface. Nature 438, 765–778.
    Voyager 1 radio-occultation and IRIS 7. 7-mm data. Icarus 79, 328–349.        Zhu, X., Strobel, D.F., 2005. On the maintenance of thermal wind balance
Lorenz, R.D., 2006. Thermal interactions of the Huygens probe with the Titan         and equatorial superrotation in Titan’s stratosphere. Icarus 176,
    environment: constraint on near-surface wind. Icarus 182, 559–566.               331–350.
```
