---
citation_key: "karkoschka2022titan"
title: "Titan's haze at opposite seasons from HST-STIS spectroscopy"
source_pdf: "data/papers/karkoschka2022titan.pdf"
source_pdf_sha256: "8179b79a8ad7111884012d6fa980ab1958b9c5c0b505136aa9732ca0343d5c1e"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                Titan's haze at opposite seasons
                  from HST-STIS spectroscopy

Item Type         Article

Authors           Karkoschka, Erich

Citation          Karkoschka, E. (2022). Titan’s haze at opposite seasons from
                  HST-STIS spectroscopy. Icarus, 387.

DOI               10.1016/j.icarus.2022.115188

Publisher         Elsevier BV

Journal           Icarus

Rights            © 2022 Elsevier Inc. All rights reserved.

Download date     06/02/2025 07:44:14

Item License      http://rightsstatements.org/vocab/InC/1.0/

Version           Final accepted manuscript

Link to Item      http://hdl.handle.net/10150/665718
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                                       1


 1         Titan’s haze at opposite seasons from HST-STIS
 2                                          spectroscopy
 3

 4                                          Erich Karkoschka
 5   Erich Karkoschka, Lunar and Planetary Laboratory, University of Arizona, Tucson, AZ 85721-
 6   0092, USA, phone 1 (520) 621-3994, fax 1 (520) 621-4933, e-mail erich@lpl.arizona.edu
 7
 8   Pages:     17
 9   Figures:      5
10   Tables:    2
11   Proposed Running Head: Titan’s haze at opposite seasons
12   Highlights:
13   > HST-STIS image cubes over 22 years provide a unique probe of atmospheric variations.
14   >Two switches of Titan’s two-component north-south asymmetry were observed.
15   >Both components at each latitude follow remarkably well a harmonic oscillation.
16   >The low-altitude component lags 2.3 years behind the high-altitude component.
17   >Our seasonal model fits all data at latitudes to +/- 70 degrees at all eight observed dates.
18
19   Key-words:        Titan; Titan atmosphere; Atmospheres, structure; Atmospheres, dynamics;
20   Satellites, atmospheres.

21   Submitted to Icarus: 2022-04-11                                  Revised: 2022-06-05, 2022-07-11
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                      2


22                                              Abstract
23         We present an analysis of three new image cubes of Titan by the Space Telescope Imaging
24   Spectrograph taken in 2017, 2018, and 2019, half a Titan year after previously analyzed image
25   cubes. Both data sets probe periods when Titan’s seasonal north-south-asymmetry switched.
26   The new observations show that the new reversal came exactly half a Titan year after the
27   previous opposite reversal. On the other hand, the phase lag of the reversals with respect to
28   Titan’s equinoxes was different indicating that the seasonal variation is close to harmonic and
29   does not follow variations due to Saturn’s orbital eccentricity. The reversal had two components,
30   a major one at altitudes below 80 km reversing two years after a minor one above 150 km. The
31   observations further revealed small temporary deviations of less than 10 % of the seasonal
32   amplitude. The new observations provide an improved seasonal model of Titan that gives
33   accurate constraints for future global circulation models.
34

35                                     Graphical Abstract




36
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                      3


37                                          1. Introduction
38         Titan’s atmosphere has a well observed seasonal variation. In 1980 and 1981, the Voyager
39   spacecraft found Titan’s atmosphere almost featureless except for a noticeable asymmetry with a
40   bright northern hemisphere and a darker southern hemisphere (Smith et al. 1981, 1982). The
41   Hubble Space Telescope (HST) took images at similar wavelengths about two Titan seasons later
42   revealing that the asymmetry had switched (Caldwell et al. 1992), consistent with a seasonal
43   change. Photometry of Titan since 1972 at 472 and 551 nm wavelength indicated an almost
44   periodic variation with a period of 15 years, half a Titan year (Lockwood and Thompson 2009).
45   The phase of the variation excluded the possibility of a pure geometric effect due to changing
46   sub-Earth latitudes.
47         In the 889 nm methane band, HST images of 1990 found the asymmetry to be opposite to
48   that of visible wavelength which was consistent with models by Toon et al. (1992) showing that
49   an increase in haze optical depth causes darkening at wavelengths below 600 nm but brightening
50   at longer wavelengths.
51         Titan’s north-south asymmetry has been further explored in wavelength and time through
52   observations from Earth, Earth orbit, and Cassini. Ground-based telescopes with adaptive optics
53   were among the first to measure the north-south asymmetry in greater spatial detail (Coustenis et
54   al., 2001; Roe et al., 2002, Ádámkovics et al., 2006). Cassini followed with much better
55   resolution (de Kok et al., 2010; Penteado et al., 2010; Teanby et al., 2012; Vinatier et al., 2015).
56   HST contributed significantly to record the temporal evolution of the asymmetry because its
57   observations are not affected by the varying seeing or atmospheric transmission of ground-based
58   telescopes and almost neither by the varying phase angle of Cassini observations.
59         Observations by the Space Telescope Imaging Spectrograph (STIS) gave more complete
60   spectral characterizations than HST imaging in a few filters (Karkoschka and Lorenz 1996;
61   Lorenz et al., 2006) since STIS can obtain image cubes with 1024 wavelengths of the whole disk
62   of Titan at full spatial resolution within one orbit around Earth. Such image cubes were taken in
63   1999 and 2000, also in 2002 with less s spatial resolution. Furthermore, partial image cubes were
64   obtained in 1997 and 2004. These five image cubes were analyzed by Karkoschka (2016). Their
65   superb spectral information revealed that the north-south asymmetry is caused by two distinct
66   cycles. The major cycle involves low altitudes (<80 km) with a north-south switch occurring in
67   2003-2004, a 90° phase lag with respect to Titan’s fall equinox in 1995. The minor cycle
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                               4

68   involves high altitudes (>150 km) with a switch occurring in 2001-2002, corresponding to a
69   smaller phase lag.
70          In this work, we analyze new STIS image cubes of the full disk at full spatial resolution
71   taken in 2017, 2018, and 2019. They record the opposite switch, including the timing and speed
72   during the switch. Both data sets were reduced in the same way in order to detect fine differences.
73   The following Section 2 explains the new observations and data reduction. Only a summary is
74   given since the details were listed in Karkoschka (2016). Section 3 details the modeling and
75   comparison with the previous data. Section 4 provides a summary.
76

77                                               2. Data reduction
78          HST took new images cubes of Titan with STIS in 2017, 2018, and 2019 (Table 1). The
79   G750L grating provided wavelengths between 532 and 1016 nm at 0.49 nm/pixel over 1024
80   pixels. The slit width was 0.05 arc-sec. The spatial sampling was 0.05 arc-sec/pixel along the
81   slit, providing up to 16 pixels across the disk of Titan. Spatial sampling perpendicular to the slit
82   was accomplished through 15-16 exposures in each year separated by 0.05 arc-sec. Fig. 1 shows
83   the slit locations on Titan’s disk according to our calibration.
84

85   Table 1: Observational parameters
86   Date, UT              Ls       lg(earth) φ(earth) φ(sun)             diameter      phase angle # of exposures
87                         (°)      (º)          (º)         (º)          (arc-sec)      (º)
88   -----------------------------------------------------------------------------------------------------------------
89   2017-05-17.72           90     225           26.5        26.8        0.775         2.8             16
90   2018-06-08.79         102      312           25.7        26.1        0.780         1.9             16
91   2019-08-10.52         115      240           24.9        24.1        0.774         3.1             15
92   ------------------------------------------------------------------------------------------------------------------
93   Note: lg and φ are sub-earth/sub-solar longitudes and latitudes, respectively. Ls is the solar
94   longitude relative to the spring equinox.
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                       5




95
96    Fig. 1: The geometry for all slit positions on Titan’s disk, shown for each observing date. North
97    on Titan is oriented upwards in each image. Titan’s equator is shown by the bright curve. The
98    direction to the sun is also shown. The arrows at the bottom give the direction of offset of the slit
99    from one exposure to the next. Even the only 15 slit positions in 2019 provided very close to
100   complete coverage of Titan’s disk.
101
102         The observations were placed near the edge of the CCD (aperture 52X0.05E1) to minimize
103   charge-transfer losses. After spatial and wavelength calibration of each data point to about 0.002
104   arc-sec and 0.03 nm, each image cube was resampled at 0.06 Titan-radii and 0.4 nm wavelength.
105   Separate deconvolutions were done for the CCD scattering and the telescope diffraction.
106         The ~1000 spectral data points were divided into 25 groups of about 40 spectral points each
107   that varied little in wavelength and methane absorption coefficient (top of Fig. 2). These 25
108   groups had eight different average wavelengths and eight different average methane absorption
109   coefficients. This allowed us to see how Titan’s reflectivity changes with wavelength at constant
110   methane absorption coefficient and vice versa.
111         This work focuses on Titan’s atmosphere, but the data also probe its surface, in particular at
112   low methane absorptions and long wavelengths (Fig. 2). The visible features correspond well
113   with Cassini ISS map PIA22770 (https://photojournal.jpl.nasa.gov/catalog/PIA22770) when
114   generated for the same viewing angle and spatial resolution (right column in Fig. 2). For each
115   spectral band, the correlation with the ISS map was determined which then allowed us to subtract
116   the surface features.
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                       6




117
118   Fig. 2: Titan images for our 25 spectral bands, combinations of central wavelength and methane
119   absorption coefficient listed on top, and three observing sessions listed on the left side. To better
120   display small variations, an average intensity distribution shown in the top row was subtracted
121   before the difference was displayed with quadruple contrast. The column at right shows Titan
122   surface albedo features measured by Cassini ISS. The bottom four rows show contributions from
123   the first two principal components to Titan’s disk reflectivity in 2017 and 2019.
124
125         Titan’s disk was divided into 10 latitude regions separated by 10°, centered on 25° South
126   up to 65° North. In each latitude region the reflectivity data points were fit to a linear function
127   with the cosine of emission angle. The fitted reflectivities at cosines of 0.6 and 0.8 represent two
128   parameters, such as reflectivity and limb darkening, that were used to compare data at different
129   latitudes and different times.
130         In summary, we reduced the data from about 600,000 original data points to 1500 data
131   points: 25 spectral bands, 10 latitudes, two center-to-limb locations, and three observing dates.
132

133                             3. Principal component analysis
134         We applied the same method of principal component analysis that was used for the STIS
135   observations of 1997-2004 (Karkoschka 2016). While the 1500 reflectivity data points are
136   functions of four parameters (spectral band, latitude, center-to-limb position, time), the method
137   tries to fit the data as it were the product of two functions, a function A1 only dependent on
138   spectral band and center-to-limb position and a function B1 only dependent on latitude and time.
139   The function A1 describes a physical variation in Titan’s atmosphere, such as a change of optical
140   depth below 80 km altitude. The function B1 describes the magnitude of this change as function
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                       7

141   of latitude and time. Note that the average state of Titan was subtracted from all data points so
142   that the principal component analysis only deals with deviations from the average state.
143         I/F(λ,μ,φ,t) = I/Faverage(λ,μ) + A1(λ,μ) B1(φ,t) + A2(λ,μ) B2(φ,t) + …                     (1)
144         where I/F is the reflectivity, λ the wavelength, μ the center-to-limb position, φ the latitude,
145   and t the time.
146         Since the first principal component, the product of A1 and B1, only approximates the data
147   to about 70 %, the residuals are fitted with the second principal component, the product of
148   functions A2 and B2, which explains another ~25 % of the variation in the data. The remaining
149   principal components are due to noise, imperfections of the surface feature subtraction, or very
150   small physical variations. The third principal component may be related with a non-linearity of
151   the first principal component (Karkoschka 2016).
152         The 2.2 year time period of the new observations is too short to describe seasonal variations
153   with a period of 29.5 years accurately. Thus, we added the previous STIS observations of 1997-
154   2004 to the data set, which gave almost identical first and second principal components as using
155   the previous data set alone. This means that the conclusions about the physical nature of both
156   components stand firm, mainly that the first component corresponds to a variation of the haze
157   optical depth below 80 km altitude, and the second one above 150 km altitude.
158         Karkoschka (2016) showed that the structure of the 50 data points each of A1 and A2 (25
159   spectral bands for two center-to-limb positions) match radiative transfer models of Titan’s
160   atmosphere best if the haze optical depth is varied and all other parameters remain constant, and
161   that A1 is best approximated if the optical depth is varied between 0 and 80 km altitude, and A2
162   similarly for altitudes above 150 km. We used the Titan’s haze model from Doose et al. (2016)
163   for the Huygens landing site during the descent of Huygens. It divides the atmosphere into four
164   layers: above 200 km, then down to 80 km, to 30 km, and below. Each layer has its own aerosol
165   size with phase functions derived from fractal particles, high single scattering albedos, up to unity
166   at 800 nm wavelength and low altitudes, and its own function of optical depth with altitude,
167   depending strongly on wavelength. The haze optical depths determined in this work at a certain
168   latitude and time divided by the haze optical depth in the Doose et al. model is called the haze
169   opacity factor, which is unity between 80 and 150 km altitude. Single scattering albedos and
170   phase functions are from Doose et al. without change.
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                                     8




171
172
173   Fig. 3:   Principal component values B1 (top) and B2 (bottom), ordered first according to
174   observing date (top of each panel), and then according to latitude (bottom scale). Error bars are
175   indicated at the top. Curves are for the seasonal model of Karkoschka (2016, dashed) and the
176   new seasonal model of this work (solid). The small circles in the 2017 panel are a prediction
177   from data half a Titan-year earlier, an interpolation between the 2000 and 2002 observations,
178   with north and south reversed.
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                       9

179
180         We also note that the seasonal model of Titan’s haze by Karkoschka (2016) using the 1997-
181   2004 observation predicts the new observations reasonably well, which is our first result (dashed
182   curves in Fig. 3). Another comparison is shown in the panel of the 2017 observations, where the
183   2017 observations (dots) are compared with the expected ones from half a Titan-year earlier
184   (crosses). The fit is remarkably good, especially for the second principal component. This is the
185   best suitable comparison at opposite seasons since the 2004 observations are less reliable due to
186   their incomplete coverage of Titan’s disk. Note that the adopted seasonal model gives the reverse
187   latitudinal distribution after half a Titan year, an assumption that could be far off. The new
188   observations show that this assumption was quite good for the 2002 versus 2017 comparison.
189         The seasonal model does not fit the 2018 and 2019 observations as well as those of other
190   years. There are at least three possibilities. First, Titan may change periodically but somewhat
191   differently at opposite seasons, such as changing at faster speeds around 2018 than around 2003.
192   Second, every change may have their individual signature. Third, Titan changes symmetrically at
193   opposite seasons, but the opposite season of the previous period ended in 2018 and the
194   extrapolation to 2019 was not accurate. We found that the third possibility is likely the correct
195   explanation.
196         We did a new seasonal model using a least square method with all data points. It fits the
197   2018 and 2019 data better without making the fit to the other data significantly worse (solid
198   curves in Fig. 3). The 1997 and 2004 observation are not fit as well, but those data are unreliable
199   since they only cover part of Titan’s disk. This was a weakness of the previous data set. Thus,
200   we could improve our model because the 2019 observations extend the temporal baseline even
201   when accounting for the anti-symmetry after 14.7 years.         The parameters of the principal
202   components B1 and B2 are listed in Table 2. The seasonal model describes B1 and B2 as
203   function of latitude φ and time t (in years):
204         Bi(φ,t)= Mi(φ) + Di(φ) sin [(t – Ei(φ)) 2π/29.46]                                    (2)
205         The 42 parameters of the seasonal model are displayed in Fig. 4.
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                                                     10




206
207   Fig. 4: The three parameters of the seasonal model for the first (solid dot and curves) and second
208   principal component (open squares and dashed curves), each as function of latitude according to
209   Eq. 2. Southern latitudes are assumed symmetric to northern ones, except for a change of sign
210   for D1 and D2.
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
                                                              11


211         Table 2: Principal component values B1 and B2
212                                B1                                      B2
213                  ------------------------------          -------------------------------
214         Latitude 2017        2018        2019            2017        2018         2019
215         ----------------------------------------------------------------------------------
216         -25°        -0.89       -0.93        -0.47        0.58        1.40         0.25
217         -15°        -0.96       -1.16        -0.71        0.99        1.87         2.35
218          -5°        -0.94       -1.16        -0.80        1.08        1.76         1.98
219           5°        -0.90       -1.20        -0.87        0.70        1.10         1.16
220         15°         -0.75       -1.20        -0.87        0.03        0.15        -0.11
221         25°         -0.54       -1.04        -0.95       -0.65        -0.70       -1.20
222         35°         -0.31       -0.85        -0.87       -1.05        -1.25       -2.06
223         45°          0.23       -0.54        -0.62       -1.35        -1.78       -2.87
224         55°          1.02       -0.21        -0.34       -1.64        -2.23       -3.47
225         65°          1.24       -0.18        -0.20       -2.01        -2.43       -4.18
226         ----------------------------------------------------------------------------------
227
228         The seasonal model has the reversal for the first principal component (low altitudes)
229   occurring during 2003 (2003.6±0.2), consistent with Karkoschka (2016), but the second principal
230   component (high altitudes) seems to switch in early 2001 (2001.3±0.2) rather than late 2001
231   suggested from the more limited earlier data set. This gives a phase shift of 2.3±0.3 years
232   between both oscillations. The phase lags of the first and second principal component with
233   respect to Titan’s seasons are 91° and 63°, respectively. These numbers correspond to Saturn’s
234   mean motion around the sun, ignoring the orbital eccentricity. The apparent variations of the
235   reversal times with latitude are close to the expected uncertainty and probably not significant.
236         The new seasonal model in terms of haze opacity factor at low and high altitude is shown in
237   Fig. 5 for all investigated latitudes. The new constraints at opposite seasons make the model
238   more reliable. However, one still needs to consider that STIS never observed Titan around the
239   equinoxes, which means that the seasonal model is an extrapolation based on the assumption of
240   harmonic oscillations. Also, high southern latitudes cannot be observed from Earth around
241   Titan’s northern summer solstice, and the same holds true for high northern latitudes near the
242   winter solstice. The seasonal model may not be accurate there.
```

<!-- PDF_PAGE: 13 -->

## PDF page 13

```text
                                                      12




243
244   Fig. 5: The variation of values B1 (top) and B2 (middle) as function of time over a whole Titan
245   year for 14 latitudes from -65° to 65° latitude, every 10°. The six latitudes within 30° of the
246   Equator are shown as solid curves, the ones further north and south as dotted and dashed,
247   respectively. The bottom panel shows the sub-solar latitude. The eight dots indicate the dates for
248   our observations, the horizontal bar the duration of the Cassini mission at Saturn, and the vertical
249   dotted lines the equinoxes and solstices. The scales at the right show approximate haze opacity
250   factors for altitudes below 80 km (top panel) and above 150 km (middle panel).
251
252         We wanted to test whether Titan’s eccentricity causes a non-harmonic oscillation in the
253   atmospheric response. The duration from fall equinox in 1995 to spring equinox in 2009 is 13.7
254   years while the following duration to the next fall equinox in 2025 is 15.7 years. This might
255   suggest that the reversals of the north-south asymmetry around 2017 occur 13.7 years after those
```

<!-- PDF_PAGE: 14 -->

## PDF page 14

```text
                                                      13

256   around 2003. However, allowing for a non-harmonic contribution, our analysis gives best fits of
257   14.5±0.3 years between reversals for the low altitude variation and 14.7±0.3 years for the high-
258   altitude variation. Both values are consistent with half a Titan year of 14.7 years but inconsistent
259   with the 13.7 years guessed above. This result might be expected considering the long phase lags
260   of both variations, especially the low altitude change. Long phase lags usually indicate a system
261   that has little response to forcing of higher frequencies. Titan’s eccentricity effectively adds a
262   small contribution of a second harmonic to the first harmonic.
263         If the second harmonic is shifted by half a season, the time between reversals remains half a
264   Titan year, but the speeds near both reversals are different. Again, there is no evidence in the
265   data set for this. An added second harmonic does not decrease residuals.
266         Seasonal variations such as in Equation (2) can also be expressed as a function of Saturn’s
267   orbital longitude, often with zero longitude defined as Titan’s spring equinox. We tried this and
268   found the residuals to be 5 % larger than when using time as in Equation (2). The fact that
269   residuals are similar can be expected since the difference between both curves is small around the
270   solstices where we have data. The fact that eccentricity does not seem to matter may also be
271   expected because the phase lags are large. The system is responding roughly to the average solar
272   illumination over the last two seasons, not particularly to the illumination at the time of
273   observation.
274         We detected small but significant short-lived deviations from the harmonic oscillations at
275   certain latitudes and times. The most significant deviation occurred in June 2018 (Fig. 3) when
276   the observed state at low altitudes (principal component 1) had an optical depth lower by 10 % of
277   the seasonal amplitude or 5 % of the maximum minus minimum. This is somewhat comparable
278   to the photometry data by Lockwood and Thompson (2009) that also showed that small, irregular
279   deviations from the harmonic oscillation occur. These short-term deviations do not repeat itself
280   after a cycle. Each cycle has a slightly different shape. Variations of the haze opacity of
281   comparable magnitude, 5-10 %, and time scales of weeks to months were tracked by Nichols-
282   Fleming et al. (2021).     The presence of these short-term variations limits the accuracy of
283   determining seasonal variations based on a handful of observation dates.
284

285                                             4. Summary
286         Titan’s north-south asymmetry has been observed for decades with various instruments.
```

<!-- PDF_PAGE: 15 -->

## PDF page 15

```text
                                                       14

287   One of the best data sets describing the asymmetry comes from STIS image cubes at ~1000
288   wavelengths between 520 and 1020 nm. Images cubes in five of the years 1997-2004 revealed
289   that the north-south asymmetry is a superposition of two distinct variations, a low-altitude
290   variation causing most of the observed changes and a high-altitude variation contributing
291   (Karkoschka 2016). Both are seasonal variations with the period of a Titan year, but the low-
292   altitude variation is two years behind the high-altitude variation in phase.
293         New STIS image cubes taken in 2017, 2018, and 2019 captured the reversals of both
294   variations at the opposite season compared to the earlier image cubes. They were specifically
295   designed for the goal of this investigation and thus cover the whole disk of Titan at full
296   resolution. For the previous STIS image cubes, this was only true for two years out of five since
297   those observations had other goals. The full set of eight image cubes was analyzed here.
298         Our main result is that the seasonal model based on the previous image cubes (Karkoschka
299   2016) predicted the new observations remarkably well. This is even true for high northern
300   latitudes that were not observable in the previous image cube. Their seasonal model was inferred
301   based on symmetry with respect to the equator except for a phase shift of half a Titan year. This
302   means that both variations at low and high altitudes can be well approximated by a simple
303   harmonic, seasonal variation.
304         We tested whether the seasonal variation can be better approximated by a slightly non-
305   harmonic oscillation that accounts for the variable orbital speed of Saturn due to its orbital
306   eccentricity which affects the timing of Titan’s seasons. We found that the observed seasonal
307   variation is not significantly affected by the finite eccentricity and much better approximated by
308   pure harmonic oscillations.     Since we only tried the simplest deviations from a harmonic
309   oscillation, we cannot say whether more sophisticated variations might better fit the data.
310         The amplitudes of the seasonal variation at both altitudes increase almost perfectly linearly
311   with latitude up to about 70° North and South. This is consistent with measurements based on
312   only the earlier image cubes.
313         For the annual average of Titan’s haze opacity, the first principal component (low-altitude
314   variation) has slightly lower values at the equator than at high latitudes while this is opposite for
315   the second principal component (high-altitude variation). This was also suggested already from
316   the first five image cubes alone.
317         The phase lags of the variation of Titan’s haze opacity with respect to Titan’s seasons is
```

<!-- PDF_PAGE: 16 -->

## PDF page 16

```text
                                                        15

318   91° at low altitudes (<80 km) and 63° at high altitudes (>150 km). The difference of 28°
319   corresponds to a relative delay of 2.3±0.3 years, a little more than the almost 2 years determined
320   from the previous smaller data set. The new value is more reliable.
321         Our data is limited to two periods around Titan’s solstices. Thus, our seasonal model may
322   be wrong around the equinoxes. Around the solstices, the tropics can be probed well from Earth,
323   but higher latitudes only on the summer hemisphere. Thus, our seasonal model may be also
324   wrong at higher latitudes around the winter solstice. Cassini data show a large polar cloud in the
325   winter hemisphere extending to mid-latitudes (Le Mouélic et al., 2018) that is clearly different
326   from our seasonal model, although our seasonal model at altitudes above 150 km has the largest
327   opacity during the winter, a qualitative agreement at least.
328         We detected small, short-lived deviations from the otherwise smooth, seasonal cycle, up to
329   about 10 % of the amplitude of the seasonal cycle. They may be comparable to weather
330   modifying the seasonal cycle on Earth.
331         The whole set of STIS observations provides some details of both seasonal variations, in
332   particular of the haze opacity at altitudes of <80 km and >150 km, the amplitudes, the phase shift
333   between both variations of 2.3 years, and the seasonal averages as function of latitude. They all
334   give constraints that can improve global circulation models of Titan (Rannou et al., 2004;
335   Crespin et al., 2008; Lora et al., 2015, 2019).
336

337                                        Acknowledgements
338         This work was supported by the STScI through General Observer program GO 14612.
339   Support for HST was provided by NASA through the Space Telescope Science Institute, which is
340   operated by the Association of Universities for Research in Astronomy, Incorporated, under
341   NASA contract NAS5-26555.
```

<!-- PDF_PAGE: 17 -->

## PDF page 17

```text
                                                      16


342                                              References
343          Ádámkovics, M., de Pater, I., Hartung, M., Eisenhauer, F., Genzel, R. 2006. Titan’s bright
344   spots: multiband spectroscopic measurement of surface diversity and hazes. J. Geophys. Res.
345   111, E07S06.
346          Brown, R.H., and 25 colleagues 2006. Observations in the Saturn system during approach
347   and orbital insertion, with Cassini’s visual and infrared mapping spectrometer (VIMS). Astron.
348   Astrophys. 446, 707-716.
349          Caldwell, J.D., Smith, P.H., Tomasko, M.G., McKay, C.P. 1992. Titan: evidence of
350   seasonal change—A comparison of Voyager and Hubble Space Telescope images. Icarus 103,
351   1-9.
352          Coustenis, A., Gendron, E., Lai, O., Véran, J.-P., Woillez, J., Combes, M., Vapillon, L.,
353   Fusco, T., Mugnier, L., Rannou, P. 2001. Images of Titan at 1.3 and 1.6 µm with adaptive optics
354   at the CFHT. Icarus 154, 501-515.
355          Crespin, A., Lebonnois, S., Vinatier, S., Bézard, B., Coustenis, A., Teanby, N.A.,
356   Achterberg, R.K., Rannou, P., Hourdin, F. 2008. Diagnostics of Titan’s stratospheric dynamics
357   using Cassini/CIRS data and the two-dimensional IPSL circulation model. Icarus 197, 556-571.
358          de Kok, R., Irwin, P.G.J., Teanby, N.A., Vinatier, S., Tosi, F., Negrão, A., Osprey, S.,
359   Ardiani, A., Moriconi, M.L., Coradini, A. 2010. A tropical haze band in Titan’s stratosphere.
360   Icarus 207, 485-490.
361          Karkoschka, E., Lorenz, R.D. 1996. Latitudinal variation of aerosol sizes inferred from
362   Titan’s shadow. Icarus 125, 369-379.
363          Karkoschka, E. 2016. Seasonal variation of Titan’s haze at low and high altitudes. Icarus
364   270, 339-354.
365          Le Mouélic S., Rodriguez, S., Robidel, R., Rousseau, B., Seignovert, B., Sotin, C., Barnes,
366   J.W., Brown, R.H., Baines, K.H., Buratti, B.J., Clark, R.N., Nicholson, P.D., Rannou, P., Cornet,
367   T. 2018. Mapping polar atmospheric features on Titan with VIMS: from the dissipation of the
368   northern cloud to onset of a southern polar vortex. Icarus 311, 371-383.
369          Lockwood, G.W., Thompson, D.T. 2009. Seasonal photometric variability of Titan, 1972-
370   2006. Icarus 200, 616-626.
371          Lora, J.M., Lunine, J.I., Russell, J.L. 2015. GCM simulations of Titan’s middle and lower
372   atmosphere and comparison to observations. Icarus 250, 516-528.
```

<!-- PDF_PAGE: 18 -->

## PDF page 18

```text
                                                     17

373        Lora, J.M., Tokano, T., Vatant d‘Ollone, J., Lebonnois, S., Lorenz, R.D. 2019. A model
374   intercomparison of Titan’s climate and low-altitude environment. Icarus 333, 113-126.
375        Lorenz, R.D., Lemmon, M.T., Smith, P.H. 2006. Seasonal evolution of Titan’s dark polar
376   hood: midsummer disappearance observed by the Hubble Space telescope. Monthly Notices
377   Royal Astron. Soc. 369, 1683-1687.
378        Nichols-Fleming, F., Corlies, P., Hayes, A.G., Ádámkovics, M., Rojo, P., Rodriguez, S.,
379   Turtle, E.P., Lora, J.M., Soderblom, J.M. 2021. Tracking short-term variations in the haze
380   distribution of Titan’s atmosphere with SINFONI VLT. Planetary Science J. 2:180.
381        Penteado, P.F., Griffith, C.A., Tomasko, M.G., Engel, S., See, C., Doose, L., Baines, K.H.,
382   Brown, R.H., Buratti, B.J., Clark, R., Nicholson, P., Sotin, C. 2010. Latitudinal variations in
383   Titan’s methane and haze from Cassini VIMS observations. Icarus 206, 352-365.
384        Rannou, P., Hourdin, F., McKay, C.P., Luz, D. 2004. A coupled dynamics-microphysics
385   model of Titan’s atmosphere. Icarus 170, 443-462.
386        Roe, H.G., de Pater, I., Macintosh, B.A., Gibbard, S.G., Max, C.E., McKay, C.P.        2002.
387   Titan’s atmosphere in late southern spring observed with adaptive opticson the W. M. Keck II
388   10-meter telescope. Icarus 157, 254-258.
389        Smith, B.A., Soderblom, L.A., Beebe, R., Boyce, J., Briggs, G., Bunker, A., Collins, S.A.,
390   Hansen, C.J., Johnson, T.V., Mitchell, J.L., Terrile, R.J., Carr, M., Cook II, A.F., Cuzzi, J.,
391   Pollack, J.B., Danielson, G.E., Ingersoll, A., Davies, M.E., Hunt, G.E., Masursky, H.,
392   Shoemaker, E., Morrison, D., Owen, T., Sagan, C., Veverka, J., Strom, R., Suomi, V. 1981.
393   Encounter with Saturn: Voyager 1 imaging results. Science 212, 163-182.
394        Smith, B.A., Soderblom, L.A., Batson, B., Bridges, P., Inoe, J., Masursky, H., Shoemaker,
395   E., Beebe, R., Boyce, J., Briggs, G., Bunker, A., Collins, S.A., Hansen, C.J., Johnson, T.V.,
396   Mitchell, J.L., Terrile, R.J., Carr, M., Cook II, A.F., Cuzzi, J., Pollack, J.B., Danielson, G.E.,
397   Ingersoll, A., Davies, M.E., Hunt, G.E., Morrison, D., Owen, T., Sagan, C., Veverka, J., Strom,
398   R., Suomi, V. 1982. A new look at the Saturn system: The Voyager 2 images. Science 215,
399   504-537.
400        Teanby, N.A., Irwin, P.G.J., Nixon, C.A., de Kok, R., Vinatier, S., Coustenis, A., Sefton-
401   Nash, E., Calcutt, S.B., Flasar, F.M.   2012. Active upper-atmosphere chemistry and dynamics
402   from polar circulation reversal on Titan. Nature 491, 732-735.
403        Vinatier, S. Bézard, B., Lebonnois, S., Teanby, N.A., Achterberg, A.K., Gorius, N.,
```

<!-- PDF_PAGE: 19 -->

## PDF page 19

```text
                                                    18

404   Mamoutkine, A., Guandique, E., Jolly, A., Jennings, D.E. Flasar, F.M. 2015. Seasonal variation
405   in Titan’s middle atmosphere during the northern spring derived from Cassini/CIRS observations.
406   Icarus 250, 95-115.
```
