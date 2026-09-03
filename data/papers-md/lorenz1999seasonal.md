---
citation_key: "lorenz1999seasonal"
title: "Seasonal change on Titan observed with the Hubble Space Telescope WFPC-2"
source_pdf: "data/papers/lorenz1999seasonal.pdf"
source_pdf_sha256: "365f116eb5feba6b67685bdcc290bcac998cfae2c9080d058c86cd34a8701692"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
Icarus 142, 391–401 (1999)
Article ID icar.1999.6225, available online at http://www.idealibrary.com on




             Seasonal Change on Titan Observed with the Hubble Space
                              Telescope WFPC-2
                                             Ralph D. Lorenz, Mark T. Lemmon, and Peter H. Smith
                                     Lunar and Planetary Laboratory, University of Arizona, Tucson, Arizona 85721–0092

                                                                               and

                                                                     G. W. Lockwood
                                        Lowell Observatory, 1400 West Mars Hill Road, Flagstaff, Arizona 86001–4470

                                                       Received November 9, 1998; revised June 8, 1999

                                                                                        Because Titan was observed with good spatial resolution only
   Recent observations with the Wide-Field Planetary Camera                          close to equinox (by Voyagers 1 and 2 in 1980/1981, and HST/
(WFPC-2) on the Hubble Space Telescope (HST) show an unex-                           WFPC-2 in 1994/1995), there was only information on the ex-
pectedly rapid change in the atmospheric albedo contrast between                     trema of the NSA cycle. It was assumed that the variation was
the north and south hemispheres. In 1994 at blue wavelengths, the                    sinusoidal (Smith et al. 1981, 1982; S81). After the asymmetry
north was around 15% brighter than the south, and was expected to                    peaked again (in the opposite sense, with north 15% brighter at
fall to about 12% in 1997, but has dropped to only 6% brighter. At                   blue wavelengths) in 1994/1995, it was predicted to fall to zero
some other wavelengths, the contrast has reversed, which was not
                                                                                     in 2002 (Lorenz et al. 1997, hereafter referred to as L97).
expected until 2002. The interhemispheric contrast has a time de-
                                                                                        In this paper, we report on analyses of new images obtained in
pendence that varies with wavelength; contrast changes in blue lag
behind changes in violet and yellow/red. The rapid change and the                    November 1997, two years after equinox. Using a new, more ac-
phase variation with wavelength are consistent with ground-based                     curate data reduction, we compare these with images obtained in
photometry. A physical model of the transport of high-altitude dark                  1994 and 1995 with the same instrument and filters. We find that
haze by meridional winds is a better description of Titan’s behav-                   seasonal change on Titan appears to occur faster than was ex-
ior than the simple sinusoidal models used to date. Investigation                    pected. We introduce a simple physical model for these changes,
with a radiative transfer model indicates that haze number density                   invoking the latitudinal transport of haze particles as suggested
changes above 160-km altitude are compatible with the observed                       by Hutzell et al. (1996) and more recently elaborately modeled
hemispheric albedo difference, and require particles >0.1 µm in                      by Tokano et al. (1999), and then attempt to quantify the number
radius. °c 1999 Academic Press                                                       density, optical properties, and altitude of the haze responsible
   Key Words: Titan; atmospheres, dynamics; image processing;                        using detailed radiative transfer models.
photometry; radiative transfer.


                                                                                             2. OBSERVATIONS AND DATA REDUCTION
                          1. INTRODUCTION
                                                                                 2.1. Qualitative Data Analysis: Titan’s Appearance to HST
   Titan’s thick hazy atmosphere not only obscured the surface
from Voyager’s cameras (Smith et al. 1981), but was itself decep-                   Titan’s appearance at equinox is well known at visible wave-
tively bland in appearance, with the most notable feature being                  lengths from the Voyager data, which show a limb-darkened
a sharp albedo difference between the north and south hemi-                      disk, with a difference in albedo between hemispheres. This
spheres (Sromovsky et al. 1981, hereafter S81). This so-called                   difference in hemispheres is strongest in the blue and green
north–south asymmetry (NSA) has a peak magnitude when the                        and weaker at violet and red wavelengths. Additionally, at high
Sun crosses Titan’s equator every 14.5 years, or every 0.5 Titan                 northern latitudes, a dark polar collar or hood is seen. It has been
year. This occurred in 1980 (when Voyager 1 arrived, in north-                   hypothesized (Yung 1987) that this latter feature is due to the
ern spring) and the asymmetry was expected to fall to zero after                 condensation of photochemical products that accumulate in the
7 years (i.e., in 1987). Early Hubble Space Telescope (HST)                      cold winter period when these latitudes are in shadow.
observations showed that the asymmetry had indeed reversed                          With HST, Titan appears broadly the same close to equinox,
(Caldwell et al. 1992) between 1981 and 1990.                                    although HST’s spatial resolution of about 300 km is far poorer
                                                                               391

                                                                                                                                              0019-1035/99 $30.00
                                                                                                                            Copyright ° c 1999 by Academic Press
                                                                                                                   All rights of reproduction in any form reserved.
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
392                                                                LORENZ ET AL.




   FIG. 1. HST Images of Titan in 1994 (top) and 1997 (bottom). From left to right the filters are F336W, F439W, F547M, FQCH4N-B (619 nm), F673N, and
FQCH4N-D (889 nm). North is up, and all images are scaled to have the same maximum brightness. It can be seen that the strong asymmetry at 439 and 547 nm
has virtually disappeared. The north at 619 nm has darkened somewhat, and the brightness at 889 nm has moved northward.



than the Voyagers’. The wavelength coverage of WFPC-2 is                      tion is poor, the polar hood seems to be present. In the next few
wider, notably permitting the surface to be imaged at near-IR                 years, our viewing geometry will continue to improve with the
wavelengths (Smith et al. 1996). The filters we use in this study             southward migration of the sub-Earth point. At the same time,
are F336W, F439M, F547W, F588N, and F673N continuum fil-                      however, the albedo contrast of the hood will decay as these
ters, with N, M, and W denoting narrow, medium, and wide                      regions accumulate more sunlight.
filters, and the narrow methane band filters FQCH4N-B and
FQCH4N-D at 619 and 889 nm (Burrows 1994).                                    2.2. Quantitative Image Analysis
    Titan as seen by HST in 1994 and 1997 is shown in Fig. 1. Con-
                                                                                 The images underwent standard STScI pipeline processing.
sidering the 1994 images first, limb darkening increases from
                                                                              The regions of the images containing Titan’s disk were ex-
violet through red wavelengths; this has the effect of making the
                                                                              tracted and each disk was fit by a model Titan, comprising two
disk look somewhat smaller at longer wavelengths, as seen in the
                                                                              model “hemispheres”: the brightness for each pixel is given by
figure (each 300-km pixel is about one-tenth of a Titan radius).
                                                                              I0 µk µk−1
                                                                                     0 , with µ and µ0 the cosine of the zenith angle and
There is also an optical radius effect [see Toon et al. (1992)
                                                                              the solar zenith angle, respectively. I0 is the normal reflectivity,
and discussion later], but this is not significant at this scale. The
                                                                              and k the Minnaert coefficient. The principal parameter we shall
889-nm methane band image has a rather different appearance
                                                                              be discussing in this paper is the ratio in I0 values for the two
(see Smith et al. 1996 and L97 for discussion): here there is limb
                                                                              hemispheres. Where we use the word contrast, this implies the
brightening since the deep atmosphere is black due to methane
                                                                              quantity I0 (bright)/I0 (dim) − 1.
absorption (with an absorption optical depth of about 100),
                                                                                 The resultant model image was convolved with a synthetic
and the thin haze above the bulk of the atmosphere is bright.
                                                                              point spread function (PSF) and the convolved image compared
Since the haze at these levels is optically thin, the longer path-
                                                                              with the observed one. The model and comparison are made
length through it leads to a bright limb.
                                                                              with 3x oversampled images since the model brightness changes
    The difference in albedo between the two hemispheres (the
                                                                              rapidly at the edge of the disk. The parameters are adjusted until
NSA) is apparent, with the difference strongest in green and blue,
                                                                              the root mean squared (rms) difference between the model and
as in the Voyager data, two seasons before. The NSA is reversed
                                                                              observed images is minimized.
in the methane band, due to bright haze being more abundant or
                                                                                 This analysis procedure improves on that in L97 [in turn de-
reflective in the south. In 1997, the picture is broadly the same,
                                                                              rived from Sromovsky et al. (1981)] in several respects.
except that the NSA appears to have reduced somewhat. The
“smile” of the methane band image also seems to have thinned                     1. The parameter fits were generated automatically, using the
and widened (the brightness is creeping northward along the                   downhill simplex method (implemented by the Interactive Data
limb).                                                                        Language procedure AMOEBA), whereas in L97 the “optimiza-
    In 1997 (and, to a lesser extent, in 1995 images not shown),              tion” was performed by hand, varying one parameter at a time.
there is some faint indication of a “flat bottom” of Titan at 336             The new procedure provides significantly better fits, since the
and 439 nm (see bottom left of Fig. 1). This is probably the dark             simplex method finds local minima in the cost function (here the
“polar hood” observed in the northern hemisphere by Voyagers                  mean square error of the fit) by systematically varying all pa-
1 and 2 (Smith et al. 1981, 1982). The hood in the north was                  rameters, whereas the manual procedure found them in a more
observed to have disappeared by 1990 (Caldwell et al. 1992), as               heuristic fashion. Further, patience limitations in the manual fit
would be expected by a seasonal model. As we are now mov-                     limited the parameter variations to increments of one part in 1000
ing into southern spring, the south polar regions that have been              or higher: the automated procedure varies the stepsize down to
shadowed are coming into view, and although the HST resolu-                   one part in 105 .
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
                                                     SEASONAL CHANGE ON TITAN                                                        393

   2. While L97 used midpoints between maximum-slope edges            tion of the real Titan so some systematic (rather than statistical)
of the images to determine the centers, the new procedure iter-       uncertainty would remain.
atively finds the best center, taking into account in the process        Our results are shown in Fig. 2. Where we have more than
any phase angle effects.                                              one image in a given filter in a given year, the values generated
   3. The µ and µ0 values for each hemisphere must be com-            by this method are always within 0.01 of each other, suggesting
puted on the basis of a radius. L97 used fixed values for the         our method is robust. This small scatter is consistent with our
radius, specifically those of Toon et al. (1992), whereas the ra-     uncertainty estimates (which describe the overall absolute un-
dius was a free parameter in the new determinations. Further, the     certainty of our estimated NSA; the relative uncertainty may in
north and south hemispheres had separate radii in the new fits.       fact be somewhat better than our error bars show).
   4. L97 reported only one determination of north/south albedo
ratio per image, with an estimated uncertainty. The new method,       2.4. Discussion and Supporting Data
with many measurements, allows a better visibility of the con-           The results and error bars from this new data reduction over-
fidence of each NSA determination and a more reliable deter-          lap with the earlier analyses of 1994 and 1995 data in L97,
mination of the errors: this aspect of the analysis is discussed in   although they are generally fractionally smaller (e.g., the 1994
the following section.                                                NSA = 1.15 at 547 nm, compared with 1.18 in L97). This con-
                                                                      firms the suspicion offered in L97 that the NSA in 1994/1995
2.3. Results
                                                                      (northern spring equinox) was rather weaker than the NSA at the
   We have devoted particular attention in this work to under-        corresponding Voyager epoch (1980/1981), where the bright/
standing the uncertainty of our measurements of the NSA, since        dark hemisphere ratio was 1.25. This difference is remarkable,
some of the changes we observe are subtle and surprising. We          but not altogether surprising. Titan’s substantial orbital eccen-
found that the RMS error in the fits was typically 3%, fraction-      tricity modulates the insolation: Tokano et al. (1999) have re-
ally better than achieved by hand in L97. The best fits (∼1.5%        cently determined that this eccentricity effect causes the peak
RMS error) were obtained in the F588N filters, with the noisier       stratospheric summer temperature over the south pole to be 10 K
F336W image reaching about 4%. The 889-nm images are not              warmer than the peak summer temperature in the north. This
as well described by the Minnaert model, but nevertheless errors      temperature difference may affect condensation processes on
were around 6%.                                                       the haze as well as the circulation that blows the haze around. It
   Repeated fits with different starting points yielded significant   may be that Titan’s haze follows an asymmetric seasonal cycle,
scatter in results. A Monte-Carlo approach was therefore tried,       in much the same way as water does on Mars with the con-
taking 20 optimized fits and taking either a mean or the fit with     sequence that the martian polar caps are different in size and
the lowest error, but this failed to produce significantly better     composition from each other.
results, the effectiveness being judged by comparing two nomi-           Sinusoidal fits to the NSA-versus-time data for the 1980–
nally identical HST images (e.g., our two exposures in 1997 us-       1995 period (L97) predicted that the blue contrast should be
ing the F439W filter; these should differ only by Poisson noise).     about 12% in 1997. However, we see (Fig. 2) that the asymmetry
For quantitative comparisons between the years, we therefore          has dropped sharply, to only 6%, and may be expected to fall
use the following method. A set (here, six) of converged fits         to zero within the next 2 years. Evidently a smooth sinusoidal
was obtained, each fit with the ratio of I0 values for the two        variation passing through the extrema of the cycle observed in
hemispheres (i.e., the NSA) held to a single fixed value. This        1980/1981 and 1994/1995 is an inappropriate description of the
constrained NSA value was varied across the expected “true”           NSA behavior.
value, typically in steps of 0.005 across a range of 0.1, mak-           The asymmetry’s complex dependence on wavelength is well
ing some 120 fits per image. A plot of the errors against the         known (Tomasko and Smith, 1982; S81; L97; Caldwell et al.
NSA shows a clear minimum, and the location of the minimum            1992) . Given the previous assumption of a sinusoidal variation
is stable between nominally identical images. The NSA values          (with time), the expectation was that the asymmetry-versus-
we report in the present paper correspond to the minimum of           wavelength curve would flip about the NSA = 1 line, like the
a quadratic fit to the error. We found in our initial analysis that   projection of a skip rope. We now observe (Fig. 2), however,
without several trial fits for each NSA value, the error envelope     that the asymmetry does not vary in phase at all wavelengths.
would not be well defined.                                               Particularly notable are the complete decay of the asymmetry
   The uncertainties we report in the NSA values correspond to        at 336 nm and its reversal at 673 nm; an in-phase sinusoidal vari-
the width of the minimum, defined by one-third of the width at        ation of the NSA would not predict these changes until 2002.
a threshold set at the minimum value of the fitted quadratic plus     The implication is that changes at violet and red wavelengths
three times the standard deviation of the quadratic fit. Performing   lead those in blue by 1–2 years. The difference in behavior be-
more image fits could presumably define the error curve rather        tween 619 and 673 nm is particularly interesting, presumably
better and hence slightly reduce the uncertainty. However, this       due to the methane absorption at 619 nm.
would be expensive computationally, and is probably unjustified          The fact that the changes in NSA in the 619- and 889-nm
given that the two-hemisphere model is not a complete descrip-        images are not in phase is also interesting. The large decrease
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
394                                                                      LORENZ ET AL.




   FIG. 2. Variation of the north/south albedo ratio with wavelength and time. Triangles connected with a dashed line denote 1994 observations, diamonds with
dotted line 1995, and circles with a solid line 1997. Pentagons are methane band images from 1996. Uncertainties are indicated for several data points (omitted
where the points are crowded.) In the methane bands at 619 and 889 nm, the asymmetry is reversed in sense. While it has decreased in magnitude at 889 and
336–588 nm, the asymmetry has intensified somewhat at 619 nm and reversed at 673 nm. Symbols are displaced ±10 nm to minimize crowding.




    FIG. 3. Disk-integrated blue (diamonds with dotted line) and yellow (crosses with dashed line) photometry obtained at Lowell observatory; dates are shown
at the top. The advanced phase of the yellow data is apparent: both the peaks and the center crossings occur about 0.5 year before the corresponding events in
blue. Note that the solar longitude is the astronomically conventional solar longitude of Saturn; Tokano et al. (1999) use the climatological convention of L s , the
saturnocentric longitude of the Sun.
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
                                                      SEASONAL CHANGE ON TITAN                                                                  395

in NSA at 889 nm (the filter probing the higher altitude) sug-
gests that the changes on Titan are occurring primarily at high
altitudes.
   Although little difference was observed between blue images
taken by Voyager 1 encounter and by Voyager 2, 9 months later
(Smith et al. 1982), there are indications (Table I of L97) that
the contrast dropped by 4% at violet and green wavelengths
over that interval, consistent with the trends we see in our HST
data. The reduced errors in our present analysis of our 1994 and
1995 HST images also allow the detection of significant change
between those years (Fig. 2).
   A consistent picture emerges, then, of the blue lagging be-
hind both longer and shorter wavelengths. There is some sup-
porting evidence from ground-based photometry for this pre-
viously unsuspected wavelength-dependent phase of the
NSA variation. Titan exhibits a change in disk-integrated albedo
over time, which has been monitored since 1971 by Lockwood
and Thompson (1979; Lockwood et al. 1986a,b). The albedo
varies approximately as a sinusoid, with an amplitude of about
4% and a period of half a Titanian year. As shown in Fig. 3,
the yellow cycle appears slightly phase-advanced with respect
to the blue. Quantitatively, a best-fit albedo of the form
a0 + a1 sin(2π (t − t0 )/14.75), with t0 indicating the phase, yields
amplitudes (a1 ) of 0.028 for yellow and 0.040 for blue, with t0
values of 1987.40 and 1988.0, each with rms errors of 0.01.
Thus the yellow is therefore around 0.6 year ahead of the blue,
consistent with our imaging observation of the NSA above.
   We do not understand why the yellow/red contrast changes
should lead those in the blue. The apparent lead of violet over            FIG. 4. Minnaert coefficient as a function of wavelength. The increasing
blue is easier to understand, in that these short wavelengths sam-      limb darkening from 336 to 588 nm and trend toward limb brightening at 889 nm
                                                                        are as noted in L97, although note the strong dip at 619 nm. Some interannual
ple only the uppermost atmosphere (see below). Tokano et al.            variability is apparent at 673 nm.
(1999) notes that haze transport by seasonally driven winds leads
to a change in haze number density at higher altitudes first; thus      certainly due to methane absorption at 619 nm causing the disk
the NSA at violet wavelengths should be quickest to respond, as         center to be less bright relative to the limb than at the adjacent
observed.                                                               wavelengths. There is also evidence of a temporal change in the
                                                                        limb-darkening coefficient: above 550 nm between 1994/1995
                                                                        and 1997 it drops by about 0.05.
2.5. Additional Imaging Results
                                                                           A separate data reduction allowing both radius and k to vary
   Our fitting procedure yields two additional parameters: a            for each hemisphere indicated a larger radius for the northern
model radius and a limb-darkening coefficient. The Minnaert             hemisphere above 550 nm and a higher Minnaert coefficient.
coefficient k shown in Fig. 4 shows a pattern consistent with           However, the scatter in these data was large (since a large radius–
L97, although absolute values are 0.05–0.1 smaller. This offset         large k and a small radius–small k may give equally good fits)
is in part due to the different data reduction techniques [L97          so we present results for fits using k constrained to be the same
used raw images, and corrected k by an empirical 0.06 to allow          for both hemispheres.
for PSF effects, while the present analysis treats the PSF explic-         The values for radius are shown in Fig. 5 and compared
itly]. An additional reason for the difference is that in L97 the       with previous measurements [although note that not all radii are
radius was forced to be equal to the Toon et al. (1992) values,         equal; we show here the best-fit Minnaert model radius, while
while in the present analysis radius is a free parameter. The re-       the Karkoschka and Lorenz (1997) results are for a grazing op-
sults here were obtained by a procedure similar to those for the        tical depth of 0.028; the analytic expression from Toon et al.
NSA—holding both hemispheres to the same variable k value               (1992) is a fit to a variety of data]. This caveat aside, the general
and selecting the value that yielded the minimum error.                 agreement is good, although the data fall below the line at 619
   A new result in Fig. 4 is that the limb-darkening coefficient is     and 673 nm: Titan’s upper atmosphere seems more transparent
smaller at 619 nm than at either 588 or 673 nm. This is almost          at these wavelengths than models indicate. The Minnaert fit is
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
396                                                                        LORENZ ET AL.


poor for the limb-brightened southern hemisphere at 889 nm;
we do not report those radius values.
   Figure 6 shows HST photometry of Titan. We measured
Titan’s total brightness (we summed pixels within a 1.6-arc-
sec-diameter circle, and corrected for Earth–Titan, Sun–Titan,
and phase angle effects, following Lockwood et al. (1986a). It is
seen that Titan has dimmed in the blue, while it has brightened
sharply at 889 nm. The change in albedo over the 3-year period
1994–1997 is broadly the same as for the 1993–1995 period, as
measured by Karkoschka (1998) using ground-based CCD spec-
troscopy. Both those data and our data indicate a slightly larger
change than do Lockwood’s photometry data, although the ∼1%
difference is probably accountable to experimental error.
   We believe the jump in brightness in our data at 889 nm is real.
This 7% increase in brightness suggests that future monitoring of
Titan’s reflectivity at this wavelength would be well worthwhile.
The brightening implies that there is more haze at high altitudes                        FIG. 6. Variation with time of Titan’s brightness. Solid symbols are relative
(above the bulk of the methane) in 1997 than in 1994. It should be                    albedo of Titan in our images, corrected for phase angle. Between 1994 and 1997
noted that the changes in the haze need to be taken into account                      Titan has become about 5% fainter in blue, while the red albedo is nearly con-
                                                                                      stant; this same behavior was observed between 1993 and 1995 by Karkoschka
                                                                                      (1998). The dramatic (∼7%) brightening at 889 nm is significantly higher than
                                                                                      for the 1993–1995 period, and suggests more haze at higher altitudes.



                                                                                      in interpreting spectroscopy in terms of surface reflectivity if
                                                                                      data spanning several years are to be analyzed, e.g., Coustenis
                                                                                      et al. (1995).

                                                                                                     3. MODELS OF SEASONAL CHANGE

                                                                                      3.1. Previous Analytic Models
                                                                                         A sinusoidally varying NSA combines with the varying con-
                                                                                      tribution of north and south hemispheres due to Titan’s obliq-
                                                                                      uity (again, roughly sinusoidal, with the same 29.5-year pe-
                                                                                      riod, but 90◦ out of phase) to give a double-frequency signal
                                                                                      as observed; see Smith et al. (1981) and especially S81. As
                                                                                      noted by Lockwood et al. (1986b) and Sromovsky et al. (1986),
                                                                                      the observed NSA was insufficient to produce the observed
                                                                                      albedo variation. The discrepancy appeared, over the 1972–
                                                                                      1986 period, to be correlated with the solar cycle. L97 exam-
                                                                                      ined these and other models in the light of HST observations in
                                                                                      the 1990–1995 period. They found that the combined NSA and
                                                                                      albedo data were best matched by a combined sinusoidal NSA
                                                                                      and sinusoidal albedo increment to both hemispheres with a
                                                                                      14.5-year period. They suggested that a change in optical radius
                                                                                      or Minnaert coefficient, corresponding to a change in haze den-
                                                                                      sity at high altitudes, might instead account for the discrepancy.
                                                                                      The solar cycle mechanism, consistent with the data until 1986,
                                                                                      falls out of phase with the required effect in the 1986–1995
   FIG. 5. Model radius as measured from the HST images analyzed in this              period.
paper, compared with previous determinations of optical radius. Dashed line
shows the analytic expression from Toon et al. (1992), while solid lines show
                                                                                         A second difficulty with the simple sinusoidal model is that the
the altitude at which various (vertical) optical depths are attained in the nominal   real cycle seems slightly asymmetric; a fit to the 1979–1981 data
Lemmon (1994) model; note the methane spikes around 889 nm.                           yields too high an amplitude for the 1990–1995 period. This may
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
                                                    SEASONAL CHANGE ON TITAN                                                       397

be corrected, perhaps artificially, if the southern hemisphere has   for the behavior at blue wavelengths, let us consider a black
a higher underlying brightness than the northern one, although       haze layer overlying a deep, brighter atmosphere, and assume
this difference must still be due to atmospheric properties, since   that the deep atmosphere does not change. Further, we consider
no blue light reaches the surface.                                   a meridional circulation, with air above this deep layer rising
    A third difficulty with the sinusoidal model is one introduced   over the subsolar point and descending at the poles. This has the
by the new data in this paper. If the phase of the NSA is tuned to   gross effect of transporting haze from the summer hemisphere
fit the extrema available to L97, the NSA falls more slowly than     to the winter one.
our observations indicate. A fourth, more philosophical, aspect          As haze is transported from the summer hemisphere to the
to these models is that while they are algebraically convenient,     winter one, the short-wavelength albedo of the winter hemi-
and useful in introducing how the effect of a seasonal asymme-       sphere drops rapidly; even a small addition of obscuring haze
try and Titan’s varying aspect combine to produce a 14.5-year        adds to the absorption optical depth and reduces the brightness.
albedo variation, they shed little light on what is actually going   The short-wavelength albedo of the spring hemisphere stays low,
on on Titan. We therefore turn our attention to more physically      however; the albedo increment in the summer hemisphere due
based models.                                                        to a quantity of haze fleeing the subsolar point is less than the
                                                                     albedo decrement it causes in the winter hemisphere. This is a
                                                                     natural consequence of the nonlinear nature of opacity. As an
3.2. Mechanisms and Physical Models
                                                                     example, if a parcel of obscuring haze with an optical depth
   All proposed mechanisms for producing the asymmetry and           δτ ∼ 0.1 moves from a dark hemisphere with a large extinc-
the albedo change invoke Titan’s haze, since the light at most       tion optical depth (1.0, for example) to one with a low optical
of the relevant wavelengths reaches unity optical depth at alti-     depth (0.1), the brightness of the darker hemisphere increases
tudes far above the surface. Photochemical production variations     by e−0.9 − e−1.0 , or only 0.038, while the brighter hemisphere is
with season (Hutzell et al. 1993) are ruled out due to the long      darkened by e−0.1 − e−0.2 , or 0.086.
residence time of haze in the atmosphere; the time for haze to           The model computes the total brightness by adding the time-
coalesce and fall to regions where it accumulates a significant      dependent contributions from two hemispheres as in L97. Each
optical depth is far longer than the seasonal time scale, so sea-    one has an albedo of the form A x exp(−τx ), with A an underlying
sonal variations in production are smoothed out. Variation due       albedo and τ representing the amount of dark haze lying above
to a difference in particle size between hemispheres is incon-       it, with subscript x denoting north or south.
sistent with the invariance of the asymmetry with observation            An insolation–haze product for each hemisphere is crudely
phase angle (Sromovsky and Fry 1989).                                computed as Px = Sτx cos(3 −φx ), where S is the relative inso-
   The most likely mechanism involves meridional circulation,        lation (which varies by some 20% over Saturn’s elliptical orbit),
either directly by transporting haze particles from one hemi-        τx is the amount of haze in the hemisphere, 3 is the subsolar
sphere to another (e.g., Hutzell et al. 1996) or by modifying        latitude, and φx is an arbitrary angle to maximize and mini-
the particle number density or optical properties by influencing     mize the Px function at the appropriate solstices; here I use
condensation (Courtin et al. 1991) and/or removal by rainout         φN = 30, φS = −30. This quantity Px represents something like
(L97). All these latter effects may play a role, but the changes     a “haze potential”; the potential in a given hemisphere is higher
we have observed are at least conceptually consistent with the       when there is more haze there, and when the Sun is furthest
haze transport mechanism, which has already been indicated           from the equator into that hemisphere. The amount of haze that
by measurements of Titan’s shadow: Karkoschka and Lorenz             moves from one hemisphere to the other is a multiple k1 of the
(1997) determined that the haze north of 5◦ S was dominated          difference between these potentials for the two hemispheres.
by 0.3-µm particles in 1995, while south of that latitude, the           Additionally, since haze being lofted by the Hadley cell dark-
haze layer was of 0.1-µm particles, at an altitude up to 100 km      ens more effectively than haze lower down (since in reality the
lower. This is consistent with upwelling air motions in the north-   dark haze and bright atmosphere are vertically mixed), both
ern (summer) hemisphere, the motion expected in a thermally          hemispheres are further darkened by an amount e−k2q , where q
direct Hadley-type circulation.                                      is the sum of the haze potentials and k2 is a constant.
                                                                         Results of the model are shown in Fig. 7. The model as de-
                                                                     scribed above (shown as the dashed curve) fits the 1994–1997
3.3. A Simple Conceptual Model
                                                                     NSA data very well, although it does not quite match the 1979–
   The haze, which may be crudely represented by tholins gen-        1981 data. The fit to the total brightness is no worse than for
erated in the laboratory (Khare et al. 1984), is dark relative       a sinusoidal model, but can be improved substantially by shift-
to the bright, Rayleigh-scattering atmosphere below at short         ing the curve to the right by 2.5 years. Both models have ini-
(<500-nm) wavelengths. For example, McKay et al. (1989) find         tial (1952) values τN = 0.9, τS = 0.12, k1 = 0.005, k2 = 30, and
that the single-scattering albedo of haze particles varies from      AS /AN = 1.03, and the time step is 0.05 year. It is emphasized
only 0.45 at 300 nm to 0.95 at 600 nm. Thus, as a crude model        that the τN and τS values refer to the haze above some “optical
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
398                                                                    LORENZ ET AL.




                                   a




                                   b




   FIG. 7. (a) Disk-averaged reflectivity (i.e., weighted sum of the two hemispheric albedos), with the photometric observations indicated by diamonds with 1σ
error bars. (b) Ratio of the two model hemispheres, together with Pioneer, Voyager, and HST measurements from L97 and the present paper (error bars are of the
order of the symbol size and so are not shown). The dashed curve corresponds to the (preferred) baseline model, while the solid curve (which has a better fit with
the brightness data, but somewhat poorer fit for the 1994–1997 HST data) is phase-delayed by 2.5 years.


surface” in the atmosphere, not the optical depth of the atmo-                    reproduce the Pioneer 11 polarimetry data (Tomasko and Smith
sphere as a whole.                                                                1982) and the ground-based geometric albedo data of Neff et al.
   The rms errors for the “raw” model are 0.024 and 0.039 for                     (1984). Although this approach is somewhat imperfect—ideally,
the brightness and albedo ratios, respectively, and 0.013 and                     if there were enough data present to constrain them, two hemi-
0.038 for the “phase-shifted” model. These may be compared                        sphere models would be made and added together to fit the
with corresponding errors of 0.015 and 0.045 for the sinusoidal                   observed spectrum—it is adequate for the present purpose. A
model mentioned in the earlier section. It may be noted that                      further imperfection, also intractable with the data on hand, is
phase-shifting the model to fit the brightness curve degrades                     that a variety of altitudes, some perhaps increasing their haze
the 1994–1997 NSA fit, and vice versa. The rapid change in                        density of one size range of haze particles while simultaneously
brightness in the early 1970s (around perihelion) is difficult for                decreasing density for another size, may be responsible for the
the models to capture, even though the model takes Saturn’s                       albedo change, while to keep the number of free parameters
orbital eccentricity into account (so that subsolar latitude varies               tractable we consider only a single additional layer.
rapidly and insolation is higher at perihelion).                                     In general, a larger layer optical depth and a higher altitude of
   This simplistic dark-haze model is clearly inappropriate for                   insertion increase the effect of the particles. The effect depends
longer wavelengths where the haze is scattering rather than ab-                   on the single-scattering albedo ω. The threshold ω above which
sorbing and light penetrates deeper into the atmosphere. Fully                    the extra haze brightens Titan rather than darkens it depends on
coupled haze/circulation models may describe the haze trans-                      the wavelength. Since all the haze is bright at long wavelengths, a
port rather better than the crude formulation above. However,                     very high ω is needed to have a brightening effect. These effects
despite its simplicity, this model has a physical basis and does a                are shown in Fig. 8.
fair job of reproducing Titan’s behavior.                                            To fit our simulated “dark hemisphere” (i.e., the baseline
                                                                                  model spectrum, divided by the 1994 NSA) we force the single-
                                                                                  scattering albedo of the particles to have values appropriate to
3.4. Radiative Transfer Modeling
                                                                                  tholins, taken from McKay et al. (1989); the range ω is allowed
   We now attempt to elucidate the altitudes responsible for                      to take is given in Table I.
the NSA using detailed radiative transfer models.An additional                       We find that the 673-nm filter is not a particularly useful
layer of particles with a specified optical depth, single-scattering              constraint, at least for the single-scattering albedo values used,
albedo is inserted into the “baseline” Titan haze microphysics                    since viable solutions are present for a large altitude and op-
and radiation transfer model (Lemmon 1994) at a specified alti-                   tical depth range (largely because the NSA is so small at this
tude. The baseline model assumes fractal particles and is tuned to                wavelength). The 336- and 439-nm data argue for a haze layer at
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
                                                                SEASONAL CHANGE ON TITAN                                                                        399

       a                                                                             b




      c                                                                              d




   FIG. 8. Radiative transfer model albedo (abscissa) of Titan, with an additional haze layer inserted at an altitude indicated by the ordinate. (a) 336 nm: The
effect of the haze is minimal below 100 km because the “baseline” haze optical depth is high even at that level (see Fig. 5). Above that altitude, bright particles
increase the albedo, with thicker layers increasing the albedo even more. Dark particles reduce the albedo. (b) 619 nm: Here the incremental layer increases the
albedo since the underlying atmosphere is dark, and has an effect down to much deeper levels in the atmosphere. The change in slope at about 80 km is due to the
rainout of haze at that altitude. There is a strong slope with altitude due to methane absorption. (c) 673 nm: This is somewhat similar to the 619-nm case, except
note that below 80 km the albedo change is relatively insensitive to the altitude of the new layer since methane absorption is weak at this wavelength. Note also
that the “background” haze is quite bright; adding haze with a single-scattering albedo of 0.9 can reduce the albedo at low altitudes. (d) 889 nm: Here the effect is
similar to that at 336 nm although this is due to the methane absorption optical depth in the atmosphere, rather than the haze.




                                                                                    high (>150-km) altitude, while the combination of the 588- and
                              TABLE I                                               619-nm data requires a layer of optical depth 0.05–0.1 if the same
             Single Scattering Albedos of the Additional                            altitude range is considered. The 889-nm data complement the
                             Haze Layer                                             other constraints quite well, forcing a narrow corridor of possi-
                                                                                    bilities.
                      Filter             Single-scattering albedo
                                                                                       The acceptable fits above 160 km are combined in Fig. 9.
             F336W                                0.4–0.5                           Overlain are relative Mie extinction efficiencies for 0.1- and
             F439W                               0.45–0.55                          0.3-µm-radius particles from Karkoschka and Lorenz (1996).
             F588N                               0.65–0.75                          It appears that the 619-nm data argue that the particles must be
             FQCH4N-B (619 nm)                    0.9–0.95                          somewhat larger than 0.1 µm. If particles are (or at least have the
             F673N                                0.9–0.97
                                                                                    optical properties of spheres of ) 0.3 µm or larger, then the layer
             FQCH4N-D (889 nm)                   0.94–0.99
                                                                                    must be quite thin, with an optical depth ∼0.1 at all wavelengths.
                Source. Reprinted, with permission from McKay                          Also shown in that figure is the change in total optical depth at
             et al. (1989).                                                         640 nm to be expected from meridonal and vertical circulation,
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
400                                                                      LORENZ ET AL.


                                                                                    in 1995 and winter solstice in 2002. Both our model and theirs
                                                                                    predict that the asymmetry will stay more-or-less fixed for the
                                                                                    2004–2008 duration of the Cassini Mission. While a disappoint-
                                                                                    ing prediction, perhaps, this agreement is at least encouraging
                                                                                    from a modeling standpoint.
                                                                                       Radiative transfer investigations indicate that the peak (1994)
                                                                                    asymmetry is compatible with an additional haze population
                                                                                    above 160 km in the southern hemisphere, with optical depths
                                                                                    of >0.1 at 439 nm and 0.04–0.1 at 889 nm, assuming haze prop-
                                                                                    erties equivalent to tholin-like particles >0.1 µm in diameter.
                                                                                    Other, more complex, haze variations may be responsible. Addi-
                                                                                    tionally, the fractal nature of the aerosol particles (e.g., Rannou
                                                                                    et al. 1995, Lemmon 1994) means this radius should be taken
                                                                                    only as a guide.
                                                                                       The north–south asymmetry appears to depend both in ampli-
                                                                                    tude and phase on wavelength. We find that changes in violet and
   FIG. 9. Assuming a single layer above 160 km is responsible for the albedo
change, the allowable optical depths are shown as a function of wavelength in
                                                                                    yellow/red lead changes in blue. The blue asymmetry may be
this figure by symbols. The small crosses indicate parameter sets that were tried   expected to fall to zero and reverse in the next 1–2 years, while
but that do not yield the required albedo. The solid line is the optical depth of   at violet and red wavelengths it has already done so. Titan’s
a layer of 0.1-µm radius spherical particles, with the thickness normalized to      behavior at red wavelengths is challenging: it is not clear why
the 889-nm value, while the dotted line corresponds to 0.3-µm radius particles.     changes at these wavelengths should lead changes in blue, nor
One-tenth-micron particles seem to be inconsistent with the data at 619 nm,
so particles larger than this must be responsible. The shaded box indicates the
                                                                                    is it obvious why the model radius of Titan should drop below
seasonal change in total optical depth throughout the atmosphere as modeled by      model estimates at these wavelengths in particular.
Tokano et al. (1999).                                                                  We find Titan darker at blue wavelengths by about 5%, while
                                                                                    its 889-nm brightness has increased by some 7% over the 3-year
                                                                                    period of this study. These changes underscore that Titan is a
as modeled by Tokano et al. (1999) using a coupled GCM/haze                         dynamic object, and it is no longer appropriate to state or use
model; the change from 0.15 to 0.35 at this wavelength seems                        Titan’s brightness in models as if it were an unchanging quantity;
compatible with our results, especially since their result is the                   seasonal changes must be considered.
vertical integral of haze changes throughout the atmosphere, not                       One useful future investigation will be to compare the ob-
just above 160 km as for our result.                                                served contrasts of specific surface features as a function of lat-
   The upward vertical winds required to levitate and transport                     itude and time with model predictions (i.e., sampling the trans-
0.1- to 1-µm particles at these altitudes are only 0.1–1 cm/s                       missivity of the haze, rather than its reflectivity). A better model
(Toon et al. 1992), quite compatible with dynamical predictions                     treatment of the haze will entail separate models for each hemi-
(Flasar et al. 1981, Hourdin et al. 1995, Tokano et al. 1999).                      sphere; spatially resolved spectroscopy (e.g., using the Space
A 1 cm/s meridional wind could transport material from 30◦ N                        Telescope Imaging Spectrometer) will facilitate this work. Full
to 30◦ S in a few years, compatible with the time scales of the                     three-dimensional models of the haze structure, probably using
changes we are observing.                                                           Monte-Carlo methods, will be indispensable tools for grappling
                                                                                    with the coupled complications of Titan’s atmosphere; radia-
                                                                                    tive effects are driven by the haze structure, which is controlled
                            4. CONCLUSIONS
                                                                                    by the temperature structure and dynamics, which are in turn
   We have analyzed a suite of images obtained between 1994                         controlled by radiation. The computational power required for
and 1997 and have detected significant change in Titan’s appear-                    running such models is no longer prohibitively expensive, so we
ance; the north–south asymmetry has varied more than previous                       expect substantial progress in this area in the coming years.
models predicted. The asymmetry over this period appears to be                         We also look forward to future observations of Titan with
somewhat weaker than in the 1980/1981 epoch.                                        HST and Cassini. Our experience suggests continuing surprises
   Until a full cycle of spatially resolved Titan observations has                  are in store.
been obtained, any prediction of Titan’s seasonal behavior must
be tentative. The changes at blue wavelengths are consistent with                                          ACKNOWLEDGMENTS
a simple physical model of upper atmospheric haze transport
                                                                                       This work is based in part on observations with the NASA/ESA Space Tele-
which we have described. The rapid changes we are observing
                                                                                    scope, and partial support was provided by NASA through Grant PI314159
now are also consistent with the more elaborate model results of                    from the STScI, which is operated by Association of Universities for Research
Tokano et al. (1999) who show rapid changes in mid- and high-                       in Astronomy, Inc., under NASA Contract NAS5-2655. G.W.L. acknowledges
latitude haze optical depth between northern autumn equinox                         the support of NASA Planetary Astronomy Programs NAG5-3948 (current) and
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
                                                              SEASONAL CHANGE ON TITAN                                                                           401

NAGW-4946 (prior). IDL is a trademark of Research Systems, Inc. We thank          Lorenz, R. D., P. H. Smith, M. T. Lemmon, E. Karkoschka, G. W. Lockwood,
the referees for useful comments, and Erich Karkoschka for making his data          and J. Caldwell 1997. Titan’s north–south asymmetry from HST and Voyager
available to us in electronic form.                                                 imaging: Comparisons with models and groundbased photometry. Icarus
                                                                                    127, 173–189.
                                                                                  Neff, J. S., D. C. Humm, J. T. Bergstrahl, A. L. Cochran, W. C. Cochran,
                             REFERENCES                                             E. S. Barker, and R. G. Tull 1984.a Absolute spectrophotometry of Titan,
                                                                                    Uranus and Neptune: 3500–10500 A. Icarus 60, 221–235.
Burrows, C. J. (Ed.) 1994. Hubble Space Telescope Wide Field and Plane-
                                                                                  Rages, K., and J. B. Pollack 1980. Titan aerosols: Optical properties and vertical
  tary Camera 2 Instrument Handbook. Space Telescope Science Institute,
  Baltimore.                                                                        distribution. Icarus 41, 119–130.
                                                                                  Rages, K., and J. B. Pollack 1983. Vertical distribution of scattering hazes in
Caldwell, J. D., P. H. Smith, M. G. Tomasko, and H. Weaver 1992. Titan: Ev-
                                                                                    Titan’s upper atmosphere. Icarus 55, 50–62.
  idence of seasonal change—A comparison of Voyager and Hubble Space
  Telescope images. Icarus 103, 1–9.                                              Rannou, P., M. Cabane, E. Chassefiere, R. Botet, C. P. McKay, and R. Courtin
                                                                                    1995. Titan’s geometric albedo: Role of the fractal structure of the aerosols.
Courtin, R., R. Wagener, C. P. McKay, J. Caldwell, K.-R. Fricke, F. Raulin, and
                                                                                    Icarus 118, 355–372.
  P. Bruston, 1991. UV spectroscopy of Titan’s atmosphere, planetary organic
  chemistry and prebiological synthesis II. Interpretation of new IUE observa-    Smith, B. A., L. A. Soderblom, R. Batson, P. Bridges, J. Inge, H. Masursky,
  tions in the 220–335 nm range. Icarus 90, 43–56.                                  E. Shoemaker, R. Beebe, J. Boyce, G. Briggs, A. Bunker, S. A. Collins,
Coustenis, A., E. Lellouch, J. P. Maillard, and C. P. McKay 1995. Titan’s           C. J. Hansen, T. V. Johnson, J. L. Mitchell, R. J. Terrile, M. Carr, A. F. Cook II,
                                                                                    J. Cuzzi, J. B. Pollack, G. E. Danielson, A. Ingersoll, M. E. Davies, G. E. Hunt,
  surface: Composition and variability from the near-infrared albedo. Icarus
                                                                                    D. Morrison, T. Owen, C. Sagan, J. Veverka, R. Strom, and V. Suomi 1982.
  118, 87–104.
                                                                                    A new look at the Saturn system: The Voyager 2 images. Science 215, 504–
Flasar, F. M., R. E. Samuelson, and B. J. Conrath 1981. Titan’s atmosphere:         537.
  Temperature and dynamics. Nature 292, 693–698.
                                                                                  Smith, B. A., L. A. Soderblom, R. Beebe, J. Boyce, G. Briggs, A. Bunker,
Hourdin, F., O. Talagrand, R. Sadourny, R. Courtin, D. Gautier, and C. P. McKay
                                                                                    S. A. Collins, C. J. Hansen, T. V. Johnson, J. L. Mitchell, R. J. Terrile, M. Carr,
  1995. Numerical simulation of the general circulation of the atmosphere of
                                                                                    A. F. Cook II, J. Cuzzi, J. B. Pollack, G. E. Danielson, A. Ingersoll,
  Titan. Icarus 117, 358–374.                                                       M. E. Davies, G. E. Hunt, H. Masursky, E. Shoemaker, D. Morrison, T. Owen,
Hubbard, W. B., and 45 colleagues 1993. The occultation of 28 Sgr by Titan.         C. Sagan, J. Veverka, R. Strom, and V. Suomi 1981. Encounter with Saturn:
  Astron. Astrophys. 269, 541–563.                                                  Voyager 1 imaging results. Science 212, 163–182.
Hutzell, W. T., C. P. McKay, and O. B. Toon 1993. Effects of time-varying haze    Smith, P. H. 1980. The radius of Titan from Pioneer Saturn data. J. Geophys.
  production on Titan’s geometric albedo. Icarus 105, 162–174.                      Res. 85, 5943–5947.
Hutzell, W. T., C. P. McKay, O. B. Toon, and F. Hourdin 1996. Simulations of      Smith, P. H., and M. T. Lemmon 1993. HST images of Titan. Bull. Am. Astron.
  Titan’s brightness by a two-dimensional haze model. Icarus 119, 112–129.          Soc. 25, 1105. [Abstract]
Karkoschka, E. 1995. Spectrophotometry of the jovian planets and Titan at         Smith, P. H., E. Karkoschka, and M. T. Lemmon 1992. Titan’s north–south
  300- to 1000-nm wavelength: The methane spectrum. Icarus 111, 174–192.            asymmetry from HST images. Bull. Am. Astron. Soc. 24, 950. [Abstract]
Karkoschka, E. 1998. Methane, ammonia, and temperature measurements of the        Smith, P. H., M. T. Lemmon, R. D. Lorenz, J. J. Caldwell, M. D. Allison, and
  jovian planets and Titan from CCD-spectrophotometry. Icarus 133, 134–146.         L. A. Sromovsky 1996. Titan’s surface, revealed by HST imaging. Icarus
Karkoschka, E., and R. D. Lorenz 1997. Latitudinal variation of aerosol sizes       119, 336–349.
  inferred from Titan’s shadow. Icarus 125, 369–379.                              Smith, P. H., R. D. Lorenz, and M. T. Lemmon 1995. New information
Khare, B. N., C. Sagan, E. T. Arakawa, F. Suits, T. A. Callcott, and M. W.          on Titan’s north–south contrast from HST. Bull. Am. Astron. Soc. 27,
  Williams 1984. Optical constants of organic tholins produced in a simulated       1104.
  titanian atmosphere: From soft X-ray to microwave frequencies. Icarus 60,       Sromovsky, L. A., V. E. Suomi, J. B. Pollack, R. J. Kraus, S. S. Limaye,
  127–137.                                                                          T. Owen, H. E. Revercomb, and C. Sagan 1981. Implications of Titan’s
Lemmon, M. T. 1994. Properties of Titan’s Haze and Surface. Ph.D. thesis,           north–south brightness asymmetry. Nature 292, 698–702.
  University of Arizona.                                                          Tokano, T., F. M. Neubauer, M. Laube, and C. P. McKay 1999. Seasonal
Lockwood, G. W., and D. T. Thompson 1979. A relationship between solar              variation of Titan’s atmospheric structure simulated by a general circulation
  activity and planetary albedos. Nature 280, 43–45.                                model. Planet. Space Sci. 47, 493–520.
Lockwood, G. W., B. L. Lutz, D. T. Thompson, and E. S. Bus 1986a. The             Tomasko, M. G., and P. H. Smith 1982. Photometry and polarimetry of Titan:
  albedo of Titan. Astrophys. J. 303, 511–530.                                      Pioneer 11 observations and their implications for aerosol properties. Icarus
Lockwood, G. W., D. T. Thompson, and L. A. Sromovsky 1986b. Photometry of           51, 65–92.
  Titan: Evidence supporting the seasonal contrast model of albedo variations.    Toon, O. B., C. P. McKay, C. A. Griffith, and R. P. Turco 1992. A physical
  Bull. Am. Astron. Soc. 18, 809.                                                   model of Titan’s aerosols. Icarus 95, 24–53.
```
