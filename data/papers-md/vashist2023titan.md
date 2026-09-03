---
citation_key: "vashist2023titan"
title: "Titan’s North--South Haze Asymmetry Ratio and Boundary at Visible Wavelengths over the Cassini Mission"
source_pdf: "data/papers/vashist2023titan.pdf"
source_pdf_sha256: "f4d43235ba0c63ae9422e3fd385f051d9980eb6942cdd02b81fb7ab7e5a6caa0"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                       https://doi.org/10.3847/PSJ/acdd05
© 2023. The Author(s). Published by the American Astronomical Society.




Titan’s North–South Haze Asymmetry Ratio and Boundary at Visible Wavelengths over
                              the Cassini Mission
              Aadvik S. Vashist1,2                , Michael F. Heslar1   , Jason W. Barnes1       , Corbin Hennen1, and Ralph D. Lorenz3
                                                        1
                                                    Department of Physics; University of Idaho; Moscow, ID 83844, USA
                                                          2
                                                            River Hill High School; Clarksville, MD 21029, USA
                                                  3
                                                Johns Hopkins University Applied Physics Laboratory; Laurel, MD 20723, USA
                                       Received 2023 February 10; revised 2023 June 3; accepted 2023 June 7; published 2023 June 30

                                                                               Abstract
             We document the evolution of the north–south asymmetry (NSA) of Titan’s haze albedo during the Cassini
             mission between 2004 and 2017. We analyze coadded cube images taken at 96 distinct wavelengths between 0.35
             and 1.05 μm by the Cassini Visual and Infrared Mapping Spectrometer (VIMS-V) instrument from 14 Titan ﬂybys.
             Over half of a Titan year, we observe a near-complete transition in the NSA boundary latitude across the
             geographic equator from the southern to the northern hemisphere, including a 3 yr fading of the boundary for
             several years after the equinox. The fading transition of the NSA matches previous observations of a reversal of the
             NSA in Hubble Space Telescope images of Titan before the winter solstice between 1997 and 2000. A comparison
             of NSA images taken at similar times but different phase angles shows the NSA boundary is detectable, albeit with
             less contrast, at moderately high phase angles (∼90°). Analysis of the NSA boundary in T61 and T67 VIMS
             images further supports a small tilt between the superrotating atmosphere and the solid body of Titan, as suggested
             in a previous analysis of 0.890 μm images from the Cassini Imaging Science Subsystem.
             Uniﬁed Astronomy Thesaurus concepts: Astronomy data modeling (1859); Astronomy image processing (2306);
             Titan (2186); Atmospheric evolution (2301); Atmospheric dynamics (2300); Albedo (2321); Computational
             astronomy (293)


                                    1. Introduction                                      and wavelength coverage of the NSA, leading to incomplete
                                                                                         records on haze circulation with large errors (Lorenz et al.
   Saturn’s moon Titan exhibits many properties not found in
                                                                                         2001, 2004). In addition, previous studies often use special
other satellites. As ﬁrst observed by Voyager 1 (Smith et al.
                                                                                         case methodologies, where results are tied to their data sets to
1981), Titan’s ubiquitous atmospheric haze prevents optical
                                                                                         calculate and subsequently compare those previous NSA
imaging of the surface. The haze distribution varies as a
                                                                                         boundary latitudes (Roman et al. 2009).
function of latitude (Sromovsky et al. 1981), altitude (Smith
                                                                                            More recent studies have the temporal coverage to study
et al. 1982; Tomasko et al. 2005), and time (Lorenz et al. 1997;
                                                                                         detailed aspects for a substantial portion of the NSA cycle with
West et al. 2011). Titan’s haze also shows albedo differences
                                                                                         individual data sets. Karkoschka (2022) modeled NSA reversal
between its northern and southern hemispheres that shift near
                                                                                         at different altitudes with the Hubble Space Telescope (HST)
the equator. As the seasons progress, atmospheric circulation
                                                                                         Space Telescope Imaging Spectrograph image cubes. Kutsop
changes the global haze distribution, culminating in a reversal
                                                                                         et al. (2022) completed an analysis of circumglobal haze bands
every 15 yr (Brown et al. 2009). The reversal presents as an
                                                                                         in a variety of Cassini imagery data sets.
albedo dichotomy in the otherwise featureless atmosphere. The
                                                                                            In this paper, we document on the seasonal changes in
existence of the asymmetry also results in a distinct boundary
                                                                                         Titan’s lower atmospheric haze near the equator using Cassini
line that separates the northern and southern hemispheres.
                                                                                         observations of visible wavelengths for the purpose of
   The Voyager 1 ﬂyby highlighted the existence of a north–
                                                                                         comparison with previous studies. The observations of seasonal
south asymmetry (NSA) between the two hemispheres (Smith
                                                                                         haze changes through visible wavelengths allow us to extend
et al. 1981), which we show in Figure 1. Previous discoveries
also found a tilt of the boundary line relative to the solid-body                        the temporal baseline of previous observations and to track one
equator of Titan (Roman et al. 2009). The boundary, as                                   seasonal cycle coherently with a single uniform data set. The
reported by Sromovsky et al. (1981) with Voyager 1 data, was                             Cassini Visual and Infrared Mapping Spectrometer instrument
located at roughly 5.5°S. More recent discoveries show a                                 (VIMS) collected spectral maps at 96 visible and near-infrared
seasonal reversal in the latitude of the asymmetry across the                            wavelengths between 0.35 and 1.05 μm, which predominantly
equator.                                                                                 sampled the stratosphere (∼70–120 km).
   The movement of the NSA boundary reveals essential details                               Section 2 describes how we modify the main NSA image
of the global atmospheric properties and circulation patterns of                         analysis algorithm from Roman et al. (2009) with considera-
Titan (Hirtzig et al. 2006). Titan disk observations from ﬂyby                           tions for the VIMS data characteristics. In Section 3, We
missions and professional telescopes provide sporadic temporal                           determine the latitude of the asymmetry at 76 of the 96 distinct
                                                                                         wavelengths, excluding atmospheric windows where the
                                                                                         visible surface precludes haze measurements (Vixie et al.
                 Original content from this work may be used under the terms
                 of the Creative Commons Attribution 4.0 licence. Any further            2012). Each distinct wavelength accesses a different altitude
distribution of this work must maintain attribution to the author(s) and the title       because of the varying atmospheric opacity. In Section 4, we
of the work, journal citation and DOI.                                                   locate the latitude of the NSA boundary on 13 distinct ﬂybys.

                                                                                     1
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                 Vashist et al.




                                                                                       Figure 2. Titanʼs NSA is evident in two individual band images from the same
                                                                                       VIMS cube in Figure 1 (CM_1634082284_1). In the left cube, the southern
                                                                                       hemisphere appears brighter at blue wavelengths. In contrast, the NSA reverses
                                                                                       in the right image within a methane absorption band at 0.886 μm. Vertical
                                                                                       striping artifacts are present in the images.

                                                                                       making the gas dim. So higher haze concentrations make a
                                                                                       hemisphere brighter at long wavelengths. The Cassini VIMS
                                                                                       ﬂybys used in this study had varied spacecraft distances,
                                                                                       leading to a range of spatial resolutions. We chose these
                                                                                       particular ﬂybys as ones where a majority or all of Titan’s disk
                                                                                       is observed. Figure 3 outlines our VIMS image processing
Figure 1. A T62 image of VIMS cube CM_1634082284_1 taken on 2009                       algorithm. An issue found in the VIMS data is the vertical
October 12 shows a typical low-phase-angle view of Titan with the NSA at
visible wavelengths from the VIMS instrument (VIMS-V). The colors are an               striping noise in the original images. We could not directly
approximation of true color using the VIMS-V channels. The overall orange              correct the striping noise because the data used to subtract the
color comes from the spectral and scattering properties of the haze particles as       background on board Cassini were not transmitted back to
well as atmospheric absorption and Rayleigh scattering, which predominate at           Earth (Brown et al. 2004). To mitigate the striping issues, we
bluer wavelengths. North is upwards in the image.
                                                                                       increased the signal-to-noise ratio by coadding images, and
                                                                                       then mapping the ﬁnal coadded image onto a cylindrical
We also determine the albedo contrast between the northern                             projection (Vixie et al. 2012). The spatial sampling of the
and southern hemispheres with regard to wavelength to                                  cylindrical images is ∼45 km or 1° of latitude. Note that the
calculate the boundary latitude, north–south (NS) ﬂux ratios,                          vertical striping appears as superimposed stripes across
and the tilt angle of the asymmetry. Finally, in Section 5, we                         cylindrical projections. Additionally, these projections do
compare our results to the existing archive of NSA boundary                            not include limb pixels with an emission angle above 60° and
observations and discuss the implications of these ﬁndings on                          thus minimize limb-darkening effects. Images at 96 distinct
the atmospheric conditions of Titan.                                                   wavelengths, also known as bands, were created for each
                                                                                       ﬂyby. After we generate a ﬁnal set of images, we adapt the
                   2. Observations and Methods                                         methods from Roman et al. (2009) to determine the latitude of
   As shown in Table 1, we analyze Cassini VIMS observa-                               the NSA boundary. We shift the images by 6°N and 6°S and
tions of Titan from 12 targeted ﬂybys from 2004 to 2015 and                            then subtract the shifted images from each other to create a
two nontargeted ﬂybys taken in 2017. Each ﬂyby observed                                high brightness contrast of each image, which highlights the
Titan from 0.356 to 1.046 μm in 96 distinct wavelength                                 presence of an asymmetry (Roman et al. 2009). These maps
channels to sample the transition from southern summer to                              were then sequentially analyzed along each longitude using a
northern summer (Brown et al. 2004). We selected these                                 sixth-order polynomial ﬁt to smooth out signal variations. We
particular ﬂybys based on boundary visibility, sufﬁcient time                          found the locations of critical points using the derivative
cadence, and baseline so as to obtain measurements spaced                              function of the ﬁt; extraneous values, such as imaginary
out over the entire period of Cassini observations. For a                              solutions and numbers outside of the latitude range, were
majority of the ﬂybys, the spatial sampling per pixel is                               removed to leave only the latitude location of the NSA at each
∼45 km or 1° of latitude. For the NSA, one hemisphere                                  longitude column. Our algorithm then ﬁnds the latitude value
appears brighter and the other dimmer, with a semidistinct line                        of the NSA transition for all the longitude columns within
near the tropics (i.e., low latitudes) dividing the two. The                           each projection and then averages them to determine the
identiﬁcation of the brighter hemisphere depends on the                                location of the NSA for each band in the ﬂyby. Since each
season and the observed wavelength that samples at the                                 column has a varied value for the NSA, we applied a moving
different altitudes as shown in Figure 2. Lorenz et al. (1997)                         average to the data to ﬂatten irregularities in the column
attribute the reversal to the separately varying single-                               brightness data. Using the latitude value found in each image,
scattering albedo of the haze and gas as a function of                                 we apply a simple average of the brightness 30°N and 30°S of
wavelength. At short visible wavelengths (Figure 2, left), the                         the NSA transition to determine the NS ﬂux ratio. We derive
haze has a low albedo, but the gas itself is relatively bright due                     I/F values using the average visible latitudes, where each
to Rayleigh scattering. Thus, more haze leads to a darker                              latitude is an average of all visible longitudinal brightness
hemisphere at short wavelengths. At near-infrared wave-                                values. We can attribute various inaccuracies in our results to
lengths close to the visible (Figure 2, right), however, the haze                      sampling area, surface wavelengths, changing subsolar and
single-scattering albedo is high, and methane starts to absorb,                        subspacecraft latitudes, image manipulation inaccuracies,

                                                                                   2
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                          Vashist et al.




Figure 3. Here we present a ﬂowchart of the image processing procedures to obtain the NSA boundary latitude and the NS ﬂux ratio from an individual VIMS cube
image.



                                                                        Table 1
                                                                 NS Boundary Observations

                                              Cassini                                     Subsolar         Boundary
Year        Month             Source          Flyby           λ (μm)          Ls (°)     Latitude (°)     Latitude (°)                   Citation
1980        Nov           Voyager/ISS                          0.450            8             4            −5.5 ± 1               Squyres et al. (1984)
1981        Aug           Voyager/ISS                          0.450            16            8            −5.5 ± 1               Squyres et al. (1984)
1990        Aug            HST/WFPC                            0.440           122           23            32 ± 10                Caldwell et al. (1992)
1992        Aug            HST/WFPC                            0.440           145           16            20 ± 10                  Smith et al. (1992)
1994        Oct           HST/WFPC2                            0.440           168           5.8            15 ± 5                 Lorenz et al. (1997)
1995        Aug           HST/WFPC2                            0.440           177           1.3            15 ± 5                 Lorenz et al. (1997)
1997        Nov           HST/WFPC2                            0.889           202          −10.7           10 ± 15                Lorenz et al. (1999)
2000      Nov–Dec         HST/WFPC2                            0.889           242          −24                L                   Lorenz et al. (2001)
2002        Dec             HST/ACS                         0.435, 889         271          −26.7          −20 ± 10         Inspection of Lorenz et al. (2006)
2003        Dec             HST/ACS                            0.435           288          −25.6          −20 ± 5          Inspection of Lorenz et al. (2006)
2004        Oct           Cassini/VIMS           Ta        0.356–1.046         300          −23.2          −12 ± 3                      This work
2004      Oct–Dec          Cassini/ISS                         0.889           300          −23             −8 ± 2                 Roman et al. (2009)
2005      Feb–Dec          Cassini/ISS                         0.889           309          −21             −8 ± 2                 Roman et al. (2009)
2005        Oct           Cassini/VIMS          T8         0.356–1.046         314          −19.6          −12 ± 3                      This work
2007      May–Dec          Cassini/ISS                         0.889          0.335         −11.5           −8 ± 2                 Roman et al. (2009)
2007        May           Cassini/VIMS         T31         0.356–1.046         333          −12.0          −12 ± 3                      This work
2009        Aug           Cassini/VIMS         T61         0.356–1.046          1            0.4           −12 ± 3                      This work
2009        Oct           Cassini/VIMS         T62         0.356–1.046          3             1            −12 ± 3                      This work
2010        Apr           Cassini/VIMS         T67         0.356–1.046          8             8            −12 ± 3                      This work
2011        Dec           Cassini/VIMS         T79         0.356–1.046          29           13            −10 ± 4                      This work
2012         Jul          Cassini/VIMS         T85         0.356–1.046          36           15            −10 ± 4                      This work
2013        Oct           Cassini/VIMS         T92         0.356–1.046          47           19             −5 ± 4                      This work
2014        May           Cassini/VIMS         T101        0.356–1.046          57           22                L                        This work
2015        Jan           Cassini/VIMS         T108        0.356–1.046          64           24                L                        This work
2015        Nov           Cassini/VIMS         T114        0.356–1.046          74           26                L                        This work
2017        Jun           Cassini/VIMS        278TI1       0.356–1.046          91           27             10 ± 3                      This work
2017        July          Cassini/VIMS        283TI1       0.356–1.046          92           27             10 ± 4                      This work

Note. ACS = Advanced Camera for Surveys.
a
  Nontargeted ﬂyby of Titan.


and/or phase angle differences within the ﬂybys. We                                                               3. NSA
determine uncertainties in our image processing algorithm
                                                                                                    3.1. Meridional Brightness Proﬁles
through the standard deviation of the derived latitude value of
every image. Additionally, instrumental artifacts and noise                          Figure 4 displays various meridional brightness proﬁles
within the VIMS-V instrument play a factor. Inherent                              from speciﬁc ﬂybys at two wavelengths: one near-infrared and
systematic errors also derive from the softness of the                            one visible. At certain wavelengths, the location of the
boundary itself and any potential offset between the atmo-                        boundary generally corresponds with either the maxima or
spheric pole and the geographic pole.                                             minima of the proﬁles, depending on the season. The values of

                                                                             3
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                 Vashist et al.




Figure 4. Meridional disk brightness (I/F) proﬁles across Titan from select Cassini ﬂybys over a Titan half year along with published HST NS ratio spectra in the top
panel. The top plot shows the proﬁle at 0.550 μm and the bottom at 1.003 μm. The error bars represent the NS boundary location and its uncertainty for each ﬂyby at
that wavelength. The shape and trends of the proﬁle did not exhibit much change between 2004 and 2007 (Ta–T31). With the onset of the vernal equinox in T61, the
meridional proﬁle developed dramatic changes resulting from both varying solar illumination and dynamic atmospheric structure, with a near-complete latitudinal
reversal by 2017 (278TI). It is important to note that in T85, 278TI, and 283TI, drop-offs are removed as the disks are terminated. Additionally, the coaddition of
longitudinal data tends to resolve the NSA boundary poorly, resulting in sometimes indistinct NSA boundary transitions, like in 278TI.



each proﬁle are averaged over longitude to minimize noise                             the early cutoff in their NS ﬂux ratio proﬁles just below the
from striping and various other artifacts. Flybys from the                            equator in Figure 4.
Prime mission (2004–2008) in southern summer have a single
relative maximum that increases in latitude as time progresses,
whereas later ﬂybys show greater variance. The movement of                                                  3.2. NS Flux Ratio Spectra
the relative extrema also serves to visualize the changes in the                         Figure 5 displays the contrast ratio, a comparison between
NS ﬂux ratio. We note that limited spatial coverage in the                            regions 30° north and south of the NS boundary, as a function of
VIMS images of the nontargeted ﬂybys (278TI, 283TI) causes                            wavelength in four panels. The different lines represent different

                                                                                 4
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                     Vashist et al.




Figure 5. NS ﬂux ratio spectra (average ratio of I/Fs 30° above and below the NSA boundary latitude) from select ﬂybys over a Titan half year. Note that wavelengths
where methane absorbs (e.g., 0.89 μm) tend to show an inverted NS ﬂux ratio relative to methane windows (e.g., 0.93 μm). The Ta and T8 spectra show distinct
features, while the T114 and 278TI spectra show those same features with a ﬂipped concavity. Changing illumination and viewing conditions can shift these NS ratio
spectra vertically. Changing illumination cannot, however, invert absorption lines as we see happen from northern winter into northern spring at 0.89 μm. Thus much
of the variation in spectral shape can be attributed to varying atmospheric structure. The other spectra from T31 to T108 that were recorded closer to the vernal equinox
show subdued or nonexistent features, indicating a more uniform meridional haze proﬁle for Titan.

ﬂybys, with ﬂybys being sorted into either northern fall, northern                      radiation received by a certain area, drives Hadley circulation
winter, equinox, or northern summer. The total observed time                            in the upper atmosphere with upwelling at the summer pole and
for Cassini equates to about half a Titan year. The top panel in                        subsidence at the winter pole (Tokano 2007; Lebonnois et al.
Figure 5 shows that the region south of the NS boundary is                              2014), driving up haze concentration in the winter hemisphere.
brighter across all wavelengths except in methane bands, like                           The observed reversal in the NS meridional brightness proﬁle
0.89 μm. At equinox, the NS ﬂux ratio increases until the proﬁle                        before and after the equinox records a clear trend of global
has NS ﬂux ratio values greater than 1, indicating a meridional                         Hadley haze circulation driven by seasonal changes in
movement of the stratospheric haze layers, vertical motion, and/                        insolation. The NS ﬂux ratio spectra show a similar reversal
or differences in aerosol properties. The extent to which changes                       (across the NS ﬂux ratio value of 1 or the red dashed line in
in the location of the NS boundary are inﬂuenced by any one of                          Figure 5) in the methane absorption band proﬁles over the
these factors is unclear, but it is likely that the changes result                      seasons at visible and infrared wavelengths, sampling altitudes
from multiple factors, each playing a role in the movement of the                       between 60 and 250 km (Robinson et al. 2014). The
boundary. The ﬂat T108 NS ﬂux ratio proﬁle near northern                                combination of the meridional proﬁle and NS ﬂux ratio spectra
summer indicates that the NSA has disappeared at all                                    observations over a Titan half year further support the idea of a
wavelengths. Then in 2017, we observe a new set of inverted                             positive feedback loop between the global atmospheric
proﬁles with NS ﬂux ratios over 1 at visible wavelengths of                             circulation patterns and haze production (Rannou et al. 2002)
0.4–0.75 μm and below 1 at infrared wavelengths of                                      where the global movement of the atmosphere, as observed by
0.8–1.05 μm. The proﬁle inversion suggests the formation of a                           the changes in the NSA, inﬂuences the production of haze. In
new NS boundary above the equator. Outlier ﬂuctuations in the                           turn, the haze particles can further exacerbate the magnitude of
NS ﬂux ratio with wavelength can be attributed to a wavelength-                         atmospheric circulation and haze concentration. This synergis-
dependent haze single-scattering albedo and atmospheric                                 tic relationship inﬂuences both the number of haze particles,
gaseous absorption. We observe a gradual increase in the NS                             but also the seasonal movements of Titan’s atmosphere. Our
ﬂux ratio as time progresses through the ﬂybys with a minimum                           results on the contrasting brightness proﬁles match well with
value of 0.773 at 0.51 μm in the Ta ﬂyby (2004), 1.034 at                               other recent works analyzing the NSA brightness differences
0.41 μm for T67 (2010), and 1.25 at 1.03 μm for T108 (2015).                            with spectral models (Kutsop et al. 2022) and modeling haze
As for averages, Ta, T67, and T108 exhibit an average ﬂux ratio                         concentrations associated with the NSA using principal
of 0.922, 1.17, and 1.29, respectively.                                                 component analysis (Karkoschka 2022).

                                                                                                                     4. NS Boundary
              3.3. Implications for Global Circulation
                                                                                                                 4.1. Boundary Latitude
   The evolution of the meridional haze brightness proﬁles over
the Cassini mission traces atmospheric circulation at different                           The straight-line interhemispheric boundary was the most
altitudes in the stratosphere. Insolation, the quantity of solar                        prominent feature in the ﬁrst high-quality spatially resolved

                                                                                   5
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                               Vashist et al.




Figure 6. NS boundary latitudes from select ﬂybys that represent a Titan half year. As expected, the data show no evidence of a relationship between wavelength and
NS boundary latitude. Even in wavelengths where the NS ﬂux ratio is 1, gradient-based indicators for the NS boundary remain due to the wide sampling latitudinal
range (±30°).

images of Titan, those taken by the Voyager 1 spacecraft in                          spanned 45°S to 0°. The difference in position between 2002
1980. The initial examination of those images suggested that                         and 2003 did not appear to be signiﬁcant (although the contrast
the boundary was within 5° of the equator near the northern                          between hemispheres did increase noticeably). Examination of
spring equinox (Smith et al. 1981). The location of the                              the 2002 images at 0.502 μm and 0.892 μm suggests a similar
boundary as seen by Voyager 2 nearly a year later was                                contrast boundary latitude within 5° at those wavelengths.
essentially identical: Squyres et al. (1984) determined the                          From the Cassini mission, Roman et al. (2009) analyzed ISS
latitude of the boundary with some precision in both data sets                       images taken from 2004 to 2007 to pin down the clear NSA
to be 5°. 5 ± 1°S. Flasar et al. (1981) observed a strong                            boundary to ∼8°S within an error margin of 2°.
superrotation of the Titan atmosphere at all latitudes through                          Now, we show observations from VIMS cube images taken
temperature variations. The slow rotation of Titan makes its                         from ﬂybys acquired over the entire Cassini mission in Figure
superrotating atmosphere more prominent, with the haze                               7. We adopt a stretch that enhances the visibility of the high-
changes documented in our study suggesting that the strato-                          contrasting band for Figure 7 which also tends to accentuate
sphere exhibits a near-seasonal evolution of circulation                             noise and instrument artifacts.
patterns. There were no resolved observations of Titan between                          In Figure 6, we observe not only the brevity of the change in
Voyager 2 in 1981 and 1990 when the new HST imaged Titan.                            latitudes as the seasons progress but also a lack of relation
However, the deconvolution of HST images ﬁt with models                              between wavelength and boundary latitude. This static location
(Caldwell et al. 1992) suggested that the boundary was located                       of the boundary as a function of wavelength suggests an
at 32° ± 10°N.                                                                       attribution of global circulation rather than truly altitude-
   Lorenz et al. (1997) reported that the best ﬁt for postrepair
                                                                                     dependent processes.
HST images acquired in 1994 and 1995 suggested that the
                                                                                        Between the Ta and T92 ﬂybys, we observe the high-
boundary was between 10 and 20°N. Further HST images in
                                                                                     contrasting band south of the equator (red dashed line in
1997 and 2000 were examined by Lorenz et al. (2004)—and
                                                                                     Figure 7). After T92, the high-contrasting band is not visible in
although the appearance in each case is not inconsistent with a
                                                                                     all targeted ﬂyby images (e.g., T108), and instead, multiple
near-equatorial boundary, the contrast is both weak and takes
the form of a ramp rather than a step, so it was difﬁcult to                         bands appear at random locations far from the equator. The
deﬁne a precise boundary latitude. During the northern winter                        extra bands are consistent with observations of secondary
from 2002 to 2003, an inspection of NS proﬁles of 0.435 μm                           bands appearing during the northern summer (Kutsop et al.
HST images by Lorenz et al. (2006) also showed a ramp in                             2022), but are not the focus of this publication. The NSA
albedo ratio (the sharpness of the albedo boundary is in part                        boundary latitudes stay at fairly constant values of 8–11°S. We
reduced owing to the telescope point-spread function), but the                       only witness the return of the high-contrasting band in two
contrast was then large enough (in a sense opposite from that in                     nontargeted ﬂybys 278TI and 283TI, imaging Titan in mid-
the mid-1990s) such that the midpoint of the ramp could deﬁne                        2017, which owing to spacecraft observation geometry only
an approximate boundary. The 2002 December image ramp                                include half-disk observations just below the equator. How-
spanned 40°S to 10°S, and so 20°S to 22°S is near the 25°S                           ever, the high-contrasting band in the 2017 images indicates
midpoint; the 2003 image ramp is a little better deﬁned and                          that the NSA boundary ﬂipped across the equator. The new

                                                                                6
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                 Vashist et al.




Figure 7. Cylindrical projection maps of the high-contrasting band for every ﬂyby with a faint demarcation of the equator in red. The color bar shows the brightness
difference between two vertically shifted images. The dark band is visible in most ﬂybys. We note two distinct bands in the T108 image, which hints at the lack of an
NSA boundary.



NSA boundary latitude was approximately 10°N within an                                high-contrasting bands in our VIMS images for select ﬂybys
error margin of 4°.                                                                   (Ta, T31, and T62) do not appear linear across all longitudes.
                                                                                      That is, they remain horizontal on the left side of the band until
                                                                                      starting a gradual upward tilt on the right side. Nonetheless, we
                           4.2. Boundary Tilt
                                                                                      chose to use the higher-quality VIMS observations from T67 to
   Looking further at the high-contrasting bands in Figure 7,                         deduce tilt measurements with a least-squares linear regression,
many of the bands appear to be slanted or tilted with respect to                      shown in Figure 8. Overall, we ﬁnd further evidence for a
the symmetry axis of the bands, indicating an offset in the                           small, detectable tilt of the NSA boundary, despite the lower
atmospheric circulation axis relative to the rotational axis of                       spatial sampling of the VIMS cube images. Similar measures of
Titan (Achterberg et al. 2008a; Roman et al. 2009; Kutsop et al.                      the tilt angle between 2° and 6° were found in an analysis of
2022). The tilt angle is positive for high-contrasting bands                          circumpolar bands in VIMS images (Kutsop et al. 2022).
oriented above the geographic equator. Roman et al. (2009)                               Consequently, we ﬁnd rough estimates for the tilt angle
made a similar observation of a tilt in the NSA during the early                      ranging from 1 to 2°, as opposed to the 4° of Roman et al.
ﬂybys from 2004 to 2007. Visually speaking, the tilts of the                          (2009). Though the method for deriving tilt is suboptimal given

                                                                                 7
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                   Vashist et al.




Figure 8. The top and bottom panels show T67 VIMS cube images at 0.886 and 1.003 μm, respectively. These maps show a visible tilt in the NSA boundary. Some
mapping artifacts and voids have been removed to show only the parts of Titan visible during the ﬂyby. The red line tracks individual measurements of the NSA
boundary latitude for image columns where the NSA is evident. The blue line shows a linear regression model ﬁt to those NSA boundary latitudes to determine a tilt
angle for the NSA boundary. The linear ﬁt indicates that at 0.886 μm, the NSA boundary is located 12°S ± 3° with an angle of −0°. 81 ± 0°. 22. At 1.003 μm, the NSA
boundary is at 12°S ± 0° with an angle of −0°. 55 ± 0°. 23. These ﬁts and their errors assume a linear tilt to the NS boundary, as opposed to a spherical offset of the
atmospheric pole from the rotational pole. Therefore the reported measurement precision may underrepresent the true accuracy after accounting for systematic errors
due to our underlying tilt assumption.


the changes in observation geometry and the fact that a linear                         differences found in our underlying assumption of a linear tilt,
ﬁt is being applied to a projection of Titan rather than the moon                      as opposed to a spherical offset of the atmospheric pole. We
itself, the current method provides the best point of comparison                       also note that the tilt angle is only one factor inﬂuencing the
to previous works. The reported measurement precision may                              geometry of the NSA boundary in the Titan images. Other
underestimate the true accuracy of the tilt due to the systematic                      factors include an azimuthal offset from the subsolar longitude

                                                                                  8
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                 Vashist et al.




Figure 9. Individual VIMS cube images at 0.886 μm taken within one year of each other at a low (37°) and high (88°) phase angle, respectively. The right plot (C)
shows a comparison of the meridional proﬁles for the two phase angles in several methane windows. At blue and green wavelengths, there is little change between the
corresponding low- and high-phase proﬁles. Differences between the two phases increase at longer wavelengths, while the gradient in I/F vs. latitude becomes more
shallow at the longest wavelengths. The red dashed lines in (A) and (B) indicate the longitude of the meridional proﬁles from each image.




Figure 10. A sequence of false-color (RGB: 0.725, 0.886, 1.003 μm, respectively), near-infrared VIMS cube images show the fading and ﬂipping of the NSA
boundary over a Titan half year. The orbital diagram is adapted from Figure 1 of Seignovert et al. (2021) to provide context with the Titan seasons. The red dashed
lines indicate the equator. The Ta image has a sharp brightness contrast between the bright (violet) and dark (indigo) hemispheres. The NSA contrast was reduced, but
the NSA boundary line was still visible below the equator by 2012 (T85). The NSA boundary line is lost in the T101 image, but some purple color remains. The lack
of the NSA persists in the T114 image with a uniform (white) disk brightness proﬁle. The NSA does not return until the near-end of the Cassini mission in mid-2017 in
the 278TI and 283TI images. The bright and dark hemispheres have ﬂipped with the subtle boundary line above the equator.


(Roman et al. 2009). The origin of the tilt may arise from an                         Roman et al. (2009) in 0.889 μm images should have been
offset between the atmospheric and geographic (solid body)                            constant throughout the seasonal cycle. Although the tilt angle
poles (Roman et al. 2009). It is not obvious that the obliquity of                    appears to have remained ﬁxed over 2004–2007 (Roman et al.
4° observed by Achterberg et al. (2008b) in thermal data and by                       2009), we ﬁnd that the tilt value was similarly persistent over

                                                                                 9
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                Vashist et al.




Figure 11. The time line of the NSA boundary latitude as measured by instruments aboard Titan ﬂyby missions and space telescopes. Before 2000, there was more
uncertainty in the NSA boundary due to the reliance on noisier HST images. The period of rapid change applies to the early HST observations of seasonal NSA
changes reported in Lorenz et al. (2001). Nearly half of Titan’s 29.5 yr seasonal cycle is precisely documented with the new VIMS NSA boundary data set. A period of
a stable NSA from 2004 to 2014 was halted by an abrupt change in the fading of the NSA boundary from 2014 until its reversal and reappearance in 2017. Note that
small systemic variations within the uncertainties exist in the NSA boundary latitude estimations between the Cassini ISS and VIMS images from 2004 to 2007. Ls
refers to solar longitude: 0° at Titan’s northern spring equinox.


the Cassini Mission (2004–2017) even though the boundary                             between the two phase angles. The high-phase proﬁles show a
does migrate over the course of a year.                                              more gradual gradient and more southerly inﬂection point from
                                                                                     60°S to 30°N relative to the low-phase proﬁles.
                     4.3. Low versus High Phase                                         At near-infrared wavelengths, high-phase-angle values
                                                                                     exhibit a generally higher I/F. These NSA observations at
   A comparison of NSA images (0.886 μm) at a low (37°) and
high phase (88°) angles is shown in Figure 9. The images were                        different phases demonstrate that the haze phase function of the
taken only nine months apart for a minimally biased                                  aerosol particles inﬂuences the overall brightness of the NS
comparison. We observe that the NSA boundary is still                                brightness proﬁle, muting the contrast when viewing at higher
detectable at a high phase angle, albeit at lower contrast.                          phase angles. The degree of the reduced contrast varies with
Differences between low and high phase angles are relatively                         wavelength, such that the contrast minimizes within the
small (less than 0.02 in I/F at boundary latitudes) for the blue-                    methane windows at optical wavelengths. In near-infrared
and green-wavelength ﬁlters. However, at longer wavelengths,                         methane windows, the phase effect is less pronounced, and the
the meridional proﬁles show more considerable differences                            contrast is similar within and outside the windows. Viewing the

                                                                                10
```

<!-- PDF_PAGE: 11 -->

## PDF page 11

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                            Vashist et al.

NSA at a higher phase angle subdues but does not erase the                                          5. Conclusion
contrast of the boundary.
                                                                             The NSA in the haze between Titan’s hemispheres tracks the
                                                                          evolution of haze abundances driven by the global atmospheric
                      4.4. Seasonal Trends                                circulation of Titan. Given the Cassini VIMS data, we can
                                                                          accurately monitor changes in the stratosphere of Titan through
   We show the seasonal evolution in the hemispheric proﬁle of
                                                                          a nearly half-year seasonal cycle using consistent instruments
the NSA using near-infrared colors in Figure 10. Purple shows
                                                                          and analysis. We ﬁnd the seasonal changes in the NSA
the darker hemisphere, while violet and white show the brighter
hemisphere with a higher haze concentration (refer to Section 3           boundary latitude and hemispheric ﬂux dichotomy follow a
for more details). The contrast between the northern and                  two-step sequence. A period of constant NSA boundary
southern hemispheres of the boundary is visible when the                  latitude with a slow changing ﬂux ratio for several years from
boundary is located away from the solid-body equator (red                 the start of northern winter in 2004 until a few years after the
dashed line in Figure 10) during 2004–2013 and 2017 but not               vernal equinox in 2011. In 2012, a rapid change in the NSA
when the boundary transitions during 2014–2016. The                       boundary shifted toward the equator and subsequently fades
boundary transitions between the north and south also                     away until reappearing in the opposite hemisphere during
experience a decrease in contrast as the boundary approaches              northern summer in 2017. The current study also ﬁnds a
the equator. Eventually, the brightness contrast between the              detectable, few degrees tilt of the NSA (−0°. 8 ± 0°. 2 and
north and south of the boundary fades away, which marks the               −0°. 6 ± 0°. 2 in T67 at 0.886 μm and 1.003 μm, respectively)
start of the transitional period in 2014.                                 that reinforces the presence of a superrotating atmosphere
   In the T101 and T114 images from Figure 10, the disk shows             found in Achterberg et al. (2008a) and Roman et al. (2009),
a uniform white color that indicates the fading of the                    who found tilts of 4° ± 2° and 3°. 8 ± 0°. 9, respectively. We
hemispheric dichotomy. The reemergence of the dichotomy                   demonstrate that the NSA boundary is detectable at higher
north of the equator does not happen until the latest nontargeted         phase angles up to 90° with reducing I/F contrast.
ﬂybys (278TI and 283TI), as noted by the ﬂip in the respective               Two products from global circulation models (GCMs) would
colors of each hemisphere. The VIMS cube images from the                  be useful to match with observations such as those described
nontargeted Titan ﬂybys have poor global coverage and                     here (and those that might be developed from Earth-based
inconsistent striping error that limits the number of usable              observations in the coming decade).
VIMS cubes.                                                                  First are tables of haze abundance versus latitude and altitude
   In our analysis, we document the migration of the NSA                  at different dates (seasons). These would allow the generation of
boundary during the Cassini Solstice Mission and prior                    synthetic images (or proﬁles of I/F versus latitude) for visible
telescope observations in Figure 11. From 2004 to 2010, the               images, and similarly (with an assumed methane proﬁle) for the
boundary remains at a steady 11.8°S, but then experiences a               near-infrared. Since different wavelengths probe different
sudden change from 2012 to 2014, where the boundary moves                 altitude ranges, and it has already been noticed (e.g., Lorenz
from 9.8°S to 5.0°S. The motion of the NSA boundary in this 2             et al. 1999) that the highest altitudes (e.g., those in the blue, or
yr period is also similar to the previous postequinox period              deep in methane bands) change ﬁrst, comparison of these data
from 1995 to 1997 (Lorenz et al. 1999). Over the next 4 yr, the           with GCM results would be a valuable constraint to reﬁne the
N/S ratio not only reverses but the NSA boundary is observed              latter. Although there have been useful discussions of GCM
to be at ∼10°N, corresponding to a change of ∼15° (∼673                   predictions of the detached haze layer (which has an optical
km). A similar report of the NSA reversal in an analysis of HST           depth that is small enough to be neglected in the images studied
Space Telescope Imaging Spectrograph taken from 1998 to                   here). The extent to which GCMs with aerosols as tracer
2004 and 2017 to 2019 (Karkoschka 2016, 2022) corroborates                particles yield sharp NSA boundaries versus brightness “ramps”
with our ﬁnding.                                                          has not, to the present authors’ knowledge, been reported.
   Our new observations resemble the report of a rapid change                Second, apart from the detached haze layer (e.g., West et al.
in the NSA at 0.889 μm between 1997 and 2000 (Lorenz et al.               2018), the quantitative association of optical albedo features in
2001) and conﬁrm the initiation of the NSA reversal across the            Titanʼs haze with speciﬁc features of the meridional circulation
equator occurs 2–3 yr after the equinox. The VIMS observa-                has not been attempted. Speciﬁcally, it is tempting to associate
tions constrain the 5 yr transition period into behaviors that are        the sharp near-equatorial NSA transition observed here and
nearly identical to previous NSA observations after the equinox           before with the boundary between two symmetric meridional
(Sromovsky et al. 1981; Lorenz et al. 1999, 2001). A few years            circulations (i.e., Hadley cells) seen in models around equinox
after the equinox, the NSA boundary begins moving toward the              (e.g., Figure 4 in Lora et al. 2015). The latter model shows a
equator at a rate of a few degrees of latitude per year for ∼2            symmetric (i.e., two-cell) meridional stream function pattern at
yrs. Afterward, the NSA boundary vanishes, reﬂecting a more               relevant altitudes (100 km) at Ls = 180, but that the pattern has
diffuse transition between the hemispheres, where the NSA is              become single cell (pole–pole) by Ls = 250. The extent to
viewed as a gradient rather a global phenomena with a                     which the symmetric circulation becomes asymmetric (with the
localized feature (the NS boundary).                                      summer cell growing, its downwelling branch progressively
   The evolution of the NSA boundary over a Titan year can be             encroaching into the winter hemisphere) versus it simply fading
broken up into two distinct periods of change. One period                 away with constant latitude boundaries to be replaced with a
includes a constant extrema latitude for several years until the          progressively intensifying pole–pole mode is not presently
equinox, followed by the second period of rapid linear change,            clear, but could be elucidated with appropriate model outputs.
and the distinct NSA fades away until the boundary reappears                 In the not-too-distant future, missions such as the Dragonﬂy
on the other side of the equator. Seasonal variations in the              drone or an orbiter (Lorenz et al. 2018; Barnes et al. 2021)
global atmospheric circulation may vary the NSA latitude for              could lead to an active probe that can measure in situ changes
each extremum of the NSA cycle.                                           in the lower atmosphere on a more localized level instead of

                                                                     11
```

<!-- PDF_PAGE: 12 -->

## PDF page 12

```text
The Planetary Science Journal, 4:118 (12pp), 2023 June                                                                                                   Vashist et al.

through changes in haze observed from orbit. Future analysis                             Brown, R. H., Baines, K. H., Bellucci, G., et al. 2004, SSRv, 115, 111
can examine the long-term effects of a superrotating atmos-                              Brown, R. H., Lebreton, J.-P., & Waite, J. H. 2009, Titan from Cassini-
phere on Titan, and data taken over a larger period of time can                             Huygens (Berlin: Springer)
                                                                                         Caldwell, J., Cunningham, C. C., Anthony, D., et al. 1992, Icar, 97, 1
aim to gain a better understanding of the movement of the                                Flasar, F., Samuelson, R., & Conrath, B. 1981, Natur, 292, 693
atmosphere relative to the rotational axis. In addition, a study                         Hirtzig, M., Coustenis, A., Gendron, E., et al. 2006, A&A, 456, 761
analyzing the correlation between the polar hood and the                                 Karkoschka, E. 2016, Icar, 270, 339
seasonal motion of the NSA boundary could reveal new details                             Karkoschka, E. 2022, Icar, 387, 115188
                                                                                         Kutsop, N., Hayes, A., Corlies, P., et al. 2022, PSJ, 3, 114
about the haze distributions.                                                            Lebonnois, S., Flasar, F. M., Tokano, T., & Newman, C. 2014, in Titan, ed.
                                                                                            I. Müller-Wodarg (Cambridge: Cambridge Univ. Press), 122
                            Acknowledgments                                              Lora, J. M., Lunine, J. I., & Russell, J. L. 2015, Icar, 250, 516
                                                                                         Lorenz, R., Young, E., & Lemmon, M. 2001, GeoRL, 28, 4453
   A.V. is supported by the Dyess Fellowship at the University                           Lorenz, R. D., Lemmon, M. T., & Smith, P. H. 2006, MNRAS, 369, 1683
of Idaho. M.F.H. and J.W.B. are supported by NASA Cassini                                Lorenz, R. D., Lemmon, M. T., Smith, P. H., & Lockwood, G. 1999, Icar,
Data Analysis Program grant 80NSSC19K0896. C.H. was                                         142, 391
                                                                                         Lorenz, R. D., Smith, P. H., Lemmon, M. T., et al. 1997, Icar, 127, 173
supported by the NASA/ESA Cassini project.                                               Lorenz, R. D., Smith, P. H., & Lemmon, M. T. 2004, GeoRL, 31, L10702
                                                                                         Lorenz, R. D., Turtle, E. P., Barnes, J. W., et al. 2018, JHATD, 34, 14
                                ORCID iDs                                                Rannou, P., Hourdin, F., & McKay, C. 2002, Natur, 418, 853
                                                                                         Robinson, T. D., Maltagliati, L., Marley, M. S., & Fortney, J. J. 2014, PNAS,
Aadvik S. Vashist https://orcid.org/0000-0002-6318-7226                                     111, 9042
Michael F. Heslar https://orcid.org/0000-0002-9304-8657                                  Roman, M. T., West, R. A., Banﬁeld, D., et al. 2009, Icar, 203, 242
Jason W. Barnes https://orcid.org/0000-0002-7755-3530                                    Seignovert, B., Rannou, P., West, R. A., & Vinatier, S. 2021, ApJ, 907, 36
Ralph D. Lorenz https://orcid.org/0000-0001-8528-4644                                    Smith, B. A., Soderblom, L., Batson, R., et al. 1982, Sci, 215, 504
                                                                                         Smith, B. A., Soderblom, L., Beebe, R., et al. 1981, Sci, 212, 163
                                                                                         Smith, P., Karkoschka, E., & Lemmon, M. 1992, BAAS, 24, 950
                                 References                                              Squyres, S., Thompson, W., & Sagan, C. 1984, BAAS, 16, 664
                                                                                         Sromovsky, L. A., Suomi, V. E., Pollack, J. B., et al. 1981, Natur, 292, 698
Achterberg, R., Conrath, B., Gierasch, P., Flasar, F., & Nixon, C. 2008a, Icar,          Tokano, T. 2007, P&SS, 55, 1990
  197, 549                                                                               Tomasko, M. G., Archinal, B., Becker, T., et al. 2005, Natur, 438, 765
Achterberg, R. K., Conrath, B. J., Gierasch, P. J., Flasar, F. M., & Nixon, C. A.        Vixie, G., Barnes, J. W., Bow, J., et al. 2012, P&SS, 60, 52
  2008b, Icar, 194, 263                                                                  West, R. A., Balloch, J., Dumont, P., et al. 2011, GeoRL, 38, L06204
Barnes, J. W., Turtle, E. P., Trainer, M. G., et al. 2021, PSJ, 2, 130                   West, R. A., Seignovert, B., Rannou, P., et al. 2018, NatAs, 2, 495




                                                                                    12
```
