---
citation_key: "snell2024titan"
title: "{Titan's Atmospheric Albedo Asymmetry and Seasonal Variability Observed through the Cassini Imaging Science Subsystem}"
source_pdf: "data/papers/snell2024titan.pdf"
source_pdf_sha256: "582635eaad3c2c61b233b364583f11333243c335152314260e218f3b1a8b2846"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                     https://doi.org/10.3847/PSJ/ad0bec
© 2024. The Author(s). Published by the American Astronomical Society.




Titan’s Atmospheric Albedo Asymmetry and Seasonal Variability Observed through the
                        Cassini Imaging Science Subsystem
                                                                         C. Snell1   and D. Banﬁeld2
                                          1
                                         Department of Astronomy, Cornell University, 122 Sciences Drive, Ithaca, NY 14853, USA
                                                        2
                                                          NASA Ames Research Center, Moffett Field, CA, USA
                                 Received 2023 March 29; revised 2023 October 10; accepted 2023 October 12; published 2024 January 18

                                                                                     Abstract
             Using images from Cassini, we analyzed the north–south albedo asymmetry that has been observed in the
             atmosphere of Saturn’s moon, Titan. Suitable images from the Cassini Imaging Science Subsystem taken at
             889 nm spanned from 2004 to 2017—around half of a Titan year—and revealed seasonal changes in the
             characteristics and orientation of the north–south asymmetry boundary. Such circumglobal features provide insight
             into the dynamics and circulation of the atmosphere more broadly. The albedo asymmetry has been observed to
             reverse for part of the Titan year, inverting the brighter and darker hemispheres; we also observed this inversion,
             along with the formation of additional banding brieﬂy during the transition (around 2014–2016). A tilt in the
             rotation axis of Titan’s atmosphere with respect to the solid body rotation has previously been noted. Using robust
             edge-detection techniques, we likewise identiﬁed a tilt offset of a few degrees in the albedo transition boundaries.
             The azimuth of this tilt axis remained roughly ﬁxed in inertial space, with some smaller possible seasonal
             ﬂuctuations around the ﬁxed direction noted.
             Uniﬁed Astronomy Thesaurus concepts: Titan (2186); Atmospheric dynamics (2300); Seasonal phenomena (1437);
             Atmospheric circulation (112)
             Supporting material: machine-readable table


                                    1. Introduction                                         (Tomasko & Smith 1982). Through subsequent observations
   The atmosphere of Titan is a compelling area of research, not                            from the Hubble Space Telescope and Cassini, seasonal
only for its uniqueness as the most substantial atmosphere of                               dependence has been observed, including in the north/south
any moon in the solar system and as a rare example of                                       albedo ratio (Lorenz et al. 1997), the overall disk brightness
atmospheric super-rotation, but also for the many similarities it                           (Lockwood & Thompson 2009), and the reversal of the bright
shares with Earth’s atmosphere. Titan is shrouded in a thick                                and dark hemispheres for half of the orbital period (Caldwell
layer of haze made up of organic species produced through                                   et al. 1992).
photochemical processes (Yung et al. 1984). There are a few                                     Since the NSA and other band-like atmospheric features
notable features that have been observed in the atmosphere,                                 likely trace the atmospheric circulation, the characteristics of
including the detached haze layer, the north–south albedo                                   such features are a useful tool to inform our understanding of
asymmetry (NSA), polar hoods, and a number of more subtle                                   Titanʼs atmospheric dynamics more broadly. Previous studies
bands at varying latitudes throughout the Titan year (Smith                                 have noted an offset or tilt in the rotation axes of several
et al. 1981, 1982; Sromovsky et al. 1981; Tomasko &                                         atmospheric features relative to the rotation axis of the solid
Smith 1982; Lorenz et al. 1997). The NSA, polar hoods, and                                  body of Titan: the north polar zone (Sromovsky et al. 1981),
other banding in particular have been observed to exhibit                                   the NSA boundary (Roman et al. 2009), middle atmosphere
seasonal variations (Caldwell et al. 1992; Lorenz et al. 1997;                              isotherms (Achterberg et al. 2008a, 2011), clouds (West et al.
Kutsop et al. 2022), and are crucial to understanding Titan’s                               2016), and gas abundances (Teanby et al. 2010; Sharkey et al.
atmospheric dynamics and photochemistry. Here, we use the                                   2020). One recent study (Kutsop et al. 2022) also investigated
Cassini Imaging Science Subsystem (ISS) to characterize the                                 seasonal trends in the orientation of polar and equatorial annuli
NSA boundary and its variation throughout the Cassini                                       using Cassini Visible and Infrared Mapping Spectrometer
mission. We also discuss its implications for Titan’s atmo-                                 (VIMS) data. These previous studies have mostly indicated a
spheric circulation more generally.                                                         tilt of around a few degrees for their respective features. In
   The atmosphere of Titan has a variation in albedo between                                studies with sufﬁcient temporal coverage (e.g., Achterberg
the northern and southern hemispheres. The difference is most                               et al. 2011 and Kutsop et al. 2022), the azimuth of the tilt axis
clearly visible in the Cassini ISS methane band ﬁlter at 889 nm                             appears to be roughly ﬁxed in inertial space, but may exhibit
(MT3), but it can also be seen in images using green, blue,                                 some seasonal ﬂuctuations around some inertially ﬁxed point
violet, and other methane band ﬁlters. The feature was ﬁrst                                 offset from the north pole. General circulation models (GCMs)
observed in the 1980s in images captured by Voyager 1 and 2                                 have shown similar results for seasonal ﬂuctuations in tilt
(Sromovsky et al. 1981; Smith et al. 1982) and Pioneer 11
                                                                                            amplitude, but differ somewhat from existing observations of
                                                                                            how the tilt azimuth changes throughout the Titan year (e.g.,
                 Original content from this work may be used under the terms
                 of the Creative Commons Attribution 4.0 licence. Any further
                                                                                            Tokano 2010). With relatively few studies of the atmospheric
distribution of this work must maintain attribution to the author(s) and the title          tilt over sufﬁciently long time spans, the seasonal dependence
of the work, journal citation and DOI.                                                      of the tilt orientation is not well constrained. Thus, further

                                                                                        1
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                      Snell & Banﬁeld

constraining the temporal trends in the tilt axis was a main              banding were still noted, however, so outliers were removed
objective in this study.                                                  with median ﬁlters, and banding was reduced by subtracting
   Overall, the tilt itself and the mechanism behind it are not yet       out the vertical and horizontal means of the dark space around
well understood. Titanʼs atmosphere is unique with its super-             the planet disk in each image and median-ﬁltering with a long,
rotation and varying insolation throughout the year (see, e.g.,           narrow boxcar.
Flasar & Achterberg 2009). Characterizing the tilt and seasonal              USGS Integrated Software for Imagers and Spectrometers
dependence of Titanʼs atmosphere is an important step toward              (ISIS; Sucharski et al. 2020) has convenient methods for
more accurate models and a more complete understanding of                 navigating SPICE kernels and generating image ﬁles with the
Titanʼs atmospheric dynamics. Thus, this work aims to better              geometric information needed for image analysis. While the
constrain the characteristics of the tilt and of the NSA and other        position and angle of the camera with respect to Titan was
atmospheric features visible at times throughout the Cassini              generally accurate enough for the analysis performed here, the
mission using images that span roughly 14 yr, or nearly half of           center pointing was often visibly inaccurate, needing adjust-
a Titan year. The expectation is that this careful analysis of the        ments of a few to several pixels for most images. Similarly to
orientation of these features presumably advecting with the               Roman et al. (2009), the center-pointing adjustments were
stratospheric winds can help constrain models of Titan’s                  determined by ﬁtting a circle to the disk formed by Titan’s
atmospheric dynamics. Additionally, this analysis expands on              outermost haze layers. This was based on the ﬁndings from
ideas and ﬁndings of previous research to provide a more                  Achterberg et al. (2008b) that atmospheric rotation decays
comprehensive description of the seasonal variability and the             above 250 km and is very slow near altitudes of 500 km,
semiannual hemispheric albedo reversal exhibited by Titanʼs               resulting in the atmosphere at this height taking on a nearly
atmosphere. This tracking of the temporal progression of these            spherical shape that is centered on the solid body. The lower
contrast features in Titan’s atmosphere should also serve as a            haze layers that are subjected to faster, variable rotation and
constraint on modeling (both theoretical and numerical) aimed             other dynamics result in a more complex, asymmetric oblate
at deciphering Titan’s atmospheric circulation. Such work is              shape without an easily identiﬁed center point. However, while
especially imperative as major upcoming missions focus in on              Roman et al. (2009) performed their ﬁt to the detached haze
Titan, and it may have important implications for other super-            layer near 500 km, which was clearly visible and relatively
rotating planetary atmospheres as well (e.g., Venus).                     invariable over the time period their study covered, later in the
                                                                          Cassini mission the detached haze layer was found to both
                                2. Methods                                decrease in altitude and become more oblate before disappear-
                  2.1. Observations and Processing                        ing into the main haze layer entirely in the years surrounding
                                                                          the equinox—it eventually reappeared in 2016 (West et al.
   The Cassini ISS collected thousands of images of Titan over
                                                                          2011, 2018; Seignovert et al. 2021). To work around this, we
the course of the mission. The ISS was made up of two
                                                                          allowed the circle radius to vary and ﬁt to the outermost visible
cameras, a wide-angle camera and a narrow-angle camera, and
                                                                          haze layer in each image. While this led to some images in the
used a variety of wavelength ﬁlters. For this work, images were
                                                                          latter half of the mission getting ﬁt at lower altitudes where the
chosen based on ﬁlter, spatial resolution, and viewing
                                                                          atmosphere is less spherical, these were typically still well
geometry. Only images from the 889 nm MT3 ﬁlter were
used, as the NSA boundary was most prominent at this                      above 250 km, and close enough to spherical to achieve a good
wavelength. We also required that the entire disk of Titan was            ﬁt. All ﬁts were also checked by eye to conﬁrm reasonable
visible in the image to allow for accurate navigation and                 results. Due to this variability and complex structure of Titan’s
calibration, and that the diameter spanned at least 100 pixels.           outermost haze layers, even after these corrections the center
Images were further constrained to those with phase angles                pointing is still a source of uncertainty of up to a few pixels,
below 70° to ensure the majority of the disk was visible. While           which is taken into account in our error bars later on.
images with phase angles greater than 30° and/or diameters                   Assuming Titan’s atmosphere at the altitudes of interest
less than 300 pixels had higher uncertainties in their                    exhibits a slightly oblate spheroid shape, using the corrected
measurements, we chose to include them to provide better                  pointing information, each pixel within the disk was assigned
temporal coverage, particularly later in the mission when more            its proper coordinates and incidence and emission angles.
suitable images were not always available. A small number of              Rather than mapping the image to a different projection, we
additional images were removed from the data set that                     chose to keep the data in image space to avoid the
contained other bodies or anomalies that interfered with the              complications associated with stretching data to different
analysis. A set of around 200 images remained that were                   resolutions.
suitable for our analysis, spanning 2004 to 2017 (see                        The polar and equatorial radii used for mapping were given
Appendix B for full listing and details of images used). For              by the radii of the solid body of Titan plus the estimated
more details about the Cassini instrumentation, see Porco                 altitude of the NSA feature. Due to spherical geometry and
et al. (2004).                                                            atmospheric opacity, data closer to the center of the disk probe
   Images were identiﬁed and downloaded from the Planetary                deeper in the atmosphere than data near the limb. At 889 nm,
Data System (PDS) using the PDS Ring-Moon Systems Node’s                  measurements at the limb only reach to around 200 km altitude,
OPUS search service.3 The images had already been calibrated              while methane absorption in the lower altitudes of Titanʼs
by the Cassini team using CISSCAL software, which removed                 atmosphere means that any bright features near the center of the
many of the imaging artifacts and bad pixels (Knowles et al.              disk stem from haze above 60 km (Lorenz et al. 2001). The
2020). Signiﬁcant noise and both horizontal and vertical                  altitude of the NSA boundary itself is not well constrained and
                                                                          spans a range or ranges of altitudes; however, it was necessary
                                                                          to map the features to a discrete altitude for this analysis. Thus,
3
    http://pds-rings.seti.org/search/                                     similar to Roman et al. (2009), a Monte Carlo–like approach

                                                                      2
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                         Snell & Banﬁeld

                                                                               Roman et al. (2009), resulting in more data points with less
                                                                               uncertainty. See Figure 1 depicting brightness based on
                                                                               incidence and emission angle, and Figure 2 for an example
                                                                               of a calibrated image. See Appendix A for a table of the
                                                                               resulting values.

                                                                                                        2.2. Analysis
                                                                                  We needed a method to identify the location of the NSA
                                                                               boundary that was robust enough to handle the variety of image
                                                                               geometries, variations in the boundaryʼs shape within and
                                                                               between images, and other changes and features that developed
                                                                               over time. When methods previously used by Roman et al.
                                                                               (2009) yielded unsatisfactory results, particularly for images
                                                                               later in the mission, edge-detection and computer-vision
                                                                               techniques were used to develop a more sophisticated
                                                                               algorithm. The method we settled on resembles a specialized
                                                                               Canny edge detector, following a similar multistep process of
                                                                               smoothing, ﬁnding the intensity gradient of the image,
Figure 1. The empirically determined photometric model for Titan in the        thresholding/identifying possible edges, and edge tracking.
889 nm methane band ﬁlter, with I/F shown as a function of incidence and          In addition to the noise removal described in the calibration
emission angles.
                                                                               steps, a low-pass blurring ﬁlter (which calculated mean pixel
                                                                               values with a 5 × 5 to 11 × 11 kernel, depending on image
was used to generate results from around 50 to 200 km above                    resolution) was applied to reduce any remaining noise,
the surface to cover this range of most likely altitudes. The                  artifacts, and smaller atmospheric features that were not of
variance in the results was later used to inform our error bars.               interest. After smoothing, the value of the intensity gradient
   Images of Titan in the methane bands exhibit signiﬁcant                     was determined at each usable pixel in the visible disk, where
limb brightening, which was found to interfere with our                        larger gradients indicate stronger albedo transitions. Since the
analysis of the atmospheric tilt. Thus, it was necessary to ﬂatten             boundaries we were interested in lie roughly along latitude
the brightness curve of the images as much as possible. While                  lines, a directional derivative of intensity was calculated for
some attempts have been made to describe this brightening as a                 every pixel in the local northward direction using a 3 ×
function of incidence and emission angles (e.g., Young et al.                  3 kernel, which weighted the surrounding pixels according to
2002), none of these yielded a ﬂat enough image for our                        the direction. This maximized the signal from the latitudinal
purposes, particularly very near the limb.                                     albedo variations, while any lingering brightness variations due
   By binning the pixels from images in our data set according                 to limb effects were suppressed since they ran roughly
to their corresponding incidence and emission angles, it was                   perpendicular to the direction vectors for most pixels in most
possible to determine the average brightness of a pixel at a                   image geometries.
given incidence and emission angle. To isolate the effects of                     The array of directional derivatives was limited to pixels
limb brightening/darkening from the albedo variations due to                   between −40° and 40° latitude, and divided longitudinally into
the NSA boundary and other atmospheric features, we selected                   100 slices of equal pixel area. The values of the derivatives in
images taken prior to the onset of the NSA reversal around                     each slice were then ﬁt to a degree-10 polynomial as a function
2014 and only included pixels north of the NSA boundary                        of latitude, where local maxima and minima were considered
region (latitudes > −5°). The albedo is relatively ﬂat north of                possible boundaries. The high degree of polynomial allowed
the NSA boundary in this time period. The selected images                      the several most signiﬁcant albedo transitions in each slice to
spanned nearly a decade and a variety of lighting geometries,                  be identiﬁed, and it provided a robust determination of the
which further reduced correlations between any speciﬁc                         approximate midpoint of each transition despite some variation
coordinates and incidence/emission angles (thus decoupling                     in the shape of the curves across the image. An example of this
albedo and limb brightening) when the pixels were binned and                   polynomial ﬁtting is shown at the bottom of Figure 3.
averaged. We noted that phase angle did not seem to have a                        With the latitudes of all possible major edges identiﬁed for
signiﬁcant impact on limb brightening over the limited range of                each slice, the ﬁnal step was to track these edge candidates
phase angles (0°–70°) included in our image set, so it was                     across the image. Each maxima and minima was sorted into a
sufﬁcient in this case to base the model on incidence and                      group with other maxima and minima, respectively, that were
emission angle only. The resulting brightness values were                      at similar latitudes in nearby slices. Any group with more than
tested on numerous images throughout the entire data set to                    90 related edge candidates (out of the 100 slices) was
ensure that this model consistently ﬂattened limb effects at all               considered a strong edge and used in the subsequent analysis.
phase angles included in our study, in both hemispheres, and in                Groups with less than 90 were thrown out as false edges or
images beyond 2014 when the NSA reversed. Some smoothing                       weak edges with too few points for accurate analysis. See
and manual ﬁne-tuning was performed as needed. This                            Figure 3 for an example of strong versus weak edge points in
empirical determination of brightness as a function of emission                an image.
and incidence angle enabled sufﬁcient ﬂattening of the images                     Finally, the location and orientation of the boundaries were
so that even subtle albedo variations near the limb could be                   determined by ﬁtting the points in each group to a plane. Least-
identiﬁed. Higher phase angles and more of each image                          squares ﬁtting was performed in two steps: the normal vector of
could be used here than in the previous similar study by                       the plane was determined ﬁrst, followed by the mean latitude.

                                                                           3
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                                                   Snell & Banﬁeld




                                                                                       Figure 3. Top: image N1496574260_1 after limb-brightening reduction. Middle:
                                                                                       the brightness gradient in the northward direction (north is to the lower right in this
                                                                                       image). The locations of gradient extrema, as determined from polynomial ﬁts, are
                                                                                       indicated on the images, where black points indicate extrema determined to belong
                                                                                       to a strong edge (used for subsequent analysis) and white points indicate extrema
Figure 2. Top: image W1536388236_1 after reduction of noise and banding                belonging to weaker edges (not used). Bottom: a few slices from the above image
artifacts in units of I/F. Only the pixels considered useful to the analysis are       gradient, demonstrating the degree-10 polynomial ﬁtting.
shown, i.e., parts of the disk with emission angle <85° and incidence angle
<87°. Middle: the surface brightness model (in I/F) produced when the                  This reduced the correlation between the tilt of the boundary
empirical photometric model is interpolated for the incidence and emission
angles of each pixel in the image. Bottom: the resulting ﬂattened image after          and the mean latitude, leading to more consistent results from
the calibrated image is divided by the brightness model.                               the ﬁtting functions.



                                                                                   4
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                       Snell & Banﬁeld

                       2.3. Error Analysis                                respect to the solid body rotation axis by roughly 4°, ﬂuctuating
                                                                          by a couple degrees apparently with the seasons. The azimuth
   Throughout the analysis, possible sources of error were
                                                                          angle of the tilt axis was roughly ﬁxed in inertial space, its
identiﬁed, mitigated, and accounted for in the reporting of the
                                                                          offset from the subsolar longitude moving westward through-
results. Uncertainties in center pointing and altitude of features,
                                                                          out the year. When plotted in an inertial reference frame, there
image artifacts and noise, physical variations, and ﬁtting errors
                                                                          is some evidence of oscillation on a period much shorter than
affected the accuracy of the calculations of atmospheric tilt. To
                                                                          the Titan year. The NSA boundary had a mean latitude around
address the many, sometimes complex sources of error, a
                                                                          9°S (when the northern hemisphere was brighter) before
Monte Carlo–like method was applied. The analysis was
                                                                          migrating southward and eventually being replaced by a new
performed repeatedly on each image while adjusting each of
                                                                          inverted boundary (at the time of this writing the southern
the image parameters over their ranges of expected uncertainty,
                                                                          hemisphere is brighter) in the northern hemisphere.
for a total of 180 runs per image. The resulting distribution of
                                                                             The tilt amplitude of the NSA boundary with respect to the
results for each image allowed us to determine the most likely
                                                                          solid body north pole versus time is shown in the upper panel
values for the atmospheric tilt and estimate the possible error.
                                                                          of Figure 4. The tilt amplitude started off around 5° and
   Both the center-pointing uncertainties and the unconstrained
                                                                          decreased to around 3° before the NSA reversal around 2014.
altitude of the NSA boundary feature contributed to possible
                                                                          The amplitude then seemed to increase before the original
mapping errors. While center-pointing inaccuracies were
                                                                          boundary faded away. The new boundary in the northern
reduced by ﬁtting to the spherical upper haze layer (as
                                                                          hemisphere had approximately a 3° amplitude. Uncertainty in
described in Section 2.1 and in Roman et al. 2009), uncertainty
                                                                          the measurements was generally 1°–2° for the ﬁrst half of the
of up to a few pixels remained in some images, particularly
                                                                          mission, but increased around the boundary transition period
those at higher phase angles where part of the disk was not
                                                                          (2012 and later) to 2°–3° for most measurements.
illuminated. The uncertain altitude of the features added further
                                                                             The tilt amplitude started with around a 70° offset from the
uncertainty to the mapping process. Both error sources become
                                                                          subsolar longitude in 2004. Throughout the course of the
more pronounced near the planet limb where the curvature of
                                                                          Cassini mission, the azimuth offset moved westward at nearly
the planet exacerbates any mapping errors, yet the points near
                                                                          the same rate as the progression of the solar longitude (Ls),
the limb are extremely valuable for constraining the NSA
                                                                          meaning the azimuth remains approximately ﬁxed in inertial
boundary orientation.
                                                                          space (see the bottom panel in Figure 4). Though generally
   The images also exhibited signiﬁcant noise and artifacts,
                                                                          following a linear trend with Ls, the results hint at some shorter-
speciﬁcally in the form of both vertical and horizontal banding,
                                                                          term ﬂuctuations, suggesting the atmospheric tilt axis may
which were strong enough to interfere with the NSA boundary
                                                                          oscillate around a ﬁxed point on timescales of less than a Titan
identiﬁcation process. As discussed above in Section 2.1, these
                                                                          year. Data at the end of the mission for the new southward
artifacts persisted after the initial CISSCAL calibration by the
                                                                          brightness boundary had high uncertainties for the azimuth
Cassini team (Knowles et al. 2020), so median ﬁlters and
                                                                          calculations due to smaller tilt amplitudes (in some cases
boxcar means were used to reduce them further. We aimed to
                                                                          approaching 180°), and it is unclear if the trend continues or if
suppress the noise and artifacts as much as possible while
                                                                          there is actually a 180° change. Similar ambiguity in azimuth
minimizing the loss of real planetary features, but this is a
                                                                          measurements from this time period appears in Kutsop et al.
lingering source of minor errors that we have accounted for.
                                                                          (2022). However, the small tilt amplitude here means that only
   We also noticed that throughout the mission, subtle “bumps”
                                                                          a slight perturbation of the tilt axis can drastically change the
in the NSA boundary could be observed, warping parts of the
                                                                          azimuth, so it is unlikely that the large variation in azimuth
boundary outside of the assumed ﬂat ring/plane. In addition,
                                                                          calculations indicates any real drastic phenomenon is
the width and gradient of the boundary varied spatially and
                                                                          occurring.
temporally, making it more complicated to consistently
                                                                             The mean latitude of the NSA boundary is shown in the middle
determine the center point of the boundary. The analysis
                                                                          panel of Figure 4. Another ﬁgure, Figure 5, displays the latitudinal
methods described in Section 2.2 were developed largely
                                                                          brightness over time to show more comprehensively the changes in
through trial and error to ensure a consistent and robust pipeline
                                                                          brightness across the whole visible disk. We found that the mean
that can handle such variations. However, some uncertainty is
                                                                          latitude remained fairly steady at 9°S until the boundary began to
still assumed, and is accounted for by injecting artiﬁcial
                                                                          migrate south starting around 2011 as the northern hemisphere
Gaussian noise into the boundary points in the various Monte
                                                                          began to darken. This original boundary faded away in 2016 at
Carlo runs.
                                                                          around 20°S when the southern hemisphere took over as the
   Lastly, there is some uncertainty in the ﬁtting functions used
                                                                          brighter hemisphere. The mean latitude measurements are
throughout the pipeline. However, these errors are more easily
                                                                          generally expected to be accurate to within a few degrees.
calculated, and thus were propagated through the analysis and
                                                                          Meanwhile, a new inverted boundary arose in the northern
incorporated into the ﬁnal uncertainty calculations.
                                                                          hemisphere as a dark cap extended southward and eventually
                                                                          encompassed most of the northern hemisphere, completing the
                                                                          NSA reversal. Figure 6 shows example images from before,
                            3. Results
                                                                          during, and after the NSA reversal. Numerous bands and
   We analyzed a few hundred Cassini ISS images distributed               brightness transitions can be seen in the middle image before
from 2004 to 2017 to determine the orientation of the NSA                 settling back into the north–south asymmetry pattern.
boundary and thus the tilt axis of the atmospheric circulation
more generally. At the 889 nm wavelength used here, the
                                                                                                    4. Discussion
northern hemisphere appeared brighter at the start of the data
set and reversed so that the southern hemisphere was brighter at             The purpose of this study was to analyze the tilt of the
the end. We found that the NSA boundary was tilted with                   rotation axis of Titan’s atmosphere, using the highest-

                                                                      5
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                                            Snell & Banﬁeld




Figure 4. The location and orientation of the NSA boundary changes over time and shows a seasonal dependence. Results are shown for both the initial major NSA
boundary where brightness increases in the northward direction (blue) and for the new reversed boundary that arises near the end of the Cassini mission (red). The top
panel shows the amplitude of the tilt with respect to the solid body north pole, the middle panel shows the mean latitude of the NSA boundary, and the bottom panel
shows the offset of the azimuth from the subsolar longitude. The dashed line in the bottom panel represents a best ﬁt to the azimuth offset if it was ﬁxed in inertial
space. Deviations from this line in the data may indicate that seasonal or other atmospheric phenomena affect the tilt azimuth.




Figure 5. The relative brightness by latitude after limb-brightening reduction is shown throughout the Cassini mission. Brightness data were interpolated through the
time gaps between images. This ﬁgure shows the NSA boundary move south and eventually fade, while a new reversed boundary arises slightly to the north. Due to
the image geometries used in this study, the data near the poles were often low quality or unavailable throughout the mission.



                                                                                  6
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                                     Snell & Banﬁeld

                                                                                        resolution imagery available (i.e., Cassini ISS), to gain further
                                                                                        insight into the atmospheric dynamics of Titan’s super-rotating
                                                                                        atmosphere. Additionally, by characterizing the location and
                                                                                        orientation of major banding features in Titan’s atmosphere
                                                                                        throughout the Cassini mission, we can determine the seasonal
                                                                                        variations of the features themselves and the atmosphere as a
                                                                                        whole. This enables a deeper understanding of the underlying
                                                                                        dynamics and provides further constraints for Titan atmo-
                                                                                        spheric models (both theoretical and numerical).
                                                                                           A seasonal dependence for the albedo asymmetry was
                                                                                        already known, as past observations revealed the reversal in
                                                                                        hemispheric brightness (e.g., Caldwell et al. 1992). However,
                                                                                        the seasonal dependence of the atmospheric tilt the asymmetry
                                                                                        reveals was less well constrained until a recent paper by Kutsop
                                                                                        et al. (2022). Our results are complementary to those of Kutsop
                                                                                        et al. (2022) in that we use an entirely different data set (Cassini
                                                                                        ISS versus VIMS) and independent methods. Our data set
                                                                                        included additional temporal coverage of the NSA, particularly
                                                                                        beyond 2012 covering the NSA reversal (measurements from
                                                                                        this time period in Kutsop et al. 2022 are primarily from the
                                                                                        north polar annulus rather than the equatorial annulus/NSA
                                                                                        boundary), and the spatial resolution of ISS images generally
                                                                                        exceeds that available in VIMS. Along with the additional steps
                                                                                        taken here during image processing and analysis to robustly
                                                                                        and precisely navigate features and their tilts, our analysis may
                                                                                        prove to be the most precise assessment of the atmospheric tilt
                                                                                        and its seasonal behavior. Overall, our results are generally in
                                                                                        accord with those of Kutsop et al. (2022), and show that the
                                                                                        azimuthal angle of the atmospheric rotation axis is roughly
                                                                                        ﬁxed in inertial space (perhaps with some slight ﬂuctuations
                                                                                        around this angle on timescales shorter than seasonal), and
                                                                                        there is a seasonal ﬂuctuation in the amplitude of the tilt angle.
                                                                                           In the top panel of Figure 4, we see the amplitude decrease
                                                                                        from around 5° to around 3°–4° near the time the NSA
                                                                                        reverses. This is in agreement with the results from the Tokano
                                                                                        (2010) GCM, which showed a similar decrease in amplitude
                                                                                        before a more rapid increase back to the previous amplitude
                                                                                        when the asymmetry reversed. This increase seems to be
                                                                                        exhibited by the original northward brightness gradient
                                                                                        boundary, but not by the new boundary that arises. However,
                                                                                        the measurements of the new boundary are quite noisy and
                                                                                        uncertain due to the more difﬁcult viewing geometries and
                                                                                        somewhat fuzzier boundary toward the end of the mission, so
                                                                                        such a trend would not necessarily fall outside our error bars. It
                                                                                        is also possible that the more drastic circulation changes
                                                                                        occurring around the transition period perturbed the boundary
                                                                                        to some extent.
                                                                                           The bottom panel of Figure 4 shows the azimuth angle offset
                                                                                        from the subsolar longitude. Throughout the mission, we see
                                                                                        this angle increase roughly along with the solar longitude (Ls),
                                                                                        indicating the atmosphere was tilted at nearly the same angle in
                                                                                        inertial space throughout the mission. This result is also generally
                                                                                        in agreement with previous results such as Achterberg et al.
                                                                                        (2011) and Kutsop et al. (2022), which similarly found that the
                                                                                        azimuth angle is roughly ﬁxed in inertial space, and Sharkey
                                                                                        et al. (2020), which found that assuming the tilt was ﬁxed in
Figure 6. Sample images from before (N1567440863_1), during                             inertial space resulted in less zonal asymmetry in thermal
(N1786775340_1), and after (N1852202278_1) the NSA reversal show                        emissions. Kutsop et al. (2022) also noted slight oscillations of
how the NSA evolves over time. The north angle for each image is oriented
toward the top of the ﬁgure (though the actual north pole may point somewhat            the tilt axis and offered additional analysis and interpretations of
into/out of the page), and the tilt in the NSA boundary is clearly visible in the       some ﬁne-grained trends. We did not deem such a ﬁne estimation
top panel.                                                                              of oscillations feasible with the noise and uncertainty in our data,


                                                                                    7
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                    Snell & Banﬁeld

so will only compare the coarsest trends. However, while we              the atmosphere. Finally, there are additional atmospheric
cannot comment speciﬁcally on any ﬁner trends, we do see what            features (i.e., the polar annuli discussed in Kutsop et al.
appear to be slight, perhaps sinusoidal ﬂuctuations in the azimuth       2022) that were not included in this work but are visible in
offset; note the data points in the bottom panel seemingly               many Cassini ISS images that could also be analyzed to offer
alternating between falling above and below a ﬁxed azimuth               more data points and compare atmospheric behavior at
offset, represented by the dashed line. Lastly, while Roman et al.       differing latitudes.
(2009) and Achterberg et al. (2008a) did not have enough
temporal coverage to determine any seasonal dependence, the                                      5. Conclusions
average values they calculated for amplitude and azimuth angle
offset support these results, as well.                                      To accurately describe and model the dynamics and
   Our observation of the NSA reversal is also in agreement              circulation of Titan’s atmosphere, observational data collected
with past observations (e.g., from Voyager and Hubble;                   over seasonal (or longer) timescales is an essential source of
Caldwell et al. 1992), where the same seasonal dependence                insight and model constraints. Here, we analyzed Cassini ISS
was noted. New to this study is the more comprehensive                   data from approximately half of a Titan year to reveal both
analysis of what happens in between, at least from the point of          qualitative and quantitative seasonal trends in the appearance
view of the 889 nm methane band. Here, we saw how the                    and orientation of Titan’s atmosphere. We extended and
original NSA boundary seemed to migrate south before fading              improved upon previous work—ﬁrst through the use of the
away, while the new boundary gradually strengthened and                  now-complete Cassini data set, and additionally through more
moved equatorward from the north. Two major boundaries                   robust image-processing and feature-detection techniques. We
coexisted brieﬂy, along with a number of other more subtle               developed an empirical photometric model to describe the
bands.                                                                   unusual limb brightening exhibited by Titan in the 889 nm
   At times throughout the mission, the NSA boundary                     ﬁlter, which increased the amount of usable data near the limb.
exhibited features that may be attributable to some weather              The NSA boundary was identiﬁed in a more robust way using
phenomena affecting the region. Such features were more                  image smoothing, directional gradients, and connecting edge
clearly evident when viewing the gradient of the image, and              candidates to capture the overall NSA boundary regardless of
consisted of a partially split NSA boundary (i.e., two discrete          minor variations in its characteristics and without false
                                                                         detections of smaller features.
latitudinal brightness changes) and bumps or waves where the
                                                                            We found that the vector normal to the NSA boundary, and
boundary deviated signiﬁcantly from a ﬂat ring. While such
                                                                         thus the rotation axis of Titan’s atmospheric mean zonal ﬂow,
features complicated our primary analysis somewhat as they
                                                                         is offset from the poles of the solid body by 4° on average. The
deviated from our expected model of the boundary, they do
                                                                         azimuth of the atmosphere’s axis is roughly ﬁxed in inertial
suggest more interesting and complicated dynamics are
                                                                         space, its offset from the subsolar longitude increasing at about
occurring at smaller scales and may offer an opportunity for             the same rate as Ls, or alongside the progression of Titan’s orbit
future deeper analysis. The phenomena occurred both during               and seasons.
and outside of the NSA reversal transition period, but during
the nontransition times such phenomena were much more
isolated and distinguishable from their more predictable                                       Acknowledgments
surroundings.                                                              C.S. is supported by the National Aeronautics and Space
   Cassini provided data for around half of a Saturn year. While         Administration (NASA) FINESST grant No. 80NSSC21K1538.
this gave us some of the best temporal coverage applied to this            Facility: Cassini(ISS).
problem so far and enabled some analysis of seasonal trends, it            Software: SciPy (Virtanen et al. 2020), USGS Integrated
would be useful to extend the coverage further to cover the              Software for Imagers and Spectrometers (ISIS) (Sucharski et al.
entire Titan year and beyond. This could potentially be done             2020).
using existing data from Voyager or Hubble, or perhaps with
future observations. The lack of usable images around 2014
was also particularly unfortunate since this was a key period                                    Appendix A
during the NSA reversal. A set of data with more complete                                Empirical Photometric Model
coverage of the transition period could provide a more                      In Table 1, we present the results of our empirically
complete look at the changes the atmosphere experiences.                 determined photometric model in I/F for every 10° of
   Additionally, only images from the 889 nm ﬁlter were used,            incidence and emission angles. To apply this model to images,
and the NSA boundary was assumed to have arisen in a certain             intermediate values can be calculated as needed using cubic
region of the atmosphere. The NSA is present in a range of               interpolation. To improve the efﬁciency of our pipeline given
wavelengths from IR to UV. We know different wavelengths                 the size of our data set, we used cubic interpolation to produce
reveal different altitudes in Titan’s atmosphere, and the NSA            an array of values at 1° intervals, then used linear interpolation
itself is wavelength dependent, with the bright and dark                 to approximate the brightness values for the particular
hemispheres inverting at wavelengths below 440 nm. Analyz-               incidence and emission angles at each relevant pixel in our
ing images taken through different ﬁlters would provide                  images. We found this was sufﬁcient to produce smooth
additional data points and probe different regions/features of           results.




                                                                     8
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                                               Snell & Banﬁeld

                                                                              Table 1
                                                          Empirical Photometric Model for Titan at 889 nm

                                                                                             Incidence Angle
                                0°            10°           20°              30°            40°           50°       60°         70°              80°             90°
              0°            0.050820       0.050700      0.050380         0.049750        0.048520     0.045820   0.041100    0.033750         0.023450       0.012490
              10°           0.051695       0.051570      0.051241         0.050551        0.049220     0.046570   0.041750    0.034170         0.023800       0.012600
              20°           0.053970       0.053810      0.053460         0.052711        0.051301     0.048570   0.043580    0.035660         0.024760       0.012900
              30°           0.057740       0.057540      0.057158         0.056340        0.054780     0.051914   0.046700    0.038420         0.026720       0.013600
Emission      40°           0.063200       0.062930      0.062441         0.061505        0.059827     0.056903   0.051534    0.042700         0.030000       0.015130
Angle         50°           0.070320       0.069970      0.069370         0.068310        0.066627     0.063733   0.058241    0.048754         0.034980       0.017870
              60°           0.079000       0.078600      0.077900         0.076920        0.075301     0.072665   0.067100    0.057100         0.042030       0.022350
              70°           0.087500       0.087100      0.086400         0.085470        0.084200     0.081940   0.077100    0.067316         0.051600       0.029449
              80°           0.090900       0.090700      0.090340         0.089735        0.088930     0.087300   0.083905    0.076577         0.062510       0.040207
              90°           0.090340       0.090290      0.090190         0.089930        0.089390     0.088310   0.086000    0.081200         0.071500       0.055500


                        Appendix B
      Image Details and NSA Boundary Measurements
   Table 2 lists all images used in our analysis along with                                the direction of the brightness gradient that was measured
relevant image details and our measurements of the location                                (northward or southward) as indicated by the subheadings
and orientation of the NSA boundary. The data are grouped by                               within the table.


                                                                                   Table 2
                                                                      Details for All Analyzed Images

Image ID                           Time                   Solar Lon.             Tilt                 NSA             Tilt         Subsolar                Offset from
                                  (UTC)                       Ls               Amplitude             Mean Lat.      Azimuth          Lon.                 Subsolar Lon.
                                                            (deg)               (deg)                 (deg)         (deg W)        (deg W)                  (deg W)
                                                                      Northward Brightness Gradient

N1477221680_2             2004-10-23T10:55:19               297.55                 5.22               −7.19         126.11            87.71                  38.39
N1477225220_2             2004-10-23T11:54:19               297.55                 5.24               −7.20         127.67            88.64                  39.03
N1477228760_2             2004-10-23T12:53:19               297.55                 4.59               −7.88         132.49            89.56                  42.93
N1477232300_2             2004-10-23T13:52:19               297.56                 4.95               −7.93         136.37            90.48                  45.88
N1477235840_2             2004-10-23T14:51:19               297.56                 3.61               −9.43         153.69            91.41                  62.29
L                                  L                          L                     L                  L              L                L                      L

                                                                      Southward Brightness Gradient

N1837244963_1             2016-03-21T08:44:08               77.44                  3.58                17.09        19.07              68.24                 310.83
N1838136389_1             2016-03-31T16:21:03               77.67                  1.75                16.64        354.60            300.92                 53.68
N1838245290_1             2016-04-01T22:36:03               77.73                  2.63                13.59        145.30            329.32                 175.99
N1839513938_1             2016-04-16T15:00:09               78.16                  5.09                21.33         9.08             300.35                 68.73
N1840893046_1             2016-05-02T14:05:03               78.64                  4.05                12.57        129.67            300.18                 189.49
L                                  L                         L                      L                   L             L                 L                      L

(This table is available in its entirety in machine-readable form.)




                                                                                     9
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
The Planetary Science Journal, 5:12 (10pp), 2024 January                                                                                            Snell & Banﬁeld

                                ORCID iDs                                                Porco, C. C., West, R. A., Squyres, S., et al. 2004, SSRv, 115, 363
                                                                                         Roman, M. T., West, R. A., Banﬁeld, D. J., et al. 2009, Icar, 203, 242
C. Snell https://orcid.org/0000-0003-3870-2369                                           Seignovert, B., Rannou, P., West, R. A., & Vinatier, S. 2021, ApJ, 907, 36
D. Banﬁeld https://orcid.org/0000-0003-2664-0164                                         Sharkey, J., Teanby, N. A., Sylvestre, M., et al. 2020, Icar, 337, 113441
                                                                                         Smith, B. A., Soderblom, L., Batson, R., et al. 1982, Sci, 215, 504
                                 References                                              Smith, B. A., Soderblom, L., Beebe, R., et al. 1981, Sci, 212, 163
                                                                                         Sromovsky, L. A., Suomi, V. E., Pollack, J. B., et al. 1981, Natur,
Achterberg, R. K., Conrath, B. J., Gierasch, P. J., Flasar, F. M., & Nixon, C. A.           292, 698
   2008a, Icar, 197, 549                                                                 Sucharski, T., Mapel, J., jcwbacker, et al. 2020, USGS-Astrogeology/ISIS3:
Achterberg, R. K., Conrath, B. J., Gierasch, P. J., Flasar, F. M., & Nixon, C. A.           ISIS 4.2.0 Public Release, v4.2.0, Zenodo, doi:10.5281/zenodo.3962369
   2008b, Icar, 194, 263                                                                 Teanby, N., Irwin, P., & de Kok, R. 2010, P&SS, 58, 792
Achterberg, R. K., Gierasch, P. J., Conrath, B. J., Michael Flasar, F., &                Tokano, T. 2010, P&SS, 58, 814
   Nixon, C. A. 2011, Icar, 211, 686                                                     Tomasko, M. G., & Smith, P. H. 1982, Icar, 51, 65
Caldwell, J., Cunningham, C. C., Anthony, D., et al. 1992, Icar, 97, 1                   Virtanen, P., Gommers, R., Oliphant, T. E., et al. 2020, NatMe, 17, 261
Flasar, F., & Achterberg, R. 2009, RSPTA, 367, 649                                       West, R., Del Genio, A., Barbara, J., et al. 2016, Icar, 270, 399
Knowles, B., West, R., Helfenstein, P., et al. 2020, P&SS, 185, 104898                   West, R. A., Balloch, J., Dumont, P., et al. 2011, GeoRL, 38, L06204
Kutsop, N. W., Hayes, A. G., Corlies, P. M., et al. 2022, PSJ, 3, 114                    West, R. A., Seignovert, B., Rannou, P., et al. 2018, NatAs, 2, 495
Lockwood, G., & Thompson, D. 2009, Icar, 200, 616                                        Young, E. F., Rannou, P., McKay, C. P., Grifﬁth, C. A., & Noll, K. 2002, AJ,
Lorenz, R. D., Smith, P. H., Lemmon, M. T., et al. 1997, Icar, 127, 173                     123, 3473
Lorenz, R. D., Young, E. F., & Lemmon, M. T. 2001, GeoRL, 28, 4453                       Yung, Y. L., Allen, M., & Pinto, J. P. 1984, ApJS, 55, 465




                                                                                    10
```
