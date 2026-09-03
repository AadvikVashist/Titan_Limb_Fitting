---
citation_key: "vixie2012mapping"
title: "Mapping Titan's surface features within the visible spectrum via Cassini VIMS"
source_pdf: "data/papers/vixie2012mapping.pdf"
source_pdf_sha256: "ee780fbfbefed42624a3ef5f7aebce2daa0d77c897f4c282a706acfcadb18305"
conversion_tool: "pdftotext -layout; tesseract OCR fallback"
ocr_pages: 0
page_marker_scheme: "PDF_PAGE and PDF page heading"
---

This is a searchable working copy. Use the linked PDF as the source of record and check it before quoting.

<!-- PDF_PAGE: 1 -->

## PDF page 1

```text
                                                                   Planetary and Space Science 60 (2012) 52–61



                                                              Contents lists available at ScienceDirect


                                                        Planetary and Space Science
                                                     journal homepage: www.elsevier.com/locate/pss




Mapping Titan’s surface features within the visible spectrum via Cassini VIMS
Graham Vixie a,, Jason W. Barnes a, Jacob Bow a, Stéphane Le Mouélic b, Sébastien Rodriguez c,
Robert H. Brown d, Priscilla Cerroni e, Federico Tosi f, Bonnie Buratti g, Christophe Sotin g,
Gianrico Filacchione e, Fabrizio Capaccioni e, Angioletta Coradini f
a
  University of Idaho, Department of Physics, P.O. Box 440903, Moscow, ID 83844-0903, United States
b
   Laboratorie de Planétologie et Géodynamique, CNRS UMR 6112, 2 rue de la Houssinie re, Université de Nantes, 44300 Nantes, France
c
  Laboratoire AIM, Université Paris Diderot - Paris 7/CNRS/CEA-Saclay, DSM/IRFU/SAp, Gif sur Yvette, France
d
   University of Arizona, Lunar and Planetary Laboratory, 1629 East University Blvd, Tucson, AZ 85721-0092, United States
e
  Istituto di Astroﬁsica Spaziale e Fisica Cosmica, Sezione di Roma, Via Fosso del Cavaliere 100, Tor Vergata, IT 00133 Roma, Italy
f
  Istituto di Fisica dello Spazio Interplanetario, Via Fosso del Cavaliere 100, IT 00133 Roma, Italy
g
   California Institute of Technology/Jet Propulsion Laboratory, 4800 Oak Grove Drive, Pasadena, CA 91109, United States




a r t i c l e i n f o                                   a b s t r a c t

Article history:                                        Titan shows its surface through many methane windows in the 1–5 mm region. Windows at shorter
Received 11 September 2010                              wavelengths also exist, polluted by scattering off of atmospheric haze that reduces the surface contrast.
Received in revised form                                At visible wavelengths, the surface of Titan has been observed by Voyager I, the Hubble Space
25 March 2011
                                                        Telescope, and ground-based telescopes. We present here global surface mapping of Titan using the
Accepted 29 March 2011
Available online 8 April 2011
                                                        visible wavelength channels from Cassini’s Visual and Infrared Mapping Spectrometer (VIMS). We show
                                                        global maps in each of the VIMS-V channels extending from 0.35 to 1.05 mm. We ﬁnd methane
Keywords:                                               windows at 0.637, 0.681, 0.754, 0.827, 0.937, and 1:046 mm and apply an RGB color scheme to the
Titan                                                   0.754, 0.827 and 0:937 mm windows to search for surface albedo variations. Our results show that Titan
Surface
                                                        appears gray at visible wavelengths; hence scattering albedo is a good approximation of the Bond
Optical wavelength
                                                        albedo. Maps of this genre have already been made and published using the infrared channels of VIMS.
Visible imaging
Cassini VIMS                                            Ours are the ﬁrst global maps of Titan shortward of 0:938 mm. We compare the older IR maps to the
                                                        new VIMS-V maps to constrain surface composition. For instance Tui Regio and Hotei Regio, referred to
                                                        as 5-mm bright spots in previous papers, do not distinguish themselves at all visible wavelengths. The
                                                        distinction between the dune areas and the bright albedo spots, however, such as the difference
                                                        between Xanadu and Senkyo, is easily discernible. We employ an empirically derived algorithm to
                                                        remove haze layers from Titan, revealing a better look at the surface contrast.
                                                                                                                        & 2011 Elsevier Ltd. All rights reserved.




1. Introduction                                                                            (1991). Titan’s surface was ﬁrst seen in optical wavelengths by
                                                                                           Voyager I as discussed by Richardson et al. (2004) using wave-
    We present here ﬁndings from the Cassini spacecraft, which is                          lengths ranging from 590 to 640 nm, and by Smith et al. (1996)
on its second extended mission, via the Visual and Infrared                                using the window at 0:94 mm. The Hubble Space Telescope (HST)
Mapping Spectrometer (VIMS) instrument (Brown et al., 2004).                               also observed Titan’s surface at 673 nm (Smith et al., 1996).
Most of Titan’s surface has already been seen from RADAR and                               Relative brightness maps of Titan were produced from the HST
infrared wavelengths, but surface features have been sparsely                              data by Smith et al. (1996) and with Cassini’s Imaging Science
identiﬁed in the visible spectrum. Methane windows do exist in                             Subsystem (ISS) by Turtle et al. (2009, 2011).
the visible, albeit not as transparently as in the IR; using them,                            The VIMS visible channel measures optical spectra from 0.351
new constraints can be placed on the composition of Titan’s                                to 1:05 mm split up over 96 channels using a slit-scanning visible
surface.                                                                                   spectrometer (Capaccioni et al., 1998). We can then peruse each
    The ability to view Titan’s surface was pioneered near the                             wavelength channel to identify which wavelengths can pierce
beginning of the 1990s with the discovery of near-infrared                                 through Titan’s atmosphere to the surface. This has been done in
methane windows by McKay et al. (1989) and Grifﬁth et al.                                  infrared channels by Barnes et al. (2007), but comprehensive
                                                                                           global maps have not been previously produced at wavelengths
                                                                                           shortward of 0:938 mm. Spectra of Titan in the visible range were
     Corresponding author. Tel.: þ 1 509 6693398.                                         published in Neff et al. (1984) and Lockwood et al. (1986) where
     E-mail address: gvixie@vandals.uidaho.edu (G. Vixie).                                 each paper suggested visible windows. Local maxima in the

0032-0633/$ - see front matter & 2011 Elsevier Ltd. All rights reserved.
doi:10.1016/j.pss.2011.03.021
```

<!-- PDF_PAGE: 2 -->

## PDF page 2

```text
                                             G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61                                       53


spectrum longward of 600nm represent the window wavelengths                   global maps of Titan have already been published in the 0:937 mm
(although their utility in sensing the surface was not realized at            wavelength courtesy of ISS. Finally, the more wavelength we can
the time), and higher signal-to-noise spectra are presented and               incorporate in to our map while still maintaining surface albedo
discussed by Karkoschka (1994). The work that we describe here                variability, the greater coverage we have of the spectrum on Titan.
is the ﬁrst to show Titan’s surface and monitor some heterogene-                 We can identify many familiar landmarks in Fig. 3 such as
ities shortward of 0:9 mm.                                                    Shangri-La, Xanadu, and the sideways ‘‘H’’ of Fensal and Aztlan.
    The goal of this study is to constrain surface composition by             Five-micron-bright regions like Tui Regio (Barnes et al., 2006) and
generating and analyzing a global map of Titan using methane                  Hotei Regio (Soderblom et al., 2009; Barnes et al., 2005) do not
windows in the optical spectrum. In Section 2 we produce a map                show strong contrast in the visible map like in the IR maps, as
using data from ﬁve of Cassini’s ﬂybys: T8, T9, T31, T34, and rev9.           shown in Fig. 4, nor do the Selk or Sinlap craters. These regions
For more information on the observations on these ﬂybys, see                  are all gray at shorter wavelengths. While the visible color is
Barnes et al. (2009). These ﬂybys were chosen to minimize                     orange, for clariﬁcation, we use the word ‘‘gray’’ to mean that the
variance in phase on the global map while maximizing area                     spectral response is ﬂat and the reﬂectivity does not vary
covered on Titan. We then combine the resulting data together                 signiﬁcantly with wavelength in the visible part of the spectrum.
at each wavelength to show surface features on Titan. The end                    The global views of Titan (orthographic projections) give an
product is a global scale, atmospherically uncorrected, 25-color              undistorted look at surface features (Fig. 3 bottom). The dark lines
map. The process of calibrating each individual cube of data and              that appear on the global maps are seams between the different
making the maps is outlined in detail in Barnes et al. (2007). The            ﬂybys. The seams are present mostly due to differing phase angles
purpose of these maps is to provide another comparative point to              between ﬂybys, limb-darkening, and not being able to fully
the IR maps in hopes of further constraining surface composition              correct atmospheric effects. The solar incidence and emission
and more accurately deﬁning the surface features, which we do in              angles change over the course of a single ﬂyby as well. Fig. 5
Sections 3–5.                                                                 shows the change in these angles and pinpoints the locations of
                                                                              nadir points. At different phase angles, the ﬂyby appears darker,
                                                                              lighter, smoother, or more noisy. The effect of changing phase
2. Processing of imaging observations                                         angles will be discussed in Section 3. Special attention was paid to
                                                                              weaving the different ﬂybys together, as we want the map to look
2.1. Mapping                                                                  as seamless as possible. The ﬁgure of merit used by Barnes et al.
                                                                              (2007) was changed to solely rely on the emission angle in an
    Titan’s surface is obscured by scattering and absorption by               effort to minimize the seams between ﬂybys.
both haze and methane in the atmosphere in optical wavelengths.                  Noise here is not instrument noise, but rather stripe noise in
At most infrared wavelengths, Titan’s atmospheric methane                     the VIMS-V instrument. At the end of each row of pixels in the slit
absorbs heavily, making the surface impossible to see. We use                 scanner, there is a column shielded that measures dark current.
Cassini VIMS to scan Titan at all wavelengths to ﬁnd certain                  The background measured by this dark stripe is actually only
windows where incident solar ﬂux can pass through nearly                      measured by a single pixel (the rest of the measurements is
unhindered and return a signal from the surface. We focus on                  thrown away) then subtracted from all the other pixels. The
certain optical wavelength windows where photons are not                      problem here is sometimes a photon will get in and be measured
absorbed the atmosphere.                                                      by this pixel giving a higher value to the background. This is also
    We start the map making process by selecting all observations             the origin of the striping on the maps. The noise then does not
where surface features are apparent. Next, we wrote software to               come from the image or the haze, but rather the instrument itself.
place all data cubes from these observations on a single cylin-                  Atmospheric scattering is what drives the different appearance
drically projected map based on each cube’s embedded latitude                 as a function of observation geometry. As light from the Sun
and longitude information. We then found certain wavelengths                  enters Titan’s atmosphere, many possible paths emerge after the
where the surface of Titan becomes apparent by looking at these               light encounters atmospheric haze particles. The light may
combined observations at every available wavelength. Fig. 1                   (ideally) reﬂect from the surface and go directly to Cassini; more
shows Titan at all the visible wavelength channels on the VIMS                probably, however, the photons collides with haze particles and
instrument. The three wavelengths showing the most surface                    scatters. This greatly decreases the fraction of photons that
features from observation are 0.754, 0.827, and 0:937 mm. The                 successfully travel to Titan’s surface and back. Part of this indirect
ﬁnal step was then to create a color composite by assigning each              light, backscattered to the spacecraft by Titan’s haze without
of these wavelengths a color and optimizing contrast in order to              reaching the surface, causes additive offsets that must be cor-
show surface variations. We assign the colors blue, green, and red,           rected (visible – Perry et al., 2005; IR – Rodriguez et al., 2006;
respectively, to deﬁne the RGB map. Fig. 2 shows all identiﬁed                Le Mouélic et al., 2008, 2010).
optical wavelength methane windows at constant contrast. The
ﬂybys and individual observations were chosen to maximize
longitudinal coverage and be close in phase. Information regard-              2.2. Haze removal and comparison
ing each ﬂyby used may be found in Table 1. We present then our
global scale, atmospherically uncorrected RGB surface reﬂectivity                Owing to high optical depth, spherical geometry, and uncer-
map of Titan in Fig. 3. This map can then be compared to other                tainties in scattering properties, it is not presently possible to
known IR maps (see Fig. 4) to identify surface characteristics.               properly correct for the atmospheric conversion of incident
    The 0:937 mm reveals the most surface features out of the                 solar ﬂux (I/F) to albedo using radiative transfer. The work of
visible channels since the atmosphere absorbs the least at this               Perry et al. (2005) on ISS data allows us to further our mapping
wavelength. However, we use two other channels for several                    algorithms by empirically compensating for some of the haze to
reasons. First, we are comparing the surface at different wave-               enhance surface contrast. The purpose of this article is present
lengths and therefore wish to identify any anomaly that is limited            maps, and thus further analysis of atmospheric or haze correc-
to a certain wavelength(s). This is achieved through the creation             tions is beyond the scope of this paper. We did try simple
of the color composite. Features that show more strongly in one               subtraction and ratio based techniques, but our best results came
(or two) of our primary colors are easier to constrain. Second,               using the ISS techniques directly.
```

<!-- PDF_PAGE: 3 -->

## PDF page 3

```text
54                                                    G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61




Fig. 1. The top left window represents the beginning of the wavelength range of Cassini VIMS’ 96 optical wavelength channels, 0:35 mm, and wavelength increases for each
row. A quick look over all of the different wavelengths in the ﬁgure above reveal that there are few that show the surface. Methane has fewer windows in the optical
spectrum than it does in the infrared, and scattering is more prevalent, but we can still see distinctions on Titan’s surface.


   Using the I/F maps in Fig. 1 as our starting point, we created a                    yields lower-quality results. By compensating for wavelengths
haze-only map using nine wavelengths where our signal is almost                        where our signal is nearly totally absorbed, we make surface
completely absorbed and that showed no surface features what-                          features more visible. The resulting map has slightly more
soever. The wavelengths used to make the haze map are 0.585,                           striping, but accounts for phase angle as to reduce the visibility
0.599, 0.607, 0.622, 0.651, 0.659, 0.666, 0.709, and 0:878 mm.                         of the seams between ﬂybys. Most importantly, the new map
Following the Perry et al. (2005) algorithm, we divided the                            shows an improved view of the surface features of Titan as well as
original map (Fig. 3) at each of the methane window wavelengths                        reveals some new areas that are barely or not at all visible
labeled in Fig. 2a by the haze map to create a new, empirically                        without correction. Comparing Figs. 3 and 6 show that the dark
haze-corrected map (Fig. 2b). Using the same color/wavelength                          areas on Titan cover more area with some new areas appearing
setup as the original, we created a new global RGB map (Fig. 6).                       south of Shangri-La in the west, through the north of Tsegihi, and
Trial and error by Perry et al. (2005) shows that a subtraction step                   south of Belet in the east. The improved contrast of the map in
```

<!-- PDF_PAGE: 4 -->

## PDF page 4

```text
                                                         G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61                                                      55




Fig. 2. On the left are I/F maps of Titan without atmospheric correction. Wavelength and contrast are listed in the top left and top right, respectively, of each image. On the
right are corrected maps, with the haze divided out according to the Perry et al. (2005) empirical algorithm described in the text. Contrast is constant in the left set of
images and set to maximize surface feature distinction in the right.


Table 1
This table summarizes the Cassini observations used in this paper.

  Flyby             Rev number                Date                            Subsolar point              No. of cubes              Phase angle              Best spatial
                                                                                                                                                             sampling (km)

  N/A               rev9                      2005 June 6                     21.11S 1451W                 6                        711                      109
  T8                rev17                     2005 October 28                 19.61S 1391W                18                        211                      42
  T9                rev19                     2005 December 26                18.91S 451W                 25                        351                      2.9
  T31               rev45                     2007 May 28                     12.01S 1561E                11                        221                      11
  T34               rev48                     2007 July 19                    11.31S 781E                  8                        381                      8

  Ta                revA                      2004 October 26                 23.21S 1651W                  1                        91                      2.6
  T3                rev3                      2005 February 15                22.21S 1551W                  1                       201                      7
  T10               rev20                     2006 January 15                 18.71S 1271W                  1                       281                      3.9
  T12               rev22                     2006 March 18                   17.91S 961W                   1                       531                      7



Fig. 6 makes the empirically corrected visible maps much more                             shows the precise outlines of methane windows at 0.637, 0.681,
useful than the uncorrected map (Fig. 3).                                                 0.754, 0.827, 0.937, and 1:046 mm. A spectrum similar to this was
    Stephan et al. (2009) created a global albedo map of Titan                            presented in Lemmon et al. (2002) by subtracting a dark area
using data collected from 2004 to 2008 by the ISS via a narrow                            spectrum from a bright area spectrum and the visible methane
band pass ﬁlter set at 0:938 mm (Porco et al., 2004). The albedo                          windows are resolved, albeit with low I/F. Lemmon et al. (2002)’s
differences were described in Turtle et al. (2009) as being                               spectrum anticipates our Fig. 7 which further resolves Titan’s
compositionally related as opposed to topographical. This map                             visible spectrum and constrains the wavelengths where surface
was also reﬁned using the same method listed in Perry et al.                              features are visible. Light from Titan’s surface may be seen within
(2005) and is compared with Fig. 6 in part in Section 4. The color                        these wavelengths by any Earth-based telescope sporting Adap-
scheme in Fig. 6 helps to distinguish the surface reﬂectivity                             tive Optics, given the right ﬁlter. Voyager I resolved Titan’s
differences. As discussed later, there are some areas present in                          surface via a slight albedo difference (Richardson et al., 2004),
our visible map that are obscured by haze in the ISS                                      though this was not recognized for 25 years after the ﬂyby.
0:938 mm map.                                                                                 The phase angle, the angle from Cassini to Titan to the Sun,
                                                                                          affects the spectra because of changing haze scattering properties.
                                                                                          The spectrum in Fig. 7 was taken from the T8 ﬂyby which had an
3. Spectroscopy                                                                           ingress phase angle of 211. We compare this T8 spectrum to
                                                                                          Tsegihi/Aztlan in Ta, T3, T10, and T12 to sample the effect of
   A spectral comparison between bright and dark surface albe-                            phase angle on surface contrast. The dates and phases of each
dos on Titan’s surface tells us how much light at every wave-                             ﬂyby are listed in Table 1. Rannou et al. (2003) did a center-to-
length channel in the VIMS instrument transmits or is absorbed                            limb contrast variation previously at 673 nm by showing a greater
going through the atmosphere. We choose areas of the same size,                           relative contrast in bright and dark surface features the further
one bright and one dark region on Titan near each other so that                           the distance from the limb.
we have similar phase angle at each location. The regions chosen                              Fig. 8 shows spectra from each ﬂyby alongside each other for
were from the T8 ﬂyby of Xanadu and Shangri-La and are boxes of                           comparison. A value of 1 indicates that the regions of Tsegihi
uniform albedo along the same latitudinal lines. The longitude for                        and Aztlan look the same at the given wavelength. The trend that
Xanadu’s region is from 1451W to 1401W and Shangri-La’s is from                           we ﬁnd here is that as phase angle increases, the systematic
1551W to 1501W with the shared latitude region from 81S to 121S                           effects of haze increase and the baseline goes above 1, reducing
for a total of 20 pixels per box (Table 1). The plot in Fig. 7 is a                       the effective I/F of transmission peaks (i.e. inhibiting surface
spectral ratio dividing Xanadu by Shangri-La. This spectral ratio                         viewing).
```

<!-- PDF_PAGE: 5 -->

## PDF page 5

```text
56                                                          G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61


                 180 W         150 W    120 W    90 W           60 W        30 W         0           30 E        60 E         90 E       120 E   150 E      180 E
        90 N                                                                                                                                                        90 N
 (North pole)                                                                                                                                                       (North pole)




         60 N                                                                                                                                                       60 N




         30 N                                                                                                                                                       30 N



                                                                        a l
             0                                                      n s                                                                                             0
     (Equator)                                                  F e                                                                                                 (Equator)
                         Shangri−
                            La         X a n a d u                             a n                  Senkyo                         B e l e t                         Huygens
                                                                           t l
                                                                       A z                                                                                           landing
                                                                                                                                                                       site
         30 S                                                                                                                                                       30 S

                                                                       Tsegihi

         60 S                                                                                                                                                       60 S




        90 S                                                                                                                                                      90 S
 (South pole)                                                                                                                                                     (South pole)
               180 W           150 W    120 W     90 W          60 W        30 W          0          30 E        60 E          90 E      120 E   150 E      180 E
           (anti−Saturn)                        (leading)                            (sub−Saturn)                           (trailing)                   (anti−Saturn)




                                                                                                             90W                                             90N
                                                     0W




                                                     270W                                                    180W                                         90S


Fig. 3. The atmospherically uncorrected I/F map of Titan shown is made via VIMS from T8, T9, T31, T34, and rev9 in cylindrical projection (a). The colors and wavelengths
used to make this map are red at 0:937 mm, green at 0:827 mm, and blue at 0:754 mm. Part a of the ﬁgure shows a simple cylindrical map of the ﬁve ﬂybys sewn together.
Parts b–g of the ﬁgure show the different faces of Titan in an orthographical projection. In comparison to infrared maps, our optical map shows less distinction between
different wavelengths. However, note that the 5-mm bright spots do not stand out with the exception of Tui Regio. (For interpretation of the references to color in this
ﬁgure legend, the reader is referred to the web version of this article.)

   The height of the intensity peak remains constant above the                               reduces the signal. Phase angles o303 minimize atmospheric
height of the baseline as phase increases. The spectrum for Ta                               factors and maximize surface contrast.
compared to the spectrum for T12 has the baseline much closer to
unity and shows less scattering; this suggests that low phase
angles yield clearer results. Judging from the level of noise in the                         4. Comparison to the infrared
T12 ﬂyby, there seems to be a phase angle at which the data
become too noisy to be useful. The lower the phase angle is, the                                 We compare optical wavelength map in Fig. 3 to the previous
less scattering occurs and the better quality surface viewing is.                            infrared map in Fig. 4 to extend the region of spectral coverage. The
When the phase angle is high, the lower I/F from limb-darkening                              optical map has less distinguished contrast within light and dark
```

<!-- PDF_PAGE: 6 -->

## PDF page 6

```text
                                                           G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61                                                             57




Fig. 4. This ﬁgure represents the work of Barnes et al. (2009) in mapping Titan using atmospheric windows in the infrared spectrum. The dark blue color on the optical
wavelength map corresponds to the darkest green on the IR map. The bright orange color in the IR map represents the 5-mm active sites (bright orange features near the poles
are clouds). Many differences can be seen comparing the infrared and visible maps. In particular Tsegihi shows up bright in the optical wavelengths, just as bright as Xanadu.
While at 5 mm Tsegihi is actually brighter than Xanadu. At longer wavelengths Tsegihi is not as prominent and does not stand out from the rest of the area in the southern
regions; but it does look different in the optical map. (For interpretation of the references to color in this ﬁgure legend, the reader is referred to the web version of this article.)

regions, whereas the infrared map described in Barnes et al. (2009)                            terrain is slightly brighter across Xanadu and Tsegihi (Fig. 3) but is
has many different distinguishable spectral variations within the light                        mostly uniform across Titan (Fig. 6).
and dark regions. Since Titan appears gray within the optical
windows, the single scattering albedo (the ratio of scattering efﬁ-
                                                                                               4.1.1. Dark
ciency to total extinction efﬁciency) must be a good approximation of
                                                                                                  The dark regime covers the majority of Titan’s equatorial
the Bond albedo. The 5-mm bright areas, Tui Regio and Hotei Regio,
                                                                                               region and corresponds to the vast sand seas littering Titan’s
for instance, do not appear distinguished from the rest of Xanadu.
                                                                                               surface (Barnes et al., 2007; Soderblom et al., 2007). The place-
    The Selk and Sinlap craters are not resolved on the visible map.
                                                                                               ment of the dark areas agrees almost exactly with the dark brown
This is not due to the wavelength directly; a surface feature of this
                                                                                               spectral unit, giving us a good comparison standard. Although the
type should not depend on the wavelength in which it is observed
                                                                                               resolution is coarser in the visible than in the IR, the boundaries
unless the feature is covered in some optically sensitive material.
                                                                                               between spectral types are sharper in the visible map in Fig. 6.
The reason these craters do not appear is because of the resolution
                                                                                                  The eastern sections of the equatorial area in each map do not
of the slit scanner on Cassini combined with the extra atmospheric
                                                                                               agree as well in the total surface features visible. Belet does not
scatter from working in the visual wavelengths. The VIMS-V scans at
                                                                                               stand out as well in the visible map from being on a seam
low resolution in order to compile a global view of Titan; however, a
                                                                                               between ﬂybys, thus making the emission angle higher, causing
crater at most appearing in two pixels (See Table 1) will still fail to
                                                                                               more haze and lower surface contrast. The section south of Adiri,
be resolved because of the high amount of atmospheric scattering,
                                                                                               however, is clearer in the visible. Just above the 301S mark on the
even though the ejecta blankets are apparent.
                                                                                               eastern end of the map, we can see the dark extend down into an
    All of the light and dark albedo features, ﬁrst described by Porco
                                                                                               open region not visible on the IR map. These new areas could be
et al. (2005), are the focus of the discussion for surface composition.
                                                                                               outlines of dune material.
The next step is to do a systematic comparison of spectral types
between the map in the visible (Figs. 3 and 6) to a well described IR
map (Fig. 4) based on Barnes et al. (2007). The type of material each                          4.1.2. Visible light blue
color and shade refers to is described in the aforementioned paper                                The light blue regime corresponds closely to the dark blue IR
and will not be discussed here. A systematic comparison of each                                spectral unit but also covers additional area. In Fig. 6, the light
spectral signature in the IR methane windows to composition types                              blue serves as the midway between the white and dark areas. The
can be found in McCord et al. (2008).                                                          light blue in the visible matches all of the dark blue areas in the IR
                                                                                               map but also extends to some other areas, most notably: northern
                                                                                               Shangri-La in the west, northeast Senkyo in the center, and
4.1. Equatorial zone                                                                           around the outer fringes of Adiri in the east.
                                                                                                  VIMS dark blue areas are thought to be dirty water ice
   The equatorial zone bounded by 301 north and south of the                                   (Rodriguez et al., 2006; Le Mouélic et al., 2008; Soderblom
equator is home to the greatest contrast on Titan’s surface. VIMS-V                            et al., 2007; Barnes et al., 2007) or organic compounds
can distinguish light and dark terrain, and the dark terrain is split                          (Clark et al., 2010). Fig. 9 compares the water ice spectrum from
into dark and light blue spectral types. The light colored (white)                             Enceladus to a spectrum of the light blue (dark blue IR) spectral unit
```

<!-- PDF_PAGE: 7 -->

## PDF page 7

```text
58                                                            G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61


                    180 W    150 W        120 W        90 W      60 W         30 W         0          30 E        60 E          90 E       120 E      150 E          180 E
         90 N                                                                                                                                                              90 N
  (North pole)                                                                                                                                                            (North pole)
                                     Incidence Angle


           60 N                                                                                                                                                              60 N




           30 N                                                                                                                                                              30 N




              0                                                                                                                                                              0
       (Equator)                                                                                                                                                             (Equator)




            30 S                                                                                                                                                             30 S




            60 S                                                                                                                                                             60 S




            90 S                                                                                                                                                            90 S
     (South pole)                                                                                                                                                          (South pole)
                 180 W       150 W        120 W     90 W         60 W         30 W          0         30 E        60 E           90 E      120 E      150 E           180 E
             (anti−Saturn)                        (leading)                            (sub−Saturn)                           (trailing)                           (anti−Saturn)

                    180 W    150 W        120 W        90 W      60 W         30 W         0          30 E        60 E          90 E       120 E      150 E          180 E
         90 N                                                                                                                                                              90 N
  (North pole)                                                                                                                                                            (North pole)
                                     Emission Angle

           60 N                                                                                                                                                              60 N




           30 N                                                                                                                                                              30 N




              0                                                                                                                                                              0
       (Equator)                                                                                                                                                             (Equator)




            30 S                                                                                                                                                             30 S




            60 S                                                                                                                                                             60 S




            90 S                                                                                                                                                            90 S
     (South pole)                                                                                                                                                          (South pole)
                 180 W       150 W        120 W     90 W         60 W         30 W          0         30 E        60 E           90 E      120 E      150 E           180 E
             (anti−Saturn)                        (leading)                            (sub−Saturn)                           (trailing)                           (anti−Saturn)


                                                                                                                                                              0°                     90°

Fig. 5. The top map represents the incidence angle as it changes over each ﬂyby. The bottom map represents the emission angle. The angle, in both cases, build up as we approach
the center of a ﬂyby, then starts to drops off. These maps identify the nadir points where the signal-to-noise ratio is the greatest in relation to the resolution in Figs. 3 and 6.



in northern Shangri-La, similar to Grifﬁth et al. (2003). The overall                          in the visible maps is not as striking. Xanadu appears a little
reﬂectivity in Enceladus produces a much higher I/F since there is no                          brighter in Fig. 3 than the rest of the bright terrain; also in Fig. 6,
atmosphere to contend with. However, the gray appearance and                                   Xanadu is distinguished from its surroundings.
greater relative brightness of the light blue spectrum in the visible
wavelengths is broadly consistent with water ice, but certainly does                           4.2. Mid latitude zone
not rule out the many possible organic compounds.
                                                                                                   The mid latitude zone on Titan extends from 301 to 601 north
                                                                                               and south. On the IR map, these regions are fuzzy and have
4.1.3. Xanadu                                                                                  nothing of real note except for the brighter albedo feature Tsegihi.
   While Xanadu stands out as a bright albedo feature in the IR,                               In the visible, there is a new area south of Shangri-La in the west
the difference between Xanadu and the rest of the bright terrain                               not visible on the IR map. The 5 mm bright features on the IR map
```

<!-- PDF_PAGE: 8 -->

## PDF page 8

```text
                                                           G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61                                                        59


                180 W         150 W    120 W    90 W           60 W        30 W          0           30 E       60 E         90 E       120 E   150 E       180 E
        90 N                                                                                                                                                      90 N
 (North pole)                                                                                                                                                    (North pole)




        60 N                                                                                                                                                        60 N




        30 N                                                                                                                                                        30 N



                                                                       a l
                                                                   n s
           0
   (Equator)
                                                               F e                                                                                                  0
                                                                                                                                                                    (Equator)
                        Shangri−
                           La         X a n a d u                             a n                   Senkyo                        B e l e t                          Huygens
                                                                          t l
                                                                      A z                                                                                            landing
                                                                                                                                                                       site
        30 S                                                                                                                                                        30 S

                                                                      Tsegihi

        60 S                                                                                                                                                        60 S




        90 S                                                                                                                                                      90 S
 (South pole)                                                                                                                                                    (South pole)
              180 W           150 W    120 W     90 W          60 W        30 W          0           30 E       60 E          90 E      120 E   150 E       180 E
          (anti−Saturn)                        (leading)                             (sub−Saturn)                          (trailing)                    (anti−Saturn)




                                                                                                             90W                                              90N
                                                    0W




                                                    270W                                                     180W                                            90S


Fig. 6. This is a corrected global map of Titan, using the same color/wavelength scheme as Fig. 3, created using an algorithm to divide the haze out of the picture. In doing
so, the surface features of Titan become more pronounced. Removing some of the haze also brings out the striping effects that are inherent to the VIMS-V system. When
compared to Fig. 3, we can see new areas emerge, such as the section south of Shangri-La in the west.




are non-existent on the visible map; however, Tui Regio, south of                            901W to roughly 201E. In Fig. 6, with the haze removed, Tsegihi
Xanadu, can be described by its outline.                                                     appears, again, to be nothing of note and instead shows some
                                                                                             splotches of light blue spectral types in the north and west areas
                                                                                             corresponding to some light brown areas on the IR map.
4.2.1. Tsegihi
   Tsegihi is the second brightest large albedo feature in the IR.
However, as seen in Fig. 3, Tsegihi is the brightest albedo                                  4.2.2. Blue white
feature—even more so than Xanadu. This brightness of albedo                                     This section is lighter in color than the light blue in the
may arise from the low phase angle of T9, the ﬂyby comprising                                equatorial region and exists south of Shangri-La. In the IR, this
```

<!-- PDF_PAGE: 9 -->

## PDF page 9

```text
60                                                               G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61


                                                                             0.937                       0.6
                                                                                                                                                   Water Ice (Enceladus)
                       1.25                                                                                                                        Light Blue (Shangri−La)
                                                                                                         0.5

                       1.20
                                                                  0.827
                                                                                                         0.4




   Xanadu/Shangri−La
                       1.15
                                                                                           1.05
                              Shangri−La              Xanadu                                       I/F   0.3
                                                          0.754
                       1.10
                                                       0.681
                                                   0.637                                                 0.2
                       1.05


                                                                                                         0.1
                       1.00


                                                                                                         0.0
                               0.4    0.5      0.6      0.7     0.8       0.9        1.0                                1         2           3           4          5
                                             Wavelength (microns)                                                                 Wavelength (Microns)
Fig. 7. A spectral ratio graph, created using the T8 ﬂyby, depicting the detected ﬂux             Fig. 9. A comparison between water ice (for reference) and the visible light blue
from a bright area relative to that in a dark area. It represents the average value of            spectral type taken from northern Shangri-La. The stars in the Enceladus line
Xanadu divided by Shangri-La versus wavelength in microns with optical wavelength                 represent the atmospheric windows in both the visible and IR. The overall gray
windows labeled and the range of the windows boxed. At wavelengths with strong                    appearance and greater relative brightness compared to the IR in the visible
atmospheric absorption, the ﬂux coming from Xanadu and that from Shangri-La are                   wavelengths in the Titan spectrum is broadly consistent with water ice, and also
about the same, barring a small difference owing to different haze reﬂections resulting           with various organic materials. Inversions from minima to maxima in these
from view geometry. Within spectral windows, where there is little or no methane                  graphs imply wavelengths where a signal may be transmitted without being
absorption, the much higher surface reﬂectivity of Xanadu results in higher ratios. The           absorbed.
ratio for longer-wavelength windows is progressively higher than that for shorter-
wavelength windows because the ratio is less affected by scatting off of haze particles.
Windows exist at 0.637, 0.681, 0.754, 0.827, 0.937, and 1:05 mm.

                                                                                                  section appears to be a group of 5-mm bright area with no real
                                                                                                  boundary. In the visible, however, we can make out a fringe
                                           T12 − 53° phase                                        between this blue white region, Tui Regio, and the surrounding
                       1.4                 T10 − 28° phase                                        area. This area in question is slightly larger than Shangri-La itself
                                           T3 − 20° phase                                         and has its southern boundary much more visible in Fig. 6.
                                           Ta − 9° phase

                       1.3                                                                        4.2.3. White
                                                                                                     The greatest area of Titan is comprised of white (Fig. 3) color




Aztlan / Tsegihi
                                                                                                  corresponding to the equatorial bright spectral unit and many of
                                                                                                  the 5-mm bright areas of the IR map. This substrate covers the
                       1.2                                                                        majority of Titan not occupied by the sand dunes.



                                                                                                  5. Conclusion
                       1.1
                                                                                                      The spectral albedo comparison of bright and dark surfaces on
                                                                                                  Titan conﬁrm transmission peaks in the optical wavelengths. This
                                                                                                  gives targets to ground-based observers with Active Optics tele-
                       1.0                                                                        scopes and allows for the possibility of resolving the surface of
                                                                                                  Titan using visible spectrum ﬁlters from Earth (Lemmon et al.,
                                                                                                  2002; Lorenz et al., 2003). This would allow more frequent data
                              0.4    0.5     0.6      0.7      0.8     0.9      1.0               acquisition from a Moon so far away.
                                                                                                      The visible spectrum mainly shows gray albedo changes for
                                            Wavelength (Microns)
                                                                                                  surface features on Titan. This serves as another constraint on
Fig. 8. The four lines above are spectral ratios from the same region as Fig. 7, T8 but at        surface composition to the IR maps published already. The main
different phase angles, increasing from Ta to T12 by ﬂyby number. The x-axis is                   larger features on Titan are distinguishable at visible wavelengths
wavelength in microns and the y-axis is albedo difference taken by dividing Xanadu                but the smaller features and the 5-mm bright areas are not. We
by Shangri-La. As the phase angle goes up, the amount of striping noise in the                    attributed this to either the lower spatial resolution (the spatial
spectrum goes up as is apparent by the rising and delinearization of the baseline. The
height of each intensity peak remains unchanged above the baseline with increasing
                                                                                                  sampling does not change) of VIMS-V or to the inherent reﬂec-
phase. The contribution from atmospheric scattering becomes more obvious in the                   tivity of material on the surface that is apparent only in the IR,
plot of T12 in that lower wavelengths scatter more, raising the baseline.                         respectively. The improved surface contrast viewing Titan with
```

<!-- PDF_PAGE: 10 -->

## PDF page 10

```text
                                                              G. Vixie et al. / Planetary and Space Science 60 (2012) 52–61                                                                61


some haze removed, however, proves useful for seeing the extent                                 Karkoschka, E., 1994. Spectrophotometry of the Jovian planets and Titan at 300- to
to which certain surface features reach.                                                            1000-nm wavelength: the methane spectrum. Icarus 111 (September),
                                                                                                    174–192.
    Our transmission spectrum in Fig. 7 is the major result from the                            Le Mouélic, S., Paillou, P., Janssen, M.A., Barnes, J.W., Rodriguez, S., Sotin, C., Brown,
visible wavelength observations for this paper. Knowing precisely                                   R.H., Baines, K.H., Buratti, B.J., Clark, R.N., Crapeau, M., Encrenaz, P.J., Jaumann,
where windows exist in the optical wavelengths can improve how                                      R., Geudtner, D., Paganelli, F., Soderblom, L., Tobie, G., Wall, S., 2008. Mapping
                                                                                                    and interpretation of Sinlap crater on Titan using Cassini VIMS and RADAR
spacecraft time is spent and what is targeted. If ground-based                                      data. Journal of Geophysical Research (Planets) 113 (April), E04003.
observers can resolve the surface of Titan, then those observations                             Le Mouélic, S., Cornet, T., Rodriguez, S., Sotin, C., Barnes, J.W., Brown, R.H., Baines,
can be used to pick out targets for Cassini or any other spacecraft                                 K.H., Buratti, B.J., Clark, R.N., Nicholson, P.D., 2010. Empirical approaches to
                                                                                                    reduce the atmospheric component in VIMS surface images of Titan. AGU Fall
that may visit the Saturn system in the future.
                                                                                                    Meeting Abstracts, December, C1546 þ.
    The surface of Titan, scanned over all the visible VIMS                                     Lemmon, M.T., Smith, P.H., Lorenz, R.D., 2002. Methane abundance on Titan,
channels, appears gray in each individual wavelength. All the                                       measured by the space telescope imaging spectrograph. Icarus 160 (Decem-
                                                                                                    ber), 375–385.
photons Cassini receives back have been scattered to some extent
                                                                                                Lockwood, G.W., Lutz, B.L., Thompson, D.T., Bus, E.S., 1986. The albedo of Titan.
but not absorbed by the atmosphere. We can make good use of                                         Astrophysical Journal 303 (April), 511–520.
the scattering albedo then as an approximation for the Bond                                     Lorenz, R.D., Dooley, J.M., West, J.D., Mitsugu, F., 2003. Backyard spectroscopy and
albedo.                                                                                             photometry of Titan, Uranus, and Neptune. Planetary and Space Science
                                                                                                    51 (February), 113–125.
    The IR wavelengths provide more spectral features for under-                                McCord, T.B., Hayne, P., Combe, J.P., Hansen, G.B., Barnes, J.W., Rodriguez, S.,
standing the surface composition; but optical maps help to                                          Le Mouélic, S., Baines, E.K.H., Buratti, B.J., Sotin, C., Nicholson, P., Jaumann, R.,
exclude surface materials not active in visible wavelengths, and                                    Nelson, R., 2008. The Cassini Vims Team, 2008. Titan’s surface: search for
                                                                                                    spectral diversity and composition using the Cassini VIMS investigation. Icarus
as such provide a useful complement to longer-wavelength                                            194, 212–242.
studies. This paper’s intent is to report on wavelengths Cassini                                McKay, C.P., Pollack, J.B., Courtin, R., 1989. The thermal structure of Titan’s
already scans, rather than arrive at speciﬁc constraints.                                           atmosphere. Icarus 80, 23–53.
                                                                                                Neff, J.S., Humm, D.C., Bergstralh, J.T., Cochran, A.L., Cochran, W.D., Barker, E.S.,
                                                                                                    Tull, R.G., 1984. Absolute spectrophotometry of Titan, Uranus, and Neptune
                                                                                                    3500–10,500 Å. Icarus 60 (November), 221–235.
Acknowledgments                                                                                 Perry, J.E., McEwen, A.S., Fussner, S., Turtle, E.P., West, R.A., Porco, C.C., Knowles, B.,
                                                                                                    Dawson, D.D., 2005. The Cassini Iss Team, 2005. Processing ISS images of
                                                                                                    Titan’s surface. In: Mackwell, S., Stansbery, E. (Eds.), 36th Annual Lunar and
    The work done here was made possible by NASA, ESA, and the
                                                                                                    Planetary Science Conference, pp. 2312–þ .
VIMS team. The authors also acknowledge funding from Grant                                      Porco, C.C., Baker, E., Barbara, J., Beurle, K., Brahic, A., Burns, J.A., Charnoz, S.,
NNX09AP34G to J.W.B. from the NASA Outer Planets Research                                           Cooper, N., Dawson, D.D., Del Genio, A.D., Denk, T., Dones, L., Dyudina, U.,
program. P. Cerroni, F. Capaccioni, A. Coradini, G. Filacchione and                                 Evans, M.W., Fussner, S., Giese, B., Grazier, K., Helfenstein, P., Ingersoll, A.P.,
                                                                                                    Jacobson, R.A., Johnson, T.V., McEwen, A., Murray, C.D., Neukum, G., Owen, W.M.,
F. Tosi acknowledge the support of ASI Grant I/015/09/0.                                            Perry, J., Roatsch, T., Spitale, J., Squyres, S., Thomas, P., Tiscareno, M., Turtle, E.P.,
                                                                                                    Vasavada, A.R., Veverka, J., Wagner, R., West, R., 2005. Imaging of Titan from the
                                                                                                    Cassini spacecraft. Nature 434, 159–168.
References
                                                                                                Porco, C.C., West, R.A., Squyres, S., McEwen, A., Thomas, P., Murray, C.D., Delgenio,
                                                                                                    A., Ingersoll, A.P., Johnson, T.V., Neukum, G., Veverka, J., Dones, L., Brahic, A.,
Barnes, J.W., Brown, R.H., Radebaugh, J., Buratti, B.J., Sotin, C., Le Mouelic, S.,                 Burns, J.A., Haemmerle, V., Knowles, B., Dawson, D., Roatsch, T., Beurle, K.,
    Rodriguez, S., Turtle, E.P., Perry, J., Clark, R., Baines, K.H., Nicholson, P.D., 2006.         Owen, W., 2004. Cassini imaging science: instrument characteristics and
    Cassini observations of ﬂow-like features in western Tui Regio, Titan. Geo-                     anticipated scientiﬁc investigations at Saturn. Space Science Reviews 115,
    physical Research Letters 33 (August), L16204.                                                  363–497.
Barnes, J.W., Brown, R.H., Soderblom, L., Buratti, B.J., Sotin, C., Rodriguez, S.,              Rannou, P., McKay, C.P., Lorenz, R.D., 2003. A model of Titan’s haze of fractal
    Le Moue lic, S., Baines, K.H., Clark, R., Nicholson, P., 2007. Global-scale surface            aerosols constrained by multiple observations. Planetary and Space Science 51,
    spectral variations on Titan seen from Cassini/VIMS. Icarus 186, 242–258.                       963–976.
Barnes, J.W., Brown, R.H., Turtle, E.P., McEwen, A.S., Lorenz, R.D., Janssen, M.,               Richardson, J., Lorenz, R.D., McEwen, A., 2004. Titan’s surface and rotation: new
    Schaller, E.L., Brown, M.E., Buratti, B.J., Sotin, C., Grifﬁth, C., Clark, R., Perry, J.,       results from Voyager 1 images. Icarus 170, 113–124.
    Fussner, S., Barbara, J., West, R., Elachi, C., Bouchez, A.H., Roe, H.G.,                   Rodriguez, S., Le Mouélic, S., Sotin, C., Clénet, H., Clark, R.N., Buratti, B., Brown, R.H.,
    Baines, K.H., Bellucci, G., Bibring, J.P., Capaccioni, F., Cerroni, P., Combes, M.,             McCord, T.B., Nicholson, P.D., Baines, K.H., 2006. The VIMS Science Team, 2006.
    Coradini, A., Cruikshank, D.P., Drossart, P., Formisano, V., Jaumann, R., Langevin, Y.,         Cassini/VIMS hyperspectral observations of the HUYGENS landing site on
    Matson, D.L., McCord, T.B., Nicholson, P.D., Sicardy, B., 2005. A 5-micron-bright               Titan. Planetary and Space Science 54, 1510–1523 (eprint0906.5476).
    spot on Titan: evidence for surface diversity. Science 310, 92–95.                          Smith, P.H., Lemmon, M.T., Lorenz, R.D., Sromovsky, L.A., Caldwell, J.J.,
Barnes, J.W., Soderblom, J.M., Brown, R.H., Buratti, B.J., Sotin, C., Baines, K.H., Clark,          Allison, M.D., 1996. Titan’s surface, revealed by HST imaging. Icarus 119,
    Jaumann, R., McCord, T.B., Nelson, R., Le Moue lic, S., Rodriguez, S., Grifﬁth, C.,            336–349.
    Penteado, P., Tosi, F., Pitman, K.M., Soderblom, L., Stephan, K., Hayne, P.,                Soderblom, L.A., Brown, R.H., Soderblom, J.M., Barnes, J.W., Kirk, R.L., Sotin, C.,
    Vixie, G., Bibring, J., Bellucci, G., Capaccioni, F., Cerroni, P., Coradini, A.,                Jaumann, R., MacKinnon, D.J., Mackowski, D.W., Baines, K.H., Buratti, B.J., Clark, R.N.,
    Cruikshank, D.P., Drossart, P., Formisano, V., Langevin, Y., Matson, D.L.,                      Nicholson, P.D., 2009. The geology of Hotei Regio, Titan: correlation of Cassini VIMS
    Nicholson, P.D., Sicardy, B., 2009. VIMS spectral mapping observations of Titan                 and RADAR. Icarus 204, 610–618.
    during the Cassini prime mission. Planetary and Space Science 57, 1950–1962.                Soderblom, L.A., Kirk, R.L., Lunine, J.I., Anderson, J.A., Baines, K.H., Barnes, J.W.,
Brown, R.H., Baines, K.H., Bellucci, G., Bibring, J.P., Buratti, B.J., Capaccioni, F.,              Barrett, J.M., Brown, R.H., Buratti, B.J., Clark, R.N., Cruikshank, D.P., Elachi, C.,
    Cerroni, P., Clark, R.N., Coradini, A., Cruikshank, D.P., Drossart, P., Formisano, V.,          Janssen, M.A., Jaumann, R., Karkoschka, E., Mouélic, S.L., Lopes, R.M., Lorenz, R.D.,
    Jaumann, R., Langevin, Y., Matson, D.L., McCord, T.B., Mennella, V., Miller, E.,                McCord, T.B., Nicholson, P.D., Radebaugh, J., Rizk, B., Sotin, C., Stofan, E.R.,
    Nelson, R.M., Nicholson, P.D., Sicardy, B., Sotin, C., 2004. The Cassini Visual and             Sucharski, T.L., Tomasko, M.G., Wall, S.D., 2007. Correlations between Cassini
    Infrared Mapping Spectrometer (VIMS) investigation. Space Science Reviews                       VIMS spectra and RADAR SAR images: implications for Titan’s surface composi-
    115, 111–168.                                                                                   tion and the character of the huygens probe landing site. Planetary and Space
Capaccioni, F., Coradini, A., Cerroni, P., Amici, S., 1998. Imaging spectroscopy of                 Science 55, 2025–2036.
    Saturn and its satellites: VIMS-V onboard Cassini. Planetary and Space Science              Stephan, K., Jaumann, R., Karkoschka, E., Barnes, J.W., Kirk, R., Tomasko, M.G.,
    46, 1263–1276.                                                                                  Turtle, E.P., Le Corre, L., Langshans, M., Le Moue lic, S., Lorenz, R., Perry, J., 2009.
Clark, R.N., Curchin, J.M., Barnes, J.W., Jaumann, R., Soderblom, L., Cruikshank, D.P.,             Mapping products of Titan’s surface. In: Brown, R.H., Lebreton, J., Waite, J.H.
    Brown, R.H., Rodriguez, S., Lunine, J., Stephan, K., Hoefen, T.M., Le Mouélic, S.,             (Eds.), Titan from Cassini–Huygens, pp. 489–510.
    Sotin, C., Baines, K.H., Buratti, B.J., Nicholson, P.D., 2010. Detection and                Turtle, E.P., Perry, J.E., McEwen, A.S., Del Genio, A.D., Barbara, J., Dawson, D.D.,
    mapping of hydrocarbon deposits on Titan. Journal of Geophysical Research                       West, R.A., Porco, C.C., 2009. Cassini imaging of Titan’s high-latitude lakes, clouds,
    (Planets) 115 (October), E10005.                                                                and south-polar surface changes. Geophysical Research Letters 36 (January),
Grifﬁth, C.A., Owen, T., Geballe, T.R., Rayner, J., Rannou, P., 2003. Evidence for the              L02204.
    exposure of water ice on Titan’s surface. Science 300, 628–630.                             Turtle, E.P., Del Genio, A.D., Barbara, J.M., Perry, J.E., Schaller, E.L., McEwen, A.S.,
Grifﬁth, C.A., Owen, T., Wagener, R., 1991. Titan’s surface and troposphere, investi-               West, R.A., Ray, T.L., 2011. Seasonal changes in Titan’s meteorology. Geophysical
    gated with ground-based, near-infrared observations. Icarus 93, 362–378.                        Research Letters 38 (February), L03203.
```
