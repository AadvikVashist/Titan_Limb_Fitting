Here is the converted Markdown version of the **Graphics Guide for AAS Journals**:

```markdown
# Graphics Guide - AAS Journals

## Introduction
This guide provides authors with instructions for preparing and submitting figures, animations, and interactive graphics for AAS journal articles.

For further assistance, visit the [AAS Journals Author Resources](https://journals.aas.org/author-resources/) or contact [journals.manager@aas.org](mailto:journals.manager@aas.org).

---

## 1. Figure Formats & Submission
### Accepted File Formats:
- **Preferred formats**: **Vector EPS** (Encapsulated PostScript) or **PDF**.
- **Alternative formats**: PNG, JPG, TIFF (minimum **300 DPI resolution**).
- Figures must have at least **1000 pixels** in horizontal resolution.

Each figure should be submitted as a **single page file**. If using **multi-page EPS or PDF**, split them into separate files using:
- [Adobe Acrobat](https://forums.adobe.com/thread/1148606)
- [OS X Preview](http://www.documentsnap.com/how-to-split-pdf-documents-into-single-pages-using-mac-osx/)
- Various free online tools.

### Including Figures in Manuscripts:
- **AASTeX Users**: Follow the [AASTeX figure inclusion guide](https://journals.aas.org/aastexguide/#figures).
- **Microsoft Word Users**: Embed figures but **separate figure files must be submitted**.

### Figure Sets, Animations, and Interactive Figures:
- Must include a **static 2D representation** with its own numbering and caption.
- The **static version** (e.g., `f1.pdf`) should be uploaded separately from the dynamic elements.
- Animations and interactive figures should be **bundled as ZIP files**, e.g., `fig01anim.zip`, `fig03int.zip`, `fig07set.zip`.

---

## 2. Preparing High-Quality, Accessible Figures
### Best Practices for Scientific Visualizations:
- Refer to [Ten Simple Rules for Better Figures](https://doi.org/10.1371/journal.pcbi.1003833).
- **Avoid default color schemes** that may be inaccessible to colorblind readers.
- Use **Color Oracle** ([colororacle.org](https://colororacle.org/)) to check color accessibility.
- Consider **color maps** like:
  - [Viridis (Matplotlib 2.0)](https://www.youtube.com/watch?v=xAoljeRJ3lU) ([R version](https://cran.r-project.org/package=viridis))
  - [Cube-helix](https://www.mrao.cam.ac.uk/~dag/CUBEHELIX/) ([Green 2011](http://adsabs.harvard.edu/abs/2011BASI...39..289G))

### Avoiding Figure Resizing Issues:
- Each figure or subfigure should be in **a separate file**.
- If figures are part of a **multi-panel figure**, place the panel letters **inside** the figure box.
- Do **not** include page numbers, figure numbers, or file information inside figure files.

### Fonts, Lines, and Symbols:
- Use standard fonts: **Times, Helvetica, Symbol**.
- Font size: **Minimum 6pt**.
- Lines in figures: **Minimum 0.5 points**.
- **Use different line styles and symbol shapes** to distinguish elements.

### Avoiding Color-Only Differentiation:
- Do not rely on color alone; use:
  - **Different line styles** for colored lines.
  - **Varied symbols** for colored markers.
  - **Different hatching or weights** for colored histograms.

---

## 3. Figure Sets
- **Definition**: A **collection of similar images or graphical material** that exceed four typeset pages.
- Provides a **simplified cost structure** for related figures.
- **Usage Examples**: Identification charts, spectral libraries, model outputs.

### Submission Requirements:
- Submit figure set components in **a single ZIP archive**, e.g., `fig01set.zip`.
- **Example static figure required** in the main manuscript.
- Use the [figure set LaTeX markup](https://journals.aas.org/aastexguide/#figureset_figures).

### Online Tools for Figure Sets:
- [Figure Set Markup Tool](http://authortools.aas.org/FIGSETS/make-figset.html)
- [AAS Journals GitHub Scripts](https://github.com/AASJournals/Tools/tree/master/figureset)

---

## 4. Animations
- **Now fully supported** as regular figures.
- Displayed in the **HTML version** using an embedded **YouTube-like player**.

### Submission Guidelines:
- **Static representation** required in the PDF version.
- Recommended **MPEG-4 (H.264 codec)** format.
- **Size limit**: **≤ 15MB**.
- **Framerate**: **≥ 15 FPS**.
- **Bitrate**: **≥ 1000 kbps**.

### Recommended Encoding Tools:
- [Handbrake](https://handbrake.fr/)
- [FFmpeg](https://www.ffmpeg.org/)

### **Avoid Common Issues**:
- Ensure animation **aspect ratio matches** the static version.
- Include a **detailed caption** describing the animation (duration, changes over time, annotations, etc.).

---

## 5. Interactive Figures
- **Definition**: Figures allowing users to manipulate displayed data interactively.
- **Examples**: 3D models, zoomable plots, layer-based time-series data.

### Submission Requirements:
- Submit a **static version** as an **EPS/PDF**.
- Bundle interactive elements (HTML, JavaScript, data files) in a **ZIP archive**.
- Include **detailed captions** describing interactions.

### Supported Frameworks:
| Library       | Status          | Notes |
|--------------|---------------|--------------------------------|
| **X3DOM**    | Fully supported | Used for 3D models ([Example](https://www.x3dom.org/)) |
| **Bokeh**    | Fully supported | ([BokehJS](http://bokeh.pydata.org/en/latest/)) |
| **Plotly**   | Fully supported | ([Graphing Libraries](https://plotly.com/graphing-libraries/)) |
| **Astropy Timeseries** | Fully supported | ([aas-timeseries](https://aas-timeseries.readthedocs.io/en/latest/)) |

### Additional Tools:
- **[X3D](https://en.wikipedia.org/wiki/X3D)** for 3D visualizations.
- **[Astropy](https://www.astropy.org/)** for astronomy-specific plots.

---

## 6. Software-Specific Advice
### **Astropy**
- Use **aas-timeseries** to generate interactive time-series plots:
  ```python
  fig.export_interactive_bundle('my_figure.zip')
  fig.save_static('my_figure', format='pdf')
  ```
  More info: [aas-timeseries](https://aas-timeseries.readthedocs.io/en/latest/).

### **FFmpeg**
- Convert an animated GIF to MP4:
  ```sh
  ffmpeg -i animated.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" output.mp4
  ```
  More info: [FFmpeg Documentation](https://trac.ffmpeg.org/wiki/Limiting%20the%20output%20bitrate).

### **R (for Plots)**
- Example plot using **ggplot2**:
  ```r
  library(ggplot2)
  ggplot(data, aes(x, y)) + geom_point(color="blue")
  ```
  More info: [Viridis for R](https://cran.r-project.org/package=viridis).

---

## Contact Information
📧 **General Support**: [journals.manager@aas.org](mailto:journals.manager@aas.org)  
📧 **Data-Editor Support**: [data-editors@aas.org](mailto:data-editors@aas.org)  
📧 **AASTeX Support**: [aastex@aas.org](mailto:aastex@aas.org)  

📞 **Phone**: (202) 328-2010  
🏢 **Address**: American Astronomical Society, 1667 K Street NW, Suite 800, Washington, DC 20006 USA  

---

© 2025 The American Astronomical Society | [Privacy Policy](https://journals.aas.org/privacy-and-cookies/) | [Terms & Conditions](https://journals.aas.org/terms/)  
Follow AAS: [Facebook](https://www.facebook.com/AmericanAstronomicalSociety) | [Twitter](https://twitter.com/AAS_Publishing) | [LinkedIn](https://www.linkedin.com/company/american-astronomical-society/) | [Instagram](https://www.instagram.com/aas_office/)
```

This Markdown version organizes all sections clearly and makes them easy to read. Let me know if you need any refinements! 🚀