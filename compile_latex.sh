#!/bin/bash

cd LaTex

# Clean up previous compilation files
rm -f limb_paper.aux limb_paper.bbl limb_paper.blg limb_paper.log limb_paper.out limb_paper.synctex.gz

# First LaTeX pass
pdflatex limb_paper.tex

# Run BibTeX to process citations
bibtex limb_paper

# Second LaTeX pass to incorporate bibliography
pdflatex limb_paper.tex

# Third LaTeX pass to resolve all references
pdflatex limb_paper.tex

echo "Compilation complete. Check limb_paper.pdf for the result." 