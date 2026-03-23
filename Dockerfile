# Dockerfile for JMLR LaTeX compilation
# Use this if you want a reproducible environment without installing LaTeX locally

FROM texlive/texlive:latest

# Set working directory
WORKDIR /paper

# Install any additional packages (if needed)
RUN tlmgr update --self && \
    tlmgr install jmlr natbib algorithm algorithmic

# Copy paper files
COPY . /paper/

# Default command: compile paper
CMD ["sh", "-c", "pdflatex jmlr_paper.tex && bibtex jmlr_paper && pdflatex jmlr_paper.tex && pdflatex jmlr_paper.tex"]

# Usage:
# 1. Save this as "Dockerfile" in your paper directory
# 2. Build: docker build -t jmlr-paper .
# 3. Run: docker run -v $(pwd):/paper jmlr-paper
# 4. Output: jmlr_paper.pdf in your current directory
