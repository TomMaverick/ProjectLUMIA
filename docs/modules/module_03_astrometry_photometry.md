# Module 3: Data Ingestion, Astrometry & Photometry Specification

## Overview
Module 3 processes incoming FITS files, performs plate-solving, handles time-referencing, and executes aperture photometry.

## Scientific Prerequisites
* **Science-Ready Data:** The module assumes that raw FITS files fetched from archives are already calibrated (Level 1 / Level 2), meaning bias, dark current, and flat-field corrections have been applied.

## Key Components
* **WCS Calibration:** Plate-solving integration, coordinate transformation, and calculation of astrometric uncertainties.
* **Time Referencing:** Conversion of observation timestamps into Barycentric Julian Date (BJD) to account for light-travel time corrections.
* **Photometry Engine:** Zero-point determination and aperture photometry using `photutils`, calculating flux magnitudes and associated errors.