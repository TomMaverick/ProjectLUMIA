# Module 10: Output, Review & Web-UI Specification

## Overview
Module 10 provides a high-performance web interface and REST/WebSocket backend for monitoring pipeline execution, reviewing candidate asteroid detections, and inspecting blink-loops. The interface enforces a strict dark mode aesthetic with deep purple accent tones.

## Key Components
* **Backend API (FastAPI):**
  * RESTful endpoints (protected by JWT authentication) serving detection results, multi-epoch `31x31` / `51x51` image cutouts, candidate triage disposition updates, and pipeline execution logs exclusively via **Module 2** (`docs/modules/module_02_pipeline.md`) repository interfaces (zero raw SQL in API routes).
  * WebSocket broadcasting for real-time task and job status updates sourced from Prefect and PostgreSQL alongside analytical detection metrics from DuckDB.
* **Frontend Dashboard (React / TypeScript + Tailwind CSS):**
  * **Theme & Styling:** Strict Dark Mode architecture utilizing deep slate/charcoal backgrounds combined with rich, dark purple accent colors (`#3b185f` to `#581c87`) for interactive elements, highlights, and borders to maintain a professional, low-fatigue observatory look.
  * **Interactive Multi-Survey Blink-Loop Viewer (`BlinkLoopViewer.tsx`):**
    * Multi-epoch animation canvas for visual verification of moving object candidates across disparate instruments (`ZTF`, `Pan-STARRS`, `Rubin`, `JWST`).
    * **Cross-Survey FOV Normalization:** Toggleable physical scale alignment mode standardizing the canvas field-of-view across disparate optical resolutions ($0.031''/\text{px}$ JWST to $1.012''/\text{px}$ ZTF), preventing jarring celestial scale changes during blink loops.
    * **Instrument Badges & Filter Bandpass Tags:** Color-coded observatory tags (e.g., Pan-STARRS cyan, Rubin emerald, JWST indigo, ZTF amber) and optical filter indicators (`g`, `r`, `i`, `F090W`).
    * **Astrometric Scale Bar & Telemetry HUD:** Calibrated angular scale bar ($1''\text{--}30''$) and real-time resolution/FOV readouts (`X.XXX"/px | FOV: XX.X"`).
  * **Candidate Triage & Active Learning Panel:** Side-by-side inspection of **Module 7** CNN classification scores (Real/Bogus), **Module 5** astrometric tracklet residuals, and photometric light curves, allowing operators to record verified Real/Bogus dispositions into PostgreSQL for **Module 8** HDF5 Active Learning dataset curation. Displays light-travel time differential ($\Delta \tau$) for space observatory pointings.