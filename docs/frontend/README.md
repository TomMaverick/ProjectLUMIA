# Frontend Design Specifications

## Overview
The `docs/frontend/` directory contains design guidelines, ergonomic mandates, and component architecture specifications for Project LUMIA's React / TypeScript web user interface.

## Interface
- [ui_design_specification.md](./ui_design_specification.md): Canonical frontend UX design specification covering night-time observatory ergonomics, zero-emoji policy, `tabular-nums` formatting, Tailwind CSS tokens, candidate blink-loops, and WebSocket state integration.

## Invariants
- Frontend implementations must strictly adhere to the night-time palette, tabular numeric alignments, and finite number policies documented herein.

## Side Effects
- None. Static design records.

## Verification
- Run: `uv run pytest tests/test_api/test_web_ui.py`
