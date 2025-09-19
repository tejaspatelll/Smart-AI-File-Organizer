## Smart File Organizer — Project Overview

Transform chaos into order with an Apple‑inspired, AI‑powered desktop app that organizes files by understanding their content and your intent.

### Project at a Glance

- **Role**: End‑to‑end Product & UX, Frontend (Electron), Backend (Python)
- **Timeframe**: 2025
- **Platforms**: macOS desktop app (Electron + Python backend)
- **AI**: Google Gemini primary, OpenAI/Claude fallbacks

### The Problem

People accumulate thousands of files across downloads, desktops, and external drives. Traditional tools sort by filename or type, not by meaning. Users need a system that understands context, preserves intent, and remains forgiving.

### The Solution

A desktop application that:

- **Understands content** (not just extensions) and **your prompts** to propose a clear, human‑readable organization plan.
- Offers **Quick & Smart** one‑click organization, **Custom Prompts**, and **Professional Templates** for common workflows.
- Prioritizes **forgiveness** with an **Undo** system and a **Review & Customize** step before changes are applied.

### UX Highlights

- **Apple‑inspired visual language**: subtle glassmorphism, depth, and motion.
- **Progressive disclosure**: simple defaults first; advanced controls when needed.
- **High information density** without overwhelm: table, card, and grouped views with real‑time filters and search.
- **Clear AI reasoning**: each change shows confidence and the “FROM → TO” path so users can trust decisions.
- **Accessibility first**: keyboard navigation, screen reader support, high‑contrast and reduced‑motion options.

### Key Features

- **Intelligent AI Organization**: content analysis, intent understanding, confidence scoring.
- **Duplicate detection** with best‑file selection.
- **Batch processing** for tens of thousands of files with **real‑time progress**.
- **Priority‑based organization**: handle high‑confidence items first.
- **Templates** for Creative, Business, Student, and Personal life.

### Interaction Flow

1. Select a folder → 2) Choose method (Quick, Prompt, Template) → 3) AI analysis → 4) Review proposed moves with filters and sorting → 5) Apply with Undo available → 6) Success with stats and next steps.

### Design Principles

- **Progressive disclosure**
- **Immediate feedback**
- **Forgiveness & safety** (Undo)
- **Performance as UX** (smooth 60fps interactions)

### Technical Architecture

- **Frontend**: Electron + HTML/CSS/JS with IPC to backend; modern design system via CSS custom properties; virtualized lists for large datasets.
- **Backend**: Python services for scanning, classification, and planning; rich metadata model with confidence and reasoning.
- **AI**: Gemini‑first with fallbacks (OpenAI/Claude); multi‑stage prompting, validation, and calibrated confidence factors.
- **Caching**: SQLite‑based smart cache keyed by MD5 to avoid redundant AI calls.

### Performance & Reliability

- **Virtual scrolling** renders only visible rows (60px height, 5‑row buffer), reducing DOM nodes from thousands to ~50–70.
- **GPU memory optimizations** via will‑change and targeted animations; no more “tile memory” warnings.
- **Throughput**: 1,000+ files/min scanning; AI processed in batches with streaming progress.
- **Resilience**: provider fallbacks, retries with backoff, structured logging, and audit trails.

### Outcomes

- **Faster decisions**: clear AI confidence and reasoning builds trust.
- **Less cognitive load**: templated flows and previews remove guesswork.
- **Delight**: fluid micro‑interactions and focused states keep users in control.

### My Contributions

- Defined UX strategy and interaction model based on progressive disclosure and forgiveness.
- Designed the visual system (glassmorphism, motion, accessibility) and implemented Electron UI.
- Built Python backend orchestration, AI prompting/validation, and caching for cost/perf wins.
- Implemented virtual scrolling and GPU optimizations to maintain 60fps with huge file lists.

### Tech Stack

- **Frontend**: Electron, HTML/CSS/JS
- **Backend**: Python (async), Typer CLI
- **AI**: google‑generativeai, openai, anthropic
- **Data**: SQLite cache, SQLAlchemy
- **DX**: pytest, black, isort

### Screens & Media

- On portfolio: include hero, method selection, review table with confidence badges, and success screen with stats.

### How It Works (Short)

- Scan → Analyze content & context → Generate plan with confidence → Let user review → Apply with undo.

### Call to Action

If you manage messy folders, this app turns hours of manual sorting into minutes—while keeping you in control.
