# ImageArcAI

AI-powered system for automatic detection and description of visual elements in digitized archival photographs — turning weeks of manual archival description into an automated process.

🔗 **Live application:** https://imagearcai.ffzg.unizg.hr
📄 **Published paper:** Stančić, Sumpor, Trbušić — Zbornik 7. kongresa hrvatskih arhivista, 2025: https://had-info.hr/publikacije

---

## What it does

A user uploads a ZIP archive of historical portrait photographs. A trained YOLO object detection model processes each image and returns structured metadata (CSV), annotated images with bounding boxes and confidence scores, and aggregated class statistics. The system supports bilingual labelling (Croatian/English) and deletes all uploaded input after processing for privacy compliance.

The model recognizes **135 object classes** in a digitized collection of late-19th/early-20th century studio portraits.

---

## My contributions

I independently led the full training pipeline for YOLOv12s — model training (200 epochs, early stopping), evaluation, iteration, and per-class analysis — on the Supek supercomputer (Red Hat Enterprise Linux). I designed and built the FastAPI web application end-to-end, including the upload endpoint, inference pipeline, bilingual output, CSV generation, annotated image output, health monitoring, and security layer. I containerized the application with Docker for delivery to production and authored the full technical documentation (60+ page paradata document covering architecture, training methodology, evaluation metrics, security testing, and operational behavior).

As a prior development phase, YOLOv8s was trained and the Supek environment set up jointly with a fellow student (contributor listed in the published paper).

---

## Dataset

1,400+ digitized portraits from the State Archives in Osijek (HR-DAOS-2035), spanning the 1870s to early 20th century, organized into 135 object classes across 9 portrait categories. Training/validation split: 263 training images / 66 validation images.

Dataset labelling carried out by a team of students within the InterPARES Trust AI project (contributors listed in the published paper).

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python), Uvicorn (ASGI) |
| Model | Ultralytics YOLO (YOLOv12s, best.pt loaded once at startup) |
| Vision | PyTorch, OpenCV |
| Frontend | Jinja2 templates + React 18 (CDN) |
| Packaging | Docker (containerized for delivery to production) |
| Output | inference-results.csv + class-statistics.csv + annotated images |

---

## Security

File-type whitelisting, magic-byte validation (JPEG: FF D8 FF), protection against unsafe ZIP path traversal, automatic deletion of all input after processing.

---

## Project context

Developed within the international research project **InterPARES Trust AI** (2021–2026) at the Faculty of Humanities and Social Sciences, University of Zagreb, under the mentorship of prof. Hrvoje Stančić and Željko Trbušić (archival and research dimension). Deployment to the institutional server was handled by system administrators.

---

*Author: Josipa Sumpor — josipasumpor1@gmail.com*