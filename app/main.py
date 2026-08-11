# app/main.py
from pathlib import Path
import uuid
import zipfile
from typing import List, Optional, Dict, Any
import time
import csv
import json
import shutil
import datetime

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
RUNS_DIR = BASE_DIR / "runs"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
I18N_DIR = BASE_DIR / "app" / "i18n"

RUNS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best.pt"
EN_CLASSES_JSON = I18N_DIR / "classes.en.json"   # HR -> EN class-name map

# -----------------------------------------------------------------------
# Allowed image extensions
# -----------------------------------------------------------------------
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Upload limits (guard against oversized uploads and zip bombs)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50 MB per uploaded ZIP
MAX_IMAGES_PER_JOB = 200              # cap images processed in one job


def is_allowed_image_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTS


def verify_image_file(path: Path) -> bool:
    """
    Confirm the file really is an image (not just a spoofed extension).
    PIL verify() checks integrity without loading the whole image into memory.
    """
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------

app = FastAPI(title="ImageArcAI - object detection on archival portraits")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# -----------------------------------------------------------------------
# Class-name translations (HR -> EN)
# -----------------------------------------------------------------------

EN_TRANSLATIONS: dict = {}


def load_en_translations():
    global EN_TRANSLATIONS
    if EN_CLASSES_JSON.exists():
        try:
            with EN_CLASSES_JSON.open("r", encoding="utf-8") as f:
                EN_TRANSLATIONS = json.load(f)
            print(f"[I18N] Loaded EN translations from {EN_CLASSES_JSON}")
        except Exception as e:
            print(f"[I18N] Failed to load {EN_CLASSES_JSON}: {e}")
            EN_TRANSLATIONS = {}
    else:
        print(f"[I18N] Warning: {EN_CLASSES_JSON} not found, using Croatian names only.")
        EN_TRANSLATIONS = {}


load_en_translations()


def get_class_name(cls_id: int, csv_language: str, model_names) -> str:
    """Return the HR or EN class name for a class ID, depending on CSV language."""
    hr_name = model_names[cls_id]
    if csv_language.lower() == "en":
        return EN_TRANSLATIONS.get(hr_name, hr_name)
    return hr_name

# -----------------------------------------------------------------------
# YOLO model
# -----------------------------------------------------------------------

yolo_model: Optional[YOLO] = None


def get_yolo_model() -> YOLO:
    global yolo_model
    if yolo_model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        print(f"[YOLO] Loading model from {MODEL_PATH}")
        yolo_model = YOLO(str(MODEL_PATH))
    return yolo_model

# -----------------------------------------------------------------------
# Inference: one CSV row per image, plus raw detections for stats/bboxes
# -----------------------------------------------------------------------

def yolo_infer_on_images(image_paths: List[Path], csv_language: str):
    """
    Returns:
      - rows: list of dicts for the detections CSV (one row per image)
      - all_image_dets: per-image detection details for stats and bbox drawing
      - total_detections: total number of detections across the set
    """
    model = get_yolo_model()
    rows: List[Dict[str, Any]] = []
    all_image_dets: List[Dict[str, Any]] = []
    total_detections = 0

    for img_path in image_paths:
        results = model(str(img_path))
        r = results[0] if isinstance(results, list) else results

        names = r.names or model.names
        boxes = r.boxes

        class_ids: List[str] = []
        class_names: List[str] = []
        scores: List[str] = []
        bboxes: List[str] = []

        img_dets: List[Dict[str, Any]] = []

        for det in boxes:
            cls_id = int(det.cls.item())
            cls_name = get_class_name(cls_id, csv_language, names)

            conf = float(det.conf.item())
            x1, y1, x2, y2 = det.xyxy[0].tolist()

            class_ids.append(str(cls_id))
            class_names.append(cls_name)
            scores.append(f"{conf:.3f}")
            bboxes.append(f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")

            img_dets.append(
                {
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        total_detections += len(img_dets)

        row = {
            "image": img_path.name,
            "class_ids": ",".join(class_ids),
            "classes": ",".join(class_names),
            "scores": ",".join(scores),
            "bboxes": " | ".join(bboxes),
        }
        rows.append(row)

        all_image_dets.append(
            {
                "image_path": img_path,
                "detections": img_dets,
            }
        )

        print(f"[YOLO] {img_path.name}: {len(boxes)} detections")

    print(f"[YOLO] Images: {len(image_paths)}, detections: {total_detections}")
    return rows, all_image_dets, total_detections

# -----------------------------------------------------------------------
# CSV: detections 
# -----------------------------------------------------------------------

def write_detections_csv(output_path: Path, rows, csv_language: str = "hr"):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    
    if csv_language.lower() == "hr":
        header = "slika;nazivi_klasa;vjerojatnosti\n"
    else:
        header = "image;classes;scores\n"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        f.write(header)
        for row in rows:
            line = (
                str(row.get("image", "")).replace(";", ",")
                + ";" + str(row.get("classes", "")).replace(";", ",")
                + ";" + str(row.get("scores", "")).replace(";", ",")
                + "\n"
            )
            f.write(line)

# -----------------------------------------------------------------------
# CSV: statistics
# -----------------------------------------------------------------------

def write_stats_csv(output_path: Path, all_image_dets, csv_language: str = "hr"):
    counts: Dict[str, int] = {}

    for img in all_image_dets:
        for det in img["detections"]:
            name = det["cls_name"]
            counts[name] = counts.get(name, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_language.lower() == "hr":
        header = "klasa;broj_detekcija\n"
    else:
        header = "class;count\n"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        f.write(header)
        for cls_name in sorted(counts.keys()):
            f.write(f"{cls_name.replace(';', ',')};{counts[cls_name]}\n")

    print(f"[STATS] Wrote statistics to {output_path} ({len(counts)} classes)")

# -----------------------------------------------------------------------
# Bounding-box images (zipped)
# -----------------------------------------------------------------------

def create_bboxes_zip(job_dir: Path, all_image_dets):
    """Build a ZIP of images with drawn bounding boxes. Assumes at least one detection exists."""
    bboxes_dir = job_dir / "bboxes_images"
    bboxes_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()

    for img in all_image_dets:
        img_path: Path = img["image_path"]
        detections = img["detections"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[BBOX] Cannot open {img_path}: {e}")
            continue

        draw = ImageDraw.Draw(image)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["cls_name"]

            draw.rectangle([(x1, y1), (x2, y2)], outline=(79, 70, 229), width=2)

            text = cls_name
            text_size = draw.textbbox((0, 0), text, font=font)
            tw = text_size[2] - text_size[0]
            th = text_size[3] - text_size[1]
            rect_bg = (x1, y1 - th - 2, x1 + tw + 4, y1)
            draw.rectangle(rect_bg, fill=(79, 70, 229))
            draw.text((x1 + 2, y1 - th - 1), text, fill=(255, 255, 255), font=font)

        out_path = bboxes_dir / img_path.name
        try:
            image.save(out_path)
        except Exception as e:
            print(f"[BBOX] Cannot save {out_path}: {e}")

    zip_path = job_dir / "bboxes.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_file in bboxes_dir.iterdir():
            if img_file.is_file():
                zf.write(img_file, arcname=img_file.name)

    shutil.rmtree(bboxes_dir, ignore_errors=True)
    print(f"[BBOX] ZIP created: {zip_path}")

# -----------------------------------------------------------------------
# Job logging
# -----------------------------------------------------------------------

JOBS_LOG = LOGS_DIR / "jobs.csv"


def log_job(job_id: str, n_images: int, csv_language: str, duration_ms: float, status: str):
    is_new = not JOBS_LOG.exists()

    with JOBS_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        if is_new:
            writer.writerow(["timestamp", "job_id", "n_images", "csv_language", "duration_ms", "status"])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            job_id,
            n_images,
            csv_language,
            f"{duration_ms:.1f}",
            status,
        ])

    print(f"[LOG] Job {job_id}: {n_images} slika, {duration_ms:.1f} ms, csv_language={csv_language}, status={status}")

# -----------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------

def cleanup_job(job_dir: Path):
    images_dir = job_dir / "images"
    zip_path = job_dir / "input.zip"

    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)
        print(f"[CLEANUP] Removed directory {images_dir}")

    if zip_path.exists():
        try:
            zip_path.unlink()
            print(f"[CLEANUP] Removed ZIP {zip_path}")
        except Exception as e:
            print(f"[CLEANUP] Error removing {zip_path}: {e}")

# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.get("/")
async def root():
    return RedirectResponse(url="/upload")


@app.get("/upload", response_class=HTMLResponse)
async def show_form(request: Request):
    ui_lang = request.query_params.get("lang", "hr").lower()
    if ui_lang not in ("hr", "en"):
        ui_lang = "hr"

    default_csv = "hr" if ui_lang == "hr" else "en"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "job_id": None,
            "ui_language": ui_lang,
            "csv_language": default_csv,
            "n_images": None,
            "duration_ms": None,
            "has_detections": False,
            "has_stats": False,
            "has_bboxes": False,
            "total_detections": None,
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(
    request: Request,
    zip_file: UploadFile = File(...),
    csv_language: str = Form("hr", alias="language"),
    outputs: List[str] = Form(default=["detections"], alias="outputs"),
):
    start_time = time.perf_counter()
    duration_ms: float = 0.0

    ui_lang = request.query_params.get("lang", "hr").lower()
    if ui_lang not in ("hr", "en"):
        ui_lang = "hr"

    if not outputs:
        outputs = ["detections"]

    job_id = uuid.uuid4().hex[:8]
    job_dir = RUNS_DIR / job_id
    images_dir = job_dir / "images"
    csv_path = job_dir / "results.csv"
    stats_path = job_dir / "stats.csv"

    images_dir.mkdir(parents=True, exist_ok=True)

    # save the uploaded ZIP
    zip_path = job_dir / "input.zip"
    with zip_path.open("wb") as f:
        content = await zip_file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            )
        f.write(content)

    # extract (allowed extensions only, then verify each is a real image)
    image_paths: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue

            filename = Path(member.filename).name
            if not filename:
                continue

            # filter by extension
            if not is_allowed_image_filename(filename):
                continue

            out_path = images_dir / filename
            with zip_ref.open(member) as src, out_path.open("wb") as dst:
                dst.write(src.read())

            # verify it is really an image; if not, delete and skip
            if not verify_image_file(out_path):
                try:
                    out_path.unlink()
                except Exception:
                    pass
                continue

            image_paths.append(out_path)

            if len(image_paths) >= MAX_IMAGES_PER_JOB:
                # stop early once the per-job image cap is reached
                break

    # no valid images: stop with an error
    if not image_paths:
        raise HTTPException(
            status_code=400,
            detail="ZIP contains no valid images (.jpg/.jpeg/.png/.tif/.tiff)."
        )

    status = "ok"
    total_detections = 0
    try:
        detections_rows, all_image_dets, total_detections = yolo_infer_on_images(
            image_paths, csv_language=csv_language
        )

        if "detections" in outputs:
            write_detections_csv(csv_path, detections_rows, csv_language=csv_language)

        if "stats" in outputs:
            write_stats_csv(stats_path, all_image_dets, csv_language=csv_language)

        if "bboxes" in outputs and total_detections > 0:
            create_bboxes_zip(job_dir, all_image_dets)

    except Exception as e:
        status = "error"
        print(f"[ERROR] Job {job_id} failed: {e}")
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        log_job(job_id, len(image_paths), csv_language, duration_ms, status)
        cleanup_job(job_dir)

    has_detections = "detections" in outputs
    has_stats = "stats" in outputs
    has_bboxes = "bboxes" in outputs and total_detections > 0

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "job_id": job_id,
            "ui_language": ui_lang,
            "csv_language": csv_language,
            "n_images": len(image_paths),
            "duration_ms": duration_ms,
            "has_detections": has_detections,
            "has_stats": has_stats,
            "has_bboxes": has_bboxes,
            "total_detections": total_detections,
        },
    )

# -----------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------

@app.get("/results/{job_id}")
async def download_results(job_id: str):
    csv_path = RUNS_DIR / job_id / "results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"results_{job_id}.csv",
    )


@app.get("/results/{job_id}/stats")
async def download_stats(job_id: str):
    stats_path = RUNS_DIR / job_id / "stats.csv"
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="Stats not found")

    return FileResponse(
        stats_path,
        media_type="text/csv",
        filename=f"stats_{job_id}.csv",
    )


@app.get("/results/{job_id}/bboxes")
async def download_bboxes(job_id: str):
    zip_path = RUNS_DIR / job_id / "bboxes.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="BBox images not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"bboxes_{job_id}.zip",
    )
