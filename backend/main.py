from pathlib import Path
from io import BytesIO
import base64
import json
import math
import threading

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL_ID = "facebook/sam2-hiera-tiny"

app = FastAPI(title="Object Area AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

_model = None
_processor = None
_model_lock = threading.Lock()


def get_model():
    global _model, _processor
    if _model is None or _processor is None:
        with _model_lock:
            if _model is None or _processor is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                _processor = Sam2Processor.from_pretrained(MODEL_ID)
                _model = Sam2Model.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
                _model.eval()
    return _model, _processor


def image_to_data_url(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay_mask(image, mask):
    base = image.convert("RGBA")
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask] = [40, 200, 120, 105]
    result = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))

    edge = mask ^ (
        np.roll(mask, 1, 0) & np.roll(mask, -1, 0) &
        np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    )
    edges = np.zeros((h, w, 4), dtype=np.uint8)
    edges[edge] = [255, 255, 255, 230]
    return Image.alpha_composite(result, Image.fromarray(edges, "RGBA")).convert("RGB")


def calibrated_area(pixel_area, scale_points, reference_cm):
    if len(scale_points) != 2 or reference_cm <= 0:
        return None
    (x1, y1), (x2, y2) = scale_points
    px = math.hypot(x2 - x1, y2 - y1)
    if px <= 0:
        return None
    return pixel_area / ((px / reference_cm) ** 2)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.post("/api/segment")
async def segment(
    image: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...),
    scale_points: str = Form(""),
    reference_cm: str = Form(""),
):
    try:
        point_list = json.loads(points)
        label_list = json.loads(labels)
        if not point_list or len(point_list) != len(label_list):
            return JSONResponse({"error": "Add matching object points and labels."}, status_code=400)

        raw = await image.read()
        pil_image = Image.open(BytesIO(raw)).convert("RGB")
        width, height = pil_image.size
        clean_points, clean_labels = [], []
        for p, label in zip(point_list, label_list):
            x = max(0.0, min(float(p[0]), width - 1))
            y = max(0.0, min(float(p[1]), height - 1))
            clean_points.append([x, y])
            clean_labels.append(1 if int(label) == 1 else 0)

        if not any(clean_labels):
            return JSONResponse({"error": "At least one positive point is required."}, status_code=400)

        model, processor = get_model()
        device = next(model.parameters()).device
        inputs = processor(
            images=pil_image,
            input_points=[[clean_points]],
            input_labels=[clean_labels],
            return_tensors="pt",
        )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=True)

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0].numpy()
        while masks.ndim > 3:
            masks = masks[0]
        scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
        best = int(np.argmax(scores[:masks.shape[0]]))
        mask = masks[best].astype(bool)
        pixel_area = int(mask.sum())

        area_cm2 = None
        if scale_points.strip() and reference_cm.strip():
            try:
                area_cm2 = calibrated_area(pixel_area, json.loads(scale_points), float(reference_cm))
            except (ValueError, TypeError, json.JSONDecodeError):
                area_cm2 = None

        return {
            "width": width,
            "height": height,
            "pixel_area": pixel_area,
            "mask_percentage": round(pixel_area / (width * height) * 100, 3),
            "score": round(float(scores[best]), 4),
            "area_cm2": round(float(area_cm2), 4) if area_cm2 is not None else None,
            "result_image": image_to_data_url(overlay_mask(pil_image, mask)),
            "device": str(device),
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
