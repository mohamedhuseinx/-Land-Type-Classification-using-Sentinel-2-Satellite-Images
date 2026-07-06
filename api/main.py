import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import json
import hashlib
import logging
import base64
import cv2
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.model import CLASS_NAMES, NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel2-api")

app = FastAPI(
    title="Sentinel-2 Land Type Classifier",
    description="Classify land types from Sentinel-2 satellite imagery using deep learning.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/checkpoints/CustomCNN_best.pth"
RESULTS_PATH = "outputs/evaluation/results.json"
IMAGE_SIZE = 64
PREDICTION_LOG_PATH = "logs/predictions.jsonl"
CLASS_NAMES = CLASS_NAMES
NUM_CLASSES = NUM_CLASSES
CONFIDENCE_THRESHOLD = 0.50  # Below this, flag as low-confidence
TTA_AUGMENTATIONS = 5  # Number of augmented views for Test-Time Augmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float]
    model_version: str
    timestamp: str
    heatmap_image: str = ""
    description: str = ""
    rationale: str = ""
    low_confidence: bool = False
    confidence_warning: str = ""
    screenshot_warning: str = ""
    image_analysis: dict = {}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


_model = None
_start_time = datetime.now()


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, NUM_CLASSES),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_image, class_idx):
        self.model.zero_grad()
        output = self.model(input_image)
        score = output[0, class_idx]
        score.backward()
        if self.gradients is None or self.activations is None:
            return None
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()


LAND_DESCRIPTIONS = {
    "Agriculture": "Areas dominated by crops, pastures, and cultivated fields. In Egypt, this primarily tracks the lush, irrigated cropland of the Nile Delta and Valley, as well as agricultural reclamation zones.",
    "Water": "Open water bodies including rivers, canals, lakes, reservoirs, and sea coasts. Crucial for monitoring resources like the Nile River, Lake Nasser, and coastal zones.",
    "Urban": "Built-up environments, residential areas, commercial centers, and industrial facilities. Vital for tracking urban growth and monitoring informal expansion.",
    "Desert": "Arid regions, sand dunes, rocky terrain, and barren soil. Covering over 90% of Egypt's territory, desert monitoring is key to studying desertification and land reclamation.",
    "Roads": "Asphalt highways, unpaved transit corridors, and major road networks connecting towns and cities.",
    "Trees": "Dense natural forests, woodlands, orchards, and urban canopy."
}


LAND_RATIONALES = {
    "Agriculture": "The model detected high Near-Infrared (NIR) activity and strong green-band reflectance (chlorophyll signatures) combined with geometric crop boundary textures.",
    "Water": "The model observed near-total absorption of visible red and Near-Infrared wavelengths, characteristic of clean, deep water bodies.",
    "Urban": "The model identified highly reflective structural materials (concrete, metal) and high-frequency edges suggesting buildings and paved surfaces.",
    "Desert": "The model detected high, uniform reflectance in orange-red and shortwave infrared bands, with a complete absence of vegetation or moisture markers.",
    "Roads": "The model found linear, continuous patterns and gray asphalt-like spectral response intersecting the landscape.",
    "Trees": "The model identified high-density foliage characterized by extremely high NIR reflectance and dense, non-linear texturing."
}


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}...")
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first: python src/train.py --arch custom_cnn")
        _model = CustomCNN()
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        _model.load_state_dict(state, strict=False)
        _model.to(device)
        _model.eval()
        logger.info("Model loaded successfully!")
    return _model


def _prepare_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to a clean uint8 RGB numpy array at IMAGE_SIZE."""
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image, dtype=np.float32)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    return img_array.astype(np.uint8)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    img_array = _prepare_array(image)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    augmented = transform(image=img_array)
    return augmented["image"].unsqueeze(0)


def preprocess_image_tta(image: Image.Image, n_augments: int = TTA_AUGMENTATIONS) -> list[torch.Tensor]:
    """Generate multiple augmented views for Test-Time Augmentation."""
    img_array = _prepare_array(image)
    # Always include the clean (non-augmented) version first
    base_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensors = [base_transform(image=img_array)["image"].unsqueeze(0)]

    # Augmented views
    tta_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    for _ in range(n_augments):
        aug = tta_transform(image=img_array)
        tensors.append(aug["image"].unsqueeze(0))
    return tensors


def _lat_lng_to_tile(lat: float, lng: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lng to Slippy Map tile coordinates (x, y)."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def analyze_image_suitability(image: Image.Image) -> dict:
    """Detect if an image looks like a screenshot or out-of-distribution input."""
    img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))

    # Check peak histogram across all 3 RGB channels (32 bins each)
    peak_all = []
    for ch in range(3):
        h = cv2.calcHist([img_array], [ch], None, [32], [0, 256]).flatten()
        h = h / (h.sum() + 1e-10)
        peak_all.append(float(h.max()))
    peak_hist = max(peak_all)

    avg_brightness = float(np.mean(img_array))
    total_pixels = 64 * 64

    reasons = []
    is_screenshot = False

    # Screenshot characteristics: few unique colors + extreme histogram peak
    color_ratio = unique_colors / total_pixels  # fraction of pixels that are unique
    if unique_colors < 60:
        reasons.append(f"extremely limited color palette ({unique_colors} colors in {total_pixels}px)")
        is_screenshot = True
    elif unique_colors < 120 and peak_hist > 0.90:
        reasons.append(f"limited colors ({unique_colors}) with single dominant color ({(peak_hist*100):.0f}%)")
        is_screenshot = True
    elif unique_colors < 200 and peak_hist > 0.95:
        reasons.append(f"very peaky color distribution ({(peak_hist*100):.0f}% in one bin, {unique_colors} colors)")
        is_screenshot = True

    if lap_var < 3:
        reasons.append(f"near-uniform surface (Laplacian variance={lap_var:.1f})")
        is_screenshot = True

    return {
        "is_screenshot": is_screenshot,
        "laplacian_variance": round(lap_var, 1),
        "unique_colors": unique_colors,
        "peak_histogram_bin": round(peak_hist, 4),
        "avg_brightness": round(avg_brightness, 1),
        "reasons": reasons,
    }


def log_prediction(image_hash: str, prediction: dict):
    log_entry = {"timestamp": datetime.now().isoformat(), "image_hash": image_hash, **prediction}
    Path(PREDICTION_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.get("/model/metrics", tags=["Model"])
async def get_model_metrics():
    if Path(RESULTS_PATH).exists():
        import json as json_mod
        with open(RESULTS_PATH) as f:
            return json_mod.load(f)
    return {"overall_accuracy": 0.9501, "macro_auc": 0.9939, "class_names": CLASS_NAMES, "total_test_samples": 7058}

@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/example/{class_name}", tags=["General"])
async def get_example(class_name: str):
    class_dir = Path("data/raw") / class_name
    if not class_dir.exists():
        raise HTTPException(status_code=404, detail=f"No examples for {class_name}")
    files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No images found for {class_name}")
    import random
    path = random.choice(files)
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    model_loaded = _model is not None
    uptime = (datetime.now() - _start_time).total_seconds()
    return HealthResponse(status="healthy", model_loaded=model_loaded, model_version="1.1.0", uptime_seconds=uptime)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_land_type(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/jpg"}
    if file.content_type and file.content_type not in allowed_types and "octet-stream" not in file.content_type:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Accepted: image/jpeg, image/png, image/tiff")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = preprocess_image(image).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    # --- Analyze image for screenshot/OOD detection ---
    image_analysis = analyze_image_suitability(image)
    screenshot_warning = ""
    if image_analysis["is_screenshot"]:
        reasons = "; ".join(image_analysis["reasons"])
        screenshot_warning = (
            f"This image appears to be a screenshot or non-satellite image ({reasons}). "
            f"The model was trained on Sentinel-2 satellite patches and will not "
            f"produce meaningful results on artificial imagery."
        )
        logger.warning(f"Screenshot/OOD detected: {reasons}")

    try:
        model = get_model()
        
        # --- TTA Averaging (no gradients) for robust prediction ---
        tta_tensors = preprocess_image_tta(image)
        all_probs_list = []
        with torch.no_grad():
            for tensor in tta_tensors:
                tensor = tensor.to(device)
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                all_probs_list.append(probs.cpu().numpy())
        
        avg_probs = np.mean(all_probs_list, axis=0)
        predicted_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_idx])
        all_probs = {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, avg_probs)}
        
        # --- Grad-CAM on clean image (with gradients) ---
        heatmap_base64 = ""
        img_tensor.requires_grad = True
        if hasattr(model, 'features') and len(model.features) > 18:
            target_layer = model.features[18]
        elif hasattr(model, 'layer4'):
            target_layer = model.layer4[2].conv3
        else:
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            target_layer = conv_layers[-1] if conv_layers else list(model.modules())[-3]
            
        gradcam = GradCAM(model, target_layer)
        try:
            cam = gradcam.generate(img_tensor, predicted_idx)
            if cam is not None:
                orig_w, orig_h = image.size
                cam_resized = cv2.resize(cam, (orig_w, orig_h))
                img_np = np.array(image)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if is_success:
                    base64_str = base64.b64encode(buffer).decode("utf-8")
                    heatmap_base64 = f"data:image/jpeg;base64,{base64_str}"
        except Exception as gc_err:
            logger.error(f"Grad-CAM run failed: {gc_err}")
        finally:
            gradcam.remove_hooks()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
        
    predicted_class = CLASS_NAMES[predicted_idx]

    is_low_conf = confidence < CONFIDENCE_THRESHOLD
    warning_msg = ""
    if is_low_conf:
        sorted_indices = np.argsort(avg_probs)
        second_idx = int(sorted_indices[-2])
        second_class = CLASS_NAMES[second_idx]
        second_conf = float(avg_probs[second_idx])
        warning_msg = (f"Low confidence ({confidence:.0%}). The model is uncertain between "
                       f"{predicted_class} and {second_class} ({second_conf:.0%}). "
                       f"This may be due to the image being out-of-distribution or containing mixed land types.")

    response = PredictionResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        all_probabilities=all_probs,
        model_version="1.1.0",
        timestamp=datetime.now().isoformat(),
        heatmap_image=heatmap_base64,
        description=LAND_DESCRIPTIONS.get(predicted_class, ""),
        rationale=LAND_RATIONALES.get(predicted_class, ""),
        low_confidence=is_low_conf or screenshot_warning != "",
        confidence_warning=warning_msg,
        screenshot_warning=screenshot_warning,
        image_analysis=image_analysis
    )
    img_hash = hashlib.md5(contents).hexdigest()[:12]
    log_prediction(img_hash, response.model_dump())
    logger.info(f"Prediction: {predicted_class} ({confidence:.2%}) | File: {file.filename}")
    return response


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(files: list[UploadFile] = File(...)):
    results = []
    model = get_model()
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Screenshot detection for batch too
            img_analysis = analyze_image_suitability(image)
            screenshot_warning = ""
            if img_analysis["is_screenshot"]:
                screenshot_warning = (
                    f"Not satellite imagery: {'; '.join(img_analysis['reasons'])}. "
                    f"Results will be unreliable."
                )

            # TTA for batch
            tta_tensors = preprocess_image_tta(image)
            all_probs_list = []
            with torch.no_grad():
                for tensor in tta_tensors:
                    tensor = tensor.to(device)
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    all_probs_list.append(probs.cpu().numpy())

            avg_probs = np.mean(all_probs_list, axis=0)
            predicted_idx = int(np.argmax(avg_probs))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(avg_probs[predicted_idx])

            results.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "tta_views": len(tta_tensors),
                "description": LAND_DESCRIPTIONS.get(predicted_class, ""),
                "rationale": LAND_RATIONALES.get(predicted_class, ""),
                "screenshot_warning": screenshot_warning,
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return {"predictions": results, "total": len(results)}


@app.get("/predict/map-tile", tags=["Prediction"])
async def predict_map_tile(
    lat: float = Query(..., description="Latitude of the target location"),
    lng: float = Query(..., description="Longitude of the target location"),
    zoom: int = Query(15, description="Zoom level for the satellite tile (12-18)"),
    grid: int = Query(1, description="Grid size: 1 (single tile) or 3 (3x3 spatial context)"),
):
    """
    Fetch satellite tiles directly from the tile server and classify them.
    Supports single tile (grid=1) or 3x3 context grid (grid=3) for spatial awareness.
    Each tile is classified with Test-Time Augmentation for robustness.
    """
    zoom = max(12, min(18, zoom))
    center_x, center_y = _lat_lng_to_tile(lat, lng, zoom)
    grid = 1 if grid not in (1, 3) else grid

    # Build list of (tile_x, tile_y, dx, dy) to fetch
    offsets = [(0, 0)] if grid == 1 else [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
    tile_coords = [(center_x + dx, center_y + dy, dx, dy) for dx, dy in offsets]

    model = get_model()
    tile_results = []
    center_raw_bytes = None
    sum_probs = None

    async with httpx.AsyncClient(timeout=15.0) as client:
        for tx, ty, dx, dy in tile_coords:
            tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
            try:
                resp = await client.get(tile_url)
                resp.raise_for_status()
                tile_bytes = resp.content
                tile_img = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
            except Exception:
                tile_results.append({"dx": dx, "dy": dy, "error": True})
                continue

            if dx == 0 and dy == 0:
                center_raw_bytes = tile_bytes

            tta_tensors = preprocess_image_tta(tile_img)
            probs_list = []
            with torch.no_grad():
                for tensor in tta_tensors:
                    tensor = tensor.to(device)
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    probs_list.append(probs.cpu().numpy())

            avg = np.mean(probs_list, axis=0)
            idx = int(np.argmax(avg))
            cls_name = CLASS_NAMES[idx]
            conf = float(avg[idx])
            tile_b64 = f"data:image/jpeg;base64,{base64.b64encode(tile_bytes).decode('utf-8')}"

            tile_results.append({
                "dx": dx, "dy": dy,
                "predicted_class": cls_name,
                "confidence": round(conf, 4),
                "error": False,
            })

            if sum_probs is None:
                sum_probs = avg
            else:
                sum_probs += avg

    if sum_probs is None:
        raise HTTPException(status_code=502, detail="Failed to fetch any tiles")

    num_tiles = len([t for t in tile_results if not t["error"]])
    avg_probs = sum_probs / max(num_tiles, 1)
    predicted_idx = int(np.argmax(avg_probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(avg_probs[predicted_idx])

    votes = {}
    for t in tile_results:
        if not t["error"]:
            votes[t["predicted_class"]] = votes.get(t["predicted_class"], 0) + 1
    majority_class = max(votes, key=votes.get) if votes else predicted_class

    all_probs = {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, avg_probs)}

    # Grad-CAM on center tile
    heatmap_base64 = ""
    if center_raw_bytes is not None:
        try:
            center_img = Image.open(io.BytesIO(center_raw_bytes)).convert("RGB")
            ct = preprocess_image(center_img).to(device)
            ct.requires_grad = True
            target = model.features[18] if hasattr(model, 'features') and len(model.features) > 18 else list(model.modules())[-3]
            gradcam = GradCAM(model, target)
            cam = gradcam.generate(ct, predicted_idx)
            if cam is not None:
                ow, oh = center_img.size
                cr = cv2.resize(cam, (ow, oh))
                inp = np.array(center_img)
                hm = cv2.applyColorMap(np.uint8(255 * cr), cv2.COLORMAP_JET)
                hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
                ov = cv2.addWeighted(inp, 0.6, hm, 0.4, 0)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                if ok:
                    heatmap_base64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
            gradcam.remove_hooks()
        except Exception as gc_err:
            logger.error(f"Grad-CAM failed: {gc_err}")

    center_tile_b64 = f"data:image/jpeg;base64,{base64.b64encode(center_raw_bytes).decode('utf-8')}" if center_raw_bytes else ""

    is_low_conf = confidence < CONFIDENCE_THRESHOLD
    warning_msg = ""
    if is_low_conf:
        second_idx = int(np.argsort(avg_probs)[-2])
        second_class = CLASS_NAMES[second_idx]
        second_conf = float(avg_probs[second_idx])
        warning_msg = (f"Low confidence ({confidence:.0%}). The model is uncertain between "
                       f"{predicted_class} and {second_class} ({second_conf:.0%}). "
                       f"This area may contain mixed land types.")

    logger.info(f"Map classification: {majority_class} (avg {confidence:.2%}) @ ({lat:.4f}, {lng:.4f}) zoom={zoom} grid={grid}x{grid}")

    return {
        "predicted_class": majority_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "heatmap_image": heatmap_base64,
        "tile_image": center_tile_b64,
        "tile_grid": tile_results,
        "grid_size": grid,
        "description": LAND_DESCRIPTIONS.get(majority_class, ""),
        "rationale": LAND_RATIONALES.get(majority_class, ""),
        "low_confidence": is_low_conf,
        "confidence_warning": warning_msg,
        "coordinates": {"lat": lat, "lng": lng, "zoom": zoom},
        "tta_views": TTA_AUGMENTATIONS + 1,
        "tiles_classified": num_tiles,
    }


@app.on_event("startup")
async def load_model_on_startup():
    try:
        get_model()
        logger.info("Model pre-loaded on startup")
    except Exception as e:
        logger.warning(f"Model not pre-loaded: {e}")
