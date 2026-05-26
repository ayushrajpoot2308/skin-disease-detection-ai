import os
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_builder import load_or_build

CLASSES_PATH = os.path.join("models", "classes.json")


def _load_class_names():
    if not os.path.exists(CLASSES_PATH):
        print(f"[predict] WARNING: '{CLASSES_PATH}' not found. Run python src/train.py first.")
        return {}
    with open(CLASSES_PATH) as f:
        class_indices = json.load(f)
    return {int(v): k for k, v in class_indices.items()}


CLASS_NAMES: dict = _load_class_names()
_model_cache: dict = {}


def get_model(name: str):
    global CLASS_NAMES
    if not CLASS_NAMES:
        CLASS_NAMES = _load_class_names()
    if not CLASS_NAMES:
        raise RuntimeError(f"Run python src/train.py first to generate '{CLASSES_PATH}'.")
    if name not in _model_cache:
        _model_cache[name] = load_or_build(name, num_classes=len(CLASS_NAMES))
    return _model_cache[name]


def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_single(image_path: str, model_name: str) -> dict:
    model = get_model(model_name)
    preds = model.predict(preprocess(image_path), verbose=0)[0]
    idx   = int(np.argmax(preds))
    return {
        "model":      model_name,
        "prediction": CLASS_NAMES.get(idx, "Unknown"),
        "confidence": round(float(preds[idx]) * 100, 2),
    }


def predict_multi(image_path: str, model_names: list) -> list:
    global CLASS_NAMES
    if not CLASS_NAMES:
        CLASS_NAMES = _load_class_names()
    results = []
    for name in model_names:
        try:
            results.append(predict_single(image_path, name))
        except Exception as exc:
            print(f"[predict] Error with {name}: {exc}")
            results.append({"model": name, "prediction": "Error",
                            "confidence": 0.0, "error": str(exc)})
    return results