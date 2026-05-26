"""
app.py  —  DermaScan AI
Flask entry point. Run this from the project ROOT:
    python app.py
"""

from flask import Flask, render_template, request, send_from_directory
import os
import base64
import uuid
from datetime import datetime
from PIL import Image
from io import BytesIO
from src.predict import predict_multi

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ─── Serve uploaded images ────────────────────────────────────────────────────
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ─── Main route ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    results    = None
    image_path = None

    if request.method == "POST":
        file     = request.files.get("file")
        captured = request.form.get("captured_image")
        models   = request.form.getlist("models")

        saved_path = None   
        # ── Case 1: File Upload ───────────────────────────────────────────────
        if file and file.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"{timestamp}_{file.filename}"
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            file.save(saved_path)

        # ── Case 2: Camera Capture ────────────────────────────────────────────
        elif captured:
            try:
                if "," in captured:
                    header, b64data = captured.split(",", 1)
                else:
                    b64data = captured

                image_data = base64.b64decode(b64data)
                image      = Image.open(BytesIO(image_data)).convert("RGB")

                unique_name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
                saved_path  = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                image.save(saved_path, "JPEG", quality=95)
                print(f"[DermaScan] Captured image saved → {saved_path}")

            except Exception as e:
                print(f"[DermaScan] Camera capture error: {e}")

        # ── Prediction ────────────────────────────────────────────────────────
        if saved_path and os.path.exists(saved_path):
            results    = predict_multi(saved_path, models)
            image_path = saved_path

    return render_template("index.html",
                           results=results,
                           image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)