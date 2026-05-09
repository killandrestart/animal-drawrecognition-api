from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import os

from src.model import SketchCNN

app = FastAPI(title="QuickDraw AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "weights", "best_model.pt")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

classes = checkpoint["classes"]

model = SketchCNN(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()


def strokes_to_image(strokes, size=28):
    canvas = np.zeros((256, 256), dtype=np.uint8)

    for stroke in strokes:
        xs, ys = stroke

        for i in range(len(xs) - 1):
            x1, y1 = int(xs[i]), int(ys[i])
            x2, y2 = int(xs[i + 1]), int(ys[i + 1])

            cv2.line(canvas, (x1, y1), (x2, y2), 255, 8)

    coords = np.column_stack(np.where(canvas > 0))

    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        canvas = canvas[y_min:y_max, x_min:x_max]

    canvas = cv2.resize(canvas, (size, size))

    canvas = canvas.astype(np.float32) / 255.0

    return canvas


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict_strokes")
async def predict_strokes(data: dict):

    strokes = data["strokes"]

    img = strokes_to_image(strokes)

    x = torch.tensor(
        img.reshape(1, 1, 28, 28),
        dtype=torch.float32
    )

    with torch.no_grad():
        out = model(x)

        probs = torch.softmax(out, dim=1)

        top_probs, top_preds = torch.topk(probs, 3)

    results = []

    for prob, pred in zip(top_probs[0], top_preds[0]):
        results.append({
            "class": classes[pred.item()],
            "confidence": float(prob.item())
        })

    return {
        "predictions": results
    }
