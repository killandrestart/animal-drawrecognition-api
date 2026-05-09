from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import os

from src.model import SketchCNN

app = FastAPI()

# 🌐 CORS (чтобы браузер не истерил)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 пути
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

checkpoint = torch.load(
    os.path.join(BASE_DIR, "weights", "best_model.pt"),
    map_location="cpu"
)

classes = checkpoint["classes"]

model = SketchCNN(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()


# 🧠 preprocessing (КЛЮЧЕВАЯ ЧАСТЬ)
def preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Empty or invalid image")

    # 🔥 1. resize (важно сохранить структуру)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 🔥 2. invert (QuickDraw style)
    img = 255 - img

    # 🔥 3. normalize
    img = img.astype(np.float32) / 255.0

    # 🔥 4. optional: slight centering boost (упрощённый вариант)
    coords = np.column_stack(np.where(img > 0.2))
    if len(coords) > 0:
        y_mean, x_mean = coords.mean(axis=0)
        shift_y = int(14 - y_mean)
        shift_x = int(14 - x_mean)

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (28, 28))

    # 🔥 5. reshape for CNN
    img = img.reshape(1, 1, 28, 28)

    return torch.tensor(img, dtype=torch.float32)

def strokes_to_image(strokes, size=28):
    canvas = np.zeros((256, 256), dtype=np.uint8)

    for stroke in strokes:
        xs, ys = stroke
        for i in range(len(xs)-1):
            x1, y1 = int(xs[i]), int(ys[i])
            x2, y2 = int(xs[i+1]), int(ys[i+1])
            cv2.line(canvas, (x1, y1), (x2, y2), 255, 8)

    coords = np.column_stack(np.where(canvas > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        canvas = canvas[y_min:y_max, x_min:x_max]

    canvas = cv2.resize(canvas, (size, size))
    canvas = canvas.astype(np.float32) / 255.0

    return canvas

# 🤖 PREDICT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    x = preprocess(img_bytes)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)

        conf, pred = torch.max(prob, 1)

    confidence = float(conf.item())
    prediction = classes[pred.item()]

    # 🚨 защита от мусорных предсказаний
    if confidence < 0.35:
        prediction = "unknown"

    return {
        "prediction": prediction,
        "confidence": confidence
    }

@app.post("/predict_strokes")
async def predict_strokes(data: dict):
    strokes = data["strokes"]

    img = strokes_to_image(strokes)

    x = torch.tensor(img.reshape(1,1,28,28), dtype=torch.float32)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)

    return {
        "prediction": classes[pred.item()],
        "confidence": float(conf.item())
    }
# 🎮 CHECK MODE
@app.post("/check")
async def check(file: UploadFile = File(...), target: str = ""):
    img_bytes = await file.read()
    x = preprocess(img_bytes)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    result = classes[pred]

    return {
        "target": target,
        "prediction": result,
        "correct": result == target
    }
