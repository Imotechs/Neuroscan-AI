import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .class_names import CLASS_NAMES
from .image_utils import load_image_bytes, preprocess_for_model
from .model_loader import load_my_model, load_transfer_model
from .gradcam import generate_ensemble_heatmaps
from .explain import ask_deepseek

app = FastAPI(title="NeuroScan AI - Brain Tumor Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

my_model = None
transfer_model = None
MY_CNN_LAST_CONV = "conv2d_2"
TRANSFER_LAST_CONV = "out_relu"


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    all_confidence: dict[str, float]
    analysis_image: str
    overlay_image: str


class ExplainRequest(BaseModel):
    message: str
    prediction: str
    confidence: float
    all_confidence: dict[str, float]


class ExplainResponse(BaseModel):
    reply: str


@app.on_event("startup")
def load_models():
    global my_model, transfer_model
    my_model = load_my_model()
    transfer_model = load_transfer_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
    contents = file.file.read()
    img_array = load_image_bytes(contents, file.filename)
    img_batch = preprocess_for_model(img_array)

    preds_my = my_model.predict(img_batch, verbose=0)
    preds_transfer = transfer_model.predict(img_batch, verbose=0)

    avg_probs = (preds_my[0] + preds_transfer[0]) / 2.0
    class_idx = int(avg_probs.argmax())
    confidence = float(avg_probs[class_idx])
    prediction_label = CLASS_NAMES[class_idx]
    all_confidence = {
        name: float(avg_probs[i]) for i, name in enumerate(CLASS_NAMES)
    }

    models_config = [
        (my_model, MY_CNN_LAST_CONV),
        (transfer_model, TRANSFER_LAST_CONV),
    ]
    analysis_image, overlay_image = generate_ensemble_heatmaps(
        models_config, img_batch, class_idx
    )

    return PredictResponse(
        prediction=prediction_label,
        confidence=confidence,
        all_confidence=all_confidence,
        analysis_image=analysis_image,
        overlay_image=overlay_image,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(body: ExplainRequest):
    reply = await ask_deepseek(
        user_message=body.message,
        prediction=body.prediction,
        confidence=body.confidence,
        all_confidence=body.all_confidence,
    )
    return ExplainResponse(reply=reply)
