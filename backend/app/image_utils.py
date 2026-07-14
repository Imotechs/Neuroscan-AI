import io
import numpy as np
from PIL import Image
import cv2

IMG_SIZE = 224

def load_image_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    if filename.lower().endswith(".dcm"):
        return _load_dicom(file_bytes)
    return _load_regular(file_bytes)

def _load_regular(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def _load_dicom(file_bytes: bytes) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    pixel_array = pixel_array.astype(np.uint8)
    if pixel_array.ndim == 2:
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
    elif pixel_array.ndim == 3 and pixel_array.shape[2] >= 3:
        pixel_array = pixel_array[:, :, :3]
    img = Image.fromarray(pixel_array).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def preprocess_for_model(pixel_array: np.ndarray) -> np.ndarray:
    return np.expand_dims(pixel_array, axis=0)
