# hw9_model.py
import numpy as np
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image
import sys
import json

# ---------------------
# Helpers (download + resize)
# ---------------------
def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img: Image.Image, target_size=(200, 200)):
    # ensure RGB and exact resizing (Image.NEAREST used in HW9 text)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# ---------------------
# Preprocessing exactly as in HW8
# ---------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = (200, 200)   # from HW8

def preprocess_for_model(img: Image.Image, target_size=TARGET_SIZE):
    """
    Returns numpy array shaped (1, C, H, W) float32,
    using the exact pipeline used in HW8:
      - resize to TARGET_SIZE
      - convert to numpy float32
      - divide by 255.0  (like torchvision.ToTensor)
      - normalize using MEAN/STD
      - transpose HWC -> CHW
      - add batch dimension
    """
    img = prepare_image(img, target_size)
    arr = np.array(img).astype(np.float32)  # shape (H, W, C), values 0..255
    arr = arr / 255.0                        # 0..1
    # normalize per-channel
    arr = (arr - MEAN[None, None, :]) / STD[None, None, :]
    # to NCHW
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    arr = np.expand_dims(arr, 0)  # (1, C, H, W)
    return arr

# ---------------------
# Q1: inspect model IO
# ---------------------
def inspect_model(model_path: str):
    sess = ort.InferenceSession(model_path)
    print("Inputs:")
    for i in sess.get_inputs():
        print("  name:", i.name, "shape:", i.shape, "type:", i.type)
    print("Outputs:")
    for o in sess.get_outputs():
        print("  name:", o.name, "shape:", o.shape, "type:", o.type)

# ---------------------
# Q3: first pixel, R channel after preprocess
# ---------------------
def first_pixel_r_value(url: str, target_size=TARGET_SIZE):
    img = download_image(url)
    X = preprocess_for_model(img, target_size)
    # X shape: (1, C, H, W)
    val = float(X[0, 0, 0, 0])  # batch 0, channel R (index 0), row 0, col 0
    print("First pixel R after preprocessing:", val)
    return val

# ---------------------
# Q4: run model and print result
# ---------------------
def run_model_on_url(model_path: str, url: str):
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    img = download_image(url)
    X = preprocess_for_model(img, TARGET_SIZE)
    preds = sess.run([output_name], {input_name: X})
    # preds is usually a list; extract scalar
    out = np.array(preds[0]).squeeze()
    print("Raw model output:", out)
    # if output is multi-dim, you can inspect shape:
    print("Output shape:", np.array(preds[0]).shape)
    return float(np.array(preds[0]).reshape(-1)[0])

# ---------------------
# CLI: make it easy to call actions
# ---------------------
def usage():
    print("Usage:")
    print("  python model_q1_q4.py inspect <model.onnx>")
    print("  python model_q1_q4.py first_pixel <image_url>")
    print("  python model_q1_q4.py predict <model.onnx> <image_url>")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "inspect":
        if len(sys.argv) != 3:
            usage(); sys.exit(1)
        inspect_model(sys.argv[2])
    elif cmd == "first_pixel":
        if len(sys.argv) != 3:
            usage(); sys.exit(1)
        first_pixel_r_value(sys.argv[2])
    elif cmd == "predict":
        if len(sys.argv) != 4:
            usage(); sys.exit(1)
        model_path = sys.argv[2]
        url = sys.argv[3]
        score = run_model_on_url(model_path, url)
        print("Scalar score:", score)
    else:
        usage()
