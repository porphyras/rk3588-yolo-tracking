import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

# Пути настраиваются переменными окружения; по умолчанию — структура этого репозитория
REPO = Path(__file__).resolve().parent.parent
TOOLS = Path(os.environ.get("TOOLS", REPO / "tools"))  # клоны airockchip и rknn_model_zoo
sys.path.insert(0, str(REPO / "v3_processes"))
OUT = str(REPO / "v5_models" / "models")
import detect as z

ROOT = Path(os.environ.get("COCO_EVAL", TOOLS / "coco_eval"))
ZOO = str(TOOLS / "rknn_model_zoo")
COCO_DIR = f"{ZOO}/datasets/COCO"
VIDEO_LIST = str(REPO / "conversion" / "kalibrovka" / "spisok.txt")
CONF = 0.001
MAX_DET = 100
ALL = np.ones(80, dtype=bool)

with open(f"{COCO_DIR}/coco_subset_20.txt") as src, open("/tmp/coco_abs.txt", "w") as dst:
    for line in src:
        if line.strip():
            dst.write(f"{COCO_DIR}/{line.strip().removeprefix('./')}\n")

ids = [l.strip() for l in open(ROOT / "ids.txt") if l.strip()]
images, gts = [], {}
for i in ids:
    img = cv2.imread(str(ROOT / "images" / f"{i}.jpg"))
    if img is None:
        continue
    h, w = img.shape[:2]
    rows = np.loadtxt(ROOT / "coco/labels/val2017" / f"{i}.txt", ndmin=2)
    cls = rows[:, 0].astype(int)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    boxes = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), axis=1).astype(np.float32)
    images.append((i, img))
    gts[i] = (boxes, cls)
print(f"изображений: {len(images)}, объектов в разметке: {sum(len(g[1]) for g in gts.values())}")


def iou_vec(b, g):
    x1 = np.maximum(b[0], g[:, 0])
    y1 = np.maximum(b[1], g[:, 1])
    x2 = np.minimum(b[2], g[:, 2])
    y2 = np.minimum(b[3], g[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (area_b + area_g - inter + 1e-9)


def evaluate(dets):
    thresholds = np.linspace(0.5, 0.95, 10)
    aps = np.full((80, 10), np.nan)
    for c in range(80):
        gt_c = {i: g[0][g[1] == c] for i, g in gts.items()}
        n_gt = sum(len(v) for v in gt_c.values())
        if n_gt == 0:
            continue
        d_c = sorted(((s, i, b) for i, cl, s, b in dets if cl == c), key=lambda x: -x[0])
        if not d_c:
            aps[c] = 0.0
            continue
        tp = np.zeros((10, len(d_c)))
        used = {i: np.zeros((10, len(v)), dtype=bool) for i, v in gt_c.items()}
        for k, (s, i, b) in enumerate(d_c):
            g = gt_c[i]
            if not len(g):
                continue
            ious = iou_vec(b, g)
            for ti, t in enumerate(thresholds):
                cand = np.where((ious >= t) & ~used[i][ti])[0]
                if len(cand):
                    j = cand[np.argmax(ious[cand])]
                    used[i][ti, j] = True
                    tp[ti, k] = 1
        for ti in range(10):
            ctp = np.cumsum(tp[ti])
            cfp = np.cumsum(1 - tp[ti])
            rec = ctp / n_gt
            prec = np.maximum.accumulate((ctp / np.maximum(ctp + cfp, 1e-9))[::-1])[::-1]
            idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
            aps[c, ti] = np.mean([prec[j] if j < len(prec) else 0.0 for j in idx])
    person = z.CLASSES.index("person")
    return np.nanmean(aps[:, 0]), np.nanmean(aps), aps[person, 0], aps[person].mean()


def run_all(infer, out_w, out_h):
    dets = []
    for i, img in images:
        lb, r, dx, dy = z.letterbox(img, out_w, out_h)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        b, c, s = z.post_process(infer(rgb), out_h, CONF, ALL)
        b = z.to_original(b, r, dx, dy, img.shape[1], img.shape[0])
        order = np.argsort(-s)[:MAX_DET]
        dets += [(i, int(c[k]), float(s[k]), b[k]) for k in order]
    return dets


MODELS = [
    ("YOLOv8n", "640x480", 640, 480, f"{TOOLS}/airockchip_yolov8/yolov8n_480.onnx", "yolov8n_480_video.rknn"),
    ("YOLOv8n", "640x640", 640, 640, f"{TOOLS}/airockchip_yolov8/yolov8n_640.onnx", "yolov8n_640.rknn"),
    ("YOLO11n", "640x480", 640, 480, f"{TOOLS}/airockchip_yolo11/yolo11n_480.onnx", "yolo11n_480_video.rknn"),
    ("YOLO11n", "640x640", 640, 640, f"{TOOLS}/airockchip_yolo11/yolo11n_640.onnx", "yolo11n_640.rknn"),
]


def report(name, size, title, m, t0):
    print(f"RESULT {name} {size} {title}: mAP50 {100 * m[0]:.1f}, mAP50-95 {100 * m[1]:.1f}, "
          f"person AP50 {100 * m[2]:.1f}, person AP50-95 {100 * m[3]:.1f}  [{time.time() - t0:.0f} с]", flush=True)


for name, size, w, h, onnx, out_name in MODELS:
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    t0 = time.time()
    report(name, size, "float (исходные веса)",
           evaluate(run_all(lambda rgb: sess.run(None, {inp: rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255}), w, h)), t0)

    board = ("int8, калибровка на видео", VIDEO_LIST) if h == 480 else ("int8, калибровка COCO", "/tmp/coco_abs.txt")
    calibs = [board] + ([("int8, калибровка COCO", "/tmp/coco_abs.txt")] if h == 480 else [])
    for k, (title, dataset) in enumerate(calibs):
        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
        rknn.load_onnx(model=onnx)
        rknn.build(do_quantization=True, dataset=dataset)
        if k == 0:
            rknn.export_rknn(f"{OUT}/{out_name}")
        rknn.init_runtime()
        t0 = time.time()
        report(name, size, title, evaluate(run_all(lambda rgb: rknn.inference(inputs=[rgb[None]], data_format="nhwc"), w, h)), t0)
        rknn.release()
