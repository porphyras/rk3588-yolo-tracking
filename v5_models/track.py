import argparse
import multiprocessing as mp
import queue
import resource
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

NMS_THRESH = 0.45
MAX_IN_FLIGHT = 4
MIN_INSIDE = 0.6
MAX_STEP = 1.6
EDGE = 2
MIN_STEP_SIM = 0.5
STEP_MIN_H = 0.25
REF_ALPHA = 0.05
DUP_OVERLAP = 0.8
DUP_FRAMES = 10
DUP_REASON = "рамка совпала с другим объектом"
LOST_SCORE = {"nano": 0.3, "vit": 0.2, "mil": 0.5}
DEFAULT_CLASSES = "person,bicycle,car,motorbike,bus,truck"
FP16 = False
MATCH_IOU = 0.3
FIX_IOU = 0.6
NEW_OVERLAP = DUP_OVERLAP
REID_SIM = 0.6
REID_SIM_ONE = 0.4
MOVE_MARGIN = 0.1
DET_MISSES = 2
DET_MISS_FRAMES = 30
DET_GAP = 5
NOTE_FRAMES = 45
DET_REASON = "детектор не видит объект"

MODEL_FILES = {
    "5": {480: "yolov5n_480_video.rknn", 640: "yolov5n_640.rknn"},
    "8": {480: "yolov8n_480_video.rknn", 640: "yolov8n_640.rknn"},
    "11": {480: "yolo11n_480_video.rknn", 640: "yolo11n_640.rknn"},
}

ANCHORS = np.array([
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]],
], dtype=np.float32)

CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

TRACKERS = ("nano", "vit", "mil")


def letterbox(img, out_w, out_h):
    h, w = img.shape[:2]
    r = min(out_w / w, out_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if nw == out_w and nh == out_h:
        return resized, r, 0, 0
    dx, dy = (out_w - nw) // 2, (out_h - nh) // 2
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, r, dx, dy


def decode(output, anchors, out_h, conf):
    _, _, gh, gw = output.shape
    p = output.reshape(3, 85, gh, gw)
    a, gy, gx = np.nonzero(p[:, 4] >= conf)
    if len(a) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)

    cand = p[a, :, gy, gx]
    stride = out_h // gh
    xy = (cand[:, 0:2] * 2 - 0.5 + np.stack((gx, gy), axis=1)) * stride
    wh = (cand[:, 2:4] * 2) ** 2 * anchors[a]
    boxes = np.concatenate((xy - wh / 2, xy + wh / 2), axis=1)

    class_ids = cand[:, 5:].argmax(axis=1)
    scores = cand[:, 4] * cand[np.arange(len(cand)), 5 + class_ids]
    return boxes, class_ids, scores


def decode_dfl(box, cls, cls_sum, out_h, conf):
    _, _, gh, gw = cls.shape
    gy, gx = np.nonzero(cls_sum[0, 0] >= conf)
    if len(gy) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, int), np.empty(0, np.float32)

    probs = cls[0][:, gy, gx]
    class_ids = probs.argmax(axis=0)
    scores = probs[class_ids, np.arange(len(gy))]
    keep = scores >= conf
    gy, gx, class_ids, scores = gy[keep], gx[keep], class_ids[keep], scores[keep]

    dist = box[0][:, gy, gx].reshape(4, 16, -1)
    dist = np.exp(dist - dist.max(axis=1, keepdims=True))
    dist = (dist * np.arange(16, dtype=np.float32)[None, :, None]).sum(axis=1) / dist.sum(axis=1)
    stride = out_h // gh
    cx, cy = gx + 0.5, gy + 0.5
    boxes = np.stack((cx - dist[0], cy - dist[1], cx + dist[2], cy + dist[3]), axis=1) * stride
    return boxes.astype(np.float32), class_ids, scores


def post_process(outputs, out_h, conf, class_mask):
    if len(outputs) == 9:
        parts = [decode_dfl(*outputs[3 * i:3 * i + 3], out_h, conf) for i in range(3)]
    else:
        parts = [decode(out, anchors, out_h, conf) for out, anchors in zip(outputs, ANCHORS)]
    boxes = np.concatenate([p[0] for p in parts])
    class_ids = np.concatenate([p[1] for p in parts])
    scores = np.concatenate([p[2] for p in parts])

    keep = (scores >= conf) & class_mask[class_ids]
    boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
    if len(scores) == 0:
        return boxes, class_ids, scores

    shifted = boxes + class_ids[:, None] * 4096
    xywh = np.concatenate((shifted[:, :2], shifted[:, 2:] - shifted[:, :2]), axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, NMS_THRESH)
    idx = np.array(idx, dtype=int).flatten()
    return boxes[idx], class_ids[idx], scores[idx]


def to_original(boxes, r, dx, dy, w, h):
    b = boxes.copy()
    b[:, [0, 2]] = ((b[:, [0, 2]] - dx) / r).clip(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - dy) / r).clip(0, h)
    return b


class Detector:
    def __init__(self, model_path, out_w, out_h, conf, class_mask):
        from rknnlite.api import RKNNLite

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise SystemExit(f"не удалось загрузить модель {model_path}")
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise SystemExit("не удалось запустить модель на NPU")
        self.out_w, self.out_h = out_w, out_h
        self.conf = conf
        self.class_mask = class_mask

    def __call__(self, frame):
        img, r, dx, dy = letterbox(frame, self.out_w, self.out_h)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[np.expand_dims(img, 0)])
        if outputs is None:
            raise SystemExit(f"модель не приняла кадр {self.out_w}x{self.out_h}: проверь --imgsz")
        boxes, class_ids, scores = post_process(outputs, self.out_h, self.conf, self.class_mask)
        return to_original(boxes, r, dx, dy, frame.shape[1], frame.shape[0]), class_ids, scores

    def close(self):
        self.rknn.release()


def silence_opencv():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def make_tracker(kind):
    if kind == "nano":
        params = cv2.TrackerNano_Params()
        params.backbone = str(MODELS / "nanotrack_backbone.onnx")
        params.neckhead = str(MODELS / "nanotrack_head.onnx")
        if FP16:
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU_FP16
        return cv2.TrackerNano.create(params)
    if kind == "vit":
        params = cv2.TrackerVit_Params()
        params.net = str(MODELS / "vittrack.onnx")
        if FP16:
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU_FP16
        return cv2.TrackerVit.create(params)
    return cv2.TrackerMIL.create()


def color_hist(frame, box):
    x, y, bw, bh = box
    xa, xb = max(int(x + 0.2 * bw), 0), min(int(x + 0.8 * bw), frame.shape[1])
    ya, yb = max(int(y + 0.1 * bh), 0), min(int(y + 0.9 * bh), frame.shape[0])
    if xb - xa < 2 or yb - ya < 2:
        return None
    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def color_sim(a, b):
    if a is None or b is None:
        return 0.0
    return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


def at_edge(box, shape):
    w = shape[1]
    x, _, bw, _ = box
    return bw < 0.5 * w and (x <= EDGE or x + bw >= w - EDGE)


class ObjectTracker:
    def __init__(self, kind, frame, box, lost_score, lost_frames, min_sim):
        self.kind = kind
        self.tracker = make_tracker(kind)
        self.tracker.init(frame, tuple(int(v) for v in box))
        self.prev_area = float(box[2] * box[3])
        self.ref_hist = color_hist(frame, box)
        self.prev_hist = self.ref_hist
        self.hist = None
        self.sim0 = 1.0
        self.edge0 = at_edge(box, frame.shape)
        self.lost_score = lost_score
        self.lost_frames = lost_frames
        self.min_sim = min_sim
        self.low = 0
        self.lost = False
        self.reason = ""

    def _problem(self, frame, ok, box, score):
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        if not ok or bw < 2 or bh < 2:
            return "трекер не нашёл объект"
        inside_w = max(0.0, min(x + bw, w) - max(x, 0))
        inside_h = max(0.0, min(y + bh, h) - max(y, 0))
        if inside_w * inside_h < MIN_INSIDE * bw * bh:
            return "ушёл за край кадра"
        if not self.edge0 and at_edge(box, frame.shape):
            return "упёрся в край кадра"
        ratio = bw * bh / max(self.prev_area, 1.0)
        if ratio > MAX_STEP or ratio < 1.0 / MAX_STEP:
            return "размер рамки резко изменился"
        if score < self.lost_score:
            return "низкая уверенность трекера"
        self.hist = color_hist(frame, box)
        self.sim0 = color_sim(self.ref_hist, self.hist)
        if self.sim0 < self.min_sim:
            return "не похож на сопровождаемый объект"
        if bh >= STEP_MIN_H * h and color_sim(self.prev_hist, self.hist) < MIN_STEP_SIM:
            return "резко изменилось содержимое рамки"
        return ""

    def drop(self, reason):
        if not self.lost:
            self.lost = True
            self.reason = reason

    def update(self, frame):
        if self.lost:
            return False, None, 0.0
        ok, box = self.tracker.update(frame)
        score = self.tracker.getTrackingScore() if self.kind != "mil" else float(ok)
        problem = self._problem(frame, ok, box, score)
        if problem:
            self.low += 1
            if self.low >= self.lost_frames:
                self.lost = True
                self.reason = problem
        else:
            self.low = 0
            self.prev_area = float(box[2] * box[3])
            self.prev_hist = self.hist
            if self.ref_hist is not None and self.hist is not None and box[3] >= STEP_MIN_H * frame.shape[0]:
                mixed = (1 - REF_ALPHA) * self.ref_hist + REF_ALPHA * self.hist
                self.ref_hist = mixed / max(float(mixed.sum()), 1e-6)
        return self.low == 0, tuple(box), float(score)


def iou_xywh(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def overlap_xywh(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / (min(a[2] * a[3], b[2] * b[3]) + 1e-9)


class DuplicateFilter:
    def __init__(self):
        self.counts = {}

    def update(self, states):
        ids = sorted(i for i, st in states.items() if st[0])
        losers = set()
        for k, a in enumerate(ids):
            for b in ids[k + 1:]:
                if overlap_xywh(states[a][1], states[b][1]) > DUP_OVERLAP:
                    self.counts[(a, b)] = self.counts.get((a, b), 0) + 1
                    if self.counts[(a, b)] >= DUP_FRAMES:
                        losers.add(a if states[a][3] < states[b][3] else b)
                else:
                    self.counts.pop((a, b), None)
        return losers


def worker(idx, kind, shm_name, frame_shape, lost_score, lost_frames, min_sim, threads, fp16, tasks, results):
    global FP16
    FP16 = fp16
    cv2.setNumThreads(threads)
    silence_opencv()
    shm = shared_memory.SharedMemory(name=shm_name)
    h, w = frame_shape[:2]
    frame_bytes = h * w * 3
    trackers = {}
    try:
        while True:
            task = tasks.get()
            if task is None:
                break
            action, seq, slot, drops, items = task
            for obj_id, reason in drops:
                if obj_id in trackers:
                    trackers[obj_id].drop(reason)
            frame = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf, offset=slot * frame_bytes)
            out = []
            if action == "add":
                for obj_id, box in items:
                    t0 = time.perf_counter()
                    old = trackers.get(obj_id)
                    tr = ObjectTracker(kind, frame, box, lost_score, lost_frames, min_sim)
                    if old is not None and old.ref_hist is not None:
                        tr.ref_hist = old.ref_hist
                    trackers[obj_id] = tr
                    out.append((obj_id, True, tuple(box), 1.0, time.perf_counter() - t0, "", 1.0))
            else:
                for obj_id, tr in list(trackers.items()):
                    if tr.lost:
                        out.append((obj_id, False, None, 0.0, 0.0, tr.reason, 0.0))
                        del trackers[obj_id]
                        continue
                    t0 = time.perf_counter()
                    visible, box, score = tr.update(frame)
                    out.append((obj_id, visible, box, score, time.perf_counter() - t0, tr.reason, tr.sim0))
            del frame
            results.put((idx, seq, slot, out))
    finally:
        shm.close()


def draw(img, objects, states, fps, fps_avg, kind, note):
    alive = 0
    for obj_id, (visible, box, score, _) in states.items():
        if not visible:
            continue
        alive += 1
        x, y, bw, bh = (int(v) for v in box)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, f"{CLASSES[objects[obj_id][0]]} #{obj_id} {score:.2f}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f"FPS: {fps:.1f}  avg: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)
    cv2.putText(img, f"tracker: {kind}  objects: {alive}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if note:
        cv2.putText(img, note, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)


def draw_detections(img, boxes, class_ids, scores):
    for i, ((x1, y1, x2, y2), c, s) in enumerate(zip(boxes.astype(int), class_ids, scores), start=1):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(img, f"{CLASSES[c]} #{i} {s:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
    cv2.putText(img, "detector (YOLOv5n, NPU)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 0), 2)


def writer_loop(path, fps, size, frames):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    while True:
        frame = frames.get()
        if frame is None:
            break
        writer.write(frame)
    writer.release()


def reader_loop(cap, frames, stop):
    while not stop.is_set():
        ok, frame = cap.read()
        item = frame if ok else None
        while not stop.is_set():
            try:
                frames.put(item, timeout=0.2)
                break
            except queue.Full:
                continue
        if item is None:
            break


def copy_to_slot(shm, slot, frame):
    view = np.ndarray(frame.shape, dtype=np.uint8, buffer=shm.buf, offset=slot * frame.nbytes)
    view[:] = frame
    del view


def class_mask_from(arg):
    if arg.strip().lower() == "all":
        return np.ones(len(CLASSES), dtype=bool)
    names = [c.strip() for c in arg.split(",") if c.strip()]
    unknown = [c for c in names if c not in CLASSES]
    if unknown:
        raise SystemExit(f"неизвестные классы: {', '.join(unknown)}. Доступны: {', '.join(CLASSES)}")
    mask = np.zeros(len(CLASSES), dtype=bool)
    for c in names:
        mask[CLASSES.index(c)] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(HERE / "test.mp4"), help="видеофайл, номер камеры или /dev/videoN")
    ap.add_argument("--yolo", default="5", choices=sorted(MODEL_FILES), help="версия YOLO: 5, 8 или 11")
    ap.add_argument("--model", default=None, help="модель детектора, по умолчанию по --yolo и ориентации видео")
    ap.add_argument("--imgsz", default=None, help="вход детектора, ШИРИНАxВЫСОТА")
    ap.add_argument("--tracker", default="nano", choices=TRACKERS)
    ap.add_argument("--workers", type=int, default=3, help="сколько процессов с трекерами, 1-4")
    ap.add_argument("--threads", type=int, default=1, help="потоков OpenCV в каждом процессе")
    ap.add_argument("--conf", type=float, default=0.4, help="порог уверенности детектора")
    ap.add_argument("--classes", default=DEFAULT_CLASSES, help="какие классы искать, через запятую, или all")
    ap.add_argument("--max-objects", type=int, default=12, help="сколько объектов сопровождать одновременно")
    ap.add_argument("--min-size", type=int, default=20, help="минимальная высота рамки в пикселях")
    ap.add_argument("--max-detect-frames", type=int, default=300,
                    help="сколько кадров детектор ищет объекты в начале, прежде чем сдаться")
    ap.add_argument("--lost-score", type=float, default=None,
                    help="уверенность трекера, ниже которой объект считается потерянным")
    ap.add_argument("--lost-frames", type=int, default=10, help="сколько кадров подряд с проблемой до потери")
    ap.add_argument("--min-sim", type=float, default=0.2,
                    help="минимальная похожесть цвета на сопровождаемый объект")
    ap.add_argument("--redetect-every", type=int, default=30,
                    help="запускать детектор раз в столько кадров, 0 - только при потере объекта")
    ap.add_argument("--reid-frames", type=int, default=900,
                    help="сколько кадров помнить потерянный объект, чтобы вернуть ему прежний номер")
    ap.add_argument("--no-redetect", action="store_true", help="детектор только в начале, как в v4")
    ap.add_argument("--frames", type=int, default=100000, help="сколько кадров обработать")
    ap.add_argument("--save", default="", help="out.mp4")
    ap.add_argument("--fp16", action="store_true", help="считать сети трекера в fp16 (nano, vit)")
    args = ap.parse_args()

    silence_opencv()
    lost_score = LOST_SCORE[args.tracker] if args.lost_score is None else args.lost_score
    n_workers = max(1, min(4, args.workers))
    redetect = not args.no_redetect
    if args.tracker == "nano":
        needed = [MODELS / "nanotrack_backbone.onnx", MODELS / "nanotrack_head.onnx"]
    elif args.tracker == "vit":
        needed = [MODELS / "vittrack.onnx"]
    else:
        needed = []
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit("нет файлов модели трекера: " + ", ".join(missing))

    src = args.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"не удалось открыть {src}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 15
    portrait = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    imgsz = args.imgsz or ("640x640" if portrait else "640x480")
    out_w, out_h = (int(v) for v in imgsz.lower().split("x"))
    model_path = args.model
    if model_path is None:
        if out_h not in MODEL_FILES[args.yolo]:
            raise SystemExit(f"для входа {imgsz} нет готовой модели, укажи --model")
        model_path = str(MODELS / MODEL_FILES[args.yolo][out_h])
    print(f"модель: {Path(model_path).name}, вход {out_w}x{out_h}")

    writer_thread = None
    frames_out = None

    def emit(img):
        nonlocal writer_thread, frames_out
        if not args.save:
            return
        if writer_thread is None:
            h, w = img.shape[:2]
            frames_out = queue.Queue(maxsize=32)
            writer_thread = threading.Thread(target=writer_loop, args=(args.save, video_fps, (w, h), frames_out),
                                             daemon=True)
            writer_thread.start()
        frames_out.put(img)

    detector = Detector(model_path, out_w, out_h, args.conf, class_mask_from(args.classes))

    def detect(frame):
        boxes, class_ids, scores = detector(frame)
        tall = (boxes[:, 3] - boxes[:, 1]) >= args.min_size
        return boxes[tall], class_ids[tall], scores[tall]

    det_frame = None
    skipped = 0
    t_det = 0.0
    while skipped < args.max_detect_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        boxes, class_ids, scores = detect(frame)
        t_det = time.perf_counter() - t0
        if len(boxes):
            det_frame = frame
            break
        emit(frame.copy())
        skipped += 1
    if not redetect:
        detector.close()
    if det_frame is None:
        detector.close()
        raise SystemExit("детектор не нашёл объектов, сопровождать нечего")

    order = np.argsort(-scores)[:args.max_objects]
    boxes, class_ids, scores = boxes[order], class_ids[order], scores[order]
    print(f"детектор: кадр {skipped}, найдено объектов: {len(boxes)} за {1000 * t_det:.1f} мс (NPU)")
    print("  " + ", ".join(f"#{i} {CLASSES[c]} {s:.2f}" for i, (c, s) in enumerate(zip(class_ids, scores), start=1)))
    if redetect:
        print(f"повторный детектор: при потере объекта"
              + (f" и раз в {args.redetect_every} кадров" if args.redetect_every > 0 else ""))
    else:
        print("повторный детектор выключен: дальше только трекер")
    vis = det_frame.copy()
    draw_detections(vis, boxes, class_ids, scores)
    emit(vis)

    frame_shape = det_frame.shape
    frame_bytes = int(np.prod(frame_shape))
    slots = MAX_IN_FLIGHT + 2
    shm = shared_memory.SharedMemory(create=True, size=slots * frame_bytes)
    ctx = mp.get_context("spawn")
    tasks = [ctx.Queue() for _ in range(n_workers)]
    results = ctx.Queue()
    procs = [ctx.Process(target=worker, daemon=True,
                         args=(k, args.tracker, shm.name, frame_shape, lost_score,
                               args.lost_frames, args.min_sim, args.threads, args.fp16, tasks[k], results))
             for k in range(n_workers)]

    base = skipped + 1
    objects = {}
    active = set()
    owner = {}
    last_box = {}
    det_seen = {}
    misses = {}
    hists = {}
    gone = {}
    suppressed = set()
    drops_for = {k: [] for k in range(n_workers)}
    duplicates = DuplicateFilter()
    stats = {"runs": 0, "on_loss": 0, "t": 0.0, "new": 0, "back": 0, "fixed": 0, "dropped": 0}
    lost_log = []
    next_id = 1
    note = ""
    note_until = -1
    need_detect = False
    after_loss = False
    last_det = 0

    def mark_lost(obj_id, n_frame, reason):
        nonlocal need_detect, after_loss
        if obj_id not in active:
            return
        active.discard(obj_id)
        last_box.pop(obj_id, None)
        gone[obj_id] = n_frame
        lost_log.append((obj_id, base + n_frame, reason))
        if redetect:
            need_detect = True
            after_loss = True

    def drop(obj_id, n_frame, reason):
        suppressed.add(obj_id)
        drops_for[owner[obj_id]].append((obj_id, reason))
        mark_lost(obj_id, n_frame, reason)

    def apply_detections(frame, n_frame, boxes, class_ids, scores, slot):
        nonlocal next_id
        dets = [((b[0], b[1], b[2] - b[0], b[3] - b[1]), int(c), float(s))
                for b, c, s in zip(boxes, class_ids, scores)]
        pairs = []
        for obj_id in active:
            box = last_box.get(obj_id)
            if box is None:
                continue
            for j, (d, c, _) in enumerate(dets):
                if c == objects[obj_id][0]:
                    v = iou_xywh(box, d)
                    if v >= MATCH_IOU:
                        pairs.append((v, obj_id, j))
        matched, used = {}, set()
        for v, obj_id, j in sorted(pairs, reverse=True):
            if obj_id in matched or j in used:
                continue
            matched[obj_id] = (j, v)
            used.add(j)

        items = {k: [] for k in range(n_workers)}
        fixed = []
        for obj_id, (j, v) in matched.items():
            det_seen[obj_id] = n_frame
            misses[obj_id] = 0
            hist = color_hist(frame, dets[j][0])
            if hist is not None:
                hists[obj_id] = hist
            if v < FIX_IOU:
                items[owner[obj_id]].append((obj_id, dets[j][0]))
                last_box[obj_id] = dets[j][0]
                fixed.append(obj_id)
        unconfirmed = sorted(o for o in active - set(matched) if o in last_box and o in hists)
        free = [j for j in range(len(dets)) if j not in used
                and all(overlap_xywh(dets[j][0], last_box[o]) <= NEW_OVERLAP for o in active if o in last_box)]
        free_hist = {j: color_hist(frame, dets[j][0]) for j in free}
        cands = []
        for obj_id, n_lost in gone.items():
            if n_frame - n_lost > args.reid_frames or obj_id not in hists:
                continue
            for j in free:
                if dets[j][1] == objects[obj_id][0]:
                    sim = color_sim(hists[obj_id], free_hist[j])
                    if sim >= REID_SIM_ONE:
                        cands.append((sim, obj_id, j, False))
        for obj_id in unconfirmed:
            now = color_sim(hists[obj_id], color_hist(frame, last_box[obj_id]))
            for j in free:
                if dets[j][1] == objects[obj_id][0]:
                    sim = color_sim(hists[obj_id], free_hist[j])
                    if sim >= REID_SIM_ONE and sim >= now + MOVE_MARGIN:
                        cands.append((sim, obj_id, j, True))
        pool_obj, pool_det = {}, {}
        for _, obj_id, j, _ in cands:
            pool_obj.setdefault(dets[j][1], set()).add(obj_id)
            pool_det.setdefault(dets[j][1], set()).add(j)
        alone = {c: len(pool_obj[c]) == 1 and len(pool_det[c]) == 1
                 and not any(objects[o][0] == c for o in active - pool_obj[c]) for c in pool_obj}
        cands = [c for c in cands if c[0] >= REID_SIM or alone[dets[c[2]][1]]]
        back, taken = [], set()
        for sim, obj_id, j, alive in sorted(cands, reverse=True):
            if obj_id in back or j in taken:
                continue
            if not alive:
                if len(active) >= args.max_objects:
                    continue
                gone.pop(obj_id)
                suppressed.discard(obj_id)
                active.add(obj_id)
            back.append(obj_id)
            taken.add(j)
            items[owner[obj_id]].append((obj_id, dets[j][0]))
            last_box[obj_id] = dets[j][0]
            det_seen[obj_id] = n_frame
            misses[obj_id] = 0
            if free_hist[j] is not None:
                hists[obj_id] = free_hist[j]

        dropped = []
        for obj_id in sorted(active - set(matched) - set(back)):
            misses[obj_id] = misses.get(obj_id, 0) + 1
            if misses[obj_id] >= DET_MISSES and n_frame - det_seen[obj_id] >= DET_MISS_FRAMES:
                drop(obj_id, n_frame, DET_REASON)
                dropped.append(obj_id)

        new = []
        load = [0] * n_workers
        for obj_id in active:
            load[owner[obj_id]] += 1
        for j in sorted((j for j in free if j not in taken), key=lambda j: -dets[j][2]):
            if len(active) >= args.max_objects:
                break
            obj_id = next_id
            next_id += 1
            box, c, s = dets[j]
            k = load.index(min(load))
            load[k] += 1
            objects[obj_id] = (c, s)
            owner[obj_id] = k
            active.add(obj_id)
            items[k].append((obj_id, box))
            last_box[obj_id] = box
            det_seen[obj_id] = n_frame
            misses[obj_id] = 0
            if free_hist[j] is not None:
                hists[obj_id] = free_hist[j]
            new.append(obj_id)

        targets = [k for k in range(n_workers) if items[k]]
        if targets:
            copy_to_slot(shm, slot, frame)
            for k in targets:
                tasks[k].put(("add", -1, slot, drops_for[k], items[k]))
                drops_for[k] = []
        t_init = 0.0
        for _ in targets:
            try:
                _, _, _, out = results.get(timeout=180)
            except queue.Empty:
                raise SystemExit("процессы с трекерами не отвечают")
            t_init += sum(o[4] for o in out)
        stats["new"] += len(new)
        stats["back"] += len(back)
        stats["fixed"] += len(fixed)
        stats["dropped"] += len(dropped)
        return new, back, fixed, dropped, t_init

    stop = threading.Event()
    reader_thread = None
    try:
        for p in procs:
            p.start()

        new, _, _, _, t_init = apply_detections(det_frame, 0, boxes, class_ids, scores, 0)
        print(f"трекер: {args.tracker}{' fp16' if args.fp16 else ''}, процессов: {n_workers}, "
              f"инициализация {1000 * t_init / max(len(new), 1):.0f} мс на объект")

        frames_in = queue.Queue(maxsize=16)
        reader_thread = threading.Thread(target=reader_loop, args=(cap, frames_in, stop), daemon=True)
        reader_thread.start()

        free_slots = deque(range(slots))
        pending = {}
        done = {}
        waiting = {}
        recent = deque(maxlen=91)
        t_track = t_obj = 0.0
        n_obj_updates = 0
        n = 0
        submitted = 0
        in_flight = 0
        reading = True
        last_frame = det_frame
        t_begin = time.perf_counter()
        t_win = t_begin

        while True:
            if redetect and args.redetect_every > 0 and n - last_det >= args.redetect_every:
                need_detect = True
            detect_now = need_detect and n - last_det >= DET_GAP
            while (reading and not detect_now and in_flight < MAX_IN_FLIGHT and free_slots
                   and submitted < args.frames):
                frame = frames_in.get()
                if frame is None:
                    reading = False
                    break
                slot = free_slots.popleft()
                copy_to_slot(shm, slot, frame)
                waiting[submitted] = frame
                pending[submitted] = {"left": n_workers, "slot": slot, "out": [], "busy": []}
                for k, q in enumerate(tasks):
                    q.put(("track", submitted, slot, drops_for[k], []))
                    drops_for[k] = []
                submitted += 1
                in_flight += 1
            if submitted >= args.frames:
                reading = False

            if detect_now and in_flight == 0 and reading:
                t0 = time.perf_counter()
                boxes, class_ids, scores = detect(last_frame)
                t_d = time.perf_counter() - t0
                on_loss = after_loss
                new, back, fixed, dropped, _ = apply_detections(last_frame, n - 1, boxes, class_ids, scores,
                                                                free_slots[0])
                stats["runs"] += 1
                stats["on_loss"] += int(on_loss)
                stats["t"] += t_d
                need_detect = False
                after_loss = False
                last_det = n
                parts = ([f"new #{', #'.join(map(str, new))}"] if new else []) + \
                        ([f"back #{', #'.join(map(str, back))}"] if back else []) + \
                        ([f"removed #{', #'.join(map(str, dropped))}"] if dropped else [])
                if parts:
                    note = "detector: " + ", ".join(parts)
                    note_until = n + NOTE_FRAMES
                    print(f"кадр {base + n - 1}: детектор {1000 * t_d:.0f} мс, найдено {len(boxes)}; "
                          + "; ".join(p.replace("new", "новые").replace("back", "вернулись")
                                      .replace("removed", "сняты") for p in parts))
                continue
            if in_flight == 0:
                break

            try:
                _, seq, slot, out = results.get(timeout=60)
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                raise SystemExit("процесс с трекерами завершился аварийно" if dead else "нет ответа от трекеров")
            entry = pending[seq]
            entry["out"] += out
            entry["busy"].append(sum(o[4] for o in out))
            entry["left"] -= 1
            if entry["left"] == 0:
                free_slots.append(entry["slot"])
                in_flight -= 1
                done[seq] = pending.pop(seq)

            while n in done:
                entry = done.pop(n)
                frame = waiting.pop(n)
                states = {}
                for obj_id, visible, box, score, dt, reason, sim0 in entry["out"]:
                    if dt > 0:
                        t_obj += dt
                        n_obj_updates += 1
                    if box is None:
                        mark_lost(obj_id, n, reason)
                        continue
                    if obj_id in suppressed:
                        continue
                    states[obj_id] = (visible, box, score, sim0)
                    last_box[obj_id] = box
                for obj_id in duplicates.update(states):
                    drop(obj_id, n, DUP_REASON)
                    states.pop(obj_id)
                t_track += max(entry["busy"])
                last_frame = frame
                n += 1
                recent.append(time.perf_counter())

                if args.save:
                    vis = frame.copy()
                    fps_now = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                    draw(vis, objects, states, fps_now, n / (recent[-1] - t_begin), args.tracker,
                         note if n <= note_until else "")
                    emit(vis)

                if n % 30 == 0:
                    now = time.perf_counter()
                    alive = sum(1 for v in states.values() if v[0])
                    print(f"кадр {n}: {30 / (now - t_win):.1f} FPS, ведётся объектов: {alive}")
                    t_win = now

        if writer_thread is not None:
            frames_out.put(None)
            writer_thread.join()

        if n == 0:
            raise SystemExit("после кадра с детекцией видео закончилось")
        total = time.perf_counter() - t_begin
        print(f"\nкадров с трекером: {n}, средний FPS: {n / total:.1f}")
        print(f"трекер: {1000 * t_obj / max(n_obj_updates, 1):.1f} мс на объект, "
              f"{1000 * t_track / n:.1f} мс на кадр в самом загруженном процессе")
        if redetect:
            print(f"детектор запускался ещё {stats['runs']} раз (из них {stats['on_loss']} после потери), "
                  f"{1000 * stats['t'] / max(stats['runs'], 1):.1f} мс на запуск; новых объектов {stats['new']}, "
                  f"вернулись под своим номером {stats['back']}, поправлено рамок {stats['fixed']}, "
                  f"снято детектором {stats['dropped']}")
        print(f"сопровождались до конца: {len(active)} из {len(objects)}")
        for obj_id, f, reason in lost_log:
            print(f"  #{obj_id} {CLASSES[objects[obj_id][0]]} потерян на кадре {f}: {reason}")
    finally:
        stop.set()
        for q in tasks:
            q.put(None)
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        if reader_thread is not None:
            reader_thread.join(timeout=5)
        cap.release()
        shm.close()
        shm.unlink()
        if redetect:
            detector.close()

    main_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"пик памяти: главный процесс {main_mb:.0f} МБ, процесс с трекерами до {child_mb:.0f} МБ")


if __name__ == "__main__":
    main()
