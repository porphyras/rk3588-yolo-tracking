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


def post_process(outputs, out_h, conf, class_mask):
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


def worker(idx, kind, shm_name, frame_shape, objects, lost_score, lost_frames, min_sim, threads, fp16, tasks, results):
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
            action, seq, slot, drops = task
            for obj_id in drops:
                trackers[obj_id].drop(DUP_REASON)
            frame = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf, offset=slot * frame_bytes)
            out = []
            if action == "init":
                for obj_id, box in objects:
                    t0 = time.perf_counter()
                    trackers[obj_id] = ObjectTracker(kind, frame, box, lost_score, lost_frames, min_sim)
                    out.append((obj_id, True, tuple(box), 1.0, time.perf_counter() - t0, "", 1.0))
            else:
                for obj_id, tr in trackers.items():
                    if tr.lost:
                        out.append((obj_id, False, None, 0.0, 0.0, tr.reason, 0.0))
                        continue
                    t0 = time.perf_counter()
                    visible, box, score = tr.update(frame)
                    out.append((obj_id, visible, box, score, time.perf_counter() - t0, tr.reason, tr.sim0))
            del frame
            results.put((idx, seq, slot, out))
    finally:
        shm.close()


def draw(img, objects, states, fps, fps_avg, kind):
    alive = 0
    for obj_id, (cls, _) in objects.items():
        visible, box, score = states.get(obj_id, (False, None, 0.0, 0.0))[:3]
        if not visible:
            continue
        alive += 1
        x, y, bw, bh = (int(v) for v in box)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, f"{CLASSES[cls]} #{obj_id} {score:.2f}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f"FPS: {fps:.1f}  avg: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)
    cv2.putText(img, f"tracker: {kind}  objects: {alive}/{len(objects)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


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
    ap.add_argument("--model", default=None, help="модель детектора, по умолчанию выбирается по ориентации видео")
    ap.add_argument("--imgsz", default=None, help="вход детектора, ШИРИНАxВЫСОТА")
    ap.add_argument("--tracker", default="nano", choices=TRACKERS)
    ap.add_argument("--workers", type=int, default=3, help="сколько процессов с трекерами, 1-4")
    ap.add_argument("--threads", type=int, default=1, help="потоков OpenCV в каждом процессе")
    ap.add_argument("--conf", type=float, default=0.4, help="порог уверенности детектора")
    ap.add_argument("--classes", default=DEFAULT_CLASSES, help="какие классы искать, через запятую, или all")
    ap.add_argument("--max-objects", type=int, default=12, help="сколько объектов взять под сопровождение")
    ap.add_argument("--min-size", type=int, default=20, help="минимальная высота рамки в пикселях")
    ap.add_argument("--max-detect-frames", type=int, default=300,
                    help="сколько кадров детектор ищет объекты, прежде чем сдаться")
    ap.add_argument("--lost-score", type=float, default=None,
                    help="уверенность трекера, ниже которой объект считается потерянным")
    ap.add_argument("--lost-frames", type=int, default=10, help="сколько кадров подряд с проблемой до потери")
    ap.add_argument("--min-sim", type=float, default=0.2,
                    help="минимальная похожесть цвета на сопровождаемый объект")
    ap.add_argument("--frames", type=int, default=100000, help="сколько кадров обработать")
    ap.add_argument("--save", default="", help="out.mp4")
    ap.add_argument("--fp16", action="store_true", help="считать сети трекера в fp16 (nano, vit)")
    args = ap.parse_args()

    silence_opencv()
    lost_score = LOST_SCORE[args.tracker] if args.lost_score is None else args.lost_score
    n_workers = max(1, min(4, args.workers))
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
    model_path = args.model or str(MODELS / ("yolov5n_640.rknn" if portrait else "yolov5n_480_video.rknn"))
    imgsz = args.imgsz or ("640x640" if portrait else "640x480")
    out_w, out_h = (int(v) for v in imgsz.lower().split("x"))

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
    det_frame = None
    skipped = 0
    t_det = 0.0
    while skipped < args.max_detect_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        boxes, class_ids, scores = detector(frame)
        t_det = time.perf_counter() - t0
        tall = (boxes[:, 3] - boxes[:, 1]) >= args.min_size
        boxes, class_ids, scores = boxes[tall], class_ids[tall], scores[tall]
        if len(boxes):
            det_frame = frame
            break
        emit(frame.copy())
        skipped += 1
    detector.close()
    if det_frame is None:
        raise SystemExit("детектор не нашёл объектов, сопровождать нечего")

    order = np.argsort(-scores)[:args.max_objects]
    boxes, class_ids, scores = boxes[order], class_ids[order], scores[order]
    objects = {i + 1: (int(c), s) for i, (c, s) in enumerate(zip(class_ids, scores))}
    print(f"детектор: кадр {skipped}, найдено объектов: {len(boxes)} за {1000 * t_det:.1f} мс (NPU, один раз)")
    print("  " + ", ".join(f"#{i} {CLASSES[c]} {s:.2f}" for i, (c, s) in objects.items()))
    vis = det_frame.copy()
    draw_detections(vis, boxes, class_ids, scores)
    emit(vis)

    xywh = [(i + 1, (b[0], b[1], b[2] - b[0], b[3] - b[1])) for i, b in enumerate(boxes)]
    by_area = sorted(xywh, key=lambda item: -item[1][2] * item[1][3])
    assigned = [by_area[k::n_workers] for k in range(n_workers)]
    assigned = [a for a in assigned if a]
    n_workers = len(assigned)

    frame_shape = det_frame.shape
    frame_bytes = int(np.prod(frame_shape))
    slots = MAX_IN_FLIGHT + 2
    shm = shared_memory.SharedMemory(create=True, size=slots * frame_bytes)
    ctx = mp.get_context("spawn")
    tasks = [ctx.Queue() for _ in range(n_workers)]
    results = ctx.Queue()
    procs = [ctx.Process(target=worker, daemon=True,
                         args=(k, args.tracker, shm.name, frame_shape, assigned[k], lost_score,
                               args.lost_frames, args.min_sim, args.threads, args.fp16, tasks[k], results))
             for k in range(n_workers)]

    stop = threading.Event()
    reader_thread = None
    try:
        for p in procs:
            p.start()

        copy_to_slot(shm, 0, det_frame)
        for q in tasks:
            q.put(("init", -1, 0, []))
        t_init = 0.0
        for _ in range(n_workers):
            try:
                _, _, _, out = results.get(timeout=180)
            except queue.Empty:
                raise SystemExit("процессы с трекерами не запустились")
            t_init += sum(o[4] for o in out)
        print(f"трекер: {args.tracker}{' fp16' if args.fp16 else ''}, процессов: {n_workers}, объектов на процесс: "
              f"{', '.join(str(len(a)) for a in assigned)}, инициализация {1000 * t_init / len(objects):.0f} мс на объект")

        frames_in = queue.Queue(maxsize=16)
        reader_thread = threading.Thread(target=reader_loop, args=(cap, frames_in, stop), daemon=True)
        reader_thread.start()

        free_slots = deque(range(slots))
        pending = {}
        done = {}
        waiting = {}
        recent = deque(maxlen=91)
        lost_at = {}
        owner = {obj_id: k for k, part in enumerate(assigned) for obj_id, _ in part}
        drops_for = {k: [] for k in range(n_workers)}
        suppressed = set()
        duplicates = DuplicateFilter()
        t_track = t_obj = 0.0
        n_obj_updates = 0
        n = 0
        submitted = 0
        in_flight = 0
        reading = True
        t_begin = time.perf_counter()
        t_win = t_begin

        while True:
            while reading and in_flight < MAX_IN_FLIGHT and free_slots and submitted < args.frames:
                frame = frames_in.get()
                if frame is None:
                    reading = False
                    break
                slot = free_slots.popleft()
                copy_to_slot(shm, slot, frame)
                waiting[submitted] = frame
                pending[submitted] = {"left": n_workers, "slot": slot, "out": [], "busy": []}
                for k, q in enumerate(tasks):
                    q.put(("track", submitted, slot, drops_for[k]))
                    drops_for[k] = []
                submitted += 1
                in_flight += 1
            if submitted >= args.frames:
                reading = False
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
                    states[obj_id] = (visible and obj_id not in suppressed, box, score, sim0)
                    if dt > 0:
                        t_obj += dt
                        n_obj_updates += 1
                    if box is None and obj_id not in lost_at:
                        lost_at[obj_id] = (skipped + 1 + n, reason)
                for obj_id in duplicates.update(states):
                    suppressed.add(obj_id)
                    drops_for[owner[obj_id]].append(obj_id)
                    lost_at.setdefault(obj_id, (skipped + 1 + n, DUP_REASON))
                    states[obj_id] = (False, None, 0.0, 0.0)
                t_track += max(entry["busy"])
                n += 1
                recent.append(time.perf_counter())

                if args.save:
                    vis = frame.copy()
                    fps_now = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                    draw(vis, objects, states, fps_now, n / (recent[-1] - t_begin), args.tracker)
                    emit(vis)

                if n % 30 == 0:
                    now = time.perf_counter()
                    alive = sum(1 for v in states.values() if v[0])
                    print(f"кадр {n}: {30 / (now - t_win):.1f} FPS, ведётся объектов: {alive}/{len(objects)}")
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
        kept = len(objects) - len(lost_at)
        print(f"сопровождались до конца: {kept} из {len(objects)}")
        if lost_at:
            for i, (f, reason) in sorted(lost_at.items()):
                print(f"  #{i} {CLASSES[objects[i][0]]} потерян на кадре {f}: {reason}")
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

    main_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"пик памяти: главный процесс {main_mb:.0f} МБ, процесс с трекерами до {child_mb:.0f} МБ")


if __name__ == "__main__":
    main()
