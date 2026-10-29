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

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IOU_MATCH = 0.25
MIN_HITS = 3
COUNT_HITS = 10
MAX_MISSES = 20
MAX_MISSES_STATIC = 100
STATIC_SPEED = 1.5
REID_DIST = 0.6
SIM_MIN = 0.35
SIM_REID = 0.5
HIST_ALPHA = 0.2
FPS_WINDOW = 90

DEFAULT_CLASSES = "person,bicycle,car,motorbike,bus,truck"

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


def color_hist(frame, box):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    xa, xb = max(int(x1 + 0.2 * bw), 0), min(int(x2 - 0.2 * bw), frame.shape[1])
    ya, yb = max(int(y1 + 0.1 * bh), 0), min(int(y2 - 0.1 * bh), frame.shape[0])
    if xb - xa < 2 or yb - ya < 2:
        return None
    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def color_sim(a, b):
    if a is None or b is None:
        return 1.0
    return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


def worker(idx, model_path, out_w, out_h, shm_name, frame_shape, conf, class_mask, tasks, results):
    from rknnlite.api import RKNNLite

    cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        results.put(("error", idx, f"не удалось загрузить модель {model_path}"))
        return
    if rknn.init_runtime(core_mask=cores[idx % len(cores)]) != 0:
        results.put(("error", idx, "не удалось запустить модель на NPU"))
        return

    shm = shared_memory.SharedMemory(name=shm_name)
    h, w = frame_shape[:2]
    frame_bytes = h * w * 3
    results.put(("ready", idx, None))
    try:
        while True:
            task = tasks.get()
            if task is None:
                break
            seq, slot = task
            frame = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf, offset=slot * frame_bytes)

            t0 = time.perf_counter()
            img, r, dx, dy = letterbox(frame, out_w, out_h)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t1 = time.perf_counter()
            outputs = rknn.inference(inputs=[np.expand_dims(img, 0)])
            if outputs is None:
                del frame
                results.put(("error", idx, f"модель не приняла кадр {out_w}x{out_h}: проверь --imgsz, "
                                           f"он должен совпадать с входом модели"))
                break
            t2 = time.perf_counter()
            boxes, class_ids, scores = post_process(outputs, out_h, conf, class_mask)
            boxes = to_original(boxes, r, dx, dy, w, h)
            hists = [color_hist(frame, b) for b in boxes]
            t3 = time.perf_counter()
            del frame

            results.put(("done", seq, (slot, boxes, class_ids, scores, hists,
                                       (t1 - t0, t2 - t1, t3 - t2), [o.shape for o in outputs])))
    finally:
        shm.close()
        rknn.release()


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.counts = {}
        self.color_rejects = 0

    def _apply(self, t, box, score, hist):
        old = t["box"]
        moved = np.array([(box[0] + box[2] - old[0] - old[2]) / 2,
                          (box[1] + box[3] - old[1] - old[3]) / 2], dtype=np.float32)
        if t["misses"] == 0:
            t["vel"] = 0.5 * t["vel"] + 0.5 * moved
            t["speed"] = 0.8 * t["speed"] + 0.2 * float(np.hypot(moved[0], moved[1]))
        if hist is not None:
            if t["hist"] is None:
                t["hist"] = hist
            else:
                mixed = (1 - HIST_ALPHA) * t["hist"] + HIST_ALPHA * hist
                t["hist"] = mixed / max(float(mixed.sum()), 1e-6)
        t["box"] = box.copy()
        t["score"] = float(score)
        t["hits"] += 1
        t["misses"] = 0
        if t["hits"] == MIN_HITS:
            t["id"] = self.next_id
            self.next_id += 1
        if t["hits"] == COUNT_HITS:
            name = CLASSES[t["cls"]]
            self.counts[name] = self.counts.get(name, 0) + 1

    @staticmethod
    def _patience(t):
        if t["hits"] < MIN_HITS:
            return 2
        if t["speed"] < STATIC_SPEED:
            return MAX_MISSES_STATIC
        return MAX_MISSES

    def update(self, boxes, class_ids, scores, hists):
        for t in self.tracks:
            t["box"] = t["box"] + np.tile(t["vel"], 2)

        boxes = boxes.astype(np.float32)
        class_ids = class_ids.astype(int)
        taken_t, taken_d = set(), set()

        if self.tracks and len(boxes):
            track_boxes = np.stack([t["box"] for t in self.tracks])
            track_cls = np.array([t["cls"] for t in self.tracks])
            same_cls = track_cls[:, None] == class_ids[None, :]

            m = iou_matrix(track_boxes, boxes)
            m[~same_cls] = 0.0
            pairs = []
            for ti, di in zip(*np.nonzero(m >= IOU_MATCH)):
                sim = color_sim(self.tracks[ti]["hist"], hists[di])
                if sim < (SIM_MIN if self.tracks[ti]["misses"] == 0 else SIM_REID):
                    self.color_rejects += 1
                    continue
                pairs.append((m[ti, di] * (0.5 + 0.5 * sim), int(ti), int(di)))
            for _, ti, di in sorted(pairs, reverse=True):
                if ti in taken_t or di in taken_d:
                    continue
                taken_t.add(ti)
                taken_d.add(di)
                self._apply(self.tracks[ti], boxes[di], scores[di], hists[di])

            lost = [ti for ti, t in enumerate(self.tracks) if ti not in taken_t and t["hits"] >= MIN_HITS]
            free = [di for di in range(len(boxes)) if di not in taken_d]
            pairs = []
            for ti in lost:
                tb = track_boxes[ti]
                th = tb[3] - tb[1]
                tc = (tb[:2] + tb[2:]) / 2
                for di in free:
                    if not same_cls[ti, di]:
                        continue
                    db = boxes[di]
                    dh = db[3] - db[1]
                    if not 0.5 < dh / max(th, 1.0) < 2.0:
                        continue
                    dc = (db[:2] + db[2:]) / 2 - tc
                    dist = float(np.hypot(dc[0], dc[1]))
                    if dist >= REID_DIST * max(th, dh):
                        continue
                    sim = color_sim(self.tracks[ti]["hist"], hists[di])
                    if sim < SIM_REID:
                        self.color_rejects += 1
                        continue
                    pairs.append((dist / max(sim, 1e-3), ti, di))
            for _, ti, di in sorted(pairs):
                if ti in taken_t or di in taken_d:
                    continue
                taken_t.add(ti)
                taken_d.add(di)
                self._apply(self.tracks[ti], boxes[di], scores[di], hists[di])

        for ti, t in enumerate(self.tracks):
            if ti not in taken_t:
                t["misses"] += 1
                t["vel"] = t["vel"] * 0.5

        for di in range(len(boxes)):
            if di in taken_d:
                continue
            self.tracks.append({
                "box": boxes[di].copy(),
                "cls": int(class_ids[di]),
                "score": float(scores[di]),
                "hist": hists[di],
                "vel": np.zeros(2, dtype=np.float32),
                "speed": 0.0,
                "hits": 1,
                "misses": 0,
                "id": 0,
            })

        self.tracks = [t for t in self.tracks if t["misses"] <= self._patience(t)]
        return [t for t in self.tracks if t["hits"] >= MIN_HITS and t["misses"] <= 2]


def draw(img, tracks, fps, fps_avg, npu_ms, counts):
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{CLASSES[t['cls']]} #{t['id']} {t['score']:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f"FPS: {fps:.1f}  avg: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)
    cv2.putText(img, f"NPU: {npu_ms:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top:
        line = "  ".join(f"{name}: {num}" for name, num in top)
        cv2.putText(img, line, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def writer_loop(path, fps, size, frames):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    while True:
        frame = frames.get()
        if frame is None:
            break
        writer.write(frame)
    writer.release()


def reader_loop(cap, first, frames, stop):
    frame = first
    while frame is not None and not stop.is_set():
        while not stop.is_set():
            try:
                frames.put(frame, timeout=0.2)
                break
            except queue.Full:
                continue
        ok, frame = cap.read()
        if not ok:
            frame = None
    while not stop.is_set():
        try:
            frames.put(None, timeout=0.2)
            break
        except queue.Full:
            continue


def copy_to_slot(shm, slot, frame):
    view = np.ndarray(frame.shape, dtype=np.uint8, buffer=shm.buf, offset=slot * frame.nbytes)
    view[:] = frame
    del view


def open_source(src):
    if src.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"не удалось прочитать {src}")
        return None, img

    is_camera = src.isdigit() or src.startswith("/dev/video")
    if is_camera:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"не удалось открыть {src}")
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"не удалось прочитать первый кадр из {src}")
    return cap, first


def class_mask_from(arg):
    if arg.strip().lower() == "all":
        return np.ones(len(CLASSES), dtype=bool), list(CLASSES)
    names = [c.strip() for c in arg.split(",") if c.strip()]
    unknown = [c for c in names if c not in CLASSES]
    if unknown:
        raise SystemExit(f"неизвестные классы: {', '.join(unknown)}. Доступны: {', '.join(CLASSES)}")
    mask = np.zeros(len(CLASSES), dtype=bool)
    for c in names:
        mask[CLASSES.index(c)] = True
    return mask, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="5", choices=sorted(MODEL_FILES), help="версия YOLO: 5, 8 или 11")
    ap.add_argument("--model", default=None, help="файл модели, по умолчанию выбирается по --yolo и --imgsz")
    ap.add_argument("--source", default=str(HERE / "test.mp4"),
                    help="картинка, видеофайл, номер камеры или /dev/videoN")
    ap.add_argument("--frames", type=int, default=1000, help="сколько кадров обработать")
    ap.add_argument("--save", default="", help="result.jpg или out.mp4")
    ap.add_argument("--imgsz", default=None, help="вход модели, ШИРИНАxВЫСОТА, по умолчанию по ориентации видео")
    ap.add_argument("--workers", type=int, default=3, help="сколько процессов и ядер NPU занять, 1-3")
    ap.add_argument("--conf", type=float, default=OBJ_THRESH, help="порог уверенности")
    ap.add_argument("--classes", default=DEFAULT_CLASSES,
                    help="какие классы искать, через запятую, или all")
    args = ap.parse_args()

    n_workers = max(1, min(3, args.workers))
    class_mask, class_names = class_mask_from(args.classes)

    src_name = args.source
    if src_name.lower().endswith((".jpg", ".jpeg", ".png")):
        cap, first = None, None
        still = open_source(src_name)[1]
        frame_shape = still.shape
    else:
        cap, first = open_source(src_name)
        still = None
        frame_shape = first.shape

    imgsz = args.imgsz or ("640x640" if frame_shape[0] > frame_shape[1] else "640x480")
    out_w, out_h = (int(v) for v in imgsz.lower().split("x"))
    model_path = args.model
    if model_path is None:
        if out_h not in MODEL_FILES[args.yolo]:
            raise SystemExit(f"для входа {imgsz} нет готовой модели, укажи --model")
        model_path = str(HERE / "models" / MODEL_FILES[args.yolo][out_h])
    print(f"модель: {Path(model_path).name}, вход {out_w}x{out_h}")

    frame_bytes = int(np.prod(frame_shape))
    max_in_flight = n_workers * 2
    slots = max_in_flight + 2
    shm = shared_memory.SharedMemory(create=True, size=slots * frame_bytes)

    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()
    procs = [ctx.Process(target=worker, daemon=True,
                         args=(i, model_path, out_w, out_h, shm.name, frame_shape,
                               args.conf, class_mask, tasks, results))
             for i in range(n_workers)]

    stop = threading.Event()
    reader_thread = None
    writer_thread = None
    frames_q = None
    try:
        for p in procs:
            p.start()
        ready = 0
        while ready < n_workers:
            try:
                kind, _, payload = results.get(timeout=120)
            except queue.Empty:
                raise SystemExit("рабочие процессы не запустились")
            if kind == "error":
                raise SystemExit(payload)
            ready += 1
        print(f"процессов и ядер NPU: {n_workers}, вход модели: {out_w}x{out_h}")
        print("ищем классы:", ", ".join(class_names) if len(class_names) < len(CLASSES) else "все")

        frames_in = None
        if cap is not None:
            frames_in = queue.Queue(maxsize=16)
            reader_thread = threading.Thread(target=reader_loop, args=(cap, first, frames_in, stop),
                                             daemon=True)
            reader_thread.start()

        tracker = Tracker()
        free_slots = deque(range(slots))
        waiting = {}
        done = {}
        recent = deque(maxlen=FPS_WINDOW + 1)
        t_pre = t_inf = t_post = t_track = 0.0
        n = 0
        submitted = 0
        in_flight = 0
        reading = True
        t_begin = time.perf_counter()
        t_win = t_begin

        while True:
            while reading and in_flight < max_in_flight and free_slots and submitted < args.frames:
                if still is not None:
                    frame = still
                else:
                    frame = frames_in.get()
                    if frame is None:
                        reading = False
                        break
                slot = free_slots.popleft()
                copy_to_slot(shm, slot, frame)
                waiting[submitted] = frame
                tasks.put((submitted, slot))
                submitted += 1
                in_flight += 1
            if submitted >= args.frames:
                reading = False

            if in_flight == 0:
                break

            try:
                kind, key, payload = results.get(timeout=60)
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                raise SystemExit("рабочий процесс завершился аварийно" if dead else "нет ответа от NPU")
            if kind == "error":
                raise SystemExit(payload)

            slot = payload[0]
            free_slots.append(slot)
            in_flight -= 1
            done[key] = payload

            while n in done:
                _, boxes, class_ids, scores, hists, times, shapes = done.pop(n)
                frame = waiting.pop(n)
                if n == 0:
                    print("выходы модели:", shapes)
                    gh, gw = shapes[0][2], shapes[0][3]
                    if gh * 8 != out_h or gw * 8 != out_w:
                        raise SystemExit(f"модель рассчитана на вход {gw * 8}x{gh * 8}, "
                                         f"запусти с --imgsz {gw * 8}x{gh * 8}")
                    found = ", ".join(f"{CLASSES[c]} {s:.2f}" for c, s in zip(class_ids, scores))
                    print("найдено на первом кадре:", found or "ничего")

                t0 = time.perf_counter()
                tracks = tracker.update(boxes, class_ids, scores, hists)
                t_track += time.perf_counter() - t0

                t_pre += times[0]
                t_inf += times[1]
                t_post += times[2]
                n += 1
                recent.append(time.perf_counter())

                if args.save:
                    vis = frame.copy()
                    fps_now = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                    fps_avg = n / (recent[-1] - t_begin)
                    draw(vis, tracks, fps_now, fps_avg, 1000 * times[1], tracker.counts)
                    if cap is None:
                        if n == 1:
                            cv2.imwrite(args.save, vis)
                    else:
                        if writer_thread is None:
                            h, w = vis.shape[:2]
                            frames_q = queue.Queue(maxsize=32)
                            writer_thread = threading.Thread(
                                target=writer_loop,
                                args=(args.save, cap.get(cv2.CAP_PROP_FPS) or 15, (w, h), frames_q),
                                daemon=True)
                            writer_thread.start()
                        frames_q.put(vis)

                if n % 30 == 0:
                    now = time.perf_counter()
                    print(f"кадр {n}: {30 / (now - t_win):.1f} FPS")
                    t_win = now

        if n == 0:
            raise SystemExit("не получено ни одного кадра")

        if writer_thread is not None:
            frames_q.put(None)
            writer_thread.join()

        total = time.perf_counter() - t_begin
        print(f"\nкадров: {n}, средний FPS: {n / total:.1f}")
        print(f"на кадр: подготовка {1000 * t_pre / n:.1f} мс, "
              f"NPU {1000 * t_inf / n:.1f} мс, постобработка {1000 * t_post / n:.1f} мс, "
              f"трекер {1000 * t_track / n:.1f} мс")
        print(f"считали параллельно в {n_workers} процессах на {n_workers} ядрах NPU")
        print("уникальных объектов:", ", ".join(f"{k}: {v}" for k, v in
                                                sorted(tracker.counts.items(), key=lambda kv: -kv[1])) or "нет")
        print(f"выдано номеров трекером: {tracker.next_id - 1}, "
              f"отклонено по цвету: {tracker.color_rejects}")
    finally:
        stop.set()
        for _ in procs:
            tasks.put(None)
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        if reader_thread is not None:
            reader_thread.join(timeout=5)
        if cap is not None:
            cap.release()
        shm.close()
        shm.unlink()

    main_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"пик памяти: главный процесс {main_mb:.0f} МБ, рабочий процесс до {child_mb:.0f} МБ")


if __name__ == "__main__":
    main()
