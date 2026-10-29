import argparse
import itertools
import queue
import resource
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

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
FPS_WINDOW = 90

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

_local = threading.local()
_counter = itertools.count()
_workers = []
_workers_lock = threading.Lock()


def init_worker(model_path, cores):
    idx = next(_counter)
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        raise SystemExit(f"не удалось загрузить модель {model_path}")
    if rknn.init_runtime(core_mask=cores[idx % len(cores)]) != 0:
        raise SystemExit("не удалось запустить модель на NPU")
    _local.rknn = rknn
    with _workers_lock:
        _workers.append(rknn)


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


def decode(output, anchors, out_h):
    _, _, gh, gw = output.shape
    p = output.reshape(3, 85, gh, gw)
    a, gy, gx = np.nonzero(p[:, 4] >= OBJ_THRESH)
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


def post_process(outputs, out_h):
    parts = [decode(out, anchors, out_h) for out, anchors in zip(outputs, ANCHORS)]
    boxes = np.concatenate([p[0] for p in parts])
    class_ids = np.concatenate([p[1] for p in parts])
    scores = np.concatenate([p[2] for p in parts])

    keep = scores >= OBJ_THRESH
    boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
    if len(scores) == 0:
        return boxes, class_ids, scores

    shifted = boxes + class_ids[:, None] * 4096
    xywh = np.concatenate((shifted[:, :2], shifted[:, 2:] - shifted[:, :2]), axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), OBJ_THRESH, NMS_THRESH)
    idx = np.array(idx, dtype=int).flatten()
    return boxes[idx], class_ids[idx], scores[idx]


def to_original(boxes, r, dx, dy, w, h):
    b = boxes.copy()
    b[:, [0, 2]] = ((b[:, [0, 2]] - dx) / r).clip(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - dy) / r).clip(0, h)
    return b


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

    def _apply(self, t, box, score):
        old = t["box"]
        moved = np.array([(box[0] + box[2] - old[0] - old[2]) / 2,
                          (box[1] + box[3] - old[1] - old[3]) / 2], dtype=np.float32)
        if t["misses"] == 0:
            t["vel"] = 0.5 * t["vel"] + 0.5 * moved
            t["speed"] = 0.8 * t["speed"] + 0.2 * float(np.hypot(moved[0], moved[1]))
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

    def update(self, boxes, class_ids, scores):
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
            for flat in np.argsort(m, axis=None)[::-1]:
                if m.flat[flat] < IOU_MATCH:
                    break
                ti, di = divmod(int(flat), m.shape[1])
                if ti in taken_t or di in taken_d:
                    continue
                taken_t.add(ti)
                taken_d.add(di)
                self._apply(self.tracks[ti], boxes[di], scores[di])

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
                    if dist < REID_DIST * max(th, dh):
                        pairs.append((dist, ti, di))
            for _, ti, di in sorted(pairs):
                if ti in taken_t or di in taken_d:
                    continue
                taken_t.add(ti)
                taken_d.add(di)
                self._apply(self.tracks[ti], boxes[di], scores[di])

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
    return cap, None


def process(frame, out_w, out_h):
    t0 = time.perf_counter()
    img, r, dx, dy = letterbox(frame, out_w, out_h)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t1 = time.perf_counter()
    outputs = _local.rknn.inference(inputs=[np.expand_dims(img, 0)])
    if outputs is None:
        raise RuntimeError(f"модель не приняла кадр {out_w}x{out_h}: проверь --imgsz, "
                           f"он должен совпадать с входом модели")
    t2 = time.perf_counter()
    boxes, class_ids, scores = post_process(outputs, out_h)
    boxes = to_original(boxes, r, dx, dy, frame.shape[1], frame.shape[0])
    t3 = time.perf_counter()
    return frame, boxes, class_ids, scores, (t1 - t0, t2 - t1, t3 - t2), [o.shape for o in outputs]


def main():
    global OBJ_THRESH

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "models" / "yolov5n_480.rknn"))
    ap.add_argument("--source", default=str(HERE / "test.mp4"),
                    help="картинка, видеофайл, номер камеры или /dev/videoN")
    ap.add_argument("--frames", type=int, default=1000, help="сколько кадров обработать")
    ap.add_argument("--save", default="", help="result.jpg или out.mp4")
    ap.add_argument("--imgsz", default="640x480", help="вход модели, ШИРИНАxВЫСОТА")
    ap.add_argument("--threads", type=int, default=3, help="сколько ядер NPU занять, 1-3")
    ap.add_argument("--conf", type=float, default=OBJ_THRESH, help="порог уверенности")
    args = ap.parse_args()
    OBJ_THRESH = args.conf

    out_w, out_h = (int(v) for v in args.imgsz.lower().split("x"))
    all_cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    cores = all_cores[:max(1, min(3, args.threads))]

    cap, still = open_source(args.source)
    pool = ThreadPoolExecutor(max_workers=len(cores), initializer=init_worker,
                              initargs=(args.model, cores))
    barrier = threading.Barrier(len(cores))
    for fut in [pool.submit(barrier.wait) for _ in range(len(cores))]:
        fut.result()
    print(f"ядер NPU: {len(cores)}, вход модели: {out_w}x{out_h}")

    tracker = Tracker()
    pending = deque()
    recent = deque(maxlen=FPS_WINDOW + 1)
    writer_thread = None
    frames_q = None
    t_pre = t_inf = t_post = t_track = 0.0
    n = 0
    submitted = 0
    reading = True
    t_begin = t_win = time.perf_counter()

    while True:
        while reading and len(pending) < len(cores) + 2 and submitted < args.frames:
            if cap is None:
                frame = still
            else:
                ok, frame = cap.read()
                if not ok:
                    reading = False
                    break
            pending.append(pool.submit(process, frame, out_w, out_h))
            submitted += 1

        if not pending:
            break

        try:
            frame, boxes, class_ids, scores, times, shapes = pending.popleft().result()
        except RuntimeError as err:
            raise SystemExit(str(err))

        if n == 0:
            print("выходы модели:", shapes)
            gh, gw = shapes[0][2], shapes[0][3]
            if gh * 8 != out_h or gw * 8 != out_w:
                raise SystemExit(f"модель рассчитана на вход {gw * 8}x{gh * 8}, "
                                 f"запусти с --imgsz {gw * 8}x{gh * 8}")

        t0 = time.perf_counter()
        tracks = tracker.update(boxes, class_ids, scores)
        t_track += time.perf_counter() - t0

        t_pre += times[0]
        t_inf += times[1]
        t_post += times[2]
        n += 1
        recent.append(time.perf_counter())

        if n == 1:
            found = ", ".join(f"{CLASSES[c]} {s:.2f}" for c, s in zip(class_ids, scores))
            print("найдено на первом кадре:", found or "ничего")

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
    print(f"считали параллельно на {len(cores)} ядрах NPU")
    print("уникальных объектов:", ", ".join(f"{k}: {v}" for k, v in
                                            sorted(tracker.counts.items(), key=lambda kv: -kv[1])) or "нет")
    print(f"выдано номеров трекером: {tracker.next_id - 1}")
    print(f"пик памяти процесса: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} МБ")

    pool.shutdown(wait=True)
    if cap is not None:
        cap.release()
    for rknn in _workers:
        rknn.release()


if __name__ == "__main__":
    main()
