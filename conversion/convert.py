import argparse
from pathlib import Path

import cv2
from rknn.api import RKNN

HERE = Path(__file__).resolve().parent


def extract_frames(video, out_dir, step):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"не удалось открыть {video}")
    paths = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            path = out_dir / f"kadr_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)
        i += 1
    cap.release()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(HERE / "models" / "yolov5n_480.onnx"))
    ap.add_argument("--out", default=str(HERE / "models" / "yolov5n_480_video.rknn"))
    ap.add_argument("--video", default=str(HERE / "test.mp4"))
    ap.add_argument("--step", type=int, default=16, help="брать каждый N-й кадр для калибровки")
    ap.add_argument("--platform", default="rk3588")
    args = ap.parse_args()

    calib_dir = HERE / "kalibrovka"
    frames = extract_frames(args.video, calib_dir, args.step)
    spisok = calib_dir / "spisok.txt"
    spisok.write_text("\n".join(str(p) for p in frames) + "\n")
    print(f"кадров для калибровки: {len(frames)}")

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.platform)
    if rknn.load_onnx(model=args.onnx) != 0:
        raise SystemExit("не удалось загрузить ONNX")
    if rknn.build(do_quantization=True, dataset=str(spisok)) != 0:
        raise SystemExit("не удалось собрать модель")
    if rknn.export_rknn(args.out) != 0:
        raise SystemExit("не удалось сохранить RKNN")
    rknn.release()
    print("готово:", args.out)


if __name__ == "__main__":
    main()
