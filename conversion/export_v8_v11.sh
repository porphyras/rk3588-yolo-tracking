#!/usr/bin/env bash
# Экспорт YOLOv8n и YOLO11n в ONNX форками Rockchip (вход 640x640 и 640x480)
set -e
TOOLS="${TOOLS:-$(cd "$(dirname "$0")/.." && pwd)/tools}"
for pair in "airockchip_yolov8 yolov8n" "airockchip_yolo11 yolo11n"; do
  set -- $pair
  cd "$TOOLS/$1"
  for size in "640 640" "480 640"; do
    set -- $pair
    h=${size% *}; w=${size#* }
    PYTHONPATH=./ python -c "
from ultralytics import YOLO
YOLO('$2.pt').export(format='rknn', imgsz=[$h, $w])
" 2>&1 | grep -iE "export|error|onnx" | tail -3
    mv $2.onnx $2_$h.onnx
  done
  ls -la *.onnx
done
