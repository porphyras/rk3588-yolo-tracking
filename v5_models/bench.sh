#!/usr/bin/env bash
cd "$(dirname "$0")"
SUFFIX="${1:-}"
VIDEO_A="${VIDEO_A:-video2.mp4}"   # несколько человек
VIDEO_B="${VIDEO_B:-person7.mp4}"  # один человек, 60 с

echo "== режим частот =="
echo "процессор: $(sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c | xargs)"
for d in /sys/class/devfreq/*; do
  case "$d" in
    *npu*|*dmc*) echo "$(basename "$d"): $(cat "$d/governor"), частота $(cat "$d/cur_freq")" ;;
  esac
done

for f in "$VIDEO_A" "$VIDEO_B"; do
  [ -f "$f" ] || { echo "нет файла $f"; exit 1; }
done

for src in "$VIDEO_A" "$VIDEO_B"; do
  name=$(basename "$src" .mp4)
  for y in 5 8 11; do
    sleep 5
    echo
    echo "== ВИДЕО $name YOLO $y"
    echo "нагрузка до запуска: $(cut -d' ' -f1-3 /proc/loadavg)"
    python track.py --yolo $y --source "$src" --save "out5_${name}_yolo$y$SUFFIX.mp4" 2>&1 \
      | grep -E "модель:|детектор: кадр|средний FPS|мс на объект|запускался|сопровождались|пик памяти"
  done
done
