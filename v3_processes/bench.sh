#!/usr/bin/env bash
cd "$(dirname "$0")"

echo "== режим частот =="
echo "процессор: $(sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c | xargs)"
for d in /sys/class/devfreq/*; do
  case "$d" in
    *npu*|*dmc*) echo "$(basename "$d"): $(cat "$d/governor"), частота $(cat "$d/cur_freq")" ;;
  esac
done

run() {
  echo
  echo "== python $*"
  python "$@" 2>&1 | grep -E "средний FPS|на кадр|пик памяти"
}

run ../v2_threads/detect.py --threads 1
run ../v2_threads/detect.py
run detect.py --workers 1
run detect.py
run detect.py --save out3.mp4
