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
  python "$@" 2>&1 | grep -E "средний FPS|на кадр|мс на объект|сопровождались|пик памяти"
}

run ../v2_threads/detect.py
run ../v3_processes/detect.py
run track.py
run track.py --tracker vit
run track.py --tracker mil
run track.py --source video2.mp4
run track.py --source video3.mp4
run track.py --source video3.mp4 --save out4.mp4
