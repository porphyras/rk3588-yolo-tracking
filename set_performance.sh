#!/usr/bin/env bash
# Pin CPU, NPU, memory controller and GPU to their maximum frequency (run on the board with sudo).
# The board is shared, so check the load (uptime, NPU usage, temperature) before benchmarking.
set -e
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done
for d in /sys/class/devfreq/*; do
  if grep -qw performance "$d/available_governors" 2>/dev/null; then echo performance > "$d/governor"; fi
done
grep . /sys/class/devfreq/*/cur_freq
