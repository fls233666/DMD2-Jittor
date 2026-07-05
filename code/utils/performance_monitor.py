"""Performance monitor helpers for local DMD2 benchmarks."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Dict, Iterable, List, Optional


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "N/A", "[N/A]", "Not Supported", "[Not Supported]"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _first_visible_gpu():
    explicit = os.environ.get("DMD2_PERF_GPU_INDEX")
    if explicit:
        return explicit.strip()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in ("-1", "none", "None"):
        return visible.split(",", 1)[0].strip()
    return "0"


class NvidiaSmiMonitor:
    """Sample GPU utilization, power, and device memory with nvidia-smi."""

    QUERY_FIELDS = (
        "index",
        "utilization.gpu",
        "power.draw",
        "power.limit",
        "memory.used",
        "memory.total",
    )

    def __init__(
        self,
        enabled: bool = False,
        interval: float = 0.5,
        gpu_index: Optional[str] = None,
        timeout: float = 1.0,
    ):
        self.requested = bool(enabled)
        self.binary = shutil.which("nvidia-smi")
        self.enabled = bool(enabled) and self.binary is not None
        self.interval = max(float(interval), 0.0)
        self.gpu_index = str(gpu_index).strip() if gpu_index else _first_visible_gpu()
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._samples: List[Dict[str, object]] = []
        self._cursor = 0
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._run_peak_memory_used_mb = None

        if self.requested and self.binary is None:
            self._error = "nvidia-smi not found"

    def start(self):
        if not self.enabled:
            return self
        if self.interval <= 0:
            return self
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval * 2.0, 1.0))
            self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self._record_sample()
            self._stop.wait(self.interval)

    def _record_sample(self):
        sample = self._query_once()
        if not sample:
            return
        memory_used = sample.get("gpu_memory_used_mb")
        if isinstance(memory_used, (int, float)):
            if self._run_peak_memory_used_mb is None:
                self._run_peak_memory_used_mb = float(memory_used)
            else:
                self._run_peak_memory_used_mb = max(self._run_peak_memory_used_mb, float(memory_used))
        with self._lock:
            self._samples.append(sample)

    def _query_once(self):
        if not self.binary:
            return None

        cmd = [
            self.binary,
            "--query-gpu=" + ",".join(self.QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ]
        if self.gpu_index:
            cmd.extend(["-i", self.gpu_index])

        try:
            output = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
            )
        except Exception as exc:
            self._error = str(exc)
            return None

        line = output.strip().splitlines()[0] if output.strip() else ""
        if not line:
            return None
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(self.QUERY_FIELDS):
            self._error = "unexpected nvidia-smi output: " + line
            return None

        return {
            "gpu_index": parts[0],
            "gpu_utilization_percent": _parse_float(parts[1]),
            "gpu_power_draw_w": _parse_float(parts[2]),
            "gpu_power_limit_w": _parse_float(parts[3]),
            "gpu_memory_used_mb": _parse_float(parts[4]),
            "gpu_memory_total_mb": _parse_float(parts[5]),
            "gpu_monitor_time": time.time(),
        }

    def collect_since_last(self):
        if not self.requested:
            return {}
        if self.enabled and self.interval <= 0:
            self._record_sample()

        with self._lock:
            samples = self._samples[self._cursor :]
            self._cursor = len(self._samples)

        record = {
            "gpu_monitor_enabled": int(self.enabled),
            "gpu_monitor_sample_count": len(samples),
        }
        if self._error:
            record["gpu_monitor_error"] = self._error
        if not samples:
            return record

        latest = samples[-1]
        record["gpu_index"] = latest.get("gpu_index")
        record["gpu_memory_total_mb"] = latest.get("gpu_memory_total_mb")
        for name in (
            "gpu_utilization_percent",
            "gpu_power_draw_w",
            "gpu_memory_used_mb",
        ):
            values = [
                float(sample[name])
                for sample in samples
                if isinstance(sample.get(name), (int, float))
            ]
            average = _mean(values)
            if average is not None:
                record[name] = average
                record[name.replace("_percent", "_max_percent").replace("_w", "_max_w").replace("_mb", "_peak_mb")] = max(values)
        if isinstance(latest.get("gpu_power_limit_w"), (int, float)):
            record["gpu_power_limit_w"] = latest["gpu_power_limit_w"]
        if self._run_peak_memory_used_mb is not None:
            record["gpu_memory_used_peak_mb_run"] = self._run_peak_memory_used_mb
        return record
