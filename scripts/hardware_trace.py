#!/usr/bin/env python3
"""Hardware resource tracer for the video-to-scene pipeline.

Logs GPU VRAM, system RAM, CPU, and disk usage at configurable intervals.
Produces a JSON trace and summary suitable for hardware sizing recommendations.
"""

import json
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path


def sample_gpus():
    """Sample GPU stats via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_used_mb": float(parts[2]),
                    "vram_total_mb": float(parts[3]),
                    "gpu_util_pct": float(parts[4]),
                    "temp_c": float(parts[5]),
                    "power_w": float(parts[6]) if parts[6] != "[N/A]" else 0,
                })
        return gpus
    except Exception:
        return []


def sample_system():
    """Sample system RAM and CPU."""
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:", "Buffers:", "Cached:"):
                    mem[parts[0].rstrip(":")] = int(parts[1])  # kB

        with open("/proc/loadavg") as f:
            load = f.read().split()

        return {
            "ram_total_mb": mem.get("MemTotal", 0) / 1024,
            "ram_available_mb": mem.get("MemAvailable", 0) / 1024,
            "ram_used_mb": (mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)) / 1024,
            "load_1m": float(load[0]),
            "load_5m": float(load[1]),
            "load_15m": float(load[2]),
        }
    except Exception:
        return {}


def sample_disk(path=os.environ.get("TRACE_DISK_PATH", str(Path(__file__).resolve().parent.parent))):
    """Sample disk usage."""
    try:
        stat = os.statvfs(path)
        total = stat.f_frsize * stat.f_blocks
        free = stat.f_frsize * stat.f_bfree
        return {
            "disk_total_gb": total / (1024**3),
            "disk_free_gb": free / (1024**3),
            "disk_used_gb": (total - free) / (1024**3),
        }
    except Exception:
        return {}


def sample_all(stage="unknown"):
    """Take a complete system sample."""
    return {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "gpus": sample_gpus(),
        "system": sample_system(),
        "disk": sample_disk(),
    }


class HardwareTracer:
    """Traces hardware usage across pipeline stages."""

    def __init__(self, output_path="hardware_trace.json"):
        self.output_path = Path(output_path)
        self.samples = []
        self.stage_peaks = {}  # stage -> peak metrics

    def sample(self, stage="unknown"):
        """Take a sample and record it."""
        s = sample_all(stage)
        self.samples.append(s)

        # Track peaks per stage
        if stage not in self.stage_peaks:
            self.stage_peaks[stage] = {
                "peak_vram_mb": 0,
                "peak_ram_mb": 0,
                "peak_gpu_util": 0,
                "peak_power_w": 0,
                "sample_count": 0,
            }

        peaks = self.stage_peaks[stage]
        peaks["sample_count"] += 1

        for gpu in s.get("gpus", []):
            peaks["peak_vram_mb"] = max(peaks["peak_vram_mb"], gpu["vram_used_mb"])
            peaks["peak_gpu_util"] = max(peaks["peak_gpu_util"], gpu["gpu_util_pct"])
            peaks["peak_power_w"] = max(peaks["peak_power_w"], gpu.get("power_w", 0))

        sys_info = s.get("system", {})
        peaks["peak_ram_mb"] = max(peaks["peak_ram_mb"], sys_info.get("ram_used_mb", 0))

        return s

    def save(self):
        """Save trace to JSON."""
        output = {
            "trace_start": self.samples[0]["timestamp"] if self.samples else None,
            "trace_end": self.samples[-1]["timestamp"] if self.samples else None,
            "total_samples": len(self.samples),
            "stage_peaks": self.stage_peaks,
            "samples": self.samples,
        }
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

    def summary(self):
        """Generate hardware sizing recommendations."""
        if not self.stage_peaks:
            return "No data collected."

        lines = ["# Hardware Requirements Trace", ""]
        lines.append("## Peak Usage by Stage\n")
        lines.append("| Stage | Peak VRAM (MB) | Peak RAM (MB) | Peak GPU% | Peak Power (W) | Samples |")
        lines.append("|-------|---------------|--------------|-----------|----------------|---------|")

        overall_vram = 0
        overall_ram = 0

        for stage, peaks in sorted(self.stage_peaks.items()):
            lines.append(
                f"| {stage} | {peaks['peak_vram_mb']:.0f} | {peaks['peak_ram_mb']:.0f} | "
                f"{peaks['peak_gpu_util']:.0f}% | {peaks['peak_power_w']:.0f}W | {peaks['sample_count']} |"
            )
            overall_vram = max(overall_vram, peaks["peak_vram_mb"])
            overall_ram = max(overall_ram, peaks["peak_ram_mb"])

        lines.append(f"\n## Minimum Hardware Recommendation\n")
        lines.append(f"- **GPU VRAM**: {max(8192, overall_vram * 1.2):.0f} MB ({max(8, int(overall_vram * 1.2 / 1024))} GB minimum)")
        lines.append(f"- **System RAM**: {max(16384, overall_ram * 1.2):.0f} MB ({max(16, int(overall_ram * 1.2 / 1024))} GB minimum)")
        lines.append(f"- **GPU**: NVIDIA with CUDA compute capability >= 7.5")
        lines.append(f"- **Disk**: 50GB+ free for models and intermediate data")
        lines.append(f"- **CPU**: 8+ cores recommended (COLMAP is CPU-intensive for matching)")

        return "\n".join(lines)


# Singleton for use across the pipeline
_tracer = None


def get_tracer(output_path=None):
    global _tracer
    if _tracer is None:
        path = output_path or str(Path(os.environ.get(
            "TEST_DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "test-data")
        )) / "gallery_output" / "hardware_trace.json")
        _tracer = HardwareTracer(path)
    return _tracer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hardware resource monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Sample interval in seconds")
    parser.add_argument("--stage", default="monitoring", help="Pipeline stage label")
    parser.add_argument("--output", default="hardware_trace.json", help="Output JSON path")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0=forever)")
    args = parser.parse_args()

    tracer = HardwareTracer(args.output)
    start = time.time()
    print(f"Monitoring hardware (stage={args.stage}, interval={args.interval}s)...")

    try:
        while True:
            s = tracer.sample(args.stage)
            gpu_info = ", ".join(f"GPU{g['index']}:{g['vram_used_mb']:.0f}/{g['vram_total_mb']:.0f}MB" for g in s["gpus"])
            ram = s["system"].get("ram_used_mb", 0)
            print(f"  [{s['timestamp']}] {gpu_info} | RAM:{ram:.0f}MB | Load:{s['system'].get('load_1m', 0):.1f}")

            if args.duration > 0 and (time.time() - start) >= args.duration:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    tracer.save()
    print(f"\nTrace saved to {args.output}")
    print(tracer.summary())
