from __future__ import annotations

import functools
import os
import time


def profile_resources(func):
    """Measure execution time, RAM usage (RSS), and GPU VRAM usage.

    Args:
    ----
    func:
        The function to profile.

    """
    import psutil
    import torch

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        process = psutil.Process(os.getpid())

        # 1. Snapshot Start State
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_vram = torch.cuda.memory_allocated() / (1024**2)
        else:
            start_vram = 0.0

        start_ram = process.memory_info().rss / (1024**2)  # Convert to MB
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_vram = torch.cuda.memory_allocated() / (1024**2)
                peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                end_vram = 0.0
                peak_vram = 0.0

            end_ram = process.memory_info().rss / (1024**2)
            end_time = time.time()

            delta_ram = end_ram - start_ram
            delta_vram = end_vram - start_vram

            print(f"\n[PROFILE] Function: {func.__name__}")
            print(f"  ├── Time:      {end_time - start_time:.4f}s")
            print(
                f"  ├── RAM (RSS): {start_ram:.2f}MB -> {end_ram:.2f}MB (Delta: {delta_ram:+.2f}MB)"  # noqa: E501
            )
            if torch.cuda.is_available():
                print(
                    f"  └── VRAM:      {start_vram:.2f}MB -> {end_vram:.2f}MB (Delta: {delta_vram:+.2f}MB, Peak: {peak_vram:.2f}MB)"  # noqa: E501
                )
            else:
                print("  └── VRAM:      N/A (CPU only)")
            print("-" * 40)

        return result

    return wrapper
