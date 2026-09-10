"""Local resource measurements without an external logging service."""

import sys
import time

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None


class RunMeasurement:
    def __init__(self, torch_module=None, device=None):
        self.torch = torch_module
        self.device = device
        self.cuda = device is not None and device.type == "cuda"
        if self.cuda:
            self.torch.cuda.synchronize(device)
            self.torch.cuda.reset_peak_memory_stats(device)
        self.started = time.perf_counter()
        self.before = resource.getrusage(resource.RUSAGE_SELF) if resource else None

    def finish(self):
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
        after = resource.getrusage(resource.RUSAGE_SELF) if resource else None
        return {
            "scope": "validated CLI through input loading and model evaluation; before artifact serialization",
            "wall_seconds": time.perf_counter() - self.started,
            "cpu_user_seconds": after.ru_utime - self.before.ru_utime
            if after
            else None,
            "cpu_system_seconds": after.ru_stime - self.before.ru_stime
            if after
            else None,
            "process_peak_rss_bytes": int(
                after.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            )
            if after
            else None,
            "peak_rss_scope": "process lifetime, including imports; not a sum across runs",
            "cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(
                self.device
            )
            if self.cuda
            else None,
            "cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device)
            if self.cuda
            else None,
            "cuda_memory_scope": "PyTorch allocator only; excludes other processes and driver allocations",
        }
