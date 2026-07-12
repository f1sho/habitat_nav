from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union


MetricValue = Union[str, int, float, None]


class ModelMetrics:
    """
    Model-level metrics shared by PyTorch, ONNX Runtime, and TensorRT.

    GPU memory is measured from the current process through nvidia-smi rather
    than torch.cuda.max_memory_allocated(). This is necessary because the
    PyTorch allocator does not track memory allocated by ONNX Runtime or
    TensorRT.

    For a fair model-only estimate, capture:
        gpu_memory_before_mb: after Habitat is initialized, before model load
        gpu_memory_after_mb: after model load and warm-up

    gpu_memory_mb is then reported as:
        max(gpu_memory_after_mb - gpu_memory_before_mb, 0)
    """

    def __init__(
        self,
        model_path: str,
        model: Optional[Any] = None,
        gpu_memory_before_mb: Optional[float] = None,
        gpu_memory_after_mb: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.model = model
        self.gpu_memory_before_mb = gpu_memory_before_mb
        self.gpu_memory_after_mb = gpu_memory_after_mb

    @staticmethod
    def get_process_gpu_memory_mb() -> Optional[float]:
        """
        Return GPU memory used by the current process across all NVIDIA GPUs.

        The value is obtained from nvidia-smi's active compute-process query.
        Returns None when nvidia-smi is unavailable, the query fails, or the
        current process does not appear in the compute-process table.
        """
        current_pid = os.getpid()

        command = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (
            FileNotFoundError,
            subprocess.SubprocessError,
            OSError,
        ):
            return None

        if completed.returncode != 0:
            return None

        total_memory_mb = 0.0
        process_found = False

        for line in completed.stdout.splitlines():
            fields = [
                field.strip()
                for field in line.split(",")
            ]

            if len(fields) < 2:
                continue

            try:
                process_pid = int(fields[0])
                used_memory_mb = float(fields[1])
            except ValueError:
                continue

            if process_pid == current_pid:
                total_memory_mb += used_memory_mb
                process_found = True

        if not process_found:
            # nvidia-smi query succeeded, but this process has not created
            # a CUDA compute context yet.
            return 0.0

        return total_memory_mb

    @property
    def model_size_mb(self) -> Optional[float]:
        path = Path(self.model_path)

        if not path.exists() or not path.is_file():
            return None

        size_bytes = path.stat().st_size
        return round(
            size_bytes / (1024.0 * 1024.0),
            2,
        )

    @property
    def parameter_count(self) -> Optional[float]:
        """
        Return parameter count in millions when a PyTorch model is available.

        ONNX and TensorRT backends normally pass model=None, so this returns
        None for those formats rather than an incorrect zero.
        """
        if self.model is None:
            return None

        parameters_method = getattr(
            self.model,
            "parameters",
            None,
        )

        if not callable(parameters_method):
            return None

        try:
            parameter_total = sum(
                parameter.numel()
                for parameter in parameters_method()
            )
        except (AttributeError, TypeError):
            return None

        return round(
            parameter_total / 1_000_000.0,
            2,
        )

    @property
    def process_gpu_memory_mb(self) -> Optional[float]:
        """
        Total GPU memory used by this process after model warm-up.

        This includes any remaining Habitat-Sim allocations in the process and
        is retained as an audit value. Use gpu_memory_mb for the model delta.
        """
        if self.gpu_memory_after_mb is not None:
            return round(
                self.gpu_memory_after_mb,
                2,
            )

        current_memory = self.get_process_gpu_memory_mb()

        if current_memory is None:
            return None

        return round(
            current_memory,
            2,
        )

    @property
    def gpu_memory_mb(self) -> Optional[float]:
        """
        Approximate model/runtime GPU-memory increment after warm-up.
        """
        if (
            self.gpu_memory_before_mb is None
            or self.gpu_memory_after_mb is None
        ):
            return None

        memory_delta_mb = (
            self.gpu_memory_after_mb
            - self.gpu_memory_before_mb
        )

        return round(
            max(memory_delta_mb, 0.0),
            2,
        )

    def summary(self) -> Dict[str, MetricValue]:
        return {
            "model_size_mb": self.model_size_mb,
            "parameter_count_m": self.parameter_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "process_gpu_memory_mb": self.process_gpu_memory_mb,
        }
