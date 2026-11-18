"""Run metadata written into every result file (hardware, library versions, threads, direct solver)."""

import importlib.util
import os
import platform
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def single_thread() -> None:
    """Pin BLAS/OpenMP to one thread; call before NumPy is imported (the timing experiments need this)."""
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def environment_info() -> dict:
    """Plain-data description of the machine and the software stack."""
    import numpy
    import scipy

    import fea_engine
    import rom_engine

    try:
        blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
        blas_info = {"name": blas.get("name"), "version": blas.get("version"), "configuration": blas.get("openblas configuration")}
    except Exception:                                 # older NumPy: no dict output
        blas_info = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "fea_engine": getattr(fea_engine, "__version__", None), "rom_engine": getattr(rom_engine, "__version__", None),
            "blas": blas_info, "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "sparse_direct_solver": "UMFPACK" if importlib.util.find_spec("scikits") else "SuperLU (scipy.sparse.linalg.splu)"}
