"""
Prints the information for tyxonq installation and environment.
"""

import platform
import sys
import numpy


def about() -> None:
    """
    Prints the information for tyxonq installation and environment.
    """
    print(f"OS info: {platform.platform(aliased=True)}")
    print(
        f"Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version: {numpy.__version__}")

    try:
        import scipy

        print(f"Scipy version: {scipy.__version__}")
    except ModuleNotFoundError:
        print(f"Scipy is not installed")

    try:
        import pandas

        print(f"Pandas version: {pandas.__version__}")
    except ModuleNotFoundError:
        print(f"Pandas is not installed")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch GPU support: {torch.cuda.is_available()}")
        print(
            f"PyTorch GPUs: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )
        if torch.version.cuda is not None:
            print(f"Pytorch cuda version: {torch.version.cuda}")
    except ModuleNotFoundError:
        print(f"PyTorch is not installed")

    try:
        import cupy

        print(f"Cupy version: {cupy.__version__}")
    except ModuleNotFoundError:
        print(f"Cupy is not installed")

    try:
        import qiskit

        print(f"Qiskit version: {qiskit.__version__}")
    except ModuleNotFoundError:
        print(f"Qiskit is not installed")


    from tyxonq import __version__

    print(f"tyxonq version {__version__}")


if __name__ == "__main__":
    about()
