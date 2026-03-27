from pathlib import Path
import platform
import sys

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "env_check.txt"


def write_log(lines):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    lines = []
    lines.append("=== CSC4005 ENVIRONMENT CHECK ===")
    lines.append(f"Python version: {sys.version}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"System: {platform.system()} {platform.release()}")

    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import sklearn
        import yaml
        import torch
        import torchvision
        import torchaudio

        lines.append("Import status: SUCCESS")
        lines.append(f"NumPy version: {np.__version__}")
        lines.append(f"Pandas version: {pd.__version__}")
        lines.append(f"Matplotlib version: {matplotlib.__version__}")
        lines.append(f"Scikit-learn version: {sklearn.__version__}")
        lines.append(f"PyYAML version: {yaml.__version__}")
        lines.append(f"Torch version: {torch.__version__}")
        lines.append(f"Torchvision version: {torchvision.__version__}")
        lines.append(f"Torchaudio version: {torchaudio.__version__}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        lines.append(f"Detected device: {device}")

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        linear = torch.nn.Linear(2, 2)
        y = linear(x)

        lines.append(f"Tensor test input shape: {tuple(x.shape)}")
        lines.append(f"Linear output shape: {tuple(y.shape)}")
        lines.append("Basic tensor + linear layer test: SUCCESS")

    except Exception as e:
        lines.append("Import status: FAILED")
        lines.append(f"Error: {repr(e)}")

    write_log(lines)

    for line in lines:
        print(line)

    print(f"\nSaved log to: {LOG_PATH}")


if __name__ == "__main__":
    main()
