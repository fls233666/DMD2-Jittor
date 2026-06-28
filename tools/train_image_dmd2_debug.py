"""Backward-compatible entry point for the image DMD2 trainer.

Use tools/train_image_dmd2.py for new runs.
"""

from train_image_dmd2 import main


if __name__ == "__main__":
    raise SystemExit(main())
