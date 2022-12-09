import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")

    args = parser.parse_args()

    imgs = []
    max_height = 0
    img_paths = list(Path(args.path).glob("*.jpg"))
    for img_path in img_paths:
        img = imageio.imread(img_path.as_posix())
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        imgs.append(img)
        max_height = max(max_height, img.shape[0])

    for img, img_path in zip(imgs, img_paths):
        img = np.pad(
            img,
            [[0, max_height - img.shape[0]], [0, 0], [0, 0]],
            mode="constant",
            constant_values=255,
        )
        imageio.imwrite(img_path, img)


if __name__ == "__main__":
    main()
