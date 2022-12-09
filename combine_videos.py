import argparse
import concurrent.futures
import functools
import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path
from string import digits
from typing import Iterable, List, Optional

import cv2
import imageio
import imutils
import numpy as np
import tqdm
from functional import seq

MARGIN = 10
TARGET_WIDTH = 270
TARGET_HEIGHT = 480


def sort_paths(paths: Iterable[Path]) -> List[Path]:
    return (
        seq(paths)
        .sorted(
            key=lambda x: int(
                "".join(
                    [char for char in x.with_suffix("").name if char in digits]
                )
            )
        )
        .to_list()
    )


def store_single_image(
    index: int,
    slider_name: Path,
    path_set: List[Path],
    text_set: List[Optional[Path]],
    tmp_folder: Path,
    width: int,
    height: int,
):
    slider = imageio.imread(slider_name)
    if len(slider.shape) == 2:
        slider = cv2.cvtColor(slider, cv2.COLOR_GRAY2RGB)
    slider = imutils.resize(slider, width=width)

    imgs = []
    for img_name, text_name in zip(path_set, text_set):
        img = imageio.imread(img_name)
        img = cv2.resize(img, (width, height))
        img = cv2.copyMakeBorder(
            img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, (0, 0, 0)
        )
        if text_name is not None:
            text_img = imageio.imread(text_name)
            if len(text_img.shape) == 2:
                text_img = cv2.cvtColor(text_img, cv2.COLOR_GRAY2RGB)
            text_img = cv2.resize(text_img, (0, 0), fx=0.6, fy=0.6)

            text_height = text_img.shape[0]
            left_pad = (img.shape[1] - text_img.shape[1]) // 2
            right_pad = img.shape[1] - text_img.shape[1] - left_pad

            text_img = np.pad(
                text_img,
                [[0, 0], [left_pad, right_pad], [0, 0]],
                mode="constant",
                constant_values=255,
            )

            center = img.shape[1] // 2
            img = np.pad(
                img,
                [[text_height + MARGIN, 0], [0, 0], [0, 0]],
                mode="constant",
                constant_values=255,
            )
            img[:text_height] = text_img
        imgs.append(img)
    imgs_array = np.concatenate(imgs, axis=1)
    slider_height = slider.shape[0]
    slider_width = slider.shape[1]
    center = imgs_array.shape[1] // 2
    start_y = imgs_array.shape[0] + MARGIN
    start_x = center - slider.shape[1] // 2

    imgs_array = np.pad(
        imgs_array,
        [[0, slider_height + MARGIN], [0, 0], [0, 0]],
        mode="constant",
        constant_values=255,
    )
    imgs_array[start_y:, start_x : start_x + slider_width] = slider

    imageio.imwrite(tmp_folder / f"{index:04d}.png", imgs_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("sliders")
    parser.add_argument("out")
    parser.add_argument("--text", nargs="+")
    parser.add_argument("--name", default="rgb")
    parser.add_argument("--width", default=TARGET_WIDTH, type=int)
    parser.add_argument("--height", default=TARGET_HEIGHT, type=int)

    tmp_folder = Path("tmp")
    tmp_folder.mkdir(exist_ok=True)

    args = parser.parse_args()
    set_of_paths = []
    set_of_texts = []
    if args.text is None:
        for path in args.paths:
            path = Path(path)
            image_names = sort_paths(path.glob(f"{args.name}*.png"))
            set_of_paths.append(image_names)
            set_of_texts.append([None] * len(image_names))
    else:
        for path, text in zip(args.paths, args.text):
            path = Path(path)
            image_names = sort_paths(path.glob(f"{args.name}*.png"))
            set_of_paths.append(image_names)
            set_of_texts.append([text] * len(image_names))
    psave = functools.partial(
        store_single_image,
        tmp_folder=tmp_folder,
        height=args.height,
        width=args.width,
    )

    sliders = sort_paths(Path(args.sliders).glob("*.jpg"))
    set_of_paths = list(zip(*set_of_paths))
    set_of_texts = list(zip(*set_of_texts))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=mp.cpu_count() - 1
    ) as pool:
        list(
            tqdm.tqdm(
                pool.map(
                    psave,
                    np.arange(len(sliders)),
                    sliders,
                    set_of_paths,
                    set_of_texts,
                ),
                total=len(sliders),
            )
        )

    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{tmp_folder.as_posix()}/%04d.png",
            "-c:v",
            "libx264",
            "-vf",
            "fps=24",
            f"{args.out}",
        ]
    )
    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    main()
