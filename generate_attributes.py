import argparse
import concurrent.futures
import functools
import multiprocessing as mp
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import tqdm

preamble = r"""\documentclass[tikz,crop]{standalone}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,shadows,shadows.blur,matrix,backgrounds}
\tikzset{white background/.style={show background rectangle,tight background,
background rectangle/.style={fill=white}}}

\def\slidersize{3.1cm}
\def\spacingslider{-0.3em}

\def\IosSevenSliderFixed#1{
    \tikz[baseline=-0.1cm]{
        \coordinate (start) at (0,0);
        \coordinate (end) at (\slidersize,0);
        \coordinate (mark) at ($(start)!#1!(end)$);
        \useasboundingbox (start|- 0,-.25) rectangle (end|- 0, .25);
        \draw[line width=0.9mm, line cap=round, blue!50!cyan] 
             (start) -- (mark) edge[lightgray] (end);
        \node[fill=white, draw=lightgray, very thin,
            blur shadow={shadow xshift=0pt, shadow opacity=20, shadow yshift=-0.9mm,
                         shadow blur steps=6, shadow blur radius=0.3mm},
            circle, minimum size=0.25cm, inner sep=0pt] at (mark) {};
    }
}

\def\IosSevenSlider#1{
    \tikz[baseline=-0.1cm]{
        \coordinate (start) at (0,0);
        \coordinate (end) at (\slidersize,0);
        \coordinate (mark) at ($(start)!#1!(end)$);
        \useasboundingbox (start|- 0,-.25) rectangle (end|- 0, .25);
        \draw[line width=0.9mm, line cap=round, blue!50!cyan] 
             (start) -- (mark) edge[lightgray] (end);
        \node[fill=white, draw=lightgray, very thin,
            blur shadow={shadow xshift=0pt, shadow opacity=20, shadow yshift=-0.9mm,
                         shadow blur steps=6, shadow blur radius=0.3mm},
            circle, minimum size=0.25cm,yshift=3.4pt, inner sep=0pt] at (mark) {};
    }
}
"""

body = """
\\begin{{document}}
\\setlength{{\\fboxsep}}{{0pt}}
\\begin{{tikzpicture}}[white background]
{}
\\end{{tikzpicture}}
\\end{{document}}
"""

first_slider_template = """
\\node (slider0) {{\IosSevenSliderFixed{{{}}}}};
\\node[left=0em of slider0.west, align=center] {{$-1$}};
\\node[right=0em of slider0.east, align=center] {{$1$}};
\\node[left=1.5em of slider0.west, align=right] {{{}}};

"""

slider_template = """
\\node[
    below=\spacingslider of slider{0}
] (slider{1}) {{\IosSevenSlider{{{2}}}}};
\\node[left=0em of slider{1}.west, align=center] {{$-1$}};
\\node[right=0em of slider{1}.east, align=center] {{$1$}};
\\node[left=1.5em of slider{1}.west, align=right] {{{3}}};
"""

MIN_VALUE = 0.0
MAX_VALUE = 1.0
NUM_FRAMES = 1200
NUM_RANDOM_POINTS = 6
BEZIER_VALUES = np.array([[0.0, 0.0], [0.6, 0.0], [0.4, 1.0], [1.0, 1.0]])

np.random.seed(0xCAFFE)


def interpolate_points(
    point1: np.ndarray, point2: np.ndarray, steps: int
) -> np.ndarray:

    t = np.linspace(0, 1, num=steps)[:, np.newaxis]
    alphas = (
        (1 - t) ** 3 * BEZIER_VALUES[0]
        + 3 * (1 - t) ** 2 * t * BEZIER_VALUES[1]
        + 3 * (1 - t) * t ** 2 * BEZIER_VALUES[2]
        + t ** 3 * BEZIER_VALUES[3]
    )[..., 1]
    alphas = alphas[:, np.newaxis]
    new_points = (1 - alphas) * point1[np.newaxis] + alphas * point2[
        np.newaxis
    ]
    return new_points


def generate_points(num_attributes: int):
    points = [[MIN_VALUE] * num_attributes, [MAX_VALUE] * num_attributes]
    for i in range(num_attributes - 1, -1, -1):
        new_point = points[-1][:]
        new_point[i] = MIN_VALUE
        points.append(new_point)
    for i in range(num_attributes - 1, -1, -1):
        new_point = points[-1][:]
        new_point[i] = MAX_VALUE
        points.append(new_point)
    random_points = np.random.uniform(
        MIN_VALUE, MAX_VALUE, size=(NUM_RANDOM_POINTS, num_attributes)
    )

    points.extend(random_points.tolist())
    points.append([0.5] * num_attributes)
    points_array = np.array(points)
    return points_array


def generate_dynamics(points: np.ndarray) -> np.ndarray:
    output = []
    duration_per_combination = NUM_FRAMES // (len(points) - 1)
    for i in range(0, points.shape[0] - 1):
        start_point = points[i]
        end_point = points[i + 1]
        output.append(
            interpolate_points(
                start_point, end_point, duration_per_combination
            )
        )
    return np.concatenate(output, axis=0)


def get_single_node(index: int, value: float, name: str) -> str:
    if index == 0:
        return first_slider_template.format(value, name)
    return slider_template.format(index - 1, index, value, name)


def render_single(i: int, dynamic: np.ndarray, attribute_names: List[str]):
    nodes = []

    for j, name in enumerate(attribute_names):
        nodes.append(get_single_node(j, dynamic[j], name))
    tex_file = [preamble, body.format("\n".join(nodes))]
    with tempfile.NamedTemporaryFile(suffix=".tex", mode="w") as temp_file:
        content = "\n".join(tex_file)
        temp_file.write(content)
        temp_file.flush()
        path = Path("frames")
        compile_command = [
            "pdflatex",
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            f"-output-dir=/tmp",
            "-recorder",
            temp_file.name,
        ]

        convert_command = [
            "convert",
            "-density",
            "300",
            Path(temp_file.name).with_suffix(".pdf"),
            path / f"frame{i:04d}.jpg",
        ]

        subprocess.call(compile_command, stdout=subprocess.DEVNULL)
        subprocess.call(convert_command, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attribute_names", nargs="+")

    args = parser.parse_args()

    points = generate_points(len(args.attribute_names))
    dynamics = generate_dynamics(points)

    prender = functools.partial(
        render_single, attribute_names=args.attribute_names
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=mp.cpu_count() - 2
    ) as pool:
        list(
            tqdm.tqdm(
                pool.map(prender, list(range(len(dynamics))), dynamics),
                total=len(dynamics),
            )
        )


if __name__ == "__main__":
    main()
