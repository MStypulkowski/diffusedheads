import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

preamble = r"""\documentclass[tikz,crop]{standalone}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,shadows,shadows.blur,matrix,backgrounds}
\tikzset{white background/.style={show background rectangle,tight background,
background rectangle/.style={fill=white}}}
"""

body = """
\\begin{{document}}
\\setlength{{\\fboxsep}}{{0pt}}
\\begin{{tikzpicture}}[white background]
{}
\\end{{tikzpicture}}
\\end{{document}}
"""

text_template = """
\\node[align=center] {{{}}};
"""


np.random.seed(0xCAFFE)


def render_single(text: str, name: str):
    node = text_template.format(text)
    tex_file = [preamble, body.format(node)]
    with tempfile.NamedTemporaryFile(suffix=".tex", mode="w") as temp_file:
        content = "\n".join(tex_file)
        temp_file.write(content)
        temp_file.flush()
        path = Path("texts")
        path.mkdir(exist_ok=True, parents=True)
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
            path / f"{name}.jpg",
        ]

        subprocess.call(compile_command, stdout=subprocess.DEVNULL)
        subprocess.call(convert_command, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("name")

    args = parser.parse_args()
    render_single(args.text, args.name)


if __name__ == "__main__":
    main()
