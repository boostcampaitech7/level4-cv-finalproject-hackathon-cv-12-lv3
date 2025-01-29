#!/usr/bin/env python3
"""A command line tool for extracting text and images from PDF and
output it to plain text, html, xml or tags.
"""

from __future__ import annotations
import argparse
import logging
import sys
import os
import json
from string import Template
from typing import List, Optional

from pdf2zh import __version__, log
from pdf2zh.high_level import translate
from pdf2zh.doclayout import OnnxModel, ModelInstance

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument(
        "files",
        type=str,
        default=None,
        nargs="*",
        help="One or more paths to PDF files.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"pdf2zh v{__version__}",
    )
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help="Use debug logging level.",
    )
    parse_params = parser.add_argument_group(
        "Parser",
        description="Used during PDF parsing",
    )
    parse_params.add_argument(
        "--vfont",
        "-f",
        type=str,
        default="",
        help="The regex to math font name of formula.",
    )
    parse_params.add_argument(
        "--vchar",
        "-c",
        type=str,
        default="",
        help="The regex to math character of formula.",
    )
    parse_params.add_argument(
        "--lang-in",
        "-li",
        type=str,
        default="en",
        help="The code of source language.",
    )
    parse_params.add_argument(
        "--lang-out",
        "-lo",
        type=str,
        default="ko",
        help="The code of target language.",
    )
    parse_params.add_argument(
        "--service",
        "-s",
        type=str,
        default="google",
        help="The service to use for translation.",
    )
    parse_params.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Output directory for files.",
    )
    parse_params.add_argument(
        "--thread",
        "-t",
        type=int,
        default=4,
        help="The number of threads to execute translation.",
    )
    return parser


def parse_args(args: Optional[List[str]]) -> argparse.Namespace:
    parsed_args = create_parser().parse_args(args=args)
    return parsed_args

def save_new_json(page_number) :
    if os.path.exists('original.json'):
        with open('original.json', 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    if os.path.exists('new.json') :
        with open('new.json', 'r') as f :
            new_data = json.load(f)
    else :
        new_data = {}

    if str(page_number) not in existing_data :
        return False

    new_data[str(page_number)] = ' '.join(existing_data[str(page_number)])

    with open('new.json', 'w') as f:
        json.dump(new_data, f,ensure_ascii=False, indent=4)

    return True
def main(args: Optional[List[str]] = None) -> int:
    logging.basicConfig()
    parsed_args = parse_args(args)
    ModelInstance.value = OnnxModel.load_available()

    print(parsed_args)
    translate(model=ModelInstance.value, **vars(parsed_args))

    idx = 0
    while True :
        if save_new_json(idx) :
            idx += 1
        else :
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
