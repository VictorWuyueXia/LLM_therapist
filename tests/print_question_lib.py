"""
Small utility to print the question library in a human-readable structured format
and count the total number of DLA items.

Notes per project conventions:
- Avoid default function parameter values.
- Avoid try/with unless necessary; let errors surface.
- Use logging for basic run information.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_question_lib(question_lib_path: Path) -> Dict[str, Any]:
    """Load the question library JSON as a nested dict.

    Parameters
    ----------
    question_lib_path: Path
        Absolute path to the question library JSON file.
    """
    file_ref = open(question_lib_path, "r", encoding="utf-8")
    text = file_ref.read()
    file_ref.close()
    return json.loads(text)


def count_total_questions(question_lib: Dict[str, Any]) -> int:
    """Count total number of question strings across all items."""
    total = 0
    for top_key in sorted(question_lib.keys(), key=int):
        section = question_lib[top_key]
        for item_key in sorted(section.keys(), key=int):
            item = section[item_key]
            questions = item["question"]
            total += len(questions)
    return total


def count_total_dla_items(question_lib: Dict[str, Any]) -> int:
    """Count total number of DLA items across all sections."""
    total = 0
    for top_key in question_lib.keys():
        section = question_lib[top_key]
        total += len(section)
    return total


def print_human_readable(question_lib: Dict[str, Any]) -> None:
    """Print only necessary info in a structured, minimal format.

    Format:
    [Section] <section_id>
      - <label> | <name> | questions: <count>
    """
    for top_key in sorted(question_lib.keys(), key=int):
        print(f"[Section] {top_key}")
        section = question_lib[top_key]
        for item_key in sorted(section.keys(), key=int):
            item = section[item_key]
            label = item["label"]
            name = item["name"]
            q_count = len(item["question"])
            print(f"  - {label} | {name} | questions: {q_count}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    question_lib_path = project_root / "data" / "libs" / "question_lib_v3_8901.json"
    logging.info(f"Loading question library: {question_lib_path}")
    question_lib = load_question_lib(question_lib_path)
    print_human_readable(question_lib)
    total_dla_items = count_total_dla_items(question_lib)
    print(f"TOTAL_DLA_ITEMS: {total_dla_items}")


if __name__ == "__main__":
    main()


