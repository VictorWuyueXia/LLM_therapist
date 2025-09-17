#!/usr/bin/env python
# test/test_RA_RV_CBT.py
# This script demonstrates basic usage of the response analyzer, reflection validation, and CBT engine modules.
# It is intended to be run as a standalone executable for quick testing and debugging.

import sys
import os

# Add the parent directory to the Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.response_analyzer import classify_dimension_and_score
from scripts.reflection_validation import rv_reasoner
from scripts.cbt_engine import stage1_guide

def main():
    # Test the response analyzer: classify a user input into dimension and score
    result1 = classify_dimension_and_score("My weight increased a lot recently.")
    print("Dimension and Score Classification:", result1)

    # Test the reflection validation: check if a follow-up response is related to the topic
    result2 = rv_reasoner(
        "Maintaining stable weight",
        "My weight increased a lot.",
        "I like movies."
    )
    print("Reflection Validation (Reasoner):", result2)

    # Test the CBT engine: generate unhelpful thoughts based on a statement
    result3 = stage1_guide("I avoid speaking in meetings.")
    print("CBT Stage 1 Guide:", result3)

if __name__ == "__main__":
    main()
