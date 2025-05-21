import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.qa_model import answer_question
from models.summarizer import summarize_text
from models.comprehension import comprehend_document

import argparse

def main():
    parser = argparse.ArgumentParser(description="Contextual Language Understanding CLI")

    subparsers = parser.add_subparsers(dest="command")

    qa_parser = subparsers.add_parser("qa")
    qa_parser.add_argument("--context_file", required=True)

    sum_parser = subparsers.add_parser("summarize")
    sum_parser.add_argument("--file", required=True)

    comp_parser = subparsers.add_parser("comprehend")
    comp_parser.add_argument("--file", required=True)

    args = parser.parse_args()

    if args.command == "qa":
        print("Answer:", answer_question(args.context_file))
    elif args.command == "summarize":
        print("Summary:", summarize_text(args.file))
    elif args.command == "comprehend":
        print("Comprehension Output:", comprehend_document(args.file))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
