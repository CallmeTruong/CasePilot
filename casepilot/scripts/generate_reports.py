"""Export reports from recorded artifacts, never synthesized measurements."""

import argparse

from casepilot.evaluation.reporting import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/reports")
    args = parser.parse_args()
    print(generate(args.results, args.output))
