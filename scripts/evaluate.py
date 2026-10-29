"""Evaluate a saved checkpoint at its configured angles."""
import argparse
from src.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device')
    ap.add_argument('--extended', action='store_true', help='Save eight-angle diagnostic separately from the primary evaluation')
    args = ap.parse_args()
    print(evaluate(args.checkpoint, args.device, args.extended))


if __name__ == '__main__':
    main()
