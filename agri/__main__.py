from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import config as C
from .data import load, save
from .pipeline import run
from .report import build_html, text_summary
from .synthetic import generate


def parse_contracts(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    keys = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = [k for k in keys if k not in C.CONTRACTS]
    if unknown:
        raise SystemExit(f"unknown contracts: {', '.join(unknown)}; choose from {', '.join(C.CONTRACTS)}")
    return keys


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contracts", help="comma separated subset, e.g. corn,wheat")
    parser.add_argument("--horizon", type=int, default=C.HORIZON_WEEKS, help="forecast horizon in weeks")
    parser.add_argument("--out", default="outputs", help="folder for the HTML report")
    parser.add_argument("--cdn", action="store_true", help="load plotly.js from a CDN instead of embedding it")


def positive_horizon(value: int) -> int:
    if value < 1:
        raise SystemExit("--horizon must be at least 1 week")
    return value


def emit(analysis, args, synthetic: bool) -> None:
    print(text_summary(analysis))
    path = build_html(analysis, Path(args.out) / "report.html", synthetic=synthetic, offline=not args.cdn)
    print(f"\nreport written to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agri", description="Agricultural commodities analytics")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the full pipeline on synthetic data")
    add_common(demo)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--save-data", help="also write the synthetic inputs as CSV to this folder")
    runner = sub.add_parser("run", help="run the pipeline on your own CSV inputs")
    add_common(runner)
    runner.add_argument("--data", required=True, help="folder containing the input CSV files")
    template = sub.add_parser("export-template", help="write an example input folder to fill in")
    template.add_argument("folder")
    template.add_argument("--start", default="2012-01-01")
    args = parser.parse_args(argv)
    if args.command == "demo":
        bundle = generate(seed=args.seed)
        if args.save_data:
            save(bundle, args.save_data)
        emit(run(bundle, parse_contracts(args.contracts), positive_horizon(args.horizon)), args, True)
    elif args.command == "run":
        try:
            bundle = load(args.data)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 2
        emit(run(bundle, parse_contracts(args.contracts), positive_horizon(args.horizon)), args, False)
    else:
        save(generate(start=args.start), args.folder)
        print(f"example inputs written to {args.folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
