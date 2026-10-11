"""CLI for the shared Robo project."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="robo", description="Robo vision tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("detect", help="Run live panel detection and save JSON logs")
    args = parser.parse_args(argv)

    if args.command == "detect":
        from .detection.cli import main as detect

        detect()
