"""Backward-compatible source entrypoint."""

try:
    from cogcappro.cli.train import main
except ModuleNotFoundError as exc:
    if exc.name != "cogcappro":
        raise
    from src.cogcappro.cli.train import main


if __name__ == "__main__":
    main()
