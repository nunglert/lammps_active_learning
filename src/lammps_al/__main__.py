"""Top-level ``lammps-al`` command dispatcher.

The one console script for this project (mirrors ``jaxrens-al``). It routes
to the sub-tools, keeping their own argument parsers untouched::

    lammps-al dashboard <calchome> ...   -> lammps_al.dashboard.cli:main
    lammps-al --version                  -> print the installed version

Also runnable as ``python -m lammps_al``.
"""

from __future__ import annotations

import sys

from lammps_al import __version__

_USAGE = f"""\
usage: lammps-al [--version] <command> [<args>...]

LAMMPS active-learning command-line interface (version {__version__}).

commands:
  dashboard   Build the multipage dashboard PDF for an AL campaign's calchome.
              (run 'lammps-al dashboard --help' for its own options)

options:
  -V, --version   Print the installed version and exit.
  -h, --help      Show this message and exit.
"""


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``lammps-al`` console script."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE, end="")
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "dashboard":
        # Lazy: the dashboard needs the [dashboard] extra (alviz, ...), which
        # `lammps-al --version`/`--help` should not.
        try:
            from lammps_al.dashboard.cli import main as dashboard_main
        except ImportError as exc:
            print(
                f"lammps-al dashboard: {exc}\n"
                "Install the dashboard dependencies with "
                "`pip install lammps_active_learning[dashboard]`.",
                file=sys.stderr,
            )
            return 1
        return dashboard_main(rest)

    print(
        f"lammps-al: unknown command {cmd!r}\n\n{_USAGE}",
        end="",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
