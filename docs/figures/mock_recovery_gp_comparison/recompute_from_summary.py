"""Recompute, from the summary table alone, the two numbers that compare the two fits of the mock recovery.

The table beside this file, ``gp_comparison_posterior_summary.csv``, holds the posterior mean and standard deviation
of eleven parameters for two fits of one mock: ``hcd_priya`` and the Gaussian-process emulator (``gp``).
``PROVENANCE.md`` beside it says what the fits are and how the table was made. This script reads only the table and
prints:

- the largest difference of the two posterior means, in units of one fit's own standard deviation: for each
  parameter the larger of ``|mean_hcd_priya - mean_gp| / sd_hcd_priya`` and ``|mean_hcd_priya - mean_gp| / sd_gp``,
  then the largest over the parameters, with the name of that parameter;
- the smallest and the largest ratio of the posterior widths, ``sd_hcd_priya / sd_gp``, over the parameters.

    python docs/figures/mock_recovery_gp_comparison/recompute_from_summary.py [TABLE.csv]

It needs the standard library only. No chain is read: the chains are not published.
"""

from __future__ import annotations

import csv
import pathlib
import sys

TABLE = pathlib.Path(__file__).with_name("gp_comparison_posterior_summary.csv")
COLUMNS = ("mean_hcd_priya", "sd_hcd_priya", "mean_gp", "sd_gp")


def read_table(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """Read the summary table.

    Parameters
    ----------
    path : pathlib.Path
        A csv file whose lines starting with ``#`` are comments, with the columns ``parameter``,
        ``mean_hcd_priya``, ``sd_hcd_priya``, ``mean_gp`` and ``sd_gp``.

    Returns
    -------
    dict of str to dict of str to float
        For each parameter, in the order of the file, its four numbers by column name.

    Raises
    ------
    ValueError
        If the file has no row, a row lacks a column, a parameter is listed twice, or a standard deviation is not
        positive.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row.get("parameter")
        if not name or any(row.get(column) in (None, "") for column in COLUMNS):
            raise ValueError(
                f"{path}: a row lacks one of the columns parameter, {', '.join(COLUMNS)}"
            )
        if name in table:
            raise ValueError(f"{path}: the parameter {name} is listed twice")
        values = {column: float(row[column]) for column in COLUMNS}
        if not (values["sd_hcd_priya"] > 0 and values["sd_gp"] > 0):
            raise ValueError(f"{path}: a standard deviation of {name} is not positive")
        table[name] = values
    if not table:
        raise ValueError(f"{path}: the table has no row")
    return table


def largest_mean_difference(table: dict[str, dict[str, float]]) -> tuple[float, str]:
    """The largest difference of the two posterior means, in units of one fit's own standard deviation.

    Parameters
    ----------
    table : dict of str to dict of str to float
        The table, as ``read_table`` returns it.

    Returns
    -------
    tuple of float and str
        The largest value over the parameters of ``|mean_hcd_priya - mean_gp|`` divided by the smaller of the two
        standard deviations (which gives the larger of the two ratios), and the parameter where it occurs.
    """
    differences = []
    for name, row in table.items():
        gap = abs(row["mean_hcd_priya"] - row["mean_gp"])
        differences.append((gap / min(row["sd_hcd_priya"], row["sd_gp"]), name))
    return max(differences)


def width_ratio_range(table: dict[str, dict[str, float]]) -> tuple[float, float]:
    """The smallest and the largest ratio of the posterior widths, hcd_priya over the Gaussian-process emulator.

    Parameters
    ----------
    table : dict of str to dict of str to float
        The table, as ``read_table`` returns it.

    Returns
    -------
    tuple of float and float
        The smallest and the largest value of ``sd_hcd_priya / sd_gp`` over the parameters.
    """
    ratios = [row["sd_hcd_priya"] / row["sd_gp"] for row in table.values()]
    return min(ratios), max(ratios)


def main(argv: list[str]) -> int:
    """Print the two numbers for a table.

    Parameters
    ----------
    argv : list of str
        The command-line arguments after the script's name: nothing, or the path of a table.

    Returns
    -------
    int
        0 after printing; 2 for a wrong number of arguments.
    """
    if len(argv) > 1:
        print("usage: recompute_from_summary.py [TABLE.csv]")
        return 2
    table = read_table(pathlib.Path(argv[0]) if argv else TABLE)
    difference, where = largest_mean_difference(table)
    smallest, largest = width_ratio_range(table)
    print(f"parameters: {len(table)}")
    print(
        f"largest difference of means, in one fit's own standard deviation: {difference:.2f} ({where})"
    )
    print(f"ratio of posterior widths, hcd_priya over GP: {smallest:.2f} to {largest:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
