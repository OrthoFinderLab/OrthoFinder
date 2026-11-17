#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert OrthoFinder orthogroup files into gene-count tables.

INPUT
  Input may be:
    - Orthogroups.tsv
    - Orthogroups.txt
    - N*.tsv hierarchical orthogroup tables
    - a directory containing any of the above

OUTPUT
  For tabular files:
      INPUT.GeneCount.tsv

  For Orthogroups.txt:
      Orthogroups.GeneCount.tsv

Examples
--------
orthogroup_gene_count Orthogroups.tsv

orthogroup_gene_count Orthogroups.txt

orthogroup_gene_count N0.tsv

orthogroup_gene_count Results_*/

orthogroup_gene_count Results_*/ -o GeneCountTables/
"""

import os
import argparse


class TabWriter(object):
    """The rows of a tab-separated file, as OrthoFinder writes them: tab-joined, unquoted (UTF-8)."""

    def __init__(self, outfile):
        self.outfile = outfile

    def writerow(self, row):
        self.outfile.write("\t".join(map(str, row)) + "\n")


def is_orthogroup_table(path):
    if not os.path.isfile(path):
        return False

    basename = os.path.basename(path)

    if basename in {"Orthogroups.tsv", "Orthogroups.txt"}:
        return True

    if basename.startswith("N") and basename.endswith(".tsv") and not basename.endswith(".GeneCount.tsv"):
        return True

    return False


def is_colon_orthogroups_file(path):
    return os.path.basename(path) == "Orthogroups.txt"


def default_output_file(input_file):
    root, ext = os.path.splitext(input_file)

    if os.path.basename(input_file) == "Orthogroups.txt":
        return root + ".txt.GeneCount.tsv"

    return root + ".GeneCount" + ext


def check_output(input_file, output_file):
    """Fails (ValueError) if output_file is input_file: writing it would destroy the input before it is read."""
    if os.path.realpath(output_file) == os.path.realpath(input_file) or \
            (os.path.exists(output_file) and os.path.samefile(output_file, input_file)):
        raise ValueError("ERROR: the output file is the input file, %s. Please write it elsewhere" % input_file)


def _replace(tmp_fn, output_file):
    os.replace(tmp_fn, output_file)


def convert_tabular_gene_counts(input_file, output_file=None):
    """The gene counts of a table of orthogroups or HOGs, with a Total column, as OrthoFinder's Orthogroups.GeneCount.tsv."""
    if output_file is None:
        output_file = default_output_file(input_file)
    check_output(input_file, output_file)
    tmp_fn = output_file + ".tmp"

    with open(input_file, encoding="utf-8") as infile, \
            open(tmp_fn, "w", encoding="utf-8", newline="\n") as outfile:
        writer = TabWriter(outfile)
        reader = (line.rstrip("\r\n").split("\t") for line in infile)
        header = next(reader)

        # HOG tables contain:
        # HOG, Level, Description, Species...
        #
        # Orthogroups.tsv contains:
        # Orthogroup, Species...
        n_col_skip = 3 if header[0] == "HOG" else 1

        writer.writerow(header + ["Total"])

        for line in reader:
            counts = [
                0 if cell == "" else len(cell.split(", "))
                for cell in line[n_col_skip:]
            ]

            writer.writerow(line[:n_col_skip] + counts + [sum(counts)])

    _replace(tmp_fn, output_file)
    print("Orthogroup gene count table written to:")
    print(output_file)


def convert_colon_gene_counts(input_file, output_file=None):
    if output_file is None:
        output_file = default_output_file(input_file)
    check_output(input_file, output_file)
    tmp_fn = output_file + ".tmp"

    with open(input_file, encoding="utf-8") as infile, \
            open(tmp_fn, "w", encoding="utf-8", newline="\n") as outfile:
        writer = TabWriter(outfile)
        writer.writerow(["Orthogroup", "GeneCount"])

        for line in infile:
            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                continue

            orthogroup, genes = line.split(":", 1)
            genes = genes.strip().split()

            writer.writerow([orthogroup, len(genes)])

    _replace(tmp_fn, output_file)
    print("Orthogroup gene count table written to:")
    print(output_file)


def convert_gene_counts(input_file, output_file=None):
    if is_colon_orthogroups_file(input_file):
        convert_colon_gene_counts(input_file, output_file)
    else:
        convert_tabular_gene_counts(input_file, output_file)


def get_input_files(input_path, output_dir=None):
    if os.path.isfile(input_path):
        if is_orthogroup_table(input_path):
            return [input_path]

        print("ERROR: input file is not a supported orthogroup file:")
        print(input_path)
        raise SystemExit(1)

    if os.path.isdir(input_path):
        files = []

        for root, dirs, filenames in os.walk(input_path):
            # (not the tables of an earlier conversion: its output directory)
            dirs[:] = [d for d in dirs if output_dir is None
                       or os.path.abspath(os.path.join(root, d)) != os.path.abspath(output_dir)]
            for filename in sorted(filenames):
                path = os.path.join(root, filename)

                if is_orthogroup_table(path):
                    files.append(path)

        return sorted(files)

    print("ERROR: input path does not exist:")
    print(input_path)
    raise SystemExit(1)


def convert_directory(input_dir, output_dir=None):
    input_dir = os.path.abspath(input_dir)

    if output_dir is None:
        output_dir = os.path.join(input_dir, "GeneCountTables")
    else:
        output_dir = os.path.abspath(output_dir)

    files_to_process = get_input_files(input_dir, output_dir)

    if not files_to_process:
        print("ERROR: no orthogroup files found in directory:")
        print(input_dir)
        raise SystemExit(1)

    os.makedirs(output_dir, exist_ok=True)

    for input_file in files_to_process:
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, rel_path)
        output_file = default_output_file(output_file)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print("\nProcessing:")
        print(input_file)

        convert_gene_counts(input_file, output_file)


def main(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Orthogroups.tsv, Orthogroups.txt, N*.tsv file, or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output filename for file input, or output directory for directory input "
            "[default: INPUT.GeneCount.tsv or INPUT/GeneCountTables]"
        ),
    )

    parsed = parser.parse_args(args)

    input_path = os.path.abspath(parsed.input)

    try:
        if os.path.isdir(input_path):
            convert_directory(input_path, parsed.output)
        else:
            convert_gene_counts(input_path, parsed.output)
    except ValueError as e:
        print(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()