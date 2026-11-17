#!/usr/bin/env python3

"""
Write the number of orthogroups shared by each pair of species, from the
results of a finished run, as Orthogroups_SpeciesOverlaps.tsv (the table of
earlier OrthoFinder versions). The diagonal is the number of orthogroups of
each species.

INPUT is an OrthoFinder results directory, which is read from
Orthogroups/Orthogroups.GeneCount.tsv (or MCL_Orthogroups.GeneCount.tsv after
-og), or a table of orthogroups or HOGs itself, such as Orthogroups.tsv,
Orthogroups.GeneCount.tsv or the HOGs of a species tree node,
Phylogenetic_Hierarchical_Orthogroups/N1.tsv for example.
The table is written to Comparative_Genomics_Statistics/ of the results
directory, or to the file given with -o.

The table is read in blocks with numpy and the counts are a sparse matrix
product, so the time grows with the species of each orthogroup, not with
every pair of species for every orthogroup.

Examples
--------
species_overlaps Results_*/

species_overlaps Results_*/Phylogenetic_Hierarchical_Orthogroups/N1.tsv -o N1_SpeciesOverlaps.tsv
"""

import argparse
import os
import sys

import numpy as np
from scipy import sparse

OUTPUT_FN = "Orthogroups_SpeciesOverlaps.tsv"
# The leading columns of the tables that are not species, by their first header cell
LEADING_COLUMNS = {"Orthogroup": 1, "HOG": 3}
# The bytes of the table read at a time
BLOCK_BYTES = 16 * 1024 * 1024
TAB, NEWLINE, CR, ZERO = 9, 10, 13, ord("0")


def table_of(results_dir):
    """The gene count table of a results directory (or its MCL orthogroups'), or None."""
    for name in ("Orthogroups", "MCL_Orthogroups"):
        fn = os.path.join(results_dir, "Orthogroups", name + ".GeneCount.tsv")
        if os.path.exists(fn):
            return fn
    return None


def read_header(fn):
    """
    (species names, the columns before them, whether the cells are gene
    counts). A table of counts is one with a Total column, as OrthoFinder's
    GeneCount.tsv and orthogroup_gene_count's, or one named *.GeneCount.*
    (those orthogroup_gene_count wrote before it had a Total column). It is
    not guessed from the cells: genes can be named by numbers.
    """
    with open(fn, "rb") as f:
        header = f.readline().rstrip(b"\r\n").decode("utf-8").split("\t")
        lead = LEADING_COLUMNS.get(header[0])
        if lead is None:
            raise ValueError("%s is not a table of orthogroups or HOGs (its first column is %r)" % (fn, header[0]))
        counts = header[-1] == "Total" or ".GeneCount." in os.path.basename(fn)
    species = header[lead:len(header) - 1 if header[-1] == "Total" else len(header)]
    if not species:
        raise ValueError("%s has no species columns" % fn)
    return species, lead, counts, header[-1] == "Total"


def present_cells(data, n_fields, lead, n_species, counts):
    """
    The (row, species) of the cells with genes in data (whole lines, bytes),
    as numpy arrays, in row order. A cell has genes if it is a species' genes
    (not empty) or its gene count (not "0", so it has a digit 1 to 9). The
    cells are found from the positions of the tabs and newlines.
    """
    a = np.frombuffer(data, dtype=np.uint8)
    if (a == CR).any():
        a = a[a != CR]
    ends = np.flatnonzero((a == TAB) | (a == NEWLINE))         # the end of each cell
    n_lines = ends.size // n_fields
    if ends.size != n_lines * n_fields or not (a[ends[n_fields - 1::n_fields]] == NEWLINE).all():
        raise ValueError("a line has not %d columns" % n_fields)
    if counts:
        # the cells of the bytes 1 to 9 (few: most counts are 0), each cell once
        digits = np.flatnonzero((a >= ZERO + 1) & (a <= ZERO + 9))
        cells = np.searchsorted(ends, digits)
        cells = cells[np.r_[True, cells[1:] != cells[:-1]]]
    else:
        starts = np.empty_like(ends)
        starts[0] = 0
        starts[1:] = ends[:-1] + 1
        cells = np.flatnonzero(ends > starts)
    rows, cols = np.divmod(cells, n_fields)
    keep = (cols >= lead) & (cols < lead + n_species)       # (not the orthogroup's name, nor its total)
    return rows[keep], cols[keep] - lead


def species_overlaps(fn, block_bytes=None):
    """
    (species names, the number of orthogroups shared by each pair of species
    (int64, species x species), the number of orthogroups) of a table.
    """
    if block_bytes is None:
        block_bytes = BLOCK_BYTES
    species, lead, counts, total = read_header(fn)
    n_sp = len(species)
    n_fields = lead + n_sp + (1 if total else 0)
    shared = np.zeros((n_sp, n_sp), dtype=np.int64)
    n_rows = 0
    with open(fn, "rb") as f:
        f.readline()
        rest = b""
        while True:
            block = f.read(block_bytes)
            data = rest + block
            if not block:
                if data and not data.endswith(b"\n"):
                    data += b"\n"
                rest = b""
            else:
                cut = data.rfind(b"\n") + 1
                data, rest = data[:cut], data[cut:]
            if b"\n\n" in data or data.startswith(b"\n"):      # (blank lines, e.g. at the end)
                data = b"".join(line + b"\n" for line in data.split(b"\n") if line.strip())
            if data:
                rows, cols = present_cells(data, n_fields, lead, n_sp, counts)
                n_block = int(data.count(b"\n"))
                # orthogroups x species, 1 where the species has genes; shared = P^T P
                p = sparse.csr_matrix((np.ones(rows.size, dtype=np.int64), (rows, cols)), shape=(n_block, n_sp))
                shared += (p.T @ p).toarray()
                n_rows += n_block
            if not block:
                break
    return species, shared, n_rows


def write_overlaps(fn, species, shared):
    with open(fn, "w", encoding="utf-8", newline="\n") as outfile:
        outfile.write("\t".join([""] + species) + "\n")
        for name, row in zip(species, shared):
            outfile.write(name + "\t" + "\t".join(map(str, row.tolist())) + "\n")


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="species_overlaps",
        description="Write the number of orthogroups shared by each pair of species (Orthogroups_SpeciesOverlaps.tsv).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("Examples"):])
    parser.add_argument("input", metavar="INPUT",
                        help="an OrthoFinder results directory, or a table of orthogroups or HOGs")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="the file to write (default Comparative_Genomics_Statistics/%s of the results "
                             "directory, or next to the table)" % OUTPUT_FN)
    a = parser.parse_args(args)

    if os.path.isdir(a.input):
        fn = table_of(a.input)
        if fn is None:
            parser.error("no Orthogroups/Orthogroups.GeneCount.tsv in %s" % a.input)
        out_dir = os.path.join(a.input, "Comparative_Genomics_Statistics")
    elif os.path.isfile(a.input):
        fn = a.input
        out_dir = os.path.dirname(os.path.abspath(fn))
    else:
        parser.error("does not exist: %s" % a.input)
    out_fn = a.output or os.path.join(out_dir, OUTPUT_FN)

    try:
        species, shared, n_rows = species_overlaps(fn)
    except (ValueError, OSError, UnicodeDecodeError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    if os.path.dirname(out_fn):
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    write_overlaps(out_fn, species, shared)
    print("%d species, %d orthogroups (%s). Written to %s" % (len(species), n_rows, os.path.basename(fn), out_fn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
