"""tools/species_overlaps.py: the orthogroups shared by each pair of species."""
import random

import numpy as np
import pytest

from tools import species_overlaps as so


def random_table(rng, kind, n_sp, n_og):
    """A table of orthogroups (kind "counts", "genes" or "hogs") and the species of each orthogroup."""
    species = ["Sp%d" % i for i in range(n_sp)]
    present = []
    lines = []
    header = {"counts": ["Orthogroup"] + species + ["Total"], "counts_no_total": ["Orthogroup"] + species,
              "genes": ["Orthogroup"] + species,
              "hogs": ["HOG", "OG", "Gene Tree Parent Clade"] + species}[kind]
    for i in range(n_og):
        n = [rng.choice([0, 0, 0, 1, 2, 10, 105]) for _ in species]
        present.append({sp for sp, k in zip(species, n) if k})
        if kind.startswith("counts"):
            cells = ["OG%07d" % i] + [str(k) for k in n] + ([str(sum(n))] if kind == "counts" else [])
        else:
            genes = [", ".join("g%d_%d" % (i, j) for j in range(k)) for k in n]
            cells = (["OG%07d" % i] if kind == "genes" else ["N1.HOG%07d" % i, "OG%07d" % i, "n%d" % i]) + genes
        lines.append("\t".join(cells))
    return "\t".join(header) + "\n" + "\n".join(lines) + "\n", species, present


def brute_force(species, present):
    return [[sum(1 for p in present if a in p and b in p) for b in species] for a in species]


@pytest.mark.parametrize("kind", ["counts", "counts_no_total", "genes", "hogs"])
@pytest.mark.parametrize("block_bytes", [1, 37, 10 ** 6])     # lines across the blocks, and one block
def test_as_counted_one_by_one(tmp_path, kind, block_bytes):
    rng = random.Random(block_bytes)
    text, species, present = random_table(rng, kind, 7, 60)
    # (counts without a Total column: told by the name, as orthogroup_gene_count's older tables)
    fn = tmp_path / ("table.GeneCount.tsv" if kind == "counts_no_total" else "table.tsv")
    fn.write_text(text)
    got_species, shared, n = so.species_overlaps(str(fn), block_bytes=block_bytes)
    assert got_species == species and n == 60
    assert shared.tolist() == brute_force(species, present)
    assert shared.dtype == np.int64


@pytest.mark.parametrize("change", ["crlf", "no final newline", "blank lines"])
def test_line_endings(tmp_path, change):
    text, species, present = random_table(random.Random(1), "counts", 5, 30)
    if change == "crlf":
        text = text.replace("\n", "\r\n")
    elif change == "no final newline":
        text = text[:-1]
    else:
        text = text.replace("\n", "\n\n", 3) + "\n\n"
    fn = tmp_path / "table.tsv"
    fn.write_text(text, newline="")
    _, shared, n = so.species_overlaps(str(fn), block_bytes=50)
    assert n == 30 and shared.tolist() == brute_force(species, present)


def test_wrong_columns(tmp_path):
    fn = tmp_path / "table.tsv"
    fn.write_text("Orthogroup\tA\tB\tTotal\nOG0000000\t1\t0\t1\nOG0000001\t1\t1\n")
    with pytest.raises(ValueError, match="columns"):
        so.species_overlaps(str(fn))
    fn.write_text("Species\tA\tB\n")
    with pytest.raises(ValueError, match="not a table"):
        so.species_overlaps(str(fn))


def test_results_directory(tmp_path, capsys):
    text, species, present = random_table(random.Random(2), "counts", 4, 20)
    (tmp_path / "Orthogroups").mkdir()
    (tmp_path / "Orthogroups" / "Orthogroups.GeneCount.tsv").write_text(text)
    assert so.main([str(tmp_path)]) == 0
    out = (tmp_path / "Comparative_Genomics_Statistics" / so.OUTPUT_FN).read_text().splitlines()
    assert out[0] == "\t" + "\t".join(species)
    assert [list(map(int, line.split("\t")[1:])) for line in out[1:]] == brute_force(species, present)
    assert [line.split("\t")[0] for line in out[1:]] == species


@pytest.mark.parametrize("header", [["Orthogroup"], ["HOG", "OG", "Gene Tree Parent Clade"]])
def test_orthogroup_gene_count_tables(tmp_path, header):
    # tools/orthogroup_gene_count writes counts without a Total column: 0 is not present
    from tools import orthogroup_gene_count
    lead = [["OG1", "x", "n"][:len(header)], ["OG2", "y", "n"][:len(header)]]
    (tmp_path / "t.tsv").write_text("\t".join(header + ["A", "B"]) + "\n" + "\t".join(lead[0] + ["a1", ""]) + "\n"
                                    + "\t".join(lead[1] + ["", "b1, b2"]) + "\n")
    orthogroup_gene_count.convert_tabular_gene_counts(str(tmp_path / "t.tsv"), str(tmp_path / "c.tsv"))
    species, shared, n = so.species_overlaps(str(tmp_path / "c.tsv"))
    assert species == ["A", "B"] and shared.tolist() == [[1, 0], [0, 1]] and n == 2
    assert so.species_overlaps(str(tmp_path / "t.tsv"))[1].tolist() == [[1, 0], [0, 1]]


def test_genes_named_by_numbers(tmp_path):
    # a table of genes, its genes numbers (gene 0 too): genes, not counts, also with no empty cells
    (tmp_path / "t.tsv").write_text("Orthogroup\tA\tB\nOG1\t12\t\nOG2\t0\t7\n")
    assert so.species_overlaps(str(tmp_path / "t.tsv"))[1].tolist() == [[2, 1], [1, 1]]
    (tmp_path / "u.tsv").write_text("Orthogroup\tA\tB\nOG1\t0\t7\nOG2\t3\t0\n")
    assert so.species_overlaps(str(tmp_path / "u.tsv"))[1].tolist() == [[2, 2], [2, 2]]
    # counts: by the Total column or the name
    (tmp_path / "v.GeneCount.tsv").write_text("Orthogroup\tA\tB\nOG1\t0\t7\nOG2\t3\t0\n")
    assert so.species_overlaps(str(tmp_path / "v.GeneCount.tsv"))[1].tolist() == [[1, 0], [0, 1]]
