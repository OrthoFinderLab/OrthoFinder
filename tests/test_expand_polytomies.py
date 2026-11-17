"""tools/expand_polytomies.py: a run's polytomies' ortholog rows per pair of children, as written before they were grouped."""

import random
import re

import pytest

from orthofinder.orthology import tree_analyser
from orthofinder.phylo import taxa as phylo_taxa
from tools import expand_polytomies

import test_polytomy_orthologs as tpo

N = tpo.N_SPECIES
SPECIES = ["Sp%d" % i for i in range(N)]


def write_results(d, trees):
    """
    A results directory of the trees ({iog: newick}): the ortholog and
    xenologue files as the run writes them (grouped), the resolved trees and
    Polytomies.tsv. Returns the files per pair of children (per_pair), as
    {(kind, species): rows}.
    """
    olog, sus, expected_o, expected_x = ([[] for _ in range(N)] for _ in range(4))
    records, tree_lines = [], []
    for iog, newick in sorted(trees.items()):
        polytomies = []
        orthologs, recon, suspect, _ = tpo.analyse(newick, polytomies)
        og = "OG%07d" % iog
        polytomies = [(og,) + r[1:] for r in polytomies]
        records.extend(polytomies)
        for rows, entries in ((olog, orthologs), (expected_o, tpo.per_pair(orthologs))):
            o, _, _ = tpo.lines(entries, suspect)
            for i in range(N):
                rows[i].extend(r.replace("OG0000007", og) for r in o[i])
        for rows, entries in ((sus, orthologs), (expected_x, tpo.per_pair(orthologs))):
            _, x, _ = tpo.lines(entries, suspect)
            for i in range(N):
                rows[i].extend(r.replace("OG0000007", og) for r in x[i])
        fn = d / ("tree_%d.txt" % iog)
        phylo_taxa.rename_tree_taxa(recon, str(fn), {g: g for g in recon.get_leaf_names()},
                                    q_support=False, q_fix_negatives=True)
        tree_lines.append("%s: %s\n" % (og, "".join(fn.read_text().split())))
    (d / "Resolved_Gene_Trees").mkdir()
    (d / "Resolved_Gene_Trees" / "Resolved_Gene_Trees.txt").write_text("".join(tree_lines))
    tree_analyser.write_polytomies(records, str(d / "Resolved_Gene_Trees"))
    for name, header, rows in (("Orthologs", "Orthogroup\tSpecies\t%s\tOrthologs", olog),
                               ("Putative_Xenologs", "Orthogroup\t%s\tOther", sus)):
        (d / name).mkdir()
        for i, sp in enumerate(SPECIES):
            (d / name / ("%s.tsv" % sp)).write_text(
                header % sp + "\n" + "".join(r + "\n" for r in sorted(rows[i])))
    expected = {}
    for kind, rows in (("o", expected_o), ("x", expected_x)):
        for i, sp in enumerate(SPECIES):
            expected[(kind, sp)] = sorted(rows[i])
    return expected, records


def read_output(d):
    out = {}
    for kind, name in (("o", "Orthologs"), ("x", "Putative_Xenologs")):
        for sp in SPECIES:
            lines = (d / "Expanded_Orthologs" / name / ("%s.tsv" % sp)).read_text().splitlines()
            out[(kind, sp)] = lines[1:]
    return out


def several_genes(seed):
    rng = random.Random(seed)
    children = []
    for i in range(40):
        sp = rng.randrange(N)
        n = rng.choice([1, 1, 2, 3])
        genes = ["%d_%d" % (sp, 10 * i + j) for j in range(n)]
        children.append(genes[0] if n == 1 else "(%s)" % ",".join(genes))
    return "(%s);" % ",".join(children)


def several_species(seed):
    rng = random.Random(seed)
    children = []
    for i in range(40):
        sps = rng.sample(range(N), rng.choice([1, 1, 2, 3]))
        genes = ["%d_%d" % (sp, 10 * i + j) for j, sp in enumerate(sps)]
        children.append(genes[0] if len(genes) == 1 else "(%s)" % ",".join(genes))
    return "(%s);" % ",".join(children)


@pytest.mark.parametrize("seed", range(3))
def test_rows_per_pair_of_children(tmp_path, seed):
    # several orthogroups, polytomies within polytomies and below binary nodes
    trees = {3: tpo.star(80, seed), 5: several_genes(seed), 8: several_species(seed),
             12: "((%s),(%s));" % (tpo.star(20, seed + 10)[1:-2], "0_900,1_901,(2_902,3_903)"),
             13: "((0_1,1_2),(2_3,3_4));"}            # (no polytomy)
    expected, records = write_results(tmp_path, trees)
    assert any(r[8] for r in records)
    assert expand_polytomies.main([str(tmp_path)]) == 0
    assert read_output(tmp_path) == expected


@pytest.mark.parametrize("newick", tpo.SUSPECT_TREES)
def test_with_suspect_genes(tmp_path, monkeypatch, newick):
    tpo.misplaced(monkeypatch, "0_900", "0_902")
    expected, _ = write_results(tmp_path, {4: newick})
    assert any(expected[("x", sp)] for sp in SPECIES)
    assert expand_polytomies.main([str(tmp_path)]) == 0
    assert read_output(tmp_path) == expected


def test_species_prefixed_tree_names(tmp_path):
    # the run's trees name genes <species>_<gene>, the ortholog files <gene>
    expected, _ = write_results(tmp_path, {3: tpo.star(30, 1)})
    fn = tmp_path / "Resolved_Gene_Trees" / "Resolved_Gene_Trees.txt"
    og, newick = fn.read_text().split(": ", 1)
    newick = re.sub(r"(?<=[(,])(\d+)_(\d+)", r"Sp\1_\1_\2", newick)
    assert "Sp0_0_" in newick and newick.rstrip().endswith("n0;")
    fn.write_text("%s: %s" % (og, newick))
    assert expand_polytomies.main([str(tmp_path)]) == 0
    assert read_output(tmp_path) == expected


def test_no_polytomies(tmp_path, capsys):
    write_results(tmp_path, {3: "((0_1,1_2),(2_3,3_4));"})
    assert expand_polytomies.main([str(tmp_path)]) == 0
    assert "No polytomies" in capsys.readouterr().out
    assert not (tmp_path / "Expanded_Orthologs").exists()


def test_files_of_another_run(tmp_path, capsys):
    write_results(tmp_path, {3: tpo.star(30, 1)})
    fn = tmp_path / "Orthologs" / "Sp0.tsv"
    lines = fn.read_text().splitlines()
    fn.write_text("\n".join(lines[:-1]) + "\n")             # (a row missing)
    assert expand_polytomies.main([str(tmp_path)]) == 1
    assert "not of the same run" in capsys.readouterr().err


def test_names_in_several_children(tmp_path, capsys):
    # (as -X can give: a gene name in two children, of two species): an error, not a guess
    write_results(tmp_path, {3: "(0_1,1_2,2_3,3_4);"})
    fn = tmp_path / "Resolved_Gene_Trees" / "Resolved_Gene_Trees.txt"
    fn.write_text(fn.read_text().replace("1_2", "0_1"))
    rows = tmp_path / "Orthologs" / "Sp1.tsv"
    rows.write_text(rows.read_text().replace("1_2", "0_1"))
    assert expand_polytomies.main([str(tmp_path)]) == 1
    assert "named the same" in capsys.readouterr().err


def test_stale_polytomies_removed(tmp_path):
    (tmp_path / tree_analyser.POLYTOMIES_FN).write_text("an earlier run\n")
    tree_analyser.write_polytomies([], str(tmp_path))
    assert not (tmp_path / tree_analyser.POLYTOMIES_FN).exists()
