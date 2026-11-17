"""Memory of large gene trees and orthogroups: the same results, held more compactly."""
import os
import random
import threading
import time

import pytest

from orthofinder.hog_orthogroups import update_trees
from orthofinder.orthology import hog_writer, rooting
from orthofinder.phylo import tree
from orthofinder.utils import parallel_task_manager


def _random_tree(rng, n_leaves, n_species):
    nodes = ["%d_%d" % (rng.randrange(n_species), i) for i in range(n_leaves)]
    while len(nodes) > 1:
        k = min(len(nodes), rng.choice([2, 2, 2, 3, 6]))       # binary nodes and polytomies
        rng.shuffle(nodes)
        nodes = ["(%s)" % ",".join(nodes[:k])] + nodes[k:]
    return tree.Tree(nodes[0] + ";")


@pytest.mark.parametrize("seed", range(20))
def test_species_masks_are_the_species_sets(seed):
    rng = random.Random(seed)
    t = _random_tree(rng, rng.randint(3, 80), rng.randint(1, 9))
    gene_map = lambda g: g.split("_")[0]
    species = {gene_map(g) for g in t.get_leaf_names()}
    bit_of = {sp: 1 << i for i, sp in enumerate(sorted(species))}
    rooting.store_species_sets(t, gene_map)
    rooting.store_species_masks(t, gene_map, bit_of)
    for n in t.traverse():
        assert n.spm_down == rooting.species_mask(n.sp_down, bit_of)
        assert n.spm_up == rooting.species_mask(n.sp_up, bit_of)


def test_species_sets_without_up():
    t = tree.Tree("((0_1,1_2),(2_3,(0_4,3_5)));")
    rooting.store_species_sets(t, lambda g: g.split("_")[0], q_up=False)
    assert t.sp_down == {"0", "1", "2", "3"}
    assert not any(hasattr(n, "sp_up") for n in t.traverse())


def test_separation_score_as_with_sets():
    rng = random.Random(3)
    species = [str(i) for i in range(12)]
    bit_of = {sp: 1 << i for i, sp in enumerate(species)}
    for _ in range(200):
        up, down, t1 = (set(rng.sample(species, rng.randint(0, 12))) for _ in range(3))
        t2 = set(species) - t1
        n1, n2 = float(len(t1) or 1), float(len(t2) or 1)
        n_recip = 1. / (n1 * n1 * n2 * n2)
        # as it was, with sets
        f_dup = len(up & t1) * len(up & t2) * len(down & t1) * len(down & t2) * n_recip
        f_a = len(up & t1) * (n2 - len(up & t2)) * (n1 - len(down & t1)) * len(down & t2) * n_recip
        f_b = (n1 - len(up & t1)) * len(up & t2) * len(down & t1) * (n2 - len(down & t2)) * n_recip
        m = rooting.species_mask
        assert rooting.outgroup_ingroup_separation_score(
            m(up, bit_of), m(down, bit_of), m(t1, bit_of), m(t2, bit_of), n_recip, n1, n2) == max(f_dup, f_a, f_b)


def test_sparse_hog_rows_written_as_dense():
    w = hog_writer.HogWriter.__new__(hog_writer.HogWriter)
    w.i_sps = ["0", "1", "2", "3"]
    row = hog_writer.SparseRow(("OG0000001", "n3", ((1, "b1, b2"), (3, "d"))))
    assert w.dense_row(row) == ["OG0000001", "n3", "", "b1, b2", "", "d"]
    assert w.dense_row(["OG", "n0", "a", ""]) == ["OG", "n0", "a", ""]
    import pickle
    assert pickle.loads(pickle.dumps(row)) == row and type(pickle.loads(pickle.dumps(row))) is hog_writer.SparseRow


def test_byte_budget_bounds_the_bytes_in_flight():
    budget = parallel_task_manager.ByteBudget(100)
    stop = threading.Event()
    assert budget.acquire(60, stop)
    got = []
    t = threading.Thread(target=lambda: got.append(budget.acquire(50, stop)))
    t.start()
    time.sleep(0.2)
    assert not got                    # 110 > 100: waits for the first to be released
    budget.release(60)
    t.join(5)
    assert got == [True] and budget.used.value == 50
    budget.release(50)
    assert budget.acquire(1000, stop)  # larger than the limit: alone
    budget.release(1000)
    assert budget.used.value == 0


@pytest.mark.parametrize("limit, fraction", [(1, 0), (update_trees.MAX_MEMORY_IN_FLIGHT, 0),   # one at a time, as before
                                             (1, update_trees.MEMORY_IN_FLIGHT_FRACTION)])      # scaled
def test_hog_trees_and_alignments_with_the_budget(tmp_path, monkeypatch, limit, fraction):
    monkeypatch.setattr(update_trees, "MAX_MEMORY_IN_FLIGHT", limit)
    monkeypatch.setattr(update_trees, "MEMORY_IN_FLIGHT_FRACTION", fraction)
    tree_dir, fa_dir, out_trees, out_msas = (tmp_path / d for d in ("t", "f", "ot", "om"))
    for d in (tree_dir, fa_dir, out_trees, out_msas):
        d.mkdir()
    ogs, hog_index, name_dict = [], {}, {}
    for i in range(12):
        og = "OG%07d" % i
        ogs.append(og)
        (tree_dir / (og + ".txt")).write_text("((0_%d:1,1_%d:2)n1:3,(2_%d:4,0_%d:1)n2:1)n0;" % (i, i, i, i + 100))
        (fa_dir / (og + ".fa")).write_text("".join(">%s\nAC-D%s\n" % (g, "-" * i) for g in
                                                  ("0_%d" % i, "1_%d" % i, "2_%d" % i, "0_%d" % (i + 100))))
        hog_index[og] = [{"HOG": "N0.HOG%07d" % i, "OG": og, "Gene Tree Parent Clade": "n0",
                          "A": "0_%d, 0_%d" % (i, i + 100), "B": "1_%d" % i, "C": "2_%d" % i}]
        name_dict["N0.HOG%07d" % i] = "N0.HOG%07d" % i
    id_dict = {"%d_%d" % (sp, g): "g%d_%d" % (sp, g) for sp in range(3) for g in list(range(12)) + list(range(100, 112))}
    update_trees.post_ogs_processing(
        ogs, str(out_trees), hog_index, name_dict, id_dict, None, ["A", "B", "C"], 4,
        {og: str(tree_dir / (og + ".txt")) for og in ogs}, {og: str(fa_dir / (og + ".fa")) for og in ogs},
        align_dir=str(out_msas), min_seq=2)
    assert sorted(os.listdir(out_trees)) == ["N0.HOG%07d.txt" % i for i in range(12)]
    assert sorted(os.listdir(out_msas)) == ["N0.HOG%07d.fa" % i for i in range(12)]
    # (the gap columns of the orthogroup removed)
    assert (out_msas / "N0.HOG0000005.fa").read_text() == ">g0_5\nACD\n>g0_105\nACD\n>g1_5\nACD\n>g2_5\nACD\n"


def test_estimated_memory_of_an_orthogroup(tmp_path):
    (tmp_path / "t.txt").write_text("x" * 100)
    (tmp_path / "a.fa").write_text("x" * 1000)
    tree_index, fasta_index = {"OG1": str(tmp_path / "t.txt")}, {"OG1": str(tmp_path / "a.fa")}
    # a parsed tree takes many times its newick, an alignment about its FASTA
    assert update_trees.estimated_memory("OG1", tree_index, fasta_index) == \
        100 * update_trees.TREE_MEMORY_FACTOR + 1000 * update_trees.ALIGNMENT_MEMORY_FACTOR
    assert update_trees.estimated_memory("OG1", {"OG1": "(a,b);"}, {}) == 6 * update_trees.TREE_MEMORY_FACTOR
    assert update_trees.estimated_memory("OG2", tree_index, fasta_index) == 0


def test_memory_budget(monkeypatch):
    from orthofinder.utils import system_info
    monkeypatch.setattr(system_info, "_usable_memory", [400 * 1024 ** 3])
    assert system_info.memory_budget(4 * 1024 ** 3, 1 / 4) == 100 * 1024 ** 3
    monkeypatch.setattr(system_info, "_usable_memory", [8 * 1024 ** 3])     # less: the minimum, as before
    assert system_info.memory_budget(4 * 1024 ** 3, 1 / 4) == 4 * 1024 ** 3
    monkeypatch.setattr(system_info, "_usable_memory", [None])               # unknown
    assert system_info.memory_budget(4 * 1024 ** 3, 1 / 4) == 4 * 1024 ** 3


def test_usable_memory_is_the_least(monkeypatch):
    from orthofinder.utils import system_info
    monkeypatch.setattr(system_info, "_usable_memory", [])
    monkeypatch.setattr(system_info, "_memory_mb", lambda: {"total": 512000, "available": 400000})
    monkeypatch.setattr(system_info, "_cgroup_memory_limit_mb", lambda: None)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "64000")                      # a job of 64 GB on the node
    assert system_info.usable_memory_bytes() == 64000 * 1024 ** 2


def test_dropped_columns_residues_by_the_trimming_gaps():
    alignment = {"a": "X?-\n", "b": "-X-\n"}
    assert update_trees.has_residues(alignment)                 # (X and ? residues with - only)
    assert not update_trees.has_residues(alignment, "-?*Xx")    # (none for clipkit's)
