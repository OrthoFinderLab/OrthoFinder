"""Polytomies of the gene trees: their orthologs per pair of species sets, counted per pair of children, and Polytomies.tsv."""

import itertools
import random
import time

import numpy as np
import pytest

from orthofinder.orthology import hog_writer, stats, tree_analyser
from orthofinder.orthology.reconciliation import get_species_neighbours
from orthofinder.phylo import tree

N_SPECIES = 6
SPECIES_TREE = "((0,1),((2,3),(4,5)));"


def species_tree():
    t = tree.Tree(SPECIES_TREE)
    t.name = "N0"
    i = 1
    for n in t.traverse():
        if not n.is_leaf() and not n.is_root():
            n.name = "N%d" % i
            i += 1
    return t


def gene_tree(newick):
    return tree.Tree(newick)


def star(n_genes, seed, n_species=N_SPECIES):
    """A polytomy of single genes (as FastTree joins identical sequences)."""
    rng = random.Random(seed)
    return "(%s);" % ",".join("%d_%d" % (rng.randrange(n_species), i) for i in range(n_genes))


def analyse(newick, polytomies=None):
    sp_tree = species_tree()
    return tree_analyser.get_orthologs_from_tree(
        7, gene_tree(newick), sp_tree, tree_analyser.gene_to_species_dash,
        get_species_neighbours(sp_tree), q_get_dups=True, polytomies=polytomies)


def per_pair(orthologs):
    """
    The orthologs with each polytomy's as an entry per pair of its children
    with no species in common (as they were written before they were
    grouped), from the children's genes alone.
    """
    out = []
    for entry in orthologs:
        if type(entry) is not tree_analyser.PolytomyOrthologs:
            out.append(entry)
            continue
        species = [set(normal) | set(suspect) for normal, suspect in entry.children]
        for i, j in itertools.combinations(range(len(entry.children)), 2):
            if not species[i] & species[j]:
                (o0, os0), (o1, os1) = entry.children[i], entry.children[j]
                out.append((o0, o1, os0, os1))
    return out


COUNTS = ("n", "n_121", "n_12m", "n_m21", "n_m2m")


def lines(orthologs, suspect, pending=False, chunk_bytes=None):
    """
    The ortholog and xenologue rows of the orthologs, each species' sorted,
    and the counts (NOrthoSp, through a PendingPairs if pending). With
    chunk_bytes, the rows as flushed in chunks of that size.
    """
    names = {"%d_%d" % (sp, g): "%d_%d" % (sp, g) for sp in range(N_SPECIES) for g in range(2000)}
    olog, sus = [""] * N_SPECIES, [""] * N_SPECIES
    chunks = [[], []]

    def flush(olog_texts, sus_texts):
        for c, texts in zip(chunks, (olog_texts, sus_texts)):
            c.extend(texts.items())

    pair_counts = stats.PendingPairs(N_SPECIES) if pending else None
    counts = tree_analyser.get_lines_for_ortholog_files(
        [(7, orthologs)], {str(i): "Sp%d" % i for i in range(N_SPECIES)},
        list(range(N_SPECIES)), names, bool(suspect), olog, sus, pair_counts=pair_counts,
        flush=flush if chunk_bytes else None, chunk_bytes=chunk_bytes)
    if pending:
        counts = stats.NOrthoSp(N_SPECIES)
        pair_counts.add_to(counts)
    for c, texts in zip(chunks, (olog, sus)):
        for i, text in c:
            texts[i] += text
    if chunk_bytes:
        assert len(chunks[0]) > N_SPECIES            # (in several chunks)
    return [sorted(t.splitlines()) for t in olog], [sorted(t.splitlines()) for t in sus], counts


def gene_pairs(orthologs, suspect, pending=False):
    """
    Every (gene, gene) related by the ortholog and xenologue rows of the
    orthologs, the number of rows, and the counts (NOrthoSp, through a
    PendingPairs if pending).
    """
    return pairs_of(*lines(orthologs, suspect, pending))


def pairs_of(olog, sus, counts):
    pairs, n_rows = set(), 0
    for rows in olog:
        for row in rows:
            _, _, mine, theirs = row.split("\t")
            n_rows += 1
            pairs.update(("o", a, b) for a in mine.split(", ") for b in theirs.split(", "))
    for rows in sus:
        for row in rows:
            _, mine, others = row.split("\t")
            pairs.update(("x", a, b) for a in mine.split(", ") for b in others.split(", "))
    return pairs, n_rows, counts


def assert_as_before(newick, q_star=False):
    """
    The orthologs of a tree's polytomies relate the same genes as an entry
    per pair of children (per_pair), and are counted the same, whole or in
    chunks. q_star: only polytomies write orthologs (no binary speciation
    node), so their records' ortholog pairs are those of the rows. Returns the number of rows of each, and the
    records.
    """
    polytomies = []
    b = analyse(newick, polytomies)
    assert polytomies
    olog_a, sus_a, counts_a = lines(per_pair(b[0]), b[2])
    pairs_a, rows_a, _ = pairs_of(olog_a, sus_a, counts_a)
    for pending in (False, True):
        pairs_b, rows_b, counts_b = gene_pairs(b[0], b[2], pending)
        assert pairs_a == pairs_b
        for m in COUNTS:
            assert np.array_equal(getattr(counts_a, m), getattr(counts_b, m)), m
            assert getattr(counts_b, m).dtype == np.uint64
    assert lines(b[0], b[2], True, chunk_bytes=200)[:2] == lines(b[0], b[2])[:2]
    if q_star:
        assert sum(r[8] for r in polytomies) == sum(p[0] == "o" for p in pairs_a) // 2
    return rows_a, rows_b, polytomies


@pytest.mark.parametrize("seed", range(5))
def test_polytomy_same_gene_pairs_and_counts_fewer_rows(seed):
    rows_a, rows_b, polytomies = assert_as_before(star(120, seed), True)
    assert rows_b <= N_SPECIES * (N_SPECIES - 1) < rows_a
    (r,) = polytomies            # (a star: one polytomy)
    assert r[:5] + r[6:8] == ("OG0000007", "n0", 120, 120, N_SPECIES, N_SPECIES, "speciation")
    leaves = [g.split("_")[0] for g in star(120, seed)[1:-2].split(",")]
    assert r[5] == max(leaves.count(sp) for sp in set(leaves))


@pytest.mark.parametrize("seed", range(5))
def test_polytomy_children_of_several_genes(seed):
    # children of one species with one or more genes (one-to-many etc. counts)
    rng = random.Random(seed)
    children = []
    for i in range(60):
        sp = rng.randrange(N_SPECIES)
        n = rng.choice([1, 1, 2, 3])
        genes = ["%d_%d" % (sp, 10 * i + j) for j in range(n)]
        children.append(genes[0] if n == 1 else "(%s)" % ",".join(genes))
    assert_as_before("(%s);" % ",".join(children), True)


def misplaced(monkeypatch, *genes):
    """resolve_overlap resolving an overlap by these genes, when below it (they are then suspect below)."""
    def resolve(overlap, sp0, sp1, ch, *args, **kwargs):
        removed = {g for c in ch for g in c.get_leaf_names() if g in genes}
        return (True, removed) if removed else (False, set())
    monkeypatch.setattr(tree_analyser, "resolve_overlap", resolve)


SUSPECT_TREES = [
    # a polytomy with a misplaced gene (suspect), below an overlap it resolves
    "((%s,0_900,(1_901,2_902)),(0_903,3_904));" % star(60, 1)[1:-2],
    "((%s,0_900,(1_901,0_902)),(0_903,1_904,4_905));" % star(60, 2)[1:-2],
]


@pytest.mark.parametrize("newick", SUSPECT_TREES)
def test_polytomy_with_suspect_genes(monkeypatch, newick):
    misplaced(monkeypatch, "0_900", "0_902")
    a = analyse(newick)
    assert a[2] and any(len(o[2]) or len(o[3]) for o in per_pair(a[0]))      # (xenologues at the polytomy)
    assert_as_before(newick)


@pytest.mark.parametrize("seed", range(5))
def test_polytomy_children_of_several_species(seed):
    rng = random.Random(seed)
    children = []
    for i in range(50):
        sps = rng.sample(range(N_SPECIES), rng.choice([1, 1, 2, 3]))
        genes = ["%d_%d" % (sp, 10 * i + j) for j, sp in enumerate(sps)]
        children.append(genes[0] if len(genes) == 1 else "(%s)" % ",".join(genes))
    assert_as_before("(%s);" % ",".join(children))


def test_polytomy_children_of_two_species():
    clades = ",".join("(%d_%d,%d_%d)" % (i % 6, 2 * i, (i + 1) % 6, 2 * i + 1) for i in range(30))
    _, _, polytomies = assert_as_before("(%s);" % clades)
    assert [r[:7] for r in polytomies] == [("OG0000007", "n0", 30, 60, 6, 10, 6)]


@pytest.mark.parametrize("seed", range(3))
def test_polytomy_of_many_species_sets(monkeypatch, seed):
    # many species sets, more than 64 species (masks of several words), in small blocks of rows
    monkeypatch.setattr(tree_analyser.DisjointGroups, "BLOCK_BYTES", 1)
    n_species = 130
    rng = random.Random(seed)
    masks = []
    for _ in range(200):
        m = 0
        for sp in rng.sample(range(n_species), rng.choice([1, 2, 3])):
            m |= 1 << sp
        masks.append(m)
    group_masks = list(dict.fromkeys(masks))
    d = tree_analyser.DisjointGroups(group_masks)
    dense = np.array([[0.0 if m0 & m1 else 1.0 for m1 in group_masks] for m0 in group_masks])
    assert np.array_equal(d.rows(0, d.k), dense.astype(bool))
    x = np.array([[rng.randrange(4) for _ in range(5)] for _ in group_masks], dtype=np.float64)
    assert np.array_equal(d.dot(x), dense @ x)
    assert all(list(d.later(a)) == [int(b) for b in np.flatnonzero(dense[a, a + 1:]) + a + 1]
               for a in range(d.k))
    one = tree_analyser.DisjointGroups([1 << sp for sp in range(100)])     # (one species each)
    assert one.words is None and np.array_equal(one.dot(x[:100]), (1 - np.eye(100)) @ x[:100])
    assert np.array_equal(one.rows(3, 7), (1 - np.eye(100))[3:7].astype(bool))


def test_polytomy_of_different_species_sets_rows_unchanged():
    # every child of a different species set: its rows are those per pair of children
    newick = "(0_1,1_2,(2_3,3_4),4_5);"
    b = analyse(newick)
    assert lines(b[0], b[2]) [:2] == lines(per_pair(b[0]), b[2])[:2]


@pytest.mark.parametrize("newick, dup, event", [
    ("(0_1,0_2,0_3,0_4);", True, "single species"),
    ("((0_1,1_1),(0_2,1_2),(0_3,1_3));", True, "duplication"),   # every pair of children shares a species
    ("(0_1,1_2,2_3,0_4);", False, "speciation"),
])
def test_polytomy_duplications(newick, dup, event):
    polytomies = []
    b = analyse(newick, polytomies)
    assert bool(b[3]) == dup
    assert [r[7] for r in polytomies] == [event]
    assert polytomies[0][8] == (0 if dup else 5)       # (0_1 and 0_4 with 1_2 and 2_3, and 1_2 with 2_3)


def test_polytomy_records_of_a_tree():
    # one record per polytomy, below or above another, none for binary nodes
    newick = "(((0_1,0_2,0_3),1_4,1_5),(2_6,3_7),(4_8,(5_9,5_10)));"
    polytomies = []
    analyse(newick, polytomies)
    assert [r[2] for r in polytomies] == [3, 3, 3]
    events = sorted(r[7] for r in polytomies)
    assert events == ["single species", "speciation", "speciation"]
    root = next(r for r in polytomies if r[3] == 10)
    assert root[4:7] == (6, 3, 3)
    inner = next(r for r in polytomies if r[3] == 5)
    assert inner[4:9] == (2, 3, 2, "speciation", 6)
    single = next(r for r in polytomies if r[7] == "single species")
    assert single[3:9] == (3, 1, 3, 1, "single species", 0)


def test_write_polytomies(tmp_path, capsys):
    from orthofinder.utils import logging as run_logging
    checkpoint = tmp_path / "checkpoint.txt"
    logger = run_logging.Logger(checkpoint, fmt="%(message)s")
    run_logging.RunLogger.set_logger(logger)
    records = [("OG0000009", "n10", 3, 3, 3, 1, 3, "speciation", 3),
               ("OG0000002", "n0", 51648, 51648, 1222, 300, 1222, "speciation", 10 ** 9),
               ("OG0000009", "n9", 4, 8, 1, 8, 1, "single species", 0),
               ("OG0000009", "n2", 3, 6, 2, 3, 2, "duplication", 0)]
    try:
        tree_analyser.write_polytomies(records, str(tmp_path))
        tree_analyser.write_polytomies([], str(tmp_path / "none"))
    finally:
        run_logging.RunLogger.set_logger(None)
        logger.close()
    fn = tmp_path / tree_analyser.POLYTOMIES_FN
    rows = [line.split("\t") for line in fn.read_text().splitlines()]
    assert rows[0] == list(tree_analyser.POLYTOMIES_HEADER)
    assert [r[:2] for r in rows[1:]] == [["OG0000002", "n0"], ["OG0000009", "n2"], ["OG0000009", "n9"],
                                         ["OG0000009", "n10"]]
    assert rows[1] == ["OG0000002", "n0", "51648", "51648", "1222", "300", "1222", "speciation", str(10 ** 9)]
    (line,) = checkpoint.read_text().splitlines()
    assert "WARNING" not in line and tree_analyser.POLYTOMY_STAGE in line
    assert "4 polytomies in 2 orthogroups (2 speciation, 1 duplication, 1 single species)" in line
    assert "largest OG0000002 node n0, 51648 children, 51648 genes, 1222 species" in line
    assert capsys.readouterr().out == ""               # (nothing on screen)
    assert not (tmp_path / "none").exists()


def test_children_all_share_species():
    assert tree_analyser.children_all_share_species([0b011, 0b110, 0b010])
    assert tree_analyser.children_all_share_species([0b011, 0b110, 0b101])     # no species in all
    assert not tree_analyser.children_all_share_species([0b001, 0b010, 0b001])


def test_evidenced_dup_level_by_distinct_levels(capsys):
    """get_evidenced_dup_level: the same as comparing every pair of children (as it did)."""
    w = hog_writer.HogWriter.__new__(hog_writer.HogWriter)    # (only the species tree's numbering is needed)
    st = species_tree()
    w.tin, w.tout = {}, {}
    clock = [0]

    def visit(n):
        w.tin[n.name] = clock[0]
        clock[0] += 1
        for c in n.get_children():
            visit(c)
        w.tout[n.name] = clock[0]
    visit(st)

    def every_pair(mrcas):
        attested = set()
        for l1, l2 in itertools.combinations(mrcas, 2):
            if l1 == l2:
                attested.add(l1)
            elif w._is_below(l2, l1):
                attested.add(l2)
            elif w._is_below(l1, l2):
                attested.add(l1)
        if len(attested) == 1:
            return attested.pop()
        for top in sorted(attested, key=lambda a: w.tin[a]):
            if all(w._is_below(other, top) for other in attested if other != top):
                return top
        return None

    levels = [n.name for n in st.traverse()]
    rng = random.Random(4)
    for _ in range(500):
        mrcas = [rng.choice(levels) for _ in range(rng.randint(2, 8))]
        assert w.get_evidenced_dup_level(list(mrcas)) == every_pair(mrcas), mrcas
    capsys.readouterr()      # (its warnings for the unexpected topologies)


def _store_species_sets_every_sibling(t, gene_map, tag="sp_"):
    """store_species_sets as it was: for each child, the union of all the other children."""
    tag_up, tag_down = tag + "up", tag + "down"
    for node in t.traverse("postorder"):
        if node.is_leaf():
            node_type = gene_map(node.name)
            setattr(node, tag_down, set() if node_type is None else {node_type})
        elif not node.is_root():
            setattr(node, tag_down, set.union(*[getattr(ch, tag_down) for ch in node.get_children()]))
    for node in t.traverse("preorder"):
        if node.is_root():
            setattr(node, tag_up, set())
        else:
            others = [ch for ch in node.up.get_children() if ch != node]
            sp_downs = set.union(*[getattr(o, tag_down) for o in others])
            setattr(node, tag_up, sp_downs if node.up.is_root() else getattr(node.up, tag_up).union(sp_downs))
    setattr(t, tag_down, set.union(*[getattr(ch, tag_down) for ch in t.get_children()]))


def _random_tree(rng, n_leaves, n_species):
    nodes = ["%d_%d" % (rng.randrange(n_species), i) for i in range(n_leaves)]
    while len(nodes) > 1:
        k = min(len(nodes), rng.choice([2, 2, 3, 5, 12]))       # binary nodes and polytomies
        rng.shuffle(nodes)
        nodes = ["(%s)" % ",".join(nodes[:k])] + nodes[k:]
    return tree.Tree(nodes[0] + ";")


@pytest.mark.parametrize("seed", range(20))
def test_store_species_sets_as_before(seed):
    from orthofinder.orthology import rooting
    rng = random.Random(seed)
    newick = _random_tree(rng, rng.randint(3, 80), rng.randint(1, 8)).write()
    # (with None for a gene of no species, as the ingroup/outgroup tagging does)
    gene_map = (lambda g: None if g.endswith("7") else g.split("_")[0]) if seed % 2 else (lambda g: g.split("_")[0])
    a, b = tree.Tree(newick), tree.Tree(newick)
    rooting.store_species_sets(a, gene_map)
    _store_species_sets_every_sibling(b, gene_map)
    for na, nb in zip(a.traverse("preorder"), b.traverse("preorder")):
        assert getattr(na, "sp_down", None) == getattr(nb, "sp_down", None)
        assert na.sp_up == nb.sp_up
    assert a.sp_down == b.sp_down


def test_children_all_share_species_of_many_children():
    # a polytomy of 60,000 children of three species sets that pairwise share a species,
    # none in all of them: the pairs of species sets checked, not of children
    masks = [0b011, 0b110, 0b101] * 20000
    start = time.monotonic()
    assert tree_analyser.children_all_share_species(masks)
    assert not tree_analyser.children_all_share_species(masks + [0b1000])
    assert time.monotonic() - start < 1


@pytest.mark.parametrize("newick", [star(120, 1), SUSPECT_TREES[0], "((0_1,1_2),(0_3,(1_4,2_5)),(3_6,4_7));"])
def test_hog_pass_without_orthologs(monkeypatch, newick):
    # (q_orthologs=False, the HOG pass: the same tree features, suspect genes and duplications)
    misplaced(monkeypatch, "0_900")
    sp_tree = species_tree()
    results = [tree_analyser.get_orthologs_from_tree(
        7, gene_tree(newick), sp_tree, tree_analyser.gene_to_species_dash,
        get_species_neighbours(sp_tree), q_get_dups=True, q_orthologs=q) for q in (True, False)]
    (o_a, t_a, s_a, d_a), (o_b, t_b, s_b, d_b) = results
    assert o_a and not o_b and s_a == s_b and d_a == d_b
    assert [(n.name, getattr(n, "dup", None), getattr(n, "sp_node", None)) for n in t_a.traverse()] == \
        [(n.name, getattr(n, "dup", None), getattr(n, "sp_node", None)) for n in t_b.traverse()]


def test_species_neighbours_as_the_dictionaries():
    # (an array of species x species, as the dictionaries of dictionaries it replaced gave)
    import random as random_module
    from orthofinder.phylo import tree as tree_module

    def dictionaries(t):
        species = t.get_leaf_names()
        levels = {s: [] for s in species}
        for n in t.traverse('postorder'):
            if n.is_leaf():
                continue
            leaf_sets = [set(ch.get_leaf_names()) for ch in n.get_children()]
            for i, own in enumerate(leaf_sets):
                others = set().union(*[x for j, x in enumerate(leaf_sets) if j != i])
                for leaf in own:
                    levels[leaf].append(others)
        return {sp: {o: d for d, others in enumerate(lev) for o in others} for sp, lev in levels.items()}

    rng = random_module.Random(1)

    def newick(names):
        if len(names) == 1:
            return names[0]
        k = rng.choice([2, 2, 3]) if len(names) >= 3 else 2
        cuts = sorted(rng.sample(range(1, len(names)), k - 1))
        return "(" + ",".join(newick(names[a:b]) for a, b in zip([0] + cuts, cuts + [len(names)])) + ")"

    for _ in range(50):
        names = [str(i) for i in range(rng.randint(2, 30))]
        rng.shuffle(names)
        t = tree_module.Tree(newick(names) + ";")
        expected, got = dictionaries(t), get_species_neighbours(t)
        assert all(got[a][b] == d for a, row in expected.items() for b, d in row.items())
