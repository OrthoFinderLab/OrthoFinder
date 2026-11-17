"""STRIDE's supported duplications with the species of each gene tree node as bitmasks (as the sets gave)."""

from orthofinder.phylo import tree
from orthofinder.species_tree import stride


def test_supported_duplications():
    # (the values the set version gave) a duplication before A+B, both copies
    # with A and B, and one in C. The one before C+D lost D in one copy and is
    # not supported
    species, dict_clades, clade_names = stride.analyse_species_tree(tree.Tree("((A,B),(C,D),(E,F));"))
    t = tree.Tree("(((A_1,B_1),(A_2,B_2)),((C_1,C_2),D_1),((C_3,D_2),(E_1,F_1)));")
    genes = set(t.get_leaf_names())
    supported, genes_post_dup = stride.supported_hierachies(
        t, genes, set(map(stride.gene_to_species_dash, genes)), stride.gene_to_species_dash,
        species, dict_clades, clade_names, "t")
    assert dict(supported) == {frozenset("AB"): 1, frozenset("C"): 1}
    assert genes_post_dup == {frozenset(["A_1", "A_2", "B_1", "B_2"])}


def test_clade_index_masks():
    species, dict_clades, _ = stride.analyse_species_tree(tree.Tree("((A,B),(C,D),(E,F));"))
    index = stride.CladeIndex(dict_clades, species)
    ab = index.bit_of["A"] | index.bit_of["B"]
    assert index.smallest_containing(ab) == {frozenset("AB")}
    assert index.species_of_bit[index.bit_of["E"]] == "E"
    assert sorted(index.expected[frozenset("AB")]) == sorted([[index.bit_of["A"]], [index.bit_of["B"]]])


def _supported(tmp_path, newick):
    species, dict_clades, clade_names = stride.analyse_species_tree(tree.Tree("((A,B),(C,D),(E,F));"))
    fn = tmp_path / "t.tre"
    fn.write_text(newick)
    supported, genes_post_dup = stride.supported_hierachies_wrapper(
        str(fn), stride.gene_to_species_dash, species, dict_clades, clade_names)
    return dict(supported), genes_post_dup


def test_node_next_to_a_polytomy_skipped(tmp_path):
    # (the A+B duplication's parent is a polytomy: the node is skipped, the tree used,
    # rather than "Expected binary tree" stopping the run)
    assert _supported(tmp_path, "((((A_1,B_1),(A_2,B_2)),C_1,D_1),(E_1,F_1),((C_2,D_2),(A_3,B_3)));") == ({}, set())


def test_gene_tree_rooted_on_two_as_unrooted(tmp_path):
    unrooted = "(((A_1,B_1),(A_2,B_2)),((C_1,C_2),D_1),((C_3,D_2),(E_1,F_1)));"
    rooted = "((((A_1,B_1),(A_2,B_2)),((C_1,C_2),D_1)),((C_3,D_2),(E_1,F_1)));"
    assert _supported(tmp_path, rooted) == _supported(tmp_path, unrooted) == (
        {frozenset("AB"): 1, frozenset("C"): 1}, {frozenset(["A_1", "A_2", "B_1", "B_2"])})
