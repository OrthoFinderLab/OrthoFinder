"""Rooting gene trees: the root that best separates the species, as in the species tree."""

import operator
from collections import Counter
import os

from ..phylo import tree as tree_lib


class RootMap(object):
    def __init__(self, set_a, set_b, GeneToSpecies):
        self.set_a = set_a
        self.set_b = set_b
        self.GeneToSpecies = GeneToSpecies
        
    def gene_map(self, gene_name):
        sp = self.GeneToSpecies(gene_name)
        if sp in self.set_a: return True 
        elif sp in self.set_b: return False
        else: return None


# Why check_and_root_tree gave no tree
MISSING = "missing or empty"
ONE_GENE = "one gene"
NOT_ROOTED = "not rooted"


def check_and_root_tree(tree_fn, species_tree_rooted, GeneToSpecies, why=None):
    """
    Check that the tree can be analysed and rooted
    Root tree
    Returns None if this fails, i.e. checks: exists, has more than one gene, can be rooted
    why: a list, to which the reason is added if it fails (MISSING, ONE_GENE or NOT_ROOTED)
    """
    if (not os.path.exists(tree_fn)) or os.stat(tree_fn).st_size == 0:
        if why is not None:
            why.append(MISSING)
        return None, False
    q_have_support = False
    try:
        tree = tree_lib.Tree(tree_fn, format=2)
        q_have_support = True
    except:
        try:
            tree = tree_lib.Tree(tree_fn)
        except:
            tree = tree_lib.Tree(tree_fn, format=3)
    if len(tree) == 1: 
        if why is not None:
            why.append(ONE_GENE)
        return None, False
    root = get_root(tree, species_tree_rooted, GeneToSpecies)
    if root == None: 
        if why is not None:
            why.append(NOT_ROOTED)
        return None, False
    # Pick the first root for now
    if root != tree:
        tree.set_outgroup(root)
    return tree, q_have_support


def get_root(tree, species_tree_rooted, GeneToSpecies):
        roots = get_roots(tree, species_tree_rooted, GeneToSpecies)
        if len(roots) > 0:
            root_dists = [r.get_closest_leaf()[1] for r in roots]
            i, _ = max(enumerate(root_dists), key=operator.itemgetter(1))
            return roots[i]
        else:
            return None # single species tree


def get_roots(tree, species_tree_rooted, GeneToSpecies):
    """
    Allow non-binary gene or species trees.
    (A,B,C) => consider splits A|BC, B|AC, C|AB - this applies to gene and species tree
    If a clean ingroup/outgroup split cannot be found then score root by geometric mean of fraction of expected species actually 
    observed for the two splits
    """
    species_observed = set([GeneToSpecies(g) for g in tree.get_leaf_names()])
    if len(species_observed) == 1:
        return [next(n for n in tree)] # arbitrary root if all genes are from the same species
    
    # use species tree to find correct outgroup according to what species are present in the gene tree
    n = species_tree_rooted
    children = n.get_children()
    leaves = [set(ch.get_leaf_names()) for ch in children]
    have = [len(l.intersection(species_observed)) != 0 for l in leaves]
    while sum(have) < 2:
        n = children[have.index(True)]
        children = n.get_children()
        leaves = [set(ch.get_leaf_names()) for ch in children]
        have = [len(l.intersection(species_observed)) != 0 for l in leaves]

    # Get splits to look for
    roots_list = []
    scores_list = []   # the fraction completeness of the two clades
#    roots_set = set()
    # Species below/above each node: the same for every split, so computed once
    # (the tree is not modified in this loop). As bitmasks, not sets: a set of
    # the species above a node holds nearly all of them at every node (GBs for
    # a tree of tens of thousands of genes of a thousand species).
    bit_of = {sp: 1 << i for i, sp in enumerate(sorted(species_observed, key=str))}
    store_species_masks(tree, GeneToSpecies, bit_of)
    for i in range(len(leaves)):
        t1 = leaves[i]
        t2 = set.union(*[l for j,l in enumerate(leaves) if j!=i])
        # G - set of species in gene tree
        # First relevant split in species tree is (A,B), such that A \cap G \neq \emptyset and A \cap G \neq \emptyset
        # label all nodes in gene tree according the whether subsets of A, B or both lie below node
        root_mapper = RootMap(t1, t2, GeneToSpecies)    
        mask1 = species_mask(t1, bit_of)
        mask2 = species_mask(t2, bit_of)
        nt1 = float(len(t1))
        nt2 = float(len(t2))
        n_recip = 1./(nt1*nt1*nt2*nt2)
        gene_map = root_mapper.gene_map
        store_species_sets(tree, gene_map, "inout_") # ingroup/outgroup identification
        # find all possible locations in the gene tree at which the root should be

        T = {True,}
        F = {False,}
        TF = set([True, False])
        for m in tree.traverse('postorder'):
            if m.is_leaf(): 
                if len(m.inout_up) == 1 and m.inout_up != m.inout_down:
                    # this is the unique root
                    return [m]
            else:
                if len(m.inout_up) == 1 and len(m.inout_down) == 1 and m.inout_up != m.inout_down:
                    # this is the unique root
                    return [m]
                nodes = m.get_children() if m.is_root() else [m] + m.get_children()
                clades = [ch.inout_down for ch in nodes] if m.is_root() else ([m.inout_up] + [ch.inout_down for ch in m.get_children()])
                # do we have the situation A | B or (A,B),S?
                if len(nodes) == 3:
                    if all([len(c) == 1 for c in clades]) and T in clades and F in clades:
                        # unique root
                        if clades.count(T) == 1:
                            return [nodes[clades.index(T)]]
                        else:
                            return [nodes[clades.index(F)]]
                    elif T in clades and F in clades:
                        #AB-(A,B) or B-(AB,A)
                        ab = [c == TF for c in clades]
                        i = ab.index(True)
                        roots_list.append(nodes[i])
                        scores_list.append(outgroup_ingroup_separation_score(
                            nodes[i].spm_up, nodes[i].spm_down, mask1, mask2, n_recip, nt1, nt2))
                    elif clades.count(TF) >= 2:  
                        # (A,A,A)-excluded, (A,A,AB)-ignore as want A to be biggest without including B, (A,AB,AB), (AB,AB,AB) 
                        i = 0
                        roots_list.append(nodes[i])
                        scores_list.append(outgroup_ingroup_separation_score(
                            nodes[i].spm_up, nodes[i].spm_down, mask1, mask2, n_recip, nt1, nt2))
                elif T in clades and F in clades:
                    roots_list.append(m)
                    scores_list.append(0)  # last choice
    # If we haven't found a unique root then use the scores for completeness of ingroup/outgroup to root
    if len(roots_list) == 0: 
        return [] # This shouldn't occur
    return [sorted(zip(scores_list, roots_list), key=lambda x: x[0], reverse=True)[0][1]]


def species_mask(species, bit_of):
    """The bitmask of species (bit_of: species -> its bit; those without one, none)."""
    mask = 0
    for sp in species:
        mask |= bit_of.get(sp, 0)
    return mask


def store_species_masks(t, gene_map, bit_of):
    """
    store_species_sets's sp_down and sp_up, as bitmasks of the species
    (bit_of: species -> its bit): spm_down and spm_up. A bitmask of a thousand
    species is ~160 bytes, a set of them ~30 KB.
    """
    for node in t.traverse('postorder'):
        if node.is_leaf():
            node.add_feature("spm_down", bit_of.get(gene_map(node.name), 0))
        elif not node.is_root():
            down = 0
            for ch in node.get_children():
                down |= ch.spm_down
            node.add_feature("spm_down", down)
    # (as store_species_sets: a pass over a polytomy's children, those in only
    # one child found with the species in two or more, twice)
    t.add_feature("spm_up", 0)
    for node in t.traverse('preorder'):
        ch = node.get_children()
        if not ch:
            continue
        up = node.spm_up
        if len(ch) == 2:
            ch[0].add_feature("spm_up", up | ch[1].spm_down)
            ch[1].add_feature("spm_up", up | ch[0].spm_down)
            continue
        once = twice = 0
        for c in ch:
            twice |= once & c.spm_down
            once |= c.spm_down
        all_up = up | once
        for c in ch:
            c.add_feature("spm_up", all_up & ~(c.spm_down & ~twice & ~up))
    down = 0
    for ch in t.get_children():
        down |= ch.spm_down
    t.add_feature("spm_down", down)


def store_species_sets(t, gene_map, tag="sp_", q_up=True):
    """
    The species below each node (tag + "down") and, if q_up, those outside
    its clade (tag + "up"), as sets (see store_species_masks for bitmasks).
    """
    if t is None:
        raise ValueError("StoreSpeciesSets got t=None (tree is None)")
    tag_up = tag + "up"
    tag_down = tag + "down"  
    for node in t.traverse('postorder'):
        if node.is_leaf():
            node_type = gene_map(node.name)
            node.add_feature(tag_down, set() if node_type is None else {node_type})
        elif node.is_root():
            continue
        else:
            node.add_feature(tag_down, set.union(*[ch.__getattribute__(tag_down) for ch in node.get_children()]))
    # The species of a node's tree outside its clade (tag_up): its parent's,
    # and those below its siblings. At a polytomy, a pass over the children
    # (not a union of every other child's for each child: quadratic, hours for
    # tens of thousands of identical sequences, FastTree's polytomies), and
    # the children with no species of their own share one set (the sets are
    # replaced, never changed in place).
    if not q_up:
        t.add_feature(tag_down, set.union(*[ch.__getattribute__(tag_down) for ch in t.get_children()]))
        return
    t.add_feature(tag_up, set())
    for node in t.traverse('preorder'):
        ch = node.get_children()
        if not ch:
            continue
        up = node.__getattribute__(tag_up)
        downs = [c.__getattribute__(tag_down) for c in ch]
        if len(ch) == 2:
            ch[0].add_feature(tag_up, up.union(downs[1]))
            ch[1].add_feature(tag_up, up.union(downs[0]))
            continue
        n_children_with = Counter()
        for down in downs:
            n_children_with.update(down)
        all_up = up.union(n_children_with)
        for c, down in zip(ch, downs):
            only_here = {sp for sp in down if n_children_with[sp] == 1 and sp not in up}
            c.add_feature(tag_up, all_up - only_here if only_here else all_up)
    t.add_feature(tag_down, set.union(*[ch.__getattribute__(tag_down) for ch in t.get_children()]))


def outgroup_ingroup_separation_score(up, down, mask1, mask2, n_recip, n1, n2):
    """(the species above and below the node, and of the two sides of the split, as bitmasks)"""
    up1, up2 = (up & mask1).bit_count(), (up & mask2).bit_count()
    down1, down2 = (down & mask1).bit_count(), (down & mask2).bit_count()
    f_dup = up1 * up2 * down1 * down2 * n_recip
    f_a = up1 * (n2 - up2) * (n1 - down1) * down2 * n_recip
    f_b = (n1 - up1) * up2 * down1 * (n2 - down2) * n_recip
    choice = (f_dup, f_a, f_b)
    return max(choice)
