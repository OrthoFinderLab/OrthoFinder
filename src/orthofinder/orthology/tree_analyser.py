# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:11:11 2017

@author: david

Perform directed 'reconciliation' first and then apply EggNOG method

1 - root gene trees on outgroup: unique one this time
2 - infer orthologs
"""
import os
import sys
import itertools
from collections import defaultdict 

import numpy as np


try:
    from rich import print
except ImportError:
    ...
from ..utils import file_handler, file_io, parallel_task_manager
from ..utils import printing
from ..utils import run_report
from ..phylo import taxa as phylo_taxa
from . import stats
from . import rooting
from .rooting import check_and_root_tree
from .resolve import resolve_tree, resolve_polytomies, RESOLVED


debug = False   # HOGs
# An unexpected error analysing a gene tree: in the run report and the run goes on
# without it, or (the tests: ORTHOFINDER_RAISE_INTERNAL_ERRORS=1) the run fails
RAISE_INTERNAL_ERRORS = os.getenv("ORTHOFINDER_RAISE_INTERNAL_ERRORS") == "1"


def gene_to_species_dash(g):
  return g.split("_", 1)[0]
  
ortho_finder_ids = gene_to_species_dash


def species_and_gene_dash(g):
  return g.split("_", 1)

  
species_and_gene_lookup = {
    gene_to_species_dash:species_and_gene_dash, 
}


class TreeAnalyser(object):
    def __init__(
            self, 
            n_ogs, 
            d_results_orthologs, 
            recon_trees_renamed_dir, 
            species_tree_rooted_labelled, 
            species_to_use, 
            GeneToSpecies, 
            sequence_dict, 
            species_dict, 
            spec_seq_dict, 
            neighbours, 
            q_no_recon, 
            dups_file_handle, 
            stride_dups, 
            orthologs_files_handles, 
            putative_xenolog_file_handles, 
            hog_writer, 
            q_split_paralogous_clades, 
            exist_msa=True,
            write_hog_tree=True,
            other_names=None,
    ):
        """other_names: {sequence ID: name} of the "Other" genes of the putative xenologues
        (gene_ids.prefixes_if_shared), None: sequence_dict"""
        self.n_ogs = n_ogs
        self.d_results_orthologs = d_results_orthologs
        self.recon_trees_renamed_dir = recon_trees_renamed_dir
        self.species_tree_rooted_labelled = species_tree_rooted_labelled
        self.species_to_use = species_to_use
        self.nspecies = len(self.species_to_use)
        self.GeneToSpecies = GeneToSpecies
        self.sequence_dict = sequence_dict
        self.species_dict = species_dict
        self.spec_seq_dict = spec_seq_dict
        self.spec_seq_dict[">%d genes" % (5*self.nspecies)] = ">%d genes" % (5*self.nspecies)   # used in Duplications
        self.neighbours = neighbours
        self.q_no_recon = q_no_recon
        self.dups_file_handle = dups_file_handle
        self.stride_dups = stride_dups
        self.orthologs_files_handles = orthologs_files_handles
        self.putative_xenolog_file_handles = putative_xenolog_file_handles
        self.hog_writer = hog_writer
        self.q_split_paralogous_clades = q_split_paralogous_clades
        self.exist_msa = exist_msa
        self.write_hog_tree = write_hog_tree
        self.other_names = other_names
        # The polytomies of the trees analysed (polytomy_record), in the pass that
        # writes the orthologs (not write_hog_tree): this process's, those of the
        # parallel workers added by the parent (write_polytomies)
        self.polytomies = []
        # The second pass: the polytomies the first resolved, ([(genes, children,
        # species sets of children)], {gene: index}), set_hog_pass_resolved
        self.hog_pass_resolved = None


    def analyse_tree(self, iog, n_ortho_acc=None, emit_chunk=None):
        """
        Analyse one gene tree.

        It returns a payload to the parent process.

        If nOrtho_acc (a stats.NOrthoSp or StreamedCounts) is given, this
        tree's ortholog counts are added to it once the whole tree has been
        analysed, and the payload's "n_orthologs" is None. Otherwise it holds a
        dense NOrthoSp for this tree, but in the HOG pass (write_hog_tree),
        whose counts are not used, None.

        emit_chunk(iog, olog_texts, sus_texts) (with n_ortho_acc): the rows of a
        large tree are sent to it in sorted chunks as they are made
        (get_lines_for_ortholog_files: flush), not in the payload. The
        payload's "chunked" is then True, or False if the tree failed after
        chunks were sent (they are to be discarded); None if none were.
        """
        og_name = "OG%07d" % iog
        n_species = len(self.species_to_use)
        chunks_sent = [False]

        def flush(olog_texts, sus_texts):
            chunks_sent[0] = True
            emit_chunk(iog, olog_texts, sus_texts)

        try:
            if self.write_hog_tree:
                tree_file = file_handler.handler.get_ogs_tree_fn(iog)
            else:
                tree_file = file_handler.handler.get_resolved_tree_id_dir() + "OG%07d.txt" % iog

            why = []
            rooted_tree_ids, q_have_support = check_and_root_tree(
                tree_file,
                self.species_tree_rooted_labelled,
                self.GeneToSpecies,
                why=why,
            )

            if rooted_tree_ids is None:
                self.report_skipped(og_name, tree_file, why[0] if why else rooting.NOT_ROOTED)
                return None

            polytomies = None if self.write_hog_tree else []
            # (the first pass: those it resolves, recorded in the second)
            resolved = [] if self.write_hog_tree else None

            orthologs, recon_tree, suspect_genes, dups = get_orthologs_from_tree(
                iog,
                rooted_tree_ids,
                self.species_tree_rooted_labelled,
                self.GeneToSpecies,
                self.neighbours,
                q_get_dups=True,
                q_no_recon=self.q_no_recon,
                polytomies=polytomies,
                q_orthologs=not self.write_hog_tree,     # (the HOG pass: not used)
                resolved=resolved,
                hog_pass_resolved=self.hog_pass_resolved,
            )

            ortholog_lines = empty_ortholog_lines(self.nspecies)
            ortholog_sus_lines = ["" for _ in range(self.nspecies)]

            pair_counts = stats.PendingPairs(n_species) if n_ortho_acc is not None else None
            q_flush = emit_chunk is not None and n_ortho_acc is not None
            if self.write_hog_tree:
                # (the HOG pass: its ortholog rows and counts are not used, those of the second pass are)
                n_orthologs_sp_pair = None
            else:
                n_orthologs_sp_pair = get_lines_for_ortholog_files(
                    [(iog, orthologs)],
                    self.species_dict,
                    self.species_to_use,
                    self.sequence_dict,
                    len(suspect_genes) > 0,
                    ortholog_lines,
                    ortholog_sus_lines,
                    pair_counts=pair_counts,
                    other_names=self.other_names,
                    flush=flush if q_flush else None,
                )
            del orthologs

            cached_hogs = get_hogs_from_tree(
                iog,
                recon_tree,
                self.hog_writer,
                self.q_split_paralogous_clades,
                q_report=not self.write_hog_tree,
            )

            self.write_resolved_tree(recon_tree, iog)

            if n_ortho_acc is not None:
                # Only count trees that were analysed completely (the last step that can fail is above).
                pair_counts.add_to(n_ortho_acc)
            if polytomies:
                self.polytomies.extend(polytomies)
            if resolved:
                self.polytomies.extend(resolved)

            return {
                "iog": iog,
                "n_orthologs": n_orthologs_sp_pair,
                "olog_lines": ortholog_lines,
                "olog_sus_lines": ortholog_sus_lines,
                "duplications": dups,
                "suspect_genes": suspect_genes,
                "cached_hogs": cached_hogs,
                "chunked": True if chunks_sent[0] else None,
            }

        except parallel_task_manager.WorkerError:
            raise       # the run cannot write its results (write_resolved_tree)
        except Exception as e:
            # (the run goes on without this tree, in the run report; in the tests, an error)
            if RAISE_INTERNAL_ERRORS:
                raise
            if isinstance(e, HogAnalysisError):
                run_report.add("hog.failed", og_name, detail=str(e))
            else:
                run_report.add("tree.error" if self.write_hog_tree else "olog.error", og_name,
                               detail="%s: %s" % (type(e).__name__, e))

            return {
                "iog": iog,
                "n_orthologs": (
                    stats.NOrthoSp(n_species) if n_ortho_acc is None and not self.write_hog_tree else None
                ),
                "olog_lines": empty_ortholog_lines(self.nspecies),
                "olog_sus_lines": ["" for _ in range(self.nspecies)],
                "duplications": [],
                "suspect_genes": set(),
                "cached_hogs": [],
                "chunked": False if chunks_sent[0] else None,
            }

    def report_skipped(self, og_name, tree_file, why):
        """A tree not analysed (why, as check_and_root_tree), in the run report."""
        if self.write_hog_tree:
            check = {rooting.MISSING: "tree.missing", rooting.ONE_GENE: "tree.one_gene"}.get(why, "tree.not_rooted")
        else:
            check = "olog.tree_missing" if why == rooting.MISSING else "olog.error"
        run_report.add(check, og_name, detail="%s (%s)" % (why, os.path.basename(tree_file)))

    def set_hog_pass_resolved(self, resolved):
        """
        The polytomies the first pass (the HOGs') resolved, (genes, children,
        species sets of children) each: the second pass reads its trees, so
        has none to resolve, and records these (by their genes, as the trees'
        orthogroups are those of the HOGs).
        """
        index = {gene: k for k, record in enumerate(resolved) for gene in record[0]}
        self.hog_pass_resolved = (resolved, index) if resolved else None

    def write_resolved_tree(self, recon_tree, iog):
        """
        The resolved gene tree: with the gene names (the second pass, the results),
        or with the gene IDs (the first pass, read by the update of the orthogroups:
        gene names need not be unique, e.g. with -X the same name in two species,
        so a tree of names cannot be turned back into the genes). If it cannot be
        written the run stops (its orthologs and HOGs are not dropped without the
        run failing): the update needs these trees, and a results file would be
        missing.
        """
        fn = self.recon_trees_renamed_dir + "OG%07d.txt" % iog
        if self.write_hog_tree:
            ids_map = {leaf.name: leaf.name for leaf in recon_tree.get_leaves()}
        else:
            ids_map = self.spec_seq_dict
        try:
            phylo_taxa.rename_tree_taxa(recon_tree, fn, ids_map, q_support=False, q_fix_negatives=True)
        except Exception as e:
            raise parallel_task_manager.WorkerError(
                "ERROR: Cannot write the resolved gene tree of OG%07d, %s (%s: %s)" % (iog, fn, type(e).__name__, e))


# The ortholog and xenologue rows of a tree a worker holds before it sends them
# on as sorted chunks (get_lines_for_ortholog_files: flush). Most trees have
# fewer, and are sent whole. Fixed (not system_info.memory_budget): whether a tree is
# sent in chunks decides how its rows are written, so where the gzip files' members end.
CHUNK_BYTES = 64 * 1024 * 1024


def sorted_rows(rows):
    """
    The rows (each ending in a newline) sorted, as one text: as sorted_lines
    (ortholog_writers), which sorts the lines without their newlines ("a\tb"
    sorts after "a", "a\tb\n" before "a\n").
    """
    return "\n".join(sorted(row[:-1] for row in rows)) + "\n"


def empty_ortholog_lines(nspecies):
    """
    One tree's ortholog rows: the text of each species (by its index), the rows
    of its file Orthologs/<species>.tsv.
    """
    return ["" for _ in range(nspecies)]


class SpeciesSets(object):
    """
    Species below every node of a gene tree, computed once in one postorder pass.

    Each set is stored as an int bitmask (one bit per species), so memory is
    about nspecies/8 bytes per node rather than a Python set per node, and a
    union is a single OR. Valid while the tree topology does not change.
    """

    def __init__(self, tree, GeneToSpecies):
        self.species_names = []
        self.species_bit = {}
        self.masks = {}
        self.leaves = {}
        self._set_cache = {}
        for n in tree.traverse("postorder"):
            if n.is_leaf():
                sp = GeneToSpecies(n.name)
                bit = self.species_bit.get(sp)
                if bit is None:
                    bit = 1 << len(self.species_names)
                    self.species_bit[sp] = bit
                    self.species_names.append(sp)
                self.masks[id(n)] = bit
                self.leaves.setdefault(n.name, n)
            else:
                mask = 0
                for c in n.get_children():
                    mask |= self.masks[id(c)]
                self.masks[id(n)] = mask

    def mask(self, node):
        return self.masks[id(node)]

    def species(self, node):
        """A new set of the species below node (callers may modify it)."""
        mask = self.masks[id(node)]
        cached = self._set_cache.get(mask)
        if cached is None:
            names = self.species_names
            out = []
            m = mask
            while m:
                low = m & -m
                out.append(names[low.bit_length() - 1])
                m ^= low
            cached = frozenset(out)
            self._set_cache[mask] = cached
        return set(cached)

    def leaf(self, name):
        return self.leaves[name]


def get_orthologs_from_tree(
        iog,
        tree,
        species_tree_rooted,
        GeneToSpecies,
        neighbours,
        q_get_dups=False,
        q_no_recon=False,
        polytomies=None,
        q_orthologs=True,
        resolved=None,
        hog_pass_resolved=None,
    ):
    """
    Infer orthologs and duplications from a reconciled gene tree.

    q_orthologs: also the orthologs (not for the HOG pass, which does not use
    them: the tree's features, suspect genes and duplications only)

    polytomies: a list, to which a record is added for each polytomy
    (polytomy_record)

    resolved: a list, to which (genes, children, species sets of children) is
    added for each polytomy resolved (resolve_polytomies), in the first pass

    hog_pass_resolved: those of the first pass, (records, {gene: index}),
    recorded in polytomies as the nodes of their genes in this tree
    """
    og_name = "OG%07d" % iog
    q_prune = False

    species_and_gene = species_and_gene_lookup[GeneToSpecies]

    orthologs = []
    duplications = []

    resolved_polytomies = []
    if not q_no_recon:
        resolved_polytomies = resolve_polytomies(tree, GeneToSpecies, species_tree_rooted)
        tree = resolve_tree(tree, GeneToSpecies)
        # (and any that resolve_tree's moves made: the tree the first pass writes,
        # which the second reads, has none)
        resolved_polytomies += resolve_polytomies(tree, GeneToSpecies, species_tree_rooted)
        if resolved is not None:
            resolved.extend((genes, n_children, n_sets) for _, genes, n_children, n_sets in resolved_polytomies)

    if q_prune:
        tree.prune(tree.get_leaf_names())

    if len(tree) == 1:
        return set(orthologs), tree, set(), duplications

    i_node = 1
    tree.name = "n0"

    suspect_genes = set()
    empty_set = set()

    # The topology is fixed from here on, so species sets and leaf lookups can
    # be computed once instead of re-walking subtrees at every node.
    sp_sets = SpeciesSets(tree, GeneToSpecies)
    mrca_cache = {}

    def species_tree_mrca(mask, species):
        node = mrca_cache.get(mask)
        if node is None:
            node = mrca_node(species_tree_rooted, species)
            mrca_cache[mask] = node
        return node

    for n in tree.traverse("preorder"):
        if n.is_leaf():
            continue

        if not n.is_root():
            n.name = "n%d" % i_node
            i_node += 1

        ch = n.get_children()

        if len(ch) == 2:
            o_size, overlap, sp0, sp1 = overlap_size(
                n,
                GeneToSpecies,
                suspect_genes,
                sp_sets=sp_sets,
            )

            sp_present = sp0.union(sp1)
            if len(sp_present) == len(sp_sets.species(n)):
                # No species removed as suspect: the set is exactly the species below n.
                st_node = species_tree_mrca(sp_sets.mask(n), sp_present)
            else:
                st_node = mrca_node(species_tree_rooted, sp_present)

            n.add_feature("sp_node", st_node.name)

            if o_size != 0:
                q_resolved, misplaced_genes = resolve_overlap(
                    overlap,
                    sp0,
                    sp1,
                    ch,
                    tree,
                    neighbours,
                    GeneToSpecies,
                    sp_sets=sp_sets,
                )

                for g in misplaced_genes:
                    nn = sp_sets.leaf(g)
                    nn.add_feature("X", True)

            else:
                misplaced_genes = empty_set

            dup = o_size != 0 and not q_resolved

            n.add_feature("dup", dup)

            if dup:
                n.add_feature("dup_type", "species_overlap_duplication")

                if q_get_dups:
                    genes0 = ch[0].get_leaf_names()
                    genes1 = ch[1].get_leaf_names()

                    duplications.append((
                        st_node.name,
                        n.name,
                        float(o_size) / len(st_node),
                        genes0,
                        genes1
                    ))

            else:
                n.add_feature("dup_type", "speciation")

                if q_orthologs:
                    orthologs.append(
                        orthologs_and_suspect(
                            ch,
                            suspect_genes,
                            misplaced_genes,
                            species_and_gene
                        )
                    )

                suspect_genes.update(misplaced_genes)

        elif len(ch) > 2:
            all_species = sp_sets.species(n)
            st_node = species_tree_mrca(sp_sets.mask(n), all_species)
            n.add_feature("sp_node", st_node.name)

            if len(all_species) == 1:
                genes = n.get_leaf_names()
                duplications.append((st_node.name, n.name, 1.0, genes, []))
                n.add_feature("dup", True)
                if polytomies is not None:
                    polytomies.append(polytomy_record(og_name, n, 1, "single species", 0, species_and_gene))
            else:
                # The children grouped by their species (polytomy_orthologs): an
                # entry per pair of children would be one ortholog relationship
                # per pair, a billion for a polytomy of ~50,000 identical
                # sequences (FastTree joins them as one node)
                masks = [sp_sets.mask(child) for child in ch]
                q_dup = children_all_share_species(masks)
                entry = None
                if q_orthologs:
                    entry = polytomy_orthologs(ch, masks, suspect_genes, species_and_gene)
                    orthologs.append(entry)
                if q_dup:
                    duplications.append((st_node.name, n.name, 1.0, n.get_leaf_names(), []))
                n.add_feature("dup", q_dup)
                if polytomies is not None and entry is not None:
                    polytomies.append(polytomy_record(og_name, n, len(entry.groups),
                                                      "duplication" if q_dup else "speciation",
                                                      entry.n_ortholog_pairs(), species_and_gene))

    if polytomies is not None:
        records = [(genes, n_children, n_sets, node) for node, genes, n_children, n_sets in resolved_polytomies]
        if hog_pass_resolved is not None:
            # (the first pass's, by the genes of each in this tree)
            hog_records, index = hog_pass_resolved
            present = defaultdict(list)
            for gene in tree.get_leaf_names():
                k = index.get(gene)
                if k is not None:
                    present[k].append(gene)
            records.extend((genes, hog_records[k][1], hog_records[k][2], None)
                           for k, genes in sorted(present.items()) if len(genes) > 1)
        for genes, n_children, n_species_sets, node in records:
            if node is None or node.get_tree_root() is not tree:
                # (the node of its genes, if not its own: one resolve_tree removed)
                node = tree.get_common_ancestor(genes)
            polytomies.append(polytomy_record(og_name, node, n_species_sets, RESOLVED, 0, species_and_gene,
                                              genes=genes, n_children=n_children))

    return orthologs, tree, suspect_genes, duplications


def mrca_node(t_rooted, taxa):
    return (t_rooted & next(taxon for taxon in taxa)) if len(taxa) == 1 else t_rooted.get_common_ancestor(taxa)


def overlap_size(node, GeneToSpecies, suspect_genes, sp_sets=None):
    if sp_sets is None:
        descendents = [{GeneToSpecies(l) for l in n.get_leaf_names()}.difference(suspect_genes) for n in node.get_children()]
    else:
        descendents = [sp_sets.species(n).difference(suspect_genes) for n in node.get_children()]
    intersection = descendents[0].intersection(descendents[1])
    return len(intersection), intersection, descendents[0], descendents[1]

def resolve_overlap(overlap, sp0, sp1, ch, tree, neighbours, GeneToSpecies, rel_overlap_cutoff=4, sp_sets=None):
    """
    Is an overlap suspicious and if so can it be resolved by identifying genes that are out of place?
    Args:
        overlap - the species with genes in both clades
        sp0 - the species below ch[0]
        sp1 - the species below ch[1]
        ch - the two child nodes
        tree - the gene tree
        neighbours - dictionary species->neighbours, where neighbours is a list of the sets of species observed at successive topological distances from the species
    Returns:
        qSuccess - has the overlap been resolved
        genes_removed - the out-of-place genes that have been removed so as to resolve the overlap
    
    Implementation:
        - The number of species in the overlap must be a 5th or less of the number of species in each clade - What if it's a single gene that's out of place? Won't make a difference then to the orthologs!
        - for each species with genes in both clades: the genes in one clade must all be more out of place (according to the 
          species tree) than all the gene from that species in the other tree
    """
    o_size = len(overlap)
    lsp0 = len(sp0)
    lsp1 = len(sp1)
    if (o_size == lsp0 or o_size == lsp1) or (rel_overlap_cutoff*o_size >= lsp0 and rel_overlap_cutoff*o_size >= lsp1): 
        return False, []
    # The overlap looks suspect, misplaced genes?
    # for each species, we'd need to be able to determine that all genes from A or all genes from B are misplaced
    genes_removed = []
    n_a_removed = 0
    n_b_removed = 0
    q_resolved = True
    for sp in overlap:
        A = [g for g in ch[0].get_leaf_names() if GeneToSpecies(g) == sp]
        B = [g for g in ch[1].get_leaf_names() if GeneToSpecies(g) == sp]
        a_levels = []
        b_levels = []
        for X, level in zip((A,B),(a_levels, b_levels)):
            for g in X:
                if sp_sets is None:
                    gene_node = tree & g
                    species_below = lambda node: set([GeneToSpecies(gg) for gg in node.get_leaf_names()])
                else:
                    gene_node = sp_sets.leaf(g)
                    species_below = sp_sets.species
                r = gene_node.up
                next_species = species_below(r)
                # having a gene from the same species isn't enough?? No, but we add to the count I think.
                while len(next_species) == 1:
                    r = r.up
                    next_species = species_below(r)
                next_species.remove(sp)
                # get the level
                # the sum of the closest and furthest expected distance topological distance for the closest genes in the gene tree (based on species tree topology)
                neigh = neighbours[sp]
                observed = [neigh[n_sp] for n_sp in next_species]
                level.append(min(observed) + max(observed))
        q_remove_a = max(b_levels) + 2 < min(a_levels)   # if the clade is one step up the tree further way (min=max) then this gives +2. There's no way this is a problem                        
        q_remove_b = max(a_levels) + 2 < min(b_levels)                           
        if q_remove_a and rel_overlap_cutoff*o_size < len(sp0):
            n_a_removed += len(a_levels)
            genes_removed.extend(A)
        elif q_remove_b and rel_overlap_cutoff*o_size < len(sp1):
            n_b_removed += len(b_levels)
            genes_removed.extend(B)
        else:
            q_resolved = False
            break
    if q_resolved:
        return True, set(genes_removed)
    else:
        return False, set()
          
# The polytomies of the gene trees, a row each, next to Resolved_Gene_Trees.txt
# (write_polytomies), and their stage in checkpoint.txt
POLYTOMIES_FN = "Polytomies.tsv"
POLYTOMY_STAGE = "Infer orthologs"
POLYTOMIES_HEADER = ("Orthogroup", "Node", "Children", "Genes", "Species", "Max Genes of a Species",
                     "Species Sets of Children", "Event", "Ortholog Pairs")


def polytomy_record(og_name, node, n_species_sets, event, n_pairs, species_and_gene, genes=None,
                    n_children=None):
    """
    The row of Polytomies.tsv of a polytomy, node (POLYTOMIES_HEADER). For a
    polytomy resolved (event RESOLVED, resolve_polytomies), its genes and
    children, and node the node of its genes in the resolved tree. Its
    orthologs are rows of binary nodes, so it has no ortholog pairs of its own.
    """
    per_species = defaultdict(int)
    for gene in (node.get_leaf_names() if genes is None else genes):
        per_species[species_and_gene(gene)[0]] += 1
    return (og_name, node.name, len(node.get_children()) if n_children is None else n_children,
            sum(per_species.values()), len(per_species), max(per_species.values()), n_species_sets, event,
            n_pairs)


class MergedGenes(list):
    """The genes of a species in several groups of a polytomy, merged for one grouped entry."""


class DisjointGroups:
    """
    Whether each pair of groups of a polytomy's children has no species in
    common, a row (group) at a time or in blocks of rows, without a groups x
    groups matrix of floats (which for tens of thousands of groups would not
    fit in memory). If every group is of one species (the usual large
    polytomy, of identical sequences), every pair of groups is (their species
    sets differ): nothing is held. Otherwise from their species sets as bits
    (words: groups x 64-bit words), held as a bool matrix if it is at most
    BLOCK_BYTES, else made a block of rows at a time when needed.
    """
    BLOCK_BYTES = 64 * 1024 ** 2      # (of a block of rows being made, or used)

    def __init__(self, group_masks):
        self.k = len(group_masks)
        self.words = None
        self.matrix = None
        if not all(m & (m - 1) == 0 for m in group_masks):
            n_words = max(m.bit_length() for m in group_masks) // 64 + 1
            self.words = np.array([[(m >> (64 * w)) & 0xFFFFFFFFFFFFFFFF for w in range(n_words)]
                                   for m in group_masks], dtype=np.uint64)
            if self.k * self.k <= self.BLOCK_BYTES:
                self.matrix = np.empty((self.k, self.k), dtype=bool)
                for start, stop in self._ranges(8):
                    self.matrix[start:stop] = self._make_rows(start, stop)

    def _ranges(self, bytes_per_cell):
        step = max(1, self.BLOCK_BYTES // max(self.k * bytes_per_cell, 1))
        for start in range(0, self.k, step):
            yield start, min(self.k, start + step)

    def _make_rows(self, start, stop):
        shared = np.zeros((stop - start, self.k), dtype=bool)
        for w in range(self.words.shape[1]):        # (a word at a time: no rows x groups x words)
            shared |= (self.words[start:stop, w, None] & self.words[None, :, w]) != 0
        return ~shared

    def rows(self, start, stop):
        """Those of the groups start to stop (rows) with every group (columns), as bool."""
        if self.words is None:
            block = np.ones((stop - start, self.k), dtype=bool)
            block[np.arange(stop - start), np.arange(start, stop)] = False
            return block
        if self.matrix is not None:
            return self.matrix[start:stop]
        return self._make_rows(start, stop)

    def later(self, a):
        """The groups after a with no species in common with it."""
        if self.words is None:
            return range(a + 1, self.k)
        return [int(b) for b in np.flatnonzero(self.rows(a, a + 1)[0, a + 1:]) + a + 1]

    def dot(self, x):
        """The matrix of whether the groups are disjoint (0/1) times x (groups x n), as float64."""
        if self.words is None:
            return x.sum(axis=0) - x            # (ones but the diagonal)
        out = np.empty_like(x, dtype=np.float64)
        for start, stop in self._ranges(8):
            out[start:stop] = self.rows(start, stop).astype(np.float64) @ x
        return out


class PolytomyOrthologs:
    """
    The orthologs of a polytomy (see polytomy_orthologs), in place of an
    (o_0, o_1, os_0, os_1) per pair of its children, which would be held
    together: a billion for a polytomy of ~50,000 identical sequences.

    children: the (o, os) of each child, as orthologs_and_suspect's
    groups: the children of each species set (indices)
    disjoint: the pairs of groups with no species in common (DisjointGroups)
    counts: the arguments of NOrthoSp.add_polytomy (species instead of
    indices), the relationships of every pair of children
    """

    def __init__(self, children, groups, disjoint, counts):
        self.children = children
        self.groups = groups
        self.disjoint = disjoint
        self.counts = counts

    def n_ortholog_pairs(self):
        """The pairs of genes written as orthologs (not suspect), of every pair of children with no species in common."""
        n = np.array([[sum(len(genes) for i in group for genes in self.children[i][0].values())]
                      for group in self.groups], dtype=np.float64)
        return int(round(float((n * self.disjoint.dot(n)).sum()) / 2))

    def grouped_entries(self):
        """
        (o_0, o_1, os_0, os_1) of the genes of each group against those of
        all the later groups with none of its species: the relationships of
        every pair of children, in rows per pair of species (rather than per
        pair of children), so their number does not grow with the children:
        for the usual polytomy of identical sequences of one species each, at
        most that of the pairs of species. A species' list is shared, not
        copied, if it is of one group.
        """
        sides = []
        for group in self.groups:
            sides.append([defaultdict(list), defaultdict(list)])
            for i in group:
                for side, genes in zip(sides[-1], self.children[i]):
                    for sp, seqs in genes.items():
                        side[sp].extend(seqs)
        for a in range(len(self.groups)):
            later = self.disjoint.later(a)
            if len(later) == 0:
                continue
            right = []
            for i_side in range(2):
                parts = defaultdict(list)
                for b in later:
                    for sp, genes in sides[b][i_side].items():
                        parts[sp].append(genes)
                right.append({sp: p[0] if len(p) == 1 else MergedGenes(itertools.chain.from_iterable(p))
                              for sp, p in parts.items()})
            yield sides[a][0], right[0], sides[a][1], right[1]


def polytomy_orthologs(ch, masks, suspect_genes, species_and_gene):
    """
    The orthologs of a polytomy of several species, its children ch, of
    species masks (SpeciesSets). The children are grouped by
    their species: the children of a group have the same species, so they are
    never orthologs of each other, and every child of a group is an ortholog
    of every child of a group with no species in common.

    Returns the PolytomyOrthologs, for any number of groups (the relationships
    of every pair of children, as of a smaller polytomy).
    """
    children = []
    for child in ch:
        normal, suspect = defaultdict(list), defaultdict(list)
        for gene in child.get_leaf_names():
            sp, seq = species_and_gene(gene)
            (suspect if gene in suspect_genes else normal)[sp].append(seq)
        children.append((normal, suspect))
    return polytomy_from_children(children, masks)


def polytomy_from_children(children, masks):
    """The PolytomyOrthologs of a polytomy's children ((o, os) of each) of species masks."""
    group_of = {}
    for m in masks:
        if m not in group_of:
            group_of[m] = len(group_of)
    k = len(group_of)

    groups = [[] for _ in range(k)]
    species, col = [], {}
    child_genes = []            # (group, species column, normal genes) of each child and species
    for i, ((normal, _), m) in enumerate(zip(children, masks)):
        g = group_of[m]
        groups[g].append(i)
        for sp, genes in normal.items():
            if sp not in col:
                col[sp] = len(species)
                species.append(sp)
            child_genes.append((g, col[sp], len(genes)))

    group_masks = list(group_of)
    disjoint = DisjointGroups(group_masks)

    c1, cm, sm = (np.zeros((k, len(species))) for _ in range(3))
    if child_genes:
        a = np.array(child_genes, dtype=np.int64)
        one = a[:, 2] == 1
        np.add.at(c1, (a[one, 0], a[one, 1]), 1)
        np.add.at(cm, (a[~one, 0], a[~one, 1]), 1)
        np.add.at(sm, (a[~one, 0], a[~one, 1]), a[~one, 2])
    return PolytomyOrthologs(children, groups, disjoint, (species, disjoint, c1, cm, sm))


def write_polytomies(records, d_out):
    """
    The polytomies of the gene trees (records of every tree, polytomy_record)
    to d_out + POLYTOMIES_FN, a row each, sorted by orthogroup and node, and
    a summary line in checkpoint.txt. No polytomies, no file (and none of an
    earlier run of the same results directory, as a restart from the trees).
    """
    fn = os.path.join(d_out, POLYTOMIES_FN)
    run_report.count("olog.polytomies_resolved", sum(1 for r in records if r[7] == RESOLVED), len(records))
    if not records:
        if os.path.exists(fn):
            os.remove(fn)
        return
    records = sorted(records, key=lambda r: (r[0], node_order(r[1])))
    with open(fn, "w", encoding="utf-8", newline="\n") as outfile:
        outfile.write("\t".join(POLYTOMIES_HEADER) + "\n")
        for record in records:
            outfile.write("\t".join(map(str, record)) + "\n")
    printing.report_notes(POLYTOMY_STAGE, [polytomies_summary(records, fn)])


def node_order(name):
    """A gene tree node's name (n<i>) as its number, to sort n2 before n10."""
    return (0, int(name[1:]), name) if name[:1] == "n" and name[1:].isdigit() else (1, 0, name)


def polytomies_summary(records, fn):
    """The line of checkpoint.txt of the polytomies (records, of POLYTOMIES_HEADER) written to fn."""
    n_ogs = len({r[0] for r in records})
    events = defaultdict(int)
    for r in records:
        events[r[7]] += 1
    largest = max(records, key=lambda r: (r[2], r[3]))
    resolved = (", %d resolved with the species tree" % events[RESOLVED]) if events[RESOLVED] else ""
    return ("%d %s in %d %s (%d speciation, %d duplication, %d single species%s); largest %s node %s, "
            "%d children, %d genes, %d species; details in %s"
            % (len(records), "polytomy" if len(records) == 1 else "polytomies",
               n_ogs, "orthogroup" if n_ogs == 1 else "orthogroups",
               events["speciation"], events["duplication"], events["single species"], resolved,
               largest[0], largest[1], largest[2], largest[3], largest[4], fn))


def children_all_share_species(masks):
    """
    Whether every pair of a polytomy's children shares a species (their
    species as bitmasks, SpeciesSets): at once if one species is in all of
    them, otherwise until the first pair that does not. Children of the same
    species share them, so only the pairs of different species sets are
    checked (the children of a large polytomy are mostly of a few).
    """
    common = masks[0]
    for m in masks[1:]:
        common &= m
    if common:
        return True
    return all(m0 & m1 for m0, m1 in itertools.combinations(set(masks), 2))


def orthologs_and_suspect(ch, suspect_genes, misplaced_genes, species_and_gene):
    """
    ch - the two child nodes that are orthologous
    suspect_genes - genes already identified as misplaced at lower levels
    misplaced_genes - genes identified as misplaced at this level

    Returns the tuple (o_0, o_1, os_0, os_1) where each element is a dictionary from species to genes from that species,
    the o are orthologs, the os are 'suspect' orthologs because the gene was previously identified as suspect
    """
    d = [defaultdict(list) for _ in range(2)]
    d_sus = [defaultdict(list) for _ in range(2)] 
    for node, di, d_susi in zip(ch, d, d_sus):
        for g in [g for g in node.get_leaf_names() if g not in misplaced_genes]:
            sp, seq = species_and_gene(g)
            if g in suspect_genes:
                d_susi[sp].append(seq)
            else:
                di[sp].append(seq)
    return d[0], d[1], d_sus[0], d_sus[1]


def get_lines_for_ortholog_files(
        orthologs_alltrees,
        species_dict,
        i_species_to_use,
        sequence_dict,
        q_contains_suspect_orthologs,
        ortholog_lines,
        ortholog_sus_lines,
        pair_counts=None,
        other_names=None,
        flush=None,
        chunk_bytes=None,
    ):
    """
    Add ortholog rows to olog_lines/olog_sus_lines.

    Returns a dense NOrthoSp for these trees. If pair_counts (a
    stats.PendingPairs) is given, the counts are added to it instead and None
    is returned, which avoids allocating five nspecies x nspecies matrices per
    tree.

    flush: for a tree with more than chunk_bytes (default CHUNK_BYTES) of rows, called with
    ({i: text}, {i: text}) (the ortholog and xenologue rows of each species,
    sorted, see sorted_rows) each time the rows held reach chunk_bytes, and for
    the rest at the end, instead of adding them to olog_lines/olog_sus_lines:
    the rows held are bounded, not those of the whole tree (the number of
    species pairs: hundreds of millions for a tree of tens of thousands of
    genes). (The caller knows whether it was called.)

    other_names: {sequence ID: name} of the "Other" genes of the putative
    xenologues (of several species: <species>_<gene> if two species have genes
    of the same name, gene_ids.prefixes_if_shared); default sequence_dict.
    """
    nsp = len(i_species_to_use)
    n_ortho = stats.NOrthoSp(nsp) if pair_counts is None else None

    # Rows are collected in lists and joined once at the end. Appending with
    # olog_lines[i][j] += row copies the whole accumulated string every time
    # (Python can't append in place to a string held in a list), which is
    # quadratic for large trees.
    ortholog_parts = defaultdict(list)   # i -> rows
    sus_parts = defaultdict(list)    # i -> rows
    if chunk_bytes is None:
        chunk_bytes = CHUNK_BYTES
    held = [0]          # characters of the rows held
    flushed = [False]

    def sorted_and_freed(parts_of):
        # each species' rows freed once sorted: not all of them twice in memory
        texts = {}
        for i in list(parts_of):
            parts = parts_of.pop(i)
            if parts:       # (a species' list can be empty: the pair loop takes it before its rows)
                texts[i] = sorted_rows(parts)
        return texts

    def flush_rows():
        flush(sorted_and_freed(ortholog_parts), sorted_and_freed(sus_parts))
        held[0] = 0
        flushed[0] = True

    sp_to_i = {str(sp): i for i, sp in enumerate(i_species_to_use)}
    sp_label = [species_dict[str(sp)] for sp in i_species_to_use]
    getrow = file_io.unquoted_line

    seq_cache = {}

    def seq_name(sp, g):
        key = str(sp) + "_" + str(g)

        v = seq_cache.get(key)

        if v is None:
            v = sequence_dict[key]
            seq_cache[key] = v

        return v

    def join_gene_names(sp, genes):
        return ", ".join(
            sorted(seq_name(sp, g) for g in genes)
        )

    def join_other_names(sp, genes):
        if other_names is None:
            return join_gene_names(sp, genes)
        return ", ".join(sorted(other_names[str(sp) + "_" + str(g)] for g in genes))

    def add_stats(i_l, i_r, n_l, n_r):
        if pair_counts is None:
            n_ortho.add_pair(i_l, i_r, n_l, n_r)
        else:
            pair_counts.add_pair(i_l, i_r, n_l, n_r)

    def add_held(n_chars):
        if flush is not None:
            held[0] += n_chars
            if held[0] >= chunk_bytes:
                flush_rows()

    def field(text):
        # as file_io.unquoted_line: without trailing line breaks
        return text.rstrip("\r\n") if ("\n" in text or "\r" in text) else text

    label_tab = [field(label) + "\t" for label in sp_label]

    # The names of a species' genes joined, by its list's id: the lists of a large
    # polytomy's genes are in many of its grouped entries. Each list is kept with
    # its text, so that its id is not reused while cached. A MergedGenes list is
    # of one grouped entry only, so not cached (nor kept).
    shared_texts = {}

    def species_texts(leaves, q_shared=False):
        """(i, its genes' names joined, number of genes) of each species with genes, in species order."""
        texts = []
        for sp in sorted(leaves.keys(), key=lambda sp: sp_to_i[str(sp)]):
            genes = leaves[sp]
            if genes:
                q_cached = q_shared and type(genes) is not MergedGenes
                text = shared_texts[id(genes)][1] if q_cached and id(genes) in shared_texts else None
                if text is None:
                    text = field(join_gene_names(str(sp), genes))
                    if q_cached:
                        shared_texts[id(genes)] = (genes, text)
                texts.append((sp_to_i[str(sp)], text, len(genes)))
        return texts

    def add_polytomy_counts(species, *counts):
        idx = [sp_to_i[str(sp)] for sp in species]
        (n_ortho if pair_counts is None else pair_counts).add_polytomy(idx, *counts)

    def add_suspect_row(og, sp_l, genes_l, sp_r, genes_r):
        sp_l = str(sp_l)
        sp_r = str(sp_r)

        if sp_l == sp_r:
            return

        if not genes_l or not genes_r:
            return

        i_l = sp_to_i[sp_l]
        i_r = sp_to_i[sp_r]

        text_l = join_gene_names(sp_l, genes_l)
        text_r = join_gene_names(sp_r, genes_r)
        other_l = join_other_names(sp_l, genes_l)
        other_r = join_other_names(sp_r, genes_r)

        row_l = getrow((og, text_l, other_r))
        row_r = getrow((og, text_r, other_l))
        sus_parts[i_l].append(row_l)
        sus_parts[i_r].append(row_r)
        add_held(len(row_l) + len(row_r))

    def add_rows(og_tab, side_l, side_r, q_count):
        # One row per pair of species, in both directions (one file per
        # species: OG, the other species, the genes of each).
        for i_l, text_l, n_l in side_l:
            parts_l = ortholog_parts[i_l]
            head_l = og_tab + label_tab[i_l]
            tail_l = "\t" + text_l + "\n"
            for i_r, text_r, n_r in side_r:
                if i_l == i_r:
                    continue
                row_l = og_tab + label_tab[i_r] + text_l + "\t" + text_r + "\n"
                row_r = head_l + text_r + tail_l
                parts_l.append(row_l)
                ortholog_parts[i_r].append(row_r)

                if q_count:
                    add_stats(i_l, i_r, n_l, n_r)
                if flush is not None:
                    held[0] += len(row_l) + len(row_r)
                    if held[0] >= chunk_bytes:
                        flush_rows()
                        parts_l = ortholog_parts[i_l]

    def add_suspect_rows(og, leaves_l, leaves_r, leaves_l_sus, leaves_r_sus):
        leaves_sus = (leaves_l_sus, leaves_r_sus)
        leaves_norm = (leaves_l, leaves_r)

        for i_pair in range(2):
            leaves_sus0 = leaves_sus[i_pair]
            leaves_sus1 = leaves_sus[1 - i_pair]
            leaves_norm1 = leaves_norm[1 - i_pair]

            partner_species = dict.fromkeys(
                list(leaves_norm1.keys()) + list(leaves_sus1.keys())
            )

            for sp0 in sorted(leaves_sus0.keys(), key=lambda sp: sp_to_i[str(sp)]):
                genes0 = leaves_sus0[sp0]

                if not genes0:
                    continue

                for sp1 in sorted(partner_species.keys(), key=lambda sp: sp_to_i[str(sp)]):
                    if str(sp0) == str(sp1):
                        continue

                    genes1 = (
                        list(leaves_norm1.get(sp1, [])) +
                        list(leaves_sus1.get(sp1, []))
                    )

                    if not genes1:
                        continue

                    add_suspect_row(
                        og,
                        sp0,
                        genes0,
                        sp1,
                        genes1
                    )

    for iog, orthologs_onetree in orthologs_alltrees:
        og = "OG%07d" % iog

        og_tab = field(og) + "\t"

        for entry in orthologs_onetree:
            if type(entry) is PolytomyOrthologs:
                # counted per pair of groups of its children
                add_polytomy_counts(*entry.counts)
                # (one grouped entry at a time: those of the later groups' genes merged are not all held)
                for leaves_l, leaves_r, leaves_l_sus, leaves_r_sus in entry.grouped_entries():
                    side_l = species_texts(leaves_l, True)
                    side_r = species_texts(leaves_r, True) if side_l else []
                    add_rows(og_tab, side_l, side_r, False)
                    if q_contains_suspect_orthologs:
                        add_suspect_rows(og, leaves_l, leaves_r, leaves_l_sus, leaves_r_sus)
                shared_texts.clear()
                continue

            leaves_l, leaves_r, leaves_l_sus, leaves_r_sus = entry
            # Each species' genes are named and joined once per node, not once
            # per pair of species.
            side_l = species_texts(leaves_l)
            side_r = species_texts(leaves_r) if side_l else []
            add_rows(og_tab, side_l, side_r, True)

            if q_contains_suspect_orthologs:
                add_suspect_rows(og, leaves_l, leaves_r, leaves_l_sus, leaves_r_sus)

    if flushed[0]:
        if any(ortholog_parts.values()) or any(sus_parts.values()):
            flush_rows()
        return n_ortho

    for i, parts in ortholog_parts.items():
        ortholog_lines[i] += "".join(parts)
    for i, parts in sus_parts.items():
        ortholog_sus_lines[i] += "".join(parts)

    return n_ortho


class HogAnalysisError(Exception):
    """The HOGs of a gene tree could not be inferred (get_hogs_from_tree)."""


def get_hogs_from_tree(
        iog,
        tree,
        hog_writer,
        q_split_paralogous_clades,
        q_report=False,
    ):
    """
    Generate HOG rows from one gene tree.

    Important:
        This function no longer writes to files.
        It returns cached_hogs to the parent process.
    """
    og_name = "OG%07d" % iog
    if debug:
        print("\n===== %s =====" % og_name)

    try:
        # (q_report: the duplications of no single level in the run report, from the pass of the results' HOGs)
        tree = hog_writer.mark_dups_below(tree, og_name if q_report else None)
        cached_hogs = []

        for n in tree.traverse("preorder"):
            cached_hogs.extend(
                hog_writer.write_clade_v2(
                    n,
                    og_name,
                    q_split_paralogous_clades
                )
            )

        return cached_hogs

    except Exception as e:
        # (the run report's "HOGs" check, not an error of the tree's orthologs)
        raise HogAnalysisError("%s: %s" % (type(e).__name__, e)) from e
