# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:11:11 2017

@author: david

Perform directed 'reconciliation' first and then apply EggNOG method

1 - root gene trees on outgroup: unique one this time
2 - infer orthologs
"""
from ..utils import logging as run_logging
import os
import sys
import glob
import resource
import operator
import itertools
import traceback
import multiprocessing as mp
from collections import Counter, defaultdict
import warnings
import queue
import time
import io

import numpy as np
try:
    from rich import print
except ImportError:
    ...

from .tree_analyser import TreeAnalyser, mrca_node, gene_to_species_dash, get_lines_for_ortholog_files, empty_ortholog_lines, \
    write_polytomies
from .hog_writer import HogWriter
from .parallel_pipeline import run_orthologs_parallel_pipeline, set_file_descriptor_limit
from .ortholog_writers import ParentOutputWriter
from ..phylo import tree
from ..utils import util, file_handler
from ..utils import printing
from ..utils import species_info
from ..utils import gene_ids
from ..phylo import taxa
from . import stats


@run_logging.RunLogger.stage("Infer orthologs")
def reconciliation_and_orthologs(
        recon_method,
        og_set,
        n_high_parallel,
        n_low_parallel,
        i_species_tree=None,
        stride_dups=None,
        q_split_para_clades=False,
        save_space=False,
        print_info=True,
        exist_msa=True,
        write_hog_tree=True,
        write_to_rd=True,
        fd_limit=None,
        write_named_hogs=True,
    ):

    species_tree_ids_fn = file_handler.handler.get_species_tree_ids_rooted_fn()

    if write_to_rd:
        taxa.rename_tree_taxa(
            species_tree_ids_fn,
            file_handler.handler.get_species_tree_results_node_labels_fn(),
            og_set.species_dict(),
            q_support=False,
            q_fix_negatives=True,
            label="N"
        )

    if print_info:
        util.print_time("Inferring orthologs from gene trees")

    q_no_recon = ("only_overlap" == recon_method)

    # Label species-tree internal nodes deterministically.
    species_tree_rooted_labelled = tree.Tree(species_tree_ids_fn)
    species_tree_rooted_labelled.name = "N0"

    i_node = 1
    node_names = [species_tree_rooted_labelled.name]

    for n in species_tree_rooted_labelled.traverse():
        if (not n.is_leaf()) and (not n.is_root()):
            n.name = "N%d" % i_node
            node_names.append(n.name)
            i_node += 1

    species_dict = og_set.species_dict()
    sequence_dict = og_set.sequence_dict()

    if fd_limit is not None:
        set_file_descriptor_limit(fd_limit)

    hog_writer = HogWriter(
        species_tree_rooted_labelled,
        node_names,
        sequence_dict,
        species_dict,
        og_set.species_to_use,
        write_to_rd=write_to_rd,
        write_named=write_named_hogs,
    )

    # The orthologs of the orthogroups of 2 or 3 genes, written after those of the trees
    last_ortholog_lines, n_orthologs_last = None, None
    if not write_hog_tree:
        last_ortholog_lines, n_orthologs_last = two_and_three_gene_orthogroups(og_set)

    try:
        n_orthologs_sp_pair = do_orthologs_for_ortho_finder(
            og_set,
            species_tree_rooted_labelled,
            gene_to_species_dash,
            stride_dups,
            q_no_recon,
            hog_writer,
            q_split_para_clades,
            n_low_parallel,
            save_space,
            print_info=print_info,
            exist_msa=exist_msa,
            write_hog_tree=write_hog_tree,
            fd_limit=fd_limit,
            last_ortholog_lines=last_ortholog_lines,
        )
        if n_orthologs_last is not None:
            n_orthologs_sp_pair += n_orthologs_last

        if print_info:
            util.print_time("Done of orthologs")

        two_and_three_gene_hogs(
            og_set,
            species_tree_rooted_labelled,
            hog_writer
        )

        hog_writer.close_files()

    finally:
        hog_writer.close_files()

    return n_orthologs_sp_pair



def two_and_three_gene_hogs(og_set, st_rooted_labelled, hog_writer):
    ogs = og_set.get_ogs_all()
    for iog, og in enumerate(ogs):
        n = len(og) 
        if n < 2 or n > 3: continue
        og_name = "OG%07d" % iog
        sp_present = set([str(g.i_sp) for g in og])
        st_node = mrca_node(st_rooted_labelled, sp_present)
        hogs_to_write = hog_writer.get_skipped_nodes(st_node, None)  
        if len(sp_present) > 1:
            # We don't create files for 'species specific HOGs'
            st_node = mrca_node(st_rooted_labelled, sp_present)
            hogs_to_write = hogs_to_write + [st_node.name]
        # sorted so the row does not depend on the gene order of the orthogroup
        genes = sorted((g.to_string() for g in og), key=species_info.seq_id_key)
        hog_writer.write_hog_genes(genes, hogs_to_write, og_name)


def two_and_three_gene_orthogroups(og_set):
    """
    The ortholog lines of the orthogroups of 2 or 3 genes (in the layout of
    get_lines_for_ortholog_files) and their number per species pair. They are
    written after those of the trees (ParentOutputWriter.write_last): these
    orthogroups have the fewest genes, so the last orthogroup numbers.
    """
    species_dict = og_set.species_dict()
    sequence_dict = og_set.sequence_dict()

    all_orthologs, nspecies, sp_to_index = get_all_orthologs(og_set)

    ortholog_lines_tot = empty_ortholog_lines(nspecies)

    ortholog_sus_lines_tot = [
        ""
        for _ in range(nspecies)
    ]

    n_orthologs_sp_pair = get_lines_for_ortholog_files(
        all_orthologs,
        species_dict,
        og_set.species_to_use,
        sequence_dict,
        q_contains_suspect_orthologs=False,
        ortholog_lines=ortholog_lines_tot,
        ortholog_sus_lines=ortholog_sus_lines_tot,
    )

    return ortholog_lines_tot, n_orthologs_sp_pair


def get_all_orthologs(og_set):
    ogs = og_set.get_ogs_all()
    all_orthologs = []
    d_empty = defaultdict(list)
    for iog, og in enumerate(ogs):
        n = len(og) 
        if n == 1: break
        elif n == 2:
            if og[0].i_sp == og[1].i_sp: continue
            # orthologs is a list of tuples of dictionaries
            # each dictionary is sp->list of genes in species
            d0 = defaultdict(list)
            d0[str(og[0].i_sp)].append(str(og[0].i_seq))
            d1 = defaultdict(list)
            d1[str(og[1].i_sp)].append(str(og[1].i_seq))
            orthologs = [(d0, d1, d_empty, d_empty)]
        elif n == 3:
            sp = [g.i_sp for g in og]
            c = Counter(sp) 
            n_sp = len(c)
            if n_sp == 3:
                g = [(str(g.i_sp), str(g.i_seq)) for g in og]
                d0 = defaultdict(list)
                d0[g[0][0]].append(g[0][1])
                d1 = defaultdict(list)
                d1[g[1][0]].append(g[1][1])
                d1[g[2][0]].append(g[2][1])
                orthologs = [(d0, d1, d_empty, d_empty)]  
                d0 = defaultdict(list)
                d0[g[1][0]].append(g[1][1])
                d1 = defaultdict(list)
                d1[g[2][0]].append(g[2][1])
                orthologs.append((d0,d1, d_empty, d_empty))
            elif n_sp == 2:             
                sp0, sp1 = list(c.keys())
                d0 = defaultdict(list)
                d0[str(sp0)] = [str(g.i_seq) for g in og if g.i_sp == sp0]
                d1 = defaultdict(list)
                d1[str(sp1)] = [str(g.i_seq) for g in og if g.i_sp == sp1]
                orthologs = [(d0, d1, d_empty, d_empty)]
            else: 
                continue # no orthologs
            
        else:
            continue
        if orthologs:
            all_orthologs.append((iog, orthologs))

    nspecies = len(og_set.species_to_use)
    sp_to_index = {str(sp):i for i, sp in enumerate(og_set.species_to_use)}
    return all_orthologs, nspecies, sp_to_index


def do_orthologs_for_ortho_finder(
        og_set,
        species_tree_rooted_labelled,
        GeneToSpecies,
        stride_dups,
        q_no_recon,
        hog_writer,
        q_split_paralogous_clades,
        n_parallel,
        save_space,
        print_info=True,
        exist_msa=True,
        write_hog_tree=True,
        fd_limit=None,
        last_ortholog_lines=None,
    ):
    try:
        species_dict = og_set.species_dict()
        sequence_dict = og_set.sequence_dict()
        nspecies = len(og_set.species_to_use)

        # none in the pass that writes only the HOG trees
        d_results_orthologs = None if write_hog_tree else file_handler.handler.get_orthologs_directory()

        neighbours = get_species_neighbours(species_tree_rooted_labelled)
        iogs4 = list(og_set.get_i_ogs4())
        recon_trees_renamed_dir = file_handler.handler.get_ogs_recon_tree_dir(True)
        spec_seq_dict = og_set.spec_seq_dict()
        # the "Other" genes of Putative_Xenologs (of the results pass only): as Orthogroups.txt
        # (decided here, before the workers are forked)
        other_names = None
        if not write_hog_tree and gene_ids.prefixes_if_shared(
                sequence_dict, species_dict, og_set.species_to_use, og_set.q_add_species_to_ids):
            other_names = spec_seq_dict
        output_writer = ParentOutputWriter(
            d_results_orthologs=d_results_orthologs,
            species_dict=species_dict,
            species_to_use=og_set.species_to_use,
            sequence_dict=sequence_dict,
            spec_seq_dict=spec_seq_dict,
            stride_dups=stride_dups,
            hog_writer=hog_writer,
            save_space=save_space,
            write_hog_tree=write_hog_tree,
            last_ortholog_lines=last_ortholog_lines,
        )

        ta = TreeAnalyser(
            len(iogs4),
            d_results_orthologs,
            recon_trees_renamed_dir,
            species_tree_rooted_labelled,
            og_set.species_to_use,
            GeneToSpecies,
            sequence_dict,
            species_dict,
            spec_seq_dict,
            neighbours,
            q_no_recon,
            None,       
            stride_dups,
            None,  
            None,  
            hog_writer,
            q_split_paralogous_clades,
            exist_msa=exist_msa,
            write_hog_tree=write_hog_tree,
            other_names=other_names,
        )

        total_tasks = len(iogs4)

        if n_parallel == 1:
            progressbar, task = printing.get_progressbar(total_tasks)
            progressbar.start()

            # (the HOG pass counts none: write_hog_tree)
            n_orthologs_sp_pair = None if write_hog_tree else stats.NOrthoSp(nspecies)
            output_writer.runs.spill_dir = file_handler.handler.get_working_directory_write()

            def emit_chunk(iog, olog_texts, sus_texts):
                # a large tree's rows, as they are made: held on disk until write_result
                output_writer.add_chunk(iog, {
                    "olog_chunks": list(olog_texts.items()),
                    "xenolog_chunks": list(sus_texts.items()),
                })

            try:
                for completed_tasks, iog in enumerate(iogs4, start=1):
                    # (the counts added to n_orthologs_sp_pair, as the parallel workers do:
                    # not a species x species NOrthoSp per tree)
                    result = ta.analyse_tree(
                        iog,
                        n_ortho_acc=n_orthologs_sp_pair,
                        emit_chunk=emit_chunk if output_writer.need_ortholog_output else None,
                    )

                    if result is None:
                        progressbar.update(task, advance=1)
                        continue

                    output_writer.write_result(result)

                    progressbar.update(task, advance=1)

                # (iogs4 is in OG order)
                output_writer.write_last()

            finally:
                progressbar.stop()
                output_writer.close()

            if print_info:
                util.print_time("Done writing orthologs")

        else:
            args_queue = mp.Queue()

            # Start the largest trees first so one big tree does not run alone
            # at the end. HOG rows are still committed in OG order.
            ogs_all = og_set.get_ogs_all()
            for iog in sorted(iogs4, key=lambda i: -len(ogs_all[i])):
                args_queue.put(iog)

            n_orthologs_sp_pair = run_orthologs_parallel_pipeline(
                ta,
                len(og_set.species_to_use),
                args_queue,
                n_parallel,
                total_tasks,
                output_writer,
                iogs_ordered=iogs4,
                n_orthologs_cache=100,
                write_hog_tree=write_hog_tree,
                fd_limit=fd_limit,
                writer_queue_size=max(4 * n_parallel, 32),
                spill_dir=file_handler.handler.get_working_directory_write(),
            )

        if not write_hog_tree:
            write_polytomies(ta.polytomies, file_handler.handler.get_user_resolved_tree_dir())

    except IOError as e:
        if str(e).startswith("[Errno 24] Too many open files"):
            printing.number_open_files_exception_advice(len(og_set.species_to_use), True)
            util.fail()
        else:
            raise

    return n_orthologs_sp_pair


class SpeciesNeighbours(object):
    """
    The topological distance of each species from the others in the species
    tree, neighbours[sp][other]: the number of nodes above sp below the one
    that has other below it too (0 for its sister). An array of species x
    species (int16, or int32 for a tree deeper than that), not a dictionary
    of dictionaries: species^2 Python entries, several GB at 5,000 species.
    """

    def __init__(self, names, levels):
        self.index = {name: i for i, name in enumerate(names)}
        self.levels = levels

    def __getitem__(self, sp):
        return _NeighbourRow(self.levels[self.index[sp]], self.index)

    def __len__(self):
        return len(self.index)


class _NeighbourRow(object):
    __slots__ = ("row", "index")

    def __init__(self, row, index):
        self.row = row
        self.index = index

    def __getitem__(self, other):
        return int(self.row[self.index[other]])


def get_species_neighbours(t):
    """
    Args: t = rooted species tree

    Returns:
    SpeciesNeighbours: neighbours[species][other_species] -> topological distance
    """
    species = t.get_leaf_names()
    index = {s: i for i, s in enumerate(species)}
    n = len(species)
    # the distance of a species from another is the number of the nodes above it before their
    # common ancestor (the level, in postorder): one pass down from each species to the root
    levels = np.zeros((n, n), dtype=np.int16 if n < np.iinfo(np.int16).max else np.int32)
    level = np.zeros(n, dtype=np.int64)
    leaves = {}     # node -> the indices of its species, until its parent is reached
    for node in t.traverse('postorder'):
        if node.is_leaf():
            leaves[node] = np.array([index[node.name]], dtype=np.int64)
            continue
        children = node.get_children()
        sets = [leaves.pop(ch) for ch in children]
        everyone = np.concatenate(sets)
        for i, own in enumerate(sets):
            others = np.concatenate([x for j, x in enumerate(sets) if j != i]) if len(sets) > 1 else everyone[:0]
            levels[np.ix_(own, others)] = level[own][:, None]
        level[everyone] += 1
        leaves[node] = everyone
    return SpeciesNeighbours(species, levels)

