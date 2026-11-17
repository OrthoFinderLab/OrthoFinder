from __future__ import absolute_import

import os
import numpy as np

import numpy.core.numeric as numeric
import multiprocessing as mp

from . import mcl, score_normalisation
from ..gene_tree import msa_trees
from . import orthogroups_set
from ..utils import util, file_handler, matrices, parallel_task_manager
from ..utils import printing
from ..utils import gene_ids


def write_graph_per_species(args):
    seqs_info, graph_fn, i_spec, d_pickle = args
    # calculate the 2-way connections for one query species
    with open(graph_fn + "_%d" % i_spec, "w", encoding="utf-8", newline="\n") as graph_file:
        B = matrices.load_matrix_array("B", seqs_info, i_spec, d_pickle)
        # The weights of each species' block (connections both ways, times the scores), a block at
        # a time, joined into one CSR of all the species' columns (column j_offset + j): a CSR of
        # each block held for every species would be a row pointer per gene per species
        rows, cols, values = [], [], []
        for j_spec in range(seqs_info.n_species):
            m1 = matrices.as_csr(matrices.load_matrix("connect", i_spec, j_spec, d_pickle))
            m2tr = numeric.transpose(
                matrices.as_csr(matrices.load_matrix("connect", j_spec, i_spec, d_pickle))
            )
            w = matrices.as_csr((m1 + m2tr).multiply(matrices.as_csr(B[j_spec])))
            del m1, m2tr
            B[j_spec] = None
            # (sorted columns: sorted_indices() always copies, only if they are not)
            w = w if w.has_sorted_indices else w.sorted_indices()
            rows.append(np.repeat(np.arange(w.shape[0]), np.diff(w.indptr)))
            cols.append(w.indices.astype(np.int64) + seqs_info.seq_starting_indices[j_spec])
            values.append(w.data)
            del w
        del B
        # (in each row the species in order, each one's columns in order: a stable sort by row)
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        indices = np.concatenate(cols)[order] if cols else np.zeros(0, dtype=np.int64)
        data = np.concatenate(values)[order] if values else np.zeros(0)
        n_query = seqs_info.n_seqs_per_species[seqs_info.species_to_use[i_spec]]
        indptr = np.zeros(n_query + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=n_query), out=indptr[1:])
        del rows, cols, values, order
        offset = seqs_info.seq_starting_indices[i_spec]
        for query in range(n_query):
            parts = ["%d    " % (offset + query)]
            a, b = indptr[query], indptr[query + 1]
            if a != b:
                parts.extend(
                    "%d:%.3f " % (j, value)
                    for j, value in zip(indices[a:b].tolist(), data[a:b].tolist())
                )
            parts.append("$\n")
            graph_file.write("".join(parts))
        if i_spec == (seqs_info.n_species - 1):
            graph_file.write(")\n")
        # util.print_time("Written final scores for species %d to graph file" % iSpec)


def write_graph_per_species_homology(args):
    seqs_info, graph_fn, i_spec, d_pickle = args
    # calculate the 2-way connections for one query species
    # W = [matrices.load_matrix("B", iSpec, jSpec, d_pickle).tolil() for jSpec in range(seqsInfo.nSpecies)]
    W = []
    for j_spec in range(seqs_info.n_species):
        w1 = matrices.as_csr(matrices.load_matrix("B", i_spec, j_spec, d_pickle))
        matrices.dump_matrix("H", (w1 > 0).tolil(), i_spec, j_spec, d_pickle)
        w2tr = numeric.transpose(matrices.as_csr(matrices.load_matrix("B", j_spec, i_spec, d_pickle)))
        W.append((w1 + w2tr > 0).tolil())  # symmetrise
    # matrices.dump_matrix_array("H", W, iSpec, d_pickle)
    with open(graph_fn + "_%d" % i_spec, "w", encoding="utf-8", newline="\n") as graph_file:
        for query in range(seqs_info.n_seqs_per_species[seqs_info.species_to_use[i_spec]]):
            offset = seqs_info.seq_starting_indices[i_spec]
            graph_file.write("%d    " % (offset + query))
            for j_spec in range(seqs_info.n_species):
                row = W[j_spec].getrowview(query)
                j_offset = seqs_info.seq_starting_indices[j_spec]
                for j in row.rows[0]:
                    graph_file.write("%d:%.3f " % (j + j_offset, 1.0))
            graph_file.write("$\n")
        if i_spec == (seqs_info.n_species - 1):
            graph_file.write(")\n")
        util.print_time("Written final scores for species %d to graph file" % i_spec)


def largest_species_first(seqs_info):
    """
    Species job indices, largest proteome first.

    Each per-species task processes one row of the species x species hit
    matrices, so its cost grows with the species' number of genes. Starting
    the largest first ("longest processing time first") stops one large
    species from running alone at the end while other workers sit idle. Only
    the order changes; every task does the same work.
    """
    n_seqs = [seqs_info.n_seqs_per_species[seqs_info.species_to_use[i]] for i in range(seqs_info.n_species)]
    return sorted(range(seqs_info.n_species), key=lambda i: -n_seqs[i])


def get_sequence_lengths(seqs_info):
    sequence_lengths = []
    for i_species, i_fasta in enumerate(seqs_info.species_to_use):
        sequence_lengths.append(np.zeros(seqs_info.n_seqs_per_species[i_fasta]))
        fasta_filename = file_handler.handler.get_species_fasta_fn(i_fasta)
        current_sequence_length = 0
        i_current_sequence = -1
        q_first_line = True
        with open(fasta_filename, encoding="utf-8") as infile:
            for row in infile:
                if len(row) > 1 and row[0] == ">":
                    if q_first_line:
                        q_first_line = False
                    else:
                        sequence_lengths[i_species][
                            i_current_sequence
                        ] = current_sequence_length
                        current_sequence_length = 0
                    _, i_current_sequence = gene_ids.get_id_pair_from_string(row[1:])
                else:
                    current_sequence_length += len(row.rstrip())
        sequence_lengths[i_species][i_current_sequence] = current_sequence_length
    return sequence_lengths


def do_orthogroups(
        options,
        species_info_obj,
        seqs_info,
        species_names_dict,
        i_unassigned=None,
        GRACE_PERIOD = 10.,
        STALL_TIMEOUT = 200.
    ):


    q_unassigned = i_unassigned is not None

    # Run Algorithm, cluster and output cluster files with original accessions
    printing.print_underline(
        "Running OrthoFinder algorithm"
        + (" for clade-specific genes" if q_unassigned else "")
    )

    # it's important to free up the memory from python used for processing the genomes
    # before launching MCL because both use sizeable amounts of memory. The only
    # way I can find to do this is to launch the memory intensive python code
    # as separate process that exits before MCL is launched.

    lengths = get_sequence_lengths(seqs_info)  # Alternatively, self-self bit scores, but it amounts to the same thing

    # Process BLAST hits
    util.print_time("Initial processing of each species")

    blast_dir_list = file_handler.handler.get_blast_results_dir()
    if q_unassigned:
        blast_dir_list = blast_dir_list[:1]  # only use latet directory with unassigned gene searches
    
    file_handler.handler.get_pickle_dir()  # create the pickle directory before the parallel processing to prevent a race condition
    cmd_queue = mp.Queue()  
    for i_species_job in largest_species_first(seqs_info):  # The i-th job, not the OrthoFinder species ID
        cmd_queue.put(i_species_job)

    for _ in range(options.n_process_alg):
        cmd_queue.put(None)
    
    total_tasks = seqs_info.n_species
    # progressbar, task = printing.get_progressbar(total_tasks)
    # update_cycle = 1
    # progressbar.start()
    result_queue = mp.Queue()
    running_processes = [
        mp.Process(
            target=score_normalisation.WaterfallMethod.worker_process_blast_hits_new,
            args=(
                seqs_info,
                blast_dir_list,
                lengths,
                cmd_queue,
                file_handler.handler.get_pickle_dir(),
                options.q_double_blast,
                options.v2_scores,
                q_unassigned,
                result_queue,
            ),
        )
        for _ in range(options.n_process_alg)
    ]

    parallel_task_manager.manage_queue_new(
        running_processes, 
        total_tasks, 
        options.n_process_alg, 
        result_queue, 
        GRACE_PERIOD = GRACE_PERIOD,
        STALL_TIMEOUT = STALL_TIMEOUT
    )

    if options.gathering_version < (3, 0):
        util.print_time("Connected putative homologues")
        ## -------------------------------------------------------------
        cmd_queue = mp.Queue()
        for i_species in largest_species_first(seqs_info):
            cmd_queue.put((seqs_info, i_species))

        for _ in range(options.n_process_alg):
            cmd_queue.put(None)
        result_queue = mp.Queue()

        running_processes = [
            mp.Process(
                target=score_normalisation.WaterfallMethod.worker_connect_cognates_new,
                args=(
                    cmd_queue,  
                    result_queue, 
                    file_handler.handler.get_pickle_dir(), 
                    options.v2_scores
                ),
            )
            for i_ in range(options.n_process_alg)
        ]
        parallel_task_manager.manage_queue_new(
            running_processes, 
            total_tasks, 
            options.n_process_alg, 
            result_queue, 
            GRACE_PERIOD = GRACE_PERIOD,
            STALL_TIMEOUT = STALL_TIMEOUT
        )

    
        graph_filename = score_normalisation.WaterfallMethod.write_graph_parallel(
            write_graph_per_species, seqs_info, options.n_process_alg, i_unassigned
        )

        # 5b. MCL
        clusters_filename, clusters_filename_pairs = (
            file_handler.handler.create_unused_clusters_fn(
                options.mcl_inflation, i_unassigned
            )
        )
        mcl.MCL.run_mcl(
            graph_filename, clusters_filename, options.n_process_alg, options.mcl_inflation
        )
        # If processing unassigned, then ignore all 'unclustered' genes - they will include any genes not included in this search
        mcl.convert_single_ids_to_id_pair(
            seqs_info, clusters_filename, clusters_filename_pairs, q_unassigned
        )

    elif options.gathering_version == (3, 2):

        graph_filename = score_normalisation.WaterfallMethod.write_graph_parallel(
            write_graph_per_species_homology, seqs_info, options.n_process_alg, i_unassigned
        )
        clusters_filename, clusters_filename_pairs = (
            file_handler.handler.create_unused_clusters_fn(
                options.mcl_inflation, i_unassigned
            )
        )
        mcl.MCL.run_mcl(
            graph_filename, clusters_filename, options.n_process_alg, options.mcl_inflation
        )
        mcl.convert_single_ids_to_id_pair(
            seqs_info, clusters_filename, clusters_filename_pairs, q_unassigned
        )

    if not q_unassigned:
        # (files of each orthogroup left by orthogroups inferred before, e.g. a -b restart)
        file_handler.handler.remove_orthogroup_files()
        post_clustering_orthogroups(
            clusters_filename_pairs,
            species_info_obj,
            seqs_info,
            species_names_dict,
            options,
        )
    return clusters_filename_pairs


def add_unclustered_genes(ogs, species_to_use, n_seqs_per_species):
    """
    Add each gene of the analysis that is in no orthogroup to ogs as a
    single-gene orthogroup, at the end in sequence-ID order (so the other
    orthogroups keep their numbers), as the final orthogroups from the HOGs do.
    With --assign the clustering of the new species' unassigned genes does not
    keep its single-gene clusters (mcl.convert_single_ids_to_id_pair), so
    without this their genes would be in no results file. Returns the number
    of genes added.
    """
    clustered = {gene for og in ogs for gene in og}
    missing = [
        f"{i_species}_{i_seq}"
        for i_species in species_to_use
        for i_seq in range(n_seqs_per_species[i_species])
        if f"{i_species}_{i_seq}" not in clustered
    ]
    ogs.extend({gene} for gene in missing)
    return len(missing)


def post_clustering_orthogroups(
        clusters_filename_pairs,
        species_info_obj,
        seqs_info,
        species_names_dict,
        options,
        q_incremental=False,
        ogs=None,
    ):
    """
    Write OGs & statistics to results files, write Fasta files.
    Args:
        q_incremental - These are not the final orthogroups, don't write results
        ogs - the orthogroups, if they are in memory and not written (--assign, before
              its clade-specific orthogroups): else read from clusters_filename_pairs
    """
    ogs_given = ogs
    if ogs is None:
        ogs = mcl.get_predicted_ogs(clusters_filename_pairs)
    # The MCL orthogroups are not results files here: they are written as MCL_Orthogroups.*
    # at the end of the run if wanted (-og, -mog, the other stop options, or a failed run:
    # run.finish.write_mcl_orthogroups)
    if options.q_stop_after_mcl_groups and not q_incremental:
        # every gene must be in an orthogroup (or Orthogroups_UnassignedGenes.tsv)
        add_unclustered_genes(ogs, species_info_obj.species_to_use, seqs_info.n_seqs_per_species)

    printing.print_underline("Writing orthogroups to file")
    ids_dict = mcl.MCL.write_orthogroup_files(
        ogs,
        [file_handler.handler.get_sequence_ids_fn()],
        None,
        clusters_filename_pairs,
        # Legacy/Orthogroups_id.txt, of the final MCL orthogroups only
        write_ids=options.keep_legacy and not q_incremental,
    )

    # Write Orthogroup FASTA files
    og_set = orthogroups_set.OrthogroupSet(
        options.min_seq,
        file_handler.handler.get_working_directory1_read(),
        species_info_obj.species_to_use,
        species_info_obj.n_sp_all,
        options.q_add_species_to_ids,
        options.tree_program,
        id_extractor=gene_ids.FirstWordExtractor,
        ogs=ogs_given,
    )


    tree_gen = msa_trees.TreesForOrthogroups(None, None, None)

    # d_seqs = files.FileHandler.get_results_seqs_dir()
    # if not os.path.exists(d_seqs):
    #     os.mkdir(d_seqs)

    # treeGen.write_fasta_files(fastaWriter, ogSet.get_ogs_all(), idsDict, False)

    d_seqs_id = file_handler.handler.get_seqs_id_dir()
    if not os.path.exists(d_seqs_id):
        os.mkdir(d_seqs_id)

    # Not Orthogroup_Sequences/: the MCL orthogroups' sequences (e.g. -og) are written by
    # tools/mcl_orthogroup_sequences.py
    tree_gen.write_fasta_files(file_handler.handler.get_species_seqs_dir(), species_info_obj.species_to_use,
                               og_set.get_ogs_all(), ids_dict, q_id=True, q_results=False,
                               n_processes=options.n_process_alg or 1)

    if not q_incremental:
        # stats.Stats(ogs, speciesNamesDict, speciesInfoObj.speciesToUse, files.FileHandler.iResultsVersion)
        # print("")
        util.print_time("Done orthogroups")
        file_handler.handler.log_ogs()
