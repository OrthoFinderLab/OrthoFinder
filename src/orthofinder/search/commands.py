from ..utils import util, file_handler
import itertools
import os

import numpy as np

class SearchCommands(object):
    """
    The search commands of pairs of species (query, database), in a given
    order, each made only when it is needed (iterated or indexed): the pairs
    are numpy arrays (16 bytes a pair with its size: 400 MB at 5,000 species), not a list of the
    commands (about 450 bytes each with the pairs and sizes: 25 million at
    5,000 species). The same commands, in the same order, as the list of
    them was. make(i, j) makes the command of the pair (i, j).
    """

    def __init__(self, make, pairs, sizes):
        self.make = make
        self.pairs = pairs
        self.sizes = sizes

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        # (the pairs as Python ints a block at a time: tolist() of all of them is the list avoided)
        make = self.make
        for start in range(0, len(self.pairs), self.BLOCK):
            for i, j in self.pairs[start:start + self.BLOCK].tolist():
                yield make(i, j)

    BLOCK = 100000

    def __getitem__(self, k):
        i, j = self.pairs[k].tolist()
        return self.make(i, j)

    def subset(self, keep):
        """The commands of keep (a numpy bool array, one for each), in the same order."""
        return SearchCommands(self.make, self.pairs[keep], self.sizes[keep])


def _pairs(rows, cols, q_all):
    """The pairs (i, j) of rows x cols (i row-major), those with i <= j unless q_all, as an n x 2 array."""
    rows = np.asarray(rows, dtype=np.int32)       # (species indices: 8 bytes a pair)
    cols = np.asarray(cols, dtype=np.int32)
    i = np.repeat(rows, len(cols))
    j = np.tile(cols, len(rows))
    if not q_all:
        keep = i <= j
        i, j = i[keep], j[keep]
    return np.column_stack((i, j))


def get_ordered_search_commands(
        seqs_info, 
        species_info_obj, 
        options, 
        prog_caller, 
        n_genes_per_species=None, 
        q_new_species_unassigned_genes=False
    ):

    """ Using the nSeq1 x nSeq2 as a rough estimate of the amount of work required for a given species-pair, returns the commands 
    ordered so that the commands predicted to take the longest come first. This allows the load to be balanced better when processing 
    the BLAST commands.
    n_genes_per_species: List[int] - number of genes per species
    Returns (SearchCommands, their task sizes as a numpy array).
    """
    i_species_previous = list(range(species_info_obj.i_first_new_species))
    i_species_new = list(range(species_info_obj.i_first_new_species, species_info_obj.n_sp_all))
    if q_new_species_unassigned_genes:
        if n_genes_per_species is not None:
            exclude = {isp for isp, n_genes in enumerate(n_genes_per_species) if n_genes == 0}
            i_species_new = list(set(i_species_new).difference(exclude))
        species_pairs = _pairs(i_species_new, i_species_new, True)
        n_genes = np.asarray(n_genes_per_species, dtype=np.int64)
    else:
        q_all = options.q_double_blast
        species_pairs = np.concatenate([_pairs(i_species_new, i_species_new, q_all),
                                        _pairs(i_species_new, i_species_previous, q_all),
                                        _pairs(i_species_previous, i_species_new, q_all)])
        # (a dict by species ID)
        n_genes = np.array([seqs_info.n_seqs_per_species[i] for i in range(species_info_obj.n_sp_all)],
                           dtype=np.int64)
    task_sizes = n_genes[species_pairs[:, 0]] * n_genes[species_pairs[:, 1]] if len(species_pairs) \
        else np.zeros(0, dtype=np.int64)

    # Smaller files should be processed first 
    q_largest_first = False
    if options.cmd_order != "ascending":
        q_largest_first = True 

    # (as sorted(zip(task_sizes, species_pairs)), reversed if largest first: by size, then the
    # pair, which is unique, so reversing the order is sorting in reverse)
    order = np.lexsort((species_pairs[:, 1], species_pairs[:, 0], task_sizes)) if len(species_pairs) \
        else np.zeros(0, dtype=np.int64)
    if q_largest_first:
        order = order[::-1]
    species_pairs, task_sizes = species_pairs[order], task_sizes[order]

    method_threads = None
    # Each species' files once, not for every pair: the FASTA file is looked for in the working directories
    fasta_fns, database_fns = {}, {}

    def fasta_fn(i):
        if i not in fasta_fns:
            fasta_fns[i] = file_handler.handler.get_species_unassigned_fasta_fn(i) if q_new_species_unassigned_genes \
                else file_handler.handler.get_species_fasta_fn(i)
        return fasta_fns[i]

    def database_fn(j):
        if j not in database_fns:
            database_fns[j] = file_handler.handler.get_species_database_n(j, options.search_program)
        return database_fns[j]

    def make(i_fasta, i_db):
        return prog_caller.get_search_method_command_search(
            options.search_program,
            fasta_fn(i_fasta),
            database_fn(i_db),
            file_handler.handler.get_blast_results_fn(i_fasta, i_db, q_for_creation=True),
            scorematrix=options.score_matrix,
            gapopen=options.gapopen,
            gapextend=options.gapextend,
            method_threads=method_threads,
            nucleotide=options.dna,
        )

    return SearchCommands(make, species_pairs, task_sizes), task_sizes


def get_ordered_search_commands_clades(
        seqs_info, 
        species_info_obj, 
        options, prog_caller,
        n_genes_per_species, 
        species_clades
    ):
    """
    Search all species
    """
    method_threads = None

    exclude = {isp for isp, n_genes in enumerate(n_genes_per_species) if n_genes == 0}
    species_pairs = []
    for clade in species_clades:
        clade = list(set(clade).difference(exclude))
        species_pairs.extend([(i, j) for i, j in itertools.product(clade, clade)])
    commands = []
    for i_fasta, i_db in species_pairs:
        if options.search_program in ["mmseqs"]:
            unassigned = file_handler.handler.get_species_unassigned_fasta_fn(i_fasta)
            query = os.path.join(os.path.dirname(unassigned), os.path.basename(unassigned).split(".", 1)[-1])
        else:
            query = file_handler.handler.get_species_unassigned_fasta_fn(i_fasta)

        commands.append(
            prog_caller.get_search_method_command_search(
                options.search_program,
                query,
                file_handler.handler.get_species_database_n(i_db, options.search_program),
                file_handler.handler.get_blast_results_fn(i_fasta, i_db, q_for_creation=True),
                scorematrix=options.score_matrix,
                gapopen=options.gapopen,
                gapextend=options.gapextend,
                method_threads=method_threads,
                nucleotide=options.dna,
            )
        )
    # commands = [
    #     prog_caller.get_search_method_command_search(
    #         options.search_program,
    #         files.FileHandler.get_species_unassigned_fasta_fn(iFasta),
    #         files.FileHandler.get_species_database_n(iDB, options.search_program),
    #         files.FileHandler.get_blast_results_fn(iFasta, iDB, qForCreation=True),
    #         scorematrix=options.score_matrix,
    #         gapopen=options.gapopen,
    #         gapextend=options.gapextend,
    #         method_threads=method_threads
    #     ) 
    #     for iFasta, iDB in speciesPairs
    # ]
    tasksize =  [
        os.stat(file_handler.handler.get_species_database_n(i_db, options.search_program)).st_size
        for i_fasta, i_db in species_pairs
        if os.path.isfile(file_handler.handler.get_species_database_n(i_db, options.search_program)) \
            and os.path.exists(file_handler.handler.get_species_database_n(i_db, options.search_program))
    ]

    return  commands, tasksize


def get_ordered_search_commands_assign(species_info_obj, diamond_db, options, prog_caller, q_one_query, threads=1):
    """ Using the nSeq1 x nSeq2 as a rough estimate of the amount of work required for a given species-pair, returns the commands
    ordered so that the commands predicted to take the longest come first. This allows the load to be balanced better when processing
    the BLAST commands.
    """
    i_species_new = list(range(species_info_obj.i_first_new_species, species_info_obj.n_sp_all))
    tasksize = None

    method_threads = None
    
    if q_one_query:
        wd = file_handler.handler.get_species_seqs_dir()[0]
        fn_single_fasta = "%sall_sequences.fa" % wd
        with open(fn_single_fasta, 'w', encoding="utf-8", newline="\n") as outfile:
            for i_fasta in i_species_new:
                with open(file_handler.handler.get_species_fasta_fn(i_fasta), 'r', encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                    outfile.write("\n")
        results = wd + "Blast_all_sequences.txt"
        # commands = [prog_caller.get_search_method_command_search(search_program, fn_single_fasta, diamond_db, results,
                                                                # scorematrix=options.score_matrix, 
                                                                # gapopen=options.gapopen, 
                                                                # gapextend=options.gapextend,#
                                                                # method_threads=options.method_threads)]
        commands = ["diamond blastp --ignore-warnings -d %s -q %s -o %s --more-sensitive -p %d --quiet -e 0.001 --compress 1" % (diamond_db, fn_single_fasta, results, threads)]
        results_files = [results + ".gz"]
    else:
        commands = [
            prog_caller.get_search_method_command_search(
                options.search_program,
                file_handler.handler.get_species_fasta_fn(i_fasta),
                diamond_db,
                file_handler.handler.get_blast_results_fn(i_fasta, -1, q_for_creation=True),
                scorematrix=options.score_matrix, 
                gapopen=options.gapopen, 
                gapextend=options.gapextend,#
                method_threads=method_threads,
                nucleotide=options.dna,
            )
            for i_fasta in i_species_new
        ]
        if options.search_program in ["mmseqs"]:
            results_files = [file_handler.handler.get_blast_results_fn(i_fasta, -1, q_for_creation=True) for i_fasta in i_species_new]
        else:
            results_files = [file_handler.handler.get_blast_results_fn(i_fasta, -1, q_for_creation=True) + ".gz" for i_fasta in i_species_new]
        tasksize = [
            os.stat(file_handler.handler.get_species_fasta_fn(i_fasta)).st_size
            for i_fasta in i_species_new
            if os.path.isfile(file_handler.handler.get_species_fasta_fn(i_fasta)) \
                and os.path.exists(file_handler.handler.get_species_fasta_fn(i_fasta))
        ]
    return commands, tasksize, results_files