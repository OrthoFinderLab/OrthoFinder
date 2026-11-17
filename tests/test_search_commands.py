"""search/commands.py: the search commands made as needed, the same and in the same order as the list of them."""
import itertools
import random
import types

from orthofinder.search import commands as sc
from orthofinder.utils import util


class Handler(object):
    def get_species_fasta_fn(self, i):
        return "/w/Species%d.fa" % i

    def get_species_unassigned_fasta_fn(self, i):
        return "/w/U.Species%d.fa" % i

    def get_species_database_n(self, j, program):
        return "/w/%sDB%d" % (program, j)

    def get_blast_results_fn(self, i, j, q_for_creation=False):
        return "/w/Blast%d_%d.txt" % (i, j)


class Caller(object):
    def get_search_method_command_search(self, method, query, db, out, **kwargs):
        return "%s %s %s %s" % (method, query, db, out)


def as_the_list(seqs, first_new, n, double, order, unassigned):
    """The commands of the earlier code: a list of them, sorted with util.sort_array_pair_by_first."""
    previous, new = list(range(first_new)), list(range(first_new, n))
    if unassigned:
        exclude = {i for i in range(n) if seqs[i] == 0}
        new = list(set(new).difference(exclude))
        pairs = list(itertools.product(new, new))
    else:
        pairs = [(i, j) for i, j in itertools.product(new, new) if double or i <= j] + \
                [(i, j) for i, j in itertools.product(new, previous) if double or i <= j] + \
                [(i, j) for i, j in itertools.product(previous, new) if double or i <= j]
    sizes = [seqs[i] * seqs[j] for i, j in pairs]
    sizes, pairs = util.sort_array_pair_by_first(sizes, pairs, q_largest_first=order != "ascending")
    query = "/w/U.Species%d.fa" if unassigned else "/w/Species%d.fa"
    return ["diamond %s /w/diamondDB%d /w/Blast%d_%d.txt" % (query % i, j, i, j) for i, j in pairs], sizes


def test_the_same_commands_in_the_same_order(monkeypatch):
    monkeypatch.setattr(sc.file_handler, "handler", Handler())
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, 12)
        first_new = rng.randint(0, n)
        seqs = {i: rng.choice([5, 7, 7, 10, 0]) for i in range(n)}      # (by species ID, as SequencesInfo)
        double, order, unassigned = rng.random() < 0.5, rng.choice(["descending", "ascending"]), rng.random() < 0.3
        options = types.SimpleNamespace(q_double_blast=double, cmd_order=order, search_program="diamond",
                                        score_matrix=None, gapopen=None, gapextend=None, dna=False)
        commands, sizes = sc.get_ordered_search_commands(
            types.SimpleNamespace(n_seqs_per_species=seqs),
            types.SimpleNamespace(i_first_new_species=first_new, n_sp_all=n), options, Caller(),
            n_genes_per_species=[seqs[i] for i in range(n)], q_new_species_unassigned_genes=unassigned)
        expected, expected_sizes = as_the_list(seqs, first_new, n, double, order, unassigned)
        assert list(commands) == expected and sizes.tolist() == expected_sizes and len(commands) == len(expected)
        if expected:
            assert commands[0] == expected[0]
            keep = sizes > 25
            assert list(commands.subset(keep)) == [c for c, k in zip(expected, keep) if k]
