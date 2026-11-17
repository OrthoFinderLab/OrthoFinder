"""The rows of a large tree sent in sorted chunks: the same files as a tree sent whole."""
import random
from collections import defaultdict

import numpy as np
import pytest

from orthofinder.orthology import stats, tree_analyser
from orthofinder.orthology.ortholog_writers import SortedRuns, partition_chunk, sorted_lines


def random_orthologs(n_species, n_genes, n_nodes, seed):
    """Random ortholog relationships (leaves_l, leaves_r, sus_l, sus_r) and gene names."""
    rng = random.Random(seed)
    names = {}
    for sp in range(n_species):
        for g in range(n_genes):
            # names of different lengths and with tabs-sorting characters, so that
            # rows sort differently with and without their newlines
            names["%d_%d" % (sp, g)] = rng.choice(["g", "g ", "gA", "g-"]) + str(rng.randrange(1000))

    def leaves(n):
        d = defaultdict(list)
        for sp in rng.sample(range(n_species), n):
            d[str(sp)] = [str(g) for g in rng.sample(range(n_genes), rng.randint(1, 3))]
        return d

    orthologs = []
    for _ in range(n_nodes):
        orthologs.append((leaves(rng.randint(1, n_species)), leaves(rng.randint(1, n_species)),
                          leaves(rng.randint(0, 2)), leaves(rng.randint(0, 2))))
    return orthologs, names


def test_sorted_rows_as_sorted_lines():
    rows = ["a\tb\n", "a\n", "a b\n", "B\n", "a\tb\n"]
    assert tree_analyser.sorted_rows(rows) == sorted_lines("".join(rows))


@pytest.mark.parametrize("chunk_bytes", [1, 500, 5000, 10 ** 9])
def test_chunks_merged_are_the_rows_sent_whole(tmp_path, chunk_bytes):
    n_species = 6
    orthologs, names = random_orthologs(n_species, 8, 60, seed=chunk_bytes)
    species_dict = {str(i): "Sp%d" % i for i in range(n_species)}
    species = list(range(n_species))

    olog, sus = [""] * n_species, [""] * n_species
    whole_counts = stats.NOrthoSp(n_species)
    pending = stats.PendingPairs(n_species)
    tree_analyser.get_lines_for_ortholog_files([(7, orthologs)], species_dict, species, names, True,
                                               olog, sus, pair_counts=pending)
    pending.add_to(whole_counts)

    runs = SortedRuns(str(tmp_path))
    n_chunks = [0]

    def flush(olog_texts, sus_texts):
        n_chunks[0] += 1
        for payload in partition_chunk(olog_texts, sus_texts, 2).values():
            for i, text in payload["olog_chunks"]:
                runs.add(7, ("o", i), text)
            for i, text in payload["xenolog_chunks"]:
                runs.add(7, ("x", i), text)

    olog2, sus2 = [""] * n_species, [""] * n_species
    chunk_counts = stats.NOrthoSp(n_species)
    pending = stats.PendingPairs(n_species, max_pairs=7)
    counts = tree_analyser.get_lines_for_ortholog_files(
        [(7, orthologs)], species_dict, species, names, True, olog2, sus2,
        pair_counts=pending, flush=flush, chunk_bytes=chunk_bytes)
    assert counts is None                   # (added to pair_counts)
    pending.add_to(chunk_counts)

    for a, b in zip(("n", "n_121", "n_12m", "n_m21", "n_m2m"), ("n", "n_121", "n_12m", "n_m21", "n_m2m")):
        assert np.array_equal(getattr(whole_counts, a), getattr(chunk_counts, b))

    if chunk_bytes == 10 ** 9:
        # below the limit: sent whole, as before
        assert n_chunks[0] == 0
        assert olog2 == olog and sus2 == sus
        return
    assert n_chunks[0] > 1
    assert olog2 == [""] * n_species and sus2 == [""] * n_species

    for kind, texts in (("o", olog), ("x", sus)):
        for i, text in enumerate(texts):
            merged = []
            if (kind, i) in runs.keys(7):
                runs.merge(7, (kind, i), merged.append)
            assert "".join(merged) == (sorted_lines(text) if text else "")
    runs.discard(7)
    assert runs.file.seek(0, 2) == 0     # nothing pending: the file is emptied
    runs.close()


@pytest.mark.parametrize("in_memory", [0, 10 ** 9])     # merged from disk, by one sort
def test_sorted_runs_merge_large_runs(tmp_path, monkeypatch, in_memory):
    from orthofinder.orthology import ortholog_writers
    monkeypatch.setattr(ortholog_writers, "RUN_READ_BYTES", 7)       # lines split across reads
    monkeypatch.setattr(ortholog_writers, "MERGED_WRITE_BYTES", 20)
    monkeypatch.setattr(ortholog_writers, "MERGE_IN_MEMORY_BYTES", in_memory)
    rng = random.Random(3)
    lines = ["OG1\t%s\tx%d" % (rng.choice("abcé"), rng.randrange(10 ** 6)) for _ in range(500)]
    runs = SortedRuns(str(tmp_path))
    for k in range(0, 500, 37):
        runs.add(1, ("o", 0), "\n".join(sorted(lines[k:k + 37])) + "\n")
    runs.add(2, ("o", 0), "other\n")
    out = []
    runs.merge(1, ("o", 0), out.append)
    assert "".join(out) == "\n".join(sorted(lines)) + "\n"
    runs.discard(1)
    assert runs.keys(2) == [("o", 0)]     # the other OG's runs are kept
    runs.close()


def test_add_pairs_as_add_pair():
    rng = np.random.default_rng(0)
    n_sp = 9
    pairs = [(int(a), int(b), int(c), int(d)) for a, b, c, d in zip(
        rng.integers(0, n_sp, 3000), rng.integers(0, n_sp, 3000),
        rng.integers(1, 4, 3000), rng.integers(1, 4, 3000)) if a != b]
    loop, vec = stats.NOrthoSp(n_sp), stats.NOrthoSp(n_sp)
    for p in pairs:
        loop.add_pair(*p)
    vec.add_pairs(pairs)
    for m in ("n", "n_121", "n_12m", "n_m21", "n_m2m"):
        assert np.array_equal(getattr(loop, m), getattr(vec, m))
        assert getattr(vec, m).dtype == np.uint64


def test_streamed_counts_as_a_worker_held():
    # (a worker's counts sent to the parent in blocks, StreamedCounts, sum to those it held in a NOrthoSp)
    rng = np.random.default_rng(1)
    n_sp = 7
    pairs = [(int(a), int(b), int(c), int(d)) for a, b, c, d in zip(
        rng.integers(0, n_sp, 5000), rng.integers(0, n_sp, 5000),
        rng.integers(1, 5, 5000), rng.integers(1, 5, 5000)) if a != b]
    held, parent = stats.NOrthoSp(n_sp), stats.NOrthoSp(n_sp)
    blocks = []
    streamed = stats.StreamedCounts(blocks.append)
    streamed.MAX_PAIRS = 300
    dense = stats.NOrthoSp(n_sp)       # (a large tree's counts, PendingPairs.dense)
    for i, p in enumerate(pairs):
        if i % 3 == 0:
            held.add_pair(*p)
            streamed.add_pair(*p)
        elif i % 3 == 1:
            held.add_pairs([p])
            streamed.add_pairs([p])
        else:
            dense.add_pair(*p)
    held += dense
    streamed += dense
    streamed.flush()
    assert len(blocks) > 2
    for block in blocks:
        parent.add_block(block)
    for m in ("n", "n_121", "n_12m", "n_m21", "n_m2m"):
        assert np.array_equal(getattr(held, m), getattr(parent, m))
