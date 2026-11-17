"""blast_file_processor: each (query, hit) pair's highest score, as the earlier sort by pair and score gave."""

import numpy as np

from orthofinder.utils import blast_file_processor


def old_best_scores(I, J, S, n_seqs_j, q_check_for_self_hits):
    # (the earlier code: sorted by pair, then score, the last entry of each pair)
    keep = S > 0
    if q_check_for_self_hits:
        keep &= I != J
    I, J, S = I[keep], J[keep], S[keep]
    key = I * n_seqs_j + J
    order = np.lexsort((S, key))
    key = key[order]
    last = np.ones(key.size, dtype=bool)
    last[:-1] = key[1:] != key[:-1]
    return key[last] // n_seqs_j, key[last] % n_seqs_j, S[order][last]


def test_best_score_of_each_pair_as_before():
    rng = np.random.default_rng(0)
    for case in range(200):
        n_i, n_j = rng.integers(1, 30, 2)
        n = int(rng.integers(1, 400))
        I, J = rng.integers(0, n_i, n), rng.integers(0, n_j, n)
        # (few distinct scores: ties; some <= 0, dropped)
        S = rng.choice(np.r_[rng.random(5) * 100, [0.0, -1.0]], n) if case % 2 else rng.random(n) * 100 - 5
        self_hits = case % 3 == 0
        if self_hits:
            n_j = n_i
            J = rng.integers(0, n_i, n)
        got = blast_file_processor._bit_score_matrix(
            I, J, S, n_i, n_j, self_hits, 0, 0 if self_hits else 1, lambda *x: x, ("empty",))
        want = old_best_scores(I, J, S, n_j, self_hits)
        if want[0].size == 0:
            assert got == ("empty",)
            continue
        for g, w in zip(got, want):
            assert g.dtype == w.dtype and np.array_equal(g, w)
