"""utils/matrices.py: blocks stored and held as CompactCSR, given back as the same CSR."""
import pickle

import numpy as np
from scipy import sparse

from orthofinder.utils import matrices


def same(a, b):
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data))


def test_compact_gives_back_the_same_csr():
    rng = np.random.default_rng(0)
    for n_rows, n_cols, density in ((1, 1, 1.0), (50, 30, 0.05), (2000, 7, 0.001), (5, 5, 0.0)):
        m = sparse.random(n_rows, n_cols, density=density, format="csr", random_state=1)
        assert same(matrices.compact(m).tocsr(), m)
    # explicit zeros and unsorted columns kept as they are
    m = sparse.csr_matrix((np.array([0.0, 2.0, 3.0]), np.array([2, 0, 1]), np.array([0, 0, 2, 2, 3])), shape=(4, 3))
    c = pickle.loads(pickle.dumps(matrices.compact(m)))
    assert same(c.tocsr(), m) and c.nnz == 3 and c.get_shape() == (4, 3)
    assert len(c.rows) == 2         # (only the rows with entries)


def test_pack_files_round_trip(tmp_path):
    d = str(tmp_path) + "/"
    blocks = [sparse.random(40, k, density=0.1, format="csr", random_state=k) for k in (3, 9, 1)]
    matrices.dump_matrix_array("B", blocks, 0, d)
    for j, b in enumerate(blocks):
        assert same(matrices.as_csr(matrices.load_matrix("B", 0, j, d)), b)
    seqs_info = type("S", (), {"n_species": 3})()
    assert all(same(matrices.as_csr(x), b) for x, b in zip(matrices.load_matrix_array("B", seqs_info, 0, d), blocks))
