"""hog_orthogroups.update_trees.threaded_reader: the reads submitted are bounded."""
import queue
import threading
import time

from orthofinder.hog_orthogroups import update_trees


def test_reads_submitted_ahead_are_bounded(monkeypatch):
    gate = threading.Event()
    taken = [0]

    def blocked_read(unique_og, *args, **kwargs):
        gate.wait()
        return (unique_og,)

    def ogs():
        for i in range(10000):
            taken[0] += 1
            yield ("OG%07d" % i,)

    monkeypatch.setattr(update_trees, "read_files", blocked_read)
    read_queue, report_queue, stop_event = queue.Queue(), queue.Queue(), threading.Event()
    n_threads = 3
    t = threading.Thread(target=update_trees.threaded_reader,
                         args=(read_queue, ogs(), {}, {}, {}, n_threads, stop_event, report_queue))
    t.start()
    time.sleep(0.5)
    # (those running, those waiting, and the one held for a free slot)
    assert taken[0] <= update_trees.READS_PER_THREAD * n_threads + 1
    gate.set()
    t.join(30)
    assert not t.is_alive() and read_queue.qsize() == 10000 and taken[0] == 10000
