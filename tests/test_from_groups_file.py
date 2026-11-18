"""-fg given an orthogroups file (clusters_OrthoFinder_*.txt_id_pairs.txt) rather than a results directory."""

import pytest

from orthofinder.run import previous_run
from orthofinder.utils import run_state


def results(tmp_path, names=("I1.5", "I3.0"), recorded="I3.0"):
    wd = tmp_path / "Results_x" / "WorkingDirectory"
    wd.mkdir(parents=True)
    fns = {}
    for name in names:
        fns[name] = wd / ("clusters_OrthoFinder_%s.txt_id_pairs.txt" % name)
        fns[name].write_text("")
    if recorded:
        run_state.record_file(str(tmp_path / "Results_x"), "orthogroups", str(fns[recorded]))
    return tmp_path / "Results_x", fns


def test_state_found_from_a_clusters_file(tmp_path):
    results_dir, fns = results(tmp_path)
    found, state = run_state.find(str(fns["I1.5"]))
    assert found.rstrip("/") == str(results_dir) and state["files"]["orthogroups"].endswith("I3.0.txt_id_pairs.txt")


def test_other_orthogroups(tmp_path):
    results_dir, fns = results(tmp_path)
    state_dir, state = run_state.find(str(fns["I1.5"]))
    # another file than the run's: its files of each orthogroup are of other genes
    assert previous_run.other_orthogroups(str(fns["I1.5"]), state_dir, state)
    assert not previous_run.other_orthogroups(str(fns["I3.0"]), state_dir, state)
    # a directory: the run's own orthogroups
    assert not previous_run.other_orthogroups(str(results_dir), state_dir, state)


def test_other_orthogroups_without_a_record(tmp_path):
    _, fns = results(tmp_path, recorded=None)
    assert previous_run.other_orthogroups(str(fns["I1.5"]), None, {})


def test_given_clusters_file_checked(tmp_path):
    fn = tmp_path / "Orthogroups.tsv"
    fn.write_text("")
    with pytest.raises(SystemExit):
        previous_run.given_clusters_file(str(fn))
    ok = tmp_path / "clusters_OrthoFinder_I1.5.txt_id_pairs.txt"
    ok.write_text("")
    assert previous_run.given_clusters_file(str(ok)) == str(ok)


def test_old_locator_given_a_clusters_file(tmp_path):
    # (results without run_state.json): rooted at the file's working directory, not at the file
    _, fns = results(tmp_path, recorded=None)
    (fns["I1.5"].parent / "Species0.fa").write_text(">0_0\nM\n")       # (what makes it a working directory)

    class Options:
        q_start_from_groups = True
        q_start_from_trees = False
        q_start_from_species_trees = False
        q_start_from_blast = False
        q_start_from_fasta = False
        species_tree_fn = None

    pfl = previous_run.PreviousFilesLocatorOld(Options(), str(fns["I1.5"]))
    assert pfl.get_home_for_results() == str(fns["I1.5"].parent) + "/OrthoFinder/"
    wd, clusters = pfl.get_start_from_ogs()
    assert clusters == str(fns["I1.5"])
