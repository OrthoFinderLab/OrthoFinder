"""--assign: the orthogroups' positions after single-gene orthogroups are removed, and the gene tree files of the first pass (issue #99)."""

from orthofinder.mcl_orthogroups import assign


def test_restart_at_the_first_removed_orthogroup():
    # core orthogroups (positions 0-4): 3 and 4 are single genes that a clade-specific
    # orthogroup takes, so they are removed and the clade-specific ones move down
    ogs = [{"0_1", "0_2"}, {"0_3", "1_3"}, {"0_4"}, {"0_5"}, {"1_6"}]
    clade = [[{"0_5", "1_6", "2_1"}, {"2_2", "2_3"}]]
    _, restart = assign.write_all_orthogroups(ogs, {}, clade, restart_index=5, write=False)
    assert ogs == [{"0_1", "0_2"}, {"0_3", "1_3"}, {"0_4"}, {"0_5", "1_6", "2_1"}, {"2_2", "2_3"}]
    # (from position 3 the first pass's files are of other orthogroups)
    assert restart == 3


def test_restart_unchanged_without_removals():
    ogs = [{"0_1", "0_2"}, {"0_3"}]
    _, restart = assign.write_all_orthogroups(ogs, {}, [[{"2_1", "2_2"}]], restart_index=2, write=False)
    assert restart == 2 and len(ogs) == 3


def test_restart_before_a_later_removal():
    # a removal after the restart position does not move it
    ogs = [{"0_1", "0_2"}, {"0_3", "0_4"}]
    _, restart = assign.write_all_orthogroups(ogs, {}, [[{"2_1"}], [{"2_1", "1_1"}]], restart_index=1,
                                              write=False)
    assert restart == 1


def test_new_species_genes_added():
    ogs = [{"0_1"}, {"0_2", "1_2"}]
    assign.write_all_orthogroups(ogs, {0: {"2_7"}}, [], write=False)
    assert ogs == [{"0_1", "2_7"}, {"0_2", "1_2"}]


def test_remove_gene_tree_files(tmp_path):
    dirs = [tmp_path / "Sequences_ids", tmp_path / "Trees_ids", tmp_path / "missing"]
    for d in dirs[:2]:
        d.mkdir()
        for i in (0, 2, 3, 12):
            (d / ("OG%07d.fa" % i)).write_text(">x\n")
        (d / "notes.txt").write_text("kept\n")
    (dirs[1] / "OG0000005.txt.raxml.log").write_text("")      # (a program's other files of an orthogroup)
    n = assign.remove_gene_tree_files(3, [str(d) for d in dirs])
    assert n == 5
    for d in dirs[:2]:
        assert sorted(p.name for p in d.iterdir()) == ["OG0000000.fa", "OG0000002.fa", "notes.txt"]


def _clusters(wd, name):
    fn = wd / ("clusters_OrthoFinder_%s.txt_id_pairs.txt" % name)
    fn.write_text("")
    return fn


def test_core_clusters_file_recorded(tmp_path):
    # orthogroups inferred again in the same results (-b): the run_state's, not the first by name
    from orthofinder.utils import run_state
    wd = tmp_path / "WorkingDirectory"
    wd.mkdir()
    _clusters(wd, "I1.5")
    new = _clusters(wd, "I1.5_1")
    _clusters(wd, "unassigned_clade_0_I1.2")
    run_state.record_file(str(tmp_path), "orthogroups", str(new))
    assert assign.core_clusters_file(str(wd) + "/") == str(new)


def test_core_clusters_file_most_recent_without_a_record(tmp_path, capsys):
    import os
    wd = tmp_path / "WorkingDirectory"
    wd.mkdir()
    old = _clusters(wd, "I1.5")
    new = _clusters(wd, "I1.5_1")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert assign.core_clusters_file(str(wd) + "/") == str(new)
    assert "most recent" in capsys.readouterr().out


def test_one_clusters_file(tmp_path):
    wd = tmp_path / "WorkingDirectory"
    wd.mkdir()
    only = _clusters(wd, "I1.5")
    assert assign.core_clusters_file(str(wd) + "/") == str(only)


def test_profiles_removed_with_the_orthogroup_files(tmp_path):
    # orthogroups inferred again (-b): --assign's profiles of the old ones are not reused
    from orthofinder.utils import file_handler
    h = file_handler._FileHandler()
    h.rd1 = str(tmp_path) + "/"
    h.wd_current = str(tmp_path / "WorkingDirectory") + "/"
    (tmp_path / "WorkingDirectory" / "Sequences_ids").mkdir(parents=True)
    for name in ("profile_sequences.10_kmeans.fa.dmnd", "profile_sequences.10_km.fa", "SpeciesIDs.txt"):
        (tmp_path / "WorkingDirectory" / name).write_text("")
    h.remove_orthogroup_files()
    assert sorted(p.name for p in (tmp_path / "WorkingDirectory").iterdir()) == ["SpeciesIDs.txt"]
