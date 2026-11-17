"""tools/species_tree_alignment.py writes the same files as the run that inferred the species tree."""

import filecmp
import glob
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.slow
def test_species_tree_alignment_tool_reproduces_the_run(tmp_path):
    fasta_dir = tmp_path / "ExampleData"
    fasta_dir.mkdir()
    for fn in glob.glob(os.path.join(ROOT, "ExampleData", "*.faa")):
        shutil.copy(fn, fasta_dir)
    subprocess.run([sys.executable, os.path.join(ROOT, "orthofinder.py"), "-f", str(fasta_dir), "-t", "2", "-n", "t"],
                   check=True, capture_output=True)
    results = fasta_dir / "OrthoFinder" / "Results_t"
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "species_tree_alignment.py"), str(results), "-o", str(out)],
                   check=True, capture_output=True)
    assert filecmp.cmp(results / "WorkingDirectory" / "Alignments_ids" / "SpeciesTreeAlignment.fa",
                       out / "SpeciesTreeAlignment.fa", shallow=False)
    with open(results / "Species_Tree" / "Orthogroups_for_concatenated_alignment.txt") as a, \
            open(out / "Orthogroups_for_concatenated_alignment.txt") as b:
        assert sorted(a) == sorted(b)


@pytest.mark.parametrize("tree_program", ["fasttree", "raxml"])
def test_orthogroups_numbered_as_in_the_run(tmp_path, monkeypatch, tree_program):
    # (a run keeps every orthogroup, whatever the tree program: each one's index is its number, the
    # name of its alignment. Dropping the small ones first numbered the others wrongly)
    from tools import species_tree_alignment as sta
    ogs = [{"0_0", "1_0"}] + [{"%d_%d" % (sp, i) for sp in range(5)} for i in range(1, 6)]
    (tmp_path / "Orthogroups").mkdir()
    align_dir = tmp_path / "WorkingDirectory" / "Alignments_ids"
    align_dir.mkdir(parents=True)
    for i in range(len(ogs)):
        (align_dir / ("OG%07d.fa" % i)).write_text("".join(">%s\nACGT\n" % g for g in sorted(ogs[i])))
    monkeypatch.setattr(sta.mcl, "get_predicted_ogs", lambda fn: ogs)
    monkeypatch.setattr(sta, "run_files", lambda d: ("x", [str(tmp_path / "WorkingDirectory")],
                                                     {"tree_program": tree_program, "min_seq": 4}))
    used = []
    monkeypatch.setattr(sta.concatenated_alignment, "create_concatenated_alignment",
                        lambda i_ogs, *args, **kwargs: used.extend(i_ogs))
    assert sta.main([str(tmp_path), "-o", str(tmp_path / "out")]) == 0
    assert sorted(used) == [1, 2, 3, 4, 5]
