"""The run report, WorkingDirectory/Run_Report/ (utils/run_report.py)."""

import multiprocessing as mp
import os

import pytest

from orthofinder.utils import file_handler, run_report


@pytest.fixture
def wd(tmp_path, monkeypatch):
    d = tmp_path / "WorkingDirectory"
    d.mkdir()
    (d / "SpeciesIDs.txt").write_text("0: Homo_sapiens.fa\n1: Mus_musculus.fa\n")
    h = file_handler._FileHandler()
    h.wd_current = str(d)
    monkeypatch.setattr(file_handler, "handler", h)
    run_report.start()
    return d / run_report.REPORT_DIR


def read(fn):
    return [line.split("\t") for line in fn.read_text().splitlines()]


def summary(d):
    return {(r[0], r[1]): r[2:] for r in read(d / run_report.SUMMARY_FN)[1:]}


def test_clean_run(wd):
    run_report.checked("tree.missing", 120)
    assert run_report.write() == []
    rows = read(wd / run_report.SUMMARY_FN)
    assert tuple(rows[0]) == run_report.SUMMARY_HEADER
    # every check, in run order; those not run "not checked"
    assert [tuple(r[:2]) for r in rows[1:]] == [(c.stage, c.text) for c in run_report.CHECKS]
    assert summary(wd)[("Gene trees", "Gene tree missing or empty")] == [
        run_report.RESULTS_MISSING, "0", "120", "", ""]
    assert summary(wd)[("Input", "Files ignored, not FASTA files")][1] == run_report.NOT_CHECKED
    assert os.listdir(wd) == [run_report.SUMMARY_FN]       # (no stage files)


def test_items_and_screen_line(wd):
    run_report.checked("tree.missing", 120)
    run_report.add("tree.missing", "OG0000012", detail="missing or empty (OG0000012.txt)")
    run_report.add("tree.missing", "OG0000003", detail="missing or empty (OG0000003.txt)")
    run_report.add("search.too_few_hits", "", species="0, 1", detail="1 hits")
    run_report.count("olog.polytomies_resolved", 5, 40)
    (line,) = run_report.write()
    assert "Gene tree missing or empty (2)" in line and "Summary.tsv" in line
    assert "polytom" not in line and "hits" not in line       # (not results missing: no screen line)
    s = summary(wd)
    assert s[("Gene trees", "Gene tree missing or empty")][1:] == [
        "2", "120", "No orthologs, duplications or HOGs of the orthogroup", "Gene_Trees.tsv"]
    assert s[("Orthologs", "Polytomies resolved with the species tree")][1:3] == ["5", "40"]
    assert s[("Orthologs", "Polytomies resolved with the species tree")][4] == "Resolved_Gene_Trees/Polytomies.tsv"
    assert read(wd / "Gene_Trees.tsv") == [list(run_report.ITEMS_HEADER),
                                           ["Gene tree missing or empty", "OG0000003", "", "missing or empty (OG0000003.txt)"],
                                           ["Gene tree missing or empty", "OG0000012", "", "missing or empty (OG0000012.txt)"]]
    # (species IDs as their names)
    assert read(wd / "Sequence_Searches.tsv")[1][2] == "Homo_sapiens, Mus_musculus"
    assert sorted(os.listdir(wd)) == ["Gene_Trees.tsv", "Sequence_Searches.tsv", "Summary.tsv"]


def _worker_add():
    run_report.add("tree.error", "OG0000007", detail="ValueError: x\tand\nmore")


def test_items_of_worker_processes(wd):
    run_report.checked("tree.error", 10)
    p = mp.get_context("fork").Process(target=_worker_add)
    p.start()
    p.join()
    (line,) = run_report.write()
    assert "Internal error analysing a gene tree (1)" in line
    assert read(wd / "Gene_Trees.tsv")[1] == ["Internal error analysing a gene tree", "OG0000007", "",
                                              "ValueError: x and more"]
    assert not (wd / run_report.PENDING_DIR).exists()


def test_pending_items_of_another_run_dropped(wd):
    (wd / run_report.PENDING_DIR).mkdir(parents=True)
    (wd / run_report.PENDING_DIR / "1_2_3.tsv").write_text("tree.error\tOG0000001\t\tx\n")
    assert run_report.write() == []
    assert summary(wd)[("Gene trees", "Internal error analysing a gene tree")][1] == run_report.NOT_CHECKED


def test_restart_keeps_the_checks_it_does_not_run(wd):
    # the first run: the input and the gene trees
    run_report.checked("input.not_fasta", 3)
    run_report.add("input.not_fasta", "notes.txt", detail="not a FASTA file extension")
    run_report.checked("tree.missing", 120)
    run_report.add("tree.missing", "OG0000012")
    run_report.write()
    # a restart from the gene trees: the gene trees again, now none missing
    run_report.start()
    run_report.checked("tree.missing", 118)
    assert run_report.write() == []
    s = summary(wd)
    assert s[("Input", "Files ignored, not FASTA files")][1:3] == ["1", "3"]
    assert s[("Gene trees", "Gene tree missing or empty")][1:3] == ["0", "118"]
    assert read(wd / "Input.tsv")[1][1] == "notes.txt"
    assert not (wd / "Gene_Trees.tsv").exists()


def test_no_report_without_a_working_directory(monkeypatch):
    monkeypatch.setattr(file_handler, "handler", file_handler._FileHandler())
    run_report.start()
    run_report.add("tree.error", "OG0000001")
    assert run_report.write() is None


def test_unknown_check():
    with pytest.raises(KeyError):
        run_report.add("no.such.check", "x")


def test_counts_of_several_batches(wd):
    # (e.g. the alignments of --assign, in two batches)
    run_report.count("aln.untrimmed", 2, 30, add=True)
    run_report.count("aln.untrimmed", 1, 400, add=True)
    run_report.checked("aln.no_dropped", 30, add=True)
    run_report.checked("aln.no_dropped", 400, add=True)
    run_report.write()
    s = summary(wd)
    assert s[("Alignments", "Trimming failed or unusable, alignment kept untrimmed")][1:3] == ["3", "430"]
    assert s[("Alignments", "Trimmed without the columns removed written (--keep-dropped)")][1:3] == ["0", "430"]
