"""tools/primary_transcript.py, tools/ncbi_primary_transcript.py and tools/orthogroup_gene_count.py."""
import os
import zipfile

from tools import ncbi_primary_transcript as ncbi
from tools import orthogroup_gene_count as ogc
from tools import primary_transcript as pt


def test_primary_transcripts_of_files_with_the_same_name(tmp_path, capsys):
    # (each species' protein.faa in a directory of its own, as NCBI downloads are: one output each)
    for sp, text in (("sp1", ">g1.1\nMKV\n>g1.2\nMKVLL\n"), ("sp2", ">h1.1\nMAAA\n")):
        (tmp_path / "in" / sp).mkdir(parents=True)
        (tmp_path / "in" / sp / "protein.faa").write_text(text)
    (tmp_path / "in" / "other.fa").write_text(">x.1\nMK\n")
    with zipfile.ZipFile(tmp_path / "in" / "arch.zip", "w") as zf:
        zf.write(tmp_path / "in" / "sp1" / "protein.faa", "sp1/protein.faa")
    pt.main([str(tmp_path / "in"), "last_dot"])
    out = tmp_path / "in" / "primary_transcripts"
    assert sorted(os.listdir(out)) == ["arch_sp1_protein.faa", "other.fa", "sp1_protein.faa", "sp2_protein.faa"]
    assert (out / "sp1_protein.faa").read_text() == ">g1.2\nMKVLL\n"
    assert (out / "sp2_protein.faa").read_text() == ">h1.1\nMAAA\n"
    assert capsys.readouterr().out.count("Processing FASTA") == 4         # (each file once)


def test_ncbi_genes_are_loci_not_symbols(tmp_path):
    # two loci with the same symbol: two genes; a locus without a symbol by its locus_tag
    gff = tmp_path / "g.gff"
    gff.write_text(
        "chr1\tRefSeq\tCDS\t1\t9\t.\t+\t0\tID=cds-XP_1;Dbxref=GeneID:101,Genbank:XP_1;gene=ABC;protein_id=XP_1\n"
        "chr1\tRefSeq\tCDS\t1\t9\t.\t+\t0\tID=cds-XP_3;Dbxref=GeneID:101;gene=ABC;protein_id=XP_3\n"
        "chr2\tRefSeq\tCDS\t1\t9\t.\t+\t0\tID=cds-XP_2;Dbxref=GeneID:202;gene=ABC;protein_id=XP_2\n"
        "chr3\tRefSeq\tCDS\t1\t9\t.\t+\t0\tID=cds-XP_4;locus_tag=LT9;protein_id=XP_4\n")
    longest = ncbi.get_longest_transcripts({"XP_1": "MKV", "XP_2": "MKVLL", "XP_3": "MKVLLLL", "XP_4": "M"},
                                           ncbi.parse_gff(str(gff)))
    assert sorted(p for p, _, _ in longest.values()) == ["XP_2", "XP_3", "XP_4"]
    ncbi.write_longest_fasta(str(tmp_path / "o.fa"), longest)
    assert (tmp_path / "o.fa").read_text() == ">XP_3 gene=ABC\nMKVLLLL\n>XP_2 gene=ABC\nMKVLL\n>XP_4 gene=LT9\nM\n"


def test_gene_count_directory_twice(tmp_path):
    # (its own output tables are not converted again)
    (tmp_path / "N0.tsv").write_text("HOG\tOG\tGene Tree Parent Clade\tA\tB\nN0.H1\tOG1\tn1\ta1\t\n")
    (tmp_path / "Orthogroups.tsv").write_text("Orthogroup\tA\tB\nOG1\ta1\t\n")
    for _ in range(2):
        ogc.convert_directory(str(tmp_path))
    assert sorted(os.listdir(tmp_path / "GeneCountTables")) == ["N0.GeneCount.tsv", "Orthogroups.GeneCount.tsv"]
    assert (tmp_path / "GeneCountTables" / "N0.GeneCount.tsv").read_text().splitlines()[1] == "N0.H1\tOG1\tn1\t1\t0\t1"


def test_output_names_distinct_as_written():
    # a/x.fa and b/x.fa become a_x.fa and b_x.fa: a file named a_x.fa itself must not be overwritten
    names = pt.output_names([("a", "x.fa"), ("b", "x.fa"), ("a_x.fa",)])
    assert len(set(names)) == 3 and "b_x.fa" in names and "a_x.fa" in names
    # names that stay the same with every directory added: a number
    names = pt.output_names([("a", "x.fa"), ("a_x.fa",), ("a_x.fa",)])
    assert len(set(names)) == 3
    # unchanged when the names are distinct already
    assert pt.output_names([("sp1", "protein.faa"), ("sp2", "protein.faa"), ("other.fa",)]) == \
        ["sp1_protein.faa", "sp2_protein.faa", "other.fa"]


def test_output_names_of_many_files_quickly():
    # (the path endings counted once, not each file against every other)
    import time
    parts = [("sp%d" % i, "protein.faa") for i in range(20000)]
    start = time.monotonic()
    names = pt.output_names(parts)
    assert time.monotonic() - start < 5 and len(set(names)) == 20000


def test_primary_transcripts_do_not_overwrite_a_file(tmp_path):
    for rel, text in (("a/x.fa", ">a.1\nMK\n"), ("b/x.fa", ">b.1\nMA\n"), ("a_x.fa", ">c.1\nMC\n")):
        (tmp_path / "in" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "in" / rel).write_text(text)
    pt.main([str(tmp_path / "in"), "last_dot"])
    out = tmp_path / "in" / "primary_transcripts"
    assert len(os.listdir(out)) == 3
    assert sorted((out / fn).read_text() for fn in os.listdir(out)) == [">a.1\nMK\n", ">b.1\nMA\n", ">c.1\nMC\n"]


def test_gene_count_never_overwrites_its_input(tmp_path, capsys):
    import pytest
    fn = tmp_path / "Orthogroups.tsv"
    fn.write_text("Orthogroup\tA\tB\nOG1\ta1, a2\tb1\n")
    before = fn.read_text()
    with pytest.raises(ValueError, match="output file is the input"):
        ogc.convert_tabular_gene_counts(str(fn), str(fn))
    with pytest.raises(SystemExit):
        ogc.main([str(fn), "-o", str(fn)])
    assert fn.read_text() == before and sorted(os.listdir(tmp_path)) == ["Orthogroups.tsv"]


def test_gene_count_total_column(tmp_path):
    (tmp_path / "Orthogroups.tsv").write_text("Orthogroup\tA\tB\nOG1\ta1, a2\tb1\nOG2\t\tb2\n")
    ogc.convert_tabular_gene_counts(str(tmp_path / "Orthogroups.tsv"), str(tmp_path / "c.tsv"))
    assert (tmp_path / "c.tsv").read_text() == "Orthogroup\tA\tB\tTotal\nOG1\t2\t1\t3\nOG2\t0\t1\t1\n"
    assert sorted(os.listdir(tmp_path)) == ["Orthogroups.tsv", "c.tsv"]      # (no temporary file left)
