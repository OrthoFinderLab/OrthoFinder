"""Trimming of the MSAs: OrthoFinder's own, trimming programs (--trim), the dropped columns (--keep-dropped)."""

import json
import os
import queue
import random
import sys
import threading
from pathlib import Path

import ete4
import numpy as np
import pytest

from orthofinder.gene_tree import trim
from orthofinder.hog_orthogroups import update_trees
from orthofinder.run import process_args
from orthofinder.utils import program_caller

from tools import trim_msa

ROOT = Path(__file__).resolve().parents[1]


def write_alignment(fn, n_seqs=15, n_cols=700, seed=1):
    """An alignment with every seventh column mostly gaps."""
    rng = random.Random(seed)
    ancestor = [rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n_cols)]
    with open(fn, "w") as f:
        for i in range(n_seqs):
            seq = "".join("-" if rng.random() < (0.95 if c % 7 == 0 else 0.02) else a for c, a in enumerate(ancestor))
            f.write(">%d_%d extra words\n%s\n" % (i % 4, i, seq))
    return dict(trim.read_fasta(fn))


def untrimmed(fn):
    """The alignment fn as it is: what a program's output that cannot be used leaves."""
    return trim.read_fasta(str(fn))


def fake_program(tmp_path, name, script):
    """A trimming program: a Python script run with this Python."""
    fn = tmp_path / (name + ".py")
    fn.write_text(script)
    return "%s %s" % (sys.executable, fn)


# A program that keeps the even columns and writes the odd ones to argv[3] (if given), as
# trimAl -complementary is a second command: argv[3] "-" writes the odd ones to argv[2]
KEEP_EVEN = """
import sys
records, name = [], None
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith(">"):
        records.append([line[1:], ""])
    elif line:
        records[-1][1] += line
def write(fn, start):
    with open(fn, "w") as f:
        for h, s in records:
            f.write(">%s\\n%s\\n" % (h.split()[0], s[start::2]))
if len(sys.argv) > 3 and sys.argv[3] == "-":
    write(sys.argv[2], 1)
else:
    write(sys.argv[2], 0)
    if len(sys.argv) > 3:
        write(sys.argv[3], 1)
"""


# -------------------------------------------------- OrthoFinder's trimming ----

def test_builtin_dropped_columns_are_the_rest(tmp_path):
    seqs = write_alignment(tmp_path / "in.fa")
    n_removed = trim.run_in_process(str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10, 0.1, 500, 0.75,
                                    str(tmp_path / "dropped.fa"))
    kept, dropped = dict(trim.read_fasta(tmp_path / "out.fa")), dict(trim.read_fasta(tmp_path / "dropped.fa"))
    assert n_removed == len(next(iter(dropped.values()))) > 0
    assert set(kept) == set(dropped) == set(seqs)
    for h, seq in seqs.items():
        # the columns kept and removed are the alignment's, each once
        assert sorted(kept[h] + dropped[h]) == sorted(seq)
        assert len(kept[h]) + len(dropped[h]) == len(seq)


def test_builtin_untrimmed_writes_no_dropped_file(tmp_path):
    write_alignment(tmp_path / "in.fa", n_cols=300)        # below the minimum length (500)
    (tmp_path / "dropped.fa").write_text(">old\n-\n")      # of an earlier run
    assert trim.run_in_process(str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10, 0.1, 500, 0.75,
                               str(tmp_path / "dropped.fa")) == 0
    assert not (tmp_path / "dropped.fa").exists()
    assert (tmp_path / "out.fa").read_text() == (tmp_path / "in.fa").read_text()


def test_builtin_in_place_untrimmed_unchanged(tmp_path):
    # (as OrthoFinder trims: the output is the input file) too short to trim: nothing added to it
    fn = tmp_path / "in.fa"
    with open(fn, "w") as f:
        f.write("".join(">%d_%d\n%s\n" % (i % 4, i, "ACDEFGHIKL" * 30) for i in range(15)))
    before = fn.read_bytes()
    assert trim.run_in_process(str(fn), str(fn), 10, 0.1, 500, 0.75) == 0
    assert fn.read_bytes() == before


def test_builtin_every_column_kept_rewritten(tmp_path):
    # no column with >90% gaps: rewritten as before (lines of 80, "*" as a gap), no dropped file
    fn = tmp_path / "in.fa"
    seqs = ["ACDEFGHIKL" * 70 + ("*" if i == 0 else "A") for i in range(15)]
    with open(fn, "w") as f:
        f.write("".join(">%d_%d\n%s\n" % (i % 4, i, seq) for i, seq in enumerate(seqs)))
    assert trim.run_in_process(str(fn), str(fn), 10, 0.1, 500, 0.75, str(tmp_path / "d.fa")) == 0
    expected = [seq.replace("*", "-") for seq in seqs]
    assert fn.read_text() == "".join(">%d_%d\n%s" % (i % 4, i, "".join(seq[j:j + 80] + "\n"
                                     for j in range(0, len(seq), 80))) for i, seq in enumerate(expected))
    assert not (tmp_path / "d.fa").exists()


def test_builtin_without_dropped_unchanged(tmp_path):
    write_alignment(tmp_path / "in.fa")
    trim.run_in_process(str(tmp_path / "in.fa"), str(tmp_path / "a.fa"), 10, 0.1, 500, 0.75)
    trim.run_in_process(str(tmp_path / "in.fa"), str(tmp_path / "b.fa"), 10, 0.1, 500, 0.75, str(tmp_path / "d.fa"))
    assert (tmp_path / "a.fa").read_text() == (tmp_path / "b.fa").read_text()


# ---------------------------------------------------- trimming programs ----

def test_trim_program_entries():
    paired = trim.TrimProgram("t", {"cmd_line_kept": "t INPUT OUTPUT > STDOUT",
                                    "cmd_line_dropped": "t INPUT OUTPUT -c", "output_mode": "paired"})
    assert paired.output_mode == "paired" and paired.writes_dropped()
    cmds = paired.commands("/a/in.fa", "/a/k.fa", "/a/d.fa")
    assert [c for c, _ in cmds] == ["t /a/in.fa /a/k.fa > %s" % os.devnull, "t /a/in.fa /a/d.fa -c"]
    assert len(paired.commands("/a/in.fa", "/a/k.fa")) == 1       # no dropped columns: one command
    complement = trim.TrimProgram("c", {"cmd_line": "c INPUT -o OUTPUT", "complement_suffix": ".complement"})
    assert complement.output_mode == "single_with_complement"
    assert complement.dropped_output("/a/k.fa", None) == "/a/k.fa.complement"
    assert not trim.TrimProgram("b", {"cmd_line": "b INPUT OUTPUT"}).writes_dropped()
    for entry in ({"cmd_line": "x", "output_mode": "self_generated"}, {"cmd_line": "x", "output_type": "mask"},
                  {"cmd_line": "x", "output_mode": "paired"}, {"output_mode": "single"}):
        with pytest.raises(trim.InvalidTrimEntry):
            trim.TrimProgram("x", entry)


@pytest.mark.parametrize("mode", ["paired", "single_with_complement"])
def test_program_kept_and_dropped(tmp_path, mode):
    seqs = write_alignment(tmp_path / "in.fa")
    cmd = fake_program(tmp_path, "keep_even", KEEP_EVEN)
    entry = ({"cmd_line": cmd + " INPUT OUTPUT", "cmd_line_dropped": cmd + " INPUT OUTPUT -"} if mode == "paired"
             else {"cmd_line": cmd + " INPUT OUTPUT OUTPUT.complement", "complement_suffix": ".complement"})
    program = trim.TrimProgram("keep_even", entry)
    aln = tmp_path / "in.fa"
    assert trim.run_program(program, str(aln), str(aln), 10, str(tmp_path / "d.fa")) == (None, None, False)
    kept, dropped = trim.read_fasta(aln), dict(trim.read_fasta(tmp_path / "d.fa"))
    # the input's headers, in its order (the program wrote only their first word)
    assert [h for h, _ in kept] == list(seqs)
    assert all(s == seqs[h][0::2] and dropped[h] == seqs[h][1::2] for h, s in kept)
    assert sorted(os.listdir(tmp_path)) == sorted(["in.fa", "d.fa", "keep_even.py"])   # no files left


def test_program_complement_removed_without_dropped(tmp_path):
    write_alignment(tmp_path / "in.fa")
    cmd = fake_program(tmp_path, "keep_even", KEEP_EVEN)
    program = trim.TrimProgram("k", {"cmd_line": cmd + " INPUT OUTPUT OUTPUT.c", "complement_suffix": ".c"})
    assert trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10) == (None, None, False)
    assert sorted(os.listdir(tmp_path)) == ["in.fa", "keep_even.py", "out.fa"]


@pytest.mark.parametrize("script, reason", [
    ("import sys; sys.exit(3)", "exited with code 3"),
    ("pass", "no output"),
    # a sequence left out (as trimAl does with one that only has gaps)
    ("import sys; t = open(sys.argv[1]).read().split('>')[1:]; open(sys.argv[2], 'w').write(''.join('>' + r for r in t[1:]))",
     "1 of the 15 sequences are not in its output"),
    # a sequence added (e.g. a consensus)
    ("import sys; open(sys.argv[2], 'w').write(open(sys.argv[1]).read() + '>consensus\\nAAAAA\\n')",
     "its output has 1 sequence(s) not in the input, e.g. consensus"),
    # a sequence left with no residues
    ("import sys\nt = [r.split('\\n', 1) for r in open(sys.argv[1]).read().split('>')[1:]]\n"
     "open(sys.argv[2], 'w').write(''.join('>%s\\n%s\\n' % (h, '-' * 5 if i == 0 else 'A' * 5) for i, (h, s) in enumerate(t)))",
     "no residues"),
])
def test_program_output_not_used(tmp_path, script, reason):
    # the alignment kept untrimmed, no columns removed
    seqs = write_alignment(tmp_path / "in.fa")
    expected = untrimmed(tmp_path / "in.fa")
    program = trim.TrimProgram("bad", {"cmd_line": fake_program(tmp_path, "bad", script) + " INPUT OUTPUT"})
    note, warning, q_failed = trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "in.fa"), 10,
                                               str(tmp_path / "d.fa"))
    assert reason in note and warning is None and q_failed is (reason != "no residues")
    assert note.endswith(", so the alignment was kept untrimmed")
    assert trim.read_fasta(tmp_path / "in.fa") == expected and not (tmp_path / "d.fa").exists()
    assert [h for h, _ in expected] == list(seqs)


def test_program_methodthread_outside_a_run(tmp_path):
    # (the dependency test and trim_msa: one thread; in a run command_runner fills it)
    write_alignment(tmp_path / "in.fa")
    # (the argument checked, then left out: not a file for the columns removed)
    cmd = fake_program(tmp_path, "keep_even", "import sys; assert sys.argv[3] == '1'; del sys.argv[3]\n" + KEEP_EVEN)
    program = trim.TrimProgram("k", {"cmd_line": cmd + " INPUT OUTPUT METHODTHREAD"})
    assert trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10) == (None, None, False)


def test_temp_files_of_a_stopped_run_removed(tmp_path):
    names = ["OG0000001.fa.trim_kept.fa", "OG0000001.fa.trim_kept.fa.complement", "OG0000001.fa.trim_dropped.fa",
             "OG0000001.fa.trim_status0", "OG0000001.fa.trim_stderr1"]
    for name in names + ["OG0000001.fa", "OG0000002.fa"]:
        (tmp_path / name).write_text("x")
    trim.remove_temp_files(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["OG0000001.fa", "OG0000002.fa"]


def test_program_not_run_on_few_sequences(tmp_path):
    write_alignment(tmp_path / "in.fa", n_seqs=8)
    program = trim.TrimProgram("bad", {"cmd_line": fake_program(tmp_path, "bad", "import sys; sys.exit(3)") + " INPUT OUTPUT"})
    assert trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10) == (None, None, False)
    assert (tmp_path / "out.fa").read_text() == (tmp_path / "in.fa").read_text()


# --------------------------------------------------- config and options ----

@pytest.mark.parametrize("config", ["src/orthofinder/run/config.json", "user_config.json"])
def test_config_files_have_trimming_programs(config):
    prog_caller = program_caller.ProgramCaller(str(ROOT / config))
    assert {"trimal", "clipkit", "bmge"} <= set(prog_caller.list_trim_methods())
    assert prog_caller.get_trim_program("trimal").writes_dropped()
    assert prog_caller.get_trim_program("clipkit").writes_dropped()
    assert not prog_caller.get_trim_program("bmge").writes_dropped()


def test_trimmiy_entries_can_be_used(tmp_path):
    # those of trimmiy's config.json with an output mode OrthoFinder reads; the others are skipped
    config = {
        "trimal": {"program_type": "trim", "cmd_line_kept": "trimal -in INPUT -out OUTPUT -automated1 -sgc > STDOUT",
                   "cmd_line_dropped": "trimal -in INPUT -out OUTPUT -automated1 -sgc -complementary > STDOUT",
                   "output_mode": "paired", "output_type": "alignment", "program_category": "compare_with"},
        "clipkit": {"program_type": "trim", "cmd_line": "clipkit INPUT -o OUTPUT -c > STDOUT",
                    "output_mode": "single_with_complement", "complement_suffix": ".complement",
                    "output_type": "alignment", "program_category": "compare_with"},
        "alifilter": {"program_type": "trim", "cmd_line": "AliFilter -i INPUT -o OUTPUT -k mask > STDOUT",
                      "output_mode": "single", "output_type": "mask"},
        "noisy": {"program_type": "trim", "cmd_line": "noisy --seqtype P INPUT > STDOUT", "output_mode": "self_generated"},
    }
    (tmp_path / "c.json").write_text(json.dumps(config))
    assert program_caller.ProgramCaller(str(tmp_path / "c.json")).list_trim_methods() == ["trimal", "clipkit"]


def read(tmp_path, *args):
    prog_caller = process_args.program_caller_with_configs([])
    return process_args._read_options(["-f", str(tmp_path)] + list(args), prog_caller)[0]


def test_trim_options(tmp_path):
    options = read(tmp_path)
    assert options.q_trim and options.trim_program == "clipkit" and not options.keep_dropped     # (the default)
    options = read(tmp_path, "--trim", "clipkit", "--keep-dropped")
    assert options.q_trim and options.trim_program == "clipkit" and options.keep_dropped
    for args in (["--trim", "none"], ["-z"], ["--trim", "none", "-z"]):
        options = read(tmp_path, *args)
        assert not options.q_trim
    options = read(tmp_path, "--trim", "orthofinder", "--keep-dropped")
    assert options.q_trim and options.trim_program is None


@pytest.mark.parametrize("args, message", [
    (["--trim", "trimal", "-z"], "-z (no trimming) cannot be used with --trim trimal"),
    (["--keep-dropped", "-z"], "the MSAs are not trimmed"),
    (["--keep-dropped", "--trim", "none"], "the MSAs are not trimmed"),
    (["--keep-dropped", "--trim", "bmge"], "--keep-dropped cannot be used with --trim bmge"),
    (["--trim", "clipkit", "-M", "dendroblast"], "also require option '-M msa'"),
    (["--trim", "none", "-M", "dendroblast"], "also require option '-M msa'"),
    (["--trim", "nope"], "Invalid argument for option --trim: nope"),
])
def test_trim_options_rejected(tmp_path, args, message):
    with pytest.raises(process_args.ArgumentError, match=message.replace("(", r"\(").replace(")", r"\)")):
        read(tmp_path, *args)


# ------------------------------------------------ the results' MSAs ----

def test_dropped_columns_pruned_with_the_msa():
    t = ete4.Tree("((0_0:1,1_0:2)n1:3,2_0:4)n0;", parser=1)
    row = {"HOG": "H0", "OG": "OG0000000", "Gene Tree Parent Clade": "n0", "A": "0_0", "C": "2_0"}
    gene_dict = {"0_0": "AC-D\n", "1_0": "AAAA\n", "2_0": "A--D\n"}
    dropped = {"0_0": "-W-\n", "1_0": "YYY\n", "2_0": "-K-\n"}
    incoming, outgoing, reports = queue.Queue(), queue.Queue(), queue.Queue()
    incoming.put(("OG0000000", t, gene_dict, dropped, 0))
    incoming.put(None)
    update_trees.process_task(incoming, outgoing, {"OG0000000": [row]}, {}, ["A", "C"], threading.Event(), reports)
    _, _, results, _ = outgoing.get_nowait()
    _, _, kept, pruned_dropped = results[0]
    # the HOG's genes, without the columns that are gaps in all of them
    assert kept == {"0_0": "ACD\n", "2_0": "A-D\n"}
    assert pruned_dropped == {"0_0": "W\n", "2_0": "K\n"}


def test_no_dropped_columns_left_for_a_hog():
    t = ete4.Tree("((0_0:1,1_0:2)n1:3,2_0:4)n0;", parser=1)
    row = {"HOG": "H0", "OG": "OG0000000", "Gene Tree Parent Clade": "n0", "A": "0_0", "C": "2_0"}
    incoming, outgoing, reports = queue.Queue(), queue.Queue(), queue.Queue()
    incoming.put(("OG0000000", t, {"0_0": "AC\n", "2_0": "AD\n"}, {"0_0": "-\n", "1_0": "Y\n", "2_0": "-\n"}, 0))
    incoming.put(None)
    update_trees.process_task(incoming, outgoing, {"OG0000000": [row]}, {}, ["A", "C"], threading.Event(), reports)
    assert outgoing.get_nowait()[2][0][3] is None       # (not written)


# --------------------------------------------------------- the tool ----

def test_trim_msa_tool(tmp_path, capsys):
    seqs = write_alignment(tmp_path / "in.fa")
    assert trim_msa.main([str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), "--keep-dropped",
                          "--program", "orthofinder"]) == 0
    dropped_fn = tmp_path / "MultipleSequenceAlignments_Dropped" / "out.fa"      # (as a run's)
    assert "column(s) removed" in capsys.readouterr().out
    kept, dropped = dict(trim.read_fasta(tmp_path / "out.fa")), dict(trim.read_fasta(dropped_fn))
    assert all(len(kept[h]) + len(dropped[h]) == len(s) for h, s in seqs.items())
    assert trim_msa.main(["--list"]) == 0
    assert "trimal" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        trim_msa.main([str(tmp_path / "in.fa"), str(tmp_path / "o.fa"), "--program", "bmge", "--keep-dropped"])
    assert not (tmp_path / "MultipleSequenceAlignments_Dropped" / "o.fa").exists()


@pytest.mark.parametrize("program", [[], ["--program", "clipkit"]])
def test_trim_msa_tool_keep_dropped_not_the_input(tmp_path, program):
    # (the columns removed by an earlier run are removed first: the input would be lost)
    (tmp_path / "MultipleSequenceAlignments_Dropped").mkdir()
    infn = tmp_path / "MultipleSequenceAlignments_Dropped" / "OG1.fa"
    seqs = write_alignment(infn)
    with pytest.raises(SystemExit):
        trim_msa.main([str(infn), str(tmp_path / "OG1.fa"), "--keep-dropped"] + program)
    assert dict(trim.read_fasta(infn)) == seqs and not (tmp_path / "OG1.fa").exists()


def test_trim_msa_tool_keep_dropped_none_removed(tmp_path, capsys):
    # (few sequences: copied, nothing removed, no directory of the columns removed left)
    seqs = write_alignment(tmp_path / "in.fa")
    assert trim_msa.main([str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), "--keep-dropped",
                          "--min-seq", "100"]) == 0
    assert dict(trim.read_fasta(tmp_path / "out.fa")) == seqs
    assert sorted(os.listdir(tmp_path)) == ["in.fa", "out.fa"] and "columns removed are in" not in capsys.readouterr().out


@pytest.mark.parametrize("program, dna", [("bmge-dna", []), ("bmge", ["-d"])])
def test_trim_msa_tool_sequence_type(tmp_path, capsys, program, dna):
    # (as --trim in a run: a program limited to one sequence type, by -d)
    write_alignment(tmp_path / "in.fa")
    with pytest.raises(SystemExit):
        trim_msa.main([str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), "--program", program] + dna)
    assert "alignments only" in capsys.readouterr().err and not (tmp_path / "out.fa").exists()


def test_trim_msa_tool_program_paths_without_spaces(tmp_path, capsys):
    # (as -o in a run: a path is words of the program's command)
    (tmp_path / "with space").mkdir()
    seqs = write_alignment(tmp_path / "with space" / "in.fa")
    with pytest.raises(SystemExit):
        trim_msa.main([str(tmp_path / "with space" / "in.fa"), str(tmp_path / "out.fa"), "--program", "clipkit"])
    assert "path with a space" in capsys.readouterr().err
    assert not (tmp_path / "out.fa").exists() and dict(trim.read_fasta(tmp_path / "with space" / "in.fa")) == seqs
    # OrthoFinder's own trimming runs no program
    assert trim_msa.main([str(tmp_path / "with space" / "in.fa"), str(tmp_path / "out.fa"), "--program",
                          "orthofinder"]) == 0


def test_trim_msa_tool_trimmed_without_dropped(tmp_path, capsys):
    # the columns removed are not written: trimmed, not reported as copied untrimmed
    seqs = write_alignment(tmp_path / "in.fa")
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"p": {
        "program_type": "trim", "cmd_line": fake_program(tmp_path, "keep_even", KEEP_EVEN) + " INPUT OUTPUT",
        "cmd_line_dropped": fake_program(tmp_path, "bad", "import sys; sys.exit(4)") + " INPUT OUTPUT"}}))
    assert trim_msa.main([str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), "--program", "p",
                          "--config", str(config), "--keep-dropped"]) == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "WARNING: trimmed, but the columns removed were not written" in err and "untrimmed" not in err
    assert all(s == seqs[h][0::2] for h, s in trim.read_fasta(tmp_path / "out.fa"))

# ------------------------------------------------ program commands ----

@pytest.mark.parametrize("words", ["PATHOGENS", "OUTPUTS", "INPUT_DATA", "BASENAMES"])
def test_variables_in_paths_not_replaced(words):
    infn = "/data/%s/WorkingDirectory/OG1.fa" % words
    cmd = trim.replace_variables("trimal -in INPUT -out OUTPUT -dir PATH -n BASENAME > STDOUT", infn, infn + ".out")
    assert cmd == "trimal -in %s -out %s.out -dir /data/%s/WorkingDirectory -n OG1.fa > %s" % (
        infn, infn, words, os.devnull)
    pc = program_caller.ProgramCaller.__new__(program_caller.ProgramCaller)
    assert pc._replace_variables("famsa INPUT OUTPUT", infn, infn + ".aln") == "famsa %s %s.aln" % (infn, infn)
    db = "/data/DATABASE_%s/diamondDBSpecies0" % words
    assert pc._replace_variables("diamond -d DATABASE -q INPUT -o OUTPUT", infn, infn + ".txt", dbname=db) == \
        "diamond -d %s -q %s -o %s.txt" % (db, infn, infn)


def test_failed_program_kept_untrimmed(tmp_path):
    seqs = write_alignment(tmp_path / "in.fa")
    program = trim.TrimProgram("failing", {"cmd_line": "false INPUT OUTPUT"})
    note, warning, q_failed = trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 2)
    assert "exited with code" in note and warning is None and q_failed
    assert trim.read_fasta(tmp_path / "out.fa") == untrimmed(tmp_path / "in.fa")
    assert len(seqs) == 15


def test_restart_continues_the_trimming_but_not_keep_dropped():
    from orthofinder.utils import run_state
    # (a state of a run with --trim clipkit --keep-dropped; keep_dropped as an earlier build kept it)
    state = {"settings": {"q_trim": True, "trim_program": "clipkit", "keep_dropped": True}}
    options = process_args.Options()
    run_state.restore_settings(state, options, ["-fg", "Results"])
    assert options.trim_program == "clipkit"        # the analysis continues as it was
    assert not options.keep_dropped                 # an extra output: only if given again


def test_restart_of_a_dendroblast_run_with_trim_rejected(tmp_path):
    # -M of the run being continued is restored after the options are read
    from orthofinder.utils import run_state
    prog_caller = process_args.program_caller_with_configs([])
    for extra in (["--trim", "trimal"], ["--trim", "none"]):
        args = ["-fg", str(tmp_path)] + extra
        options = process_args._read_options(args, prog_caller)[0]
        assert process_args.trimming_error(options, prog_caller) is None
        run_state.restore_settings({"settings": {"q_msa_trees": False}}, options, args)
        assert "also require option '-M msa'" in process_args.trimming_error(options, prog_caller)


# ------------------------------------------------------ in a run ----

def run_steps(steps, results):
    """An orthogroup's steps run as a run runs them (command_runner)."""
    from orthofinder.utils import command_runner
    command_runner.run_parallel_commands_and_move_results_file(
        1, [steps], True, on_function_result=results.append)


@pytest.mark.parametrize("mode", ["paired", "single_with_complement"])
def test_program_in_a_run(tmp_path, mode):
    from orthofinder.utils import run_state
    seqs = write_alignment(tmp_path / "in.fa")
    cmd = fake_program(tmp_path, "keep_even", KEEP_EVEN)
    entry = ({"cmd_line": cmd + " INPUT OUTPUT", "cmd_line_dropped": cmd + " INPUT OUTPUT -"} if mode == "paired"
             else {"cmd_line": cmd + " INPUT OUTPUT OUTPUT.complement", "complement_suffix": ".complement"})
    trimmer = trim.Trimmer(trim.TrimProgram("keep_even", entry), keep_dropped=True)
    aln, dropped_fn = str(tmp_path / "in.fa"), str(tmp_path / "d.fa")
    shell = trimmer.program.shell_command(aln, aln, True)
    # the program's own command (stopped with the others), noted as that program's
    assert run_state.command_key(shell) == os.path.basename(sys.executable)
    results = []
    run_steps([(shell, None), (trim.trim_msa, (trimmer, aln, 10, dropped_fn, True))], results)
    assert results == [("in.fa", trim.TRIMMED, 15, None, None)]
    kept, dropped = trim.read_fasta(aln), dict(trim.read_fasta(dropped_fn))
    assert [h for h, _ in kept] == list(seqs)
    assert all(s == seqs[h][0::2] and dropped[h] == seqs[h][1::2] for h, s in kept)
    assert sorted(os.listdir(tmp_path)) == sorted(["in.fa", "d.fa", "keep_even.py"])   # no files left


def test_failed_program_in_a_run_is_reported_not_raised(tmp_path):
    write_alignment(tmp_path / "in.fa")
    script = "import sys; sys.stderr.write('first\\nbad input\\n'); sys.exit(3)"
    trimmer = trim.Trimmer(trim.TrimProgram("bad", {"cmd_line": fake_program(tmp_path, "bad", script) + " INPUT OUTPUT"}))
    aln = str(tmp_path / "in.fa")
    expected = untrimmed(aln)
    results = []
    run_steps([(trimmer.program.shell_command(aln, aln, False), None), (trim.trim_msa, (trimmer, aln, 10, None, True))],
              results)
    assert results == [("in.fa", trim.UNTRIMMED, 15, "bad exited with code 3: bad input, so the alignment was kept untrimmed",
                        None)]
    assert trim.read_fasta(aln) == expected                   # (no other program)
    assert sorted(os.listdir(tmp_path)) == ["bad.py", "in.fa"]


def test_alignment_kept_untrimmed_when_the_program_removes_everything(tmp_path):
    # (no columns removed: none written)
    rng = random.Random(4)
    with open(tmp_path / "in.fa", "w") as outfile:
        for i in range(20):
            gappy = ["-"] * 2000
            gappy[rng.randrange(2000)] = "A"
            outfile.write(">s%d\n%s\n" % (i, "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(600))
                                          + "".join(gappy)))
    everything = "import sys; open(sys.argv[2], 'w').write(''.join(l for l in open(sys.argv[1]) if l.startswith('>')))"
    trimmer = trim.Trimmer(trim.TrimProgram("p", {
        "cmd_line": fake_program(tmp_path, "everything", everything) + " INPUT OUTPUT",
        "cmd_line_dropped": fake_program(tmp_path, "nothing", "import sys; open(sys.argv[2], 'w').close()")
        + " INPUT OUTPUT"}), keep_dropped=True)
    aln, dropped_fn = str(tmp_path / "in.fa"), str(tmp_path / "d.fa")
    import shutil
    shutil.copy(aln, tmp_path / "copy.fa")
    results = []
    run_steps([(trimmer.program.shell_command(aln, aln, True), None),
               (trim.trim_msa, (trimmer, aln, 10, dropped_fn, True))], results)
    assert results == [("in.fa", trim.UNTRIMMED, 20, "p removed every column, so the alignment was kept untrimmed", None)]
    assert trim.read_fasta(aln) == trim.read_fasta(tmp_path / "copy.fa")
    assert not os.path.exists(dropped_fn)


def test_few_columns_of_a_program_that_rewrites_unknowns_used(tmp_path):
    # (as BMGE: "*" written as "?") 6 columns, fewer than N-3, each sequence with a residue: used
    seqs = write_alignment(tmp_path / "in.fa")
    with open(tmp_path / "in.fa", "w") as outfile:
        for h, seq in seqs.items():
            outfile.write(">%s\n*%s\n" % (h, seq[1:]))
    rewrite = ("import sys\nt = open(sys.argv[1]).read().split('>')[1:]\n"
               "r = [x.split('\\n', 1) for x in t]\n"
               "open(sys.argv[2], 'w').write(''.join('>%s\\n%s\\n' % (h.split()[0], ''.join(s.split())[:6].replace('*', '?'))"
               " for h, s in r))\n")
    program = trim.TrimProgram("p", {"cmd_line": fake_program(tmp_path, "rewrite", rewrite) + " INPUT OUTPUT",
                                     "gap_characters": {"protein": "-?X"}})
    assert trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10) == (None, None, False)
    assert [s for _, s in trim.read_fasta(tmp_path / "out.fa")] == [("*" + s[1:])[:6].replace("*", "?")
                                                                     for s in seqs.values()]


def write_columns(fn, columns):
    """An alignment of these columns (strings, a character of each sequence)."""
    with open(fn, "w") as f:
        for i in range(len(columns[0])):
            f.write(">s%d\n%s\n" % (i, "".join(c[i] for c in columns)))


def test_trimming_warnings_summarised(tmp_path, monkeypatch, capsys):
    # in checkpoint.txt (no file of their own): the alignments of 1000 sequences or more kept untrimmed, a
    # summary of all; the warnings also summarised to the screen and Log.txt
    from orthofinder.gene_tree import msa_trees
    from orthofinder.utils import file_handler
    from orthofinder.utils import logging as run_logging

    checkpoint = tmp_path / "checkpoint.txt"

    class Handler:
        def get_check_point_fn(self):
            return str(checkpoint)

        def write_to_log(self, text):
            log.append(text)

    log = []
    monkeypatch.setattr(file_handler, "handler", Handler())
    logger = run_logging.Logger(checkpoint, fmt="%(message)s")
    run_logging.RunLogger.set_logger(logger)
    try:
        msa_trees.write_trimming_warnings([("OG0000002.fa", trim.UNTRIMMED, 1000, "b", None),
                                           ("OG0000001.fa", trim.UNTRIMMED, 2000, "a", None),
                                           ("OG0000004.fa", trim.UNTRIMMED, 999, "d", None),        # (not listed)
                                           ("OG0000005.fa", trim.FEW, 4, None, None),
                                           ("OG0000006.fa", trim.ALL_KEPT, 40, None, None),
                                           ("OG0000003.fa", trim.TRIMMED, 30, None, "c")], "trimal")
    finally:
        run_logging.RunLogger.set_logger(None)
        logger.close()
    out = " ".join(capsys.readouterr().out.split())        # (rich wraps it)
    assert out.count("WARNING") == 1 and log and "untrimmed" not in out
    assert "1 alignment was trimmed by trimal without the columns removed written" in out
    stage = "[%s] " % msa_trees.TRIMMING_STAGE
    assert checkpoint.read_text() == (
        stage + "OG0000001.fa a\n" + stage + "OG0000002.fa b\n"
        + stage + "Trimming with trimal, 6 alignments: 1 trimmed, 1 with every column kept, 3 kept untrimmed (the "
        "trimming program failed, removed every column or left a sequence with no residues), 1 not trimmed (10 "
        "sequences or fewer)\n"
        + stage + "WARNING: OG0000003.fa trimmed, c\n")
    assert os.listdir(tmp_path) == ["checkpoint.txt"]


@pytest.mark.parametrize("dropped_script", [
    "import sys; sys.exit(4)",
    # not the input's sequences
    "import sys; open(sys.argv[2], 'w').write('>x\\nAC\\n')",
])
def test_keep_dropped_does_not_change_the_alignment(tmp_path, dropped_script):
    # the columns removed cannot be written: the trimmed alignment is used, as without --keep-dropped
    seqs = write_alignment(tmp_path / "in.fa")
    keep_even = fake_program(tmp_path, "keep_even", KEEP_EVEN)
    entry = {"cmd_line": keep_even + " INPUT OUTPUT",
             "cmd_line_dropped": fake_program(tmp_path, "bad", dropped_script) + " INPUT OUTPUT"}
    trimmer = trim.Trimmer(trim.TrimProgram("p", entry), keep_dropped=True)
    aln, dropped_fn = str(tmp_path / "in.fa"), str(tmp_path / "d.fa")
    results = []
    run_steps([(trimmer.program.shell_command(aln, aln, True), None),
               (trim.trim_msa, (trimmer, aln, 10, dropped_fn, True))], results)
    assert len(results) == 1 and results[0][:4] == ("in.fa", trim.TRIMMED, 15, None)
    assert results[0][4].startswith("the columns removed were not written, p")
    assert all(s == seqs[h][0::2] for h, s in trim.read_fasta(aln))
    assert not os.path.exists(dropped_fn)
    # the same by the tool
    write_alignment(tmp_path / "in.fa")
    note, warning, _ = trim.run_program(trimmer.program, aln, aln, 10, dropped_fn)
    assert warning.startswith("the columns removed were not written") and note is None
    assert all(s == seqs[h][0::2] for h, s in trim.read_fasta(aln))


def test_columns_removed_but_not_written_reported(tmp_path):
    # a complement file not written: none removed only if the alignment is not shorter
    seqs = write_alignment(tmp_path / "in.fa")
    keep_even = fake_program(tmp_path, "keep_even", KEEP_EVEN)
    trimmer = trim.Trimmer(trim.TrimProgram("p", {"cmd_line": keep_even + " INPUT OUTPUT",
                                                  "complement_suffix": ".complement"}), keep_dropped=True)
    aln, dropped_fn = str(tmp_path / "in.fa"), str(tmp_path / "d.fa")
    results = []
    run_steps([(trimmer.program.shell_command(aln, aln, True), None),
               (trim.trim_msa, (trimmer, aln, 10, dropped_fn, True))], results)
    n = len(next(iter(seqs.values())))
    assert results == [("in.fa", trim.TRIMMED, 15, None, "the columns removed were not written, p removed %d column(s) but did not "
                        "write them" % (n - len(next(iter(seqs.values()))[0::2])))]
    assert all(s == seqs[h][0::2] for h, s in trim.read_fasta(aln))       # (trimmed)


def test_dropped_command_not_run_after_a_failure(tmp_path):
    write_alignment(tmp_path / "in.fa")
    marker = tmp_path / "ran"
    second = fake_program(tmp_path, "second", "open(%r, 'w').close()" % str(marker))
    trimmer = trim.Trimmer(trim.TrimProgram("p", {"cmd_line": "false INPUT OUTPUT",
                                                  "cmd_line_dropped": second + " INPUT OUTPUT"}), keep_dropped=True)
    aln = str(tmp_path / "in.fa")
    results = []
    run_steps([(trimmer.program.shell_command(aln, aln, True), None),
               (trim.trim_msa, (trimmer, aln, 10, str(tmp_path / "d.fa"), True))], results)
    assert "exited with code" in results[0][3] and not marker.exists()


def test_dropped_command_of_several_not_run_after_a_failure(tmp_path):
    # a cmd_line of several commands: all of it skipped, its messages kept (a group)
    write_alignment(tmp_path / "in.fa")
    m1, m2 = tmp_path / "ran1", tmp_path / "ran2"
    trimmer = trim.Trimmer(trim.TrimProgram("p", {
        "cmd_line": "echo bad input >&2; false INPUT OUTPUT",
        "cmd_line_dropped": "touch %s; touch %s INPUT OUTPUT" % (m1, m2)}), keep_dropped=True)
    aln = str(tmp_path / "in.fa")
    results = []
    run_steps([(trimmer.program.shell_command(aln, aln, True), None),
               (trim.trim_msa, (trimmer, aln, 10, str(tmp_path / "d.fa"), True))], results)
    assert results[0][3].startswith("p exited with code 1: bad input, so") and results[0][4] is None
    assert not m1.exists() and not m2.exists()


@pytest.mark.parametrize("gaps, q_untrimmed", [(None, False), ({"protein": "-?*X"}, True)])
def test_program_gaps_decide_a_residue(tmp_path, gaps, q_untrimmed):
    # a sequence left with only X: a residue by default (- and *), none for a program with X a gap
    seqs = write_alignment(tmp_path / "in.fa")
    first = next(iter(seqs))
    script = KEEP_EVEN + "\nimport re\n" + (
        "t = open(sys.argv[2]).read().split('\\n')\n"
        "t[1] = re.sub('[^-]', 'X', t[1])\n"
        "open(sys.argv[2], 'w').write('\\n'.join(t))\n")
    entry = {"cmd_line": fake_program(tmp_path, "x_out", script) + " INPUT OUTPUT"}
    if gaps:
        entry["gap_characters"] = gaps
    program = trim.TrimProgram("p", entry)
    note, warning, q_failed = trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10)
    assert (note is not None) is q_untrimmed and warning is None and not q_failed
    if q_untrimmed:
        # the input kept untrimmed instead
        assert "no residues, e.g. %s" % first.split()[0] in note
        assert trim.read_fasta(tmp_path / "out.fa") == untrimmed(tmp_path / "in.fa")
    assert program.gaps(True) == trim.DEFAULT_GAP_CHARACTERS        # (none given for dna)


@pytest.mark.parametrize("gaps", [{"amino": "-X"}, {"protein": ""}, "-X", {"dna": 5}])
def test_program_gaps_checked(gaps):
    with pytest.raises(trim.InvalidTrimEntry, match="gap_characters"):
        trim.TrimProgram("p", {"cmd_line": "p INPUT OUTPUT", "gap_characters": gaps})


def test_configured_gaps():
    # (each program's gaps and unknowns, from its source: see config.json)
    prog_caller = process_args.program_caller_with_configs([])
    expected = {"trimal": ("-X", "-N"), "clipkit": ("-?*Xx", "-?*XxNn"), "clipkit-kpic-smart-gap": ("-?*Xx", "-?*XxNn"),
                "bmge": ("-?X", trim.DEFAULT_GAP_CHARACTERS), "bmge-dna": (trim.DEFAULT_GAP_CHARACTERS, "-?NX")}
    for name, (protein, dna) in expected.items():
        program = prog_caller.get_trim_program(name)
        assert (program.gaps(False), program.gaps(True)) == (protein, dna), name
    options = process_args.Options()
    options.trim_program, options.dna = "clipkit", True
    assert process_args.trim_gaps(options, prog_caller) == "-?*XxNn"
    options.trim_program = None                       # (OrthoFinder's trimming)
    assert process_args.trim_gaps(options, prog_caller) == "-*"
    assert trim.Trimmer(prog_caller.get_trim_program("trimal"), nucleotide=True).gaps() == "-N"


def test_program_not_run_in_a_run_on_few_sequences(tmp_path):
    from orthofinder.gene_tree import msa_trees
    write_alignment(tmp_path / "in.fa", n_seqs=8)
    before = (tmp_path / "in.fa").read_text()
    trimmer = trim.Trimmer(trim.TrimProgram("bad", {"cmd_line": "false INPUT OUTPUT"}))
    steps = msa_trees.trim_commands(trimmer, 0, str(tmp_path / "in.fa"), 10, 8)
    assert [command for command, _ in steps] == [trim.trim_msa]       # (only the step after it)
    results = []
    run_steps(steps, results)
    assert results == [("in.fa", trim.FEW, 8, None, None)] and (tmp_path / "in.fa").read_text() == before


@pytest.mark.parametrize("args, message", [
    (["-d", "--trim", "bmge"], "--trim bmge trims protein alignments only (sequence_type in its config.json "
                               "entry), but the input is nucleotide (-d). Use --trim bmge-dna instead"),
    (["--trim", "bmge-dna"], "--trim bmge-dna trims nucleotide alignments only"),
])
def test_trim_sequence_type(tmp_path, args, message):
    prog_caller = process_args.program_caller_with_configs([])
    options = read(tmp_path, *args)         # (-d may be that of a run being continued: checked once restored)
    assert process_args.trimming_error(options, prog_caller).startswith(message)
    assert process_args.trimming_error(read(tmp_path, "-d", "--trim", "bmge-dna"), prog_caller) is None


def test_trim_sequence_type_of_a_restart(tmp_path):
    from orthofinder.utils import run_state
    prog_caller = process_args.program_caller_with_configs([])
    args = ["-fg", str(tmp_path), "--trim", "bmge-dna"]
    options = process_args._read_options(args, prog_caller)[0]     # (no -d: that of the run, restored)
    run_state.restore_settings({"settings": {"dna": True}}, options, args)
    assert process_args.trimming_error(options, prog_caller) is None


def test_dna_test_alignment():
    lines = program_caller.test_alignment(nucleotide=True).split("\n")
    seqs = [line for line in lines[1::2] if line]
    assert set("".join(seqs)) <= set("ACGT-")
    assert {len(seq) for seq in seqs} == {900}


def test_program_stopped_when_another_command_fails(tmp_path):
    import subprocess
    import time
    from orthofinder.utils import command_runner
    write_alignment(tmp_path / "in.fa")
    pid_fn = tmp_path / "pid"
    script = "import os, sys, time; open(%r, 'w').write(str(os.getpid())); time.sleep(60)" % str(pid_fn)
    trimmer = trim.Trimmer(trim.TrimProgram("slow", {"cmd_line": fake_program(tmp_path, "slow", script) + " INPUT OUTPUT"}))
    aln = str(tmp_path / "in.fa")
    slow = [(trimmer.program.shell_command(aln, aln, False), None), (trim.trim_msa, (trimmer, aln, 10, None, True))]
    failing = [("while [ ! -s %s ]; do sleep 0.05; done; false" % pid_fn, None)]
    start = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):
        command_runner.run_parallel_commands_and_move_results_file(2, [slow, failing], True)
    assert time.monotonic() - start < 30
    pid = int(pid_fn.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# ------------------------------------------- a trimmer's own commands ----


def test_many_sequences_command(tmp_path):
    entry = {"cmd_line": "t INPUT OUTPUT", "cmd_line_large_n": "t -m kpic INPUT OUTPUT", "large_n": 1000}
    program = trim.TrimProgram("t", entry)
    assert program.command(999) == "t INPUT OUTPUT" and program.command(1000) == "t -m kpic INPUT OUTPUT"
    assert program.command() == "t INPUT OUTPUT"
    assert "-m kpic" in program.shell_command("/a/in.fa", "/a/in.fa", False, 1000)
    assert "-m kpic" not in program.shell_command("/a/in.fa", "/a/in.fa", False, 10)
    for bad in ({"cmd_line": "t", "cmd_line_large_n": "u"}, {"cmd_line": "t", "large_n": 5},
                {"cmd_line": "t", "cmd_line_large_n": "u", "large_n": 0},
                {"cmd_line": "t", "cmd_line_large_n": "u", "large_n": "5"}):
        with pytest.raises(trim.InvalidTrimEntry):
            trim.TrimProgram("t", bad)
    # in a run: the number of sequences decides
    from orthofinder.gene_tree import msa_trees
    trimmer = trim.Trimmer(program)
    steps = msa_trees.trim_commands(trimmer, 0, "/a/in.fa", 10, 1500)
    assert "-m kpic" in steps[0][0]


@pytest.mark.parametrize("config", ["src/orthofinder/run/config.json", "user_config.json"])
def test_clipkit_kpic_with_many_sequences(config):
    program = program_caller.ProgramCaller(str(ROOT / config)).get_trim_program("clipkit")
    assert program.large_n == 1000 and "-m kpic " in program.command(1000) and "-m" not in program.command(999)


def test_default_clipkit_trims_with_the_tool(tmp_path):
    import shutil
    if shutil.which("clipkit") is None:
        pytest.skip("clipkit is not installed")
    seqs = write_alignment(tmp_path / "in.fa")
    assert trim_msa.main([str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), "--keep-dropped"]) == 0
    kept = dict(trim.read_fasta(tmp_path / "out.fa"))
    dropped = dict(trim.read_fasta(tmp_path / "MultipleSequenceAlignments_Dropped" / "out.fa"))
    assert [h.split()[0] for h in kept] == [h.split()[0] for h in seqs]
    assert all(len(kept[h]) + len(dropped[h]) == 700 for h in kept)


def test_program_that_keeps_every_column_used(tmp_path):
    write_alignment(tmp_path / "in.fa")
    program = trim.TrimProgram("copy", {"cmd_line": "cp INPUT OUTPUT"})
    assert trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10) == (None, None, False)
    assert trim.read_fasta(tmp_path / "out.fa") == untrimmed(tmp_path / "in.fa")


def test_orthofinder_trimming_in_a_run_unchanged(tmp_path):
    # OrthoFinder's own trimming alone, also of a short alignment and with every column kept
    for name, cols in (("long.fa", None), ("short.fa", ["A" * 12, "AC" * 6, "-" * 12])):
        if cols is None:
            write_alignment(tmp_path / name)
        else:
            write_columns(tmp_path / name, cols)
        trim.run_in_process(str(tmp_path / name), str(tmp_path / ("plain_" + name)), 10, 0.1, 500, 0.75)
        record = trim.trim_msa(trim.Trimmer(), str(tmp_path / name), 10)
        assert record == (name, trim.TRIMMED if cols is None else trim.ALL_KEPT, 15 if cols is None else 12, None, None)
        assert (tmp_path / name).read_bytes() == (tmp_path / ("plain_" + name)).read_bytes()


@pytest.mark.parametrize("complement, written", [
    # headers only: none of the 350 columns removed written
    ("open(sys.argv[2] + '.c', 'w').write(''.join('>%s\\n\\n' % h.split()[0] for h, _ in records))", 0),
    # some of them only
    ("open(sys.argv[2] + '.c', 'w').write(''.join('>%s\\n%s\\n' % (h.split()[0], s[1:20:2]) for h, s in records))", 10),
])
def test_columns_removed_not_all_written_reported(tmp_path, complement, written):
    seqs = write_alignment(tmp_path / "in.fa")
    script = KEEP_EVEN.replace("if len(sys.argv) > 3 and", "if False and") + "\n" + complement + "\n"
    program = trim.TrimProgram("p", {"cmd_line": fake_program(tmp_path, "p", script) + " INPUT OUTPUT",
                                     "complement_suffix": ".c"})
    note, warning, q_failed = trim.run_program(program, str(tmp_path / "in.fa"), str(tmp_path / "out.fa"), 10,
                                               str(tmp_path / "d.fa"))
    removed = 700 - 350
    assert note is None and not q_failed
    assert warning == ("the columns removed were not written, p removed %d column(s) but did not write them" % removed
                       if written == 0 else
                       "the columns removed were not written, p removed %d column(s) but wrote %d" % (removed, written))
    assert not (tmp_path / "d.fa").exists()
    assert all(s == seqs[h][0::2] for h, s in trim.read_fasta(tmp_path / "out.fa"))       # (trimmed)
