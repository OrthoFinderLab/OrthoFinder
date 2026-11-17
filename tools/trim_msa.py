#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trim a multiple sequence alignment as an OrthoFinder run does, with a
trimming program configured in OrthoFinder's config.json (--program, as
--trim in a run, default clipkit) or with OrthoFinder's own trimming
(--program orthofinder), optionally saving the columns removed
(--keep-dropped, as in a run, in MultipleSequenceAlignments_Dropped/ next to
OUTPUT, with OUTPUT's file name). An alignment of --min-seq sequences or
fewer is not trimmed (copied).

A trimming program's output is used if every sequence is still there with a
residue and it kept at least one column. Otherwise, or if the program fails,
the alignment is kept untrimmed, as in a run.

OrthoFinder's own trimming removes the columns with more than 90% gaps,
keeping at least 500 columns and 75% of the residues. It does not trim an
alignment of 500 columns or fewer.

Examples
--------
trim_msa OG0000000.fa OG0000000.trimmed.fa

trim_msa OG0000000.fa OG0000000.trimmed.fa --keep-dropped

trim_msa OG0000000.fa OG0000000.trimmed.fa --program orthofinder

trim_msa OG0000000.fa OG0000000.trimmed.fa --program trimal

trim_msa OG0000000.fa OG0000000.trimmed.fa --program my_trimmer --config my_config.json
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
src_dir = os.path.join(repo_root, "src")

if os.path.isdir(os.path.join(src_dir, "orthofinder")) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from orthofinder.gene_tree import trim

DROPPED_DIR = "MultipleSequenceAlignments_Dropped"       # (as a run's, --keep-dropped)
DEFAULT_PROGRAM = "clipkit"     # (as a run's, process_args.DEFAULT_TRIM_PROGRAM)


def dropped_file(outfn):
    """--keep-dropped: the file of the columns removed, as a run's (the trimmed alignment's name)."""
    return os.path.join(os.path.dirname(os.path.abspath(outfn)), DROPPED_DIR, os.path.basename(outfn))


def same_file(fn1, fn2):
    """Whether two paths are the same file (or would be, if neither exists yet)."""
    if os.path.exists(fn1) and os.path.exists(fn2):
        return os.path.samefile(fn1, fn2)
    return os.path.realpath(fn1) == os.path.realpath(fn2)


def trim_programs(config_files):
    """The configured trimming programs: config.json, ~/config_orthofinder_user.json and config_files."""
    from orthofinder.run import process_args
    args = []
    for fn in config_files:
        args += ["--config", fn]
    return process_args.program_caller_with_configs(args)


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="trim_msa",
        description="Trim a multiple sequence alignment (FASTA) as an OrthoFinder run does.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("Examples"):])
    parser.add_argument("input", metavar="INPUT", nargs="?", help="the alignment (FASTA)")
    parser.add_argument("output", metavar="OUTPUT", nargs="?", help="the trimmed alignment (FASTA), may be INPUT")
    parser.add_argument("--keep-dropped", action="store_true",
                        help="also save the columns removed (FASTA), in %s/ next to OUTPUT with OUTPUT's file "
                             "name, as in a run (not written if none are)" % DROPPED_DIR)
    parser.add_argument("--program", metavar="NAME",
                        help="a trimming program of OrthoFinder's config.json (program_type trim), "
                             "e.g. trimal, or orthofinder for OrthoFinder's own trimming (default clipkit, as a run)")
    parser.add_argument("--config", metavar="FILE", action="append", default=[],
                        help="a config file with more trimming programs (can be given more than once)")
    parser.add_argument("--list", action="store_true", help="list the trimming programs configured and stop")
    parser.add_argument("-d", "--dna", action="store_true",
                        help="the alignment is of nucleotides, for the gap characters of the program's config entry")
    parser.add_argument("--min-seq", type=int, default=10,
                        help="an alignment of this many sequences or fewer is not trimmed (default 10)")
    parser.add_argument("-f", type=float, default=trim.DEFAULT_F,
                        help="OrthoFinder's trimming, the fraction of residues a column needs at "
                             "least (default %(default)s)")
    parser.add_argument("--min-length", type=int, default=trim.DEFAULT_N_MIN,
                        help="OrthoFinder's trimming, the columns kept at least (default %(default)s)")
    parser.add_argument("-c", type=float, default=trim.DEFAULT_C,
                        help="OrthoFinder's trimming, the fraction of the residues kept at least "
                             "(default %(default)s)")
    a = parser.parse_args(args)
    if a.list:
        print("\n".join(["orthofinder"] + [name + (" (the default)" if name == DEFAULT_PROGRAM else "")
                                            for name in trim_programs(a.config).list_trim_methods()]))
        return 0
    if a.input is None or a.output is None:
        parser.error("the following arguments are required: INPUT, OUTPUT")
    if not os.path.isfile(a.input):
        parser.error("the alignment does not exist: %s" % a.input)
    dropped = dropped_file(a.output) if a.keep_dropped else None
    if dropped is not None and same_file(dropped, a.input):
        parser.error("with --keep-dropped the columns removed are written to %s, which is INPUT. Please "
                     "write OUTPUT elsewhere" % dropped)
    made = dropped is not None and not os.path.isdir(os.path.dirname(dropped))
    if made:
        os.makedirs(os.path.dirname(dropped))
    try:
        returncode = trim_alignment(parser, a, dropped)
    finally:
        if made and not os.listdir(os.path.dirname(dropped)):
            os.rmdir(os.path.dirname(dropped))       # (none removed: no directory left either)
    if dropped is not None and os.path.exists(dropped):
        print("The columns removed are in %s" % dropped)
    return returncode


def trim_alignment(parser, a, dropped):
    """main's trimming, of a.input to a.output and the columns removed to dropped (if not None)."""

    if a.program is None:
        a.program = DEFAULT_PROGRAM
    if a.program == "orthofinder":
        try:
            n_removed = trim.run_in_process(a.input, a.output, a.min_seq, a.f, a.min_length, a.c, dropped)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        print("%d column(s) removed" % n_removed)
        return 0

    # (as in a run, -o: the files' paths are words of the program's command)
    spaced = [fn for fn in (a.input, a.output) if any(c.isspace() for c in os.path.abspath(fn))]
    if spaced:
        parser.error("a trimming program cannot be given a path with a space: %s. Please use paths without "
                     "spaces" % spaced[0])
    prog_caller = trim_programs(a.config)
    if a.program not in prog_caller.list_trim_methods():
        parser.error("no trimming program called %s is configured, those that are: %s"
                     % (a.program, ", ".join(["orthofinder"] + prog_caller.list_trim_methods())))
    program = prog_caller.get_trim_program(a.program)
    # (as a run, process_args.trimming_error: a program limited to one sequence type, by -d)
    input_type = "dna" if a.dna else "protein"
    if program.sequence_type is not None and program.sequence_type != input_type:
        others = [name for name in prog_caller.list_trim_methods()
                  if prog_caller.get_trim_program(name).sequence_type == input_type]
        parser.error("%s trims %s alignments only (sequence_type in its config.json entry), but the alignment is "
                     "%s%s" % (a.program, "nucleotide" if program.sequence_type == "dna" else "protein",
                               "nucleotide (-d)" if a.dna else "protein (give -d for a nucleotide alignment)",
                               ". Use --program %s instead" % others[0] if others else ""))
    if dropped is not None and not program.writes_dropped():
        parser.error("%s's config entry does not say how it writes the columns it removes "
                     "(cmd_line_dropped or complement_suffix)" % a.program)
    # (the programs found as in a run: its search order, parallel_task_manager.my_env's PATH)
    from orthofinder.utils import parallel_task_manager
    note, warning, q_failed = trim.run_program(program, a.input, a.output, a.min_seq, dropped,
                                               env=parallel_task_manager.my_env.copy(), nucleotide=a.dna)
    if note is not None:
        # (as a run: kept untrimmed, the program's output could not be used)
        print("%s." % note, file=sys.stderr)
    if q_failed:
        return 1
    if warning is not None:
        print("WARNING: trimmed, but %s." % warning, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
