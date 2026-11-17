#!/usr/bin/env python3
"""
Trimming of multiple sequence alignments: OrthoFinder's own (run_in_process),
or a trimming program configured in config.json ("program_type": "trim",
TrimProgram, the default ClipKIT), with the columns removed optionally
written to a file of their own (the dropped columns). If a program removes
every column, fails, or leaves a sequence without residues, the alignment is
kept untrimmed (_keep_whole). It needs nothing else of OrthoFinder: the
trim_msa tool (tools/trim_msa.py) runs it on its own.
"""

import glob
import os
import re
import shlex
import shutil
import subprocess
import numpy as np
try:
    from rich import print
except ImportError:
    ...

# OrthoFinder's trimming: the default parameters (see run_in_process)
DEFAULT_F = 0.1
DEFAULT_N_MIN = 500
DEFAULT_C = 0.75

class MSA(object):
    """
    An alignment as one array: a row of characters per sequence (np.frombuffer
    of the sequences' bytes), so the columns' counts and the trimming are
    single passes in numpy rather than loops over the characters.
    Gaps are "-" and "*" (as in the counts; "*" is written as "-").
    """
    def __init__(self, fn):
        records = read_fasta(fn)
        self.names = [name for name, _ in records]      # list of names
        seqs = [seq for _, seq in records]
        self.n = len(self.names)    # number of sequences
        self.length = len(seqs[0]) if seqs else 0   # number of columns in MSA
        for i, seq in enumerate(seqs[1:], 2):
            if len(seq) != self.length:
                # (an error, not sys.exit(): that is a successful exit, and the run would
                # end as if stopped normally, without its results)
                raise ValueError("ERROR: Malformed alignment %s: sequence %d has length %d, previous was %d"
                                 % (fn, i, len(seq), self.length))
        # one byte a character; other characters (not Latin-1) as 4-byte code points
        try:
            self.encoding, dtype = "latin-1", np.uint8
            data = "".join(seqs).encode(self.encoding)
        except UnicodeEncodeError:
            self.encoding, dtype = "utf-32-le", np.uint32
            data = "".join(seqs).encode(self.encoding)
        self.M = np.frombuffer(data, dtype=dtype).reshape(self.n, self.length)
        self.residues = (self.M != ord("-")) & (self.M != ord("*"))   # non-gap characters

    def write_msa(self, i_cols, outfn):
        kept = np.where(self.residues[:, i_cols], self.M[:, i_cols], ord("-")).astype(self.M.dtype)
        write_fasta(outfn, ((name, row.tobytes().decode(self.encoding)) for name, row in zip(self.names, kept)))

def main(infn, outfn, msa_min_seq=10, f=DEFAULT_F, n_min=DEFAULT_N_MIN, c=DEFAULT_C, dropped_fn=None):
    """OrthoFinder's trimming of infn to outfn (see run_in_process)."""
    return run_in_process(infn, outfn, msa_min_seq, f, n_min, c, dropped_fn)


def run_in_process(infn, outfn, msa_min_seq, f, n_min, c, dropped_fn=None):
    """
    Trim the alignment file. Auto reduce f parameter if required. OrthoFinder default
    parameters are used by default.
    Args:
        infn - input FASTA filename
        outfn - output FASTA filename
        f - Lower limit for fraction of meaningful characters
        n_min - The min trimmed length
        c - The min fraction of non-gap characters to conserve
        dropped_fn - if given, the columns removed are written to it (no file
                     if none are)
    Returns:
        the number of columns removed
    Notes:
        If f is too high then it will reduce f s.t. len >= n_min or the full 
        alignment is kept (if less than n_min).
        In some way fraction of gaps is unsatisfactory since as the number of sequences
        in the MSA gets larger a column could contain a high fraction of gaps and 
        yet still contain useful phylogenetic signal for (many) sequences that are
        closely related and don't have a gap in that column. For this reason the 
        c parameter is included which specifies the minimum total fraction of the 
        original non-gap characters that must be preserved.
        An alignment is not trimmed (it is copied) if a sequence would be left
        with no residues.
    """
    if not os.path.exists(infn):
        raise FileNotFoundError("ERROR: The alignment to trim does not exist: %s" % infn)
    try:
        # BEWARE: Input and output files may be the same. (Opened only, to check it
        # can be written: nothing added, the alignment may be kept as it is.)
        with open(outfn, 'a', encoding="utf-8", newline="\n"):
            pass
    except OSError as e:
        raise OSError("ERROR: Cannot write the trimmed alignment %s: %s" % (outfn, e)) from e

    remove_file(dropped_fn)     # (of an earlier run)
    msa = MSA(infn)
    n = msa.n
    if msa.length <= n_min or n <= msa_min_seq:
        copy_input_to_output(infn, outfn)
        # print("Already below min length: %d" % msa.length)
        return 0

    length = msa.length
    max_gap = (1.-f)*n
    aa_counts = msa.residues.sum(axis=0)   # per column
    gap_counts = n - aa_counts
    aa_before = int(aa_counts.sum())
    i_keep = np.where(gap_counts <= max_gap)
    n_keep = i_keep[0].size     # it's an I, J tuple
    aa_after = aa_counts[i_keep].sum()
    # Check we have kept enough columns and characters
    if n_keep < n_min or aa_after < c*aa_before:
        # print("%0.3f: %d columns, %0.1f%% of characters retained" % (f, n_keep, 100.*aa_after/aa_before))
        f, i_keep = get_satifactory_f(gap_counts, aa_counts, n, f, n_min, c)

    to_keep = i_keep[0]
    # trimmed only if every sequence keeps a residue
    trim = bool(msa.residues[:, to_keep].any(axis=1).all())
    n_keep = to_keep.size
    if not trim:
        copy_input_to_output(infn, outfn)
        return 0
    if dropped_fn is not None and n_keep < length:
        dropped = np.setdiff1d(np.arange(length), to_keep)
        msa.write_msa(dropped, dropped_fn)
    # (also with every column kept, as before: rewritten, "*" as a gap)
    msa.write_msa(to_keep, outfn)
    return length - n_keep
    # print("%0.3f: %d->%d, %0.1f%% characters retained. Trimmed %s" % (f, length, to_keep.size, 100.*aa_after/aa_before, infn))

def get_satifactory_f(gap_counts, aa_counts, N, f_orig, n_min, c, tol = 0.001):
    """
    The f used was too large, get an f that gives n_min columns
    Args:
        gap_counts = np array of gaps per column
        aa_counts = np array of amino acids per column
        N - number of sequences
        f_orig - starting f for the search
        n_min - minimum alignment length
        c - minimum fraction of amino acids to keep
        tol - convergence tolerance
    Implementation:
        down - lower bound on non-gap percentage, giving larger alignment
        up - upper bound on non-gap character percentage
    """
    # binary search 
    aa_before = aa_counts.sum()
    up = f_orig
    down = 0.
    x_prev = f_orig
    x = 0.5*(down+up)
    while abs(x-x_prev) > tol:
        max_gap = (1.-x)*N
        i_keep = np.where(gap_counts <= max_gap)
        n_keep = i_keep[0].size
        aa_new = aa_counts[i_keep].sum()
        # print("%0.3f: %d columns, %0.1f%% of characters retained" % (x, n_keep, 100.*aa_new/aa_before))
        if n_keep < n_min or aa_new < c*aa_before:
            up = x
        elif (n_keep == n_min and aa_new >= c*aa_before) or (n_keep >= n_min and aa_new == c*aa_before):
            # one on the limit, the other satisfied - stop exactly here (x is used)
            down = x
            break
        else:
            down = x
        x_prev = x
        x = 0.5*(down+up)
    # Use the value which gave just above the min length
    x = down
    max_gap = (1.-x)*N
    i_keep = np.where(gap_counts <= max_gap)
    aa_new = aa_counts[i_keep].sum()
    # print("%0.3f: %d columns, %0.1f%% characters retained" % (x, n_keep, 100.*aa_new/aa_before))
    return x, i_keep


def copy_input_to_output(infn, outfn):
    # if the outfn doesn't exist doesn't the check for being the same is redundant
    if (not os.path.exists(outfn)) or (not os.path.samefile(infn, outfn)):
        shutil.copy(infn, outfn)


def remove_file(fn):
    if fn is not None and os.path.exists(fn):
        os.remove(fn)


# ------------------------------------------------- trimming programs ----

# The ways a trimming program writes the columns it removes (config.json "output_mode")
OUTPUT_MODES = ("paired", "single_with_complement", "single")

# The sequences a trimming program's command can be limited to (config.json "sequence_type")
SEQUENCE_TYPES = ("protein", "dna")
# The gaps of OrthoFinder's trimming (MSA.residues), and of a program whose entry
# gives none (TrimProgram.gaps)
DEFAULT_GAP_CHARACTERS = "-*"


class InvalidTrimEntry(ValueError):
    pass


class TrimProgram(object):
    """
    A trimming program of config.json ("program_type": "trim"), the same
    entries as trimmiy's (only the output modes OrthoFinder can read):
        "cmd_line" (or "cmd_line_kept"): writes the trimmed alignment to OUTPUT
        "cmd_line_dropped": (output mode "paired") writes the columns removed to OUTPUT
        "cmd_line_large_n", "large_n": cmd_line of an alignment of large_n
            sequences or more (optional, both or neither)
        "complement_suffix": (output mode "single_with_complement") cmd_line
            also writes the columns removed to OUTPUT + complement_suffix
        "output_mode": one of OUTPUT_MODES (default: from the entries above);
            "single": the program does not write the columns it removes
        "sequence_type": one of SEQUENCE_TYPES if the command only trims
            those (e.g. it tells the program the type), default either
        "gap_characters": {sequence type: the characters the program treats as
            gaps or unknown}, e.g. {"protein": "-?*X", "dna": "-?*XN"}: a
            sequence it leaves with only these has no residues (default
            DEFAULT_GAP_CHARACTERS)
    The commands can use INPUT, OUTPUT, BASENAME, PATH, BASEOUTNAME and STDOUT
    (where the program's output goes, it is not kept).
    """

    def __init__(self, name, entry):
        self.name = name
        self.cmd = entry.get("cmd_line", entry.get("cmd_line_kept"))
        if not self.cmd:
            raise InvalidTrimEntry("'cmd_line' entry is missing")
        self.cmd_dropped = entry.get("cmd_line_dropped")
        self.cmd_large_n = entry.get("cmd_line_large_n")
        self.large_n = entry.get("large_n")
        if (self.cmd_large_n is None) != (self.large_n is None):
            raise InvalidTrimEntry("'cmd_line_large_n' and 'large_n' go together")
        if self.large_n is not None and (isinstance(self.large_n, bool) or not isinstance(self.large_n, int)
                                           or self.large_n < 1):
            raise InvalidTrimEntry("'large_n' must be a whole number, 1 or more")
        self.complement_suffix = entry.get("complement_suffix")
        default_mode = ("paired" if self.cmd_dropped else
                        "single_with_complement" if self.complement_suffix else "single")
        self.output_mode = entry.get("output_mode", default_mode)
        if self.output_mode not in OUTPUT_MODES:
            raise InvalidTrimEntry("output mode '%s' is not supported, only %s"
                                   % (self.output_mode, ", ".join(OUTPUT_MODES)))
        if entry.get("output_type", "alignment") not in ("alignment", "kept-only-alignemnt", "kept-only-alignment"):
            raise InvalidTrimEntry("output type '%s' is not supported, only alignments" % entry["output_type"])
        if self.output_mode == "paired" and not self.cmd_dropped:
            raise InvalidTrimEntry("output mode 'paired' needs 'cmd_line_dropped'")
        if self.output_mode == "single_with_complement" and not self.complement_suffix:
            raise InvalidTrimEntry("output mode 'single_with_complement' needs 'complement_suffix'")
        self.sequence_type = entry.get("sequence_type")
        if self.sequence_type is not None and self.sequence_type not in SEQUENCE_TYPES:
            raise InvalidTrimEntry("sequence type '%s' is not supported, only %s"
                                   % (self.sequence_type, ", ".join(SEQUENCE_TYPES)))
        gaps = entry.get("gap_characters", {})
        if not isinstance(gaps, dict) or any(t not in SEQUENCE_TYPES or not isinstance(c, str) or not c
                                             for t, c in gaps.items()):
            raise InvalidTrimEntry("gap_characters must be {sequence type: characters}, the types %s"
                                   % ", ".join(SEQUENCE_TYPES))
        self.gap_characters = gaps
        self.skip_check = entry.get("skip_check") == "true"

    def gaps(self, nucleotide=False):
        """The characters the program treats as gaps or unknown, of protein or nucleotide (-d) alignments."""
        return self.gap_characters.get("dna" if nucleotide else "protein", DEFAULT_GAP_CHARACTERS)

    def writes_dropped(self):
        """Whether the program writes the columns it removes."""
        return self.output_mode != "single"

    def command(self, n_seqs=None):
        """cmd_line, or cmd_line_large_n for an alignment of large_n sequences or more."""
        if self.cmd_large_n is not None and n_seqs is not None and n_seqs >= self.large_n:
            return self.cmd_large_n
        return self.cmd

    def commands(self, infn, kept_fn, dropped_fn=None, n_seqs=None):
        """[(command, the file it writes)]: the trimmed alignment (of n_seqs sequences), and the dropped columns if dropped_fn."""
        cmds = [(replace_variables(self.command(n_seqs), infn, kept_fn), kept_fn)]
        if dropped_fn is not None and self.output_mode == "paired":
            cmds.append((replace_variables(self.cmd_dropped, infn, dropped_fn), dropped_fn))
        return cmds

    def dropped_output(self, kept_fn, dropped_fn):
        """
        The file the program writes the dropped columns to: with output mode
        "single_with_complement" next to kept_fn (also when they are not
        wanted, dropped_fn None), otherwise dropped_fn.
        """
        if self.output_mode == "single_with_complement":
            return kept_fn + self.complement_suffix
        return dropped_fn

    def shell_command(self, infn, outfn, keep_dropped, n_seqs=None):
        """
        A run's trimming of infn to outfn by the program as one shell command,
        a step of the orthogroup's commands run and stopped as its MSA and tree
        commands are (msa_trees). It does not fail: each of its commands' exit
        code and messages go to files that finish_program reads, so a failure
        has the alignment kept untrimmed (_trim_with_program) rather than
        stopping the run.
        """
        kept_tmp, dropped_tmp, _ = _temp_files(self, outfn, keep_dropped)
        # (the program first: the run notes each kind of command by its first program, run_state;
        # the second, the columns removed, only if the first succeeded: test, a shell word, is
        # not taken for a program, programs.with_paths. Each in a group, { ...; }: the messages
        # and the guard are of all of a cmd_line of several commands, e.g. "a > t; mv t OUTPUT")
        q = shlex.quote
        status = _status_files(outfn)
        return "; ".join("%s{ %s; } 2> %s; echo $? > %s"
                         % ('test "$(cat %s)" = 0 && ' % q(status[0][0]) if i else "", cmd, q(stderr_fn), q(status_fn))
                         for i, ((cmd, _), (status_fn, stderr_fn))
                         in enumerate(zip(self.commands(infn, kept_tmp, dropped_tmp, n_seqs), status)))


_STDOUT_RE = re.compile(r"\bSTDOUT\b")
_METHODTHREAD_RE = re.compile(r"(?<![A-Za-z0-9_])METHODTHREADS?(?![A-Za-z0-9_])")     # (as command_runner's)


def replace_variables(cmd, infn, outfn):
    """A command of config.json with its file variables replaced (as utils.program_caller does)."""
    from ..utils import util
    return util.substitute_variables(_STDOUT_RE.sub(os.devnull, cmd), util.file_variables(infn, outfn))


def read_fasta(fn):
    """[(header, sequence)] of a FASTA file, the sequence without whitespace."""
    records = []
    with open(fn, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.startswith(">"):
                records.append([line[1:].strip(), []])
            elif records:
                records[-1][1].append("".join(line.split()))
    return [(header, "".join(parts)) for header, parts in records]


def read_headers(fn):
    """(the headers of a FASTA file, the length of its first sequence without whitespace), not holding its sequences."""
    headers, length = [], 0
    with open(fn, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.startswith(">"):
                headers.append(line[1:].strip())
            elif len(headers) == 1:
                length += len("".join(line.split()))
    return headers, length


def write_fasta(fn, records, n_char=80):
    with open(fn, "w", encoding="utf-8", newline="\n") as outfile:
        for header, seq in records:
            outfile.write(">" + header + "\n")
            outfile.write("".join(seq[i:i + n_char] + "\n" for i in range(0, len(seq), n_char)))


def _first_word(header):
    return header.split()[0] if header.split() else header


def program_output(headers, fn, require_residues, gaps=DEFAULT_GAP_CHARACTERS):
    """
    A trimming program's output fn as [(header, sequence)] in the order of the
    input's headers, or a reason it cannot be used: every sequence of the
    input (by the first word of its header: programs may change the rest),
    once, all the same length, and if require_residues each with a residue,
    a character not in gaps (the program's gaps and unknown characters: a
    tree needs every sequence; programs may leave out or empty some).
    """
    if not os.path.exists(fn) or os.path.getsize(fn) == 0:
        return None, "no output"
    try:
        records = read_fasta(fn)
    except (OSError, UnicodeDecodeError) as e:
        return None, "cannot read its output (%s)" % e
    seqs = {}
    for header, seq in records:
        word = _first_word(header)
        if word in seqs:
            return None, "sequence %s is in its output twice" % word
        seqs[word] = seq
    missing = [h for h in headers if _first_word(h) not in seqs]
    if missing:
        return None, "%d of the %d sequences are not in its output" % (len(missing), len(headers))
    extra = sorted(set(seqs) - {_first_word(h) for h in headers})
    if extra:
        return None, "its output has %d sequence(s) not in the input, e.g. %s" % (len(extra), extra[0])
    if len(seqs) != len(headers):
        return None, "the input's sequence names are not distinct by their first word"
    lengths = {len(seq) for seq in seqs.values()}
    if len(lengths) != 1:
        return None, "its output is not an alignment (sequences of different lengths)"
    if require_residues:
        if lengths == {0}:
            return None, "it removed every column"
        empty = [w for w, seq in seqs.items() if not seq.strip(gaps)]
        if empty:
            return None, "it left %d sequence(s) with no residues, e.g. %s" % (len(empty), empty[0])
    return [(h, seqs[_first_word(h)]) for h in headers], None


def _temp_files(program, outfn, keep_dropped):
    """(the trimmed alignment, the dropped columns if keep_dropped, the file the program writes them to) of a trimming of outfn."""
    kept_tmp = outfn + ".trim_kept.fa"
    dropped_tmp = outfn + ".trim_dropped.fa" if keep_dropped else None
    return kept_tmp, dropped_tmp, program.dropped_output(kept_tmp, dropped_tmp)


def _status_files(outfn):
    """(exit code, messages) of each of TrimProgram.shell_command's commands: the trimmed alignment's, the dropped columns'."""
    return [(outfn + ".trim_status%d" % i, outfn + ".trim_stderr%d" % i) for i in range(2)]


# (_temp_files and _status_files: the program's outputs, with a complement_suffix too)
_TEMP_PATTERNS = ("*.trim_kept.fa*", "*.trim_dropped.fa", "*.trim_status[01]", "*.trim_stderr[01]")


def remove_temp_files(directory):
    """
    The files of a run's trimming programs left in directory, by a run that
    was stopped: removed before the alignments are trimmed again, so that an
    output a program does not write again is not taken for its own.
    """
    for pattern in _TEMP_PATTERNS:
        for fn in glob.glob(os.path.join(glob.escape(directory), pattern)):
            remove_file(fn)


def run_program(program, infn, outfn, msa_min_seq, dropped_fn=None, env=None, run=None, nucleotide=False):
    """
    Trim infn to outfn (they may be the same file) with a trimming program
    (a TrimProgram), the columns it removes to dropped_fn if given. An
    alignment of msa_min_seq sequences or fewer is not trimmed, as by
    OrthoFinder's trimming. If its output cannot be used, the alignment is
    kept untrimmed (_trim_with_program).
    Returns (a note of why it was kept untrimmed, for checkpoint.txt, or
    None; a warning if the columns removed could not be written, or None;
    whether the program failed).
    run(command, env) -> (returncode, stdout, stderr): how a command is run
    (default subprocess, in a shell). nucleotide: the alignment is of
    nucleotides (-d), for the program's gaps (TrimProgram.gaps).
    """
    headers = read_headers(infn)
    n_seqs = len(headers[0])
    if n_seqs <= msa_min_seq:
        remove_file(dropped_fn)     # (of an earlier run; otherwise by _trim_with_program)
        copy_input_to_output(infn, outfn)
        return None, None, False

    def run_commands(kept_tmp, dropped_tmp):
        for fn in (kept_tmp, program.dropped_output(kept_tmp, dropped_tmp)):
            remove_file(fn)     # (of an earlier run)
        results = []
        for cmd, _ in program.commands(infn, kept_tmp, dropped_tmp, n_seqs):
            # (one thread: in a run, command_runner fills METHODTHREAD)
            returncode, _, stderr = (run or _run_shell)(_METHODTHREAD_RE.sub("1", cmd), env)
            results.append((returncode, stderr))
            if returncode != 0:
                break
        return results

    return _trim_with_program(program, infn, outfn, dropped_fn, run_commands, headers, nucleotide)


def finish_program(program, infn, outfn, dropped_fn=None, q_ran=True, nucleotide=False):
    """
    run_program for a run, after the program's shell_command(infn, outfn,
    dropped_fn is not None, N) has run, if q_ran (it is not run for an
    alignment of few sequences, msa_trees.trim_commands): its output used,
    or the reason it is not. Returns as run_program.
    """
    if not q_ran:
        remove_file(dropped_fn)     # (of an earlier run; otherwise by _trim_with_program)
        copy_input_to_output(infn, outfn)
        return None, None, False

    def read_status(kept_tmp, dropped_tmp):
        results = []
        for _, (status_fn, stderr_fn) in zip(program.commands(infn, kept_tmp, dropped_tmp), _status_files(outfn)):
            try:
                with open(status_fn, "r") as infile:
                    returncode = int(infile.read().strip())
            except (OSError, ValueError):
                returncode = None
            try:
                with open(stderr_fn, "rb") as infile:
                    stderr = infile.read()
            except OSError:
                stderr = None
            results.append((returncode, stderr))
        return results

    try:
        return _trim_with_program(program, infn, outfn, dropped_fn, read_status, nucleotide=nucleotide)
    finally:
        for fns in _status_files(outfn):
            for fn in fns:
                remove_file(fn)


def _keep_whole(infn, outfn, why):
    """A trimming program's output cannot be used (why): infn kept untrimmed, copied to outfn. Returns the note for checkpoint.txt."""
    copy_input_to_output(infn, outfn)
    return "%s, so the alignment was kept untrimmed" % why


def _trim_with_program(program, infn, outfn, dropped_fn, run_commands, headers=None, nucleotide=False):
    """
    run_program and finish_program: run_commands(kept_tmp, dropped_tmp) runs the
    program's commands (or reads how they ended), returning [(exit code, its
    stderr)] of those that ran, the exit code None if one did not finish.
    If the program removed every column, failed, wrote no alignment or left a
    sequence without residues, the alignment is kept untrimmed (_keep_whole).
    Otherwise its output is used as it is. The trimmed alignment is used whether or not the columns removed can be
    written: --keep-dropped, only an extra output, does not change the trees.
    headers: read_headers(infn), if read already.
    Returns as run_program.
    """
    remove_file(dropped_fn)     # (of an earlier run)
    headers, length = read_headers(infn) if headers is None else headers
    gaps = program.gaps(nucleotide)
    kept_tmp, dropped_tmp, dropped_out = _temp_files(program, outfn, dropped_fn is not None)
    produced = [kept_tmp] + ([dropped_out] if dropped_out is not None else [])   # (all removed after)
    try:
        results = run_commands(kept_tmp, dropped_tmp)
        failure = _command_failure(program, results[0])
        kept = None
        if failure is None:
            kept, reason = program_output(headers, kept_tmp, require_residues=False)
            if kept is None:
                failure = "%s: %s" % (program.name, reason)
            elif not kept[0][1]:
                return _keep_whole(infn, outfn, "%s removed every column" % program.name), None, False
        if failure is not None:
            return _keep_whole(infn, outfn, failure), None, True
        n_kept = len(kept[0][1])
        empty = [_first_word(h) for h, seq in kept if not seq.strip(gaps)]
        if empty:
            why = "%s left %d sequence(s) with no residues, e.g. %s" % (program.name, len(empty), empty[0])
            return _keep_whole(infn, outfn, why), None, False
        write_fasta(outfn, kept)
        if dropped_tmp is None:
            return None, None, False
        # (a second command, output mode paired, or with the first)
        failure = _command_failure(program, results[1]) if len(program.commands(infn, kept_tmp, dropped_tmp)) > 1 \
            else None
        dropped = None
        if failure is None:
            dropped, reason = program_output(headers, dropped_out, require_residues=False)
            if dropped is not None and len(dropped[0][1]) == 0:
                dropped, reason = None, "no output"     # (headers only: none written)
            if dropped is None and reason != "no output":
                failure = "%s: %s" % (program.name, reason)
            elif dropped is None and n_kept < length:
                # (no output is none removed, unless the alignment is shorter)
                failure = "%s removed %d column(s) but did not write them" % (program.name, length - n_kept)
            elif dropped is not None and n_kept + len(dropped[0][1]) != length:
                failure = "%s removed %d column(s) but wrote %d" % (program.name, length - n_kept, len(dropped[0][1]))
                dropped = None
        if failure is None and dropped is not None:
            write_fasta(dropped_fn, dropped)
        return None, (None if failure is None else "the columns removed were not written, " + failure), False
    finally:
        for fn in produced:
            remove_file(fn)


def _command_failure(program, result):
    """Why a command of the program failed ((exit code, stderr), None if it did not run), or None."""
    returncode, stderr = result if result is not None else (None, None)
    if returncode is None:
        return "%s did not finish" % program.name
    if returncode != 0:
        return "%s exited with code %d%s" % (program.name, returncode, _last_line(stderr))
    return None


def _run_shell(cmd, env):
    p = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return p.returncode, b"", p.stderr


def _last_line(stderr):
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    lines = [line for line in (stderr or "").splitlines() if line.strip()]
    return ": " + lines[-1].strip() if lines else ""


class Trimmer(object):
    """
    How a run trims its MSAs: OrthoFinder's trimming (program None) or a
    trimming program (a TrimProgram, by default ClipKIT), whether the dropped columns are kept
    (keep_dropped), and whether the alignments are of nucleotides (-d, for the
    program's gaps). Passed to trim_msa, in another process: plain data.
    """

    def __init__(self, program=None, keep_dropped=False, nucleotide=False):
        self.program = program
        self.keep_dropped = keep_dropped
        self.nucleotide = nucleotide

    def gaps(self):
        """The characters of gaps and unknown residues of the trimming (TrimProgram.gaps)."""
        return DEFAULT_GAP_CHARACTERS if self.program is None else self.program.gaps(self.nucleotide)

    def name(self):
        return "OrthoFinder" if self.program is None else self.program.name


# What happened to an alignment (trim_msa, for the run's summary)
TRIMMED = "trimmed"
ALL_KEPT = "every column kept"
UNTRIMMED = "kept untrimmed"      # (the program's output could not be used)
FEW = "too few sequences"          # (not trimmed, msa_min_seq)


def trim_msa(trimmer, aln_fn, msa_min_seq, dropped_fn=None, q_program_ran=True):
    """
    Trim an orthogroup's alignment in place (a step of its commands, see
    msa_trees): by trimmer, a Trimmer, the dropped columns to dropped_fn. A
    trimming program has already run (TrimProgram.shell_command), if
    q_program_ran. If its output cannot be used the alignment is kept
    untrimmed (_trim_with_program). Returns (the alignment's file name, what
    happened: TRIMMED, ALL_KEPT, UNTRIMMED or FEW, the number of sequences, a
    note for checkpoint.txt or None, a warning or None).
    """
    headers, length = read_headers(aln_fn)
    n_seqs = len(headers)
    name = os.path.basename(aln_fn)
    if trimmer.program is None:
        n_removed = run_in_process(aln_fn, aln_fn, msa_min_seq, DEFAULT_F, DEFAULT_N_MIN, DEFAULT_C, dropped_fn)
        status = FEW if n_seqs <= msa_min_seq else TRIMMED if n_removed else ALL_KEPT
        return name, status, n_seqs, None, None
    note, warning, _ = finish_program(trimmer.program, aln_fn, aln_fn, dropped_fn, q_program_ran,
                                      trimmer.nucleotide)
    if not q_program_ran:
        status = FEW
    elif note is not None:
        status = UNTRIMMED
    else:
        status = TRIMMED if read_headers(aln_fn)[1] < length else ALL_KEPT
    return name, status, n_seqs, note, warning
