"""
The run report, WorkingDirectory/Run_Report/: what each stage of a run checked
and what it found that changed or left out results.

Summary.tsv has a row for every check, in run order, also those that found
nothing (so a report of zeros shows the checks ran). The items of a stage's
checks (orthogroups, files, species) are in a file of the stage, e.g.
Gene_Trees.tsv, written only if it has any. The report is written once, at the
end of the run, also if it fails or is stopped (write).

A check is marked as run by the main process (checked), with what it is out of
(e.g. the number of gene trees). Its items can be added by any process (add):
those of the worker processes are appended to a file each in
Run_Report/.pending/, read when the report is written. A restart keeps the rows
and items of the checks it does not run again (those of the stages before it),
from the report of the run it continues.
"""

import collections
import os
import re
import time

from . import file_handler

# What a problem does to the results
RESULTS_MISSING = "results missing"
RESULTS_CHANGED = "results changed"
INFORMATION = "information"

REPORT_DIR = "Run_Report"
SUMMARY_FN = "Summary.tsv"
PENDING_DIR = ".pending"
NOT_CHECKED = "not checked"

SUMMARY_HEADER = ("Stage", "Check", "Severity", "Count", "Out of", "Effect on results", "Details")
ITEMS_HEADER = ("Check", "Item", "Species", "Detail")

# The stages, in run order, and the file of their items
STAGES = collections.OrderedDict([
    ("Input", "Input.tsv"),
    ("Sequence searches", "Sequence_Searches.tsv"),
    ("Orthogroups", "Orthogroups.tsv"),
    ("Alignments", "Alignments.tsv"),
    ("Gene trees", "Gene_Trees.tsv"),
    ("Species tree", "Species_Tree.tsv"),
    ("HOGs", "HOGs.tsv"),
    ("Orthologs", "Orthologs.tsv"),
    ("Output", "Output.tsv"),
])

Check = collections.namedtuple("Check", "id stage text severity effect details")

CHECKS = [
    Check("input.not_fasta", "Input", "Files ignored, not FASTA files", INFORMATION,
          "These files are not species of the run", None),
    Check("input.not_utf8", "Input", "FASTA headers read as Windows-1252, not UTF-8", RESULTS_CHANGED,
          "Gene names may have other characters than intended", None),
    Check("input.renamed", "Input", "Gene names the same once cleaned, renamed", RESULTS_CHANGED,
          "These genes have other names than in the FASTA files", None),
    Check("input.shared_names", "Input", "Gene names in more than one species", INFORMATION,
          "Told apart by their species", None),
    Check("search.no_self_hits", "Sequence searches", "Species with no hits against itself", RESULTS_CHANGED,
          "Its genes are in no orthogroup with each other", None),
    Check("search.too_few_hits", "Sequence searches", "Species pairs with too few hits to normalise the scores",
          RESULTS_CHANGED, "The hits between these species are not used", None),
    Check("assign.no_genes_assigned", "Orthogroups", "New species with no genes assigned to the core orthogroups",
          RESULTS_MISSING, "These species are only in clade-specific orthogroups", None),
    Check("assign.mcl_fallback", "Orthogroups", "MCL orthogroups used as the fallback", INFORMATION,
          "The orthogroups are the MCL orthogroups, not the HOGs", None),
    Check("aln.untrimmed", "Alignments", "Trimming failed or unusable, alignment kept untrimmed", RESULTS_CHANGED,
          "The gene tree is of the untrimmed alignment", None),
    Check("aln.no_dropped", "Alignments", "Trimmed without the columns removed written (--keep-dropped)",
          INFORMATION, "No MultipleSequenceAlignments_Dropped file", None),
    Check("aln.not_for_species_tree", "Alignments", "Alignments not used for the species tree", RESULTS_CHANGED,
          "Left out of the concatenated alignment", None),
    Check("tree.missing", "Gene trees", "Gene tree missing or empty", RESULTS_MISSING,
          "No orthologs, duplications or HOGs of the orthogroup", None),
    Check("tree.one_gene", "Gene trees", "Gene tree of one gene", RESULTS_MISSING,
          "No orthologs, duplications or HOGs of the orthogroup", None),
    Check("tree.not_rooted", "Gene trees", "Gene tree could not be rooted", RESULTS_MISSING,
          "No orthologs, duplications or HOGs of the orthogroup", None),
    Check("tree.error", "Gene trees", "Internal error analysing a gene tree", RESULTS_MISSING,
          "No orthologs, duplications or HOGs of the orthogroup", None),
    Check("st.unreadable", "Species tree", "Gene trees not used for the species tree", RESULTS_CHANGED,
          "Left out of the species tree", None),
    Check("st.multiple_roots", "Species tree", "Equally good roots of the species tree", INFORMATION,
          "Only the first is analysed", None),
    Check("hog.failed", "HOGs", "Internal error inferring the HOGs of a gene tree", RESULTS_MISSING,
          "No HOGs of the orthogroup", None),
    Check("hog.topology", "HOGs", "Unexpected gene tree topology", RESULTS_CHANGED,
          "A duplication not placed on the species tree", None),
    Check("hog.tree_not_made", "HOGs", "HOG gene tree or alignment not made", RESULTS_MISSING,
          "No orthologs of the HOG's orthogroup", None),
    Check("olog.tree_missing", "Orthologs", "Gene tree of an orthogroup missing", RESULTS_MISSING,
          "No orthologs or duplications of the orthogroup", None),
    Check("olog.error", "Orthologs", "Internal error inferring the orthologs of a gene tree", RESULTS_MISSING,
          "No orthologs or duplications of the orthogroup", None),
    Check("olog.polytomies_resolved", "Orthologs", "Polytomies resolved with the species tree", INFORMATION,
          "Their orthologs are those of a binary tree", "Resolved_Gene_Trees/Polytomies.tsv"),
    Check("out.mcl_failed", "Output", "MCL orthogroups not written", RESULTS_MISSING,
          "No Orthogroups/MCL_Orthogroups files", None),
    # preview (--orthoxml: not in main, packaging/make_main.sh)
    Check("out.orthoxml_failed", "Output", "Orthogroups not written in OrthoXML", RESULTS_MISSING,
          "No OrthoXML file", None),
    # /preview
]
CHECKS_BY_ID = {c.id: c for c in CHECKS}
assert len(CHECKS_BY_ID) == len(CHECKS) and all(c.stage in STAGES for c in CHECKS)


class _State:
    def __init__(self):
        self.owner = None           # (the main process: its pid, and the run's token)
        self.token = None
        self.checked = {}           # {check id: out of}
        self.counts = {}            # {check id: count}, of checks counted, not itemised
        self.items = collections.defaultdict(list)    # {check id: [(item, species, detail)]}


_state = _State()


def start():
    """A new run, in this process (the main one): no checks yet."""
    global _state
    _state = _State()
    _state.owner = os.getpid()
    _state.token = "%d_%d" % (os.getpid(), time.time_ns())


def report_dir():
    """WorkingDirectory/Run_Report/ of the run, or None before the working directory is known."""
    wd = file_handler.handler.wd_current
    return None if wd is None else os.path.join(wd, REPORT_DIR)


def _clean(value):
    return re.sub(r"[\t\r\n]+", " ", str(value)).strip()


def checked(check_id, out_of=None, add=False):
    """
    The check ran (in the main process), of out_of items (e.g. gene trees), if
    given; add: in addition to those it already ran on in this run (e.g. a
    second batch of alignments).
    """
    if check_id not in CHECKS_BY_ID:
        raise KeyError(check_id)
    before = _state.checked.get(check_id)
    if out_of is None:
        out_of = before
    elif add and before is not None:
        out_of += before
    _state.checked[check_id] = out_of


def count(check_id, n, out_of=None, add=False):
    """A check with a count (e.g. polytomies resolved), in the main process; add: as checked."""
    checked(check_id, out_of, add)
    _state.counts[check_id] = n + (_state.counts.get(check_id, 0) if add else 0)


def add(check_id, item="", species="", detail=""):
    """
    An item a check found (e.g. an orthogroup), in any process. Never fails:
    the report must not stop the run.
    """
    if check_id not in CHECKS_BY_ID:
        raise KeyError(check_id)
    row = (_clean(item), _clean(species), _clean(detail))
    if _state.owner is None:
        return
    if os.getpid() == _state.owner:
        _state.items[check_id].append(row)
        return
    try:
        d = report_dir()
        if d is None:
            return
        d = os.path.join(d, PENDING_DIR)
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, "%s_%d.tsv" % (_state.token, os.getpid()))
        with open(fn, "a", encoding="utf-8", newline="\n") as outfile:
            outfile.write("\t".join((check_id,) + row) + "\n")
    except Exception:
        pass


def _read_pending(d):
    """The items the worker processes added (and the pending files removed, also those of other runs)."""
    pending = os.path.join(d, PENDING_DIR)
    if not os.path.isdir(pending):
        return
    for name in sorted(os.listdir(pending)):
        fn = os.path.join(pending, name)
        if name.startswith(_state.token + "_"):
            with open(fn, encoding="utf-8") as infile:
                for line in infile:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) == 4 and fields[0] in CHECKS_BY_ID:
                        _state.items[fields[0]].append(tuple(fields[1:]))
        os.remove(fn)
    os.rmdir(pending)


def _read_tsv(fn):
    if not os.path.exists(fn):
        return []
    with open(fn, encoding="utf-8") as infile:
        rows = [line.rstrip("\n").split("\t") for line in infile]
    return rows[1:]


def _species_names():
    """{species ID: name} of the run (SpeciesIDs.txt), {} if it has none yet."""
    fn = os.path.join(file_handler.handler.wd_current, "SpeciesIDs.txt")
    names = {}
    if os.path.exists(fn):
        with open(fn, encoding="utf-8") as infile:
            for line in infile:
                if ": " in line:
                    i, name = line.rstrip("\n").split(": ", 1)
                    names[i.strip()] = os.path.splitext(name.strip())[0]
    return names


def _species_column(value, names):
    """Species IDs (e.g. "3, 5") as their names."""
    if not value or not re.fullmatch(r"\d+(, \d+)*", value):
        return value
    return ", ".join(names.get(i, i) for i in value.split(", "))


def _natural(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def write():
    """
    Write the report (once, at the end of the run). Returns the line for the
    screen and Log.txt, if this run left results out ([] otherwise), or None
    if there is no report (no working directory).
    """
    if _state.owner is None or os.getpid() != _state.owner:
        return None
    d = report_dir()
    if d is None or not os.path.isdir(file_handler.handler.wd_current):
        return None
    os.makedirs(d, exist_ok=True)
    _read_pending(d)
    ran = set(_state.checked) | set(_state.counts) | {k for k, v in _state.items.items() if v}
    # (the rows and items of the checks this run did not run: those of the run it continues)
    previous = {(r[0], r[1]): r for r in _read_tsv(os.path.join(d, SUMMARY_FN)) if len(r) == len(SUMMARY_HEADER)}
    names = _species_names()
    summary = []
    missing = []
    for stage, stage_fn in STAGES.items():
        stage_checks = [c for c in CHECKS if c.stage == stage]
        carried = {c.text for c in stage_checks if c.id not in ran}
        items = [r for r in _read_tsv(os.path.join(d, stage_fn)) if len(r) == len(ITEMS_HEADER) and r[0] in carried]
        for c in stage_checks:
            if c.id not in ran:
                row = previous.get((c.stage, c.text))
                summary.append(list(row) if row else [c.stage, c.text, c.severity, NOT_CHECKED, "", "", ""])
                continue
            these = _state.items.get(c.id, [])
            n = _state.counts.get(c.id, len(these))
            out_of = _state.checked.get(c.id)
            items.extend([c.text, item, _species_column(sp, names), detail] for item, sp, detail in these)
            details = (c.details or (stage_fn if these else "")) if n else ""
            summary.append([c.stage, c.text, c.severity, str(n), "" if out_of is None else str(out_of),
                            c.effect if n else "", details])
            if n and c.severity == RESULTS_MISSING:
                missing.append("%s (%d)" % (c.text, n))
        order = {c.text: i for i, c in enumerate(stage_checks)}
        items.sort(key=lambda r: (order.get(r[0], len(order)), _natural(r[1]), r[2], r[3]))
        fn = os.path.join(d, stage_fn)
        if items:
            _write_tsv(fn, ITEMS_HEADER, items)
        elif os.path.exists(fn):
            os.remove(fn)
    _write_tsv(os.path.join(d, SUMMARY_FN), SUMMARY_HEADER, summary)
    if not missing:
        return []
    return ["WARNING: results are missing: %s. See %s" % ("; ".join(missing), os.path.join(d, SUMMARY_FN))]


def _write_tsv(fn, header, rows):
    with open(fn, "w", encoding="utf-8", newline="\n") as outfile:
        outfile.write("\t".join(header) + "\n")
        for row in rows:
            outfile.write("\t".join(row) + "\n")
