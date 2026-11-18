#!/usr/bin/env python3

"""
Write the orthologs of a run's polytomies one row per pair of children.

A run writes the orthologs of a polytomy (a gene tree node of more than two
children) one row per pair of the species sets of its children, for example
"A1, A2, A3" against "B1, B2", rather than one row per pair of children (A1
against B1, A1 against B2 and so on). The gene pairs and the counts are the
same either way, but per pair of children a polytomy of 50,000 identical
sequences would have billions of rows.

This writes a copy of the ortholog and putative xenologue files with the rows
of every polytomy one per pair of children, to Expanded_Orthologs/Orthologs and
Expanded_Orthologs/Putative_Xenologs of the results directory (or of the
directory given with -o). The other rows are copied as they are, and the run's
own files are not changed. It reads the polytomies from
Resolved_Gene_Trees/Polytomies.tsv and their children from
Resolved_Gene_Trees/Resolved_Gene_Trees.txt. A polytomy that the run
resolved (with a child of several species) has rows of binary nodes, as any
other node, so they are copied as they are. The rows can be many: about 2.7
billion and 160 GB for a node of 50,000 children, with as much again of
temporary files next to the output while they are written.

Examples
--------
expand_polytomies Results_*/

expand_polytomies Results_*/ -o /scratch/expanded
"""

import argparse
import gzip
import io
import os
import sys
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
src_dir = os.path.join(repo_root, "src")

if os.path.isdir(os.path.join(src_dir, "orthofinder")) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from orthofinder.orthology import ortholog_writers, tree_analyser
from orthofinder.phylo.tree import Tree

OUTPUT_DIR = "Expanded_Orthologs"
TREES_DIR = "Resolved_Gene_Trees"
TREES_FN = "Resolved_Gene_Trees.txt"
# The rows of an orthogroup held before they are added to its runs (expand_file)
HELD_ROWS = 100000
AMBIGUOUS = -1


def open_text(path, mode):
    """A text file (gzipped if .gz) as OrthoFinder writes them: UTF-8, "\n" line endings."""
    newline = None if "r" in mode else "\n"
    if path.endswith(".gz"):
        # (no time or name in the header, as the run writes them)
        binary = gzip.GzipFile(path, mode + "b", compresslevel=6, mtime=0) if "w" in mode else gzip.open(path, "rb")
        return io.TextIOWrapper(binary, encoding="utf-8", newline=newline)
    return open(path, mode, encoding="utf-8", newline=newline)


def read_polytomies(results):
    """{orthogroup: [node]} of the polytomies with orthologs (Polytomies.tsv), and {(orthogroup, node): ortholog pairs}."""
    fn = os.path.join(results, TREES_DIR, tree_analyser.POLYTOMIES_FN)
    nodes, pairs = defaultdict(list), {}
    if not os.path.exists(fn):
        return nodes, pairs
    with open(fn, encoding="utf-8") as infile:
        header = next(infile).rstrip("\n").split("\t")
        i_pairs = header.index("Ortholog Pairs")
        for line in infile:
            row = line.rstrip("\n").split("\t")
            n = int(row[i_pairs])
            if n:           # (a polytomy without orthologs has no rows to expand)
                nodes[row[0]].append(row[1])
                pairs[(row[0], row[1])] = n
    return nodes, pairs


class Polytomy:
    """
    The child of each gene of a polytomy (node), by the gene's name in the
    tree (<species>_<gene>, or <gene> with -X), by (species, gene) and by
    <gene> alone if that is of one child only.
    """

    def __init__(self, og, node, species):
        self.og = og
        self.name = node.name
        self.child_of = {}
        self.by_species = {}
        self.plain = {}
        for i, child in enumerate(node.get_children()):
            for leaf in child.get_leaf_names():
                self.child_of[leaf] = i if self.child_of.get(leaf, i) == i else AMBIGUOUS
                sp = next((s for s in species if leaf.startswith(s + "_")), None)
                if sp is not None:
                    gene = leaf[len(sp) + 1:]
                    self.by_species[(sp, gene)] = i
                    self.plain[gene] = i if self.plain.get(gene, i) == i else AMBIGUOUS

    def child(self, gene, species=None):
        """The child of a gene of the ortholog files (of species, if known), or None if not below the node."""
        i = self.by_species.get((species, gene)) if species is not None else None
        if i is None:
            i = self.child_of.get(gene)
        if i is None and species is None:
            i = self.plain.get(gene)
        if i == AMBIGUOUS:
            raise ValueError("%s node %s: the gene %s is named the same in more than one of its children "
                             "(a run with -X?), so its rows cannot be split by child"
                             % (self.og, self.name, gene))
        return i


def read_tree_polytomies(results, nodes, species):
    """{orthogroup: [Polytomy]} of the polytomies of nodes ({orthogroup: [node]}), from the resolved gene trees."""
    fn = os.path.join(results, TREES_DIR, TREES_FN)
    found = {}
    species = sorted(species, key=len, reverse=True)       # (the longest species name that is a prefix)
    with open(fn, encoding="utf-8") as infile:
        for line in infile:
            og, _, newick = line.partition(": ")
            if og not in nodes:
                continue
            t = Tree(newick.strip(), format=1)
            wanted = set(nodes[og])
            polytomies = [Polytomy(og, n, species) for n in t.traverse() if n.name in wanted]
            if len(polytomies) != len(wanted):
                raise ValueError("%s: the nodes %s of %s are not in its tree"
                                 % (fn, ", ".join(sorted(wanted - {p.name for p in polytomies})), og))
            found[og] = polytomies
    missing = set(nodes) - set(found)
    if missing:
        raise ValueError("%s: no tree of %s" % (fn, ", ".join(sorted(missing))))
    return found


def split_side(genes, polytomy, species):
    """{child: [genes]} of the genes of one side of a row, or None if one is not below the polytomy."""
    by_child = defaultdict(list)
    for gene in genes:
        i = polytomy.child(gene, species)
        if i is None:
            return None
        by_child[i].append(gene)
    return by_child


def expand_row(fields, polytomies, kind, species):
    """
    The rows per pair of children of a row (its fields) of the file of
    species, its polytomy and its gene pairs, if it is of one (its genes
    below it, in more than one of its children); otherwise None. kind: "o"
    (Orthogroup, Species, genes, genes: those of species, then of Species)
    or "x" (Orthogroup, genes, Other genes: those of species, then of any).
    """
    if kind == "o":
        og, other_sp, mine, theirs = fields
        head = (og, other_sp)
    else:
        (og, mine, theirs), other_sp = fields, None
        head = (og,)
    mine, theirs = mine.split(", "), theirs.split(", ")
    for p in polytomies:
        left = split_side(mine, p, species)
        if left is None:
            continue
        right = split_side(theirs, p, other_sp)
        if right is None or len(set(left) | set(right)) < 2:
            continue        # (a row of a node below the polytomy, within one child)
        rows = ["\t".join(head + (", ".join(genes_i), ", ".join(genes_j))) + "\n"
                for _, genes_i in sorted(left.items()) for _, genes_j in sorted(right.items())]
        return rows, p, len(mine) * len(theirs)
    return None


def species_files(d):
    """{species name: file} of the files of each species in d, <species>.tsv[.gz]."""
    files = {}
    if os.path.isdir(d):
        for name in os.listdir(d):
            for ext in (".tsv.gz", ".tsv"):
                if name.endswith(ext) and os.path.isfile(os.path.join(d, name)):
                    files[name[:-len(ext)]] = os.path.join(d, name)
                    break
    return files


def expand_file(src, dst, kind, species, tree_polytomies, runs, pairs_found):
    """
    src to dst with the rows of the polytomies of tree_polytomies per pair of
    children, the rows of each orthogroup with a polytomy sorted as the run
    sorts them (runs, a SortedRuns). pairs_found[(og, node)]: the gene pairs
    of its ortholog rows (each pair is in the files of both its species).
    """
    key = (kind, species)
    iog_of = {og: int(og[2:]) for og in tree_polytomies}
    with open_text(src, "r") as infile, open_text(dst, "w") as outfile:
        outfile.write(next(infile, ""))
        current = None
        held = []

        def add_held():
            if held:
                runs.add(iog_of[current], key, tree_analyser.sorted_rows(held))
                held.clear()

        def finish():
            add_held()
            iog = iog_of[current]
            if key in runs.runs.get(iog, {}):
                runs.merge(iog, key, outfile.write)
                del runs.runs[iog][key]

        # (by the orthogroup of each block of rows, not their order: the rows of the
        # orthogroups of 2 or 3 genes are after those of the trees)
        for line in infile:
            og = line[:line.find("\t")]
            if current is not None and og != current:
                finish()
                current = None
            if og not in tree_polytomies:
                outfile.write(line)
                continue
            current = og
            expanded = expand_row(tuple(line.rstrip("\n").split("\t")), tree_polytomies[og], kind, species)
            if expanded is None:
                held.append(line if line.endswith("\n") else line + "\n")
            else:
                rows, p, n_pairs = expanded
                held.extend(rows)
                if kind == "o":
                    pairs_found[(og, p.name)] += n_pairs
            if len(held) >= HELD_ROWS:
                add_held()
        if current is not None:
            finish()


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="expand_polytomies",
        description="Write the orthologs of a run's polytomies one row per pair of children.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("Examples"):])
    parser.add_argument("results", metavar="RESULTS", help="an OrthoFinder results directory")
    parser.add_argument("-o", "--output", metavar="DIR",
                        help="the directory to write Orthologs and Putative_Xenologs to (default %s/ of "
                             "the results directory)" % OUTPUT_DIR)
    a = parser.parse_args(args)

    d_orthologs = os.path.join(a.results, "Orthologs")
    if not os.path.isdir(d_orthologs) or not os.path.exists(os.path.join(a.results, TREES_DIR, TREES_FN)):
        parser.error("no Orthologs and %s/%s in %s" % (TREES_DIR, TREES_FN, a.results))
    nodes, pairs = read_polytomies(a.results)
    if not nodes:
        print("No polytomies with orthologs in %s: the ortholog files have every row per pair of children"
              % a.results)
        return 0
    d_out = a.output or os.path.join(a.results, OUTPUT_DIR)
    if os.path.realpath(d_out) == os.path.realpath(a.results):
        parser.error("the output directory is the results directory: its files would be overwritten")
    os.makedirs(d_out, exist_ok=True)

    ortholog_files = species_files(d_orthologs)
    runs = ortholog_writers.SortedRuns(spill_dir=d_out)
    pairs_found = defaultdict(int)
    try:
        tree_polytomies = read_tree_polytomies(a.results, nodes, ortholog_files)
        for kind, name in (("o", "Orthologs"), ("x", "Putative_Xenologs")):
            d = os.path.join(d_out, name)
            os.makedirs(d, exist_ok=True)
            for species, src in sorted(species_files(os.path.join(a.results, name)).items()):
                expand_file(src, os.path.join(d, os.path.basename(src)), kind, species, tree_polytomies,
                            runs, pairs_found)
        # (each gene pair is in the files of both its species)
        wrong = [(og, node) for (og, node), n in sorted(pairs.items()) if pairs_found[(og, node)] != 2 * n]
        if wrong:
            raise ValueError("the ortholog rows of %s node %s are not those of %s: the files are not of "
                             "the same run as %s/%s"
                             % (wrong[0][0], wrong[0][1], tree_analyser.POLYTOMIES_FN, TREES_DIR,
                                tree_analyser.POLYTOMIES_FN))
    except (ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    finally:
        runs.close()
    if any(e.name.startswith("Orthologs_") and e.is_dir() for e in os.scandir(d_orthologs)):
        # (as -pof: the pairwise files too)
        ortholog_writers.split_ortholog_files(os.path.join(d_out, "Orthologs"))
    print("Written to %s" % d_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
