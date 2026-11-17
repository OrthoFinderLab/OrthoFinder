#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:30:34 2024

@author: user

Create primary transcript FASTA files from NCBI ZIP genome downloads.

INPUT
  The input may be:
    - a single .zip file
    - a directory containing .zip files

The script expects each ZIP archive to contain:
  - a protein FASTA file ending with: protein.faa
  - a GFF file ending with: genomic.gff

For each gene, the longest protein sequence is written to the output FASTA file.
A gene is a locus, by its NCBI GeneID (else its locus_tag, else its gene symbol),
not a gene symbol alone, which different loci can share.
"""

import os
import sys
import argparse
import zipfile
import tempfile
from collections import defaultdict

from Bio import SeqIO


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def find_files(root_dir):
    protein_fasta = None
    gff_file = None

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith("protein.faa"):
                protein_fasta = os.path.join(root, filename)
            elif filename.endswith("genomic.gff"):
                gff_file = os.path.join(root, filename)

    return protein_fasta, gff_file


def parse_fasta(fasta_file):
    protein_dict = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_dict[record.id] = str(record.seq)

    return protein_dict


def gff_attributes(text):
    """The attributes (column 9) of a GFF3 line, as {key: value}."""
    attributes = {}
    for item in text.split(";"):
        key, sep, value = item.partition("=")
        if sep:
            attributes[key.strip()] = value.strip()
    return attributes


def gene_locus(attributes):
    """
    (the gene locus of a CDS's attributes, its gene symbol): the locus by its
    NCBI GeneID (Dbxref), else its locus_tag, else its symbol (gene). A symbol
    alone is not a locus: different loci can have the same symbol.
    """
    symbol = attributes.get("gene")
    for xref in attributes.get("Dbxref", "").split(","):
        if xref.startswith("GeneID:"):
            return xref, symbol
    if "locus_tag" in attributes:
        return "locus_tag:" + attributes["locus_tag"], symbol
    return (None if symbol is None else "gene:" + symbol), symbol


def parse_gff(gff_file):
    """{protein_id: (its gene locus, its gene symbol)} of the CDSs of a GFF3 file."""
    gene_map = {}

    with open(gff_file, "r", encoding="utf-8") as gff:
        for line in gff:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")

            if len(fields) < 9:
                continue

            attributes = gff_attributes(fields[8])
            protein_id = attributes.get("protein_id")
            locus, symbol = gene_locus(attributes)
            if protein_id is not None and locus is not None:
                gene_map[protein_id] = (locus, symbol)

    return gene_map


def get_longest_transcripts(protein_dict, gene_map):
    """{gene locus: (protein_id, sequence, symbol)}: the longest protein of each locus."""
    gene_transcripts = defaultdict(list)

    for protein_id, sequence in protein_dict.items():
        if protein_id in gene_map:
            locus, symbol = gene_map[protein_id]
            gene_transcripts[locus].append((protein_id, sequence, symbol))

    longest_transcripts = {}

    for locus, transcripts in gene_transcripts.items():
        longest_transcripts[locus] = max(transcripts, key=lambda x: len(x[1]))

    return longest_transcripts


def write_longest_fasta(output_file, longest_transcripts):
    # (by gene symbol as before, a locus without one by its locus)
    with open(output_file, "w", encoding="utf-8", newline="\n") as out_fasta:
        for locus, (protein_id, sequence, symbol) in sorted(
                longest_transcripts.items(), key=lambda item: (item[1][2] or item[0], item[0])):
            out_fasta.write(f">{protein_id} gene={symbol or locus.split(':', 1)[1]}\n")
            out_fasta.write(f"{sequence}\n")


def process_zip_file(zip_file_path, output_dir):
    print("\nProcessing ZIP: %s" % zip_file_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        unzip_file(zip_file_path, tmpdir)

        protein_fasta, gff_file = find_files(tmpdir)

        if not protein_fasta or not gff_file:
            print("WARNING: could not find required files in %s" % zip_file_path)
            return

        protein_dict = parse_fasta(protein_fasta)
        gene_map = parse_gff(gff_file)
        longest_transcripts = get_longest_transcripts(protein_dict, gene_map)

        if not longest_transcripts:
            print("WARNING: no gene/protein mappings found in %s" % zip_file_path)
            return

        base = os.path.basename(zip_file_path)
        if base.endswith(".zip"):
            base = base[:-4]

        output_fasta_path = os.path.join(output_dir, base + "_lt.fasta")

        write_longest_fasta(output_fasta_path, longest_transcripts)

        print("Found %d proteins" % len(protein_dict))
        print("Found %d protein-to-gene mappings" % len(gene_map))
        print("Wrote %d longest transcripts" % len(longest_transcripts))
        print("Output: %s" % output_fasta_path)


def find_zip_files(input_path, output_dir=None):
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir) if output_dir is not None else None

    if os.path.isfile(input_path):
        if input_path.endswith(".zip"):
            return [input_path]
        print("ERROR: input file is not a .zip archive: %s" % input_path)
        sys.exit(1)

    if os.path.isdir(input_path):
        zip_files = []

        for root, dirs, files in os.walk(input_path):
            root_abs = os.path.abspath(root)

            if output_dir is not None:
                dirs[:] = [
                    d for d in dirs
                    if os.path.abspath(os.path.join(root_abs, d)) != output_dir
                ]

            for filename in files:
                if filename.endswith(".zip"):
                    zip_files.append(os.path.join(root_abs, filename))

        return sorted(zip_files)

    print("ERROR: input path does not exist: %s" % input_path)
    sys.exit(1)

def process_extracted_directory(input_dir, output_dir):
    print("\nProcessing directory: %s" % input_dir)

    protein_fasta, gff_file = find_files(input_dir)

    if not protein_fasta or not gff_file:
        print("WARNING: could not find required files in %s" % input_dir)
        return

    protein_dict = parse_fasta(protein_fasta)
    gene_map = parse_gff(gff_file)
    longest_transcripts = get_longest_transcripts(protein_dict, gene_map)

    if not longest_transcripts:
        print("WARNING: no gene/protein mappings found in %s" % input_dir)
        return

    base = os.path.basename(os.path.abspath(input_dir))
    output_fasta_path = os.path.join(output_dir, base + "_lt.fasta")

    write_longest_fasta(output_fasta_path, longest_transcripts)

    print("Found %d proteins" % len(protein_dict))
    print("Found %d protein-to-gene mappings" % len(gene_map))
    print("Wrote %d longest transcripts" % len(longest_transcripts))
    print("Output: %s" % output_fasta_path)

def main(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Input .zip file or directory containing .zip files",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory [default: input_dir/primary_transcripts]",
    )

    parsed = parser.parse_args(args)

    input_path = os.path.abspath(parsed.input)

    if parsed.output is None:
        if os.path.isdir(input_path):
            output_dir = os.path.join(input_path, "primary_transcripts")
        else:
            output_dir = os.path.join(
                os.path.dirname(input_path),
                "primary_transcripts",
            )
    else:
        output_dir = os.path.abspath(parsed.output)

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        zip_files = find_zip_files(input_path, output_dir)

        for zip_file in zip_files:
            process_zip_file(zip_file, output_dir)

    elif os.path.isdir(input_path):
        zip_files = find_zip_files(input_path, output_dir)

        if zip_files:
            for zip_file in zip_files:
                process_zip_file(zip_file, output_dir)
        else:
            process_extracted_directory(input_path, output_dir)

    else:
        print("ERROR: input path does not exist: %s" % input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()