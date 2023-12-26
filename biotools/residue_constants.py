import numpy as np

# fmt: off
# reference: https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/primary-sequences-and-the-pdb-format
# Standard (L-) Amino Acids: ALA, CYS, ASP, GLU, PHE, GLY, HIS, ILE, LYS, LEU, MET, ASN, PRO, GLN, ARG, SER, THR, VAL, TRP, TYR (20 in total)
# except: PYL (pyrrolysine)*, SEC (selenocysteine) *
# Deoxyribonucleotides: DA, DC, DG, DT
# except: DI
# Ribonucleotides: A, C, G, U
# except: I
# since ATCG overlap with protein, use []() instead
# since AUCG overlap with protein, use {}:; instead
restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK", # Protein unknown
    "{": "  A", # A, RNA Adenine
    ":": "  C", # C, RNA Cytosine
    ";": "  G", # G, RNA Guanine
    "}": "  U", # U, RNA Uracil
    "!": "RNU", # RNA unknown
    "[": " DA", # A, DNA Adenine
    "(": " DC", # C, DNA Cytosine
    ")": " DG", # G, DNA Guanine
    "]": " DT", # T, DNA Thymine
    "?": "DNU", # DNA unknown
}

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X", # Protein unknown
    "{", # A, RNA Adenine
    ":", # C, RNA Cytosine
    ";", # G, RNA Guanine
    "}", # U, RNA Uracil
    "!", # RNA unknown
    "[", # A, DNA Adenine
    "(", # C, DNA Cytosine
    ")", # G, DNA Guanine
    "]", # T, DNA Thymine
    "?", # DNA unknown
]

special_to_canonical = {
    "{": "A", ":": "C", ";": "G", "}": "U", "!": "X",
    "[": "A", "(": "C", ")": "G", "]": "T", "?": "X",
}
for k, v in restype_1to3.items():
    if k not in special_to_canonical:
        special_to_canonical[k] = k

# for atom types, see https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# for torsion angles, see https://www.researchgate.net/figure/Torsion-angles-of-nucleic-acids-A-Torsion-angles-along-the-backbone-a-to-z-within_fig4_221914038
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
    # add DNA/RNA atom types. exceed 37
    # backbone of nucleic acids
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'", # exists in RNA not in DNA
    "C1'",
    # Adenosine
    "N9",
    "C8",
    "N7",
    "C5",
    "C6",
    "N6",
    "N1",
    "C2",
    "N3",
    "C4",
    # Cytidine, these ones are the same as adenosine: N1, C2, N3, C4, C5, C6
    "O2",
    "N4",
    # Guanosine, these ones are the same as adenosine: N9, C8, N7, C5, C6, N1, C2, N3, C4
    "O6",
    "N2",
    # Uridine, these ones are the same as adenosine or cytidine: N1, C2, N3, C4, C5, C6
    "O4",
    # Thymidine
    "C7", # one more carbon than uridine
]

restype_name_to_atom_names = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "UNK": [],
    "  A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "  C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "  G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "  U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "RNU": [],
    " DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    " DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    " DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    " DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"], # Thymidine has one more carbon than uridine (C7) TODO check?
    "DNU": [],
} # TODO: add DNA/RNA atom types ? will exceed 14?

# fmt: on


def _make_standard_atom_mask():
    """Returns [num_res_types, num_atom_types] mask array."""
    # +1 to account for unknown (all 0s).
    mask = np.zeros([restype_num, atom_type_num], dtype=np.int32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3[restype_letter]
        atom_names = restype_name_to_atom_names[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask


restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)
restype_3to1 = {v: k for k, v in restype_1to3.items()}
atom_order = {atom: i for i, atom in enumerate(atom_types)}
atom_type_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)
STANDARD_ATOM_MASK = _make_standard_atom_mask()


def aatype_to_str_sequence(aatype, special_na_token=True):
    if special_na_token:
        return ''.join([restypes[aatype[i]] for i in range(len(aatype))])
    else:
        return ''.join([special_to_canonical[restypes[aatype[i]]] for i in range(len(aatype))])
