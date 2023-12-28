# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
import io
from typing import Any, Sequence, Mapping, Optional
import string
import gzip

from . import residue_constants
from Bio.PDB import PDBParser  # pip install biopython
import numpy as np


FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclasses.dataclass(frozen=False)  # (frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    # modified: support-resolution-20230619
    # protein resolution
    resolution: any = None

    # modified: support-release-date-20230619
    # release date
    release_date: str = None

    # modified: support-multi-model-20230619
    n_models: int = 1


def from_pdb_string(
    pdb_str: str,
    chain_id=None,  # None = all chains, "A" = chain A, ["A", "B"] = chains A and B
    allow_multi_model=True,
    allow_insertion="keep",  # False: raise error, "keep": keep it, True and others: ignore
    skip_hetero=True,
    allow_lower_case_and_digit_chain_id=True,
    skip_x=True,
) -> Protein:
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)

    # modified: support-resolution-20230619
    resolution = structure.header["resolution"]
    if resolution is None:
        resolution = 0.0

    # modified: support-release-date-20230619
    release_date = structure.header["release_date"]

    models = list(structure.get_models())
    if len(models) != 1:
        if allow_multi_model == False:  # modified: support-multi-model-20230619
            raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
        else:
            pass
    model = models[0]  # TODO: support loading other models

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None:
            if hasattr(chain_id, '__contains__'):
                if chain.id not in chain_id:
                    continue
            elif chain.id != chain_id:
                continue

        insertion_offset = 0  # offset due to amino acid insertion
        for res in chain:
            if res.id[2] != " ":
                if allow_insertion == False:  # modified: support-insertion-20230619
                    raise ValueError(f"PDB contains an insertion code at chain {chain.id} and residue index {res.id[1]}. These are not supported.")
                elif allow_insertion == "keep":
                    insertion_offset += 1
                else:  # elif allow_insertion == "skip":
                    continue
            if res.id[0] != " ":
                if skip_hetero:
                    continue
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            if skip_x and res_shortname == "X":
                continue

            restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    if atom.name not in ["OP3"] and atom.name[0] not in ["H", "D"]:
                        print(f'Unknown atom: [{atom.name}] in chain [{chain.id}] res [{res.resname}]')
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1] + insertion_offset)  # modified: support-insertion-20230619
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    parents = None
    parents_chain_index = None
    if "PARENT" in pdb_str:
        parents = []
        parents_chain_index = []
        chain_id = 0
        for l in pdb_str.split("\n"):
            if "PARENT" in l:
                if not "N/A" in l:
                    parent_names = l.split()[1:]
                    parents.extend(parent_names)
                    parents_chain_index.extend([chain_id for _ in parent_names])
                chain_id += 1

    if not allow_lower_case_and_digit_chain_id:  # modified: support-lower-case-and-digit-chain-20230619
        chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    else:
        chain_id_mapping = {cid: n for n, cid in enumerate(PDB_CHAIN_IDS)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        parents=parents,
        parents_chain_index=parents_chain_index,
        resolution=resolution,  # modified: support-resolution-20230619
        n_models=len(models),  # modified: support-multi-model-20230619
        release_date=release_date,  # modified: support-release-date-20230619
    )


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents_chain_index is not None:
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]

    # TODO: what is parents here?
    if parents is None or len(parents) == 0:
        parents = ["N/A"]
    else:
        pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    out_pdb_lines = []
    lines = pdb_str.split("\n")

    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f"REMARK {remark}")

    parents_per_chain = None
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        if prot.parents_chain_index is not None:
            parent_dict = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(prot.parents)
    else:
        parents_per_chain = [["N/A"]]

    make_parent_line = lambda p: f"PARENT {' '.join(p)}"

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))

    chain_counter = 0
    for i, l in enumerate(lines):
        if "PARENT" not in l and "REMARK" not in l:
            out_pdb_lines.append(l)
        if "TER" in l and not "END" in lines[i + 1]:
            chain_counter += 1
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            out_pdb_lines.append(make_parent_line(chain_parents))

    return "\n".join(out_pdb_lines)


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    pdb_lines.append("MODEL     1")
    headers = get_pdb_headers(prot)
    if len(headers) > 0:
        pdb_lines.extend(headers)

    n = aatype.shape[0]
    atom_index = 1
    prev_chain_index = chain_index[0]
    chain_tags = PDB_CHAIN_IDS
    # Add all atom sites.
    for i in range(n):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""

            chain_tag = "A"
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = i == n - 1
        if chain_index is not None:
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = f"{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i]):>3} {chain_tag:>1}{residue_index[i]:>4}"
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if i != n - 1:
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def get_chain_ids(po):
    return np.unique(po.chain_index)


def chain_id_to_char(chain_id):
    return PDB_CHAIN_IDS[chain_id]


def get_chain_by_id(po, chain_id):
    mask = po.chain_index == chain_id
    return Protein(
        aatype=po.aatype[mask],
        atom_positions=po.atom_positions[mask],
        atom_mask=po.atom_mask[mask],
        residue_index=po.residue_index[mask],
        b_factors=po.b_factors[mask],
        chain_index=po.chain_index[mask],
        remark=po.remark,
        parents=po.parents,
        parents_chain_index=po.parents_chain_index,
    )


def to_sequence(po, special_na_token=True, only_protein=False, only_rna=False, only_dna=False):
    aatype = po.aatype
    assert sum([only_protein, only_rna, only_dna]) <= 1, "only one of only_protein, only_rna, only_dna can be True"
    if only_protein:
        aatype = aatype[aatype <= 20]  # 0 ~ 20 is protein
    elif only_rna:
        aatype = aatype[(21 <= aatype) & (aatype <= 25)]  # 21 ~ 25 is RNA
    elif only_dna:
        aatype = aatype[(26 <= aatype) & (aatype <= 30)]  # 26 ~ 30 is DNA
    return residue_constants.aatype_to_str_sequence(aatype, special_na_token)


def from_pdb_file(fname, *args, **kwargs):
    with open(fname) as f:
        return from_pdb_string(f.read(), *args, **kwargs)


def from_pdb_gz(fname, *args, **kwargs):
    with gzip.open(fname, 'rb') as f:
        return from_pdb_string(gzip.open(fname).read().decode('utf-8'), *args, **kwargs)


def to_pdb_file(po, fname):
    with open(fname, 'w') as f:
        f.write(to_pdb(po))


if __name__ == '__main__':
    # test with 1 data
    # po = from_pdb_file('biotools/notes/1b7f.pdb')
    # to_pdb_file(po, 'biotools/notes/1b7f_recon.pdb')
    # po = from_pdb_file('biotools/notes/1njy.pdb')
    # to_pdb_file(po, 'biotools/notes/1njy_recon.pdb')
    # po = from_pdb_file('biotools/notes/1vq8.pdb')
    # to_pdb_file(po, 'biotools/notes/1vq8_recon.pdb')
    po = from_pdb_file('biotools/notes/8gwo.pdb')
    to_pdb_file(po, 'biotools/notes/8gwo_recon.pdb')
    print([chain_id_to_char(i) for i in get_chain_ids(po)])
    print(to_sequence(po, special_na_token=True, only_protein=True))
    print(to_sequence(po, special_na_token=True, only_protein=False))
