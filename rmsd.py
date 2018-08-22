
from Bio import PDB
import numpy as np
import copy
import argparse
import sys


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation
    """
    D = len(V[0])
    N = len(V)
    rmsd = 0.0
    for v, w in zip(V, W):
        rmsd += sum([(v[i] - w[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)


def write_coordinates(atoms, V, title=""):
    """
    Print coordinates V with corresponding atoms to stdout in XYZ format.
    Parameters
    ----------
    atoms : list
        List of atomic types
    V : array
        (N,3) matrix of atomic coordinates
    title : string (optional)
        Title of molecule
    """
    N, D = V.shape

    print(str(N))
    print(title)

    for i in range(N):
        atom = atoms[i]
        atom = atom[0].upper() + atom[1:]
        print("{0:2s} {1:15.8f} {2:15.8f} {3:15.8f}".format(
                atom, V[i, 0], V[i, 1], V[i, 2]))


class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving. """
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)


def get_coordinates_pdb(id, filepath, chains):
    V = list()
    atoms = list()

    parser = PDB.PDBParser()
    structure = parser.get_structure(id, filepath)

    for model in structure:
        for chain in model:
            if chain.get_id() in chains:
                for atom in chain.get_atoms():
                    if atom.get_id() in ("H", "C", "N", "O", "S", "P") and atom.get_full_id()[3][0] == ' ':
                        atoms.append(atom.get_id())
                        V.append(np.asarray(atom.get_coord()))

    V = np.asarray(V)
    atoms = np.asarray(atoms)
    assert(V.shape[0] == atoms.size)
    return atoms, V

def main():

    # parser = argparse.ArgumentParser(usage='%(prog)s [options] structure_a structure_b')
    #
    # parser.add_argument('structure_a', metavar='structure_a', type=str, help='Structure in .xyz or .pdb format')
    # parser.add_argument('structure_b', metavar='structure_b', type=str)
    #
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    #
    # args = parser.parse_args()

    # As default, load the extension as format
    # if args.format == None:
    #     args.format = args.structure_a.split('.')[-1]


    pdbList = PDB.PDBList()

    pdb_fn = pdbList.retrieve_pdb_file("1acb", file_format="pdb")

    p_atoms, p_all = get_coordinates_pdb("1cse", pdb_fn, "I")
    q_atoms, q_all = get_coordinates_pdb("1cse", '/tmp/trans2.pdb', "I")

    P = copy.deepcopy(p_all)
    Q = copy.deepcopy(q_all)

    print(p_atoms.shape)
    print(q_atoms.shape)
    if np.count_nonzero(p_atoms != q_atoms):
        exit("Atoms not in the same order")

    normal_rmsd = rmsd(P, Q)
    print("Normal RMSD: {0}".format(normal_rmsd))

    return

if __name__ == "__main__":
    main()