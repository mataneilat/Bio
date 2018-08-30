"""
    This module contains the contact map repository and its parsers
"""
import numpy as np
import os


def parse_gcnn_file(atoms_count, gcnn_file):
    """
    Parses a gcnn file that is received as a part of RaptorX's output.
    The gcnn file is simply a matrix in text format.

    :param atoms_count: The number of atoms to define the returned matrix dimensions.
    :param gcnn_file:   The path to the gcnn file
    :return:    A matrix representing the probabilities of each two residues being in contact.
    """
    if gcnn_file is None:
        return None
    if not os.path.isfile(gcnn_file):
        return None
    mat = np.zeros(shape=(atoms_count, atoms_count))
    with open(gcnn_file) as handle:
        content = handle.readlines()
        for i, line in enumerate(content):
            for j, element in enumerate(line.split()):
                mat[i,j] = float(element)

    print(mat[mat==0].size)
    return mat


def parse_rr_file(atoms_count, rr_file):
    """
    Parses an rr file that is received as a part of RatporX's output
    :param atoms_count: The number of atoms to define the returned matrix dimensions.
    :param rr_file:     The path to the gcnn file
    :return:    A matrix representing the probabilities of each two residues being in contact.
                Pairs of residues for which no contact information is given are filled with -1
    """
    if rr_file is None:
        return None
    if not os.path.isfile(rr_file):
        return None
    mat = np.full(shape=(atoms_count, atoms_count), fill_value=-1.0)
    for i in range(atoms_count):
        mat[i,i] = 1
    with open(rr_file) as handle:
        content = handle.readlines()
        for line in content:
            tokens = line.split(' ')
            if len(tokens) == 5:
                i = int(tokens[0]) - 1
                j = int(tokens[1]) - 1
                p = float(tokens[4].strip('\n'))
                mat[i,j] = p
                mat[j,i] = p

    return mat


class ContactMapRepository:
    """
    The repository contact maps.
    The repository works in a lazy manner, meaning that files are only parsed when requested.
    """
    def __init__(self, contact_map_directory):
        """
        Constructs the repository with the directory containing the contact map files.

        :param contact_map_directory:   The directory containing the contact map files.
        """
        self.contact_map_directory = contact_map_directory

    def get_contact_map_gcnn(self, morph_id, residue_count):
        """
        Parses and returns the contact map corresponding to the given morph id, given as a .gcnn file.

        :param morph_id:    The morph for which contact map is required.
        :param residue_count:   The protein's residue count.
        :return:    The contact map of the protein represented by the given morph id.
        """
        return parse_gcnn_file(residue_count, os.path.join(self.contact_map_directory, morph_id + '.gcnn'))

    def get_contact_map_rr(self, morph_id, residue_count):
        """
        Parses and returns the contact map corresponding to the given morph id, given as an .rr file.

        :param morph_id:    The morph for which contact map is required.
        :param residue_count:   The protein's residue count.
        :return:    The contact map of the protein represented by the given morph id.
        """
        return parse_rr_file(residue_count, os.path.join(self.contact_map_directory, morph_id + '.rr'))



