
import numpy as np
import os

def parse_gcnn_file(atoms_count, gcnn_file):
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

    def __init__(self, contact_map_directory):
        self.contact_map_directory = contact_map_directory

    def get_contact_map_gcnn(self, morph_id, residue_count):
        return parse_gcnn_file(residue_count, os.path.join(self.contact_map_directory, morph_id + '.gcnn'))

    def get_contact_map_rr(self, morph_id, residue_count):
        return parse_rr_file(residue_count, os.path.join(self.contact_map_directory, morph_id + '.rr'))



