
from Bio import PDB
import numpy as np

def apply(chain, rx, ry, rz, tx, ty, tz):
    translation = np.array((tx, ty, tz), 'f')

    rotation_x = PDB.rotaxis2m(rx, PDB.Vector(1, 0, 0))
    rotation_y = PDB.rotaxis2m(-ry, PDB.Vector(0, 1, 0))
    rotation_z = PDB.rotaxis2m(rz, PDB.Vector(0, 0, 1))

    rotation = rotation_z.dot(rotation_y).dot(rotation_x)

    chain.transform(rotation.transpose(), translation)




if __name__ == "__main__":

    pdbList = PDB.PDBList()

    parser = PDB.PDBParser()
    writer = PDB.PDBIO()

    # pdb_path = pdbList.retrieve_pdb_file("1cse", file_format="pdb")
    #
    # (pdb_dir, pdb_fn) = os.path.split(pdb_path)
    # pdb_id = pdb_fn[:4]

    structure = parser.get_structure("1cse", "/tmp/trans.pdb")

    for model in structure:
        for chain in model:
            if chain.get_id() == 'I':
                #apply(chain, 2.21, -0.49, 2.25, 71.95, -2.97, 36.30)
                apply(chain, 1.85, -1.05, -1.74, 52.90, 24.89, 49.65)

    writer.set_structure(structure)
    writer.save("/tmp/trans2.pdb")