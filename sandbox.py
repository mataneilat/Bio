
from Bio import PDB
from bio.chain_filter import ChainFilter

if __name__ == "__main__":
    """ Parses PDB id's desired chains, and creates new PDB structures. """
    # import sys
    # if not len(sys.argv) == 2:
    #     print("Usage: $ python %s 'pdb.txt'" % __file__)
    #     sys.exit()

    pdb_textfn = "/tmp/1acb.pdb"

    pdbList = PDB.PDBList()
    splitter = ChainFilter("/tmp")  # Change me.

    pdb_fn = pdbList.retrieve_pdb_file("1acb", file_format="pdb")
    splitter.make_pdb(pdb_fn, ["E"])
    splitter.make_pdb(pdb_fn, ["I"])


