
from Bio import PDB
from Bio import SeqIO
from Bio.SeqIO import PdbIO

def main():
    pdbList = PDB.PDBList()
    #pdb = '1ggg'
    #pdb_fn = pdbList.retrieve_pdb_file(pdb, file_format="pdb")

    pdb = '663817-9353'
    pdb_fn = '/tmp/663817-9353/ff0.pdb'
    with open(pdb_fn) as handle:
        records = PdbIO.PdbAtomIterator(handle)
        records_list = list(records)

        with open("/tmp/%s.txt" % pdb, "w") as out_file:
            SeqIO.write(records_list, out_file, "fasta")
            out_file.close()

if __name__ == '__main__':
    main()