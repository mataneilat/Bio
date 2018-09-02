
from Bio import SeqIO
from Bio.SeqIO import PdbIO
from morphs_repository import *
from morphs_atlas_parser import *
from utils import *

morphs_repository = MorphsRepository(parse_morphs_atlas_from_text('./hingeatlas.txt'),
                                    '/Users/mataneilat/Downloads/hinge_atlas_nonredundant')

def main():

    morphs_ids = list(morphs_repository.atlas_morphs.keys())

    morphs_chunks = list(partition_generator(morphs_ids, 20))

    def pdb_to_fasta(morph, file_path, ubi, header, out_dir, batch_name):
        morph_id = morph.morph_id
        with open("%s/%s.fasta" % (out_dir, batch_name), "a") as out_file:
            with open(file_path) as handle:
                records = PdbIO.PdbAtomIterator(handle)
                records_list = list(records)
                if len(records_list) > 1:
                    raise ValueError("Only one sequence is expected")
                records_list[0].id = morph_id
                records_list[0].description = ''
                SeqIO.write(records_list, out_file, "fasta")


    for batch_count, morphs_chunk in enumerate(morphs_chunks):
        morphs_repository.perform_on_some_morphs_in_directory(lambda x: x in morphs_chunk,
                                                              pdb_to_fasta, out_dir="/Users/mataneilat/Documents/BioInfo/raptor_input",
                                                              batch_name="batch_%s" % batch_count)




if __name__ == '__main__':
    main()