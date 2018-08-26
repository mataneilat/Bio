
from pathlib import Path
from prody import parsePDB
import os

class MorphsRepository:

    def __init__(self, atlas_morphs, morphs_pdb_directory):
        self.atlas_morphs = atlas_morphs
        self.morphs_pdb_directory = morphs_pdb_directory

    def perform_on_morph_in_atlas(self, morph_id, func, **func_kwargs):
        morph = self.atlas_morphs.get(morph_id)
        if morph is None:
            return
        ff0_path = '%s/%s/ff0.pdb' % (self.morphs_pdb_directory, morph_id)
        if Path(ff0_path).is_file():
            ubi, header = parsePDB(ff0_path, subset='calpha', header=True)
            func(morph, ubi, header, **func_kwargs)

    def perform_on_all_morphs_in_directory(self, func, **kwargs):
        self.perform_on_some_morphs_in_directory(lambda x: True, func, **kwargs)

    def perform_on_some_morphs_in_directory(self, morph_filter, func, **kwargs):
        directory = os.fsencode(self.morphs_pdb_directory)

        for file in os.listdir(directory):
            morph_filename = os.fsdecode(file)
            if morph_filter(morph_filename):
                self.perform_on_morph_in_atlas(morph_filename, func, **kwargs)
