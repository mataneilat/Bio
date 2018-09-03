
"""
    Module containing the MorphsRepository class
"""
from pathlib import Path
from prody import parsePDB
import os

class MorphsRepository:
    """
    The morphs repository allows convenient uniform access to the atlas's morphs
    """

    def __init__(self, atlas_morphs, morphs_pdb_directory):
        """
        Initializes the repository with the parsed AtlasMorph objects and path to the directory containing the protein
        structures.

        :param atlas_morphs:    Dictionary from morph_id to the parsed AtlasMorph object
        :param morphs_pdb_directory:    The path to the directory containing the protein structures
        """
        self.atlas_morphs = atlas_morphs
        self.morphs_pdb_directory = morphs_pdb_directory

    def perform_on_morph_in_atlas(self, morph_id, func, **func_kwargs):
        """
        Performs the given function of the specified morph.

        :param morph_id:    The ID of the morph on which to perform function
        :param func:    The function to perform. Receives the following on execution:
                        AtlasMorph object, protein structure file path, parsed AtomGroup object, parsed header,
                        func_kwargs.
        :param func_kwargs:     Additional arguments passed to func on execution.
        """
        morph = self.atlas_morphs.get(morph_id)
        if morph is None:
            return
        ff0_path = '%s/%s/ff0.pdb' % (self.morphs_pdb_directory, morph_id)
        if Path(ff0_path).is_file():
            ubi, header = parsePDB(ff0_path, subset='calpha', header=True)
            func(morph, ff0_path, ubi, header, **func_kwargs)

    def perform_on_some_morphs_in_directory(self, morph_filter, func, **kwargs):
        """
        Performs the given function on morphs the agree with the given filter

        :param morph_filter:    A function that receives a morph's filename and returns a boolean value indicating
                                whether to execute func on this morph or not.
        :param func:    The function to perform on each filtered morph. Receives the following on execution:
                        AtlasMorph object, protein structure file path, parsed AtomGroup object, parsed header,
                        func_kwargs.
        :param kwargs:  Additional arguments passed to func on execution.
        """
        directory = os.fsencode(self.morphs_pdb_directory)

        for file in os.listdir(directory):
            morph_filename = os.fsdecode(file)
            if morph_filter(morph_filename):
                self.perform_on_morph_in_atlas(morph_filename, func, **kwargs)
