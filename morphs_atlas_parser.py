"""
    Module providing methods responsible for parsing the atlas text file
"""

from atlas_objects import *

def parse_atlas_residue(line):
    """
    Parser a single residue's information from a line in the atlas text file
    :param line:    The line containing the residue's information
    :return:    The AtlasResidueInfo object with the parsed information.
    """
    tokens = line.split()
    morph_id = tokens[0]
    residue_idx = int(tokens[1])
    residue_type = tokens[2]
    is_hinge = (tokens[3].strip() == '1')

    return AtlasResidueInfo(morph_id, residue_idx, residue_type, is_hinge)


def parse_morphs_atlas_from_text(txt_file):
    """
    Parses the atlas information from the given text file

    :param txt_file:    Text file containing the atlas information
    :return:    Dictionary mapping morph_id to morph object.
    """
    morphs = []
    current_morph = None
    with open(txt_file) as f:
        lines = f.readlines()
        for line in lines[1:]:
            atlas_residue = parse_atlas_residue(line)
            if current_morph is None:
                current_morph = AtlasMorph(atlas_residue.morph_id)
            if current_morph.morph_id != atlas_residue.morph_id:
                morphs.append(current_morph.copy())
                current_morph = AtlasMorph(atlas_residue.morph_id)
            current_morph.add_residue(atlas_residue)

    return {morph.morph_id:morph for morph in morphs}
