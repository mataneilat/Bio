"""
    Module containing the atlas access objects
"""
class AtlasResidueInfo:
    """
    Class representing a single residue from the atlas
    """
    def __init__(self, morph_id, residue_idx, residue_type, is_hinge):
        self.morph_id = morph_id
        self.residue_idx = residue_idx
        self.residue_type = residue_type
        self.is_hinge = is_hinge

    def __repr__(self):
        return '<id: %s, idx: %s, type: %s, is_hinges %s>' % (self.morph_id, self.residue_idx, self.residue_type, self.is_hinge)


class AtlasMorph:
    """
    Class representing a entire morph information from the atlas.
    """
    def __init__(self, morph_id, residues=None):
        self.morph_id = morph_id
        if residues is None:
            residues = []
        self.residues = residues

    def add_residue(self, residue):
        self.residues.append(residue)

    def get_hinges(self):
        return [res.residue_idx for res in filter(lambda res: res.is_hinge, self.residues)]

    def copy(self):
        return AtlasMorph(self.morph_id, residues=self.residues.copy())
