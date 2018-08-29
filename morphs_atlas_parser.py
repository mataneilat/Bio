
class AtlasResidueInfo:

    def __init__(self, morph_id, residue_idx, residue_type, is_hinge):
        self.morph_id = morph_id
        self.residue_idx = residue_idx
        self.residue_type = residue_type
        self.is_hinge = is_hinge

    def __repr__(self):
        return '<id: %s, idx: %s, type: %s, is_hinges %s>' % (self.morph_id, self.residue_idx, self.residue_type, self.is_hinge)


class AtlasMorph:

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


def parse_atlas_residue(line):
    tokens = line.split()
    morph_id = tokens[0]
    residue_idx = int(tokens[1])
    residue_type = tokens[2]
    is_hinge = (tokens[3].strip() == '1')

    return AtlasResidueInfo(morph_id, residue_idx, residue_type, is_hinge)


def parse_morphs_atlas_from_text(txt_file):
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


def main():
    morphs = parse_morphs_atlas_from_text('./hingeatlas.txt')
    for morph in morphs:
        print(morph.morph_id, morph.get_hinges())

if __name__ == '__main__':
    main()