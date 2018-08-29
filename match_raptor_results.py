
import os
from shutil import copyfile
import zipfile
import shutil


def produce_raptor_results_with_proper_id(morph_ids, sequences, batch_raptor_output_dir_path,
                                          gcnn_output_dir, rr_output_dir):

    temp_work_dir = '/tmp/raptor_work'

    for filename in os.listdir(batch_raptor_output_dir_path):
        if not filename.endswith('.zip'):
            continue
        raptor_id = filename.split('.')[0]

        filepath = os.path.join(batch_raptor_output_dir_path, filename)

        # Extract zip to zip_dest
        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(temp_work_dir)
        zip_ref.close()

        zip_dest = os.path.join(temp_work_dir, filename[:-4])

        morph_id = None
        for fn in os.listdir(zip_dest):
            file_path = os.path.join(zip_dest, fn)
            if fn.endswith('.seq'):
                morph_id = figure_morph_id(morph_ids, sequences, file_path)

        copyfile(os.path.join(zip_dest, raptor_id + '.gcnn'),
                 os.path.join(gcnn_output_dir, morph_id + '.gcnn'))

        copyfile(os.path.join(zip_dest, raptor_id + '.rr'),
                 os.path.join(rr_output_dir, morph_id + '.rr'))

        shutil.rmtree(zip_dest)


def parse_batches(batches_dir_path):
    morph_ids = []
    sequences = []
    current_morph_id = None
    current_sequence = None
    for filename in os.listdir(batches_dir_path):
        filepath = os.path.join(batches_dir_path, filename)
        with open(filepath, "r") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for line in content:
                splitted_line = line.split('>')
                if len(splitted_line) > 2:
                    raise ValueError("Line is of invalid format")
                if len(splitted_line) == 2:
                    if len(splitted_line[0]) > 0:
                        raise ValueError("Line is of invalid format")
                    if current_morph_id is not None and current_sequence is not None:
                        morph_ids.append(current_morph_id)
                        sequences.append(current_sequence)

                    current_morph_id = splitted_line[1]
                    current_sequence = ''
                if len(splitted_line) == 1:
                    if current_sequence is None:
                        current_sequence = ''
                    current_sequence += splitted_line[0].strip()
        print("Finished with", filename)
    morph_ids.append(current_morph_id)
    sequences.append(current_sequence)
    return morph_ids, sequences


def figure_morph_id(morph_ids, sequences, seq_file_path):
    with open(seq_file_path, "r") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        if len(content) != 2:
            raise ValueError("Seq file of wrong format")
        sequence = content[1].strip()
        sequence_idx = sequences.index(sequence)
        return morph_ids[sequence_idx]

def copy_contact_maps(raptor_id_to_morph_id, batch_raptor_output_dir_path, contact_maps_dir_path):

    for filename in os.listdir(batch_raptor_output_dir_path):
        if not filename.endswith('.rr'):
            continue
        src_path = os.path.join(batch_raptor_output_dir_path, filename)

        filename_trimmed = filename[:-3]
        dst_filename = raptor_id_to_morph_id[filename_trimmed] + '.rr'
        dst_path = os.path.join(contact_maps_dir_path, dst_filename)
        copyfile(src_path, dst_path)


if __name__ == '__main__':

    current_batch = 11

    batches_dir_path = '/Users/mataneilat/Documents/BioInfo/raptor_input'
    morph_ids, sequences = parse_batches(batches_dir_path)

    batch_raptor_output_dir_path = '/Users/mataneilat/Downloads/batch_%s.all_in_one' % current_batch

    produce_raptor_results_with_proper_id(morph_ids, sequences, batch_raptor_output_dir_path,
                                          '/Users/mataneilat/Documents/BioInfo/raptor_output/contact_maps_new',
                                          '/Users/mataneilat/Documents/BioInfo/raptor_output/contact_maps')



