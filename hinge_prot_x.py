
import argparse
import sys
from prody import parsePDB, LOGGER
import logging

from contact_map_repository import ContactMapRepository
from morphs_repository import MorphsRepository
from morphs_atlas_parser import parse_morphs_atlas_from_text
from prediction.tensorflow_prediction import *
from prediction.deterministic_prediction import *
from result_measures import *

LOCAL_SENSITIVITY = 7


def initialize_ndl_predictor(morphs_repository, morphs_ids, contact_map_repository):
    tensorflow_predictor = NeighborhoodDeepLearningPredictor(LOCAL_SENSITIVITY)
    tensorflow_predictor.train_model(morphs_repository, morphs_ids, contact_map_repository)

    return 'Neighborhood Deep Learning', tensorflow_predictor


def setup_ndl_predictor(args):
    contact_map_repository = None
    if args.atlas_contact_map_dir is not None:
        contact_map_repository = ContactMapRepository(args.atlas_contact_map_dir)

    morphs_repository = MorphsRepository(parse_morphs_atlas_from_text(args.hinge_atlas_file),
                                         args.atlas_pdb_directory)

    morphs_ids = list(morphs_repository.atlas_morphs.keys())

    return initialize_ndl_predictor(morphs_repository, morphs_ids, contact_map_repository)


def setup_cca_predictor(args):
    return 'Cross Correlation Average', CrossCorrelationAvgPredictor(LOCAL_SENSITIVITY)


def setup_nca_predictor(args):
    return 'Near Correlations Averages', NearCorrelationAvgsPredictor(LOCAL_SENSITIVITY)


def setup_cvd_predictor(args):
    return 'Correlation Vectors Distance', CorrelationVectorsDistancePredictor(LOCAL_SENSITIVITY)


def predict(args):

    predictors = []

    possible_predictors = ('cca', 'nca', 'cvd', 'ndl')

    for possible_predictor in possible_predictors:
        if args.method == possible_predictor or args.method == 'all':
            setup_method = globals()['setup_%s_predictor' % possible_predictor]
            predictors.append(setup_method(args))

    ubi, header = parsePDB(args.pdb, subset='calpha', header=True)

    k_inv = None
    if args.contact_map is None:
        k_inv = calc_gnm_k_inv(ubi, header)
    else:
        k_inv = calc_gnm_k_inv(ubi, header, contact_map=args.contact_map)

    for desc, predictor in predictors:

        hinges = predictor.predict_hinges(k_inv)
        print('Predictor:', desc, ';', 'Hinges:', hinges)

    if args.method == 'prody' or args.method == 'all':
        print('ProDy Predictor Hinges:', get_hinges_default(ubi, header))



def show_results(args):

    contact_map_repository = None
    if args.atlas_contact_map_dir is not None:
        contact_map_repository = ContactMapRepository(args.atlas_contact_map_dir)

    morphs_repository = MorphsRepository(parse_morphs_atlas_from_text(args.hinge_atlas_file), args.atlas_pdb_directory)

    morphs_ids = list(morphs_repository.atlas_morphs.keys())
    test_morph_ids = morphs_ids

    predictors = [setup_cca_predictor(args), setup_nca_predictor(args), setup_cvd_predictor(args)]

    if not args.without_dl:

        # In case we use machine learning, a portion of the morphs are are used to train the machine, and the others
        # are being used as test morphs
        train_morph_ids = morphs_ids[:150]
        test_morph_ids = morphs_ids[150:]

        predictors.append(initialize_ndl_predictor(morphs_repository, train_morph_ids, contact_map_repository))

    total_scores = [0] * len(predictors)

    total_default_score = 0

    def collect_prediction_results(morph, file_path, ubi, header):
        nonlocal predictors, total_scores, total_default_score

        total_residue_count = len(morph.residues)

        k_inv = None
        if contact_map_repository is None:
            k_inv = calc_gnm_k_inv(ubi, header)
        else:
            contact_map = contact_map_repository.get_contact_map_rr(morph.morph_id, len(ubi))
            if contact_map is None:
                return
            k_inv = calc_gnm_k_inv(ubi, header, contact_map=contact_map)

        logging.debug('-----------------')
        logging.debug('Annotated Hinges: %s' % morph.get_hinges())

        for i, predictor in enumerate(predictors):
            desc = predictor[0]
            predictor_class = predictor[1]

            predicted_hinges = predictor_class.predict_hinges(k_inv)
            prediction_score = calculate_mcc(predicted_hinges, morph.get_hinges(), total_residue_count)

            logging.debug('Predictor: %s ; Hinges: %s ; MCC: %s' % (desc, predicted_hinges, prediction_score))
            total_scores[i] += prediction_score

        default_hinges = get_hinges_default(ubi, header)
        default_score = calculate_mcc(default_hinges, morph.get_hinges(), total_residue_count)

        logging.debug('ProDy predictor hinges: %s ; MCC: %s' % (default_hinges, default_score))
        logging.debug('-----------------')
        total_default_score += default_score

    morphs_repository.perform_on_some_morphs_in_directory(lambda morph_id : morph_id in test_morph_ids,
            collect_prediction_results)

    for i, predictor in enumerate(predictors):
        desc = predictor[0]
        print(desc, "Average MCC", total_scores[i] / len(test_morph_ids))

    print("Prody Average MCC:", total_default_score / len(test_morph_ids))


def validate_predict(parser, args):
    if args.method == 'ndl' or args.method == 'all':
        if args.hinge_atlas_file is None or args.atlas_pdb_directory is None:
            parser.print_help()
            sys.stderr.write("Error: When choosing the machine learning method, one must specify the atlas file and directory")
            sys.exit(2)


def validate_arguments(parser, args):
    if args.command is None:
        parser.print_help()
        sys.stderr.write("Error: No method was chosen")
        sys.exit(2)
    if args.command == 'predict':
        validate_predict(parser, args)


def add_show_results_parser(subparsers):

    show_results_parser = subparsers.add_parser('show_results', help='The command for showing hinge prediction results')

    show_results_parser.add_argument('-without_dl', action='store_true')

    show_results_parser.add_argument('--hinge_atlas_file', dest='hinge_atlas_file', required=True,
                                   help='The atlas of hinge annotations that should be given in case of machine learning prediction')

    show_results_parser.add_argument('--atlas_pdb_directory', dest='atlas_pdb_directory', required=True,
                                   help='The directory containing the protein structure files for the atlas annotations')

    show_results_parser.add_argument('--atlas_contact_map_dir', dest='atlas_contact_map_dir',
                        help='Directory of contact maps for atlas proteins')

def add_prediction_parser(subparsers):

    predict_parser = subparsers.add_parser('predict', help='The command for normal hinge prediction')

    predict_parser.add_argument('pdb', help='The protein structure for which hinge prediction is required')
    predict_parser.add_argument('method', choices=['all', 'cca', 'nca', 'cvd', 'ndl', 'prody'],
                                   help='The method to use for prediction')
    predict_parser.add_argument('--contact_map', dest='contact_map', help='Directory of contact maps for atlas proteins')
    predict_parser.add_argument('--hinge_atlas_file', dest='hinge_atlas_file',
                                   help='The atlas\' hinge annotations that should be given in case of machine learning prediction')
    predict_parser.add_argument('--atlas_pdb_directory', dest='atlas_pdb_directory',
                                   help='The directory containing the protein structure files for the atlas annotations')
    predict_parser.add_argument('--atlas_contact_map_dir', dest='atlas_contact_map_dir',
                        help='Directory of contact maps')


def setup_arg_parser():
    parser = argparse.ArgumentParser(prog='Hinge prediction software')

    parser.add_argument('-log_level', default='INFO', choices=['INFO', 'DEBUG', 'CRITICAL'], help='The logging level')
    subparsers = parser.add_subparsers(title='subcommands', dest='command', help='The different execution modes')

    add_prediction_parser(subparsers)

    add_show_results_parser(subparsers)

    return parser


def execute(args):
    if args.command == 'show_results':
        show_results(args)
    if args.command == 'predict':
        predict(args)


if __name__ == '__main__':

    parser = setup_arg_parser()

    # Disable debug logging from prody
    LOGGER._logger.level = logging.INFO

    args = parser.parse_args()

    validate_arguments(parser, args)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    execute(args)

  #  Benchmark().plot_benchmark()


