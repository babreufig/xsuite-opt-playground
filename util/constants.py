import os

#####################################################
############### FILE PATHS ##########################
#####################################################

# /xsuite-opt-playground/util
UTIL_DIR = os.path.dirname(os.path.realpath(__file__))

# /xsuite-opt-playground
BASE_DIR = os.path.dirname(UTIL_DIR)

# /xsuite-opt-playground/lattice_data
LATTICE_DATA_PATH = os.path.join(BASE_DIR, 'lattice_data')

# Files from /xsuite-opt-playground/lattice_data
HLLHC15_THICK_PATH = os.path.join(LATTICE_DATA_PATH, 'hllhc15_collider_thick.json')
LHC_THICK_KNOBS_PATH = os.path.join(LATTICE_DATA_PATH, 'lhc_thick_with_knobs.json')
LHC_PATH = os.path.join(LATTICE_DATA_PATH, 'lhc.json')
OPT_150_1500_PATH = os.path.join(LATTICE_DATA_PATH, 'opt_round_150_1500.madx')
