import mvn
from numpy import asarray
from matplotlib.pyplot import get_cmap

# Default colors
CMAP = get_cmap("tab10")
# data directory
DATA_DIR = "../data/"
# Subject table
SUBJECT_TABLE = DATA_DIR + "subject_table.csv"

# Constants
GRAVITY = (
    9.80665  # m/s/s  # Acceleration due to gravity acts in the negative z direction...
)
MAX_EASTING = 900000  # m approximate maximum easting coordinate in m.

POLE_DENSITY = 0.192 / 1.5875  # kg/m of length
SKI_DENSITY = 0.706 / (2.01 * 0.04445)  # kg/m^2 of area (length x width)

SEGMENT_NAMES = [
    "l_foot",
    "l_lower_leg",
    "l_upper_leg",
    "r_foot",
    "r_lower_leg",
    "r_upper_leg",
    "l_hand",
    "l_forearm",
    "l_upper_arm",
    "r_hand",
    "r_forearm",
    "r_upper_arm",
    "lower_trunk",
    "mid_trunk",
    "upper_trunk",
    "head",
]

# segment proportional mass of total body mass
# See table 4.1 in DeLava (1995)
SEGMENT_MASS_MALE = {
    "head": 6.94 / 100,
    "upper_trunk": 15.96 / 100,
    "mid_trunk": 16.33 / 100,
    "lower_trunk": 11.17 / 100,
    "upper_arm": 2.71 / 100,
    "forearm": 1.62 / 100,
    "hand": 0.61 / 100,
    "upper_leg": 14.16 / 100,
    "lower_leg": 4.33 / 100,
    "foot": 1.37 / 100,
}
SEGMENT_MASS_FEMALE = {
    "head": 6.68 / 100,
    "upper_trunk": 15.45 / 100,
    "mid_trunk": 14.65 / 100,
    "lower_trunk": 12.47 / 100,
    "upper_arm": 2.55 / 100,
    "forearm": 1.38 / 100,
    "hand": 0.56 / 100,
    "upper_leg": 14.78 / 100,
    "lower_leg": 4.81 / 100,
    "foot": 1.29 / 100,
}
# Segment length proportions from full height
SEGMENT_LENGTH_MALE = {
    "head": 0.2033 / 1.741,
    "upper_trunk": 0.1707 / 1.741,
    "mid_trunk": 0.2155 / 1.741,
    "lower_trunk": 0.1457 / 1.741,
    "upper_arm": 0.2817 / 1.741,
    "forearm": 0.2689 / 1.741,
    "hand": 0.0862 / 1.741,
    "upper_leg": 0.4222 / 1.741,
    "lower_leg": 0.434 / 1.741,
    "foot": 0.2581 / 1.741,
}
SEGMENT_LENGTH_FEMALE = {
    "head": 0.2002 / 1.735,
    "upper_trunk": 0.1425 / 1.735,
    "mid_trunk": 0.2053 / 1.735,
    "lower_trunk": 0.1815 / 1.735,
    "upper_arm": 0.2751 / 1.735,
    "forearm": 0.2643 / 1.735,
    "hand": 0.0780 / 1.735,
    "upper_leg": 0.3685 / 1.735,
    "lower_leg": 0.4323 / 1.735,
    "foot": 0.2283 / 1.735,
}

# segment com positions as a function of segment length relative to proximal end
SEGMENT_COM_MALE = {
    "head": 59.76 / 100,
    "upper_trunk": 29.99 / 100,
    "mid_trunk": 45.02 / 100,
    "lower_trunk": 61.15 / 100,
    "upper_arm": 57.72 / 100,
    "forearm": 45.74 / 100,
    "hand": 79.00 / 100,
    "upper_leg": 40.95 / 100,
    "lower_leg": 44.59 / 100,
    "foot": 44.15 / 100,
}

SEGMENT_COM_FEMALE = {
    "head": 58.94 / 100,
    "upper_trunk": 20.77 / 100,
    "mid_trunk": 45.12 / 100,
    "lower_trunk": 49.20 / 100,
    "upper_arm": 57.54 / 100,
    "forearm": 45.59 / 100,
    "hand": 74.74 / 100,
    "upper_leg": 36.12 / 100,
    "lower_leg": 44.16 / 100,
    "foot": 40.14 / 100,
}

# Segment radius of gyration as a function of segment length
# sagittal, transverse, longitudinal
SEGMENT_RADII_GYRATION_MALE = {
    "head": asarray([36.2, 37.6, 31.2]) / 100,
    "upper_trunk": asarray([71.6, 45.4, 65.9]) / 100,
    "mid_trunk": asarray([48.2, 38.3, 46.8]) / 100,
    "lower_trunk": asarray([61.5, 55.1, 58.7]) / 100,
    "upper_arm": asarray([28.5, 26.9, 15.8]) / 100,
    "forearm": asarray([27.6, 26.5, 12.1]) / 100,
    "hand": asarray([62.8, 51.3, 40.1]) / 100,
    "upper_leg": asarray([32.9, 32.9, 14.9]) / 100,
    "lower_leg": asarray([25.5, 24.9, 10.3]) / 100,
    "foot": asarray([25.7, 24.5, 12.4]) / 100,
}

SEGMENT_RADII_GYRATION_FEMALE = {
    "head": asarray([33.0, 35.9, 31.8]) / 100,
    "upper_trunk": asarray([74.6, 50.2, 71.8]) / 100,
    "mid_trunk": asarray([43.3, 35.4, 41.5]) / 100,
    "lower_trunk": asarray([43.3, 40.2, 44.4]) / 100,
    "upper_arm": asarray([27.8, 26.0, 14.8]) / 100,
    "forearm": asarray([26.1, 25.7, 9.4]) / 100,
    "hand": asarray([53.1, 45.4, 33.5]) / 100,
    "upper_leg": asarray([36.9, 36.4, 16.2]) / 100,
    "lower_leg": asarray([27.1, 26.7, 9.3]) / 100,
    "foot": asarray([29.9, 27.9, 13.9]) / 100,
}

# Map segment to xsens
SEGMENT_TO_XSENS = {
    "head": [mvn.SEGMENT_HEAD, mvn.SEGMENT_NECK],
    "upper_trunk": [mvn.SEGMENT_T8],
    "mid_trunk": [mvn.SEGMENT_T12, mvn.SEGMENT_L3],
    "lower_trunk": [mvn.SEGMENT_L5, mvn.SEGMENT_PELVIS],
    "l_upper_arm": [mvn.SEGMENT_LEFT_UPPER_ARM],
    "r_upper_arm": [mvn.SEGMENT_RIGHT_UPPER_ARM],
    "l_forearm": [mvn.SEGMENT_LEFT_FOREARM],
    "r_forearm": [mvn.SEGMENT_RIGHT_FOREARM],
    "l_hand": [mvn.SEGMENT_LEFT_HAND],
    "r_hand": [mvn.SEGMENT_RIGHT_HAND],
    "l_upper_leg": [mvn.SEGMENT_LEFT_UPPER_LEG],
    "r_upper_leg": [mvn.SEGMENT_RIGHT_UPPER_LEG],
    "l_lower_leg": [mvn.SEGMENT_LEFT_LOWER_LEG],
    "r_lower_leg": [mvn.SEGMENT_RIGHT_LOWER_LEG],
    "l_foot": [mvn.SEGMENT_LEFT_FOOT, mvn.SEGMENT_LEFT_TOE],
    "r_foot": [mvn.SEGMENT_RIGHT_FOOT, mvn.SEGMENT_RIGHT_TOE],
}

# Helper functions
def strip_utf8(s):
    if s[0] == "\ufeff":
        s = s[1 : len(s)]
    return s


def strip_endline(s):
    if s[-1] == "\n":
        s = s[0:-1]
    return s
