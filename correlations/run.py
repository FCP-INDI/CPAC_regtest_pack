#!/usr/bin/env python

from typing import Optional, NamedTuple, Tuple, Union

import os
import glob
import numpy as np
import scipy
from scipy import stats
import pandas as pd

from multiprocessing import Pool
import itertools
from compare_pipelines import compare_pipelines
from utils import read_yml_file

def main():

    import os
    import argparse

    from multiprocessing import Pool
    import itertools

    parser = argparse.ArgumentParser()
    parser.add_argument("input_yaml", type=str, 
                        help="file path of the script's input YAML")
    args = parser.parse_args()

    # get the input info
    input_dct = read_yml_file(args.input_yaml)

    # check for already completed stuff (pickles)
    output_dir = os.path.join(os.getcwd(), 
                              "correlations_{0}".format(input_dct['settings']['run_name']))
    pickle_dir = os.path.join(output_dir, "pickles")

    if not os.path.exists(pickle_dir):
        try:
            os.makedirs(pickle_dir)
        except:
            err = "\n\n[!] Could not create the output directory for the " \
                  "correlations. Do you have write permissions?\nAttempted " \
                  "output directory: {0}\n\n".format(output_dir)
            raise Exception(err)

    input_dct['settings'].update({'output_dir': output_dir})
    input_dct['settings'].update({'pickle_dir': pickle_dir})

    compare_pipelines(input_dct, dir_type='output_dir')
    #compare_pipelines(input_dct, dir_type='work_dir')


if __name__ == "__main__":
    main()
