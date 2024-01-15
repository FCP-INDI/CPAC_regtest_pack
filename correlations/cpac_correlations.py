#!/usr/bin/env python

from typing import Optional, NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
from multiprocessing import Pool
from s3_utils import download_from_s3
Axis = Union[int, Tuple[int, ...]]

class CorrValue(NamedTuple):
    concor: np.ndarray
    pearson: np.ndarray

def batch_correlate(
    x: np.ndarray, y: np.ndarray, axis: Optional[Axis] = None
) -> CorrValue:
    """
    Compute a batch of concordance and Pearson correlation coefficients between
    x and y along an axis (or axes).

    References:
        https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    """
    # Summary stats for x
    try:
        if x.ndim == 3:
            x= x[..., np.newaxis]
        x_mean = np.mean(x, axis=axis, keepdims=True)
        x_var = np.var(x, axis=axis, keepdims=True)
        x_std = np.sqrt(x_var)
        # NOTE: Not trying to fix NaNs
        x_norm = (x - x_mean) / x_std

    except Exception as e:
        print(f"Batch Correlate Exception for X: {e}\n{x.shape}")       
    try:
        # Summary stats for y
        if y.ndim == 3:
            y= y[..., np.newaxis]
        y_mean = np.mean(y, axis=axis, keepdims=True)
        y_var = np.var(y, axis=axis, keepdims=True)
        y_std = np.sqrt(y_var)
        y_norm = (y - y_mean) / y_std

    except Exception as e:
        print(f"Batch Correlate Exception for Y: {e}\n{y.shape}")
    
    try:
        # Correlation coefficients
        pearson = np.mean(x_norm * y_norm, axis=axis, keepdims=True)
        concor = 2 * pearson * x_std * y_std / (x_var + y_var + (x_mean - y_mean) ** 2)

    except Exception as e:
        print(f"Batch Correlate Exception for Correlation Coefficients: {e}\n{pearson.shape}")

    # Squeeze reduced singleton dimensions
    if axis is not None:
        concor = np.squeeze(concor, axis=axis)
        pearson = np.squeeze(pearson, axis=axis)
    
    pearson = np.nanmean(pearson)
    concor = np.nanmean(concor)
    return (concor, pearson)

def correlate_text_based(txt1, txt2):
    # TODO: why do we drop columns containing na?
    oned_one = pd.read_csv(txt1, delimiter=None, comment="#").dropna(axis=1).values
    oned_two = pd.read_csv(txt2, delimiter=None, comment="#").dropna(axis=1).values

    concor, pearson = batch_correlate(oned_one, oned_two, axis=0)
    concor = np.nanmean(concor)
    pearson = np.nanmean(pearson)
    return concor, pearson

def calculate_correlation(args_tuple):

    import os
    import subprocess
    import nibabel as nb
    import numpy as np
    import scipy.stats.mstats
    import scipy.stats
    import math
   
    category = args_tuple[0]
    old_path = args_tuple[1]
    new_path = args_tuple[2]
    local_dir = args_tuple[3]
    s3_creds = args_tuple[4]
    verbose = args_tuple[5]

    if verbose:
        print("Calculating correlation between {0} and {1}".format(old_path, new_path))

    corr_tuple = None

    if s3_creds:
        try:
            # full filepath with filename
            old_local_file = os.path.join(local_dir, "s3_input_files", \
                old_path.replace("s3://",""))
            # directory without filename
            old_local_path = old_local_file.replace(old_path.split("/")[-1],"")

            new_local_file = os.path.join(local_dir, "s3_input_files", \
                new_path.replace("s3://",""))
            new_local_path = new_local_file.replace(new_path.split("/")[-1],"")

            if not os.path.exists(old_local_path):
                os.makedirs(old_local_path)
            if not os.path.exists(new_local_path):
                os.makedirs(new_local_path)

        except Exception as e:
            err = "\n\nLocals: {0}\n\n[!] Could not create the local S3 " \
                  "download directory.\n\nError details: {1}\n\n".format((locals(), e))
            raise Exception(e)

        try:
            if not os.path.exists(old_local_file):
                old_path = download_from_s3(old_path, old_local_path, s3_creds)
            else:
                old_path = old_local_file
        except Exception as e:
            err = "\n\nLocals: {0}\n\n[!] Could not download the files from " \
                  "the S3 bucket. \nS3 filepath: {1}\nLocal destination: {2}" \
                  "\nS3 creds: {3}\n\nError details: {4}\n\n".format(locals(), 
                                                                     old_path, 
                                                                     old_local_path, 
                                                                     s3_creds, e)
            raise Exception(e)

        try:
            if not os.path.exists(new_local_file):
                new_path = download_from_s3(new_path, new_local_path, s3_creds)
            else:
                new_path = new_local_file
        except Exception as e:
            err = "\n\nLocals: {0}\n\n[!] Could not download the files from " \
                 "the S3 bucket. \nS3 filepath: {1}\nLocal destination: {2}" \
                  "\nS3 creds: {3}\n\nError details: {4}\n\n".format(locals(), 
                                                                     new_path, 
                                                                     new_local_path, 
                                                                     s3_creds, e)
            raise Exception(e)

    ## nibabel to pull the data from the re-assembled file paths
    if os.path.exists(old_path) and os.path.exists(new_path):

        if ('.csv' in old_path and '.csv' in new_path) or \
                ('spatial_map_timeseries.txt' in old_path and 'spatial_map_timeseries.txt' in new_path) or \
                    ('.1D' in old_path and '.1D' in new_path) or \
                        ('.tsv' in old_path and '.tsv' in new_path):
            try:
                concor, pearson = correlate_text_based(old_path, new_path)

                if concor > 0.980:
                    corr_tuple = (category, [concor], [pearson])
                else:
                    corr_tuple = (category, [concor], [pearson], (old_path, new_path))
                if verbose:
                    print("Success - {0}".format(str(concor)))

            except Exception as e:
                corr_tuple = ("file reading problem: {0}".format(e), 
                              old_path, new_path)
                if verbose:
                    print(str(corr_tuple))

            return corr_tuple

        else:
            try:
                old_file_img = nb.load(old_path)
                old_file_hdr = old_file_img.header
                new_file_img = nb.load(new_path)
                new_file_hdr = new_file_img.header

                data_1 = nb.load(old_path).get_fdata()
                data_2 = nb.load(new_path).get_fdata()

                old_file_dims = data_1.shape
            
            except Exception as e:
                corr_tuple = ("file reading problem: {0}".format(e), 
                              old_path, new_path)
                if verbose:
                    print(str(corr_tuple))
                return corr_tuple

        ## set up and run the Pearson correlation and concordance correlation
        if data_1.flatten().shape == data_2.flatten().shape:
            try:
                if len(old_file_dims) > 3:
                    axis = tuple(range(3, len(old_file_dims)))
                    concor, pearson = batch_correlate(data_1, data_2, axis=axis)
                else:
                    concor, pearson = batch_correlate(data_1, data_2)
            except Exception as e:
                corr_tuple = ("correlating problem: {0}".format(e), 
                              old_path, new_path)
                if verbose:
                    print(str(corr_tuple))
                return corr_tuple
            if concor > 0.980:
                corr_tuple = (category, [concor], [pearson])
            else:
                corr_tuple = (category, [concor], [pearson], (old_path, new_path))
            if verbose:
                print("Success - {0}".format(str(concor)))
        else:
            corr_tuple = ("different shape", old_path, new_path)
            if verbose:
                print(str(corr_tuple))

    else:
        if not os.path.exists(old_path):
            corr_tuple = ("file doesn't exist", [old_path], None)
            if verbose:
                print(str(corr_tuple))
        if not os.path.exists(new_path):
            if not corr_tuple:
                corr_tuple = ("file doesn't exist", [new_path], None)
                if verbose:
                    print(str(corr_tuple))
            else:
                corr_tuple = ("file doesn't exist", old_path, new_path)
                if verbose:
                    print(str(corr_tuple))

    return corr_tuple

def run_correlations(matched_dct, input_dct, source='output_dir', quick=False, verbose=False):

    all_corr_dct = {
        'pearson': {},
        'concordance': {},
        'sub_optimal': {}
    }

    args_list = []

    quick_list = [
        'anatomical_brain',
        'anatomical_csf_mask',
        'anatomical_gm_mask',
        'anatomical_wm_mask',
        'anatomical_to_standard',
        'functional_preprocessed',
        'functional_brain_mask',
        'mean_functional_in_anat',
        'functional_nuisance_residuals',
        'functional_nuisance_regressors',
        'functional_to_standard',
        'roi_timeseries'
    ]

    matched_path_dct = matched_dct['matched']
    output_dir = input_dct['settings']['correlations_dir']
    s3_creds = input_dct['settings']['s3_creds']

    for category in matched_path_dct.keys():

        if quick:
            if category not in quick_list:
                continue

        for file_id in matched_path_dct[category].keys():
            
            old_path = matched_path_dct[category][file_id][0]
            new_path = matched_path_dct[category][file_id][1]

            if source == 'work_dir':
                args_list.append((file_id, old_path, new_path, output_dir, s3_creds, verbose))
            else:
                args_list.append((category, old_path, new_path, output_dir, s3_creds, verbose))

    print("\nNumber of correlations to calculate: {0}\n".format(len(args_list)))
    total = len(args_list)
    print("Running correlations...")
    p = Pool(input_dct['settings']['n_cpus'])
    corr_tuple_list = p.map(calculate_correlation, args_list)
    p.close()
    p.join()

    print("\nCorrelations of the {0} are done.\n".format(source))

    for corr_tuple in corr_tuple_list:
        if not corr_tuple:
            continue
        if corr_tuple[0] not in all_corr_dct['concordance'].keys():
            all_corr_dct['concordance'][corr_tuple[0]] = []
        if corr_tuple[0] not in all_corr_dct['pearson'].keys():
            all_corr_dct['pearson'][corr_tuple[0]] = []
        all_corr_dct['concordance'][corr_tuple[0]] += corr_tuple[1]
        all_corr_dct['pearson'][corr_tuple[0]] += corr_tuple[2]

        if len(corr_tuple) > 3:
            if corr_tuple[0] not in all_corr_dct['sub_optimal'].keys():
                all_corr_dct['sub_optimal'][corr_tuple[0]] = []
            try:
                all_corr_dct['sub_optimal'][corr_tuple[0]].append("{0}:\n{1}\n{2}"
                                                                  "\n\n".format(corr_tuple[1][0], 
                                                                                corr_tuple[3][0],
                                                                                corr_tuple[3][1]))
            except TypeError as te:
                print(f'Type Error Ocurred. \n{te}')
                pass

    return all_corr_dct

def post180_organize_correlations(concor_dct, corr_type="concordance", quick=False):

    corr_map_dct = {"correlations": {}}
    for key in concor_dct:
        if "problem" in key:
            continue
        # shouldn't need this - FIX
        rawkey = key.replace('acq-', '').replace('run-', '')
        datatype = rawkey.split("_")[-1]

        if datatype not in corr_map_dct["correlations"]:
            corr_map_dct["correlations"][datatype] = {}
        corr_map_dct["correlations"][datatype][rawkey] = concor_dct[key]

    return corr_map_dct

def organize_correlations(concor_dict, corr_type="concordance", quick=False):
    # break up all of the correlations into groups - each group of derivatives
    # will go into its own boxplot

    regCorrMap = {}
    native_outputs = {}
    template_outputs = {}
    timeseries = {}
    functionals = {}

    core = {}

    corr_map_dict = {}
    corr_map_dict["correlations"] = {}

    derivs = [
        'alff', 
        'dr_tempreg', 
        'reho', 
        'sca_roi', 
        'timeseries', 
        'ndmg']
    anats = [
        'anatomical', 
        'seg'
    ]
    time_series = [
        'functional_freq',
        'nuisance_residuals',
        'functional_preprocessed',
        'functional_to_standard',
        'ica_aroma_',
        'motion_correct',
        'slice_time',
    ]
    funcs = [
        'functional',
        'displacement']

    for key in concor_dict:

        if quick:
            core[key] = concor_dict[key]
            continue

        if 'xfm' in key or 'mixel' in key:
            continue

        if 'centrality' in key or 'vmhc' in key or 'sca_tempreg' in key:
            template_outputs[key] = concor_dict[key]
            continue

        for word in anats:
            if word in key:
                regCorrMap[key] = concor_dict[key]
                continue

        for word in derivs:
            if word in key and 'standard' not in key:
                native_outputs[key] = concor_dict[key]
                continue
            elif word in key:
                template_outputs[key] = concor_dict[key]
                continue

        for word in time_series:
            if word in key and 'mean' not in key and 'mask' not in key:
                timeseries[key] = concor_dict[key]
                continue

        for word in funcs:
            if word in key:
                functionals[key] = concor_dict[key]

    if quick:
        group = "{0}_core_outputs".format(corr_type)
        if len(core.values()) > 0:
            corr_map_dict["correlations"][group] = core
        else:
            print("No values in {0}".format(group))
        return corr_map_dict

    group = "{0}_registration_and_segmentation".format(corr_type)
    if len(regCorrMap.values()) > 0:
        corr_map_dict["correlations"][group] = regCorrMap
    else:
        print("No values in {0}".format(group))
 
    group = "{0}_native_space_outputs".format(corr_type)
    if len(native_outputs.values()) > 0:
        corr_map_dict["correlations"][group] = native_outputs
    else:
        print("No values in {0}".format(group))

    group = "{0}_template_space_outputs".format(corr_type)
    if len(template_outputs.values()) > 0:
        corr_map_dict["correlations"][group] = template_outputs
    else:
        print("No values in {0}".format(group))

    group = "{0}_timeseries_outputs".format(corr_type)
    if len(timeseries.values()) > 0:
        corr_map_dict["correlations"][group] = timeseries
    else:
        print("No values in {0}".format(group))

    group = "{0}_functional_outputs".format(corr_type)
    if len(functionals.values()) > 0:
        corr_map_dict["correlations"][group] = functionals
    else:
        print("No values in {0}".format(group))

    return corr_map_dict
