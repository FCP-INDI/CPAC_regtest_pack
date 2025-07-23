#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SBATCH -N 1
# SBATCH -p RM-shared
# SBATCH -t 01:00:00
# SBATCH --ntasks-per-node=4
"""Calculate correlation coefficients for C-PAC outputs."""
import argparse
from collections.abc import Generator
from fcntl import flock, LOCK_EX, LOCK_UN
from itertools import chain
import json
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
from typing import (
    Any,
    cast,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Protocol,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd
import yaml

import nibabel as nb

Axis = Union[int, Tuple[int, ...]]


class CorrValue(NamedTuple):
    """Correlation values"""

    concor: np.ndarray | float
    pearson: np.ndarray | float


DirType = Literal["output_dir", "work_dir", "log_dir"]
Replacement = tuple[str, str] | str


class InputDctSettings(TypedDict):
    """Dictionary for input settings."""

    n_cpus: int
    correlations_dir: str | Path
    run_name: str
    s3_creds: Optional[str]
    quick: bool
    verbose: bool
    output_dir: str | Path
    pickle_dir: str | Path


class InputDctPipelines(TypedDict):
    """Dictionary for input pipelines."""

    output_dir: str | Path
    work_dir: str | Path
    log_dir: str | Path
    pipe_config: str | Path
    replacements: Optional[list[Replacement]]


class InputDct(TypedDict):
    """Dictionary for correlation inputs."""

    settings: InputDctSettings
    pipelines: dict[str, InputDctPipelines]


OutFilesSubDct = dict[tuple[str, str, str], list[str | Path]]
OutFilesDct = dict[str, OutFilesSubDct]


class MapDict(TypedDict):
    """Dictionary for maps of correlations."""

    correlations: OutFilesDct
    pipeline_names: list[str]


class MatchedFilepaths(TypedDict):
    """Dictionary for matched / unmatched filepaths."""

    matched: OutFilesDct
    missing_old: list[OutFilesSubDct | tuple[str, str, str]]
    missing_new: list[OutFilesSubDct]


CorrelationMap = dict[str, dict[str, float]]
QuickSummary = dict[str, list[str]]


class CorrelationsDct(TypedDict):
    """Dictionary for all correlations."""

    pearson: dict[str, list[float]]
    concordance: dict[str, list[float]]
    sub_optimal: OutFilesSubDct


class OrganizeFxn(Protocol):
    """Function to organize correlation dictionaries."""

    def __call__(
        self, concor_dct: dict, corr_type: str, quick: bool = False
    ) -> MapDict:
        ...


def read_yml_file(yml_filepath):
    with open(yml_filepath, "r") as f:
        yml_dict = yaml.safe_load(f)

    return yml_dict


def write_yml_file(yml_dict, out_filepath):
    with open(out_filepath, "wt") as f:
        yaml.safe_dump(yml_dict, f)


def read_txt_file(txt_file):
    with open(txt_file, "r") as f:
        strings = f.read().splitlines()
    return strings


def write_txt_file(text_lines, out_filepath):
    with open(out_filepath, "wt") as f:
        for line in text_lines:
            f.write("{0}\n".format(line))


def read_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        dct = pickle.load(f)
    return dct


def write_pickle(dct, out_filepath):
    with open(out_filepath, "wb") as f:
        pickle.dump(dct, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_dct(
    dct: Optional[dict] = None,
    text_lines: Optional[list[str]] = None,
    outname: str = "",
) -> dict[Any, Any]:
    if not text_lines or not outname:
        return {}
    if not dct:
        dct = {outname: text_lines}
    else:
        dct.update({outname: text_lines})
    return dct


def gather_local_filepaths(output_folder_path: str) -> list[str]:
    """Given a local path, return relevant paths within that directory"""
    filepaths = []

    print("Gathering file paths from {0}\n".format(output_folder_path))
    for root, _dirs, files in os.walk(output_folder_path):
        # loops through every file in the directory
        for filename in files:
            # checks if the file is a nifti (.nii.gz)
            if (
                ".nii" in filename
                or ".csv" in filename
                or ".txt" in filename
                or ".1D" in filename
                or ".tsv" in filename
            ):
                filepaths.append(os.path.join(root, filename))

    if len(filepaths) == 0:
        raise FileNotFoundError(
            "\n\n[!] No filepaths were found given the output folder!\n\n"
        )

    return filepaths


class SummaryStats:
    def __init__(self, array: np.ndarray, axis: Optional[int | str] = None) -> None:
        self.mean = np.mean(array, axis=axis)
        self.var = np.var(array, axis=axis)
        self.std = np.sqrt(self.var)
        self.norm = (array - self.mean) / self.std


def pull_NIFTI_file_list_from_s3(s3_directory, s3_creds):
    try:
        from indi_aws import fetch_creds
    except:
        err = (
            "\n\n[!] You need the INDI AWS package installed in order to "
            "pull from an S3 bucket. Try 'pip install indi_aws'\n\n"
        )
        raise Exception(err)

    s3_list = []

    s3_path = s3_directory.replace("s3://", "")
    bucket_name = s3_path.split("/")[0]
    bucket_prefix = s3_path.split(bucket_name + "/")[1]

    bucket = fetch_creds.return_bucket(s3_creds, bucket_name)

    # Build S3-subjects to download
    # maintain the "s3://<bucket_name>" prefix!!
    print("Gathering file paths from {0}\n".format(s3_directory))
    for bk in bucket.objects.filter(Prefix=bucket_prefix):
        if ".nii" in str(bk.key):
            s3_list.append(os.path.join("s3://", bucket_name, str(bk.key)))

    if len(s3_list) == 0:
        err = "\n\n[!] No filepaths were found given the S3 path provided!\n\n"
        raise Exception(err)

    return s3_list


def download_from_s3(s3_path, local_path, s3_creds):
    try:
        from indi_aws import aws_utils, fetch_creds
    except:
        err = (
            "\n\n[!] You need the INDI AWS package installed in order to "
            "pull from an S3 bucket. Try 'pip install indi_aws'\n\n"
        )
        raise Exception(err)

    s3_path = s3_path.replace("s3://", "")
    bucket_name = s3_path.split("/")[0]
    bucket_prefix = s3_path.split(bucket_name + "/")[1]

    filename = s3_path.split("/")[-1]
    local_file = os.path.join(local_path, filename)

    if not os.path.exists(local_file):
        bucket = fetch_creds.return_bucket(s3_creds, bucket_name)
        aws_utils.s3_download(bucket, ([bucket_prefix], [local_file]))

    return local_file


def parse_csv_data(csv_lines):
    parsed_lines = []
    for line in csv_lines:
        if "#" not in line:
            new_row = [float(x.rstrip("\n")) for x in line.split("\t") if x != ""]
            parsed_lines.append(new_row)
        csv_np_data = np.asarray(parsed_lines)

    return csv_np_data


def batch_correlate(
    x: np.ndarray, y: np.ndarray, axis: Optional[Axis] = None
) -> CorrValue:
    """
    Compute a batch of concordance and Pearson correlation coefficients between
    x and y along an axis (or axes).

    References:
        https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    """
    # summary stats
    try:
        summary_stats = {"x": SummaryStats(x), "y": SummaryStats(y)}
    except ZeroDivisionError:
        return CorrValue(np.nan, np.nan)

    # Correlation coefficients
    pearson = np.mean(summary_stats["x"].norm * summary_stats["y"].norm, axis=axis)
    concor = (
        2
        * pearson
        * summary_stats["x"].std
        * summary_stats["y"].std
        / (
            summary_stats["x"].var
            + summary_stats["y"].var
            + (summary_stats["x"].mean - summary_stats["y"].mean) ** 2
        )
    )
    # Squeeze reduced singleton dimensions
    if axis is not None and 1 in concor.shape:
        concor = np.squeeze(concor, axis=axis)
        pearson = np.squeeze(pearson, axis=axis)
    return CorrValue(concor, pearson)


def correlate_text_based(txts: Union[list, tuple]) -> Generator:
    delimiters = tuple(delimiter_from_filepath(path) for path in txts)
    # TODO: why do we drop columns containing na?
    initial_load = [
        pd.read_csv(txt, delimiter=delimiters[i], comment="#").dropna(axis=1)
        for i, txt in enumerate(txts)
    ]
    for i, df in enumerate(initial_load):
        # if we read a value-row as a header, fix that
        try:
            df.columns.astype(float)
            initial_load[i] = pd.read_csv(
                txts[i], delimiter=delimiters[i], comment="#", header=None
            ).dropna(axis=1)
        except ValueError:
            pass
    # assume string columns are indices and not values to correlate
    indices = []
    for i in range(len(initial_load)):
        indices.append(
            np.where(df.apply(lambda _: _.dtype == np.dtypes.ObjectDType))[0]
        )
    oned = []
    for i, index in enumerate(indices):
        if index.shape[0]:
            oned.append(
                pd.read_csv(  # type: ignore
                    txts[i], delimiter=delimiters[i], comment="#", index_col=indices[i]
                )
                .dropna(axis=1)
                .values
            )
        else:
            oned.append(initial_load[i].values)
    return (np.nanmean(measure) for measure in batch_correlate(*oned, axis=0))  # type: ignore


def create_unique_file_dict(
    filepaths: list[str],
    output_folder_path: str,
    replacements: Optional[list[Replacement]] = None,
) -> OutFilesDct:
    """
    Parameters
    ----------
    filepaths : list of str
      list of output filepaths from a CPAC output directory
    output_folder_path : str
      the CPAC output directory the filepaths are from
    replacements : list of str, optional
      a list of strings to be removed from the filepaths should
      they occur

    Returns
    -------
    files_dict : dict
        a dictionary of dictionaries, format:
        files_dict["centrality"] =
            {("centrality", midpath, nums): <filepath>, ..}
    """
    files_dict: OutFilesDct = {}

    for filepath in filepaths:
        if "_stack" in filepath:
            continue

        if ("itk" in filepath) or ("xfm" in filepath) or ("montage" in filepath):
            continue
        path_changes = []
        real_filepath = filepath
        if replacements:
            for word_couple in replacements:
                if isinstance(word_couple, tuple):
                    word, new = word_couple
                else:
                    if "," not in word_couple:
                        raise SyntaxError(
                            "\n\n[!] In the replacements text file, the old "
                            "substring and its replacement must be separated "
                            "by a comma.\n\n"
                        )
                    word, new = word_couple.split(",")
                if word in filepath:
                    path_changes.append(f"old: {filepath}")
                    filepath = filepath.replace(word, new)
                    path_changes.append(f"new: {filepath}")
        if path_changes:
            with open(os.path.join(os.getcwd(), "path_changes.txt"), "wt") as f:
                for path in path_changes:
                    f.write(path)
                    f.write("\n")

        filename = filepath.split("/")[-1]

        # name of the directory the file is in
        folder = filepath.split("/")[-2]

        midpath = filepath.replace(output_folder_path, "")
        midpath = midpath.replace(filename, "")

        pre180 = False
        if pre180:
            # name of the output type/derivative
            try:
                category = midpath.split("/")[2]
            except IndexError as e:
                continue

            if "eigenvector" in filepath:
                category = category + ": eigenvector"
            if "degree" in filepath:
                category = category + ": degree"
            if "lfcd" in filepath:
                category = category + ": lfcd"
        else:
            category = filename
            category = category.rstrip(".gz").rstrip(".nii")

            excl_tags = ["sub-", "ses-", "task-", "run-", "acq-"]

            # len(filetag) == 1 is temporary for broken/missing ses-* tag
            for filetag in filename.split("_"):
                for exctag in excl_tags:
                    if exctag in filetag or len(filetag) == 1:
                        category = category.replace(f"{filetag}_", "")

        # this provides a way to safely identify the specific file
        # without relying on a full string of the filename (because
        # this can change between versions depending on what any given
        # processing tool appends to output file names)
        nums_in_folder = [int(s) for s in folder if s.isdigit()]
        nums_in_filename = [int(s) for s in filename if s.isdigit()]

        file_nums = ""

        for num in nums_in_folder:
            file_nums = file_nums + str(num)

        for num in nums_in_filename:
            file_nums = file_nums + str(num)

        # load these settings into the tuple so that the file can be
        # identified without relying on its full path (as it would be
        # impossible to match files from two regression tests just
        # based on their filepaths)
        file_tuple = (category, midpath, file_nums)

        temp_dict: OutFilesSubDct = {}
        temp_dict[file_tuple] = [real_filepath]

        if category not in files_dict.keys():
            files_dict[category] = {}

        files_dict[category].update(temp_dict)

    return files_dict


def gather_all_files(
    input_dct: InputDct, pickle_dir: str | Path, source: DirType = "output_dir"
) -> tuple[OutFilesDct, OutFilesDct]:
    """
    Given an input dictionary, a pickle directory, and (optionally) a source,
    returns a pair of dicts
    """
    file_dct_list: list[OutFilesDct] = [{}, {}]
    assert len(input_dct["pipelines"]) == 2
    for index, (key, pipe_dct) in enumerate(input_dct["pipelines"].items()):
        pipe_outdir = str(pipe_dct[source])

        if input_dct["settings"]["s3_creds"]:
            if not "s3://" in pipe_outdir:
                raise ValueError(
                    "\n\n[!] If pulling output files from an S3 bucket, the "
                    "output folder path must have the s3:// prefix.\n\n"
                )
        else:
            pipe_outdir = os.path.abspath(pipe_outdir).rstrip("/")

        # pipeline_name = pipe_outdir.split("/")[-1]

        # if source == "output_dir" and "pipeline_" not in pipeline_name:
        #    err = "\n\n[!] Your pipeline output directory has to be a specific " \
        #          "one that has the 'pipeline_' prefix.\n\n(Not the main output " \
        #          "directory that contains all of the 'pipeline_X' subdirectories," \
        #          "and not a specific participant's output subdirectory either.)\n"
        #    raise Exception(err)

        output_pkl = os.path.join(pickle_dir, f"{key}_{source}_paths.p")

        if os.path.exists(output_pkl):
            print(
                f"Found output list pickle for {key}, skipping output file"
                "path parsing.."
            )
            pipeline_files_dct: OutFilesDct = read_pickle(output_pkl)
        else:
            if input_dct["settings"]["s3_creds"]:
                pipeline_files_list = pull_NIFTI_file_list_from_s3(
                    pipe_outdir, input_dct["settings"]["s3_creds"]
                )
            else:
                pipeline_files_list = gather_local_filepaths(pipe_outdir)
            pipeline_files_dct = create_unique_file_dict(
                pipeline_files_list, pipe_outdir, pipe_dct["replacements"]
            )
            write_pickle(pipeline_files_dct, output_pkl)
        file_dct_list[index] = pipeline_files_dct

    return cast(tuple[OutFilesDct, OutFilesDct], tuple(file_dct_list))


def match_filepaths(
    old_files_dict: OutFilesDct,
    new_files_dict: OutFilesDct,
) -> MatchedFilepaths:
    """Returns a dictionary mapping each filepath from the first C-PAC
    run to the second one, matched to derivative, strategy, and scan.

    Parameters
    ----------
    old_files_dict, new_files_dict
        each key is a derivative name, and each value is another
        dictionary keying (derivative, mid-path, last digit in path)
        tuples to a list containing the full filepath described by
        the tuple that is the key

    Returns
    -------
    matched_path_dict
        same as the input dictionaries, except the list in the
        sub-dictionary value has both file paths that are matched
    """

    # file path matching
    matched_path_dict: OutFilesDct = {}
    missing_in_old: list[OutFilesSubDct | tuple[str, str, str]] = []
    missing_in_new: list[OutFilesSubDct] = []

    for key in new_files_dict:
        # for types of derivative...
        if key in old_files_dict.keys():
            for file_id in new_files_dict[key]:
                if file_id in old_files_dict[key].keys():
                    if key not in matched_path_dict.keys():
                        matched_path_dict[key] = {}

                    matched_path_dict[key][file_id] = (
                        old_files_dict[key][file_id] + new_files_dict[key][file_id]
                    )

                else:
                    missing_in_old.append(file_id)  # new_files_dict[key][file_id])
        else:
            missing_in_old.append(new_files_dict[key])

    # find out what is in the last version's outputs that isn't in the new
    # version's outputs
    for key in old_files_dict:
        if new_files_dict.get(key) != None:
            missing_in_new.append(old_files_dict[key])

    if len(matched_path_dict) == 0:
        err = (
            "\n\n[!] No output paths were successfully matched between "
            "the two CPAC output directories!\n\n"
        )
        raise Exception(err)

    matched_files_dct: MatchedFilepaths = {
        "matched": matched_path_dict,
        "missing_old": missing_in_old,
        "missing_new": missing_in_new,
    }

    return matched_files_dct


def delimiter_from_filepath(filepath: Union[Path, str]) -> Optional[str]:
    """
    Given a filepath, return expected value-separator delimiter
    """
    filepath = str(filepath)
    if filepath.endswith(".tsv"):
        return "\t"
    if filepath.endswith(".csv"):
        return ","
    with open(filepath, "r", encoding="utf8") as _f:
        first_line = "#"
        while first_line.lstrip().startswith("#"):
            first_line = _f.readline()
        for delimiter in ["\t", ",", " "]:
            if delimiter in first_line:
                if delimiter == " ":
                    return r"\s+"
                return delimiter
    return None


def report_missing(matched_dct, pipelines, output_dir):
    flat_missing_dct = {"missing_old": {}, "missing_new": {}}

    for missing_type in flat_missing_dct:
        for subdct in matched_dct[missing_type]:
            if type(subdct) is not dict:
                continue
            for key, path_list in subdct.items():
                if key[0] not in flat_missing_dct[missing_type].keys():
                    flat_missing_dct[missing_type][key[0]] = []
                flat_missing_dct[missing_type][key[0]].append(path_list[0])

    report_msg = ""

    if flat_missing_dct["missing_new"]:
        report_msg += (
            "\nThese outputs are in {0}, and are either missing "
            "in {1} or were not picked up by this script's file parsing:"
            "\n\n".format(list(pipelines)[0], list(pipelines)[1])
        )
        for output_type in flat_missing_dct["missing_new"]:
            report_msg += "\n{0}:\n\n".format(output_type)
            for path in flat_missing_dct["missing_new"][output_type]:
                report_msg += "    {0}\n".format(path)
        report_file = os.path.join(output_dir, "report_missing_new.txt")
        with open(report_file, "wt") as f:
            f.write(report_msg)

    if flat_missing_dct["missing_old"]:
        report_msg += (
            "\nThese outputs are in {0}, and missing "
            "in {1} or were not picked up by this script's file parsing:"
            "\n\n".format(list(pipelines)[1], list(pipelines)[0])
        )
        for output_type in flat_missing_dct["missing_old"]:
            report_msg += "\n{0}:\n\n".format(output_type)
            for path in flat_missing_dct["missing_old"][output_type]:
                report_msg += "    {0}\n".format(path)
        report_file = os.path.join(output_dir, "report_missing_old.txt")
        with open(report_file, "wt") as f:
            f.write(report_msg)


def calculate_correlation(args_tuple):
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
            old_local_file = os.path.join(
                local_dir, "s3_input_files", old_path.replace("s3://", "")
            )
            # directory without filename
            old_local_path = old_local_file.replace(old_path.split("/")[-1], "")

            new_local_file = os.path.join(
                local_dir, "s3_input_files", new_path.replace("s3://", "")
            )
            new_local_path = new_local_file.replace(new_path.split("/")[-1], "")

            if not os.path.exists(old_local_path):
                os.makedirs(old_local_path)
            if not os.path.exists(new_local_path):
                os.makedirs(new_local_path)

        except Exception as e:
            err = (
                "\n\nLocals: {0}\n\n[!] Could not create the local S3 "
                "download directory.\n\nError details: {1}\n\n".format(locals(), e)
            )
            raise Exception(err)

        try:
            if not os.path.exists(old_local_file):
                old_path = download_from_s3(old_path, old_local_path, s3_creds)
            else:
                old_path = old_local_file
        except Exception as e:
            err = (
                "\n\nLocals: {0}\n\n[!] Could not download the files from "
                "the S3 bucket. \nS3 filepath: {1}\nLocal destination: {2}"
                "\nS3 creds: {3}\n\nError details: {4}\n\n".format(
                    locals(), old_path, old_local_path, s3_creds, e
                )
            )
            raise Exception(e)

        try:
            if not os.path.exists(new_local_file):
                new_path = download_from_s3(new_path, new_local_path, s3_creds)
            else:
                new_path = new_local_file
        except Exception as e:
            err = (
                "\n\nLocals: {0}\n\n[!] Could not download the files from "
                "the S3 bucket. \nS3 filepath: {1}\nLocal destination: {2}"
                "\nS3 creds: {3}\n\nError details: {4}\n\n".format(
                    locals(), new_path, new_local_path, s3_creds, e
                )
            )
            raise Exception(e)

    ## nibabel to pull the data from the re-assembled file paths
    if os.path.exists(old_path) and os.path.exists(new_path):
        if (
            (".csv" in old_path and ".csv" in new_path)
            or (".txt" in old_path and ".txt" in new_path)
            or (".1D" in old_path and ".1D" in new_path)
            or (".tsv" in old_path and ".tsv" in new_path)
        ):
            try:
                concor, pearson = correlate_text_based((old_path, new_path))
            except Exception as e:
                return category, e, (old_path, new_path)

            if concor > 0.980:
                corr_tuple = (category, [concor], [pearson])
            else:
                corr_tuple = (category, [concor], [pearson], (old_path, new_path))
            if verbose:
                print("Success - {0}".format(str(concor)))

            return corr_tuple

        try:
            old_file_img = nb.load(old_path)
            old_file_hdr = old_file_img.header
            new_file_img = nb.load(new_path)
            new_file_hdr = new_file_img.header

            old_file_dims = old_file_hdr.get_zooms()
            # new_file_dims = new_file_hdr.get_zooms()

            old_img = nb.load(old_path)
            new_img = nb.load(new_path)

            if nb.orientations.aff2axcodes(
                old_img.affine
            ) != nb.orientations.aff2axcodes(new_img.affine):
                # reorient old image to new image's orientation
                old_img = nb.processing.resample_from_to(old_img, new_img)

            data_1 = old_img.get_fdata()
            data_2 = new_img.get_fdata()

        except Exception as e:
            corr_tuple = (f"file reading problem: {e}", old_path, new_path)
            if verbose:
                print(str(corr_tuple))
            return corr_tuple

        ## set up and run the Pearson correlation and concordance correlation
        if data_1.flatten().shape == data_2.flatten().shape:
            try:
                if len(old_file_dims) > 3:
                    axis = tuple(range(3, len(old_file_dims)))
                    concor, pearson = batch_correlate(data_1, data_2, axis=axis)
                    concor = np.nanmean(concor)
                    pearson = np.nanmean(pearson)
                else:
                    concor, pearson = batch_correlate(data_1, data_2)
            except Exception as e:
                corr_tuple = (f"correlating problem: {e}", old_path, new_path)
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


def run_correlations(
    matched_dct, input_dct: InputDct, source="output_dir", quick=False, verbose=False
):
    all_corr_dct: CorrelationsDct = {
        "pearson": {},
        "concordance": {},
        "sub_optimal": {},
    }

    args_list = []

    quick_list = [
        "anatomical_brain",
        "anatomical_csf_mask",
        "anatomical_gm_mask",
        "anatomical_wm_mask",
        "anatomical_to_standard",
        "functional_preprocessed",
        "functional_brain_mask",
        "mean_functional_in_anat",
        "functional_nuisance_residuals",
        "functional_nuisance_regressors",
        "functional_to_standard",
        "roi_timeseries",
    ]

    matched_path_dct = matched_dct["matched"]
    output_dir = input_dct["settings"]["correlations_dir"]
    s3_creds = input_dct["settings"]["s3_creds"]

    for category in matched_path_dct.keys():
        if quick:
            if category not in quick_list:
                continue

        for file_id in matched_path_dct[category].keys():
            old_path = matched_path_dct[category][file_id][0]
            new_path = matched_path_dct[category][file_id][1]

            if source == "work_dir":
                args_list.append(
                    (file_id, old_path, new_path, output_dir, s3_creds, verbose)
                )
            else:
                args_list.append(
                    (category, old_path, new_path, output_dir, s3_creds, verbose)
                )

    print("\nNumber of correlations to calculate: {0}\n".format(len(args_list)))

    print("Running correlations...")
    p = Pool(int(input_dct["settings"]["n_cpus"]))
    corr_tuple_list = p.map(calculate_correlation, args_list)
    p.close()
    p.join()

    print("\nCorrelations of the {0} are done.\n".format(source))

    failures = []

    for corr_tuple in corr_tuple_list:
        if not corr_tuple:
            continue
        if isinstance(corr_tuple[1], Exception):
            failures.append(
                (corr_tuple[0], corr_tuple[1], " | ".join(str(corr_tuple[2])))
            )
            continue
        if corr_tuple[0] not in all_corr_dct["concordance"].keys():
            all_corr_dct["concordance"][corr_tuple[0]] = []
        if corr_tuple[0] not in all_corr_dct["pearson"].keys():
            all_corr_dct["pearson"][corr_tuple[0]] = []
        all_corr_dct["concordance"][corr_tuple[0]] += corr_tuple[1]
        all_corr_dct["pearson"][corr_tuple[0]] += corr_tuple[2]

        if len(corr_tuple) > 3:
            if corr_tuple[0] not in all_corr_dct["sub_optimal"].keys():
                all_corr_dct["sub_optimal"][corr_tuple[0]] = []
            try:
                all_corr_dct["sub_optimal"][corr_tuple[0]].append(
                    f"{corr_tuple[1][0]}:\n{corr_tuple[3][0]}\n{corr_tuple[3][1]}\n\n"
                )
            except TypeError:
                pass

    return all_corr_dct, failures


def post180_organize_correlations(
    concor_dct, corr_type="concordance", quick=False
) -> MapDict:
    corr_map_dct = cast(MapDict, {"correlations": {}})
    for key in concor_dct:
        if "problem" in key:
            continue
        # shouldn't need this - FIX
        rawkey = key.replace("acq-", "").replace("run-", "")
        datatype = rawkey.split("_")[-1]

        if datatype not in corr_map_dct["correlations"]:
            corr_map_dct["correlations"][datatype] = {}
        corr_map_dct["correlations"][datatype][rawkey] = concor_dct[key]

    return corr_map_dct


def organize_correlations(concor_dct, corr_type="concordance", quick=False) -> MapDict:
    # break up all of the correlations into groups - each group of derivatives
    # will go into its own boxplot

    regCorrMap = {}
    native_outputs = {}
    template_outputs = {}
    timeseries = {}
    functionals = {}

    core = {}

    corr_map_dict = cast(MapDict, {"correlations": {}})

    derivs = ["alff", "dr_tempreg", "reho", "sca_roi", "timeseries", "ndmg"]
    anats = ["anatomical", "seg"]
    time_series = [
        "functional_freq",
        "nuisance_residuals",
        "functional_preprocessed",
        "functional_to_standard",
        "ica_aroma_",
        "motion_correct",
        "slice_time",
    ]
    funcs = ["functional", "displacement"]

    for key in concor_dct:
        if quick:
            core[key] = concor_dct[key]
            continue

        if "xfm" in key or "mixel" in key:
            continue

        if "centrality" in key or "vmhc" in key or "sca_tempreg" in key:
            template_outputs[key] = concor_dct[key]
            continue

        for word in anats:
            if word in key:
                regCorrMap[key] = concor_dct[key]
                continue

        for word in derivs:
            if word in key and "standard" not in key:
                native_outputs[key] = concor_dct[key]
                continue
            if word in key:
                template_outputs[key] = concor_dct[key]
                continue

        for word in time_series:
            if word in key and "mean" not in key and "mask" not in key:
                timeseries[key] = concor_dct[key]
                continue

        for word in funcs:
            if word in key:
                functionals[key] = concor_dct[key]

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


def quick_summary(
    dct: QuickSummary, corr_map_dct: MapDict, output_dir: str | Path
) -> QuickSummary:
    for corr_group in corr_map_dct["correlations"].keys():
        lines: list[str] = []
        for output_type, corr_vec in dict(
            corr_map_dct["correlations"][corr_group]
        ).items():
            try:
                corrmean = np.mean(np.asarray(corr_vec))
            except TypeError:
                continue
            lines.append("{0}: {1}".format(output_type, corrmean))

            write_txt_file(
                lines, os.path.join(output_dir, "average_{0}.txt".format(corr_group))
            )
            dct = write_dct(dct, lines, str(output_type))
    return dct


def create_boxplot(corr_group, corr_group_name, pipeline_names=None, current_dir=None):
    from matplotlib import pyplot

    if not pipeline_names:
        pipeline_names = ["pipeline_1", "pipeline_2"]
    if not current_dir:
        current_dir = os.getcwd()

    allData = []
    labels = list(corr_group.keys())
    labels.sort()

    for label in labels:
        if "file reading problem" in label:
            continue
        try:
            allData.append(np.asarray(corr_group[label]).astype(float))
        except ValueError as ve:
            continue
            # raise Exception(ve)

    pyplot.boxplot(allData)
    pyplot.xticks(range(1, (len(corr_group) + 1)), labels, rotation=85)
    pyplot.margins(0.5, 1.0)
    # pyplot.ylim(0.5,1.2)
    pyplot.xlabel("Derivatives")
    pyplot.title(
        f"Correlations between {list(pipeline_names)[0]} and "
        f"{list(pipeline_names)[1]}\n( {corr_group_name} )"
    )

    output_filename = os.path.join(
        current_dir,
        (
            corr_group_name
            + "_"
            + list(pipeline_names)[0]
            + "_and_"
            + list(pipeline_names)[1]
        ),
    )

    pyplot.savefig(
        "{0}.png".format(output_filename), format="png", dpi=200, bbox_inches="tight"
    )
    pyplot.close()


@overload
def compare_pipelines(
    input_dct: InputDct, dir_type: Literal["log_dir", "work_dir"]
) -> None:
    ...


@overload
def compare_pipelines(
    input_dct: InputDct, dir_type: Literal["output_dir"]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    ...


def compare_pipelines(
    input_dct: InputDct, dir_type: DirType = "output_dir"
) -> Optional[
    tuple[CorrelationMap, CorrelationMap] | tuple[QuickSummary, QuickSummary]
]:
    """
    Given an input dict containing keys 'settings', gather previously
    generated pickles or all relevant output and working files

    Returns
    -------
    corr_map : dict

    pearson_map : dict
    """
    output_dir = input_dct["settings"]["output_dir"]
    pickle_dir = input_dct["settings"]["pickle_dir"]

    corrs_pkl = os.path.join(pickle_dir, f"{dir_type}_correlations.p")
    failures_pkl = os.path.join(pickle_dir, f"{dir_type}_failures.p")
    matched_pkl = os.path.join(pickle_dir, f"{dir_type}_matched_files.p")

    all_corr_dct: Optional[CorrelationsDct] = None
    matched_dct: Optional[MatchedFilepaths] = None
    if os.path.exists(corrs_pkl):
        print(
            f"\n\nFound the correlations pickle: {corrs_pkl}\n\n"
            "Starting from there..\n"
        )
        all_corr_dct = read_pickle(corrs_pkl)
    if os.path.exists(matched_pkl):
        print(
            f"\n\nFound the matched filepaths pickle: {matched_pkl}\n\n"
            "Starting from there..\n"
        )
        matched_dct = read_pickle(matched_pkl)

        if dir_type == "output_dir":
            report_missing(matched_dct, input_dct["pipelines"].keys(), output_dir)

    else:
        # gather all relevant output and working files
        outfiles1_dct, outfiles2_dct = gather_all_files(
            input_dct, pickle_dir, source=dir_type
        )
        matched_dct = match_filepaths(outfiles1_dct, outfiles2_dct)
        write_pickle(matched_dct, matched_pkl)

        if dir_type == "output_dir":
            report_missing(matched_dct, input_dct["pipelines"].keys(), output_dir)

    if not matched_dct:
        return {}, {}

    subjects: list[str] = [
        key[2] for key in matched_dct["matched"][list(matched_dct["matched"].keys())[0]]
    ]

    if not all_corr_dct:
        all_corr_dct, failures = run_correlations(
            matched_dct,
            input_dct,
            source=dir_type,
            quick=input_dct["settings"]["quick"],
            verbose=input_dct["settings"]["verbose"],
        )
        write_pickle(all_corr_dct, corrs_pkl)
        if failures:
            write_pickle(failures, failures_pkl)

    correlations_json: list[dict[str, str]] = [
        {"rowid": key, "columnid": subjects[i % len(subjects)], "value": str(value)}
        for correlation_type in ["pearson", "concordance"]
        for key, value_list in all_corr_dct[correlation_type].items()
        for i, value in enumerate(value_list)
    ]
    correlations_json_dir: Path = Path(output_dir).parent / "correlations"
    if not correlations_json_dir.exists():
        correlations_json_dir.mkdir(parents=True, exist_ok=True)
    correlations_json_path: Path = correlations_json_dir / "correlations.json"

    if correlations_json_path.exists():
        with correlations_json_path.open("r", encoding="utf8") as json_file:
            _loaded_json: list[dict[str, str]] = []
            try:
                _loaded_json = json.load(json_file)
            except json.decoder.JSONDecodeError:
                pass  # empty or corrupted file
            correlations_json = [*_loaded_json, *correlations_json]
    with correlations_json_path.open("w", encoding="utf8") as json_file:
        try:
            flock(json_file.fileno(), LOCK_EX)  # Lock the file
            json.dump(correlations_json, json_file, indent=2)
        finally:
            flock(json_file.fileno(), LOCK_UN)  # Unlock the file
    if all_corr_dct["sub_optimal"]:
        write_yml_file(
            all_corr_dct["sub_optimal"], os.path.join(output_dir, "sub_optimal.yml")
        )

    if dir_type == "work_dir":
        sorted_vals = []
        # sorted_keys = sorted(all_corr_dct, key=all_corr_dct.get)
        for key in all_corr_dct.keys():  # sorted_keys:
            if (
                "file reading problem:" in key
                or "different shape" in key
                or "correlating problem" in key
            ):
                continue
            sorted_vals.append("{0}: {1}".format(all_corr_dct[key], key))
        working_corrs_file = os.path.join(output_dir, "work_dir_correlations.txt")
        with open(working_corrs_file, "wt") as f:
            for line in sorted_vals:
                f.write(line)
                f.write("\n")
        return None

    pre180 = False
    if pre180:
        organize: OrganizeFxn = organize_correlations
    else:
        organize = post180_organize_correlations

    corr_map_dict: MapDict = organize(
        all_corr_dct["concordance"],
        "concordance",
        quick=input_dct["settings"]["quick"],
    )
    corr_map_dict["pipeline_names"] = list(input_dct["pipelines"].keys())

    pearson_map_dict: MapDict = organize(
        all_corr_dct["pearson"], "pearson", quick=input_dct["settings"]["quick"]
    )
    pearson_map_dict["pipeline_names"] = list(input_dct["pipelines"].keys())
    dct: QuickSummary = {}
    corr_map = quick_summary(dct, corr_map_dict, output_dir)
    pearson_map = quick_summary(dct, pearson_map_dict, output_dir)

    for corr_group_name in corr_map_dict["correlations"].keys():
        corr_group = corr_map_dict["correlations"][corr_group_name]
        create_boxplot(
            corr_group, corr_group_name, corr_map_dict["pipeline_names"], output_dir
        )

    for corr_group_name in pearson_map_dict["correlations"].keys():
        corr_group = pearson_map_dict["correlations"][corr_group_name]
        create_boxplot(
            corr_group, corr_group_name, pearson_map_dict["pipeline_names"], output_dir
        )

    return corr_map, pearson_map


class CpacCorrelationsNamespace(argparse.Namespace):
    """Namespace with attributes required for cpac_correlations."""

    branch: str
    data_source: str
    input_yaml: str

    @property
    def cli_args(self) -> list[str]:
        """Return list of CLI arg strings.

        >>> CpacCorrelationsNamespace(branch="develop", data_source="HNU_1",
        ...                           input_yaml="input.yaml").cli_args
        ['--branch', 'develop', '--data_source', 'HNU_1', 'input.yaml']
        """
        return [
            *list(
                chain.from_iterable(
                    [
                        [f"--{key}", value]
                        for key, value in vars(self).items()
                        if key != "input_yaml"
                    ]
                )
            ),
            self.input_yaml,
        ]


def parse_args() -> CpacCorrelationsNamespace:
    """Parse commandline args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_yaml", type=str, help="file path of the script's input YAML"
    )
    parser.add_argument("--data_source", type=str, help="Which site data comes from")
    parser.add_argument("--branch", type=str, help="Branch name")
    return parser.parse_args(namespace=CpacCorrelationsNamespace())


def main(
    args: Optional[CpacCorrelationsNamespace] = None,
) -> tuple[list[str], str, str]:
    """Correlate two C-PAC runs.

    • Parse commandline arguments
    • Read input YAML
    • Check for already completed stuff (pickles)
    """
    if args is None:
        args = parse_args()
    data_source: str = args.data_source
    branch: str = args.branch

    # get the input info
    input_dct: InputDct = read_yml_file(args.input_yaml)

    # check for already completed stuff (pickles)
    output_dir: str = os.path.join(
        os.getcwd(), f"correlations_{input_dct['settings']['run_name']}"
    )
    pickle_dir: str = os.path.join(output_dir, "pickles")

    if not os.path.exists(pickle_dir):
        try:
            os.makedirs(pickle_dir)
        except:
            err: str = (
                "\n\n[!] Could not create the output directory for the "
                "correlations. Do you have write permissions?\nAttempted "
                f"output directory: {output_dir}\n\n"
            )
            raise Exception(err)

    input_dct["settings"].update({"output_dir": output_dir, "pickle_dir": pickle_dir})

    corr_map: CorrelationMap
    pearson_map: CorrelationMap
    corr_map, pearson_map = compare_pipelines(input_dct, dir_type="output_dir")
    corr_map_keys = list(corr_map.keys())
    all_keys: list[str] = []
    for key in corr_map_keys:
        keys = list(corr_map[key])
        for i in keys:
            all_keys.append(i)
    return all_keys, data_source, branch


if __name__ == "__main__":
    main()

cpac_correlations = main

__all__ = [
    "batch_correlate",
    "calculate_correlation",
    "compare_pipelines",
    "correlate_text_based",
    "CorrValue",
    "CpacCorrelationsNamespace",
    "cpac_correlations",
    "create_boxplot",
    "create_unique_file_dict",
    "delimiter_from_filepath",
    "download_from_s3",
    "gather_all_files",
    "gather_local_filepaths",
    "main",
    "match_filepaths",
    "organize_correlations",
    "parse_csv_data",
    "post180_organize_correlations",
    "pull_NIFTI_file_list_from_s3",
    "quick_summary",
    "read_pickle",
    "read_txt_file",
    "read_yml_file",
    "report_missing",
    "run_correlations",
    "SummaryStats",
    "write_dct",
    "write_pickle",
    "write_txt_file",
    "write_yml_file",
]
