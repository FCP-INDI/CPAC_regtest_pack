"""Generate a correlation matrix between two output directories

Copyright (C) 2022  C-PAC Developers
This file is part of CPAC_regtest_pack.
CPAC_regtest_pack is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
CPAC_regtest_pack is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with CPAC_regtest_pack. If not, see
<https://www.gnu.org/licenses/>"""
# coding=utf-8
# pylint: disable=wrong-import-position
import sys

if sys.version_info < (3, 6):
    raise EnvironmentError("This module requires Python 3.6 or newer.")

import argparse
# import glob
# from itertools import chain
# import pandas as pd
# import scipy.io as sio
# from afnipy.lib_afni1D import Afni1D
# from scipy.stats import pearsonr
# from tabulate import tabulate

from .directory import determine_software_and_root, iterate_features
from .subjects import gather_unique_ids

# sorted_keys = list(feature_headers.keys())
# sorted_keys.sort(key=str.lower)
# feat_def_table = tabulate(
#     [
#         [
#             key,
#             feature_headers[key].get("name"),
#             feature_headers[key].get("link")
#         ] for key in sorted_keys
#     ],
#     headers=["key", "feature name", "documentation link"]
# )
# del sorted_keys


def main():
    parser = argparse.ArgumentParser(
        description="Create a correlation matrix between two C-PAC output "
                    "directories.",
        # epilog="The following features currently have available definitions "
        #        "to calculate Pearson's \x1B[3mr\x1B[23m between C-PAC and "
        #        f"fmriprep:\n\n{feat_def_table}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("outputs_path", nargs=2, type=str,
                        help="path to an outputs directory")

    # parser.add_argument("--subject_list", type=str, nargs="*")

    # parser.add_argument("--session", type=str,
    #                     help="limit to a single given session")

    # parser.add_argument("--feature_list", type=str, nargs="*",
    #                     default=regressor_list + motion_list,
    #                     help="default: %(default)s")

    parser.add_argument("--num_cores", type=int, default=4,
                        help="number of cores to use - will calculate "
                             "correlations in parallel if greater than 1")

    parser.add_argument("run_name", type=str,
                        help="name for the correlations run")

    args = parser.parse_args()

    a_software, a_root = determine_software_and_root(args.outputs_path[0])
    b_software, b_root = determine_software_and_root(args.outputs_path[1])
    a_ids = gather_unique_ids(a_root)
    b_ids = gather_unique_ids(b_root)
    unique_ids = a_ids[0].intersection(b_ids[0])
    most_specific_ids = [id for id in unique_ids if
                         len(id) == max(len(id) for id in unique_ids)]
    a_files = a_ids[1].get_files()
    b_files = b_ids[1].get_files()
    features = {}
    for specific_id in most_specific_ids:
        for file in a_files:
            features = iterate_features(features, file, specific_id,
                                        a_software, 'a')
        for file in b_files:
            features = iterate_features(features, file, specific_id,
                                        b_software, 'b')
    print(features)

    # subject_list = args.subject_list if (
    #     "subject_list" in args and args.subject_list is not None
    # ) else generate_subject_list_for_directory(args.outputs_path[0])

    # if "session" in args and args.session is not None:
    #     subject_list = [
    #         sub for sub in subject_list if sub.endswith(str(args.session))
    #     ]

    # corrs = Correlation_Matrix(
    #     subject_list,
    #     args.feature_list,
    #     [{
    #         "software": args.new_outputs_software,
    #         "run_path": args.new_outputs_path if args.new_outputs_path.endswith(
    #             "/"
    #         ) else f"{args.new_outputs_path}/"
    #     }, {
    #         "software": args.old_outputs_software,
    #         "run_path": args.old_outputs_path if args.old_outputs_path.endswith(
    #             "/"
    #         ) else f"{args.old_outputs_path}/"
    #     }]
    # )

    # path_table = corrs.print_filepaths(plaintext=True)

    # if args.save:
    #     output_dir = os.path.join(
    #         os.getcwd(), "correlations_{0}".format(args.run_name)
    #     )

    #     if not os.path.exists(output_dir):
    #         try:
    #             os.makedirs(output_dir)
    #         except:
    #             err = ("\n\n[!] Could not create the output directory for the "
    #                    "correlations. Do you have write permissions?\n "
    #                    f"Attempted output directory: {output_dir}\n\n")
    #             raise Exception(err)

    #     path_table.to_csv(os.path.join(output_dir, "filepaths.csv"))
    #     sio.savemat(
    #         os.path.join(output_dir, "corrs.mat"), {'corrs':corrs.corrs}
    #     )

    # generate_heatmap(
    #     reshape_corrs(corrs.corrs),
    #     args.feature_list,
    #     subject_list,
    #     save_path=os.path.join(
    #         output_dir, "heatmap.png"
    #     ) if args.save else args.save,
    #     title=f"{args.new_outputs_software} "
    #     f"{args.new_outputs_path.split('/')[-1]} vs "
    #     f"{args.old_outputs_software} {args.old_outputs_path.split('/')[-1]}"
    # )


# class Subject_Session_Feature:
#     """
#     A class for (subject × session) × feature data
#     """
#     def __init__(self, subject, feature, runs):
#         """
#         Parameters
#         ----------
#         subject: str
#             (subject × session)

#         feature: str

#         runs: list of dicts
#             [{"software": str, "run_path": str}]
#         """
#         if "_" in subject:
#             self.subject, self.session = subject.split("_", 1)
#         else:
#             self.subject = subject
#             self.session = None
#         self.feature = feature
#         self.paths = (
#             get_paths(
#                 self.subject,
#                 self.feature,
#                 runs[0]["run_path"],
#                 runs[0]["software"],
#                 self.session
#             ),
#             get_paths(
#                 self.subject,
#                 self.feature,
#                 runs[1]["run_path"],
#                 runs[1]["software"],
#                 self.session
#             )
#         )
#         self.data = (
#             self.read_feature(
#                 self.paths[0],
#                 self.feature,
#                 runs[0]["software"]
#             ),
#             self.read_feature(
#                 self.paths[1],
#                 self.feature,
#                 runs[1]["software"]
#             )
#         )
#         if self.data[0] is not None:
#             print(f"{runs[0]['software']} {self.feature}: {len(self.data[0])}")
#         if self.data[1] is not None:
#             print(f"{runs[1]['software']} {self.feature}: {len(self.data[1])}")

#     def read_feature(self, files, feature, software="C-PAC"):
#         """
#         Method to read a feature from a given file

#         Parameters
#         ----------
#         files: list of str
#             paths to files

#         feature: str

#         software: str

#         Returns
#         -------
#         feature: np.ndarray or list or None
#         """
#         if not files:
#             return(None)
#         software = "C-PAC" if software.lower() in [
#             "c-pac",
#             "cpac"
#         ] else software.lower()

#         feature_label = get_feature_label(feature, software)

#         if software == "C-PAC":
#             for file in files:
#                 if file.endswith(".1D"):
#                     data = Afni1D(file)
#                     if "compcor" in file.lower():
#                         return(data.mat[int(feature_label[1][-1])][1:])
#                     header = data.header[-1] if len(data.header) else ""
#                     header_list = header.split('\t')
#                     if isinstance(feature_label, list):
#                         for fl in feature_label:
#                             if fl in header_list:
#                                 return data.mat[header_list.index(fl)]
#                     else:
#                         return data.mat[header_list.index(feature_label)] if (
#                             feature_label in header_list
#                         ) else data.mat[0][1:] if (
#                             len(data.mat[:]) == 1
#                         ) else ([None] * len(data.mat[0][1:]))
#                 elif file.endswith('.csv'):
#                     return list(pd.read_csv(
#                         file, sep="\t"
#                     )["Sub-brick"][1:].dropna().astype(float).values)

#         elif software == "fmriprep":
#             for file in files:
#                 if file.endswith(".tsv"):
#                     data = pd.read_csv(file, sep='\t')
#                     if feature_label in data.columns:
#                         return(data[feature_label])
#                 elif file.endswith(".txt"):
#                     with open(file) as f:
#                         return([
#                             float(x) for x in [
#                                 x.strip() for x in f.readlines()
#                             ][1:]
#                         ])
#         return


# class Correlation_Matrix:
#     """
#     A class for (subject × session) × feature correlation matrices
#     """
#     def __init__(self, subject_sessions, features, runs):
#         """
#         Parameters
#         ----------
#         subject_sessions: list of strings
#             ["subject_session", ...]

#         features: list of strings
#             ["feature", ...]

#         runs: list of dicts
#             [{"software": str, "run_path": str}]
#         """
#         self.subjects = subject_sessions
#         self.features = features
#         self.runs = runs
#         self.data = {
#             subject: {
#                 feature: Subject_Session_Feature(
#                     subject, feature, runs
#                 ) for feature in features
#             } for subject in subject_sessions
#         }
#         self.corrs = np.zeros((len(subject_sessions), len(features)))
#         self.run_pearsonsr()

#     def print_filepaths(self, plaintext=False):
#         """
#         Function to print a table
#         """
#         columns = ["\n".join([
#             self.runs[i]["software"], self.runs[i]["run_path"]
#         ]) for i in range(2)]
#         plaintext_columns = ["\n".join([
#             self.runs[i]["software"], wrap(self.runs[i]["run_path"])
#         ]) for i in range(2)]
#         path_table = pd.DataFrame([[
#             "Not found" if not
#             self.data[sub][feat].paths[i] else (
#                 self._join_paths(self.data[sub][feat].paths, i)
#             ) for i in range(2)
#         ] for sub in self.data for feat in self.data[sub]],
#                                   columns=columns,
#                                   index=[f"{sub} {feat}" for sub in
#                                          self.subjects for feat in
#                                          self.features])
#         if plaintext:
#             plaintext_path_table = pd.DataFrame([[
#                 f"\u001b[3m\u001b[31mNot found\u001b[0m{' '*13}" if not
#                 self.data[sub][feat].paths[i] else wrap(
#                     self._join_paths(self.data[sub][feat].paths, i)
#                 ) for i in range(2)
#             ] for sub in self.data for feat in self.data[sub]],
#                                                columns=plaintext_columns,
#                                                index=[f"{sub} {feat}" for
#                                                       sub in self.subjects for
#                                                       feat in self.features])
#             print(tabulate(plaintext_path_table,
#                            headers=plaintext_columns))
#         else:
#             stored_options = (
#                 pd.options.display.max_rows,
#                 pd.options.display.max_colwidth
#             )
#             pd.options.display.max_rows = 999
#             pd.options.display.max_colwidth = 1000
#             try:
#                 from IPython.display import display
#                 display(path_table)
#             except ImportError:
#                 print(path_table)
#             (
#                 pd.options.display.max_rows,
#                 pd.options.display.max_colwidth
#             ) = stored_options
#             del stored_options
#         return path_table

#     def run_correlation(self, subject, feature, data1, data2):
#         """
#         A method to fill a cell in a correlation matrix with Pearson's r

#         Parameters
#         ----------
#         subject: int
#             subject index

#         feature: int
#             feature index

#         data1: np.ndarray or list

#         data2: np.ndarray or list
#         """
#         corr = calc_corr(data1, data2)
#         print(
#             f"Running subject: {subject} {feature} "
#             f"correlation score: {str(corr)}"
#         )
#         self.corrs[subject][feature] = round(corr, 3)

#     def run_pearsonsr(self):
#         for i, subject in enumerate(self.data):
#             for j, feature in enumerate(self.data[subject]):
#                 self.run_correlation(i, j, *self.data[subject][feature].data)

#     def _join_paths(self, data_paths, index):
#         return "\n".join([
#             data_path.replace(
#                 self.runs[index]["run_path"], "", 1
#             ) if data_path.startswith(
#                 self.runs[index]["run_path"]
#             ) else data_path for data_path in data_paths[index]])


# def wrap(string, at=25):
#     return '\n'.join([string[i:i+at] for i in range(0, len(string), at)])


if __name__ == "__main__":
    main()
