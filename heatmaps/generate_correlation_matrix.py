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
import os
import numpy as np

from correlation_directory import determine_software_and_root, \
                                  entities_from_featurekey, \
                                  feature_label_from_filename
from correlation_features import CalculateCorrelationBetween, Feature, \
                                 Features, FEATURES, SOFTWARE
from correlation_heatmaps import generate_heatmap
from correlation_subjects import gather_unique_ids

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
    correlation_matrices = list({feature_label_from_filename(file) for file in
                                 list(a_files.keys()) + list(b_files.keys())
                                 if file.endswith('_correlations.tsv')})
    for feature in correlation_matrices:
        FEATURES[feature] = Feature(feature)
        FEATURES[feature].set_software_entities(
            SOFTWARE["C-PAC"], entities_from_featurekey(feature))
        FEATURES[feature].set_software_endswith(SOFTWARE["C-PAC"],
                                                '_correlations.tsv')
    features = {specific_id: Features() for specific_id in most_specific_ids}
    for specific_id in most_specific_ids:
        for file in a_files:
            features[specific_id].iterate_features(FEATURES, file, specific_id,
                                                   a_software, 'a')
        for file in b_files:
            features[specific_id].iterate_features(FEATURES, file, specific_id,
                                                   b_software, 'b')
    corr_data = []
    var_set = set()
    for feature_dict in features.values():
        sub_data = []
        for label, feature in feature_dict.items():
            if 'a' in feature and 'b' in feature:
                var_set.add(label)
                if label.endswith('_correlations'):
                    corr = FEATURES[label].correlation_method.run(
                        CalculateCorrelationBetween(feature['a'],
                                                    feature['b'],
                                                    filetype='matrix'))
                else:
                    corr = FEATURES[label].correlation_method.run(
                        CalculateCorrelationBetween(feature['a'],
                                                    feature['b']))
                corr_coeff = corr if isinstance(corr, float) else corr[0]
                sub_data.append(corr_coeff)
        corr_data.append(sub_data)
        output_dir = os.path.join(
            os.getcwd(), f"correlations_{args.run_name}"
        )
        os.makedirs(output_dir, exist_ok=True)
    if corr_data and var_set and most_specific_ids:
        generate_heatmap(np.array(corr_data).T, list(var_set),
                         most_specific_ids,
                         save_path=os.path.join(output_dir, "heatmap.svg"))


if __name__ == "__main__":
    main()
