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
from traits.api import Undefined
from .features import CalculateCorrelationBetween
from .heatmaps import generate_heatmap
from .matchup import Matchup


def main():
    """Main function to run when called from commandline"""
    parser = argparse.ArgumentParser(
        description="Create a correlation matrix between two C-PAC output "
                    "directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("outputs_path", nargs=2, type=str,
                        help="path to an outputs directory")

    parser.add_argument("--num_cores", type=int, default=4,
                        help="number of cores to use - will calculate "
                             "correlations in parallel if greater than 1")
    parser.add_argument("run_name", type=str,
                        help="name for the correlations run")
    args = parser.parse_args()
    matchup = Matchup(args.outputs_path)
    matchup.set_method_for_suffix("C-PAC", "_bold.nii.gz",
                                  "Pearson_3dTcorrelate")
    matchup.set_entities_for_software("XCPD", "space-template_"
                                      "desc-preproc_bold.nii.gz",
                                      {"space": "*"}, "_bold.nii.gz")
    matchup.set_method_for_entities("C-PAC", {"desc": "mean"}, "_bold.nii.gz",
                                    "Pearson")
    matchup.set_method_for_suffix("C-PAC", "_correlations.tsv", "Spearman",
                                  "matrix")
    matchup.iterate_features()

    corr_data = []
    var_list = []
    for label, feature in matchup.features.items():
        if feature is not Undefined:
            for filepair in feature.files:
                if str(label).endswith('_correlations'):
                    corr = feature.correlation_method.run(
                        CalculateCorrelationBetween(*filepair,
                                                    filetype='matrix'))
                else:
                    corr = feature.correlation_method.run(
                        CalculateCorrelationBetween(*filepair))
                corr_coeff = corr if isinstance(corr, float) else corr[0]
                if corr_coeff is not np.nan:
                    if feature.correlation_method.basenames[
                        0
                    ] == feature.correlation_method.basenames[1]:
                        var_list.append(label)
                    else:
                        var_list.append('\n'.join(
                            feature.correation_method.basenames))
                    corr_data.append(corr_coeff)
        output_dir = os.path.join(
            os.getcwd(), f"correlations_{args.run_name}"
        )
        os.makedirs(output_dir, exist_ok=True)
    if corr_data and var_list and matchup.most_specific_ids:
        for imagetype in ["png", "svg"]:
            # pylint: disable=consider-using-f-string
            print(corr_data)
            print(var_list)
            print(matchup.most_specific_ids)
            generate_heatmap(np.array(corr_data).T, list(var_list),
                             matchup.most_specific_ids,
                             save_path=os.path.join(output_dir,
                                                    f"heatmap.{imagetype}"),
                             title='{} vs. {}'.format(*matchup.software))


if __name__ == "__main__":
    main()
