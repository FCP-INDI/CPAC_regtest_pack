<!-- Copyright (C) 2022  C-PAC Developers
This file is part of CPAC_regtest_pack.
CPAC_regtest_pack is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
CPAC_regtest_pack is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with CPAC_regtest_pack. If not, see <https://www.gnu.org/licenses/> -->
# CPAC_regtest_pack
## heatmaps

<!-- USAGE START -->
```bash
$ python3 correlation_matrix.py --help
usage: correlation_matrix.py [-h] [--subject_list [SUBJECT_LIST [SUBJECT_LIST ...]]] [--session SESSION] [--feature_list [FEATURE_LIST [FEATURE_LIST ...]]]
                             [--num_cores NUM_CORES]
                             outputs_path outputs_path run_name

Create a correlation matrix between two C-PAC output directories.

positional arguments:
  outputs_path          path to an outputs directory
  run_name              name for the correlations run

optional arguments:
  -h, --help            show this help message and exit
  --subject_list [SUBJECT_LIST [SUBJECT_LIST ...]]
  --session SESSION     limit to a single given session
  --feature_list [FEATURE_LIST [FEATURE_LIST ...]]
                        default: ['GS', 'CSF', 'WM', 'tCompCor0', 'aCompCor0', 'aCompCor1', 'aCompCor2', 'aCompCor3', 'aCompCor4', 'FD']
  --num_cores NUM_CORES
                        number of cores to use - will calculate correlations in parallel if greater than 1

The following features currently have available definitions to calculate Pearson's r between C-PAC and fmriprep:

key       feature name              documentation link
--------  ------------------------  ----------------------------------------------------------------------------------
aCompCor  aCompCor                  https://fcp-indi.github.io/docs/user/nuisance.html#acompcor
CSF       mean cerebrospinal fluid  https://fcp-indi.github.io/docs/user/nuisance.html#mean-white-matter-csf
FD        framewise displacement    https://fcp-indi.github.io/docs/user/nuisance.html#regression-of-motion-parameters
GS        global signal regression  https://fcp-indi.github.io/docs/user/nuisance.html#global-signal-regression
tCompCor  tCompCor                  https://fcp-indi.github.io/docs/user/nuisance.html#tcompcor
WM        mean white matter         https://fcp-indi.github.io/docs/user/nuisance.html#mean-white-matter-csf
```
<!-- USAGE END -->
