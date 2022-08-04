'''Heatmap defaults

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
<https://www.gnu.org/licenses/>'''
from traits.api import Undefined

feature_headers = {
    'GS': {
        'name': 'global signal regression',
        'link': 'https://fcp-indi.github.io/docs/user/nuisance.html#'
                'global-signal-regression',
        'C-PAC': ['GlobalSignalMean0', 'GlobalSignal_mean'],
        'fmriprep': 'global_signal'
    },
    'CSF': {
        'name': 'mean cerebrospinal fluid',
        'link': 'https://fcp-indi.github.io/docs/user/nuisance.html#'
                'mean-white-matter-csf',
        'C-PAC': ['CerebrospinalFluidMean0', 'CerebrospinalFluid_mean'],
        'fmriprep': 'csf'
    },
    'WM': {
        'name': 'mean white matter',
        'link': 'https://fcp-indi.github.io/docs/user/nuisance.html#'
                'mean-white-matter-csf',
        'C-PAC': ['WhiteMatterMean0', 'WhiteMatter_mean'],
        'fmriprep': 'white_matter'
    },
    'aCompCor': {
        'name': 'aCompCor',
        'link': 'https://fcp-indi.github.io/docs/user/nuisance.html#acompcor',
        'C-PAC': ['aCompCorPC', 'aCompCor'],
        'fmriprep': 'aCompCor_comp_cor_0'
    },
    'tCompCor': {
        'name': 'tCompCor',
        'link': 'https://fcp-indi.github.io/docs/user/nuisance.html#tcompcor',
        'C-PAC': ['tCompCorPC', 'tCompCor'],
        'fmriprep': 'tCompCor_comp_cor_0'
    },
    'FD': {
        'name': 'framewise displacement',
        'link': 'https://fcp-indi.github.io/docs/user/nuisance.html#'
                'regression-of-motion-parameters'
    }
}
motion_list = ['FD']
regressor_list = [
    'GS',
    'CSF',
    'WM',
    'tCompCor0',
    'aCompCor0',
    'aCompCor1',
    'aCompCor2',
    'aCompCor3',
    'aCompCor4'
]


class Software:
    """Class to store software name and version"""
    # pylint: disable=too-few-public-methods
    def __init__(self, name, version=Undefined):
        """
        Parameters
        ----------
        name : str

        version : str, Undefined or None
        """
        if not isinstance(version, (str, type(Undefined))):
            raise TypeError('Software version must be string or Undefined')
        self.name = name
        self.version = f'v{version.lstrip("v")}' if isinstance(
            version, str) else version

    def __repr__(self):
        if isinstance(self.version, str):
            return ' '.join([self.name, self.version])
        return f'{self.name} unknown version'


software = [Software("C-PAC"), Software("fMRIPrep"), Software("XCP-D"),
            Software("xcpEngine")]
