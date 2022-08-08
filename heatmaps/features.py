'''Heatmap features

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
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import dice
from traits.api import Undefined


CORRELATION_METHODS = {
    'Dice': dice,
    'Pearson': pearsonr,
    'Spearman': spearmanr
}
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


class SoftwareFeature:
    """Class to store software information related to a feature"""
    # pylint: disable=too-few-public-methods
    def __init__(self, name=None):
        self.name = name
        self.entities = []
        self.regex = None
        self.endswith = None

    def __repr__(self):
        if self.name is None:
            return '<unnamed software feature>'
        return str(self.name)


class Feature:
    """Class to store features to compare"""
    def __init__(self, name, **kwargs):
        self.name = name
        self.link = kwargs.get('link', Undefined)
        self.software = {}
        self.correlation_method = {}

    def __repr__(self):
        return self.name

    def _check_self_software(self, software):
        if software not in self.software:
            self.software[software] = SoftwareFeature(': '.join([
                str(software), self.name]))

    def set_correlation_method(self, correlation_method):
        """Set feature's correlation method

        Parameters
        ----------
        correlation_method : str
            key in CORRELATION_METHODS
        """
        self.correlation_method['label'] = correlation_method
        self.correlation_method['function'] = CORRELATION_METHODS[
            correlation_method]

    def set_software_endswith(self, software, endswith):
        """Define required filename ending for a feature

        Parameters
        ----------
        software : Software

        endswith : str
        """
        self._check_self_software(software)
        self.software[software].endswith = endswith
        if endswith == '_bold.nii.gz':
            self.set_correlation_method('Pearson')
        if endswith == '_correlations.tsv':
            self.set_correlation_method('Spearman')
        if endswith == '_mask.nii.gz':
            self.set_correlation_method('Dice')

    def set_software_entities(self, software, entities):
        """Define entities to match for a feature

        Parameters
        ----------
        software : Software

        entities : dict of str
            key, value pairs of BIDS keys and regular expressions to
            match BIDS values
        """
        self._check_self_software(software)
        self.software[software].entities = entities

    def set_software_regex(self, software, regex):
        """Define regex for finding feature"""
        self._check_self_software(software)
        self.software[software].regex = regex


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
            raise TypeError("Software version must be string or Undefined")
        self.name = name
        self.version = f"v{version.lstrip('v')}" if isinstance(
            version, str) else version

    def __repr__(self):
        if isinstance(self.version, str):
            return " ".join([self.name, self.version])
        return f"{self.name} unknown version"


SOFTWARE = {key: Software(key) for key in
            ["C-PAC", "fMRIPrep", "XCP-D", "xcpEngine"]}

FEATURES = {key: Feature(key) for key in ["space-template_desc-preproc_bold"]}
FEATURES["space-template_desc-preproc_bold"].set_software_entities(
    SOFTWARE["C-PAC"], {'space': 'template', 'desc': 'preproc.*'})
for feature in ["space-template_desc-preproc_bold"]:
    FEATURES[feature].set_software_endswith(SOFTWARE["C-PAC"], '_bold.nii.gz')
