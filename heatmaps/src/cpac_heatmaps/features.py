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
<https://www.gnu.org/licenses/>
'''
import inspect
import os
import re
from logging import warning
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple, Union
import nibabel as nib
import numpy as np
from nipype.interfaces.afni import Resample as NipypeResample, TCorrelate
from nipype.interfaces.afni.base import AFNICommand
from nipype.interfaces.afni.utils import ResampleInputSpec
from nipype.interfaces.base import File, TraitedSpec
from nipype.pipeline import engine as pe
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import dice
from traits.api import Undefined
from .subjects import ATTRIBUTES, UniqueId

UndefinedType = type(Undefined)


class _VariableValue(str):
    """
    Singleton to identify a value that should be dynamically defined
    for a given key
    """
    def __repr__(self):
        return "«Variable Value»"

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __ne__(self, other):
        return type(self) is not type(other)


VariableValue = _VariableValue()  # pylint: disable=invalid-name


class AFNICommandOutputSpec(TraitedSpec):
    """
    Copyright (c) 2009-2016, Nipype developers

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Prior to release 0.12, Nipype was licensed under a BSD license.

    Modified to allow output file to not exist
    """
    out_file = File(desc="output file", exists=False)


class CalculateCorrelation:
    """Class to check/adjust data shapes and calculate correlations

    Attributes
    ----------
    basenames : 2-tuple of str

    paths : 2-tuple of str

    working_directory : str
    """
    def __init__(self, *args, **kwargs):
        """Not intended to be initialized directly. Use
        ``CalculateCorrelationBetween`` or another specific subclass to
        initialize"""
        if args or kwargs:
            raise NotImplementedError(
                "The base class ``CalculateCorrelation`` is not intended to "
                "be initialized directly. Use "
                "``CalculateCorrelationBetween`` to initialize a new "
                "``CalculateCorrelation`` object.")
        self._loaded = None

    @property
    def basenames(self):
        """Memoized basenames"""
        return self._loaded.basenames

    def cleanup(self):
        """Remove temporary working directory"""
        rmtree(self._loaded.working_directory)

    @property
    def paths(self):
        """Paths to images"""
        return tuple(self._loaded.paths)

    @property
    def working_directory(self):
        """Temporary working directory"""
        return self._loaded.working_directory


class CalculateCorrelationBetween(CalculateCorrelation):
    """Class to initialze an object to check/adjust data shapes and
    calculate correlation between pairs of files"""
    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def __new__(cls, path1, path2, filetype=None):
        """
        Parameters
        ----------
        path1, path2 : str
            path to data

        filetype : str, optional
            'NIfTI' or 'matrix'
        """
        if filetype is None:
            if any(path.endswith(".nii.gz") for path in [path1, path2]):
                filetype = "nifti"
            else:
                filetype = Undefined
        if isinstance(filetype, str):
            if filetype.lower() == "nifti":
                return CalculateImageCorrelation(path1, path2)
            if filetype.lower() == "matrix":
                return CalculateMatrixCorrelation(path1, path2)
        raise TypeError("Could not determine type of files. Please provide a "
                        "filetype or try another pair of files")

    __init__.__signature__ = inspect.Signature([
        v for k, v in inspect.signature(__new__).parameters.items() if
        k != 'cls'])
    __init__.__doc__ = __new__.__doc__


class CalculateImageCorrelation(CalculateCorrelation):
    """Class to check/adjust image shapes and calculate correlations"""
    def __init__(self, path1, path2):
        """
        Parameters
        ----------
        path1, path2 : str
            path to data
        """
        super().__init__()
        self._loaded = _Images(path1, path2)

    def check_and_adjust_nifti_shapes(self):
        """If differing spatial resolution, reduce higher res to match
        lower res.
        If differing time duration, drop timepoints from beginning of
        longer timeseries.
        """
        if self.spaces[0] > self.spaces[1]:
            self.resample_space(0, 1)
        elif self.spaces[1] > self.spaces[0]:
            self.resample_space(1, 0)
        if len(self.images[0].shape) > 3:
            if self.images[0].shape[3] > self.images[1].shape[3]:
                self.truncate(0, 1)
            elif self.images[1].shape[3] > self.images[0].shape[3]:
                self.truncate(1, 0)

    @property
    def images(self):
        """Loaded images (via nibabel)"""
        return tuple(self._loaded.images)

    def resample_space(self, i, j):
        """Resample self.images[i] to self.images[j] space

        Parameters
        ----------
        i : int
            index of image to resample

        j : int
            index of target image
        """
        resample = pe.Node(interface=Resample(),
                           name=f'resampleImage{i}',
                           base_dir=self.working_directory)
        resample.inputs.master = self.paths[j]
        resample.inputs.outputtype = 'NIFTI_GZ'
        resample.inputs.resample_mode = 'Cu'
        resample.inputs.in_file = self.paths[i]
        self._loaded[i] = resample.run().outputs.out_file

    @property
    def spaces(self):
        """Spatial dimensions of images"""
        return tuple(image.shape[:3] for image in self.images)

    def truncate(self, i, j):
        """Truncate self.images[i] to self.images[j] length

        Parameters
        ----------
        i : int
            index of image to resample

        j : int
            index of target image
        """
        self._loaded[i] = self.images[i].slicer[...,
                                                (self.images[i].shape[3] -
                                                 self.images[j].shape[3]):]


class CalculateMatrixCorrelation(CalculateCorrelation):
    """Class to check/adjust image shapes and calculate correlations"""
    def __init__(self, path1, path2):
        """
        Parameters
        ----------
        path1, path2 : str
            path to data
        """
        super().__init__()
        self._loaded = _Matrices(path1, path2)

    @property
    def matrices(self):
        """Loaded matrices (via numpy)"""
        return tuple(self._loaded.matrices)


class CorrelationCoefficient(float):
    """A subclass of float to hold extra correlation-coefficient-
    specific attributes

    Parameters
    ----------
    x : float_like
        correlation coefficient

    correlation_type : str
        Pearson, Spearman, etc.

    correlation_series : array_like
        point-wise correlation coefficients
    """
    # pylint: disable=unused-argument
    def __new__(cls, x, correlation_type, correlation_series=None):
        return float.__new__(cls, x)

    def __init__(self, x, correlation_type, correlation_series=None):
        float.__init__(x)
        self.correlation_series = [
        ] if correlation_series is None else correlation_series
        self.correlation_type = correlation_type


class CorrelationMethod:
    """Methods to correlate pairs of files

    Attributes
    ----------
    calculate_correlation : CalculateCorrelation

    correlation_method : str
    """
    def __init__(self) -> None:
        self.calculate_correlation = Undefined
        self.correlation_method = Undefined
        self._run = Undefined

    def __repr__(self):
        return self.correlation_method

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value
        if __name == 'correlation_method' and __value is not Undefined:
            correlation_method = __value.lower()
            if hasattr(self, correlation_method):
                self._run = getattr(self, correlation_method)

    def cleanup(self):
        """Convenience method to clean up"""
        self.calculate_correlation.cleanup()

    def dice(self) -> float:
        """Dice coefficient

        Returns
        -------
        dice_coefficient : float
        """
        files = [image.get_fdata(caching="unchanged").ravel() for image in
                 self.calculate_correlation.images]
        for image in self.calculate_correlation.images:
            image.uncache()
        value = dice(*files)
        del files
        return value

    def identity(self) -> float:
        """Return 1.

        Returns
        -------
        float
        """
        # pylint: disable=unused-argument
        self.correlation_method = 'identity'
        return 1.

    def pearson(self):
        """Pearson correlation

        Returns
        -------
        pearson_r : CorrelationCoefficient
        """
        self.correlation_method = 'Pearson'
        # pylint: disable=no-member
        pearson = pearsonr(*[image.get_fdata(caching="unchanged").ravel() for
                             image in self.calculate_correlation.images])
        value = CorrelationCoefficient(pearson[0] if
                                       isinstance(pearson, tuple) else
                                       pearson.statistic, 'Pearson')
        del pearson
        return value

    def pearson_3dcorrelate(self) -> Tuple[float, float, float]:
        """Pearson correlation via 3dcorrelate

        Returns
        -------
        CorrelationCoefficient
        """
        self.correlation_method = 'Pearson_3dTcorrelate'
        tcorrelate = pe.Node(
            TCorrelate(), '3dTcorrelate',
            base_dir=self.calculate_correlation.working_directory)
        tcorrelate.inputs.xset, tcorrelate.inputs.yset = (
            self.calculate_correlation.paths)
        tcorrelate.inputs.out_file = os.path.join(
            self.calculate_correlation.working_directory,
            'functional_tcorrelate.nii.gz')
        tcorrelate.inputs.pearson = True
        correlation_image = nib.load(tcorrelate.run().outputs.out_file)
        non_zeroes = [point for point in
                      correlation_image.get_fdata(caching="unchanged").ravel()
                      if point != 0]
        value = CorrelationCoefficient(np.mean(non_zeroes), 'mean(Pearson)',
                                       non_zeroes)
        del correlation_image, non_zeroes, tcorrelate
        return value

    def pearson_3dtcorrelate(self) -> Tuple[float, float, float]:
        """Pearson correlation via 3dTcorrelate

        Returns
        -------
        CorrelationCoefficient
        """
        self.correlation_method = 'Pearson_3dTcorrelate'
        tcorrelate = pe.Node(
            TCorrelate(), '3dTcorrelate',
            base_dir=self.calculate_correlation.working_directory)
        tcorrelate.inputs.xset, tcorrelate.inputs.yset = (
            self.calculate_correlation.paths)
        tcorrelate.inputs.out_file = os.path.join(
            self.calculate_correlation.working_directory,
            'functional_tcorrelate.nii.gz')
        tcorrelate.inputs.pearson = True
        correlation_image = nib.load(tcorrelate.run().outputs.out_file)
        non_zeroes = [point for point in
                      correlation_image.get_fdata(caching="unchanged").ravel()
                      if point != 0]
        value = CorrelationCoefficient(np.mean(non_zeroes), 'mean(Pearson)',
                                       non_zeroes)
        del correlation_image, non_zeroes, tcorrelate
        return value

    def run(self, calculate_correlation):
        """Calculate correlation coefficient

        Parameters
        ----------
        calculate_correlation : CalculateCorrelation

        Returns
        -------
        CorrelationCoefficient
        """
        # pylint: disable=broad-except,lost-exception
        self.calculate_correlation = calculate_correlation
        coefficient = CorrelationCoefficient(np.nan, self.correlation_method)
        try:
            coefficient = self._run()
        except Exception as exception:
            warning(str(exception))
        finally:
            self.cleanup()
            return coefficient

    def spearman(self):
        """Spearman correlation

        Returns
        -------
        spearman_r : CorrelationCoefficient
        """
        if (self.calculate_correlation.matrices[0].shape ==
                self.calculate_correlation.matrices[2].shape):
            return CorrelationCoefficient(
                spearmanr(*(matrix.ravel() for matrix in
                            self.calculate_correlation.matrices)).correlation,
                'Spearman')
        return CorrelationCoefficient(np.nan, 'Spearman')


def _forgiving_load(path, fxn, *args, **kwargs):
    """Try to load a file; return None if fail"""
    try:
        return fxn(path, *args, **kwargs)
    except OSError as os_error:
        warning(os_error)
        return None


class _LoadedPaths:  # pylint: disable=too-few-public-methods
    def __init__(self, path1, path2):
        self._basenames = None
        self.paths = [path1, path2]
        # pylint: disable=consider-using-with
        self.working_directory = TemporaryDirectory().name
        os.makedirs(self.working_directory)

    @property
    def basenames(self):
        """Memoized basenames"""
        if self._basenames is None:
            self._basenames = tuple(os.path.basename(path) for path in
                                    self.paths)
        return self._basenames


class _Images(_LoadedPaths):
    def __init__(self, nifti1, nifti2):
        super().__init__(nifti1, nifti2)
        self.images = [img for img in [
            _forgiving_load(nifti, nib.load) for nifti in [nifti1, nifti2]
        ] if img is not None]

    def __setitem__(self, key, value):
        if isinstance(value, str):
            img = _forgiving_load(value, nib.load)
            if img is not None:
                self.images[key] = nib.load(value)
                self.paths[key] = value
        elif isinstance(value, nib.nifti1.Nifti1Image):
            self.images[key] = value
            basename, ext = splitext(self.basenames[key])
            self.paths[key] = os.path.join(self.working_directory,
                                           f'{basename}_modified.{ext}')
            nib.save(value, self.paths[key])


class _Matrices(_LoadedPaths):
    def __init__(self, path1, path2):
        super().__init__(path1, path2)
        self.matrices = [matrix for matrix in [
            _forgiving_load(path, load_matrix_array) for path in [
                path1, path2]] if matrix is not None]

    def __setitem__(self, key, value):
        if isinstance(value, str):
            matrix = _forgiving_load(value, load_matrix_array)
            if matrix is not None:
                self.matrices[key] = matrix
                self.paths[key] = value
        elif isinstance(value, np.ndarray):
            self.matrices[key] = value
            basename, ext = splitext(self.basenames[key])
            self.paths[key] = os.path.join(self.working_directory,
                                           f'{basename}_modified.{ext}')
            np.savetxt(self.paths[key], value,
                       delimiter=',' if ext == 'csv' else '\t')


class Software:
    """Class to store software name and version

    Attributes
    ----------
    config : str
        name of the pipeline config

    name : str
        name of software package

    version : str
        version of software package
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, name: str,
                 version: Optional[Union[str, UndefinedType]] = Undefined,
                 config: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        name : str

        version : str, Undefined, optional

        config : str, optional
           name of pipeline config
        """
        if version is not None and not isinstance(version,
                                                  (str, UndefinedType)):
            raise TypeError("Software version must be string or Undefined")
        self.config = config if config is not None else Undefined
        self.name = name
        self.version = f"v{version.lstrip('v')}" if isinstance(
            version, str) else version

    def __eq__(self, other):
        try:
            return self.name == other.name and self.version == other.version
        except AttributeError:
            return str(self) == str(other)

    def __hash__(self):
        return hash((self.name, self.version))

    def __repr__(self) -> None:
        if isinstance(self.version, str):
            _repr = " ".join([self.name, self.version])
        else:
            _repr = f"{self.name} unknown version"
        if self.config is not Undefined:
            return f"{_repr}: {self.config}"
        return _repr


class SoftwareFeature:
    """Class to store software information related to a feature

    Attributes
    ----------
    endswith : str
        filename ending of features for given software

    entities : list of dicts of strings
        features to match for given software and feature

    name : str
        name of feature

    regex : str
        pattern to look for for given software and feature
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, name: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        name : str, optional
        """
        self.name = name
        self.entities = []
        self.regex = None
        self.endswith = None

    def __repr__(self) -> None:
        if self.name is None:
            return '<unnamed software feature>'
        return str(self.name)


class Feature:
    """Class to store features to compare

    Attributes
    ----------
    basenames : 2-tuple of str

    correlation_method : CorrelationMethod

    files : list of 2-tuples of strings
        pairs of files to compare

    filetype : str

    link : str

    name : str

    softwarefeature : dict of {Software: SoftwareFeature}
    """
    def __init__(self, name: str, **kwargs) -> None:
        """
        Parameters
        ----------
        name : str

        correlation_method, filetype, link : str, optional
        """
        self.correlation_method = Undefined
        self._files = ([], [])  # each list is all matching files for one run
        self.filetype = kwargs.get('filetype', Undefined)
        self.link = kwargs.get('link', Undefined)
        self.name = name
        self.softwarefeature = {}
        if 'correlation_method' in kwargs:
            self.set_correlation_method(kwargs['correlation_method'])

    def __repr__(self) -> None:
        return self.name

    def add_file(self, position, file):
        """Add a file to list of matching files

        Parameters
        ----------
        position : int
           0 or 1

        file : str
           filepath
        """
        if position == 0 and file not in self._files[0]:
            self._files = (self._files[0] + [file], self._files[1])
        elif position == 1 and file not in self._files[1]:
            self._files = (self._files[0], self._files[1] + [file])

    @property
    def basenames(self):
        """Memoized basenames"""
        try:
            return self.correlation_method.calculate_correlation.basenames
        except AttributeError:
            return (Undefined, Undefined)

    def _check_software(self, software: 'Software') -> None:
        if software not in self.softwarefeature:
            self.softwarefeature[software] = SoftwareFeature(self.name)

    def filepath_match_output(self, filepath, software):
        """Check if a filepath matches configuration for a given feature
        and software

        Parameters
        ----------
        filepath : str

        software : Software

        Returns
        -------
        bool
        """
        try:
            software_features = self.softwarefeature[software]
        except KeyError:
            return False
        if hasattr(software_features, 'endswith'):
            if not filepath.endswith(software_features.endswith):
                return False
        if hasattr(software_features, 'entities'):
            if not filepath_match_entity(filepath, software_features.entities):
                return False
        return True

    @property
    def files(self):
        """list of 2-tuples of strings (pairs of files to compare)"""
        return [(_0, _1) for _0 in self._files[0] for _1 in self._files[1]]

    def set_correlation_method(self, correlation_method: str) -> None:
        """Set feature's correlation method in-place

        Parameters
        ----------
        correlation_method : str
        """
        self.correlation_method = CorrelationMethod()
        self.correlation_method.correlation_method = correlation_method

    def set_software_endswith(self, software: 'Software', endswith: str
                              ) -> None:
        """Define required filename ending for a feature and associated
        correlation method

        Parameters
        ----------
        software : Software

        endswith : str
        """
        self._check_software(software)
        self.softwarefeature[software].endswith = endswith

    def set_software_entities(self, software: 'Software',
                              entities: Dict[str, str]) -> None:
        """Define BIDS entities to match for a feature for the given software

        Parameters
        ----------
        software : Software

        entities : dict of str
            key, value pairs of BIDS keys and regular expressions to
            match BIDS values
            """
        self._check_software(software)
        self.softwarefeature[software].entities = entities

    def set_software_regex(self, software: 'Software', regex: str) -> None:
        """Define regex for finding feature

        Parameters
        ----------
        software : Software

        regex : str
        """
        self._check_software(software)
        self.softwarefeature[software].regex = regex


class Features(dict):
    """
    """


class Resample(AFNICommand):
    # pylint: disable=abstract-method
    _cmd = "3dresample"
    input_spec = ResampleInputSpec
    output_spec = AFNICommandOutputSpec


Resample.__doc__ = """
    Copyright (c) 2009-2016, Nipype developers

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Prior to release 0.12, Nipype was licensed under a BSD license.

    modified from https://github.com/nipy/nipype/blob/4a91e887/nipype/interfaces/afni/utils.py#L2632-L2653
    to not require the output file to exist.

    """ + NipypeResample.__doc__


def filepath_match_entity(filepath, key, value=None):
    '''Does this filepath match the given entity?

    Parameters
    ----------
    filepath : str

    key : str, UniqueId, dict, or iterable

    value : str, optional
        if numeric, tries to match integer value
        can include wildcards '*' or '?'
        can be negated by prefixing with '!'

    Returns
    -------
    bool
    '''
    # pylint: disable=too-many-branches,too-many-return-statements
    if filepath is Undefined:
        return False
    basename = os.path.basename(filepath)
    if isinstance(key, UniqueId):
        return all(
            filepath_match_entity(filepath, ATTRIBUTES[attribute],
                                  getattr(key, attribute)) for
            attribute in key.get_own_attributes())
    if isinstance(key, (list, tuple, set)):
        if isinstance(key, set):
            key = list(key)
        return all(filepath_match_entity(filepath, attribute) for
                   attribute in key)
    if isinstance(key, dict):
        return all(
            filepath_match_entity(filepath, d_key, d_value) for
            d_key, d_value in key.items())
    if isinstance(key, str):
        # ------------------------------------------------------------
        # If key starts with '!', return not-match
        # If both key and value start with '!', the double-negative
        # will cancel out
        if key.startswith('!'):
            return not filepath_match_entity(filepath, key[1:], value)
        f_key = f'{key}-'
        if f_key not in basename:
            if value.startswith('!'):
                return True
            return False
        # ------------------------------------------------------------
        check_value = basename.split(f_key, 1)[-1].split('_')[0]
        check_value = strip_int_suffix(check_value)
        value = strip_int_suffix(value)
        if isinstance(value, (str, int, float)):
            if isinstance(value, str):
                if value.startswith('!'):
                    # return bool(not-match)
                    return not filepath_match_entity(filepath, key, value[1:])
                # convert wildcards to regex
                value = value.replace('*', '.*').replace('?', '.?')
            if str(value).isnumeric():
                try:
                    return int(check_value) == int(value)
                except ValueError:
                    return check_value == value
            return bool(re.match(str(value), str(check_value)))
    return True


def load_matrix_array(filepath):
    """Load a TSV or CSV of a square matrix

    Parameters
    ----------
    filepath : str

    Returns
    -------
    np.ndarray
    """
    if filepath.endswith('.tsv'):
        matrix_array = _load_matrix_array(filepath, '\t,')
    elif filepath.endswith('.csv'):
        matrix_array = _load_matrix_array(filepath, ',\t')
    else:
        matrix_array = np.loadtxt(filepath)
    return matrix_array


def reconstruct_matrix_from_vector(vector, side_length=None):
    """
    Examples
    --------
    >>> reconstruct_matrix_from_vector([1])
    array([[1., 1.],
           [1., 1.]])
    >>> reconstruct_matrix_from_vector([1, 2])
    Traceback (most recent call last):
    ValueError: vector seems not to be a valid length
    >>> reconstruct_matrix_from_vector([1, 2, 3])
    array([[1., 1., 2.],
           [1., 1., 3.],
           [2., 3., 1.]])
    >>> reconstruct_matrix_from_vector([1, 2, 3, 4])
    Traceback (most recent call last):
    ValueError: vector seems not to be a valid length
    >>> reconstruct_matrix_from_vector([1, 2, 3, 4, 5])
    Traceback (most recent call last):
    ValueError: vector seems not to be a valid length
    >>> reconstruct_matrix_from_vector([1, 2, 3, 4, 5, 6])
    array([[1., 1., 2., 3.],
           [1., 1., 4., 5.],
           [2., 4., 1., 6.],
           [3., 5., 6., 1.]])
    """
    invalid_length_msg = ''
    if side_length is None:
        # try to figure out side length to create a square matrix from
        # this vector
        i = len(vector)
        side_length = (.5) + abs((.5 - (2 * i))) ** 0.5
    if side_length > 2 and side_length - side_length // 1 < 0.8:
        raise ValueError(invalid_length_msg)
    side_length = round(side_length)
    vector = list(vector)[::-1]
    matrix = np.ndarray((side_length,) * 2)  # square
    for i in range(side_length):
        for j in range(side_length):
            if i == j:
                matrix[i][j] = 1
            elif j > i:
                try:
                    matrix[i][j] = vector.pop()
                except IndexError as index_error:
                    raise ValueError(invalid_length_msg) from index_error
                matrix[j][i] = matrix[i][j]
    return matrix


def splitext(filename):
    """Customization of os.path.splitext to consider small combined
    extension sequences like '.nii.gz'

    Parameters
    ----------
    filename : str

    Returns
    -------
    basename, ext : str

    Examples
    --------
    >>> import os
    >>> os.path.splitext('file.yml')
    'file', 'yml'
    >>> splitext('file.yml')
    'file', 'yml'
    >>> os.path.splitext('file.nii.gz')
    'file.nii', 'gz'
    >>> splitext('file.nii.gz')
    'file', 'nii.gz'
    """
    ignore_area, extension_area = filename[:-8], filename[-8:]
    extension_area, ext = (extension_area.split('.', 1) if
                           '.' in extension_area else (extension_area, ''))
    return ''.join([ignore_area, extension_area]), ext


def strip_int_suffix(label):
    """Drop integer suffixes from C-PAC fork descriptions

    Examples
    --------
    >>> strip_int_suffix('desc-mean-1_bold.nii.gz')
    'desc-mean_bold.nii.gz'
    >>> strip_int_suffix({'desc': 'mean-1'})
    {'desc': 'mean'}
    """
    if isinstance(label, dict):
        return {key: (value.rstrip('-0123456789') if
                      key == 'desc' and isinstance(value, str) and
                      '-' in value else value) for key, value in label.items()}
    if isinstance(label, str):
        parts = label.split('_')
        index = [i for i, part in enumerate(parts) if part.startswith('desc-')]
        if index:
            index = index[0]
            parts[index] = '-'.join(
                list(strip_int_suffix(dict(
                    [parts[index].split('-', 1)])).items())[0])
        return '_'.join(parts)
    return label


def _load_matrix_array(filepath, delimiters=None):
    # pylint: disable=raise-missing-from
    """
    filepath : str

    delimiters : 2-length iterable of 1-length str, optional

    Returns
    -------
    matrix_array : np.array
    """
    non_iterable_delimiter = TypeError('_load_matrix_array "delimiters" '
                                       'argument must be a 2-length iterable')
    if not isinstance(delimiters, list):
        try:
            delimiters = list(delimiters)
        except TypeError:
            raise non_iterable_delimiter
        if len(delimiters) != 2:
            raise non_iterable_delimiter
    try:
        matrix_array = np.loadtxt(filepath, delimiter=delimiters.pop(0))
    except ValueError:
        matrix_array = np.loadtxt(filepath, delimiter=delimiters.pop(0))
    if (len(matrix_array.shape) != 2 and
            matrix_array.shape[0] != matrix_array.shape[1]):
        try:
            matrix_array = reconstruct_matrix_from_vector(matrix_array)
        except ValueError:
            raise ValueError(f'File "{filepath}" does not seem to represent '
                             'a square matrix')
    if all(matrix_array[i, i] == 0 for i in range(matrix_array.shape[0])):
        # replace all-0 diagonal with all-1 diagonal
        matrix_array = matrix_array + np.eye(matrix_array.shape[0])
    return matrix_array
