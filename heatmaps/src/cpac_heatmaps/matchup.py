'''Matchup information to compare

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
from traits.api import Undefined
from .directory import determine_software_and_root, entities_from_featurekey, \
                       feature_label_from_filename, separate_working_files
from .features import Feature, VariableValue
from .subjects import gather_unique_ids


class Matchup:  # pylint: disable=too-many-instance-attributes
    """A class to hold head-to-head information for pair of output directories

    Attributes
    ----------
    features : dict of {UniqueId: dict}
        subset of unique_ids with the greatest number of entities

    files : 2-length list of lists of str
        for each output, all found files

    most_specific_ids : list of UniqueId

    root : 2-tuple of str
        root paths to each output

    software : tuple of Software

    unique_ids : list of UniqueId
        intersection of ids for both outputs
    """
    def __init__(self, paths):
        """
        Parameters
        ----------
        paths : (str, str)
            paths to output directories to compare
        """
        self._bids_layout = [Undefined, Undefined]
        self.files = self.Filelist((Undefined, Undefined))
        self._ids = [Undefined, Undefined]
        self._root = [Undefined, Undefined]
        self._software = [Undefined, Undefined]
        for i in range(2):
            self._software[i], self._root[i] = determine_software_and_root(
                paths[i])
            self._ids[i], self._bids_layout[i] = gather_unique_ids(
                self.root[i])
            self.files[i] = list(self._bids_layout[i].get_files().keys())
            if hasattr(self.software[i],
                       'name') and self.software[i].name == 'C-PAC':
                self.files[i] = separate_working_files(self.files[i])
        # pylint: disable=no-member
        self.unique_ids = self._ids[0].intersection(self._ids[1])
        self.features = {id: {} for id in self.unique_ids if
                         len(id) == max(len(id) for id in self.unique_ids)}

    def __repr__(self):
        return f'{self.software[0]} vs. {self.software[1]}'

    class Filelist(list):
        """Subclass of list with extra 'flat' attribute

        Attributes
        ----------
        flat : list
        """
        @property
        def flat(self):
            """Flat list of output files"""
            if not isinstance(self, list) or len(self) != 2:
                raise TypeError('Something is unexpected about Filelist \n'
                                f'{self}')
            files = [[], []]
            for i in range(2):
                if isinstance(self[i], tuple):
                    for j in range(3):
                        if not files[i]:
                            files[i] = self[j]
                else:
                    files[i] = self[i]
            return files[0] + files[1]

    @property
    def ids(self):
        """2-tuple of lists of UniqueIds"""
        return tuple(self._ids)

    def iterate_files(self):
        """Iterate through features and find matching files.

        This method updates the 'files' attribute of each feature"""
        # pylint: disable=too-many-branches,too-many-nested-blocks
        for feature_dict in self.features.values():
            for feature in feature_dict.values():
                for i in range(2):
                    if (isinstance(self.files[i], tuple) and
                            len(self.files[i]) == 3):
                        for file in self.files[i][0]:
                            if feature.filepath_match_output(file,
                                                             self.software[i]):
                                feature.add_file(i, file)
                    else:
                        for file in self.files[i]:
                            if feature.filepath_match_output(file,
                                                             self.software[i]):
                                feature.add_file(i, file)

    @property
    def most_specific_ids(self):
        """List of UniqueIds"""
        return list(self.features.keys())

    @property
    def root(self):
        """2-tuple of root output directories"""
        return tuple(self._root)

    # pylint: disable=too-many-arguments
    def set_method(self, software, method, feature_keys=Undefined,
                   entities=None, endswith=None, filetype=Undefined):
        """
        For given software, set correlation method, optionally also
        setting feature_keys, entities, endswith, and/or filetype

        Parameters
        ----------
        software : str

        method : str

        feature_keys : list of str, or str, optional

        entities : dict, optional

        endswith : str, optional

        filetype : str, optional
        """
        if entities is None:
            entities = entities_from_featurekey(feature_keys) if (
                endswith is None and filetype is Undefined) else {}
        if endswith is None:
            endswith = ''
        if feature_keys is Undefined:
            matching_features = list({feature_label_from_filename(file) for
                                      file in [file for output in
                                      self.files.flat for file in output] if
                                      file.endswith(endswith) and
                                      all('-'.join(entity) in file for
                                      entity in entities.items())})
        else:
            matching_features = feature_keys if (
                isinstance(feature_keys, list)) else [feature_keys]
        for feature in matching_features:
            print(f'setting method {method} for {feature}')
            self.set_filename_matching(software, feature, entities,
                                       endswith, filetype)
            for feature_dict in self.features.values():
                feature_dict[feature].set_correlation_method(method)
        return matching_features

    def set_filename_matching(self, software, feature_key, entities=None,
                              endswith=None, filetype=Undefined):
        """
        For given software, set feature_key, optionally also setting
        entities, endswith, and/or filetype. If only feature_key is
        provided, entities is inferred from feature_key

        Parameters
        ----------
        software : str

        feature_key : str

        entities : dict, optional

        endswith : str, optional

        filetype : str, optional
        """
        if entities is None and endswith is None:
            entities = feature_label_from_filename(feature_key)
        if any(_ is VariableValue for _ in entities.values()):
            # Dynamically set variable values
            entities.update({_k: _v for _k, _v in
                             entities_from_featurekey(
                                feature_label_from_filename(feature_key)
                            ).items() if _k in entities and
                            entities[_k] is VariableValue})

        for feature_dict in self.features.values():
            if feature_key not in feature_dict:
                feature_dict[feature_key] = Feature(feature_key,
                                                    filetype=filetype)
            if entities is not None:
                feature_dict[feature_key].set_software_entities(software,
                                                                entities)
            if endswith is not None:
                feature_dict[feature_key].set_software_endswith(software,
                                                                endswith)

    @property
    def software(self):
        """2-tuple of software whose outputs we're comparing"""
        return tuple(self._software)
