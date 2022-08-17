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
                       feature_label_from_filename, filepath_match_entity
from .features import Feature, SOFTWARE
from .subjects import gather_unique_ids


class Features(dict):
    """Subclass of dict to hold features but add a method for iterating"""
    def iterate_features(self, file, specific_id, software, position):
        """Loop through features for a given filename, ID, feature,
        software, and position, returning the first match

        Parameters
        ----------
        features : list of str
            features to try to find

        file : str

        specific_id : UniqueId

        software : Software

        postion : int
            0 or 1

        Returns
        -------
        dict
        """
        if filepath_match_entity(file, specific_id):
            for _feature in self:
                if self.filepath_match_output(file, _feature, software):
                    if _feature not in self:
                        self[_feature] = [Undefined, Undefined]
                    else:
                        self[_feature][position] = file
                    return self
        return self

    def filepath_match_output(self, filepath, feature, software):
        """Check if a filepath matches configuration for a given feature
        and software

        Parameters
        ----------
        filepath : str

        feature : str

        software : Software

        Returns
        -------
        bool
        """
        if software.name not in SOFTWARE:
            return False
        software_features = self[feature].software[SOFTWARE[software.name]]
        if hasattr(software_features, 'endswith'):
            if not filepath.endswith(software_features.endswith):
                return False
        if hasattr(software_features, 'entities'):
            if not filepath_match_entity(filepath, software_features.entities):
                return False
        return True


class Matchup:  # pylint: disable=too-many-instance-attributes
    """A class to hold head-to-head information for pair of output directories

    Attributes
    ----------
    files : 2-tuple of list of str
        for each output, all found files

    ids : 2-tuple of list of 2-tuples
        for each output, list of strings of BIDS entities describing
        one or more outputs

    most_specific_ids : list of str
        subset of unique_ids with the greatest number of entities

    root : 2-tuple of str
        root paths to each output

    software : tuple of cpac_heatmaps.features.Software

    unique_ids : list of str
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
        self.files = [Undefined, Undefined]
        self._ids = [Undefined, Undefined]
        self._root = [Undefined, Undefined]
        self._software = [Undefined, Undefined]
        for i in range(2):
            self._software[i], self._root[i] = determine_software_and_root(
                paths[i])
            self._ids[i], self._bids_layout[i] = gather_unique_ids(
                self.root[i])
            self.files[i] = list(self._bids_layout[i].get_files().keys())
        # pylint: disable=no-member
        self.unique_ids = self._ids[0].intersection(self._ids[1])
        self.most_specific_ids = [id for id in self.unique_ids if
                                  len(id) == max(len(id) for id in
                                                 self.unique_ids)]
        self.features = {specific_id: Features() for specific_id in
                         self.most_specific_ids}

    @property
    def ids(self):
        """
        2-tuple of lists of cpac_heatmaps.subjects.UniqueID
        """
        return tuple(self._ids)

    def iterate_features(self):
        """
        Once feature keys are defined, run this method to iterate files
        looking for defined features.
        """
        for specific_id in self.most_specific_ids:
            for i in range(2):
                for file in self.files[i]:
                    self.features[specific_id].iterate_features(
                        file, specific_id, self.software[i], i)
            for label, feature in self.features.items():
                if Undefined in feature.software:
                    # drop undefined-on-one-side
                    del self.features[label]

    @property
    def root(self):
        """2-tuple of root output directories"""
        return tuple(self._root)

    def set_entities_for_software(self, software, feature_key, entities,
                                  endswith):
        """
        Set entities and endswith for one software for a feature that
        is already defined for another software.

        Parameters
        ----------
        software : str

        feature_key : str

        entities : dict

        endswith : str
            f'_{suffix}.{extension}'
        """


    def set_method_for_entities(self, software, entities, endswith, method,
                                filetype=Undefined):
        """Set correlation method for files with given entities and suffix.
        If a method was already set for matching features, this method
        will override.

        Parameters
        ----------
        software : str

        entities : dict

        endswith : str
            f'_{suffix}.{extension}'

        method : str

        filetype : str, optional
        """
        matching_features = list({feature_label_from_filename(file) for file in
                                  [file for output in self.files for file in
                                   output] if file.endswith(endswith) and
                                  all('-'.join(entity) in file for entity in
                                      entities.items())})
        for feature in matching_features:
            self.features[feature] = Feature(feature,
                                             correlation_method=method,
                                             filetype=filetype)
            self.features[feature].set_software_entities(
                SOFTWARE[software], entities_from_featurekey(feature))
            self.features[feature].set_software_endswith(SOFTWARE[software],
                                                         endswith)

    def set_method_for_suffix(self, software, endswith, method,
                              filetype=Undefined):
        """Set correlation method for files with given suffix.
        If a method was already set for matching features, this method
        will override.

        Parameters
        ----------
        software : str

        endswith : str
            f'_{suffix}.{extension}'

        method : str

        filetype : str, optional
        """
        matching_features = list({feature_label_from_filename(file) for file in
                                  [file for output in self.files for file in
                                   output] if file.endswith(endswith)})
        for feature in matching_features:
            self.features[feature] = Feature(feature,
                                             correlation_method=method,
                                             filetype=filetype)
            self.features[feature].set_software_entities(
                SOFTWARE[software], entities_from_featurekey(feature))
            self.features[feature].set_software_endswith(SOFTWARE[software],
                                                         endswith)

    @property
    def software(self):
        """2-tuple of software whose outputs we're comparing"""
        return tuple(self._software)
