"""Determine subjects to compare

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
import numpy as np
from bids import BIDSLayout
# from itertools import chain
# from traits.api import Undefined

_attributes = {'subject': 'sub', 'session': 'ses', 'acquisition': 'acq',
               'run': 'run', 'task': 'task'}


def gather_unique_ids(path):
    '''Given a root directory, return a list of UniqueIDs

    Parameters
    ----------
    path : str

    Returns
    -------
    set of UniqueID'''
    layout = BIDSLayout(path, validate=False)
    layout_df = layout.to_df()
    unique_ids = set()
    for row in layout_df[
        [col for col in _attributes if col in layout_df.columns]
    ].drop_duplicates().iterrows():
        try:
            unique_ids.add(UniqueID(**row[1].to_dict()))
        except TypeError:
            pass
    return unique_ids


class UniqueID:
    '''A representation of BIDS information

    Comparisons operate first on number of BIDS entities, then on
    string comparisons'''
    def __init__(self, subject, **kwargs):
        '''
        Parameters
        ----------
        subject : str

        session, acquisition, task, run : str or int, optional'''
        if not isinstance(subject, str):
            raise TypeError('``subject`` must be a string.')
        self.subject = subject
        for attr in ['session', 'acquisition', 'task', 'run']:
            value = kwargs.get(attr)
            if value is not None and value is not np.nan:
                setattr(self, attr, value)

    def __eq__(self, other):
        return str(self) == str(other)

    def __ge__(self, other):
        if self.entity_count != other.entity_count:
            return self.entity_count >= other.entity_count
        return str(self) >= other.self

    def __gt__(self, other):
        if self.entity_count != other.entity_count:
            return self.entity_count > other.entity_count
        return str(self) > str(other)

    def __hash__(self):
        return hash(str(self))

    def __le__(self, other):
        if self.entity_count != other.entity_count:
            return self.entity_count <= other.entity_count
        return str(self) <= other.self

    def __lt__(self, other):
        if self.entity_count != other.entity_count:
            return self.entity_count < other.entity_count
        return str(self) < str(other)

    def __repr__(self):
        return ('_'.join(
            '-'.join([_attributes[attr], str(getattr(self, attr))]) for
            attr in self._get_own_attributes()))

    @property
    def entity_count(self):
        '''Return the number of BIDS entities in a given UniqueID'''
        return str(self).count('_') + 1

    def _get_own_attributes(self):
        return (attr for attr in _attributes if hasattr(self, attr))
