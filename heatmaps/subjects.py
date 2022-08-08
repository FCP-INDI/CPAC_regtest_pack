'''Determine subjects to compare

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
import numpy as np
from bids import BIDSLayout

ATTRIBUTES = {'subject': 'sub', 'session': 'ses', 'acquisition': 'acq',
              'run': 'run', 'task': 'task'}


def gather_unique_ids(path):
    '''Given a root directory, return a list of UniqueIds and
    a BIDSLayout

    Parameters
    ----------
    path : str

    Returns
    -------
    set of UniqueId

    BIDSLayout
    '''
    layout = BIDSLayout(path, validate=False)
    layout_df = layout.to_df()
    unique_ids = set()
    for row in layout_df[
        [col for col in ATTRIBUTES if col in layout_df.columns]
    ].drop_duplicates().iterrows():
        try:
            unique_ids.add(UniqueId(**row[1].to_dict()))
        except TypeError:
            pass
    return unique_ids, layout


class UniqueId:
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

    def __len__(self):
        '''Number of entites in UniqueId'''
        return self.entity_count

    def __lt__(self, other):
        if self.entity_count != other.entity_count:
            return self.entity_count < other.entity_count
        return str(self) < str(other)

    def __repr__(self):
        return ('_'.join(
            '-'.join([ATTRIBUTES[attr], str(getattr(self, attr))]) for
            attr in self.get_own_attributes()))

    @property
    def entity_count(self):
        '''Return the number of BIDS entities in a given UniqueId'''
        return str(self).count('_') + 1

    def get_own_attributes(self):
        '''Return generator of BIDS attributes that are present'''
        return (attr for attr in ATTRIBUTES if hasattr(self, attr))
