'''Directory parsing for heatmap generation

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
<https://www.gnu.org/licenses/>.
'''
from logging import warning
import os
from traits.api import Undefined
from .configs.defaults import Software


def determine_software_and_root(outputs_path):
    '''Given the path of an output directory, determine the software
    that generated the outputs and the root for this utility to
    generate its heatmaps.

    Parameters
    ----------
    outputs_path : str
        path to the output directory of supported preprocessing software

    Returns
    -------
    software : Software

    outputs_root : str
    '''
    software = outputs_root = None
    outputs_path = outputs_path.rstrip('/')  # drop trailing slash if present
    ls_outputs_path = os.listdir(outputs_path)

    def _fx_version(look_for, line_start, delimiter=None):
        if delimiter is None:
            delimiter = line_start
        if look_for in ls_outputs_path:
            _fx_dir = os.path.join(outputs_path, look_for)
            if os.path.isdir(_fx_dir):
                log_dir = os.path.join(_fx_dir, 'logs')
                if os.path.exists(log_dir) and os.path.isdir(log_dir):
                    citation = os.path.join(log_dir, 'CITATION.md')
                    if os.path.exists(citation):
                        version = get_version_from_loghead(
                            citation, line_start, delimiter)
                        if version is not Undefined:
                            return version
                sub_dirs = [sub_dir for sub_dir in [
                    os.path.join(_fx_dir, dir) for dir in os.listdir(_fx_dir)
                ] if os.path.isdir(sub_dir)]
                for sub_dir in sub_dirs:
                    fig_path = os.path.join(sub_dir, 'figures')
                    if os.path.exists(fig_path) and os.path.isdir(fig_path):
                        for report in [
                            path for path in os.listdir(fig_path) if
                            'about' in path and path.endswith('html')
                        ]:
                            version = get_version_from_loghead(
                                os.path.join(fig_path, report),
                                '<li>xcp_abcd version:')
                            if version:
                                return version
            return Undefined
        return None

    # C-PAC
    if 'log' in ls_outputs_path and 'output' in ls_outputs_path:
        log_dir = os.path.join(outputs_path, 'log')
        pipelines = [dir for dir in os.listdir(log_dir) if
                     os.path.isdir(os.path.join(log_dir, dir))]
        # look for C-PAC version in log header, return after finding one
        for pipeline in pipelines:
            pipeline_log_dir = os.path.join(log_dir, pipeline)
            runs = os.listdir(pipeline_log_dir)
            for run in runs:
                pypeline_logfile = os.path.join(pipeline_log_dir, run,
                                                'pypeline.log')
                if os.path.exists(pypeline_logfile):
                    return Software('C-PAC',
                                    get_version_from_loghead(pypeline_logfile,
                                                             'C-PAC', ':')
                                    ), outputs_path
        return Software('C-PAC'), outputs_path
    # fMRIPrep / XCPD
    version = _fx_version('fmriprep', 'performed using *fMRIPrep*')
    if version:
        return Software('fMRIPrep', version), outputs_path
    version = _fx_version('xcp_abcd',
                          'The eXtensible Connectivity Pipeline (XCP)',
                          'version')
    if version is Undefined:
        version = _fx_version('xcp_abcd', '<li>xcp_abcd version:')
    if version:
        return Software('XCP-D', version), outputs_path
    # XCPengine
    if os.path.basename(outputs_path).startswith('xcpengine'):
        group_deps_dir = os.path.join(outputs_path, 'group', 'dependencies')
        if os.path.exists(group_deps_dir):
            for argonaut in [os.path.join(group_deps_dir, filename) for
                             filename in os.listdir(group_deps_dir) if
                             filename.endswith('Description.json')]:
                version = get_version_from_loghead(argonaut, '"Processing"',
                                                   'xcpEngine-v')
                if version is not Undefined:
                    return Software('xcpEngine', version), outputs_path
        return Software('xcpEngine', outputs_path)
    # in case we're given on level deeper than expected
    if os.path.basename(outputs_path) in [
        'fmriprep', 'output', 'xcp_abcd'] or os.path.basename(
            os.path.dirname(outputs_path)).startswith('xcpengine'):
        return determine_software_and_root(os.path.dirname(outputs_path))
    return software, outputs_root


def get_version_from_loghead(log_path, line_start, delimiter=None):
    '''Function to grab a version from a logfile's head

    Parameters
    ----------
    log_path : str

    line_start : str

    delimiter : str, optional

    Returns
    -------
    version : str or Undefined
    '''
    if delimiter is None:
        delimiter = line_start
    with open(log_path, 'r', encoding='utf-8') as log_file:
        # for line in log_file.readline():
        #     line = line.strip()
        #     print(line)
        #     print(line_start)
        #     print(line.startswith(line_start))
        try:
            version_loglines = [line.strip() for line in [
                log_file.readline() for _ in range(10)] if
                line.lstrip().startswith(line_start)]
        except StopIteration:
            version_loglines = []
    if version_loglines:
        for line in version_loglines:
            if delimiter in line:
                version = line.split(delimiter, 1)[1].strip().rstrip(
                    '"\' ,.')
                if version.endswith('</li>'):
                    version = version[:-5]
                return version
    warning(f'Version not found in {log_path}')
    return Undefined
