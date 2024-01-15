import os

from flask import Flask, send_from_directory
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from summary_app.correlations.utils import read_pickle, read_txt_file
from summary_app.config import Config


class Summary:
    correlations_dir: str
    pickle_dir: str
    output_dir_correlations: dict
    output_dir_matched_files: dict
    missing_outputs = {
        'missing_in_new': {},
        'missing_in_old': {}
    }
    template = "MNI152NLin6ASym"
    corr_list = [
        'desc-preproc_T1w',
        'desc-brain_mask',
        'label-CSF_mask',
        'label-WM_mask',
        'label-CSF_desc-preproc_mask',
        'label-WM_desc-preproc_mask',
        f'space-{template}_desc-preproc_T1w',
        'desc-mean_bold',
        'sbref',
        'space-T1w_sbref',
        f'space-{template}_sbref',
        'desc-preproc_bold',
        'desc-confounds_timeseries',
        f'space-{template}_desc-preproc_bold',
        f'space-{template}_reg-default_desc-preproc_bold'
    ]

    def __init__(self, correlations_dir):
        self.correlations_dir = correlations_dir
        self.pickle_dir = os.path.join(self.correlations_dir, "pickles")
        self.get_pickles()
        self.get_missing_outputs()

    def get_pickles(self):
        self.output_dir_correlations = read_pickle(os.path.join(self.pickle_dir, 'output_dir_correlations.p'))
        self.output_dir_matched_files = read_pickle(os.path.join(self.pickle_dir, 'output_dir_matched_files.p'))

    def get_missing_outputs(self):
        self.missing_outputs['missing_in_new'] = read_txt_file(
            os.path.join(self.correlations_dir, 'report_missing_new.txt'))
        self.missing_outputs['missing_in_old'] = read_txt_file(
            os.path.join(self.correlations_dir, 'report_missing_old.txt'))

