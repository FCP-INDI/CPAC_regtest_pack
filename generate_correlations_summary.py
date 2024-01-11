import os

class Summary:
    title = "Summary of the CPAC Correlations"
    quick_summary = []
    to_plot = [
        'desc-preproc_T1w',
        'desc-brain_mask',
        'label-CSF_mask',
        'label-WM_mask',
        'label-CSF_desc-preproc_mask',
        'label-WM_desc-preproc_mask',
        'space-{template}_desc-preproc_T1w',
        'desc-mean_bold',
        'sbref',
        'space-T1w_sbref',
        'space-{template}_sbref',
        'desc-preproc_bold',
        'desc-confounds_timeseries',
        'space-{template}_desc-preproc_bold'
        ]
    
    def generate_summary(self):
        
        pass