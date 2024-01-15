
from summary_app.config import Config
from summary_app.main.utils import find_corr_directories

class All_Correlations:
    correlations_list:list
    directory_list:list
    

    def __init__(self):
        self.directory_list, self.correlations_list = find_corr_directories()