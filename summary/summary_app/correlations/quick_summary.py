import numpy as np
import os
from summary_app.correlations.utils import write_txt_file

def quick_summary(corr_map_dct, output_dir):
    for corr_group in corr_map_dct["correlations"].keys():
        cat_dct = {}
        lines = []
        for output_type, corr_vec in dict(corr_map_dct["correlations"][corr_group]).items():
            try:
                corrmean = np.mean(np.asarray(corr_vec))
            except TypeError:
                continue
            lines.append("{0}: {1}".format(output_type, corrmean))
        write_txt_file(lines, os.path.join(output_dir, "average_{0}.txt".format(corr_group)))
 