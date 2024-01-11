import os
from summary_app.correlations.utils import read_pickle, write_pickle, write_yml_file      
from summary_app.correlations.error_handling import report_missing
from summary_app.correlations.file_utils import gather_all_files, match_filepaths
from summary_app.correlations.cpac_correlations import organize_correlations, post180_organize_correlations, run_correlations
from summary_app.correlations.plot import create_boxplot
from summary_app.correlations.quick_summary import quick_summary

def compare_pipelines(input_dct, dir_type='output_dir'):

    output_dir = input_dct['settings']['output_dir']
    pickle_dir = input_dct['settings']['pickle_dir']

    corrs_pkl = os.path.join(pickle_dir, "{0}_correlations.p".format(dir_type))
    matched_pkl = os.path.join(pickle_dir, "{0}_matched_files.p".format(dir_type))
    
    all_corr_dct = None
    if os.path.exists(corrs_pkl):
        print("\n\nFound the correlations pickle: {0}\n\n"
              "Starting from there..\n".format(corrs_pkl))
        all_corr_dct = read_pickle(corrs_pkl)
    elif os.path.exists(matched_pkl):
        print("\n\nFound the matched filepaths pickle: {0}\n\n"
              "Starting from there..\n".format(matched_pkl))
        matched_dct = read_pickle(matched_pkl)

        if dir_type == 'output_dir':
            report_missing(matched_dct, input_dct["pipelines"].keys(), output_dir)

    else:
        # gather all relevant output and working files
        outfiles1_dct, outfiles2_dct = gather_all_files(input_dct, pickle_dir, 
                                                        source=dir_type)
        
        matched_dct = match_filepaths(outfiles1_dct, outfiles2_dct)
        write_pickle(matched_dct, matched_pkl)

        if dir_type == 'output_dir':
            report_missing(matched_dct, input_dct["pipelines"].keys(), output_dir)

    if not all_corr_dct:
        all_corr_dct = run_correlations(matched_dct,
                                        input_dct, 
                                        source=dir_type,
                                        quick=input_dct['settings']['quick'],
                                        verbose=input_dct['settings']['verbose'])
        write_pickle(all_corr_dct, corrs_pkl)
    
    if dir_type == 'work_dir':
        sorted_vals = []
        #sorted_keys = sorted(all_corr_dct, key=all_corr_dct.get)
        for key in all_corr_dct.keys(): #sorted_keys:
            if 'file reading problem:' in key or 'different shape' in key or 'correlating problem' in key:
                continue
            else:
                sorted_vals.append("{0}: {1}".format(all_corr_dct[key], key))
        working_corrs_file = os.path.join(output_dir, "work_dir_correlations.txt")
        with open(working_corrs_file, 'wt') as f:
            for line in sorted_vals:
                f.write(line)
                f.write("\n")

    else:
        pre180 = False
        if pre180:
            organize = organize_correlations
        else:
            organize = post180_organize_correlations

        corr_map_dict = organize(all_corr_dct["concordance"], "concordance",
                                 quick=input_dct['settings']['quick'])
        corr_map_dict["pipeline_names"] = input_dct["pipelines"].keys()
    
        pearson_map_dict = organize(all_corr_dct["pearson"], "pearson",
                                    quick=input_dct['settings']['quick'])
        pearson_map_dict["pipeline_names"] = input_dct["pipelines"].keys()
        
        quick_summary(corr_map_dict, output_dir)
        quick_summary(pearson_map_dict, output_dir)

        if all_corr_dct['sub_optimal']:
            write_yml_file(all_corr_dct['sub_optimal'], os.path.join(output_dir, "sub_optimal.yml"))

        for corr_group_name in corr_map_dict["correlations"].keys():
            corr_group = corr_map_dict["correlations"][corr_group_name]
            create_boxplot(corr_group, corr_group_name,
                           corr_map_dict["pipeline_names"], output_dir)

        for corr_group_name in pearson_map_dict["correlations"].keys():
            corr_group = pearson_map_dict["correlations"][corr_group_name]
            create_boxplot(corr_group, corr_group_name,
                           pearson_map_dict["pipeline_names"], output_dir)


