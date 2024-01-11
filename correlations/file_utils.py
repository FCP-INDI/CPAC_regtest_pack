import os
from s3_utils import pull_NIFTI_file_list_from_s3
from utils import read_pickle, write_pickle


def create_unique_file_dict(filepaths, output_folder_path, replacements=None):

    # filepaths:
    #   list of output filepaths from a CPAC output directory
    # output_folder_path:
    #   the CPAC output directory the filepaths are from
    # replacements:
    #   (optional) a list of strings to be removed from the filepaths should
    #   they occur

    # output
    #   files_dict
    #     a dictionary of dictionaries, format:
    #     files_dict["centrality"] = 
    #         {("centrality", midpath, nums): <filepath>, ..}

    files_dict = {}

    for filepath in filepaths:

        if "_stack" in filepath:
            continue

        if ("itk" in filepath) or ("xfm" in filepath) or ("montage" in filepath):
            continue
        path_changes = []
        real_filepath = filepath
        if replacements:
            for word_couple in replacements:
                if "," not in word_couple:
                    err = "\n\n[!] In the replacements text file, the old " \
                          "substring and its replacement must be separated " \
                          "by a comma.\n\n"
                    raise Exception(err)
                word = word_couple.split(",")[0]
                new = word_couple.split(",")[1]
                if word in filepath:
                    path_changes.append("old: {0}".format(filepath))
                    filepath = filepath.replace(word, new)
                    path_changes.append("new: {0}".format(filepath))
        if path_changes:
            import os
            with open(os.path.join(os.getcwd(), "path_changes.txt"), "wt") as f:
                for path in path_changes:
                    f.write(path)
                    f.write("\n")

        filename = filepath.split("/")[-1]

        # name of the directory the file is in
        folder = filepath.split("/")[-2]

        midpath = filepath.replace(output_folder_path, "")
        midpath = midpath.replace(filename, "")

        pre180 = False
        if pre180:
            # name of the output type/derivative
            try:
                category = midpath.split("/")[2]
            except IndexError as e:
                continue

            if "eigenvector" in filepath:
                category = category + ": eigenvector"
            if "degree" in filepath:
                category = category + ": degree"
            if "lfcd" in filepath:
                category = category + ": lfcd"
        else:
            tags = []
            category = filename
            category = category.rstrip('.gz').rstrip('.nii')

            excl_tags = ['sub-', 'ses-', 'task-', 'run-', 'acq-']

            # len(filetag) == 1 is temporary for broken/missing ses-* tag
            for filetag in filename.split("_"):
                for exctag in excl_tags:
                    if exctag in filetag or len(filetag) == 1:
                        category = category.replace(f'{filetag}_', '')

        # this provides a way to safely identify the specific file
        # without relying on a full string of the filename (because
        # this can change between versions depending on what any given
        # processing tool appends to output file names)
        nums_in_folder = [int(s) for s in folder if s.isdigit()]
        nums_in_filename = [int(s) for s in filename if s.isdigit()]

        file_nums = ''

        for num in nums_in_folder:
            file_nums = file_nums + str(num)

        for num in nums_in_filename:
            file_nums = file_nums + str(num)

        # load these settings into the tuple so that the file can be
        # identified without relying on its full path (as it would be
        # impossible to match files from two regression tests just
        # based on their filepaths)
        file_tuple = (category, midpath, file_nums)

        temp_dict = {}
        temp_dict[file_tuple] = [real_filepath]

        if category not in files_dict.keys():
            files_dict[category] = {}

        files_dict[category].update(temp_dict)
        
    return files_dict


def gather_all_files(input_dct, pickle_dir, source='output_dir'):

    file_dct_list = []

    for key, pipe_dct in input_dct['pipelines'].items():

        pipe_outdir = pipe_dct[source]

        if input_dct['settings']['s3_creds']:
            if not "s3://" in pipe_outdir:
                err = "\n\n[!] If pulling output files from an S3 bucket, the "\
                      "output folder path must have the s3:// prefix.\n\n"
                raise Exception(err)
        else:
            pipe_outdir = os.path.abspath(pipe_outdir).rstrip('/')

        pipeline_name = pipe_outdir.split('/')[-1]

        #if source == "output_dir" and "pipeline_" not in pipeline_name:
        #    err = "\n\n[!] Your pipeline output directory has to be a specific " \
        #          "one that has the 'pipeline_' prefix.\n\n(Not the main output " \
        #          "directory that contains all of the 'pipeline_X' subdirectories," \
        #          "and not a specific participant's output subdirectory either.)\n"
        #    raise Exception(err)

        output_pkl = os.path.join(pickle_dir, "{0}_{1}_paths.p".format(key, source))

        if os.path.exists(output_pkl):
            print("Found output list pickle for {0}, skipping output file" \
                  "path parsing..".format(key))
            pipeline_files_dct = read_pickle(output_pkl)
        else:
            if input_dct['settings']['s3_creds']:
                pipeline_files_list = pull_NIFTI_file_list_from_s3(pipe_outdir, 
                                                                   input_dct['settings']['s3_creds'])
            else:
                pipeline_files_list = gather_local_filepaths(pipe_outdir)

            pipeline_files_dct = create_unique_file_dict(pipeline_files_list,
                                                         pipe_outdir,
                                                         pipe_dct['replacements'])

            write_pickle(pipeline_files_dct, output_pkl)

        file_dct_list.append(pipeline_files_dct)

    return (file_dct_list[0], file_dct_list[1])


def match_filepaths(old_files_dict, new_files_dict):
    """Returns a dictionary mapping each filepath from the first CPAC run to the
    second one, matched to derivative, strategy, and scan.

    old_files_dict: each key is a derivative name, and each value is another
                    dictionary keying (derivative, mid-path, last digit in path)
                    tuples to a list containing the full filepath described by
                    the tuple that is the key
    new_files_dict: same as above, but for the second CPAC run

    matched_path_dict: same as the input dictionaries, except the list in the
                       sub-dictionary value has both file paths that are matched
    """

    # file path matching
    matched_path_dict = {}
    missing_in_old = []
    missing_in_new = []

    for key in new_files_dict:
        # for types of derivative...
        if key in old_files_dict.keys():
            for file_id in new_files_dict[key]:
                if file_id in old_files_dict[key].keys():

                    if key not in matched_path_dict.keys():
                        matched_path_dict[key] = {}

                    matched_path_dict[key][file_id] = \
                        old_files_dict[key][file_id] + new_files_dict[key][file_id]

                else:
                    missing_in_old.append(file_id)#new_files_dict[key][file_id])
        else:
            missing_in_old.append(new_files_dict[key])

    # find out what is in the last version's outputs that isn't in the new
    # version's outputs
    for key in old_files_dict:
        if new_files_dict.get(key) != None:
            missing_in_new.append(old_files_dict[key])

    if len(matched_path_dict) == 0:
        err = "\n\n[!] No output paths were successfully matched between " \
              "the two CPAC output directories!\n\n"
        raise Exception(err)

    matched_files_dct = {
        "matched": matched_path_dict,
        "missing_old": missing_in_old,
        "missing_new": missing_in_new
    }

    return matched_files_dct


def gather_local_filepaths(output_folder_path):
    import os
    filepaths = []

    print("Gathering file paths from {0}\n".format(output_folder_path))
    for root, dirs, files in os.walk(output_folder_path):
        # loops through every file in the directory
        for filename in files:
            # checks if the file is a nifti (.nii.gz)
            if '.nii' in filename or '.csv' in filename or '.txt' in filename \
                    or '.1D' in filename or '.tsv' in filename:
                filepaths.append(os.path.join(root, filename))

    if len(filepaths) == 0:
        err = "\n\n[!] No filepaths were found given the output folder!\n\n"
        raise Exception(err)

    return filepaths