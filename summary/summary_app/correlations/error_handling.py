import os
def report_missing(matched_dct, pipelines, output_dir):

    flat_missing_dct = {'missing_old': {}, 'missing_new': {}}

    for missing_type in flat_missing_dct:
        for subdct in matched_dct[missing_type]:
            if type(subdct) is not dict:
                continue
            for key, path_list in subdct.items():
                if key[0] not in flat_missing_dct[missing_type].keys():
                    flat_missing_dct[missing_type][key[0]] = []
                flat_missing_dct[missing_type][key[0]].append(path_list[0])

    report_msg = ""

    if flat_missing_dct['missing_new']:
        report_msg += "\nThese outputs are in {0}, and are either missing " \
                      "in {1} or were not picked up by this script's file parsing:" \
                      "\n\n".format(list(pipelines)[0], list(pipelines)[1])
        for output_type in flat_missing_dct['missing_new']:
            report_msg += "\n{0}:\n\n".format(output_type)
            for path in flat_missing_dct['missing_new'][output_type]:
                report_msg += "    {0}\n".format(path)
        report_file = os.path.join(output_dir, "report_missing_new.txt")
        #print("{0} \nPlease check {1} for details.".format(report_msg, report_file))
        with open(report_file, 'wt') as f:
            f.write(report_msg)

    if flat_missing_dct['missing_old']:
        report_msg += "\nThese outputs are in {0}, and missing " \
                      "in {1} or were not picked up by this script's file parsing:" \
                     "\n\n".format(list(pipelines)[1], list(pipelines)[0])
        for output_type in flat_missing_dct['missing_old']:
            report_msg += "\n{0}:\n\n".format(output_type)
            for path in flat_missing_dct['missing_old'][output_type]:
                report_msg += "    {0}\n".format(path)
        report_file = os.path.join(output_dir, "report_missing_old.txt")
        #print("{0} \nPlease check {1} for details.".format(report_msg, report_file))
        with open(report_file, 'wt') as f:
            f.write(report_msg)
