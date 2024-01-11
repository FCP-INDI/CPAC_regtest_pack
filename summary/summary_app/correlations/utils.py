import numpy as np

def read_yml_file(yml_filepath):
    import yaml
    with open(yml_filepath,"r") as f:
        yml_dict = yaml.safe_load(f)

    return yml_dict


def write_yml_file(yml_dict, out_filepath):
    import yaml
    with open(out_filepath, "wt") as f:
        yaml.safe_dump(yml_dict, f)


def read_txt_file(txt_file):
    with open(txt_file,"r") as f:
        strings = f.read().splitlines()
    return strings


def write_txt_file(text_lines, out_filepath):
    with open(out_filepath, "wt") as f:
        for line in text_lines:
            f.write("{0}\n".format(line))


def read_pickle(pickle_file):
    import pickle
    with open(pickle_file, "rb") as f:
        dct = pickle.load(f)
    return dct


def write_pickle(dct, out_filepath):
    import pickle
    with open(out_filepath, "wb") as f:
        pickle.dump(dct, f, protocol=pickle.HIGHEST_PROTOCOL)

def parse_csv_data(csv_lines):
    parsed_lines = []
    for line in csv_lines:
        if '#' not in line:
            new_row = [float(x.rstrip('\n')) for x in line.split('\t') if x != '']
            parsed_lines.append(new_row)
        csv_np_data = np.asarray(parsed_lines)

    return csv_np_data

