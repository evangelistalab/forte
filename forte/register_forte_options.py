import os
import yaml

# This file contains functions for parsing the YAML file (options.yaml)
# that contains the default options for forte and registering them with
# the options object.


def parse_yaml_file(file_name="options.yaml"):
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    file_path = os.path.join(script_directory, file_name)

    with open(file_path, "r") as file:
        try:
            yaml_dic = yaml.safe_load(file)
            return yaml_dic
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {file_path}: {e}")
            return None


def register_forte_options(options):
    yaml_dic = parse_yaml_file()
    for group in yaml_dic:
        options.set_group(group)
        yaml_dict_group = yaml_dic[group]
        for key in yaml_dict_group:
            register_option(key, yaml_dict_group[key], options)


def register_option(key, vals, options):
    opt_typ = vals["type"]
    opt_val = vals["default"]
    opt_msg = vals["help"]

    if opt_typ == "bool":
        options.add_bool(key, opt_val, opt_msg)
    if opt_typ == "double":
        if opt_val == "None":
            options.add_double(key, None, opt_msg)
        else:
            options.add_double(key, opt_val, opt_msg)
    if opt_typ == "double_list":
        options.add_double_list(key, opt_msg)
    if opt_typ == "int":
        if opt_val == "None":
            options.add_int(key, None, opt_msg)
        else:
            options.add_int(key, opt_val, opt_msg)
    if opt_typ == "int_list":
        options.add_int_list(key, opt_msg)
    if opt_typ == "list":
        options.add_list(key, opt_msg)
    if opt_typ == "str":
        if "choices" in vals:
            choices = vals["choices"]
            options.add_str(key, opt_val, choices, opt_msg)
        else:
            options.add_str(key, opt_val, opt_msg)
    if opt_typ == "deprecated":
        options.add_deprecated(key, opt_msg)
>>>>>>> main
