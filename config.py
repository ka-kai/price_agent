import os
from ast import literal_eval as make_tuple
import yaml
from pathlib import Path


class Config:
    """
    Config class. Used to store all data from the config.yaml file and the agent arguments in a usable form within the application.
    Note: Pass parameter groups (e.g. weights) as dictionaries. This facilitates passing it to wandb when initializing the run
    and replacing a setting specified via the command line (--config).
    """
    def __init__(self, file_config="config.yaml"):
        # Load the config file
        d = Path(__file__).resolve().parent / "config"
        with open(d / file_config, "r") as f:
            dict_config = yaml.safe_load(f)

        # Append all items to the config object
        # Process specific values such that using them in the scripts is simplified
        dict_config = self._process_tuples(dict_config)
        for k, v in dict_config.items():
            if "path" in k:
                setattr(self, k, Path(v))
            elif "file" in k:
                setattr(self, k.replace("file", "path"), self.path_data / v)
            else:
                setattr(self, k, v)

        # Add agent parameters
        with open(d / f"args_{self.a['algo'].lower()}.yaml", "r") as f:
            dict_config_algo = yaml.safe_load(f)
        for k, v in dict_config_algo.items():
            self.a[k] = v

        # Values from environment variables
        self.project = os.environ["WANDB_PROJECT"]
        self.entity = os.environ["WANDB_ENTITY"]

        return

    def _process_tuples(self, dictionary):
        """
        Turn strings that represent tuples into tuples.

        :param dict dictionary: dictionary to be processed
        :return: dictionary with processed tuples
        :rtype: dict
        """
        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = self._process_tuples(v)
            elif isinstance(v, str) and v[0] == "(" and v[-1] == ")":
                dictionary[k] = make_tuple(v)
        return dictionary

    def _replace_value_in_nested_dict(self, dictionary, key, new_value):
        """
        Recursive function to replace the value of a given key in a (potentially) nested dictionary.

        :param dict dictionary: dictionary to be searched
        :param string key: key to be replaced
        :param new_value: new value
        :return: flag indicating whether the key was found
        :rtype: bool
        """
        # Recursive function to replace parameters in the config
        flag_found = False
        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k], flag_found = self._replace_value_in_nested_dict(v, key, new_value)
                if flag_found:
                    return dictionary, True
            elif k == key:
                if isinstance(v, bool):
                    dictionary[k] = True if new_value == "True" else False  # bool("False") is True --> treat differently
                else:
                    dictionary[k] = type(v)(new_value)  # cast the new value to the type of the old value
                return dictionary, True

        return None, flag_found

    def replace_value(self, key, new_value):
        """
        Replace the value of a given key in the config object.

        :param string key: key to be replaced
        :param new_value: new value
        :return:
        """
        flag_found = False  # flag indicating whether the key was found
        for k, v in self.__dict__.items():
            if k == key:  # if the given key is a direct attribute of the config object, replace it
                if isinstance(v, bool):
                    self.__dict__[k] = True if new_value == "True" else False  # bool("False") is True --> treat differently
                else:
                    self.__dict__[k] = type(v)(new_value)  # cast the new value to the type of the old value
                flag_found = True
                break
            elif isinstance(v, dict):  # if the attribute is a dictionary, search for the key in the dictionary
                dict_new, flag_found = self._replace_value_in_nested_dict(dictionary=v, key=key, new_value=new_value)
                if flag_found:
                    self.__dict__[k] = dict_new
                    break
        assert flag_found, f"Configuration setting '{key}' could not be found."

        print(f"\nReplaced the value for attribute '{key}' in the config object with "
              f"the value specified in the command line, i.e., {new_value}.")

        return
