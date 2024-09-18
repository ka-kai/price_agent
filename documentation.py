from datetime import datetime
from deepdiff import DeepDiff
import logging
import os
from pathlib import Path
import shutil
import sys
import yaml


def setup_and_documentation(args, config):
    # Setup
    if sys.gettrace() is not None:  # script is run in debugging mode
        flag_debug = True
        # Set parameters that are usually set via console if script is run via debugging
        config.mode = "train"
        config.dev_ctrl = "dynamic"
        config.policy = "prop_infl_per_day"
        config.path_pretrained_model = None
        config.model = None
        config.scripts_prev = False
        config.n_eps = 1
        config.notes = "debugging"
        config.flag_rb = False
        config.a["model_args"]["learning_starts"] = 0   # start training earlier when debugging to reduce waiting time;
                                                        # when number of transitions in the replay buffer < batch_size, it still samples <batch_size> transitions
                                                        # --> batch contains transitions multiple times
    else:
        flag_debug = False

        # Check if inputs are as expected and attach the settings to config
        assert args.dev_ctrl in ["none", "ripple", "dynamic"], "Please specify a valid external device controller as --dev_ctrl=none, ripple, dynamic."
        config.dev_ctrl = args.dev_ctrl
        assert args.policy in ["none", "prop_infl_per_day", "rl"], "Please specify a valid price controller as --policy=none, prop_infl, rl."
        config.policy = args.policy
        if config.policy == "rl":
            assert args.mode in ["train", "resume", "eval"], "Please specify a valid mode as --mode=train, resume or eval."
            config.mode = args.mode
            if args.mode in ["resume", "eval"]:
                assert args.path_pretrained_model is not None, f"The chosen mode is '{args.mode}'. Please provide the path to the trained model."
                config.path_pretrained_model = Path(args.path_pretrained_model)
            else:
                config.path_pretrained_model = None
            config.model = args.model
            if args.mode in ["train", "resume"]:
                assert args.n_eps is not None, "Please specify for how many episodes the model should be trained."
                config.n_eps = args.n_eps
            else:
                config.n_eps = None
        else:
            config.mode = "eval"
            config.path_pretrained_model = None
            config.n_eps = None
        config.scripts_prev = args.scripts_prev
        config.notes = args.notes
        # Adjust specified config values
        if args.config is not None:
            dict_args_config = {}
            # Expected input: string with k1:v1,k2:v2,....
            for i in args.config.replace(" ", "").split(","):
                k, v = i.split(":")
                v = float(v) if v.replace(".", "").isnumeric() else v  # turn value into float if possible
                dict_args_config[k] = v
                config.replace_value(key=k, new_value=v)

    # Create results folder
    if config.mode != "eval" or config.policy != "rl":
        # Directory with all the results
        # If a trained RL agent is evaluated, the results are stored in the corresponding directory (see elif)
        config.output_path = Path(os.environ["PATH_SIM_RESULTS"])
        if flag_debug:  # debugging mode
            config.run_name = "test"
            config.output_path = config.output_path / config.run_name
            if os.path.exists(config.output_path):
                shutil.rmtree(config.output_path)   # delete the old results
        else:
            config.run_name = f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{config.policy}_{config.dev_ctrl}"
            config.output_path = config.output_path / config.run_name
    elif config.mode == "eval":  # add an index to the folder name in case multiple evaluations are run
        n = 0
        while os.path.exists(config.path_pretrained_model / f"eval_{n}"):
            n += 1
        config.output_path = config.path_pretrained_model / f"eval_{n}"
    os.makedirs(config.output_path, mode=0o700)
    print(f"Created the following folder for the results: {config.output_path}\n")

    # Setup logging
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(config.output_path / "log_file.log"), logging.StreamHandler(sys.stdout)])

    # Copy the current scripts and config files into the result folder
    if config.mode != "eval" or config.policy != "rl":
        if config.scripts_prev:
            logging.info(f"Attention: You specified --scripts_prev=True, i.e., the Config and SimEnv from the original training run are used. "
                         f"The currents scripts are not copied to the results folder.\n")
        else:
            os.makedirs(config.output_path / "scripts", mode=0o700)
            for d in ["config", "model", "sim", ""]:
                if d != "":
                    os.makedirs(config.output_path / "scripts" / d, mode=0o700)
                p = Path(".") / d  # Path(".") is the project directory
                for t in ["*.py", "config.yaml", f"args_{config.a['algo'].lower()}.yaml"]:
                    for f in p.glob(t):
                        with open(f, "r") as f_in:
                            with open(config.output_path / "scripts" / d / f.name, "w") as f_out:
                                for line in (f_in.readlines()):
                                    if (f.suffix == ".yaml" and not flag_debug) and args.config is not None:
                                        # Overwrite config values that were specified differently in the console
                                        param = line.split(":")[0]
                                        param_ = param.replace(" ", "").replace("\t", "")  # remove white space and tabs
                                        if param_ in dict_args_config.keys():
                                            line = f"{param}: {dict_args_config[param_]}\n"
                                    print(line, end="", file=f_out)

    # Output the differences between the current config file and the one used for the original training
    if config.mode == "resume":
        for t in ["config/config.yaml", f"config/args_{config.a['algo'].lower()}.yaml"]:
            with open(config.path_pretrained_model / "scripts" / t, "r") as f:
                c_pretrained = yaml.safe_load(f)
            with open(Path(t), "r") as f:
                c_current = yaml.safe_load(f)
            ddiff = DeepDiff(c_pretrained, c_current, ignore_order=True).to_dict()
            logging.info(f"Comparison between current '{t}' and the one used for the original training:")
            for change, entries in ddiff.items():
                logging.info(f"# {change}")
                if isinstance(entries, dict):
                    for k, v in entries.items():
                        logging.info(str(k.ljust(30)) + " " + str(v))
                else:
                    for v in entries:
                        logging.info(str(v))
            logging.info(f"By default, the current file + changes based on --config are used; "
                         f"If you specified --scripts_prev=True, the original file + changes based on --config are used.\n")

    return config
