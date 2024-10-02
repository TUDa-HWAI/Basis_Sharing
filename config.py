import argparse
import yaml


def add_args():
    paser = argparse.ArgumentParser(description="ShareModel")
    paser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    paser.add_argument(
        "--calibration_size",
        "--cs",
        help="calibration size",
        type=int,
        default=256,
    )
    paser.add_argument(
        "--dataset_name",
        help="dataset for load",
        type=str,
        default="wikitext",
    )
    paser.add_argument(
        "--dataset_cache_dir",
        help="change dataset cache dir",
        type=str,
        default=None,
    )
    args, unknown = paser.parse_known_args()
    return args


class ShareConfig:
    name_map = {
        'meta-llama/Llama-2-7b-hf': "llama2-7b",
        "jeffwan/llama-7b-hf": "llama2-7b",
        "jeffwan/llama-13b-hf": "llama2-13b",
        "jeffwan/llama-30b-hf": "llama2-30b",
        'gpt2': "gpt2",
        'facebook/opt-6.7b': 'opt-6.7b',
        "mistralai/Mistral-7B-v0.1": "mistral-7b"
    }

    weight_info = {
        "llama2-7b": {
            "self_attn.k_proj": (4096, 4096),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 4096),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 11008),
            "mlp.gate_proj": (4096, 11008),
            "mlp.down_proj": (11008, 4096),
        },

        "llama2-13b": {
            "self_attn.k_proj": (5120, 5120),
            "self_attn.q_proj": (5120, 5120),
            "self_attn.v_proj": (5120, 5120),
            "self_attn.o_proj": (5120, 5120),
            "mlp.up_proj": (5120, 13824),
            "mlp.gate_proj": (5120, 13824),
            "mlp.down_proj": (13824, 5120),
        },

        "llama2-30b": {
            "self_attn.k_proj": (6656, 6656),
            "self_attn.q_proj": (6656, 6656),
            "self_attn.v_proj": (6656, 6656),
            "self_attn.o_proj": (6656, 6656),
            "mlp.up_proj": (6656, 17920),
            "mlp.gate_proj": (6656, 17920),
            "mlp.down_proj": (17920, 6656),
        },

        "gpt2": {
            "attn.c_attn": (768, 2304),
            "attn.c_proj": (768, 768),
            "mlp.c_fc": (768, 3072),
            "mlp.c_proj": (3072, 768)
        },

        "opt-6.7b": {
            "self_attn.k_proj": (4096, 4096),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 4096),
            "self_attn.out_proj": (4096, 4096),
            "fc1": (4096, 16384),
            "fc2": (16384, 4096),
        },
        "mistral-7b": {
            "self_attn.k_proj": (4096, 1024),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 14336),
            "mlp.gate_proj": (4096, 14336),
            "mlp.down_proj": (14336, 4096),
        },

    }

    def __init__(self, cmd_args):
        cmd_args_dict = cmd_args.__dict__
        self.configuration = self.load_yaml_config(cmd_args.yaml_config_file)
        self.set_attr_from_config(self.configuration)
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)

    @staticmethod
    def load_yaml_config(yaml_path):
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    def set_attr_from_config(self, configuration):
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)
