import os.path
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from accelerate import load_checkpoint_and_dispatch

from config import ShareConfig
from utils import match_state_dict
from calib import Calib
from prepare_data import prepare_data
from utils import compute_num_basis
from group import change_model, update_model
from models.gpt2 import ShareGPT2LMHeadModel
from models.llama import ShareLlamaForCausalLM
from models.opt import ShareOPTForCausalLM
from models.mistral import ShareMistralForCausalLM


def do_update_model(config, model, dataset, tokenizer, data_collator):
    if os.path.exists(config.updated_model_path):
        print("Start load model!")
        print("Load: {}".format(config.updated_model_path))
        if config.model_type == "gpt2":
            model = ShareGPT2LMHeadModel.from_pretrained(config.updated_model_path, device_map='auto')
        elif config.model_type == "llama2":
            model = ShareLlamaForCausalLM.from_pretrained(config.updated_model_path, device_map='auto')
        elif config.model_type == "opt":
            model = ShareOPTForCausalLM.from_pretrained(config.updated_model_path, device_map='auto')
        elif config.model_type == "mistral":
            model = ShareMistralForCausalLM.from_pretrained(config.updated_model_path, device_map='auto')
        else:
            raise ValueError
    else:
        std_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="cpu")
        std_model.config.use_cache = False
        model = load_checkpoint_and_dispatch(model, config.untrained_model_path, device_map="auto")

        # Prepare Dataloader for calibration data
        torch.manual_seed(2023)
        index = torch.randperm(len(dataset))
        index = index[:config.calibration_size]
        subset = Subset(dataset, index)
        dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False, collate_fn=data_collator,
                                pin_memory=True, num_workers=4)

        if config.build_update_calib:
            print("Start build update calib!")
            names = config.share_part + config.private_part
            basis_name = []
            for name in names:
                if name == "q" or name == "v" or name == "gate":
                    continue
                basis_name.append(name + "_basis")

            Calib.build_update_dataset(model, dataloader, basis_name, config.model_type, config.update_calib_path)

        model_config = model.config
        short_model_name = ShareConfig.name_map[config.model_name]

        names = config.share_part + config.private_part
        for name in names:
            print("Update {}".format(name))
            model = update_model(std_model=std_model,
                                 model=model,
                                 model_type=config.model_type,
                                 groups=getattr(model_config, name + "_groups"),
                                 name=getattr(config, name + "_name"),
                                 step=
                                 ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")][
                                     1],
                                 num_basis=getattr(model_config, "num_basis_" + name),
                                 basis_name=name + "_basis",
                                 calib_path=config.update_calib_path,
                                 )
        if config.save_updated_model:
            model.save_pretrained(config.updated_model_path, safe_serialization=False)
            tokenizer.save_pretrained(config.updated_model_path)
    return model


def create_model(config):
    if os.path.exists(config.untrained_model_path):
        model_path = config.untrained_model_path
        print("Start load model!")
        print("Start load: {}".format(config.untrained_model_path))
        if config.model_type == "gpt2":
            model = ShareGPT2LMHeadModel.from_pretrained(model_path, device_map='auto', )
        elif config.model_type == "llama2":
            if "30b" in config.untrained_model_path:
                model = ShareLlamaForCausalLM.from_pretrained(model_path, device_map='auto',
                                                              torch_dtype=torch.float16)
            else:
                model = ShareLlamaForCausalLM.from_pretrained(model_path, device_map='cpu')
        elif config.model_type == "opt":
            model = ShareOPTForCausalLM.from_pretrained(model_path, device_map='auto')
        elif config.model_type == "mistral":
            model = ShareMistralForCausalLM.from_pretrained(model_path, device_map='auto')
        else:
            raise ValueError

    else:
        if config.model_type == "llama2":
            tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = "[PAD]"
        print("Start create model!")
        model_config = AutoConfig.from_pretrained(config.model_name)
        model_config.use_cache = False
        if config.model_name == "jeffwan/llama-30b-hf":
            std_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto",
                                                             torch_dtype=torch.float16)
        else:
            std_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")

        if config.build_calib:
            train_dataset, val_dataset, tokenized_test, data_collator = prepare_data(config.dataset_name, tokenizer,
                                                                                     config.context_length, config.dataset_cache_dir)
            # Prepare Dataloader for calibration data
            torch.manual_seed(2023)
            index = torch.randperm(len(train_dataset))
            index = index[:config.calibration_size]
            subset = Subset(train_dataset, index)
            dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False, collate_fn=data_collator,
                                    pin_memory=True, num_workers=4)

            print("Start create calib!")
            calib_names = []
            if hasattr(config, "k_name"):
                # calibration data for k, q, v is the same
                calib_names.append(config.k_name)
            if hasattr(config, "attn_name"):
                calib_names.append(config.attn_name)
            calib_names.append(config.o_name)
            calib_names.append(config.up_name)
            calib_names.append(config.down_name)
            Calib.build_calibration_dataset(std_model, dataloader, calib_names, config.model_type, config.calib_path)
            print("Calib build done!")

        short_model_name = ShareConfig.name_map[config.model_name]

        # Share Part
        names = config.share_part
        for name in names:
            print("Config for {}".format(name))
            nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
            num_group = model_config.num_hidden_layers // config.group_size
            rest = model_config.num_hidden_layers % config.group_size
            gs = config.group_size
            group = [[gs * i + j for j in range(config.group_size)] for i in range(num_group)]
            if rest != 0:
                group += [[num_group * config.group_size + i for i in range(rest)]]
            setattr(model_config, name + "_groups", group)
            num_basis = compute_num_basis(nx, nf, config.group_size, config.compression_ratio)
            setattr(model_config, "num_basis_" + name, num_basis)
            print("num_basis {}".format(num_basis))

        # Private Part
        names = config.private_part
        for name in names:
            print("Config for {}".format(name))
            setattr(model_config, name + "_groups", [[i] for i in range(model_config.num_hidden_layers)])
            nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
            num_basis = compute_num_basis(nx, nf, 1, config.compression_ratio)
            setattr(model_config, "num_basis_" + name, num_basis)
            print("num_basis {}".format(num_basis))

        if config.model_type == "llama2":
            if "30b" in config.model_name:
                model_config.torch_dtype = torch.float16
            model = ShareLlamaForCausalLM(model_config)
        elif config.model_type == "gpt2":
            model = ShareGPT2LMHeadModel(model_config)
        elif config.model_type == "opt":
            model = ShareOPTForCausalLM(model_config)
        elif config.model_type == "mistral":
            model = ShareMistralForCausalLM(model_config)
        else:
            raise NotImplementedError

        print("Model init finished!")
        if not hasattr(config, "tfs"):
            matched_state_dict, _ = match_state_dict(model.state_dict(), std_model.state_dict())
            model.load_state_dict(matched_state_dict, strict=False)

            # Share Part
            names = config.share_part + config.private_part
            for name in names:
                print("Change {}".format(name))
                model = change_model(std_model=std_model,
                                     model=model,
                                     model_type=config.model_type,
                                     groups=getattr(model_config, name + "_groups"),
                                     name=getattr(config, name + "_name"),
                                     step=ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")][1],
                                     num_basis=getattr(model_config, "num_basis_" + name),
                                     basis_name=name + "_basis",
                                     calib_path=config.calib_path,
                                     )

            if config.save_untrained_model:
                model.save_pretrained(config.untrained_model_path, safe_serialization=False)
                tokenizer.save_pretrained(config.untrained_model_path)

    return model
