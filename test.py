from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, AutoTokenizer
from model_factory import create_model, AutoModelForCausalLM
from config import ShareConfig, add_args
from prepare_data import prepare_data


def compute_ppl(max_length, stride, data, model, device):
    model.to(device)
    model = model.eval()
    seq_len = data.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            output = model(input_ids, labels=target_ids)

            neg_log_likelihood = output.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


if __name__ == '__main__':
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print(config.compression_ratio)
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = "[PAD]"
    train_dataset, val_dataset, test_dataset, data_collator = prepare_data(config.dataset_name, tokenizer,
                                                                           config.context_length,
                                                                           config.dataset_cache_dir)
    model = create_model(config)
    print(compute_ppl(config.context_length, config.stride, test_dataset, model, "cuda"))
