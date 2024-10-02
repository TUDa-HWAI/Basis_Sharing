import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from prepare_data import prepare_data
from model_factory import create_model
from config import ShareConfig, add_args

if __name__ == '__main__':
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print("Use update: {}".format(config.update))
    print(config.untrained_model_path)

    tasks = ["openbookqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag", "piqa", "mathqa"]

    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print(config.compression_ratio)
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = "[PAD]"
    # model = create_model(config)
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = model.cuda()

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=128, max_batch_size=256)
    res = lm_eval.simple_evaluate(hflm, tasks=tasks, num_fewshot=0, batch_size=128, max_batch_size=256,
                                  device=model.device)
    print(res["results"])
