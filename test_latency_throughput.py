from tqdm import tqdm
import numpy as np
import torch
import time
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, AutoTokenizer
from model_factory import create_model
from config import ShareConfig, add_args
from prepare_data import prepare_data

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
                                                                           32, config.dataset_cache_dir)
    model = create_model(config)
    model = model.cuda()
    model.config.use_cache = False
    model.eval()
    model = torch.compile(model)
    index = len(val_dataset)
    index = [i for i in range(index //4)] # make it run faster
    subset = Subset(val_dataset, index)

    batch_size = 512
    warm_up = True
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            collate_fn=data_collator,
                            pin_memory=True, num_workers=4)
    total_time = []
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            if warm_up:
                print("start warm up")
                for _ in tqdm(range(10)):
                    out = model(**batch)
                warm_up = False
            torch.cuda.synchronize(device=0)
            start = time.time()
            out = model(**batch)
            torch.cuda.synchronize(device=0)
            total_time.append(time.time() - start)
            del out
    print("throughput is: {}".format(512*32 / np.median(total_time)))

