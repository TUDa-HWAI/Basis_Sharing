import os
from transformers import LlamaTokenizer, AutoTokenizer
from peft import LoraConfig, TaskType
from peft import get_peft_model, PeftModel
from model_factory import create_model
from transformers import TrainingArguments, Trainer
from config import ShareConfig, add_args
from test import compute_ppl
from prepare_data import prepare_data


def lora_finetune(config):
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = "[PAD]"
    train_dataset, val_dataset, test_dataset, data_collator = prepare_data(config.dataset_name, tokenizer,
                                                                           config.context_length,
                                                                           config.dataset_cache_dir)
    model = create_model(config)
    if os.path.exists(config.lora_output_dir):
        print("Start lora test!")
        model = PeftModel.from_pretrained(model, config.lora_output_dir)
        print(compute_ppl(config.context_length, config.stride, test_dataset, model, "cuda"))
    else:
        os.environ["WANDB_PROJECT"] = ShareConfig.name_map[config.model_name]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
        )

        model = get_peft_model(model, peft_config)

        trainer_config = TrainingArguments(output_dir=config.lora_output_dir,
                                           overwrite_output_dir=True,
                                           evaluation_strategy="epoch",
                                           per_device_train_batch_size=config.lora_train_batch_size,
                                           per_device_eval_batch_size=config.lora_train_batch_size,
                                           gradient_accumulation_steps=1,
                                           lr_scheduler_type="constant",
                                           logging_steps=1,
                                           learning_rate=config.lora_learning_rate,
                                           save_total_limit=1,
                                           seed=42,
                                           data_seed=0,
                                           save_safetensors=False,
                                           bf16=True,
                                           num_train_epochs=config.lora_train_epoch,
                                           save_strategy="epoch",
                                           load_best_model_at_end=True,
                                           metric_for_best_model='loss',
                                           run_name=config.lora_run_name,
                                           )
        trainer = Trainer(model=model,
                          args=trainer_config,
                          data_collator=data_collator,
                          train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          tokenizer=tokenizer)

        model.print_trainable_parameters()

        trainer.train()
        if config.save_lora:
            trainer.save_model()

        print(compute_ppl(config.context_length, config.stride, test_dataset, model, "cuda"))


if __name__ == '__main__':
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print(config.compression_ratio)
    lora_finetune(config)
