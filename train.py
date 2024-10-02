from transformers import Trainer, LlamaTokenizer, AutoTokenizer
from transformers import set_seed
from transformers import TrainingArguments
from model_factory import create_model
from config import ShareConfig, add_args
from prepare_data import prepare_data


def train(config):
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    set_seed(2024)
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = "[PAD]"
    train_dataset, val_dataset, tokenized_test, data_collator = prepare_data(config.dataset_name, tokenizer,
                                                                             config.context_length,
                                                                             config.dataset_cache_dir)
    model = create_model(config)

    trainer_config = TrainingArguments(output_dir=config.trained_model_path,
                                       overwrite_output_dir=True,
                                       evaluation_strategy='epoch',
                                       per_device_train_batch_size=1,
                                       per_device_eval_batch_size=1,
                                       gradient_accumulation_steps=8,
                                       lr_scheduler_type="constant",
                                       logging_steps=1,
                                       learning_rate=2e-6,
                                       save_total_limit=1,
                                       seed=42,
                                       data_seed=0,
                                       weight_decay=0.001,
                                       max_grad_norm=0.01,
                                       bf16=True,
                                       num_train_epochs=3,
                                       save_strategy="epoch",
                                       load_best_model_at_end=True,
                                       metric_for_best_model='loss',
                                       run_name="llama2-7b",
                                       )
    trainer = Trainer(model=model,
                      args=trainer_config,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=tokenizer)

    trainer.train()
    model = trainer.model
    model.save_pretrained(config.trained_model_path,
                          safe_serialization=False,
                          is_main_process=trainer.accelerator.is_main_process,
                          save_function=trainer.accelerator.save,
                          state_dict=trainer.accelerator.get_state_dict(model, unwrap=False))


if __name__ == '__main__':
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print(config.compression_ratio)
    train(config)
