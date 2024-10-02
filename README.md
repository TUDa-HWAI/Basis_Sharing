# Basis Sharing
This is the code for the paper "BASIS SHARING: CROSS-LAYER PARAMETER SHARING
FOR LARGE LANGUAGE MODEL COMPRESSION". Some config examples are added in config directory.

## Run Basis Sharing
To run Basis Sharing on LLaMA-7B for generation tasks, run
```
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
```
make sure to set *build_calib* as true for a model, when you want to compress it for the first time.

After compress with WikiText, to test with other dataset run
~~~
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml --dataset_name <ptb, C4, WikiText>
~~~
For C4 you need to download them from [link](https://drive.google.com/drive/folders/123Id1MkZVsKySGy_sMO4RgiJKrtPcvUp?usp=drive_link). Don't forget to update *dataset_cache_dir* in config file.

## Run LoRA
~~~
python lora.py  --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## Run Reasoning tasks
~~~
python test_adapter.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## Run Throughput tasks
~~~
python test_throughput.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

Reference

@misc{{parametersharing2024,

title={Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression},

author={Jingcun Wang and Yu-Guang Chen and Ing-Chao Lin and Bing Li and Grace Li Zhang},

archivePrefix={arXiv},

year={2024} 
}
