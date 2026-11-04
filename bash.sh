#!/bin/bash

# python recovery_experiment.py \
#   --dataset_type klar \
#   --checkpoint /root/autodl-tmp/models/llama3.1-8b/llama3.1-8b-awq-int4 \
#   --quant_type awq-int4 \
#   --model_type llama3.1 \
#   --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
#   --batch_size 32 \
#   --save_path results \
#   --max_new_tokens 8 \
#   --stages 1,2 \
#   --save_path result/recovery \
#   --resume true \
#   --calculate_ppl false \
#   --seed 42 \
#   --fp16_checkpoint /root/autodl-tmp/models/llama3.1-8b/llama3.1-8b-instruct \


# python recovery_experiment.py \
#   --dataset_type klar \
#   --checkpoint /root/autodl-tmp/models/llama3.1-8b/llama3.1-8b-instruct \
#   --quant_type bnb-nf4 \
#   --model_type llama3.1 \
#   --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
#   --batch_size 32 \
#   --save_path results \
#   --max_new_tokens 8 \
#   --stages 1 \
#   --save_path result/recovery \
#   --resume true \
#   --calculate_ppl false \
#   --seed 42 \
#   --fp16_checkpoint /root/autodl-tmp/models/llama3.1-8b/llama3.1-8b-instruct \

python recovery_experiment.py \
  --dataset_type klar \
  --checkpoint /root/autodl-tmp/models/llama3.1-8b/llama3.1-8b-gptq-int4 \
  --quant_type gptq-int4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results \
  --max_new_tokens 8 \
  --stages 1,2 \
  --save_path result/recovery \
  --resume true \
  --calculate_ppl false \
  --seed 42 \
  --fp16_checkpoint /root/autodl-tmp/models/llama3.1-8b/llama3.1-8b-instruct \

# python main.py \
#   --dataset_type klar \
#   --checkpoint /root/autodl-tmp/models/llama3.1-8b/llama3.1-8b-instruct \
#   --quant_type fp16 \
#   --model_type llama3.1 \
#   --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
#   --batch_size 32 \
#   --save_path results \
#   --max_new_tokens 8 \
#   --calculate_ppl false \
  