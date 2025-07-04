---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-generation
tags:
- ERNIE4.5
---

<div align="center" style="line-height: 1;">
  <a href="https://ernie.baidu.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖_Chat-ERNIE_Bot-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/baidu" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Baidu-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/PaddlePaddle/ERNIE" target="_blank" style="margin: 2px;">
    <img alt="Github" src="https://img.shields.io/badge/GitHub-ERNIE-000?logo=github&color=0000FF" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://ernie.baidu.com/blog/ernie4.5" target="_blank" style="margin: 2px;">
    <img alt="Blog" src="https://img.shields.io/badge/🖖_Blog-ERNIE4.5-A020A0" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="#license" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache2.0-A5de54" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

# ERNIE-4.5-0.3B-Base

## ERNIE 4.5 Highlights

The advanced capabilities of the ERNIE 4.5 models, particularly the MoE-based A47B and A3B series, are underpinned by several key technical innovations:

1. **Multimodal Heterogeneous MoE Pre-Training:** Our models are jointly trained on both textual and visual modalities to better capture the nuances of multimodal information and improve performance on tasks involving text understanding and generation, image understanding, and cross-modal reasoning. To achieve this without one modality hindering the learning of another, we designed a *heterogeneous MoE structure*, incorporated *modality-isolated routing*, and employed *router orthogonal loss* and *multimodal token-balanced loss*. These architectural choices ensure that both modalities are effectively represented, allowing for mutual reinforcement during training.

2. **Scaling-Efficient Infrastructure:** We propose a novel heterogeneous hybrid parallelism and hierarchical load balancing strategy for efficient training of ERNIE 4.5 models. By using intra-node expert parallelism, memory-efficient pipeline scheduling, FP8 mixed-precision training and finegrained recomputation methods, we achieve remarkable pre-training throughput. For inference, we propose *multi-expert parallel collaboration* method and *convolutional code quantization* algorithm to achieve 4-bit/2-bit lossless quantization. Furthermore, we introduce PD disaggregation with dynamic role switching for effective resource utilization to enhance inference performance for ERNIE 4.5 MoE models. Built on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle), ERNIE 4.5 delivers high-performance inference across a wide range of hardware platforms.

3. **Modality-Specific Post-Training:** To meet the diverse requirements of real-world applications, we fine-tuned variants of the pre-trained model for specific modalities. Our LLMs are optimized for general-purpose language understanding and generation. The VLMs focuses on visuallanguage understanding and supports both thinking and non-thinking modes. Each model employed a combination of *Supervised Fine-tuning (SFT)*, *Direct Preference Optimization (DPO)* or a modified reinforcement learning method named *Unified Preference Optimization (UPO)* for post-training.

## Model Overview

ERNIE-4.5-0.3B-Base is a text dense Base model. The following are the model configuration details:

| Key            | Value       |
| -------------- | ----------- |
| Modality       | Text        |
| Training Stage | Pretraining |
| Params         | 0.36B       |
| Layers         | 18          |
| Heads(Q/KV)    | 16 / 2      |
| Context Length | 131072      |

## Quickstart

### Model Finetuning with ERNIEKit

[ERNIEKit](https://github.com/PaddlePaddle/ERNIE) is a training toolkit based on PaddlePaddle, specifically designed for the ERNIE series of open-source large models. It provides comprehensive support for scenarios such as instruction fine-tuning (SFT, LoRA) and alignment training (DPO), ensuring optimal performance.

Usage Examples:

```bash
# Download Model
huggingface-cli download baidu/ERNIE-4.5-0.3B-Base-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Base-Paddle
# SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml model_name_or_path=baidu/ERNIE-4.5-0.3B-Base-Paddle
# DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_8k.yaml model_name_or_path=baidu/ERNIE-4.5-0.3B-Base-Paddle
```

For more detailed examples, including SFT with LoRA, multi-GPU configurations, and advanced scripts, please refer to the examples folder within the [ERNIEKit](https://github.com/PaddlePaddle/ERNIE) repository.

### FastDeploy Inference

Service deployment can be quickly completed using FastDeploy in the following command. For more detailed usage instructions, please refer to the [FastDeploy Repository](https://github.com/PaddlePaddle/FastDeploy).

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

### Using `transformers` library

**Note**: Before using the model, please ensure you have the `transformers` library installed. (version 4.50.0 or higher)

The following contains a code snippet illustrating how to use the model generate content based on given inputs.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-4.5-0.3B-Base-PT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

prompt = "Large language model is"
model_inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024
)
result = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
print("result:", result)
```

### vLLM inference

vLLM is currently being adapted, priority can be given to using our forked repository [vllm](https://github.com/CSWYF3634076/vllm/tree/ernie). We are working with the community to fully support ERNIE4.5 models, stay tuned.

```bash
vllm serve baidu/ERNIE-4.5-0.3B-Base-PT --trust-remote-code
```

## License

The ERNIE 4.5 models are provided under the Apache License 2.0. This license permits commercial use, subject to its terms and conditions. Copyright (c) 2025 Baidu, Inc. All Rights Reserved.

## Citation

If you find ERNIE 4.5 useful or wish to use it in your projects, please kindly cite our technical report:

```bibtex
@misc{ernie2025technicalreport,
      title={ERNIE 4.5 Technical Report},
      author={Baidu ERNIE Team},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={}
}
```
