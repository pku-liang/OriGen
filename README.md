# OriGen: Enhancing RTL Code Generation with Code-to-Code Augmentation and Self-Reflection

## 📖 Introduction

OriGen is a fine-tuned LoRA model designed for Verilog code generation. It's built on top of DeepSeek Coder 7B, leveraging datasets generated through code-to-code augmentation and self-reflection techniques.

### 🔧 Models

1. **OriGen**: Focused on Verilog code generation, trained on [origen_dataset_instruction](https://huggingface.co/datasets/henryen/origen_dataset_instruction)
2. **OriGen_Fix**: Specialized in fixing syntax errors in Verilog code, based on OriGen, further trained using [origen_dataset_debug](https://huggingface.co/datasets/henryen/origen_dataset_debug) 

## 🔗 Quick Links

- 🤗 **Hugging Face Models**:
  - [OriGen](https://huggingface.co/henryen/OriGen)
  - [OriGen_Fix](https://huggingface.co/henryen/OriGen_Fix)

- 📊 **Datasets**:
  - [Instruction Dataset](https://huggingface.co/datasets/henryen/origen_dataset_instruction)
  - [Debug Dataset](https://huggingface.co/datasets/henryen/origen_dataset_debug)
  - [Description Dataset](https://huggingface.co/datasets/henryen/origen_dataset_description)

- 📁 **GitHub Repository**: [pku-liang/OriGen](https://github.com/pku-liang/OriGen)

## 🚀 Quick Start

### Environment Setup

```bash
conda create -n origen python=3.11
conda activate origen
pip install -r requirements.txt
```

### Usage Example
Here is an example of how to use the model. Please note that the base model, DeepSeek Coder 7B, is loaded in float16 precision, even though its default precision is bfloat16. This choice is made because our experiments show that Lora trained in float16 outperforms those trained in bfloat16.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from peft import PeftModel

model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
).to("cuda")

model = PeftModel.from_pretrained(model, model_id="henryen/OriGen")
model.eval()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "### Instruction: Please act as a professional Verilog designer. and provide Verilog code based on the given instruction. Generate a concise Verilog module for a 8 bit full adder, don't include any unnecessary code.\n### Response: "

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens=1000, 
    do_sample=False, 
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    streamer=streamer
)
```

### Output

```verilog
module full_adder(
    input [7:0] a,
    input [7:0] b,
    input cin,
    output [7:0] sum,
    output cout
);

assign {cout, sum} = a + b + cin;

endmodule
```

## 📊 Evaluation

We've released scripts for the Verilog-Eval benchmark. For details, please refer to the [evaluation README](./evaluation/README.md).

![OriGen Evaluation Results](figures/evaluation.png)

## 📄 Paper

**arXiv**: [https://arxiv.org/abs/2407.16237](https://arxiv.org/abs/2407.16237)

If you find this work useful, please cite our paper:

```bibtex
@article{2024origen,
  title={OriGen: Enhancing RTL Code Generation with Code-to-Code Augmentation and Self-Reflection},
  author={Cui, Fan and Yin, Chenyang and Zhou, Kexing and Xiao, Youwei and Sun, Guangyu and Xu, Qiang and Guo, Qipeng and Song, Demin and Lin, Dahua and Zhang, Xingcheng and others},
  journal={arXiv preprint arXiv:2407.16237},
  year={2024}
}
```