import os
import argparse
import json
from tqdm import tqdm
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Code generation using a fine-tuned language model.")
    parser.add_argument("--cuda_device", type=str, help="CUDA device to use, e.g., '0' or '0,1'")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-7b-instruct-v1.5", help="Model ID to use")
    parser.add_argument("--data_type", type=str, help="Data type to use (e.g., float16, float32)")
    parser.add_argument("--peft_config", type=str, help="PEFT config file")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample from the model")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for nucleus sampling")
    parser.add_argument("--input_file", type=str, default="./hint_human_act_as_expert.jsonl", help="Input file")
    parser.add_argument("--output_file", type=str, help="Output file")
    parser.add_argument("--n", type=int, default=1, help="Number of completions for each problem to generate")
    return parser.parse_args()

def setup_environment(args):
    os.environ['TOKENIZERS_PARALLELISM'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    data_type = getattr(torch, args.data_type)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=data_type,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, model_id=args.peft_config)
    model.eval()
    return model, tokenizer

def generate_completion(model, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_length = inputs["input_ids"].size(1)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs.sequences[0][input_length:], skip_special_tokens=True)

def post_process_generated_code(completion: str):
    pattern_code_env = r"```verilog\n(.*?)\n```"
    match = re.search(pattern_code_env, completion, re.DOTALL)
    verilog_code = match.group(1) if match else completion

    pattern_module_def = r"module\s+\w+\s*\(.*?\);\n\n"
    return re.sub(pattern_module_def, '', verilog_code, flags=re.DOTALL)

def process_input_file(args, model, tokenizer):
    with open(args.input_file, "r", encoding="utf-8") as input_file, \
         open(args.output_file, "w", encoding="utf-8") as output_file:
        for line in tqdm(input_file, desc="Processing inputs"):
            data = json.loads(line)
            prompt, task_id = data["hint"], data["task_id"]
            for _ in range(args.n):
                generated_code = generate_completion(model, tokenizer, prompt, args)
                completion = post_process_generated_code(generated_code)
                output = {"task_id": task_id, "completion": completion}
                json.dump(output, output_file)
                output_file.write("\n")

def main():
    args = parse_arguments()
    setup_environment(args)
    model, tokenizer = load_model_and_tokenizer(args)
    process_input_file(args, model, tokenizer)

if __name__ == "__main__":
    main()