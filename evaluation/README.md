## Introduction
We provide a test script to evaluate the performance of OriGen on the Verilog-Eval Human benchmark.

Before running the script, please install the required dependencies:
```bash
cd ..
conda create -n origen python=3.11
conda activate origen
pip install -r requirements.txt
```

## Usage
For simple evaluation, you can run the following command:
```bash
python generate_lora.py \
 --cuda_device "0" \
 --data_type "float16" \
 --peft_config "henryen/OriGen" \
 --output_file "../results/origen.jsonl"
```
The default setting for `do_sample` is False to ensure reproducibility. You can find the reference result in `../results/reference_verilog.jsonl`. The completions you generate will be the same as the reference result.

To evaluate the completions, please install [verilog-eval](https://github.com/NVlabs/verilog-eval) first. You might need to checkout to release/1.0.0 branch.

Then run the following command to evaluate the completions:
```bash
evaluate_functional_correctness path/to/origen.jsonl  --problem_file data/VerilogEval_Human.jsonl
```

The output should be :
```bash
{'pass@1': 0.5384615384615384}
```

For more complex evaluation, you can run the command as follows:
```bash
python generate_lora.py \
 --cuda_device "0" \
 --data_type "float16" \
 --peft_config "henryen/OriGen" \
 --output_file "../results/origen.jsonl" \
 --do_sample True \
 --temperature 0.7 \
 --top_p 0.95 \
 --n 5
```

## Details 
The prompt file `hint_human_act_as_expert.jsonl` is generated from verilog-eval v1.0.0. For each problem in Human benchmark, we concat its prompt and detail_description using verilog specific chat template.

The template is as follows:
```
### Instruction: Please act as a professional Verilog designer and provide Verilog code based on the given description. {detailed description}
### Response: {prompt(module head)}
```