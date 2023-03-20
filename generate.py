import sys

import torch
from peft import PeftModel
import transformers

from finetune import generate_prompt

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

weights = 'lora-gesture'
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        weights,
        torch_dtype=torch.float16
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        weights,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        weights,
        device_map={"": device},
    )

model.eval()


def evaluate(
        input_sequence,
        max_new_tokens=256,
        temperature=0,
        top_p=0.75,
        top_k=1,
        num_beams=1,
        **kwargs,
):
    prompt = generate_prompt(input_sequence)
    inputs = tokenizer(prompt, return_tensors="pt")

    stop_sequence = tokenizer("<")["input_ids"]

    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=stop_sequence
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print("Output: ")
    print(output)
    return output.split("<response>")[-1].strip()


if __name__ == "__main__":
    for instruction in sys.argv[1:]:
        print("Input:", instruction)
        print("Response:", evaluate(instruction))
        print()