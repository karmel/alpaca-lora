import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_dataset
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 2  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 300  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 20 # TODO: this only works for the mini-test.


def generate_prompt_from_dict(data_point):
    return generate_prompt(data_point['input'], output=data_point['output'])


def generate_prompt(gesture, output=None):
    prompt = ("<instruction>These are points made by human fingers touching an x, y plane. "
              "Determine what gesture was made.")
    prompt = prompt + f"""
<gesture>
{gesture}

<response>
"""
    if output: prompt = prompt + output
    return prompt


def tokenize(prompt, tokenizer):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def run_finetuning(path_to_ckpt, data_dir):
    model = LlamaForCausalLM.from_pretrained(
        path_to_ckpt,
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        path_to_ckpt, add_eos_token=True
    )

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    data_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)
                  if fname[-5:] == '.json']
    print('Data files: ')
    print(data_files)
    data = load_dataset("json", data_files=data_files)

    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]

    train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt_from_dict(x), tokenizer))
    val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt_from_dict(x), tokenizer))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir="lora-gesture",
            save_total_limit=3,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train()

    model.save_pretrained("lora-gesture")


if __name__ == '__main__':
    path_to_ckpt: str = "decapoda-research/llama-7b-hf"
    if "PATH_TO_CKPT" in os.environ:
        path_to_ckpt = os.environ["PATH_TO_CKPT"]

    if "DATA_DIR" in os.environ:
        data_dir: str = os.environ["DATA_DIR"]
    else:
        raise ValueError('No env variable $DATA_DIR found. Please set $DATA_DIR.')

    run_finetuning(path_to_ckpt, data_dir)