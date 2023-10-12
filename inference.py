from transformers import GenerationConfig, TextStreamer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch
import os
import json
import pandas as pd

torch_dtype = torch.bfloat16
device_map = {"": 0}

# model_id = "NousResearch/Llama-2-7b-hf"
model_id = "minhbui/viettel_v3.2"
# peft_id = "PEFT" 

tokenizer = LlamaTokenizer.from_pretrained(model_id)
model_gen = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    device_map=device_map,
    torch_dtype=torch_dtype
)


def get_answer(prompt, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model_gen.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model_gen.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )

    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

lst_answer = []

from datasets import load_dataset
import re
def delete_citation(example):
    pattern = r"\[\s*\d+\s*\]"  # Regular expression pattern to match the desired patterns
    new_string = re.sub(pattern, "", example)
    return new_string

def generate_file(filename):
    lst_prompt = load_dataset("json", data_files=filename, split="train")
    lst_response = []
    for prompt in lst_prompt:
        out = get_answer(prompt['prompt'])
        lst_response.append(out)

    df_submit = pd.DataFrame({"answer_predict":lst_response, "quest_id":[x for x in range(len(lst_response))]})
    df_submit['answer_predict'] = df_submit['answer_predict'].map(lambda x : delete_citation(x))
    df_submit.to_csv("results.csv")



def main():
    parser = argparse.ArgumentParser(prog='keywords')
    parser.add_argument('--filename', help='filename to run generate results')
    args = parser.parse_args()
    generate_file(args['filename'])


if __name__ == "__main__":
    main()
