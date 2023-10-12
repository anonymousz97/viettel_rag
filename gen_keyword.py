from enum import Enum
from typing import Generator, Optional, Union
from transformers import GenerationConfig, TextStreamer
from copy import deepcopy
from transformers import GenerationConfig, TextStreamer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch
import os
import json
import pandas as pd
import argparse

torch_dtype = torch.bfloat16
device_map = {"": 0}

# model_id = "NousResearch/Llama-2-7b-hf"
model_id = "minhbui/viettel_v3.2"
# peft_id = "PEFT" 

tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    device_map=device_map,
    torch_dtype=torch_dtype
)

class PromptStyle(Enum):
    """
    Enum for prompt styles
    """

    INSTRUCT = "instruct"
    CHAT = "chat"
    CHATML = "chatml"

class AlpacaPrompter:
    """
    Base class for alpaca prompters
    """

    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    system_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    system_format: str = "{system}"
    turn_format: str
    turn_no_input_format: str
    prompt_style: Optional[PromptStyle] = None

    def __init__(self, prompt_style=PromptStyle.INSTRUCT.value):
        self.prompt_style = prompt_style if prompt_style else PromptStyle.INSTRUCT.value
        self.match_prompt_style()

    def match_prompt_style(self):
        # pylint: disable=duplicate-code
        if self.prompt_style == PromptStyle.INSTRUCT.value:
            self.turn_format = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            self.turn_no_input_format = (
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            self.system_format = "### System:\n{system}\n\n"
        if self.prompt_style == PromptStyle.CHAT.value:
            self.turn_format = "USER: {instruction}\n{input}\nASSISTANT:"
            self.turn_no_input_format = "USER: {instruction}\nASSISTANT:"
            self.system_format = "SYSTEM: {system}\n"
        if self.prompt_style == PromptStyle.CHATML.value:
            self.turn_format = "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"
            self.turn_no_input_format = (
                "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            )
            self.system_format = "<|im_start|>system\n{system}<|im_end|>\n"

    def build_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = (
                self.system_format.format(system=self.system_prompt)
                if self.system_prompt
                else ""
            ) + self.turn_format.format(instruction=instruction, input=input)
        else:
            res = (
                self.system_format.format(system=self.system_no_input_prompt)
                if self.system_prompt
                else ""
            ) + self.turn_no_input_format.format(instruction=instruction)
        if output:
            res = f"{res}{output}"
        yield res
        
def gen_keyword_template1(question):
    instruction = "You are an AI assistant that follows instruction extremely well. Help as much as you can."
    input_template = (
        "Liệt kê ra tất cả các từ khóa quan trọng trích ra từ câu sau. "
        "Đặc biệt chú ý đến các tên riêng và các mốc thời gian."
        "Lưu ý các từ khóa chỉ được lấy từ câu được cho bên dưới, không tự ý tìm từ bên ngoài."
        "\nCâu: {question}"
    )
    
    input = input_template.format(question=question)
    
    prompter = AlpacaPrompter(prompt_style=PromptStyle.INSTRUCT.value)
    prompt = prompter.build_prompt(instruction=input, input="", output="Từ khoá:").__next__()
    return prompt

def gen_keyword_template2(question):
    instruction = "You are an AI assistant that follows instruction extremely well. Help as much as you can."
    input_template = """Liệt kê ra các cụm từ quan trọng xuất hiện trong câu sau. \nCâu: {question}"""
    input = input_template.format(question=question)
    
    prompter = AlpacaPrompter(prompt_style=PromptStyle.INSTRUCT.value)
    prompt = prompter.build_prompt(instruction=input, input="", output="Từ khoá:").__next__()
    return prompt

from transformers import GenerationConfig, TextStreamer


def get_answer(prompt, max_new_tokens=256, repetition_penalty=1.13):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=repetition_penalty,
            # repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
            # temperature=0.2,
            # top_p=0.95,
            # top_k=20,
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
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )
        
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

import pandas as pd
import json 
from tqdm import tqdm

def gen_keyword(filename, outfile):
    private_questions = pd.read_csv(filename)["question"].to_list()
    keywords = []
    for question in tqdm(private_questions):
        prompt1 = gen_keyword_template1(question)
        key_string1 = get_answer(prompt1)
        prompt2 = gen_keyword_template2(question)
        key_string2 = get_answer(prompt2)
        keywords.append({
            "question": question,
            "key_strings": [key_string1, key_string2]
        })

    with open(outfile, "w", encoding="utf8") as file:
        json.dump(keywords, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(prog='keywords')
    parser.add_argument('--filename', help='filename to run generate keywords')
    parser.add_argument('--outfile', help='filename keyword output')
    args = parser.parse_args()
    gen_keyword(args['filename'], args['outfile'])


if __name__ == "__main__":
    main()
