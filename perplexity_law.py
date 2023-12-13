import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='')
parser.add_argument("--window-size", type=int, default=4096)
parser.add_argument("--data", type=str, default='viettel_raw_data/wiki_raw.json')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model, 
    trust_remote_code=True,
    token='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')

try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        token='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',
        torch_dtype=torch.bfloat16
    )
except Exception as _:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        token='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',
    )
model.tie_weights()
model.eval()

with open(args.data, 'r') as file:
    # Load JSON data from the file
    data = json.load(file)
    samples = data['data']

encodings = tokenizer("\n\n".join(samples), return_tensors="pt")

max_length = args.window_size
stride = 512
seq_len = encodings.input_ids.size(1)
num_chars = len("\n\n".join(samples))
device = "cuda"

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack([nll.to('cuda:0') for nll in nlls]).mean())
print(ppl, seq_len, num_chars)
model_name_to_save = args.model.replace('/', '_')
data_name = args.data[:-4]
with open(f'{data_name}_{model_name_to_save}_{args.window_size}.txt', 'w') as f:
    f.write(str(ppl.item()))
    f.write('\n')
    f.write(', '.join([str(nll.item()) for nll in nlls]))