from transformers import GenerationConfig, TextStreamer
def get_answer(prompt, max_new_tokens=768):
  input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
	model.eval()
	with torch.no_grad():
	    generation_config = GenerationConfig( repetition_penalty=1.1,max_new_tokens=max_new_tokens,temperature=0.2,top_p=0.95,top_k=40,
	   # bos_token_id=tokenizer.bos_token_id,
	         # eos_token_id=tokenizer.eos_token_id,
	        # eos_token_id=0, # for open-end generation.
	        pad_token_id=tokenizer.pad_token_id,
	        do_sample=True,
	        use_cache=True,
	        return_dict_in_generate=True,
	        output_attentions=False,
	        output_hidden_states=False,
	        output_scores=False,
	        )
	    streamer = TextStreamer(tokenizer, skip_prompt=True)
	    generated = model.generate(inputs=input_ids,generation_config=generation_config,streamer=streamer,
	    )
	    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
	    output = tokenizer.batch_decode(gen_tokens)[0]
	    output = output.split(tokenizer.eos_token)[0]
	    return output.strip()
