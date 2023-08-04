from threading import Thread
from typing import Iterator
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = 'meta-llama/Llama-2-7b-chat-hf'

class InferlessPythonModel:
    def get_prompt(self, message, chat_history,
               system_prompt):
        texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        texts.append(f'{message.strip()} [/INST]')
        return ''.join(texts)

    
    def get_input_token_length(self, message, chat_history, system_prompt):
        prompt = self.get_prompt(message, chat_history, system_prompt)
        input_ids = self.tokenizer([prompt], return_tensors='np')['input_ids']
        return input_ids.shape[-1]


    def run_function(self, message,
        chat_history,
        system_prompt,
        max_new_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        top_k=5):
        print("Starting Time -->", datetime.now().strftime("%H:%M:%S"))
        prompt = self.get_prompt(message, chat_history, system_prompt)
        print("Prompt Finished -->", datetime.now().strftime("%H:%M:%S"))
        inputs = self.tokenizer([prompt], return_tensors='pt').to('cuda')
        print("Tokenizer Finished -->", datetime.now().strftime("%H:%M:%S"))

        streamer = TextIteratorStreamer(self.tokenizer,
                                        timeout=10.,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        
        print("Streamer Finished -->", datetime.now().strftime("%H:%M:%S"))
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        print("Thread Started -->", datetime.now().strftime("%H:%M:%S"))

        outputs = ''
        for text in streamer:
            outputs += text

        print("Outputs Finished -->", datetime.now().strftime("%H:%M:%S"))

        return outputs


    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token='<your_token>')

        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map='auto',
                use_auth_token='<your_token>'
            )
        else:
            self.model = None

    def infer(self, inputs):
        message = inputs['message']
        chat_history = inputs['chat_history'] if 'chat_history' in inputs else []
        system_prompt = inputs['system_prompt'] if 'system_prompt' in inputs else ''
        result = self.run_function(
            message=message,
            chat_history=chat_history,
            system_prompt=system_prompt,
        )
        return {"generated_text": result}

    def finalize(self):
        self.tokenizer = None
        self.model = None
