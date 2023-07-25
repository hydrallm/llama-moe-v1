import torch
import os
import time


from peft_model import PeftModel
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    LlamaTokenizer,
    Pipeline
)

from bigmodelvis import Visualization
# expert to example prompt
PROMPT_DICT = {
    'math':  """
If the equation of a line is y = 2x + 3, find the coordinates of the point where the line intersects the y-axis.
""",
    'science': """
What is the probability of finding a particle with a given energy in a one-dimensional infinite square well potential when the potential width is 2 nm and the particle has a mass of 5x10^-26 kg? Use the Schrödinger equation to solve for the allowed energy states and their wave functions."
""",
    'sharegpt': """
    Hey GPT I am applying to transfer to the following colleges: UT Austin, NYU, Boston University, Cornell University. I will need your help writing good essays and provide you with the necessary prompts but first I'd like to know what information about me would help you
"""
}

# expert to lora adapter
ADAPTER_DICT = {
    'sharegpt': 'HydraLM/sharegpt_llama2-7b',
    'math': 'HydraLM/camel_math_llama2-7b',
    'science': 'HydraLM/camel_science_llama2-7b'
}


class InferenceModel:

    def __init__(self, model: PeftModel):
        self.model = model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2').encode
        self.cursor = 'cursor'

        self.adapter_names = ['sharegpt', 'math', 'science']
        self.adapter_dir = 'adapters'
        self.adapter_weights = []

        for name in self.adapter_names:
            self.model.load_adapter(ADAPTER_DICT[name], name)

        self.model.delete_adapter('default')

        # Visualization(self.model).structure_graph()

        # start = time.time()
        # self.model.add_weighted_adapter(self.adapter_names, [0.3, 0.3, 0.4], self.cursor, combination_type='svd')
        # end = time.time()

        # print(end - start)

    def __call__(prompt: str) -> str:
        """
        Returns generations from mixture of LoRA experts, weighted based on 
        cosine similarity from PROMPT_DICT
        """

        # dict of name -> weight (add up to 1)
        weights = self.compute_weights(prompt)

        for name, value in weights:
            self.adjust_alphas()


if __name__ == '__main__':
    config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        # target_modules='all',
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_auth_token=True,
        # load_in_8bit=True,
    )

    model = PeftModel.from_pretrained(model,
                                      'HydraLM/sharegpt_llama2-7b',
                                      peft_config=config,
                                      adapter_name='default'
                                      )

    model.update_alphas({'default': 3})
    model.update_alphas({'default': 0})
    model.update_alphas({'default': 16})

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    prompt = 'hello world'

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, do_sample=True,
                            top_p=0.95, top_k=0, max_new_tokens=10)

    print(output)
