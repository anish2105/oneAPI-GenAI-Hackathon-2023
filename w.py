from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
import torch

model_id = "Writer/camel-5b-hf"


tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id ,torch_dtype=torch.bfloat16
)

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)

print(local_llm('What is the capital of France? '))