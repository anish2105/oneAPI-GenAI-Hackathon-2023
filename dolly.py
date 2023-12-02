from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
from instruct_pipeline import InstructionTextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)



pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=3000,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.2,
)

local_llm = HuggingFacePipeline(pipeline=pipe)

print(local_llm('What is the capital of France? '))