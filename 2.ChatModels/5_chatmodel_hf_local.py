from pyexpat import model
from unittest import result

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'C:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="SupraLabs/Supra2-100M-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=30
    )
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Who is Elon Musk?")

print(result.content)