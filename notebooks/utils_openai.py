from dotenv import load_dotenv
import os
# from openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from typing import List, Optional
import numpy as np

load_dotenv()


# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model=os.getenv("emb_model"),
    deployment_name="embedding-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)


def get_embedding(text, embed_model=embed_model):
   return embed_model.get_text_embedding(text)


def get_embedding_score(text: str, other_texts: List) -> dict:
    text_emb = get_embedding(text)
    cosine_similarities = np.dot(text_emb, np.array(other_texts).T)
    return cosine_similarities
