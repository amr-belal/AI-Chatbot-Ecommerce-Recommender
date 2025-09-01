import pandas as pd 
from sentence_transformers import SentenceTransformer
from typing import List,Dict 

data = pd.read_csv("data\ecommerce_data.csv")

print(data.head())


sentenc_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = sentenc_model.encode(data.tolist())

    