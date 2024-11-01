import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship
from huggingface_hub import login

load_dotenv()
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
hf_token = os.getenv("HF_TOKEN") 


login(token=hf_token)

df = pd.read_json("hf://datasets/fhai50032/Symptoms_to_disease_7k/Symptoms_to_disease_7k.json")
df['query'] = df['query'].apply(lambda x: x.split('Patient:I may have ', 1)[-1])
df['response'] = df['response'].apply(lambda x: x.split('You may have ', 1)[-1])
df['text'] = df.apply(lambda row: f'Symptoms: {row["query"]} Disease: {row["response"]}', axis=1)
df.drop(columns=['query','response'], inplace=True)
df = df[df['text'].apply(lambda x: len(x.split()) <= 30)]

graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

for _, row in df.iterrows():
    text = row['text']
    symptoms_text = text.split("Symptoms: ")[1].split(" Disease: ")[0]
    disease_text = text.split(" Disease: ")[1]
    disease_node = Node("Disease", name=disease_text.strip())
    graph.merge(disease_node, "Disease", "name")
    
    symptoms = symptoms_text.replace('.', ',').split(',')
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom: 
            symptom_node = Node("Symptom", name=symptom)
            graph.merge(symptom_node, "Symptom", "name")
            relationship = Relationship(disease_node, "HAS_SYMPTOM", symptom_node)
            graph.merge(relationship)
