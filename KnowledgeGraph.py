import os
from dotenv import load_dotenv
import pandas as pd
from py2neo import Graph
from huggingface_hub import login

load_dotenv()
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)

df = pd.read_json("hf://datasets/fhai50032/Symptoms_to_disease_7k/Symptoms_to_disease_7k.json")

df['query'] = df['query'].apply(lambda x: x.split('Patient:I may have ', 1)[-1].replace('_', ' '))
df['response'] = df['response'].apply(lambda x: x.split('You may have ', 1)[-1].replace('_', ' '))
df['text'] = df.apply(lambda row: f'Symptoms: {row["query"]} Disease: {row["response"]}', axis=1)
df.drop(columns=['query', 'response'], inplace=True)
df = df[df['text'].apply(lambda x: len(x.split()) <= 30)]

graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
batch_size = 100
disease_symptom_pairs = []
def execute_batch(pairs):
    graph.run(
        """
        UNWIND $pairs AS pair
        MERGE (d:Disease {name: pair[0]})
        MERGE (s:Symptom {name: pair[1]})
        MERGE (d)-[:HAS_SYMPTOM]->(s)
        """, parameters={"pairs": pairs}
    )

for idx, row in df.iterrows():
    text = row['text']
    symptoms_text = text.split("Symptoms: ")[1].split(" Disease: ")[0]
    disease_text = text.split(" Disease: ")[1].strip()
    symptoms = [symptom.strip() for symptom in symptoms_text.replace('.', ',').split(',') if symptom.strip()]
    for symptom in symptoms:
        disease_symptom_pairs.append((disease_text, symptom))
        print(f'(:Disease {{name: "{disease_text}"}})-[:HAS_SYMPTOM]->(:Symptom {{name: "{symptom}"}})')
    if (idx + 1) % batch_size == 0:
        execute_batch(disease_symptom_pairs)
        disease_symptom_pairs = []
        
if disease_symptom_pairs:
    execute_batch(disease_symptom_pairs)

print("Data upload to Neo4j completed.")
