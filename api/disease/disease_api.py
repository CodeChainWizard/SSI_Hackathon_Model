from fastapi import FastAPI
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.load_state_dict(torch.load("../../models/disease_prediction/checkpoints/disease_bert_model.pth"))
model.eval()

CSV_FILE_PATH = "symptoms_data.csv"

@app.post("/predict_disease")
def predict_disease(data: dict):
    df = pd.DataFrame([data])
    df.to_csv(CSV_FILE_PATH, index=False)

    loaded_df = pd.read_csv(CSV_FILE_PATH)
    text = loaded_df["symptoms"].iloc[0]
    
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**inputs).logits.argmax().item()

    return {"predicted_disease_risk": output}
