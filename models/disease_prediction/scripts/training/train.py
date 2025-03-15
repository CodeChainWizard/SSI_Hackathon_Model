import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class DiseaseDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        # Load dataset
        self.data = pd.read_csv(file_path)

        # Ensure required columns exist
        required_columns = ["fever", "cough", "shortness_of_breath", "infection_risk"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Fill missing values with "Unknown" or a neutral value
        self.data.fillna({"fever": "Unknown", "cough": "Unknown", "shortness_of_breath": "Unknown", "infection_risk": "Low"}, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Convert symptoms into a textual format
        row = self.data.iloc[index]
        text = f"Fever: {row['fever']}, Cough: {row['cough']}, Shortness of Breath: {row['shortness_of_breath']}"

        # Label Mapping
        label_mapping = {"Low": 0, "Medium": 1, "High": 2}
        label = label_mapping.get(row["infection_risk"], 0)  # Default to "Low" if unknown

        # Tokenize input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long),
        )

# 🚀 Load pre-trained BERT tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

dataset = DiseaseDataset("/Users/yashcomputers/Desktop/SSI_Hackathon/AI-Traffic-Disease-Prediction/data/disease/disease_data.csv", tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "checkpoints/disease_bert_model.pth")
print("✅ Disease Model Training Complete & Saved!")

model.load_state_dict(torch.load("checkpoints/disease_bert_model.pth", map_location=device))
model.eval()

sample_text = "Fever: 38.5, Cough: Severe, Shortness of Breath: Yes"
encoding = tokenizer(sample_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

with torch.no_grad():
    output = model(input_ids, attention_mask=attention_mask)
    predicted_label = torch.argmax(output.logits, dim=1).item()

label_names = ["Low", "Medium", "High"]
print(f"🚀 Predicted Infection Risk: {label_names[predicted_label]}")
