import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from typing import List, Tuple, Dict
import pickle
import os

class NameNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize and encode
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class BERTNameNER(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-uncased'):
        super(BERTNameNER, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

class NameNERProcessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_map = {
            'O': 0,      # Outside
            'B-TITLE': 1, # Beginning of Title
            'B-FNAME': 2, # Beginning of First Name
            'B-MNAME': 3, # Beginning of Middle Name
            'B-SNAME': 4, # Beginning of Surname
            'B-SUFFIX': 5, # Beginning of Suffix
            'I-TITLE': 6,  # Inside Title
            'I-FNAME': 7,  # Inside First Name
            'I-MNAME': 8,  # Inside Middle Name
            'I-SNAME': 9,  # Inside Surname
            'I-SUFFIX': 10 # Inside Suffix
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def create_bio_labels(self, text: str, title: str, forename: str, 
                         middle_name: str, surname: str, suffix: str) -> List[int]:
        """Create BIO labels for the given text and components"""
        # Clean and prepare components
        components = {
            'TITLE': str(title).strip() if pd.notna(title) else '',
            'FNAME': str(forename).strip() if pd.notna(forename) else '',
            'MNAME': str(middle_name).strip() if pd.notna(middle_name) else '',
            'SNAME': str(surname).strip() if pd.notna(surname) else '',
            'SUFFIX': str(suffix).strip() if pd.notna(suffix) else ''
        }
        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text.lower())
        labels = [self.label_map['O']] * len(tokens)  # Initialize all as 'O'
        
        # Add special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels = [self.label_map['O']] + labels + [self.label_map['O']]
        
        # Find and label each component
        text_lower = text.lower()
        for comp_type, comp_value in components.items():
            if comp_value and len(comp_value) > 0:
                comp_tokens = self.tokenizer.tokenize(comp_value.lower())
                if comp_tokens:
                    # Find the position of this component in the tokenized text
                    for i in range(len(tokens) - len(comp_tokens) + 1):
                        if tokens[i:i+len(comp_tokens)] == comp_tokens:
                            # Label the first token as B- and rest as I-
                            labels[i] = self.label_map[f'B-{comp_type}']
                            for j in range(1, len(comp_tokens)):
                                if i + j < len(labels):
                                    labels[i + j] = self.label_map[f'I-{comp_type}']
                            break
        
        # Pad or truncate to max_length
        max_length = 128
        if len(labels) > max_length:
            labels = labels[:max_length]
        else:
            labels.extend([self.label_map['O']] * (max_length - len(labels)))
            
        return labels
    
    def load_data(self, csv_file: str) -> Tuple[List[str], List[List[int]]]:
        """Load and process the CSV data"""
        df = pd.read_csv(csv_file)
        
        texts = []
        all_labels = []
        
        for _, row in df.iterrows():
            text = row['text']
            title = row.get('TITLE', '')
            forename = row.get('FORENAME', '')
            middle_name = row.get('MIDDLE NAME', '')
            surname = row.get('SURNAME', '')
            suffix = row.get('SUFFIX', '')
            
            labels = self.create_bio_labels(text, title, forename, middle_name, surname, suffix)
            
            texts.append(text)
            all_labels.append(labels)
        
        return texts, all_labels

class NameNERTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train(self, train_loader, val_loader, num_epochs=5, learning_rate=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                logits = self.model(input_ids, attention_mask)
                
                # Reshape for loss calculation
                loss = criterion(logits.view(-1, self.model.num_labels), labels.view(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            val_accuracy = self.evaluate(val_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Training Loss: {avg_loss:.4f}')
            print(f'  Validation Accuracy: {val_accuracy:.4f}')
    
    def evaluate(self, data_loader):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                
                # Only count non-padding tokens
                mask = attention_mask.bool()
                correct_predictions += (predictions == labels).masked_select(mask).sum().item()
                total_predictions += mask.sum().item()
        
        self.model.train()
        return correct_predictions / total_predictions if total_predictions > 0 else 0
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_labels': self.model.num_labels,
                'model_name': 'bert-base-uncased'
            }
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['model_config']

class NameNERPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.processor = NameNERProcessor()
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['model_config']
        
        self.model = BERTNameNER(config['num_labels'], config['model_name'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
    
    def predict(self, text: str) -> Dict[str, str]:
        """Predict name components from text"""
        # Tokenize
        encoding = self.processor.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        # Decode predictions
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_labels = [self.processor.reverse_label_map[pred.item()] 
                          for pred in predictions[0]]
        
        # Extract entities
        entities = self.extract_entities(tokens, predicted_labels)
        return entities
    
    def extract_entities(self, tokens: List[str], labels: List[str]) -> Dict[str, str]:
        """Extract named entities from tokens and labels"""
        entities = {
            'TITLE': '',
            'FORENAME': '',
            'MIDDLE_NAME': '',
            'SURNAME': '',
            'SUFFIX': ''
        }
        
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    entity_text = self.processor.tokenizer.convert_tokens_to_string(current_tokens)
                    entities[current_entity] = entity_text.strip()
                
                # Start new entity
                current_entity = label[2:]  # Remove 'B-'
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                current_tokens.append(token)
                
            else:
                # Save current entity and reset
                if current_entity and current_tokens:
                    entity_text = self.processor.tokenizer.convert_tokens_to_string(current_tokens)
                    entities[current_entity] = entity_text.strip()
                current_entity = None
                current_tokens = []
        
        # Save final entity
        if current_entity and current_tokens:
            entity_text = self.processor.tokenizer.convert_tokens_to_string(current_tokens)
            entities[current_entity] = entity_text.strip()
        
        return entities

def main():
    """Main training function"""
    # Configuration
    CSV_FILE = 'generated_names.csv'  # Your CSV file path
    MODEL_SAVE_PATH = 'bert_name_ner_model.pth'
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize processor
    processor = NameNERProcessor()
    
    # Load data
    print("Loading data...")
    texts, labels = processor.load_data(CSV_FILE)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = NameNERDataset(train_texts, train_labels, processor.tokenizer)
    val_dataset = NameNERDataset(val_texts, val_labels, processor.tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    num_labels = len(processor.label_map)
    model = BERTNameNER(num_labels)
    
    # Initialize trainer
    trainer = NameNERTrainer(model, device)
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # Save model
    trainer.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()