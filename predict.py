#!/usr/bin/env python3
"""
BERT Name NER Runner Script
Direct execution script - modify variables below as needed
"""

import torch
import pandas as pd
import json
from faker import Faker
from train import NameNERPredictor, NameNERProcessor, BERTNameNER, NameNERTrainer, NameNERDataset
from torch.utils.data import DataLoader

# ================================
# CONFIGURATION VARIABLES - MODIFY THESE AS NEEDED
# ================================

# Training Configuration
TRAIN_CSV_FILE = 'training_data.csv'           # Input CSV for training
MODEL_SAVE_PATH = 'bert_name_ner_model.pth'    # Where to save/load model
NUM_EPOCHS = 5                                  # Training epochs
BATCH_SIZE = 16                                 # Training batch size
LEARNING_RATE = 2e-5                           # Learning rate

# Data Generation Configuration
GENERATE_SAMPLES = 10000                         # Number of samples to generate
GENERATED_CSV_OUTPUT = 'generated_names.csv'   # Output file for generated data

# Prediction Configuration  
PREDICT_INPUT_CSV = 'names_to_predict.csv'     # CSV file with names to predict
PREDICT_OUTPUT_CSV = 'predictions.csv'         # Output file for predictions
EVALUATION_CSV = 'test_names.csv'              # CSV file for model evaluation

# What to run - Set these to True/False to control execution
RUN_GENERATE_DATA = True          # Generate synthetic data
RUN_TRAIN_MODEL = True            # Train the model
RUN_PREDICT_INTERACTIVE = True    # Run interactive prediction mode
RUN_PREDICT_CSV = False           # Predict from CSV file
RUN_EVALUATE = False              # Evaluate model performance

# ================================
# END CONFIGURATION
# ================================

def create_sample_csv():
    """Create a sample CSV file for testing"""
    data = {
        'text': [
            'MR THOMAS SURRAY FISHER',
            'MR CONRAD A FISHER JR',
            'MS SARAH JANE SMITH',
            'DR WILLIAM HENRY JONES III',
            'MRS ELIZABETH MARY BROWN',
            'MISS CATHERINE ROSE WILSON',
            'PROF JOHN MICHAEL DAVIS SR'
        ],
        'TITLE': ['MR', 'MR', 'MS', 'DR', 'MRS', 'MISS', 'PROF'],
        'FORENAME': ['THOMAS', 'CONRAD', 'SARAH', 'WILLIAM', 'ELIZABETH', 'CATHERINE', 'JOHN'],
        'MIDDLE NAME': ['SURRAY', 'A', 'JANE', 'HENRY', 'MARY', 'ROSE', 'MICHAEL'],
        'SURNAME': ['FISHER', 'FISHER', 'SMITH', 'JONES', 'BROWN', 'WILSON', 'DAVIS'],
        'SUFFIX': ['', 'JR', '', 'III', '', '', 'SR']
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_names.csv', index=False)
    print("Created sample_names.csv for testing")
    return 'sample_names.csv'

def train_model(csv_file, model_path='bert_name_ner_model.pth', epochs=5, batch_size=16):
    """Train the BERT model"""
    print(f"Training model with data from {csv_file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize processor
    processor = NameNERProcessor()
    
    # Load data
    print("Loading and processing data...")
    texts, labels = processor.load_data(csv_file)
    print(f"Loaded {len(texts)} samples")
    
    # Split data (80-20 split)
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = NameNERDataset(train_texts, train_labels, processor.tokenizer)
    val_dataset = NameNERDataset(val_texts, val_labels, processor.tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    num_labels = len(processor.label_map)
    model = BERTNameNER(num_labels)
    
    # Initialize trainer
    trainer = NameNERTrainer(model, device)
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs, 2e-5)
    
    # Save model
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")

def predict_from_csv(input_file, model_path='bert_name_ner_model.pth', output_file='predictions.csv'):
    """Predict name components for texts from CSV file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load predictor
    predictor = NameNERPredictor(model_path, device)
    
    # Load input data
    df = pd.read_csv(input_file)
    
    results = []
    for _, row in df.iterrows():
        text = row['text'] if 'text' in row else str(row.iloc[0])
        prediction = predictor.predict(text)
        
        result = {
            'text': text,
            'predicted_title': prediction['TITLE'],
            'predicted_forename': prediction['FORENAME'], 
            'predicted_middle_name': prediction['MIDDLE_NAME'],
            'predicted_surname': prediction['SURNAME'],
            'predicted_suffix': prediction['SUFFIX']
        }
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return results_df

def evaluate_model(test_csv, model_path='bert_name_ner_model.pth'):
    """Evaluate the model on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load predictor
    predictor = NameNERPredictor(model_path, device)
    
    # Load test data
    df = pd.read_csv(test_csv)
    
    correct_predictions = {'TITLE': 0, 'FORENAME': 0, 'MIDDLE_NAME': 0, 'SURNAME': 0, 'SUFFIX': 0}
    total_predictions = len(df)
    
    print("Evaluating model...")
    for _, row in df.iterrows():
        text = row['text']
        prediction = predictor.predict(text)
        
        # Compare predictions with ground truth
        ground_truth = {
            'TITLE': str(row.get('TITLE', '')).strip(),
            'FORENAME': str(row.get('FORENAME', '')).strip(),
            'MIDDLE_NAME': str(row.get('MIDDLE NAME', '')).strip(),
            'SURNAME': str(row.get('SURNAME', '')).strip(),
            'SUFFIX': str(row.get('SUFFIX', '')).strip()
        }
        
        for key in correct_predictions:
            if prediction[key].upper() == ground_truth[key].upper():
                correct_predictions[key] += 1
    
    # Calculate and display accuracy
    print("\nEvaluation Results:")
    print("-" * 40)
    for key, correct in correct_predictions.items():
        accuracy = (correct / total_predictions) * 100
        print(f"{key:12}: {accuracy:.2f}% ({correct}/{total_predictions})")
    
    overall_accuracy = sum(correct_predictions.values()) / (total_predictions * 5) * 100
    print(f"{'Overall':12}: {overall_accuracy:.2f}%")

def predict_interactive(model_path='bert_name_ner_model.pth'):
    """Interactive mode for testing individual names"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        predictor = NameNERPredictor(model_path, device)
        print("BERT Name NER - Interactive Mode")
        print("Enter names to extract components (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            text = input("\nEnter name: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
                
            if not text:
                continue
                
            try:
                result = predictor.predict(text)
                
                print(f"\nInput: {text}")
                print("Extracted components:")
                print(f"  Title:       {result['TITLE']}")
                print(f"  First Name:  {result['FORENAME']}")
                print(f"  Middle Name: {result['MIDDLE_NAME']}")
                print(f"  Surname:     {result['SURNAME']}")
                print(f"  Suffix:      {result['SUFFIX']}")
                
            except Exception as e:
                print(f"Error processing name: {e}")
                
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

def generate_uk_names_dataset_faker(num_samples=2000, output_file='generated_names.csv'):
    """Generate synthetic UK names dataset using Faker for better quality"""
    import random
    
    # Initialize Faker with UK locale
    fake_uk = Faker('en_GB')  # UK English
    fake_us = Faker('en_US')  # For additional diversity
    
    # UK-specific titles with weights
    uk_titles = [
        ('MR', 40), ('MRS', 25), ('MS', 20), ('MISS', 8), 
        ('DR', 4), ('PROF', 1), ('SIR', 0.5), ('LADY', 0.5), 
        ('LORD', 0.5), ('REV', 0.5)
    ]
    
    # Create weighted title list
    weighted_titles = []
    for title, weight in uk_titles:
        weighted_titles.extend([title] * int(weight * 10))
    
    # UK-specific suffixes
    uk_suffixes = ['', '', '', '', '', '', '', '', '', 'JR', 'SR', 'II', 'III', 'IV']
    
    generated_data = []
    
    print(f"Generating {num_samples} UK name samples using Faker...")
    
    for i in range(num_samples):
        # Use faker to generate names
        if random.random() < 0.5:  # 50% male names
            first_name = fake_uk.first_name_male().upper()
            title = random.choice(['MR', 'DR', 'PROF', 'SIR', 'LORD', 'REV'])
        else:  # 50% female names  
            first_name = fake_uk.first_name_female().upper()
            title = random.choice(['MRS', 'MS', 'MISS', 'DR', 'PROF', 'LADY'])
        
        # Sometimes use weighted title selection
        if random.random() < 0.8:
            title = random.choice(weighted_titles)
            
        # Generate surname using faker
        surname = fake_uk.last_name().upper()
        
        # Generate middle name (70% chance)
        middle_name = ''
        if random.random() < 0.7:
            if random.random() < 0.3:  # 30% chance of initial
                middle_name = random.choice('ABCDEFGHIJKLMNPRSTW')
            else:  # 70% chance of full middle name
                middle_name = random.choice([
                    fake_uk.first_name().upper(),
                    fake_us.first_name().upper()
                ])
        
        # Add suffix (10% chance)
        suffix = random.choice(uk_suffixes)
        if suffix == '' and random.random() < 0.05:  # 5% chance of less common suffixes
            suffix = random.choice(['ESQ', 'QC', 'OBE', 'MBE', 'CBE'])
        
        # Build the full name text
        name_parts = [title, first_name]
        if middle_name:
            name_parts.append(middle_name)
        name_parts.append(surname)
        if suffix:
            name_parts.append(suffix)
        
        full_name = ' '.join(name_parts)
        
        # Add to dataset
        generated_data.append({
            'text': full_name,
            'TITLE': title,
            'FORENAME': first_name,
            'MIDDLE NAME': middle_name,
            'SURNAME': surname,
            'SUFFIX': suffix
        })
        
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(generated_data)
    df.to_csv(output_file, index=False)
    
    print(f"Dataset saved to {output_file}")
    print(f"Sample data:")
    print(df.head(10))
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique titles: {df['TITLE'].nunique()}")
    print(f"Unique first names: {df['FORENAME'].nunique()}")
    print(f"Unique surnames: {df['SURNAME'].nunique()}")
    print(f"Names with middle names: {len(df[df['MIDDLE NAME'] != ''])}")
    print(f"Names with suffixes: {len(df[df['SUFFIX'] != ''])}")
    
    return output_file

def main():
    """Main execution function - runs based on configuration variables"""
    
    print("BERT Name NER - Direct Execution Mode")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Generate Data
    if RUN_GENERATE_DATA:
        print(f"\n1. GENERATING DATA ({GENERATE_SAMPLES} samples)")
        print("-" * 30)
        generate_uk_names_dataset_faker(GENERATE_SAMPLES, GENERATED_CSV_OUTPUT)
        
    # 2. Train Model
    if RUN_TRAIN_MODEL:
        print(f"\n2. TRAINING MODEL")
        print("-" * 30)
        
        # Use generated data if it exists, otherwise use specified training file
        training_file = GENERATED_CSV_OUTPUT if RUN_GENERATE_DATA else TRAIN_CSV_FILE
        
        print(f"Training with data from: {training_file}")
        train_model(training_file, MODEL_SAVE_PATH, NUM_EPOCHS, BATCH_SIZE)
        
    # 3. Interactive Prediction
    if RUN_PREDICT_INTERACTIVE:
        print(f"\n3. INTERACTIVE PREDICTION MODE")
        print("-" * 30)
        predict_interactive(MODEL_SAVE_PATH)
        
    # 4. CSV Prediction
    if RUN_PREDICT_CSV:
        print(f"\n4. CSV PREDICTION")
        print("-" * 30)
        predict_from_csv(PREDICT_INPUT_CSV, MODEL_SAVE_PATH, PREDICT_OUTPUT_CSV)
        
    # 5. Model Evaluation
    if RUN_EVALUATE:
        print(f"\n5. MODEL EVALUATION")
        print("-" * 30)
        evaluate_model(EVALUATION_CSV, MODEL_SAVE_PATH)
    
    print("\n" + "=" * 50)
    print("Execution completed!")

if __name__ == "__main__":
    main()