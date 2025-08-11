# mockRepo
Mock to push unwanted or code sharing


pip install numpy
pip install pandas
pip install scikit-learn
pip install tensorflow
pip install matplotlib
pip install seaborn
pip install Faker


# BERT Name NER for UK Names

A BERT-based Named Entity Recognition model specifically designed for extracting name components (Title, First Name, Middle Name, Surname, Suffix) from UK names.

## Requirements

```
torch>=1.9.0
transformers>=4.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
faker>=18.0.0
```

## Installation

1. Install Python 3.11 (if not already installed)
2. Install the required packages:

```bash
pip install torch transformers pandas scikit-learn numpy faker
```

## File Structure

```
bert_name_ner/
├── bert_name_ner.py          # Main model implementation
├── bert_ner_runner.py        # Runner script
├── names.csv                 # Your training data (CSV format)
└── bert_name_ner_model.pth   # Trained model (generated after training)
```

## CSV Format

Your CSV file should have the following columns:
- `text`: The full name string
- `TITLE`: Title (MR, MRS, DR, etc.)
- `FORENAME`: First name
- `MIDDLE NAME`: Middle name(s)
- `SURNAME`: Last name
- `SUFFIX`: Suffix (JR, SR, III, etc.)

Example:
```csv
text,TITLE,FORENAME,MIDDLE NAME,SURNAME,SUFFIX
"MR THOMAS SURRAY FISHER",MR,THOMAS,SURRAY,FISHER,
"MR CONRAD A FISHER JR",MR,CONRAD,A,FISHER,JR
```

## Usage (Direct Execution)

The script now runs directly without command-line arguments. Simply modify the configuration variables at the top of `bert_ner_runner.py` and run:

```python
# Edit these variables in bert_ner_runner.py:
GENERATE_SAMPLES = 2000                    # Number of samples to generate
NUM_EPOCHS = 5                            # Training epochs
RUN_GENERATE_DATA = True                  # Generate synthetic data
RUN_TRAIN_MODEL = True                    # Train the model  
RUN_PREDICT_INTERACTIVE = True            # Run interactive mode
RUN_PREDICT_CSV = False                   # Process CSV files
RUN_EVALUATE = False                      # Evaluate model
```

### Then simply run:
```bash
python bert_ner_runner.py
```

This will:
1. Generate 2000 UK names using Faker
2. Train the BERT model  
3. Start interactive prediction mode

### Configuration Options

**Data Generation:**
- `GENERATE_SAMPLES`: Number of synthetic names to create
- `GENERATED_CSV_OUTPUT`: Where to save generated data

**Training:**  
- `NUM_EPOCHS`: Training epochs (3-10 recommended)
- `BATCH_SIZE`: Training batch size (8-32)
- `MODEL_SAVE_PATH`: Where to save trained model

**Execution Control:**
- `RUN_GENERATE_DATA`: Generate new training data  
- `RUN_TRAIN_MODEL`: Train the model
- `RUN_PREDICT_INTERACTIVE`: Interactive name testing
- `RUN_PREDICT_CSV`: Batch process CSV files
- `RUN_EVALUATE`: Evaluate model performance

## Model Architecture

- **Base Model**: BERT-base-uncased from Hugging Face
- **Tokenizer**: BERT tokenizer (no tensorflow-text dependency)
- **Labels**: BIO tagging scheme (B-TITLE, I-TITLE, B-FNAME, I-FNAME, etc.)
- **Classification**: Token-level classification with linear layer on top of BERT

## Label Schema

The model uses BIO (Beginning-Inside-Outside) tagging:
- `O`: Outside any entity
- `B-TITLE`: Beginning of title
- `I-TITLE`: Inside title (continuation)
- `B-FNAME`: Beginning of first name
- `I-FNAME`: Inside first name
- `B-MNAME`: Beginning of middle name
- `I-MNAME`: Inside middle name  
- `B-SNAME`: Beginning of surname
- `I-SNAME`: Inside surname
- `B-SUFFIX`: Beginning of suffix
- `I-SUFFIX`: Inside suffix

## Data Generator Features (Using Faker)

The enhanced data generator now uses Faker library for realistic UK names:

- **UK-localized names**: Uses `Faker('en_GB')` for authentic UK names
- **Gender-appropriate titles**: Matches titles to gender (MR for male, MRS/MS for female)
- **Weighted distributions**: Common titles (MR, MRS) appear more frequently
- **Mixed name sources**: Combines UK and US Faker for diversity
- **Professional titles**: DR, PROF, REV, SIR, LADY, LORD, OBE, MBE, CBE
- **Realistic patterns**: 
  - 70% have middle names (mix of full names and initials)
  - 10% have suffixes (JR, SR, II, III, IV)
  - 5% have honors/qualifications (ESQ, QC, OBE, etc.)

Generated names look like:
```
MR JAMES WILLIAM THOMPSON
DR SARAH ELIZABETH PATEL  
PROF MICHAEL A HARRISON JR
MS CHARLOTTE ROSE WILLIAMS
SIR HENRY ARTHUR DAVIDSON
```

Statistics are shown after generation:
- Total samples created
- Unique names and titles
- Distribution analysis

## Training Tips

1. **Data Quality**: Ensure your CSV data is clean and properly formatted
2. **Epochs**: Start with 5 epochs, increase if needed for better performance  
3. **Batch Size**: Adjust based on your GPU memory
