import numpy as np
import pickle
import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class AddressNERRunner:
    def __init__(self, model_path, vocab_path):
        """Initialize the NER runner with trained model and vocabulary"""
        self.model = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.max_seq_length = 0
        self.vocab_size = 0
        self.num_classes = 0
        
        self.load_model_and_vocab(model_path, vocab_path)
    
    def load_model_and_vocab(self, model_path, vocab_path):
        """Load trained model and vocabulary"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
            
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']
            self.label_to_idx = vocab_data['label_to_idx']
            self.idx_to_label = vocab_data['idx_to_label']
            self.max_seq_length = vocab_data['max_seq_length']
            self.vocab_size = vocab_data['vocab_size']
            self.num_classes = vocab_data['num_classes']
            
            print(f"Vocabulary loaded from {vocab_path}")
            print(f"Vocabulary size: {self.vocab_size}")
            print(f"Max sequence length: {self.max_seq_length}")
            
        except Exception as e:
            print(f"Error loading model or vocabulary: {e}")
            raise
    
    def preprocess_address_fields(self, add1="", add2="", add3="", add4=""):
        """Preprocess 4-field address input format"""
        fields = [add1, add2, add3, add4]
        # Remove empty fields and join
        address = ' '.join([field.strip() for field in fields if field.strip()])
        return address.upper()
    
    def tokenize_and_encode(self, address_text):
        """Tokenize address and convert to model input format"""
        # Tokenize
        tokens = address_text.upper().split()
        
        # Convert tokens to indices
        token_indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 0)) 
                        for token in tokens]
        
        # Pad sequence
        X = pad_sequences([token_indices], maxlen=self.max_seq_length, 
                         padding='post', value=self.word_to_idx.get('<PAD>', 0))
        
        return tokens, X
    
    def predict_single_address(self, address_text):
        """Predict NER tags for a single address"""
        tokens, X = self.tokenize_and_encode(address_text)
        
        # Make prediction
        predictions = self.model.predict(X, verbose=0)
        
        # Get predicted labels
        pred_labels = []
        for i in range(len(tokens)):
            pred_idx = np.argmax(predictions[0][i])
            pred_labels.append(self.idx_to_label[pred_idx])
        
        # Extract entities
        entities = self.extract_entities(tokens, pred_labels)
        
        return {
            'address': address_text,
            'tokens': tokens,
            'predicted_labels': pred_labels,
            'entities': entities
        }
    
    def predict_from_fields(self, add1="", add2="", add3="", add4=""):
        """Predict NER tags from 4-field address format"""
        address = self.preprocess_address_fields(add1, add2, add3, add4)
        
        if not address.strip():
            return {
                'address': '',
                'tokens': [],
                'predicted_labels': [],
                'entities': {
                    'FLAT_NUMBER': '',
                    'HOUSE_NUMBER': '',
                    'HOUSE_NAME': '',
                    'STREET': ''
                },
                'error': 'Empty address provided'
            }
        
        result = self.predict_single_address(address)
        result['original_fields'] = {
            'ADD1': add1,
            'ADD2': add2,
            'ADD3': add3,
            'ADD4': add4
        }
        
        return result
    
    def extract_entities(self, tokens, labels):
        """Extract entities from tokens and labels using BIO tagging"""
        entities = {
            'FLAT_NUMBER': '',
            'HOUSE_NUMBER': '',
            'HOUSE_NAME': '',
            'STREET': ''
        }
        
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if current_entity == 'FLAT':
                        entities['FLAT_NUMBER'] = entity_text
                    elif current_entity == 'HOUSE_NUM':
                        entities['HOUSE_NUMBER'] = entity_text
                    elif current_entity == 'HOUSE_NAME':
                        entities['HOUSE_NAME'] = entity_text
                    elif current_entity == 'STREET':
                        entities['STREET'] = entity_text
                
                # Start new entity
                current_entity = label[2:]  # Remove 'B-'
                current_tokens = [token]
            
            elif label.startswith('I-') and current_entity == label[2:]:
                current_tokens.append(token)
            
            else:
                # Save current entity and reset
                if current_entity and current_tokens:
                    entity_text = ' '.join(current_tokens)
                    if current_entity == 'FLAT':
                        entities['FLAT_NUMBER'] = entity_text
                    elif current_entity == 'HOUSE_NUM':
                        entities['HOUSE_NUMBER'] = entity_text
                    elif current_entity == 'HOUSE_NAME':
                        entities['HOUSE_NAME'] = entity_text
                    elif current_entity == 'STREET':
                        entities['STREET'] = entity_text
                
                current_entity = None
                current_tokens = []
        
        # Handle last entity
        if current_entity and current_tokens:
            entity_text = ' '.join(current_tokens)
            if current_entity == 'FLAT':
                entities['FLAT_NUMBER'] = entity_text
            elif current_entity == 'HOUSE_NUM':
                entities['HOUSE_NUMBER'] = entity_text
            elif current_entity == 'HOUSE_NAME':
                entities['HOUSE_NAME'] = entity_text
            elif current_entity == 'STREET':
                entities['STREET'] = entity_text
        
        return entities
    
    def predict_batch(self, addresses):
        """Predict NER tags for a batch of addresses"""
        results = []
        
        for address in addresses:
            if isinstance(address, str):
                result = self.predict_single_address(address)
            elif isinstance(address, dict):
                # Assume it's a 4-field format
                result = self.predict_from_fields(
                    address.get('ADD1', ''),
                    address.get('ADD2', ''),
                    address.get('ADD3', ''),
                    address.get('ADD4', '')
                )
            else:
                result = {'error': f'Invalid address format: {address}'}
            
            results.append(result)
        
        return results
    
    def format_output(self, result, format_type='dict'):
        """Format prediction output"""
        if format_type == 'dict':
            return result
        
        elif format_type == 'json':
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv_row':
            entities = result['entities']
            return {
                'address': result['address'],
                'flat_number': entities['FLAT_NUMBER'],
                'house_number': entities['HOUSE_NUMBER'],
                'house_name': entities['HOUSE_NAME'],
                'street': entities['STREET']
            }
        
        elif format_type == 'readable':
            output = f"Address: {result['address']}\n"
            output += f"Tokens: {' | '.join(result['tokens'])}\n"
            output += f"Labels: {' | '.join(result['predicted_labels'])}\n"
            output += "Extracted Entities:\n"
            for entity_type, entity_value in result['entities'].items():
                output += f"  {entity_type}: '{entity_value}'\n"
            return output
        
        else:
            return result
    
    def save_predictions(self, results, output_path, format_type='json'):
        """Save predictions to file"""
        if format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv':
            import pandas as pd
            csv_data = []
            for result in results:
                if 'error' not in result:
                    csv_row = self.format_output(result, 'csv_row')
                    csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run Address NER Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.h5 file)')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to vocabulary file (.pkl file)')
    parser.add_argument('--input', type=str,
                        help='Input address string or JSON file with addresses')
    parser.add_argument('--add1', type=str, default='',
                        help='Address field 1 (32 bytes max)')
    parser.add_argument('--add2', type=str, default='',
                        help='Address field 2 (32 bytes max)')
    parser.add_argument('--add3', type=str, default='',
                        help='Address field 3 (32 bytes max)')
    parser.add_argument('--add4', type=str, default='',
                        help='Address field 4 (32 bytes max)')
    parser.add_argument('--batch_file', type=str,
                        help='JSON file containing batch of addresses')
    parser.add_argument('--output', type=str,
                        help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv', 'readable'], default='readable',
                        help='Output format (default: readable)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize runner
    print("Loading model and vocabulary...")
    runner = AddressNERRunner(args.model_path, args.vocab_path)
    print("Model loaded successfully!\n")
    
    # Interactive mode
    if args.interactive:
        print("=== Interactive Address NER ===")
        print("Enter addresses or 4-field format (type 'quit' to exit)")
        print("Format options:")
        print("1. Single address: 'ADDRESS_TEXT'")
        print("2. Four fields: 'ADD1|ADD2|ADD3|ADD4'\n")
        
        while True:
            try:
                user_input = input("Enter address: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if '|' in user_input:
                    # 4-field format
                    fields = user_input.split('|')
                    fields += [''] * (4 - len(fields))  # Pad to 4 fields
                    result = runner.predict_from_fields(fields[0], fields[1], fields[2], fields[3])
                else:
                    # Single address
                    result = runner.predict_single_address(user_input)
                
                print("\n" + runner.format_output(result, 'readable'))
                print("-" * 60)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
        return
    
    # Single address prediction
    if args.input:
        result = runner.predict_single_address(args.input)
        output = runner.format_output(result, args.format)
        
        if args.output:
            if args.format == 'json':
                runner.save_predictions([result], args.output, 'json')
            else:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output)
                print(f"Result saved to {args.output}")
        else:
            print(output)
    
    # 4-field prediction
    elif any([args.add1, args.add2, args.add3, args.add4]):
        result = runner.predict_from_fields(args.add1, args.add2, args.add3, args.add4)
        output = runner.format_output(result, args.format)
        
        if args.output:
            if args.format == 'json':
                runner.save_predictions([result], args.output, 'json')
            else:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output)
                print(f"Result saved to {args.output}")
        else:
            print(output)
    
    # Batch prediction
    elif args.batch_file:
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        if isinstance(batch_data, list):
            addresses = batch_data
        elif isinstance(batch_data, dict) and 'addresses' in batch_data:
            addresses = batch_data['addresses']
        else:
            print("Invalid batch file format. Expected list of addresses or dict with 'addresses' key.")
            return
        
        print(f"Processing {len(addresses)} addresses...")
        results = runner.predict_batch(addresses)
        
        if args.output:
            output_format = 'json' if args.format == 'json' else 'csv' if args.format == 'csv' else 'json'
            runner.save_predictions(results, args.output, output_format)
        else:
            for i, result in enumerate(results):
                print(f"\n=== Address {i+1} ===")
                print(runner.format_output(result, 'readable'))
    
    else:
        # Demo with example
        print("=== Address NER Demo ===")
        example_addresses = [
            "39A ST JOHNS STREET PERTH",
            "FLAT 12 WESTMINSTER HOUSE BAKER STREET",
            "25 THE OAKS VICTORIA ROAD",
            "UNIT 5 142B COLLINS AVENUE"
        ]
        
        print("Processing example addresses...\n")
        for address in example_addresses:
            result = runner.predict_single_address(address)
            print(runner.format_output(result, 'readable'))
            print("-" * 60)

if __name__ == "__main__":
    main()