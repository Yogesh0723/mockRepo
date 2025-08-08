import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class AddressNERRunner:
    def __init__(self):
        # ✅ Updated model path to use .keras format
        model_path = "./models/address_ner_model.keras"
        vocab_path = "./models/vocabulary.pkl"

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
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")

            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)

            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']
            self.label_to_idx = vocab_data['label_to_idx']
            self.idx_to_label = vocab_data['idx_to_label']
            self.max_seq_length = vocab_data['max_seq_length']
            self.vocab_size = vocab_data['vocab_size']
            self.num_classes = vocab_data['num_classes']

            print(f"✅ Vocabulary loaded from: {vocab_path}")

        except Exception as e:
            print(f"❌ Error loading model or vocabulary: {e}")
            raise

    def tokenize_and_encode(self, address_text):
        tokens = address_text.upper().split()
        token_indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 0)) for token in tokens]
        X = pad_sequences([token_indices], maxlen=self.max_seq_length, padding='post', value=self.word_to_idx.get('<PAD>', 0))
        return tokens, X

    def extract_entities(self, tokens, labels):
        entities = {'FLAT_NUMBER': '', 'HOUSE_NUMBER': '', 'HOUSE_NAME': '', 'STREET': ''}
        current_entity = None
        current_tokens = []

        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current_entity and current_tokens:
                    self._assign_entity(entities, current_entity, current_tokens)
                current_entity = label[2:]
                current_tokens = [token]
            elif label.startswith('I-') and current_entity == label[2:]:
                current_tokens.append(token)
            else:
                if current_entity and current_tokens:
                    self._assign_entity(entities, current_entity, current_tokens)
                current_entity = None
                current_tokens = []

        if current_entity and current_tokens:
            self._assign_entity(entities, current_entity, current_tokens)

        return entities

    def _assign_entity(self, entities, entity_type, tokens):
        entity_text = ' '.join(tokens)
        if entity_type == 'FLAT':
            entities['FLAT_NUMBER'] = entity_text
        elif entity_type == 'HOUSE_NUM':
            entities['HOUSE_NUMBER'] = entity_text
        elif entity_type == 'HOUSE_NAME':
            entities['HOUSE_NAME'] = entity_text
        elif entity_type == 'STREET':
            entities['STREET'] = entity_text

    def predict_single_address(self, address_text):
        tokens, X = self.tokenize_and_encode(address_text)
        predictions = self.model.predict(X, verbose=0)
        pred_labels = [self.idx_to_label[np.argmax(pred)] for pred in predictions[0][:len(tokens)]]
        entities = self.extract_entities(tokens, pred_labels)
        return {
            'address': address_text,
            'tokens': tokens,
            'predicted_labels': pred_labels,
            'entities': entities
        }


def main():
    print("🏠 Address NER Inference")
    runner = AddressNERRunner()

    while True:
        address_input = input("\n🔍 Enter address (or type 'exit' to quit): ").strip()
        if address_input.lower() == 'exit':
            print("👋 Exiting.")
            break

        result = runner.predict_single_address(address_input)

        print("\n📌 Prediction Result:")
        print("Tokens:   ", ' | '.join(result['tokens']))
        print("Labels:   ", ' | '.join(result['predicted_labels']))
        print("Entities:")
        for key, value in result['entities'].items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
