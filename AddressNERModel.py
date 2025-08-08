import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Embedding, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns


class AddressNERModel:
    def __init__(self, max_seq_length=20, embedding_dim=100, lstm_units=128):
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.label_to_idx = {
            'O': 0, 'B-FLAT': 1, 'I-FLAT': 2, 'B-HOUSE_NUM': 3, 'I-HOUSE_NUM': 4,
            'B-HOUSE_NAME': 5, 'I-HOUSE_NAME': 6, 'B-STREET': 7, 'I-STREET': 8
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.vocab_size = 0
        self.num_classes = len(self.label_to_idx)
        self.history = None

    def load_data_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        tokens_list = []
        tags_list = []

        for item in json_data:
            tokens_list.append(item['tokens'])
            tags_list.append(item['bio_tags'])

        print(f"Loaded {len(tokens_list)} samples from JSON")
        return tokens_list, tags_list

    def load_data_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)

        if 'token' not in df.columns or 'bio_tag' not in df.columns:
            raise ValueError("CSV must have 'token' and 'bio_tag' columns.")
    
        tokens_list = []
        tags_list = []

        # Group tokens by address if 'address' column exists
        if 'address' in df.columns:
            grouped = df.groupby('address')
            for _, group in grouped:
                tokens_list.append(group['token'].tolist())
                tags_list.append(group['bio_tag'].tolist())
        else:
        # Fallback: assume sequential rows belong to one sentence if no grouping info
            raise ValueError("CSV must have an 'address' column to group tokens.")

        print(f"Loaded {len(tokens_list)} samples from CSV")
        return tokens_list, tags_list    

    def build_vocabulary(self, all_tokens):
        vocab = set()
        for tokens in all_tokens:
            vocab.update(tokens)

        vocab.add('<PAD>')
        vocab.add('<UNK>')

        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample vocabulary: {list(self.word_to_idx.keys())[:10]}")

    def encode_sequences(self, tokens_list, labels_list=None):
        X = []
        y = []

        for i, tokens in enumerate(tokens_list):
            token_indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
            X.append(token_indices)

            if labels_list is not None:
                label_indices = [self.label_to_idx[label] for label in labels_list[i]]
                y.append(label_indices)

        X = pad_sequences(X, maxlen=self.max_seq_length, padding='post', value=self.word_to_idx['<PAD>'])

        if labels_list is not None:
            y = pad_sequences(y, maxlen=self.max_seq_length, padding='post', value=0)
            y = np.array([to_categorical(seq, num_classes=self.num_classes) for seq in y])
            return X, y

        return X

    def build_model(self):
        input_layer = Input(shape=(self.max_seq_length,), name='input')
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_seq_length,
            mask_zero=True,
            name='embedding'
        )(input_layer)

        bi_lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
        bi_lstm1 = Dropout(0.3)(bi_lstm1)

        bi_lstm2 = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(bi_lstm1)
        bi_lstm2 = Dropout(0.3)(bi_lstm2)

        output = TimeDistributed(Dense(self.num_classes, activation='softmax'))(bi_lstm2)

        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.model.summary())
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, model_save_path='best_model.keras'):
        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6, verbose=1),
            ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        print(f"Starting training with {X_train.shape[0]} samples...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=-1)
        y_true_classes = np.argmax(y_test, axis=-1)

        y_true_flat = []
        y_pred_flat = []

        for i in range(len(y_true_classes)):
            for j in range(len(y_true_classes[i])):
                if y_true_classes[i][j] != 0:
                    y_true_flat.append(y_true_classes[i][j])
                    y_pred_flat.append(y_pred_classes[i][j])

        print("\nClassification Report:")
        print(classification_report(
            y_true_flat,
            y_pred_flat,
            labels=list(self.idx_to_label.keys()),
            target_names=[self.idx_to_label[i] for i in self.idx_to_label],
            zero_division=0
        ))

        return loss, accuracy

    def plot_training_history(self, save_path='training_history.png'):
        if self.history is None:
            print("No training history found")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title("Loss over Epochs")
        ax1.legend()

        ax2.plot(self.history.history['accuracy'], label='Train Acc')
        ax2.plot(self.history.history['val_accuracy'], label='Val Acc')
        ax2.set_title("Accuracy over Epochs")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Training history saved to {save_path}")

    def predict(self, X):
        return self.model.predict(X)

    def save_model_and_vocab(self, model_path, vocab_path):
        self.model.save(model_path)  # Save in Keras format

        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'max_seq_length': self.max_seq_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes
        }

        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)

        print(f"Model saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")

    def load_model_and_vocab(self, model_path, vocab_path):
        self.model = tf.keras.models.load_model(model_path)

        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)

        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.label_to_idx = vocab_data['label_to_idx']
        self.idx_to_label = vocab_data['idx_to_label']
        self.max_seq_length = vocab_data['max_seq_length']
        self.embedding_dim = vocab_data['embedding_dim']
        self.lstm_units = vocab_data['lstm_units']
        self.vocab_size = vocab_data['vocab_size']
        self.num_classes = vocab_data['num_classes']

        print("Model and vocabulary loaded successfully")

def main():
    # Hardcoded values
    data_path = Path("data/uk_address_bio.json")  # or use .csv here
    data_format = 'json'  # or 'csv'
    output_dir = Path("./models")
    max_seq_length = 15
    embedding_dim = 100
    lstm_units = 128
    epochs = 10
    batch_size = 32
    test_size = 0.2
    val_size = 0.2

    # Create output directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = AddressNERModel(
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units
    )

    # Load data
    print(f"Loading data from {data_path}...")
    if data_format == 'csv':
        tokens_list, tags_list = model.load_data_from_csv(data_path)
    else:
        tokens_list, tags_list = model.load_data_from_json(data_path)

    # Build vocabulary
    print("Building vocabulary...")
    model.build_vocabulary(tokens_list)

    # Encode sequences
    print("Encoding sequences...")
    X, y = model.encode_sequences(tokens_list, tags_list)

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Train model
    model_save_path = output_dir / 'best_address_ner_model.keras'
    print("Starting training...")

    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=str(model_save_path)
    )

    # Evaluate model
    print("\nEvaluating model...")
    model.evaluate_model(X_test, y_test)

    # Plot training history
    history_plot_path = output_dir / 'training_history.png'
    model.plot_training_history(str(history_plot_path))

    # Save final model and vocabulary
    final_model_path = output_dir / 'address_ner_model.keras'
    vocab_path = output_dir / 'vocabulary.pkl'

    model.save_model_and_vocab(str(final_model_path), str(vocab_path))

    print(f"\nTraining completed successfully!")
    print(f"Models saved in: {output_dir}")
    print(f"Files created:")
    print(f"  - {final_model_path}")
    print(f"  - {vocab_path}")
    print(f"  - {model_save_path}")
    print(f"  - {history_plot_path}")
if __name__ == "__main__":
    main()
