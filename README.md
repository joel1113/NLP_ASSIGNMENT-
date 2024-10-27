
Use a simple dataset for English-to-French translation. You can either use a small dataset like this or download a more extensive dataset such as the Tab-delimited Bilingual Sentence Pairs dataset from Tatoeba or Parallel Corpus from the European Parliament.

Example data (small English to French pairs)

data = [ ("hello", "bonjour"), ("how are you", "comment ça va"), ("I am fine", "je vais bien"), ("what is your name", "comment tu t'appelles"), ("my name is", "je m'appelle"), ("thank you", "merci"), ("goodbye", "au revoir") ] [CO4]

(a) Data Preprocessing

(b) Build Seq2Seq Model

(c) Preparing the Data for Training

(d) Train the model on the dataset

(e) Inference Setup for Translation

(f) Translate New Sentences

(g) Experimenting and Improving the Model by large dataset and hyper tune parameter.


import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
data = [
    ("hello", "bonjour"),
    ("how are you", "comment ça va"),
    ("I am fine", "je vais bien"),
    ("what is your name", "comment tu t'appelles"),
    ("my name is", "je m'appelle"),
    ("thank you", "merci"),
    ("goodbye", "au revoir")
]

# (a) Data Preprocessing
def preprocess_data(data):
    # Split English and French sentences
    eng_texts = [pair[0] for pair in data]
    fra_texts = [' ' + pair[1] + ' ' for pair in data]

    # Create tokenizers
    eng_tokenizer = Tokenizer()
    fra_tokenizer = Tokenizer()

    # Fit tokenizers
    eng_tokenizer.fit_on_texts(eng_texts)
    fra_tokenizer.fit_on_texts(fra_texts)

    # Convert texts to sequences
    eng_sequences = eng_tokenizer.texts_to_sequences(eng_texts)
    fra_sequences = fra_tokenizer.texts_to_sequences(fra_texts)

    # Find maximum lengths
    max_eng_len = max(len(seq) for seq in eng_sequences)
    max_fra_len = max(len(seq) for seq in fra_sequences)

    # Pad sequences
    eng_padded = pad_sequences(eng_sequences, maxlen=max_eng_len, padding='post')
    fra_padded = pad_sequences(fra_sequences, maxlen=max_fra_len, padding='post')

    return eng_padded, fra_padded, eng_tokenizer, fra_tokenizer, max_eng_len, max_fra_len

# Process the data
eng_data, fra_data, eng_tokenizer, fra_tokenizer, max_eng_len, max_fra_len = preprocess_data(data)

# Get vocabulary sizes
eng_vocab_size = len(eng_tokenizer.word_index) + 1
fra_vocab_size = len(fra_tokenizer.word_index) + 1

# Print shapes for debugging
print("Input shapes:")
print(f"English data shape: {eng_data.shape}")
print(f"French data shape: {fra_data.shape}")

# (b) Build Seq2Seq Model
def build_model(input_vocab, output_vocab, input_length, output_length):
    # Encoder
    encoder_inputs = Input(shape=(input_length,))
    encoder_embedding = Embedding(input_vocab, 50)(encoder_inputs)
    encoder = LSTM(100, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(output_length,))
    decoder_embedding = Embedding(output_vocab, 50)(decoder_inputs)
    decoder_lstm = LSTM(100, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Create model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# (c) Prepare Data for Training
decoder_input_data = fra_data[:, :-1]  # Remove last token
decoder_target_data = fra_data[:, 1:]  # Remove first token

# Print shapes for debugging
print("\nTraining data shapes:")
print(f"Decoder input shape: {decoder_input_data.shape}")
print(f"Decoder target shape: {decoder_target_data.shape}")

# (d) Train the Model
model = build_model(
    eng_vocab_size,
    fra_vocab_size,
    max_eng_len,
    max_fra_len - 1
)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    [eng_data, decoder_input_data],
    decoder_target_data,
    batch_size=2,
    epochs=100,
    validation_split=0.2
)

# (e) Inference Setup
class Translator:
    def __init__(self, model, eng_tokenizer, fra_tokenizer, max_eng_len, max_fra_len):
        self.model = model
        self.eng_tokenizer = eng_tokenizer
        self.fra_tokenizer = fra_tokenizer
        self.max_eng_len = max_eng_len
        self.max_fra_len = max_fra_len
        self.fra_index_word = {v: k for k, v in fra_tokenizer.word_index.items()}

    def translate(self, text):
        # Tokenize input text
        sequence = self.eng_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_eng_len, padding='post')

        # Initialize target sequence
        target_seq = np.zeros((1, self.max_fra_len - 1))

        # Generate translation
        prediction = self.model.predict([padded, target_seq])

        # Convert prediction to text
        output_sequence = np.argmax(prediction[0], axis=1)
        translated_text = []

        for idx in output_sequence:
            if idx != 0:
                word = self.fra_index_word.get(idx, '')
                if word and word not in ['', '']:
                    translated_text.append(word)

        return ' '.join(translated_text)

# (f) Translate New Sentences
translator = Translator(model, eng_tokenizer, fra_tokenizer, max_eng_len, max_fra_len)

# Test translations
test_sentences = [
    "hello",
    "thank you",
    "your name",
    "how are you"
]

print("\nTranslations:")
for sentence in test_sentences:
    translation = translator.translate(sentence)
    print(f"English: {sentence}")
    print(f"French: {translation}\n")
     
Input shapes:
English data shape: (7, 4)
French data shape: (7, 5)

Training data shapes:
Decoder input shape: (7, 4)
Decoder target shape: (7, 4)
Epoch 1/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 4s 222ms/step - accuracy: 0.1781 - loss: 2.7703 - val_accuracy: 0.1250 - val_loss: 2.7523
Epoch 2/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.2500 - loss: 2.7511 - val_accuracy: 0.2500 - val_loss: 2.7370
Epoch 3/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.2500 - loss: 2.7321 - val_accuracy: 0.2500 - val_loss: 2.7147
Epoch 4/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 0.2500 - loss: 2.7038 - val_accuracy: 0.2500 - val_loss: 2.6822
Epoch 5/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.2500 - loss: 2.6697 - val_accuracy: 0.2500 - val_loss: 2.6381
Epoch 6/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.2500 - loss: 2.6374 - val_accuracy: 0.2500 - val_loss: 2.5761
Epoch 7/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - accuracy: 0.2500 - loss: 2.5589 - val_accuracy: 0.2500 - val_loss: 2.4734
Epoch 8/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - accuracy: 0.2500 - loss: 2.4826 - val_accuracy: 0.2500 - val_loss: 2.3248
Epoch 9/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 0.2500 - loss: 2.2995 - val_accuracy: 0.2500 - val_loss: 2.1605
Epoch 10/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.2500 - loss: 2.1589 - val_accuracy: 0.2500 - val_loss: 2.1536
Epoch 11/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.2500 - loss: 2.1695 - val_accuracy: 0.2500 - val_loss: 2.1748
Epoch 12/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.2500 - loss: 2.0422 - val_accuracy: 0.2500 - val_loss: 2.1187
Epoch 13/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.2500 - loss: 2.0129 - val_accuracy: 0.2500 - val_loss: 2.0938
Epoch 14/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 0.3219 - loss: 1.9148 - val_accuracy: 0.5000 - val_loss: 2.1059
Epoch 15/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.4344 - loss: 1.9494 - val_accuracy: 0.5000 - val_loss: 2.1314
Epoch 16/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.3562 - loss: 2.0671 - val_accuracy: 0.5000 - val_loss: 2.1567
Epoch 17/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 0.5156 - loss: 1.9194 - val_accuracy: 0.5000 - val_loss: 2.1717
Epoch 18/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5469 - loss: 1.8345 - val_accuracy: 0.6250 - val_loss: 2.2120
Epoch 19/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.5156 - loss: 1.8150 - val_accuracy: 0.6250 - val_loss: 2.2781
Epoch 20/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.4844 - loss: 1.8538 - val_accuracy: 0.6250 - val_loss: 2.3406
Epoch 21/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.4844 - loss: 1.7311 - val_accuracy: 0.6250 - val_loss: 2.3896
Epoch 22/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.4844 - loss: 1.7434 - val_accuracy: 0.5000 - val_loss: 2.4392
Epoch 23/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5000 - loss: 1.5845 - val_accuracy: 0.5000 - val_loss: 2.5053
Epoch 24/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5156 - loss: 1.5518 - val_accuracy: 0.5000 - val_loss: 2.6016
Epoch 25/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.4844 - loss: 1.5211 - val_accuracy: 0.5000 - val_loss: 2.7236
Epoch 26/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5156 - loss: 1.4369 - val_accuracy: 0.5000 - val_loss: 2.8668
Epoch 27/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 0.4375 - loss: 1.4898 - val_accuracy: 0.3750 - val_loss: 2.9898
Epoch 28/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 0.5156 - loss: 1.2883 - val_accuracy: 0.5000 - val_loss: 3.1014
Epoch 29/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5781 - loss: 1.0952 - val_accuracy: 0.5000 - val_loss: 3.2311
Epoch 30/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 0.5156 - loss: 1.1805 - val_accuracy: 0.3750 - val_loss: 3.3717
Epoch 31/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 0.5000 - loss: 1.1128 - val_accuracy: 0.5000 - val_loss: 3.5108
Epoch 32/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5406 - loss: 1.0409 - val_accuracy: 0.3750 - val_loss: 3.5962
Epoch 33/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5281 - loss: 1.1311 - val_accuracy: 0.3750 - val_loss: 3.7048
Epoch 34/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.7094 - loss: 0.9598 - val_accuracy: 0.5000 - val_loss: 3.8452
Epoch 35/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.6719 - loss: 0.9779 - val_accuracy: 0.5000 - val_loss: 3.9010
Epoch 36/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.8875 - loss: 0.7809 - val_accuracy: 0.3750 - val_loss: 3.9765
Epoch 37/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.9281 - loss: 0.7983 - val_accuracy: 0.5000 - val_loss: 4.0720
Epoch 38/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.9594 - loss: 0.6173 - val_accuracy: 0.5000 - val_loss: 4.1424
Epoch 39/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.9281 - loss: 0.5946 - val_accuracy: 0.5000 - val_loss: 4.1963
Epoch 40/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 0.9281 - loss: 0.5609 - val_accuracy: 0.3750 - val_loss: 4.2502
Epoch 41/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 1.0000 - loss: 0.5357 - val_accuracy: 0.5000 - val_loss: 4.3377
Epoch 42/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.4433 - val_accuracy: 0.5000 - val_loss: 4.3683
Epoch 43/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.4010 - val_accuracy: 0.5000 - val_loss: 4.4035
Epoch 44/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.3414 - val_accuracy: 0.5000 - val_loss: 4.5080
Epoch 45/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.9750 - loss: 0.3553 - val_accuracy: 0.5000 - val_loss: 4.5311
Epoch 46/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 1.0000 - loss: 0.3182 - val_accuracy: 0.3750 - val_loss: 4.5412
Epoch 47/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.2551 - val_accuracy: 0.5000 - val_loss: 4.6365
Epoch 48/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.2719 - val_accuracy: 0.5000 - val_loss: 4.6560
Epoch 49/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - accuracy: 1.0000 - loss: 0.2372 - val_accuracy: 0.3750 - val_loss: 4.6881
Epoch 50/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 1.0000 - loss: 0.2445 - val_accuracy: 0.3750 - val_loss: 4.6891
Epoch 51/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 1.0000 - loss: 0.1692 - val_accuracy: 0.5000 - val_loss: 4.7024
Epoch 52/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - accuracy: 1.0000 - loss: 0.1815 - val_accuracy: 0.5000 - val_loss: 4.6976
Epoch 53/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.1655 - val_accuracy: 0.5000 - val_loss: 4.6744
Epoch 54/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.1515 - val_accuracy: 0.5000 - val_loss: 4.7021
Epoch 55/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.1487 - val_accuracy: 0.5000 - val_loss: 4.7448
Epoch 56/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.1500 - val_accuracy: 0.5000 - val_loss: 4.7661
Epoch 57/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.1342 - val_accuracy: 0.5000 - val_loss: 4.7637
Epoch 58/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - accuracy: 1.0000 - loss: 0.1151 - val_accuracy: 0.5000 - val_loss: 4.7665
Epoch 59/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.1146 - val_accuracy: 0.5000 - val_loss: 4.7818
Epoch 60/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.1138 - val_accuracy: 0.5000 - val_loss: 4.7988
Epoch 61/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - accuracy: 1.0000 - loss: 0.1038 - val_accuracy: 0.5000 - val_loss: 4.8136
Epoch 62/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.0931 - val_accuracy: 0.5000 - val_loss: 4.8193
Epoch 63/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.0942 - val_accuracy: 0.5000 - val_loss: 4.8203
Epoch 64/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 1.0000 - loss: 0.0908 - val_accuracy: 0.5000 - val_loss: 4.8233
Epoch 65/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.0835 - val_accuracy: 0.5000 - val_loss: 4.8330
Epoch 66/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.0834 - val_accuracy: 0.5000 - val_loss: 4.8473
Epoch 67/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - accuracy: 1.0000 - loss: 0.0784 - val_accuracy: 0.5000 - val_loss: 4.8589
Epoch 68/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 1.0000 - loss: 0.0798 - val_accuracy: 0.5000 - val_loss: 4.8676
Epoch 69/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.0720 - val_accuracy: 0.5000 - val_loss: 4.8760
Epoch 70/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 1.0000 - loss: 0.0697 - val_accuracy: 0.5000 - val_loss: 4.8821
Epoch 71/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 1.0000 - loss: 0.0674 - val_accuracy: 0.5000 - val_loss: 4.8889
Epoch 72/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.0643 - val_accuracy: 0.5000 - val_loss: 4.8969
Epoch 73/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.0598 - val_accuracy: 0.5000 - val_loss: 4.9038
Epoch 74/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.0591 - val_accuracy: 0.5000 - val_loss: 4.9101
Epoch 75/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step - accuracy: 1.0000 - loss: 0.0548 - val_accuracy: 0.5000 - val_loss: 4.9142
Epoch 76/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - accuracy: 1.0000 - loss: 0.0519 - val_accuracy: 0.5000 - val_loss: 4.9161
Epoch 77/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - accuracy: 1.0000 - loss: 0.0507 - val_accuracy: 0.5000 - val_loss: 4.9189
Epoch 78/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 1.0000 - loss: 0.0494 - val_accuracy: 0.5000 - val_loss: 4.9246
Epoch 79/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 1.0000 - loss: 0.0472 - val_accuracy: 0.5000 - val_loss: 4.9347
Epoch 80/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step - accuracy: 1.0000 - loss: 0.0472 - val_accuracy: 0.5000 - val_loss: 4.9453
Epoch 81/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 1.0000 - loss: 0.0464 - val_accuracy: 0.5000 - val_loss: 4.9539
Epoch 82/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - accuracy: 1.0000 - loss: 0.0449 - val_accuracy: 0.5000 - val_loss: 4.9613
Epoch 83/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 1.0000 - loss: 0.0414 - val_accuracy: 0.5000 - val_loss: 4.9655
Epoch 84/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - accuracy: 1.0000 - loss: 0.0419 - val_accuracy: 0.5000 - val_loss: 4.9699
Epoch 85/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 1.0000 - loss: 0.0411 - val_accuracy: 0.5000 - val_loss: 4.9758
Epoch 86/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 1.0000 - loss: 0.0394 - val_accuracy: 0.5000 - val_loss: 4.9831
Epoch 87/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 1.0000 - loss: 0.0365 - val_accuracy: 0.5000 - val_loss: 4.9904
Epoch 88/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step - accuracy: 1.0000 - loss: 0.0364 - val_accuracy: 0.5000 - val_loss: 4.9974
Epoch 89/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 1.0000 - loss: 0.0360 - val_accuracy: 0.5000 - val_loss: 5.0044
Epoch 90/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - accuracy: 1.0000 - loss: 0.0333 - val_accuracy: 0.5000 - val_loss: 5.0099
Epoch 91/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step - accuracy: 1.0000 - loss: 0.0323 - val_accuracy: 0.5000 - val_loss: 5.0148
Epoch 92/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 1.0000 - loss: 0.0326 - val_accuracy: 0.5000 - val_loss: 5.0196
Epoch 93/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - accuracy: 1.0000 - loss: 0.0307 - val_accuracy: 0.5000 - val_loss: 5.0246
Epoch 94/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step - accuracy: 1.0000 - loss: 0.0306 - val_accuracy: 0.5000 - val_loss: 5.0297
Epoch 95/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - accuracy: 1.0000 - loss: 0.0300 - val_accuracy: 0.5000 - val_loss: 5.0358
Epoch 96/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.0295 - val_accuracy: 0.5000 - val_loss: 5.0422
Epoch 97/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 1.0000 - loss: 0.0285 - val_accuracy: 0.5000 - val_loss: 5.0483
Epoch 98/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 1.0000 - loss: 0.0275 - val_accuracy: 0.5000 - val_loss: 5.0537
Epoch 99/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 1.0000 - loss: 0.0262 - val_accuracy: 0.5000 - val_loss: 5.0581
Epoch 100/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - accuracy: 1.0000 - loss: 0.0261 - val_accuracy: 0.5000 - val_loss: 5.0626

Translations:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 291ms/step
English: hello
French: bonjour end

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
English: thank you
French: comment end

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
English: your name
French: comment end

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
English: how are you
French: comment ça va end
