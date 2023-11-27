import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pickle
import h5py 
print("---------------------- Downloading Dataset -------------------------\n")

dataset = pd.read_csv(r"C:\Users\hp\Downloads\SMSSpamCollection.txt", sep='\t', names=['label', 'message'])

dataset.head(5)
print("----------------------  -------------------------\n")
print(dataset.head())
print("----------------------  -------------------------")
print(dataset.groupby('label').describe())
print("----------------------  -------------------------")
dataset['label'] = dataset['label'].map({'spam': 1, 'ham': 0})
X = dataset['message'].values
y = dataset['label'].values

print("---------------------- Train Test Split -------------------------\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
print(X_train[:5])

tokeniser.fit_on_texts(X_train)

print("run3")
encoded_train = tokeniser.texts_to_sequences(X_train)
print("run4")
encoded_test = tokeniser.texts_to_sequences(X_test)
print("run5")
print(encoded_train[0:2])
print("----------------------  Padding  -------------------------\n")
max_length = 10
padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_train[0:2])

n_features = padded_train.shape[1]

# Modelling a sample DNN
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(max_length,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Training Started.")
history = model.fit(padded_train, y_train, epochs=200, batch_size=16)
loss, acc = model.evaluate(padded_test, y_test)
print("Training Finished.")

print(f'Test Accuracy: {round(acc * 100)}')
import pickle

# Assuming 'model' is your machine learning model and 'tokeniser' is your tokenizer
filename_model = r'D:\S3\stream_model\CNN\saved_dnn.pkl'
filename_tokenizer = r'D:\S3\stream_model\CNN\saved_tokeniser.pkl'

# Save the model
with open(filename_model, 'wb') as file_model:
    pickle.dump(model, file_model)

# Save the tokenizer
with open(filename_tokenizer, 'wb') as file_tokenizer:
    pickle.dump(tokeniser, file_tokenizer) 

print(f"Model saved as '{filename_model}'")
print(f"Tokenizer saved as '{filename_tokenizer}'")
