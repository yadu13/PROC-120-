# Model Training Lib
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
# from tf.keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from keras.models import load_model


from data_preprocessing import preprocess_train_data

def train_bot_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile Model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    # Fit & Save Model
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    model.save('helpbot_model.h5', history)
    print("Model File Created & Saved")


# Calling Methods to Train Model
train_x, train_y = preprocess_train_data()

train_bot_model(train_x, train_y)

