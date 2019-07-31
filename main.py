# ===============[ IMPORTS ]===============
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential

import numpy as np
import cv2
import os


# ===============[ HYPERPARAMETERS ]===============
EPOCHS = 32
BATCH_SIZE = 32

CHARACTERS = 'abcdefghijklmnopqrstuvwxyz0123456789'
NUM_CHARACTERS = len(CHARACTERS)
NUM_CODE_CHARACTERS = 5

DATA_PATH = 'data'
CHECKPOINTS_PATH = 'checkpoints/weights-{epoch:02d}-{loss:.2f}.hdf5'

IMG_LIST = os.listdir(DATA_PATH)
IMG_SHAPE = (50, 200, 1)


# ===============[ DATASET INITIALIZATION ]===============
m = len(IMG_LIST)

X = np.zeros((m, *IMG_SHAPE))
Y = np.zeros((NUM_CODE_CHARACTERS, m, NUM_CHARACTERS))


# ===============[ DATASET POPULATION (ONE-HOT ENCODING) ]===============
for i, img_name in enumerate(IMG_LIST):

    img = cv2.imread(os.path.join(DATA_PATH, img_name), cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    img = np.reshape(img, IMG_SHAPE)

    one_hot_target = np.zeros((NUM_CODE_CHARACTERS, NUM_CHARACTERS))
    for j, char in enumerate(img_name[:-4]):
        index = CHARACTERS.find(char)
        one_hot_target[j, index] = 1

    X[i] = img
    Y[:, i] = one_hot_target


# ===============[ PREDICTION ]===============
def predict(model, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0

    predicted_code = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    predicted_code = predicted_code.reshape(NUM_CODE_CHARACTERS, NUM_CHARACTERS)

    indexes = []
    probabilities = []

    for char_vector in predicted_code:
        indexes.append(np.argmax(char_vector))
        probabilities.append(np.max(char_vector))

    code = ''
    confidence = (sum(probabilities) / NUM_CODE_CHARACTERS) * 100
    for i in indexes:
        code += CHARACTERS[i]


    return code, confidence


# ===============[ MODEL ]===============
def create_model():
    input_img = Input(shape=IMG_SHAPE)
    output_code = []

    x = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(padding='same')(x)
    x = Flatten()(x)

    for _ in range(NUM_CODE_CHARACTERS):
        dense = Dense(64, activation='relu')(x)
        dropout = Dropout(0.5)(dense)
        prediction = Dense(NUM_CHARACTERS, activation='sigmoid')(dropout)

        output_code.append(prediction)

    model = Model(input_img, output_code)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ===============[ TRAINING ]===============
checkpoint_callback = ModelCheckpoint(CHECKPOINTS_PATH, verbose=0, monitor='loss', mode='min', save_best_only=False, save_weights_only=True)

model = create_model()
model.fit(X, [Y[i] for i in range(NUM_CODE_CHARACTERS)], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_split=0.2, callbacks=[checkpoint_callback])
