<!--- Banner -->
<br />
<p align="center">
<a href="#"><img src="https://i.imgur.com/5XWoAcH.png"></a>
<h3 align="center">CAPTCHA Breaker</h3>
<p align="center">Extracting text out of CAPTCHA images using Keras.</p>


<!--- About --><br />
## About
This is an Optical Character Recognition algorithm that uses a Deep Convolutional Neural Network trained on the [Wilhelmy, Rodrigo & Rosas, Horacio. (2013). captcha dataset](https://www.researchgate.net/publication/248380891_captcha_dataset) in order to extract text out of [CAPTCHA images](http://www.captcha.net/).


<!--- Architecture --><br />
## Architecture
The base architecture of the network is defined as follows:
```python
x = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
x = MaxPooling2D(padding='same')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(padding='same')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(padding='same')(x)
x = Flatten()(x)
```

The `200x50` images are reduced  to a 64 dimensional vector using 3 convolutional layers of  `3x3` filters.
Batch normalization and Max Pooling layers are also used.

```python
for _ in range(NUM_CODE_CHARACTERS):
	dense = Dense(64, activation='relu')(x)
	dropout = Dropout(0.5)(dense)
	prediction = Dense(NUM_CHARACTERS, activation='sigmoid')(dropout)
	output_code.append(prediction)
```
Five fully-connected layers (one for each character) are used to predict a character using a sigmoid activation.
The output of the model is a list of the 5 predictions made for each character.


```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
The model is then compiled using the categorical cross-entropy loss function and the Adam optimizer.
Finally, an inference function is used to convert the network's prediction into a string corresponding to the CAPTCHA code, along with the network's confidence in its prediction.


<!--- Built with... --><br />
## Built with...
* [Keras](https://keras.io/) — creating the model, training and making predictions
* [Numpy](https://numpy.org/) — math operations, data preprocessing
* [OpenCV](https://opencv.org/) — reading and converting images
