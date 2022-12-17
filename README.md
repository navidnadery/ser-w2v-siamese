# Speech Emotion Recognition (SER) using Siamese networks and Wav2Vec 2.0 speech representation

## DATA
### EmoDB (Berlin)
The Berlin Database of Emotional Speech (EMODB) is a freely available German emotional database recorded by 5 professional actors and 5 actresses, each. Each person speaks up to 10 sentences for each of the 7 different emotions: neutral, anger, fear, happiness, sadness, disgust, and boredom.

To Extract and Save convolutional feature encoder from a pre-traiend Wav2vec 2.0 from PyTorch Hub, Run:
```
python3 feature_extract.py
```
It asks for the path where wav files exist in, enter the absolout path and press Enter. The `.pt` files are saved into speaker subdirectories of the wav path.