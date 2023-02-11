# Speech Emotion Recognition (SER) using Siamese networks and Wav2Vec 2.0 speech representation

## DATA
### EmoDB (Berlin)
The Berlin Database of Emotional Speech (EMODB) is a freely available German emotional database recorded by 5 professional actors and 5 actresses, each. Each person speaks up to 10 sentences for each of the 7 different emotions: neutral, anger, fear, happiness, sadness, disgust, and boredom.

To Extract and Save convolutional feature encoder from a pre-traiend Wav2vec 2.0 from PyTorch Hub, Run:
```
python3 feature_extract.py
```
It asks for the path where wav files exist in, enter the absolout path and press Enter. The `.pt` files are saved into speaker subdirectories of the wav path.

## SER Model
The model is based on a MultiheadAttention with 8 heads and embedding dimension of 768. The output dimension of the MultiheadAttention is reduced to 64 which seems small enough to abstract information of whole 768 channels. Consequently, the dimensions of the Input/Output of the Attention module for an input file are:
- Input: $$Batch * 1 * Time * 768$$
- Output: $$Batch * 1 * Time * 64$$

Then, the mean and std are applied to the time dimension, which results in:
- $$Batch * 1 * 2 * 64$$

So, the dimensions of the Input/Output of the Attention module for both inputs (source and target) of Siamese network are:
- Input: $$Batch * 2 * Time * 768$$
- Output: $$Batch * 2 * Time * 64$$

and the dimension of mean and std output would be:
- $$Batch * 2 * 2 * 64$$

Which needs a 4*64 linear mapping to get the result of the Siamese network (same/not). So, the classifier layer gets the input of dimension:
- $$Batch * 256$$

and outputs a matrix with dimension of $Batch * 1$ with values in range [0,1] which shows how much these two samples are close.

# LOSS
TODO: Implement triplet loss function in pyTorch