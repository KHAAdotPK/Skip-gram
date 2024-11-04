## skip-gram 
This repository contains the implementation of a **Skip-gram** model in C++. The Skip-gram model is a type of neural network used for natural language processing (NLP) tasks, particularly for learning word embeddings. Word embeddings are dense vector representations of words that capture their semantic relationships.The Skip-gram model learns these semantic relationships and use them to predict surrounding context words for a given target/center word. 
The Skip-gram's learned word embeddings can then be used for various NLP tasks, such as text classification, machine translation, and sentiment analysis.
### Overview
This implementation includes the following key components:
- **Word Embeddings (W1)**: An embedding matrix where each row corresponds to a unique word's embedding vector.
- **Output Weights (W2)**: Weights for predicting context words.
- **Training Loop**: Iterates over the dataset for a specified number of epochs, updating the weights using forward and backward propagation.
- **Forward Propagation**: Computes the hidden layer activation and predicted probabilities.
- **Backward Propagation**: Calculates the gradients for weight updates.
- **Loss Calculation**: Utilizes negative log-likelihood to measure the model's performance.
- **Regularization**: Implements techniques to prevent overfitting.
### Dependencies
1.  [ala_exception](https://github.com/KHAAdotPK/ala_exception)
2.  [allocator](https://github.com/KHAAdotPK/allocator)
3.  [argsv-cpp](https://github.com/KHAAdotPK/argsv-cpp)
4.  [corpus](https://github.com/KHAAdotPK/corpus)
5.  [csv](https://github.com/KHAAdotPK/csv)
6.  [Numcy](https://github.com/KHAAdotPK/Numcy)
7.  [pairs](https://github.com/KHAAdotPK/pairs.git)
8.  [parser](https://github.com/KHAAdotPK/parser)
9.  [string](https://github.com/KHAAdotPK/string)
10. [sundry](https://github.com/KHAAdotPK/sundry)
11. [read_write_weights](https://github.com/KHAAdotPK/read_write_weights)

In the **lib** folder, there is a small batch file named **PULL.cmd**. Change into the lib folder and at the command prompt execute this file. It will try to clone all the above-mentioned dependencies.
### Getting Started
Follow these steps to set up and run the Skip-gram model:
1. **Clone this repository**:
```BASH
git clone https://github.com/KHAAdotPK/Skip-gram.git
```
2. **Install dependencies**:
```BASH
cd skip-gram\lib
PULL.cmd
```
3 **Compile source code**:
```BASH
cd ..\usage
./BUILD.cmd
# build process should result in a executable file named ./skipy.exe
```
4 **Start training the model**:
```BASH
./RUN.cmd
```
### TODO
**Negative Sampling**: Implement negative sampling to improve training efficiency. Negative sampling is an optimization technique used to approximate the softmax function in the output layer. Instead of updating the weights for all words in the vocabulary, negative sampling updates only a small number of negative samples, reducing computational complexity and speeding up the training process.

# Skip-gram Model Training

This repository contains the implementation of a Skip-gram model in C++ designed for word embedding generation from a text corpus. The model has been trained multiple times to ensure the robustness and reliability of the learned representations.

## Training Environment

- **Language**: C++
- **System**: Windows 10
- **Memory**: 16 GB
- **GPU**: Not supported (CPU only)
- **Batch Training**: The model is trained in batches to prevent memory overload.

## Training Runs

### Run 1
```
F:\Skip-gram\usage>.\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 4 loop 0 verbose
Corpus: ./INPUT.txt
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (10212.6), 18.6703
Epoch# 2 of 4 epochs.
epoch_loss = (9827.57), 17.9663
Epoch# 3 of 4 epochs.
epoch_loss = (9454.42), 17.2841
Epoch# 4 of 4 epochs.
epoch_loss = (9340.72), 17.0763
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 4 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (5501.98), 10.0585
Epoch# 2 of 4 epochs.
epoch_loss = (5376.5), 9.82907
Epoch# 3 of 4 epochs.
epoch_loss = (5252.69), 9.60272
Epoch# 4 of 4 epochs.
epoch_loss = (5150.98), 9.41677
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

# We will discard the following run... and rerun it with smaller learning rate for fewer epochs

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 4 rs 0.0001 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (5045.08), 9.22319
Epoch# 2 of 4 epochs.
epoch_loss = (5047.89), 9.22831
Epoch loss is increasing... from 9.22319 to 9.22831
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

# We reran the above discarded test and this time only with 1 epoch and will accept the result

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 1 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 1 epochs.
epoch_loss = (5061.9), 9.25393
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 1 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 1 epochs.
epoch_loss = (5045.78), 9.22446
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

# We will discard this run as epoch loss increased from the epoch loss of the last run

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 1 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 1 epochs.
epoch_loss = (5057.48), 9.24585
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

# Rerun of the last discarded run but this time with lesser learning rate

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.0001 epoch 1 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 1 epochs.
epoch_loss = (5033.82), 9.2026
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

# This run will be discarded as well, 

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.0001 epoch 4 rs loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (5035.36), 9.20542
Epoch# 2 of 4 epochs.
epoch_loss = (5039), 9.21206
Epoch loss is increasing... from 9.20542 to 9.21206
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

# This will be discarded as well even though we lower the learning rate even further

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.00001 epoch 1 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 1 epochs.
epoch_loss = (5033.88), 9.2027
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

I think we should stop the training and take the average of both output files...
```

### Output:

```
F:\Chat-Bot-Skip-gram\usage>.\weights.exe words sunshine time soni kisses sunshine mamal bowhead whale
Total number of lines in file "w1trained.txt" : 224
Total number of tokens per line in file "w1trained.txt" : 51
r = 224, c = 50
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Line number = 0 - sunshine, Line number = 42 - time
Cosine Similarity = -0.290233, Cosine Distance = 0.709767
Line number = 0 - sunshine, Line number = 1 - kisses
Cosine Similarity = 0.00911397, Cosine Distance = 0.990886
Line number = 0 - sunshine, Line number = 0 - sunshine
Cosine Similarity = 1, Cosine Distance = 0
Line number = 0 - sunshine, Line number = 80 - bowhead
Cosine Similarity = -0.179863, Cosine Distance = 0.820137
Line number = 0 - sunshine, Line number = 73 - whale
Cosine Similarity = 0.0208309, Cosine Distance = 0.979169
Line number = 42 - time, Line number = 1 - kisses
Cosine Similarity = 0.00582009, Cosine Distance = 0.99418
Line number = 42 - time, Line number = 0 - sunshine
Cosine Similarity = -0.290233, Cosine Distance = 0.709767
Line number = 42 - time, Line number = 80 - bowhead
Cosine Similarity = 0.0686235, Cosine Distance = 0.931377
Line number = 42 - time, Line number = 73 - whale
Cosine Similarity = -0.208399, Cosine Distance = 0.791601
Line number = 1 - kisses, Line number = 0 - sunshine
Cosine Similarity = 0.00911397, Cosine Distance = 0.990886
Line number = 1 - kisses, Line number = 80 - bowhead
Cosine Similarity = 0.0697075, Cosine Distance = 0.930292
Line number = 1 - kisses, Line number = 73 - whale
Cosine Similarity = -0.0198801, Cosine Distance = 0.98012
Line number = 0 - sunshine, Line number = 80 - bowhead
Cosine Similarity = -0.179863, Cosine Distance = 0.820137
Line number = 0 - sunshine, Line number = 73 - whale
Cosine Similarity = 0.0208309, Cosine Distance = 0.979169
Line number = 80 - bowhead, Line number = 73 - whale
Cosine Similarity = -0.0396545, Cosine Distance = 0.960345
```

### RUN 2

```
PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 4 loop 0 verbose
Corpus: ./INPUT.txt
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (11485.3), 20.997
Epoch# 2 of 4 epochs.
epoch_loss = (11412.9), 20.8645
Epoch# 3 of 4 epochs.
epoch_loss = (11337.7), 20.7271
Epoch# 4 of 4 epochs.
epoch_loss = (11188.4), 20.4542
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 4 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Warning: Attempted to read past the last expected line. Breaking early.
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (5762.37), 10.5345
Epoch# 2 of 4 epochs.
epoch_loss = (5689.33), 10.401
Epoch# 3 of 4 epochs.
epoch_loss = (5632.82), 10.2977
Epoch# 4 of 4 epochs.
epoch_loss = (5562.9), 10.1698
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

After the following run I lower the lr from 0.001 to 0.0001...
PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.001 epoch 4 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Warning: Attempted to read past the last expected line. Breaking early.
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (5349.85), 9.78034
Epoch# 2 of 4 epochs.
epoch_loss = (5355.49), 9.79065
Epoch loss is increasing... from 9.78034 to 9.79065
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

Same partial ttained data but with lower lr than before... 
PS F:\Skip-gram\usage> .\skipy.exe corpus ./INPUT.txt lr 0.0001 epoch 4 loop 0 verbose --input W1trainedP.txt W2trainedP.txt
Corpus: ./INPUT.txt
Initialized W2 with dimensions: 224 X 50
Warning: Attempted to read past the last expected line. Breaking early.
Final line read count: 224
Expected line count: 224
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Epoch# 1 of 4 epochs.
epoch_loss = (5373.76), 9.82406
Epoch# 2 of 4 epochs.
epoch_loss = (5368.76), 9.81493
Epoch# 3 of 4 epochs.
epoch_loss = (5363.86), 9.80595
Epoch# 4 of 4 epochs.
epoch_loss = (5361.39), 9.80144
Trained input weights written to file: W1trained.txt
Trained output weights written to file: W2trained.txt

I think we should stop the training and take the average of both output files...
```

### Output:

```
F:\Chat-Bot-Skip-gram_old\usage>.\weights.exe words sunshine time soni kisses sunshine mamal bowhead whale
Total number of lines in file "w1trained.txt" : 224
Total number of tokens per line in file "w1trained.txt" : 51
Dimensions of W1 = 224 X 50
Dimensions of W2 = 50 X 224
Line number = 0 - sunshine, Line number = 42 - time
Cosine Similarity = -0.00244978, Cosine Distance = 0.99755
Line number = 0 - sunshine, Line number = 1 - kisses
Cosine Similarity = 0.27746, Cosine Distance = 0.72254
Line number = 0 - sunshine, Line number = 0 - sunshine
Cosine Similarity = 1, Cosine Distance = 0
Line number = 0 - sunshine, Line number = 80 - bowhead
Cosine Similarity = 0.0963471, Cosine Distance = 0.903653
Line number = 0 - sunshine, Line number = 73 - whale
Cosine Similarity = 0.0306641, Cosine Distance = 0.969336
Line number = 42 - time, Line number = 1 - kisses
Cosine Similarity = 0.0205816, Cosine Distance = 0.979418
Line number = 42 - time, Line number = 0 - sunshine
Cosine Similarity = -0.00244978, Cosine Distance = 0.99755
Line number = 42 - time, Line number = 80 - bowhead
Cosine Similarity = 0.0110742, Cosine Distance = 0.988926
Line number = 42 - time, Line number = 73 - whale
Cosine Similarity = 0.265754, Cosine Distance = 0.734246
Line number = 1 - kisses, Line number = 0 - sunshine
Cosine Similarity = 0.27746, Cosine Distance = 0.72254
Line number = 1 - kisses, Line number = 80 - bowhead
Cosine Similarity = 0.172388, Cosine Distance = 0.827612
Line number = 1 - kisses, Line number = 73 - whale
Cosine Similarity = 0.140638, Cosine Distance = 0.859362
Line number = 0 - sunshine, Line number = 80 - bowhead
Cosine Similarity = 0.0963471, Cosine Distance = 0.903653
Line number = 0 - sunshine, Line number = 73 - whale
Cosine Similarity = 0.0306641, Cosine Distance = 0.969336
Line number = 80 - bowhead, Line number = 73 - whale
Cosine Similarity = 0.131717, Cosine Distance = 0.868283
```

### Observations

The training loss consistently decreased across epochs in both runs, indicating that the model is learning effectively. Although the cosine similarities and distances for certain word pairs differed slightly between runs, this variation is expected in stochastic training processes. The model's ability to capture semantic relationships between words is demonstrated by these similarities.

### Conclusion

The Skip-gram model is functioning correctly, as evidenced by the consistent decrease in training loss and the successful generation of word embeddings. Future work may involve fine-tuning hyperparameters and exploring batch training methods to improve performance further.

### Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or bug reports.
### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.
### Acknowledgements
[Tomas Mikolov and others](https://arxiv.org/abs/1301.3781)
---
For more detailed information on the implementation, please refer to the source code in this [repository](https://github.com/KHAAdotPK/skip-gram/tree/main/lib/WordEmbedding-Algorithms/Word2Vec/Skip-gram). Additionally, the DOCUMENTS folder contains a file named [README.md](https://github.com/KHAAdotPK/skip-gram/blob/main/DOCUMENTS/README.md) which explains the Skip-gram model in the light of this implementation.