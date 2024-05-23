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
1. [ala_exception](https://github.com/KHAAdotPK/ala_exception)
2. [allocator](https://github.com/KHAAdotPK/allocator)
3. [argsv-cpp](https://github.com/KHAAdotPK/argsv-cpp)
4. [corpus](https://github.com/KHAAdotPK/corpus)
5. [csv](https://github.com/KHAAdotPK/csv)
6. [Numcy](https://github.com/KHAAdotPK/Numcy)
7. [parser](https://github.com/KHAAdotPK/parser)
8. [string](https://github.com/KHAAdotPK/string)
9. [sundry](https://github.com/KHAAdotPK/sundry)

In the **lib** folder, there is a small batch file named **PULL.cmd**. Change into the lib folder and at the command prompt execute this file. It will try to clone all the above-mentioned dependencies.
### Getting Started
Follow these steps to set up and run the Skip-gram model:
1. **Clone this repository**:
```BASH
git clone https://github.com/KHAAdotPK/skip-gram.git
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
# build process should result in a executable file named ./spikey.exe
```
4 **Start training the model**:
```BASH
./RUN.cmd
```
### Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or bug reports.
### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.
### Acknowledgements
[Tomas Mikolov and others](https://arxiv.org/abs/1301.3781)

---
For more detailed information on the implementation, please refer to the source code in this [repository](https://github.com/KHAAdotPK/skip-gram/tree/main/lib/WordEmbedding-Algorithms/Word2Vec/Skip-gram). Additionally, the DOCUMENTS folder contains a file named [README.md](https://github.com/KHAAdotPK/skip-gram/blob/main/DOCUMENTS/README.md) which explains the Skip-gram model in the light of this implementation.