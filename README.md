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
- **Negative Sampling**: An optimization technique that approximates the softmax function by updating only a small number of negative samples.
- **Learning Rate Decay**: A training optimization technique where the learning rate gradually decreases over time, allowing for faster initial learning and fine-tuning in later stages.
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

### Command Line Options
The Skip-gram model implementation provides various command-line options to customize the training process and control program behavior. Below is a comprehensive list of available options, grouped by their functionality.

#### Basic Options
| Option | Aliases | Description |
|--------|---------|-------------|
| `--help` | `-h`, `help`, `/help`, `?`, `/?` | Displays the help screen, listing available commands and their descriptions |
| `--version` | `-v`, `version`, `/v` | Shows the current version of the software |
| `--verbose` | `verbose` | Enables detailed output for each operation during execution |

#### Training Parameters
| Option | Aliases | Description |
|--------|---------|-------------|
| `--epoch` | `-e`, `epoch`, `/e` | Sets the epoch count, determining the number of iterations for the training loop |
| `--lr` | `lr`, `learningrate` | Defines the learning rate parameter to control the rate of convergence |
| `--rs` | `rs` | Sets the regularization strength, used to prevent overfitting |
| `--loop` | `loop` | Repeats the training loop on previously trained weights at least one additional time |
| `--learning_rate_decay` | `learning_rate_decay`, `--lr_decay` `lr_decay`, `learning_rate_scheduling` | Controls the rate at which learning rate decreases during training. Set to 1 for constant learning rate. When enabled, the learning rate gradually decreases over time, starting with a larger value for faster initial learning and decreasing for fine-tuning |
| `--random_number_generator_seed` | `random_number_generator_seed` | Sets the seed for the random number generator |

#### Input/Output Options
| Option | Aliases | Description |
|--------|---------|-------------|
| `--corpus` | `corpus` | Specifies the path to the file containing the training data |
| `--input` | `input` | Specifies the filenames to retrieve the partially input and output trained weights during training |
| `--output` | `output` | Specifies the filenames to store the input and output trained weights after completion of training |
| `--batch` | `batch` | Loads initial weights from a specified file, allowing batch processing with predefined starting weights |
| `--save_initial_weights` | `save_initial_weights` | Saves the initial "randomly initialized weights" for the embedding matrices W1 W2 to the files before training begins |

#### Advanced Options
| Option | Aliases | Description |
|--------|---------|-------------|
| `--negative_sampling` | `-ns`, `--ns` | Enables negative sampling to approximate the softmax function by drawing a few examples from non-context samples |
| `--show_pairs` | `show_pairs` | Displays pairs of target/center words and their surrounding context words (window size determined by SKIP_GRAM_WINDOW_SIZE) |
| `--shuffle_target_context_pairs` | `shuffle_target_context_pairs` | Shuffles the target/center word and its context words during training at the start of each epoch |

#### Example Usage
```bash
# Basic training with default parameters
./skipy.exe --corpus data/training.txt --epoch 10

# Advanced training with negative sampling and custom learning rate
./skipy.exe --corpus data/training.txt --epoch 20 --lr 0.01 --negative_sampling --verbose

# Continue training from previous weights
./skipy.exe --corpus data/training.txt --input previous_weights.txt --loop

# Training with specific random seed and shuffling
./skipy.exe --corpus data/training.txt --random_number_generator_seed 42 --shuffle_target_context_pairs

# Training with learning rate decay
./skipy.exe --corpus data/training.txt --epoch 20 --lr 0.01 --learning_rate_decay 0.95
```

### TODO
Currently, there are no pending major implementation tasks. Check the issues page for any minor improvements or bug fixes that need attention.

### Experimental Features
The following features have been implemented but are currently in an experimental state and need further testing:

1. **Negative Sampling**: An optimization technique that approximates the softmax function by updating only a small number of negative samples. While implemented, this feature requires additional testing and validation before being recommended for production use.

2. **Regularization**: Implementation of techniques to prevent overfitting. Currently available but undergoing testing to ensure optimal performance and reliability.

⚠️ **Warning**: These features are provided for experimental purposes only. Use them with caution as they may not perform as expected in all scenarios. We recommend waiting for thorough testing and validation before using them in production environments.

### Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or bug reports.

### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.

### Acknowledgements
[Tomas Mikolov and others](https://arxiv.org/abs/1301.3781)

---
For more detailed information on the implementation, please refer to the source code in this [repository](https://github.com/KHAAdotPK/skip-gram/tree/main/lib/WordEmbedding-Algorithms/Word2Vec/Skip-gram). Additionally, the DOCUMENTS folder contains a file named [README.md](https://github.com/KHAAdotPK/skip-gram/blob/main/DOCUMENTS/README.md) which explains the Skip-gram model in the light of this implementation.