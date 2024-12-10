/*
    usage/main.hh
    Q@khaa.pk
 */

// .\skipy.exe corpus ./INPUT.txt lr 0.01 epoch 10 rs 0.0000009 loop 0 verbose --input w1-7.dat w2-7.dat --output w1-8.dat w2-8.dat

#include <iostream>

#ifndef WORD_EMBEDDING_ALGORITHMS_SKIP_GRAM_USAGE_MAIN_HH
#define WORD_EMBEDDING_ALGORITHMS_SKIP_GRAM_USAGE_MAIN_HH

#define SKIP_GRAM_DEFAULT_CORPUS_FILE ".\\data\\corpus.txt"
#define TRAINED_INPUT_WEIGHTS_FILE_NAME     "W1trained.dat"
#define TRAINED_OUTPUT_WEIGHTS_FILE_NAME    "W2trained.dat"

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#define GRAMMAR_END_OF_TOKEN_MARKER ' '

#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif
#define GRAMMAR_END_OF_LINE_MARKER '\n'

#ifdef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE
#undef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE
#endif
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 16

// Start with small values (e.g., 0.001) and adjust based on performance.
#ifdef SKIP_GRAM_REGULARIZATION_STRENGTH
#undef SKIP_GRAM_REGULARIZATION_STRENGTH
#endif
#define SKIP_GRAM_REGULARIZATION_STRENGTH 0.001

#ifdef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#undef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#endif
#define SKIP_GRAM_CONTEXT_WINDOW_SIZE 4

#define DEFAULT_TRAINING_LOOP_PATIENCE 3

#include "../lib/argsv-cpp/lib/parser/parser.hh"
#include "../lib/sundry/cooked_read_new.hh"
#include "../lib/sundry/cooked_write_new.hh"
#include "../lib/read_write_weights/header.hh"
#include "../lib/WordEmbedding-Algorithms/Word2Vec/skip-gram/header.hh"

#define COMMAND "h -h help --help ? /? (Displays the help screen, listing available commands and their descriptions.)\n\
v -v version --version /v (Shows the current version of the software.)\n\
e epoch --epoch /e (Sets the epoch count, determining the number of iterations for the training loop.)\n\
corpus --corpus (Specifies the path to the file containing the training data.)\n\
verbose --verbose (Enables detailed output for each operation during execution.)\n\
lr --lr learningrate (Defines the learning rate parameter to control the rate of convergence.)\n\
rs --rs (Sets the regularization strength, used to prevent overfitting.)\n\
loop --loop (Repeats the training loop on previously trained weights at least one additional time.)\n\
batch --batch (Loads initial weights from a specified file, allowing batch processing with predefined starting weights.)\n\
input --input (Specifies the filenames to retrieve the partially input and output trained weights during training.)\n\
output --output (Specifies the filenames to store the input and output trained weights after completion of training.)\n\
ns --ns negative_sampling --negative_sampling (This flag enables negative sampling. It is the process of approximating the softmax function by just drawing a few examples from the set of samples that do not appear in the context of the main word.)\n"

#endif