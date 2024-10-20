### Training "hidden layer to output layer" word-embeddings using Skip-gram: A C++ Implementation 
---
#### Summary of key steps...
1. **Word Embedding Extraction**: The word embedding for the `center word` is extracted from `W1` using the word's index.
2. **Dot Product**: The extracted word embedding is passed through the hidden layer, where a `dot product` between the `center word's embedding` and the `output weight matrix W2` is calculated. This results in `unnormalized probabilities` (**logits**) for predicting context words.
3. **One-Hot Encoding**: A `one-hot vector` is created to represent the true `context words` associated with the `center word`. This serves as the target for comparison.
4. **Softmax Application**: The **logits** (`raw scores`) are passed through a `softmax function` to convert them into `predicted probabilities` (**y_pred**) for all words in the vocabulary, representing the likelihood of each being a `context word`.
5. **Gradient Computation**: The gradient (grad_u) is calculated by subtracting the one-hot encoded vector from the predicted probability distribution (y_pred). This gradient reflects the error in the model’s prediction.
6. **Weight Update**: Finally, Ggradients are computed for `W2` (**hidden-to-output**) weight matrix. These gradients are used to update the weight matrix using the `learning rate`, ensuring the model learns the correct word embeddings over time.

```C++
/*
    Extract the corresponding word embedding from the weight matrix 𝑊1.
    Instead of directly using a one-hot input vector, this implementation uses a linked list of word pairs.
    Each pair provides the index of the center word, which serves to extract the relevant embedding from 𝑊1.
    The embedding for the center word is stored in the hidden layer vector h.
 */
T* h_ptr = cc_tokenizer::allocator<T>().allocate(W1.getShape().getNumberOfColumns()); // Allocate memory for the hidden layer vector h

/*
    Loop through the columns of W1 to extract the embedding for the center word
 */
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
{
    *(h_ptr + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];

    // Check if the value retrieved for h contains an invalid number (NaN), and throw an exception if true.
    if (_isnanf(h_ptr[i]))
    {        
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: Hidden layer at ") + cc_tokenizer::String<char>("(W1 row) center word index ") +  cc_tokenizer::String<char>(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + cc_tokenizer::String<char>(" and (column index) i -> ") + cc_tokenizer::String<char>(i) + cc_tokenizer::String<char>(" -> [ ") + cc_tokenizer::String<char>("_isnanf() was true") + cc_tokenizer::String<char>("\" ]"));
    } 
}

/*
    Create a collective object h to store the hidden layer vector with dimensions (embedding size, 1)
 */
Collective<T> h = Collective<T>{h_ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}};

/*	
    Represents an intermediat gradient.	 
    This vector has shape (1, len(vocab)), similar to y_pred. 
    It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
    The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
    numerical representation of how likely each word in the vocabulary is to be a context word of a given target word (within the skip-gram model).
    
    "Each element in the resulting vector u represents how much the center word (represented by h) is related to each word in the vocabulary 
    (represented by each column of W2). "

    The variable "u" serves as an intermediary step in the forward pass, representing the activations before applying the “softmax” function to generate the predicted probabilities. It represents internal state in the neural network during the working of "forward pass". This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         
    The dot product gives us the logits or unnormalized probabilities (u), which can then be transformed into probabilities using a softmax function
 */            
Collective<T> u = Numcy::dot(h, W2);

/*
    The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
    The softmax function converts the raw scores into probabilities.

    In `Skip-gram`, this output represents the likelihood of each word being one of the context words for the given center word
 */
Collective<T> y_pred = softmax(u);

/* 
    The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency)
 */
Collective<T> oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});

/*
    The following code block, iterates through the context word indices (left and right) from the pair object.
    For each valid context word index (i), it sets the corresponding element in the oneHot vector to 1.
    This effectively creates a one-hot encoded representation of the context words
 */
for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
{       
    if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
    {
        oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
    }
}
for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
{
    if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
    {
        oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
    }        
}

/* 
    The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency)
*/
Collective<T> grad_u;
/* 
    The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency).
    Compute the gradient for the center word's embedding by subtracting the one-hot vector of the actual context word from the predicted probability distribution.
    `fp.predicted_probabilities` contains the predicted probabilities over all words in the vocabulary.
    `oneHot` is a one-hot vector representing the true context word in the vocabulary.
    The result, `grad_u`, is the error signal for updating the center word's embedding in the Skip-gram model.
    
    what is an error signal?
    -------------------------
    1. For the correct context word (where oneHot is 1), the gradient is (predicted_probabilities - 1), meaning the model's prediction was off by that much.
    2. For all other words (where oneHot is 0), the gradient is simply predicted_probabilities, meaning the model incorrectly assigned a nonzero probability to these words(meaning the model's prediction was off by that much, which the whole of predicted_probability for that out of context word)       
 */        
grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);

/*
    Dimensions of grad_u is (1, len(vocab) without redundency)
    Dimensions of fp.intermediate_activation (1, len(vocab) without redundency)

    Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency). A square matrix        
 */
Collective<T> grad_W2  = Numcy::outer(fp.intermediate_activation, grad_u);

/* 
    Reshape and Update W2: Creates a temporary variable W2_reshaped of type Collective<t> to hold the reshaped output weights held by W2. We need reshaped W2 vector for the later substraction operation between W2 vector and the other one. 

    Function reshape works when first vector is smaller in shape than the other vector
 */
Collective<T> W2_reshaped = Numcy::reshape(W2, bp.grad_weights_hidden_to_output);
/*
    Update Weights
 */
W2_reshaped -= bp.grad_weights_hidden_to_output * lr;
/* 
    Update W2 
*/
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
{
    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getNumberOfColumns(); j++)
    {
        W2[i*W2.getShape().getNumberOfColumns() + j] = W2_reshaped[i*W2_reshaped.getShape().getNumberOfColumns() + j];
    }
}
```
