/*
    lib/WordEmbedding-Algorithms/Word2Vec/skip-gram/skip-gram.hh
    Q@khaa.pk
 */

#ifndef WORD_EMBEDDING_ALGORITHMS_SKIP_GRAM_HH
#define WORD_EMBEDDING_ALGORITHMS_SKIP_GRAM_HH

#define GO_FOR_L1_CODE_BLOCK

#include "header.hh"

template<typename E>
struct backward_propogation; 

template<typename E>
struct forward_propogation 
{
    /*
        In the first constructor, forward_propagation(),
        member variables hidden_layer_vector, predicted_probabilities, and intermediate_activation
        are initialized directly in the initialization list.
        This approach is cleaner and more efficient than assigning them inside the constructor body.
     */
    forward_propogation(void) : hidden_layer_vector(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), predicted_probabilities(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), intermediate_activation(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {        
    }

    /*
        TODO, 
        Use of Initialization Lists: Utilize constructor initialization lists to initialize
        member variables rather than assigning them inside the constructor body. This improves efficiency and readability...
        implemented but still commented out from the implementation of function.
     */
    //forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u) : hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u)
    forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u) /*: hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u) */
    {           
        E* ptr = NULL;

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(h.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < h.getShape().getN(); i++)
            {
                ptr[i] = h[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        hidden_layer_vector = Collective<E>{ptr, h.getShape().copy()};

        try
        {                 
            ptr = cc_tokenizer::allocator<E>().allocate(y_pred.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < y_pred.getShape().getN(); i++)
            {
                ptr[i] = y_pred[i];
            }
        } 
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        predicted_probabilities = Collective<E>{ptr, y_pred.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(u.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < u.getShape().getN(); i++)
            {
                ptr[i] = u[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        intermediate_activation = Collective<E>{ptr, u.getShape().copy()};
    }

    forward_propogation<E>(forward_propogation<E>& other) 
    {           
        E* ptr = cc_tokenizer::allocator<E>().allocate(other.hidden_layer_vector.getShape().getN());
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.hidden_layer_vector.getShape().getN(); i++)
        {
            ptr[i] = other.hidden_layer_vector[i];
        }
        hidden_layer_vector = Collective<E>{ptr, other.hidden_layer_vector.getShape().copy()};

        ptr = cc_tokenizer::allocator<E>().allocate(other.predicted_probabilities.getShape().getN());
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.predicted_probabilities.getShape().getN(); i++)
        {
            ptr[i] = other.predicted_probabilities[i];
        }
        predicted_probabilities = Collective<E>{ptr, other.predicted_probabilities.getShape().copy()};

        ptr = cc_tokenizer::allocator<E>().allocate(other.intermediate_activation.getShape().getN());
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.intermediate_activation.getShape().getN(); i++)
        {
            ptr[i] = other.intermediate_activation[i];
        }
        intermediate_activation = Collective<E>{ptr, other.intermediate_activation.getShape().copy()};
    }

    forward_propogation<E>& operator= (forward_propogation<E>& other)    
    { 
        if (this == &other)
        {
            return *this;
        }

        E* ptr = NULL;          

        try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.hidden_layer_vector.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.hidden_layer_vector.getShape().getN(); i++)
            {
                ptr[i] = other.hidden_layer_vector[i];
            }        
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        hidden_layer_vector = Collective<E>{ptr, other.hidden_layer_vector.getShape().copy()};

        try
        {                
            ptr = cc_tokenizer::allocator<E>().allocate(other.predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.predicted_probabilities[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        predicted_probabilities = Collective<E>{ptr, other.predicted_probabilities.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(other.intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.intermediate_activation[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        intermediate_activation = Collective<E>{ptr, other.intermediate_activation.getShape().copy()};
        
        return *this;
    }
    
    /*
        Hidden Layer Vector accessor methods
     */
    E hlv(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= hidden_layer_vector.getShape().getN())
        {
            throw ala_exception("forward_propogation::hlv() Error: Provided index value is out of bounds.");
        }

        return hidden_layer_vector[((i/hidden_layer_vector.getShape().getNumberOfColumns())*hidden_layer_vector.getShape().getNumberOfColumns() + i%hidden_layer_vector.getShape().getNumberOfColumns())];
    }
    DIMENSIONS hlvShape(void)
    {
        return *(hidden_layer_vector.getShape().copy());
    }

    /*
        Predicted Probabilities accesssor methods
     */
    E pb(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= predicted_probabilities.getShape().getN())
        {
            throw ala_exception("forward_propogation::pb() Error: Provided index value is out of bounds.");
        }

        return predicted_probabilities[((i/predicted_probabilities.getShape().getNumberOfColumns())*predicted_probabilities.getShape().getNumberOfColumns() + i%predicted_probabilities.getShape().getNumberOfColumns())];
    }
    DIMENSIONS pbShape(void)
    {
        return *(predicted_probabilities.getShape().copy());
    }

     /*
        Intermediate Activation accesssor methods
     */
    E ia(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= intermediate_activation.getShape().getN())
        {
            throw ala_exception("forward_propogation::ia() Error: Provided index value is out of bounds.");
        }

        return intermediate_activation[((i/intermediate_activation.getShape().getNumberOfColumns())*intermediate_activation.getShape().getNumberOfColumns() + i%intermediate_activation.getShape().getNumberOfColumns())];
    }
    DIMENSIONS iaShape(void)
    {
        return *(intermediate_activation.getShape().copy());
    }

    /*
        Declare forward as a friend function within the struct. It is templated, do we need it like this.
     */    
    /*
        Documentation Note:
        -------------------
        The default argument for the template parameter is causing the following error during compilation:
    
        D:\ML\Embedding-Algorithms\Word2Vec\skip-gram\ML\Embedding-Algorithms\Word2Vec\skip-gram\skip-gram.hh(263): warning C4348: 'forward': redefinition of default parameter: parameter 1
        D:\ML\Embedding-Algorithms\Word2Vec\skip-gram\ML\Embedding-Algorithms\Word2Vec\skip-gram\skip-gram.hh(355): note: see declaration of 'forward'
        D:\ML\Embedding-Algorithms\Word2Vec\skip-gram\ML\Embedding-Algorithms\Word2Vec\skip-gram\skip-gram.hh(272): note: the template instantiation context (the oldest one first) is
        main.cpp(169): note: see reference to class template instantiation 'forward_propagation<double>' being compiled

        This error occurs at compile time because the friend declaration and the actual definition of the function both use the default argument for the template parameter. 
        To resolve this error, remove the default argument from either the friend declaration or the definition. 

        Example problematic friend declaration:
    
        template <typename T = double>
        friend forward_propagation<T> forward(Collective<T>&, Collective<T>&, CORPUS_REF, WORDPAIRS_PTR, bool) throw (ala_exception);

        Additional details about the friend declaration:
        The above friend declaration is ineffective because no instance of the vector/composite class is being passed to this function as an argument.
        Therefore, the function cannot access the private or protected members of the vector/composite class it is declared as a friend of.
     */

    template <typename T>
    friend backward_propogation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propogation<T>&, WORDPAIRS_PTR, bool) throw (ala_exception);

    /*
        TODO, uncomment the following statement and make all variables/properties of this vector private.
     */                       
    /*private:*/
        /*
            In the context of our CBOW/Skip-Gram model, h refers to the hidden layer vector obtained by averaging the embeddings of the context words.
            It is used in both the forward and backward passes of the neural network.

            The shape of this array is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), 
            a single row vector with the size of the embedding dimension.
         */
        //E* h;
        Collective<E> hidden_layer_vector;
        /*
            y_pred is a Numcy array of predicted probabilities of the output word given the input context. 
            In our implementation, it is the output of the forward propagation step.

            The shape of this array is (1, len(vocab)), indicating a single row vector with the length of the vocabulary and 
            where each element corresponds to the predicted probability of a specific word.
         */
        //E* y_pred;
        Collective<E> predicted_probabilities;  

        /*	
            Represents an intermediat gradient.	 
            This vector has shape (1, len(vocab)), similar to y_pred. 
            It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
            The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
            numerical representation of how likely each word in the vocabulary is to be a context word of a given target 
            word (within the skip-gram model).

            The variable "u" serves as an intermediary step in the forward pass, representing the activations before applying 
            the “softmax” function to generate the predicted probabilities. 

            It represents internal state in the neural network during the working of "forward pass".
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         */
        //E* u;
        Collective<E> intermediate_activation;  
};

/*
    The following structure is a container designed to hold gradients calculated during backpropagation
    in a two-layer neural network used for word embeddings. The presence of grad_W1 and grad_W2 implies a 
    neural network with two layers. W1 represents the weights between the input layer and the first hidden layer, 
    and W2 represents the weights between the first hidden layer and the output layer.
    - The gradients (partial derivatives of the loss function) with respect to the network's weights
    - Backpropagation, this structure plays a crucial role in backpropagation, 
      an algorithm used to train neural networks. Backpropagation calculates the 
      gradients (partial derivatives of the loss function) with respect to the network's weights.
      These gradients are then used to update the weights in a way that minimizes the loss function

    In summary, the backward_propogation<E> structure is a container designed to hold gradients calculated during 
                backpropagation in a two-layer neural network used for word embeddings.  
 */
template<typename E>
struct backward_propogation 
{  
    /*
        In the first constructor, forward_propagation(),
        member variables hidden_layer_vector, predicted_probabilities, and intermediate_activation
        are initialized directly in the initialization list.
        This approach is cleaner and more efficient than assigning them inside the constructor body.
     */         
    backward_propogation() : grad_weights_input_to_hidden(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_weights_hidden_to_output(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {
        
    }

    /*
        TODO, 
        Use of Initialization Lists: Utilize constructor initialization lists to initialize
        member variables rather than assigning them inside the constructor body. This improves efficiency and readability...
        implemented but still commented out from the implementation of function.
     */
    backward_propogation(Collective<E>& grad_W1, Collective<E>& grad_W2, Collective<E>& grad_center_word) /*: grad_weights_input_to_hidden(grad_W1), grad_weights_hidden_to_output(grad_W2), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})*/
    {
        E* ptr = NULL;

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(grad_W1.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getN(); i++)
            {
                ptr[i] = grad_W1[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_input_to_hidden = Collective<E>{ptr, grad_W1.getShape().copy()};

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(grad_W2.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W2.getShape().getN(); i++)
            {
                ptr[i] = grad_W2[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_hidden_to_output = Collective<E>{ptr, grad_W2.getShape().copy()};

        //grad_hidden_with_respect_to_center_word = Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}};

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(grad_center_word.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_center_word.getShape().getN(); i++)
            {
                ptr[i] = grad_center_word[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_hidden_with_respect_to_center_word = Collective<E>{ptr, grad_hidden_with_respect_to_center_word.getShape().copy()};
    }

    backward_propogation<E>& operator= (backward_propogation<E>& other)    
    { 
        if (this == &other)
        {
            return *this;
        }

        E* ptr = NULL;

        try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.grad_weights_input_to_hidden.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.grad_weights_input_to_hidden.getShape().getN(); i++)
            {
                ptr[i] = other.grad_weights_input_to_hidden[i];
            }        
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_input_to_hidden = Collective<E>{ptr, other.grad_weights_input_to_hidden.getShape().copy()};

        try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.grad_weights_hidden_to_output.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.grad_weights_hidden_to_output.getShape().getN(); i++)
            {
                ptr[i] = other.grad_weights_hidden_to_output[i];
            }        
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_hidden_to_output = Collective<E>{ptr, other.grad_weights_hidden_to_output.getShape().copy()};

         try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.grad_hidden_with_respect_to_center_word.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.grad_hidden_with_respect_to_center_word.getShape().getN(); i++)
            {
                ptr[i] = other.grad_hidden_with_respect_to_center_word[i];
            }        
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_hidden_with_respect_to_center_word = Collective<E>{ptr, other.grad_hidden_with_respect_to_center_word.getShape().copy()};

        return *this;
    }

    /*
       Gradiant Weights Input to Hidden accessor methods
     */
    E gw1(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= grad_weights_input_to_hidden.getShape().getN())
        {
            throw ala_exception("forward_propogation::gw1() Error: Provided index value is out of bounds.");
        }

        return grad_weights_input_to_hidden[((i/grad_weights_input_to_hidden.getShape().getNumberOfColumns())*grad_weights_input_to_hidden.getShape().getNumberOfColumns() + i%grad_weights_input_to_hidden.getShape().getNumberOfColumns())];
    }
    DIMENSIONS gw1Shape(void)
    {
        return *(grad_weights_input_to_hidden.getShape().copy());
    }

    /*        
        Gradiant Weights Hidden to Output accessor methods
     */
    E gw2(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= grad_weights_hidden_to_output.getShape().getN())
        {
            throw ala_exception("forward_propogation::gw2() Error: Provided index value is out of bounds.");
        }

        return grad_weights_hidden_to_output[((i/grad_weights_hidden_to_output.getShape().getNumberOfColumns())*grad_weights_hidden_to_output.getShape().getNumberOfColumns() + i%grad_weights_hidden_to_output.getShape().getNumberOfColumns())];
    }
    DIMENSIONS gw2Shape(void)
    {
        return *(grad_weights_hidden_to_output.getShape().copy());
    }

     /*        
        Gradiant Hidden with respect_to Center Word accessor methods
     */
    E ghcw(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= grad_hidden_with_respect_to_center_word.getShape().getN())
        {
            throw ala_exception("forward_propogation::ghcw() Error: Provided index value is out of bounds.");
        }

        return grad_hidden_with_respect_to_center_word[((i/grad_hidden_with_respect_to_center_word.getShape().getNumberOfColumns())*grad_hidden_with_respect_to_center_word.getShape().getNumberOfColumns() + i%grad_hidden_with_respect_to_center_word.getShape().getNumberOfColumns())];
    }
    DIMENSIONS ghcwShape(void)
    {
        return *(grad_hidden_with_respect_to_center_word.getShape().copy());
    }

    /*        
        Declare backward as a friend function within the struct. It is templated, do we need it like this.
     */
    template <typename T>
    friend backward_propogation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propogation<T>&, WORDPAIRS_PTR, bool) throw (ala_exception);
    
    /*
        TODO, uncomment the following statement and make all variables/properties of this vector private.
     */
    /*private:*/
        /*
            Both arrays has shape which is (corpus::len(), REPLIKA_HIDDEN_SIZE) and (REPLIKA_HIDDEN_SIZE, corpus::len()) respectovely
         */
        //E* grad_W1;
        /*
            Stores the gradients(The gradients (partial derivatives of the loss function) with respect to the network's weights)
            for the first layer weights (W1)
         */
        //Collective<E> grad_W1;
        /*
         * grad_weights_input_to_hidden: This collective object stores the gradients with respect to the weights between the input layer and the hidden layer (W1).
         * It has a shape of (corpus::len(), REPLIKA_HIDDEN_SIZE).
         */
        Collective<E> grad_weights_input_to_hidden;
        //E* grad_W2;
        /*
            Similar to grad_W1, this member stores the gradients for the second layer weights (W2)
         */
        //Collective<E> grad_W2;
        /*
         * grad_weights_hidden_to_output: This collective object stores the gradients with respect to the weights between the hidden layer and the output layer (W2).
         * It has a shape of (REPLIKA_HIDDEN_SIZE, corpus::len()).
         */
        Collective<E> grad_weights_hidden_to_output;
        /*
            Which are the gradients of the loss function with respect to the first layer weights, second layer weights, and the center word input, respectively.
            (REPLIKA_VOCABULARY_LENGTH,, SKIP_GRAM_HIDDEN_SIZE)
         */
        /*
            This member stores the gradients with respect to the center word input (likely the word used as a reference in the word embedding task)
         */
        //E* grad_h_with_respect_to_center_or_target_word;
        //Collective<E> grad_h_with_respect_to_center_or_target_word;
        /*
         * grad_hidden_with_respect_to_center_word: This collective object stores the gradients with respect to the center word input (the word used as a reference in the word embedding task).
         * It has a shape of (REPLIKA_VOCABULARY_LENGTH, SKIP_GRAM_HIDDEN_SIZE).
         */
        Collective<E> grad_hidden_with_respect_to_center_word;
};

/*
    The softmax function is a mathematical function that takes a vector of real numbers as input and normalizes
    them into a probability distribution.
    This distribution ensures that all the output values lie between 0 and 1, and they sum up to 1. 
    It's particularly useful in scenarios where you want to interpret the output as probabilities,
    such as the probabilities of a word belonging to different categories.
 */
template <typename T>
Collective<T> softmax(Collective<T>& a, bool verbose = false) throw (ala_exception)
{
    Collective<T> m; // max
    Collective<T> a_m; // a minus m 
    Collective<T> e_a_m; // exp over a_m
    Collective<T> s_e_a_m; // sum of e_a_m
    Collective<T> e_a_minus_max_divided_by_e_a_minus_max_sum;    

    try
    {
        m = Numcy::max(a);
        a_m = Numcy::subtract(a, m); 
        e_a_m = Numcy::exp(a_m);  
        s_e_a_m = Numcy::sum(e_a_m);
        /*
            m is max
            a_m, a minus m
            e_a_m, exp over a_m
            s_e_a_m, sum of e_a_m
         */  
        e_a_minus_max_divided_by_e_a_minus_max_sum = Numcy::divide(e_a_m, s_e_a_m);     
    }
    catch(ala_exception& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("softmax() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    
    return e_a_minus_max_divided_by_e_a_minus_max_sum;
} 

/*
    Performs part of the forward propagation step in a Skip-gram model
    
    @W1, embedding matrix (Collective object) 
    @W2, output layer weight matrix (Collective object)
    @vocab, instance of corpus class
    @pair, a pointer to a word pair object (containing center word index and context word indices) 
 */
template <typename T = double>
forward_propogation<T> forward(Collective<T>& W1, Collective<T>& W2, CORPUS_REF vocab, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("forward() Error: Index of center word is out of bounds of W1.");
    }

    Collective<T> h;
    Collective<T> u;
    Collective<T> y_pred;
        
    try 
    {
        /*
            Extract the corresponding word embedding from the weight matrix 𝑊1.
            Instead of directly using a one-hot input vector, this implementation uses a linked list of word pairs.
            Each pair provides the index of the center word, which serves to extract the relevant embedding from 𝑊1.
            The embedding for the center word is stored in the hidden layer vector h.
         */
        T* h_ptr = cc_tokenizer::allocator<T>().allocate(W1.getShape().getNumberOfColumns());
        // Loop through the columns of W1 to extract the embedding for the center word
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
        {
            *(h_ptr + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];

            if (_isnanf(h_ptr[i]))
            {        
                throw ala_exception(cc_tokenizer::String<char>("forward() Error: Hidden layer at ") + cc_tokenizer::String<char>("(W1 row) center word index ") +  cc_tokenizer::String<char>(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + cc_tokenizer::String<char>(" and (column index) i -> ") + cc_tokenizer::String<char>(i) + cc_tokenizer::String<char>(" -> [ ") + cc_tokenizer::String<char>("_isnanf() was true") + cc_tokenizer::String<char>("\" ]"));
            }             
        }

        h = Collective<T>{h_ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}};
        /*	
            Represents an intermediat gradient.	 
            This vector has shape (1, len(vocab)), similar to y_pred. 
            It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
            The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
            numerical representation of how likely each word in the vocabulary is to be a context word of a given target 
            word (within the skip-gram model).

            The variable "u" serves as an intermediary step in the forward pass, representing the activations before applying 
            the “softmax” function to generate the predicted probabilities. 

            It represents internal state in the neural network during the working of "forward pass".
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         
            The dot product gives us the logits or unnormalized probabilities (u), 
            which can then be transformed into probabilities using a softmax function
         */            
        u = Numcy::dot(h, W2);        
        /*
            The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
            The softmax function converts the raw scores into probabilities.

            Ensure that y_pred has valid probability values (between 0 and 1). The implementation is correct and that it normalizes the probabilities correctly, 
            i.e., all values should sum up to 1. If there's a bug in the softmax calculation, it might return incorrect values (like 0 or very small numbers).

            In `Skip-gram`, this output represents the likelihood of each word being one of the context words for the given center word.
         */
        y_pred = softmax(u);
                        
        cc_tokenizer::allocator<T>().deallocate(h_ptr);
        h_ptr = NULL;
    }
    catch (std::length_error& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (std::bad_alloc& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (ala_exception& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
                
    return forward_propogation<T>{h, y_pred, u};
}

template <typename T = double>
backward_propogation<T> backward(Collective<T>& W1, Collective<T>& W2, CORPUS_REF vocab, forward_propogation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{
    /* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    Collective<T> oneHot;
    /* The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency) */
    Collective<T> grad_u;
    /*
        Dimensions of grad_u is (1, len(vocab) without redundency)
        Dimensions of fp.intermediate_activation (1, len(vocab) without redundency)

        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)        
     */
    Collective<T> grad_W2;
    /*
        Dimensions of W2 is (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, len(vocab) without redundency)
        Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)        
     */
    Collective<T> W2_T;
    /*
       Dimensions of grad_u is (1, len(vocab) without redundency)
       Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)

       Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<T> grad_h;
    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<T> grad_W1;

    /*
        Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
        This creates a zero-filled column vector with a length equal to the vocabulary size
     */
    try 
    {
        oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});

        /*
            The following code block, iterates through the context word indices (left and right) from the pair object.
            For each valid context word index (i), it sets the corresponding element in the oneHot vector to 1.
            This effectively creates a one-hot encoded representation of the context words.
            
            Note: We are only handling unique tokens. Therefore, in some cases,
            the number of one-hot encoded vectors may be less than the total number
            of context words. This occurs when a single one-hot vector represents 
            multiple occurrences of the same token within the context.            
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
            Note: We are only handling unique tokens. Therefore, in some cases,
            the number of one-hot encoded vectors may be less than the total number
            of context words. This occurs when a single one-hot vector represents 
            multiple occurrences of the same token within the context.

            for (int i = 0; i < vocab.numberOfUniqueTokens(); i++)
            {
                if (oneHot[i] == 1)
                {
                    std::cout<< oneHot[i] << ", ";
                }
            }
            std::cout<< std::endl;
         */
    
        /* 
            The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency)

            Compute the gradient for the center word's embedding by subtracting the one-hot vector of the actual context word
            from the predicted probability distribution.
            `fp.predicted_probabilities` contains the predicted probabilities over all words in the vocabulary. 
                -> If fp.predicted_probabilities[i] is a small probability (close to zero like 1.70794e-18), it means the model is quite confident that class i is not the correct label.
                -> Floating-Point Precision: Very small probabilities close to zero (like 1.7×10e^−18) can sometimes appear as exactly zero due to precision limits, 
                   but this is generally fine for gradient computation as the training process accounts for it.
            `oneHot` is a one-hot vector representing the true context word in the vocabulary.
            The result, `grad_u`, is the error signal for updating the center word's embedding in the Skip-gram model.

            what is an error signal?
            -------------------------
            1. For the correct context word (where oneHot is 1), the gradient is (predicted_probabilities - 1), meaning the model's prediction was off by that much.
            2. For all other words (where oneHot is 0), the gradient is simply predicted_probabilities, meaning the model incorrectly assigned a nonzero probability to these words(meaning the model's prediction was off by that much, which the whole of predicted_probability for that out of context word).
            3. If the large gradients cause instability, consider gradient clipping. So, a gradient of −1 or even 1 in this context is manageable and not unusual.
               When we mention "large gradients" in the context of gradient clipping, we’re generally referring to situations where values might spike significantly higher,
               leading to instability—often in ranges much higher than 1, sometimes reaching orders of magnitude greater depending on the scale of your loss function and the learning rate.
         */        
        grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);

        /*
        for (int i = 0; i < vocab.numberOfUniqueTokens(); i++)
        {                        
            //std::cout<< grad_u[i] << ", ";

            if (_isnanf(grad_u[i]))
            {        
                throw ala_exception(cc_tokenizer::String<char>("backward() Error: gradient for the center word ") + cc_tokenizer::String<char>("(grad_u) at index ") +  cc_tokenizer::String<char>(i) + cc_tokenizer::String<char>("_isnanf() was true"));
            }
            else if (grad_u[i] == 0)
            {
                throw ala_exception("backward() Error: gradient for the center word is zero in grad_u"); 
            }

            if (oneHot[i] == 1)
            {
                std::cout << "y_pred[i] = " << fp.predicted_probabilities[i] << "  <<--->>  " <<  fp.predicted_probabilities[i] - oneHot[i] << " , " << grad_u[i] << " --- ";
            }  
        }
        std::cout<< std::endl;
         */

        grad_W2 = Numcy::outer(fp.intermediate_activation, grad_u);
        
        W2_T = Numcy::transpose(W2);

        /*
        std::cout<< grad_u.getShape().getNumberOfColumns() << " ------- " << grad_u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
        std::cout<< W2_T.getShape().getNumberOfColumns() << " ------- " << W2_T.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
         */
                
        grad_h = Numcy::dot(grad_u, W2_T);

        /*
        std::cout<< grad_h.getShape().getNumberOfColumns() << " ------------------------- " << grad_h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
         */

        grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});

        /*
            Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
            Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
         */
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
        {
            grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
        }
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)
     */ 
    return backward_propogation<T>{grad_W1, grad_W2, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}};
}

/*
    Breakdown of code
    -----------------
    Training Loop: The loop iterates over the specified number of epochs and performs training steps in each iteration.
    Shuffling Word Pairs: The training data (word pairs) are shuffled before each epoch to prevent biases in weight updates.
                          This is a good practice to ensure the model learns effectively from the data.
    Forward and Backward Propagation: Inside the loop, forward propagation is performed to calculate the hidden layer activation
                          and predicted probabilities using the current word pair, embedding matrix, and output weights.
                          Then, backward propagation calculates the gradients with respect to the input and output layer weights.
    Updating Weights: After calculating gradients, the weights are updated using gradient descent. Both the embedding matrix (W1)
                      and the output weights (W2) are updated.
    Updating Weights: After calculating gradients, the weights are updated using gradient descent.
                      Both the embedding matrix (W1) and the output weights (W2) are updated. 
    Loss Calculation: The negative log-likelihood (NLL) loss function is used to evaluate the model's performance.
                      Lower values indicate better performance. 

    Limitations and missing components
    ----------------------------------
    Optimization Algorithms: The code uses a basic gradient descent approach for weight updates.
                             More advanced optimization algorithms like Adam, RMSProp,
                             or Adagrad could be implemented to improve convergence speed and performance.
    Evaluation Metrics: While the loss function provides feedback on the model's performance during training,
                        additional evaluation metrics like accuracy or precision-recall could be included to assess
                        the model's effectiveness in capturing semantic relationships.
    Debugging and Logging: While the code includes verbose mode for printing additional information during training,
                           more comprehensive debugging and logging mechanisms could be implemented to facilitate
                           troubleshooting and monitoring during training.

    Next TODO, Implement L1/L2 regularization technique
    ---------------------------------------------------                       
    Regularization techniques: Implementing L1 and L2 regularization techniques in the Skip-gram model depends on
                               various factors such as the specific requirements of your application, the size of the dataset,
                               the complexity of the model, and the desired level of generalization.

                               Considerations for implementing regularization in your code
                               -----------------------------------------------------------
                               Overfitting: Regularization techniques like L1 and L2 can help prevent overfitting,
                                            especially when dealing with large datasets or complex models.
                                            If you observe that your model is performing well on the training data
                                            but poorly on unseen data (validation or test set), regularization might be beneficial.
                               Model Complexity: Skip-gram models with a large number of parameters or embedding dimensions are more
                                                 prone to overfitting. Regularization techniques can help control the complexity of
                                                 the model and improve generalization.
                               Training Data Size: If you have a relatively small training dataset, regularization can be particularly
                                                   useful in preventing the model from memorizing the training examples and instead learning
                                                   more generalizable patterns. 
                               Generalization: Regularization encourages the model to learn simpler patterns that generalize better to
                                               unseen data. If you're interested in building a Skip-gram model that performs well on a wide
                                               range of contexts and words, regularization might be beneficial.

                               Consideration against implementing regularization technique in you code
                               -----------------------------------------------------------------------                
                               Computational Resources: Regularization introduces additional computational overhead during training, especially
                                             for L1 regularization, which involves absolute value penalties. Consider whether your computational
                                             resources can handle the increased training time.
                               Hyperparameter Tuning: Regularization introduces additional hyperparameters (e.g., regularization strength) that
                                                      need to be tuned alongside other model hyperparameters. Ensure you have a proper validation
                                                      strategy to tune these hyperparameters effectively. 

                               In summary, while L1 and L2 regularization techniques can be beneficial for controlling overfitting and improving generalization 
                               in Skip-gram models, their implementation should be based on your specific requirements, dataset characteristics, and available
                               computational resources.
                               If you observe signs of overfitting or poor generalization, experimenting with regularization techniques could be worthwhile.                                                                                                   

    Training loop arguments
    -----------------------
    @epoch, number of times the training loop would iterate
    @W1, embedding matrix. Each row in W1 is a unique word's embedding vector, representing its semantic relationship with other words
    @W2, output layer. Weights for predicting context words
    @el, epoch loss
    @vocab, instance of class corpus
    @pairs, inctance of class skip gram pairs. The target/center word and its context words
    @lr, learning rate. The learning rate controls the step size at each iteration of the optimization process
    @rs, regulirazation strength. To prevent the model over-learning from the data
    @t, data type. Used as argument to templated types and functions
    @verbose, when true puts more text on screen to help debug code    
 */
#define SKIP_GRAM_TRAINING_LOOP(epoch, W1, W2, el, vocab, pairs, lr, rs, t, verbose)\
{\
    /* Epoch loop */\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)\
    {\
        /* Initializes the epoch loss to 0 before accumulating errors from word pairs */\
        el = 0;\
        /* Conditional block that prints the current epoch number if verbose is True */\
        if (verbose)\
        {\
            std::cout<< "Epoch# " << i << " of " << epoch << " epochs." << std::endl;\
        }\
        /* Shuffle Word Pairs: Shuffles the training data (word pairs) before each epoch to avoid biases in weight updates */\
        Numcy::Random::shuffle<PAIRS>(pairs, pairs.get_number_of_word_pairs());\
        /* Iterates through each word pair in the training data  */\
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())\
        {\
            /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
            forward_propogation<t> fp;\
            backward_propogation<t> bp;\
            /* Reshape and Update W2: Creates a temporary variable W2_reshaped of type Collective<t> to hold the reshaped\
               output weights held by W2. We need reshaped W2 vector for the later substraction operation between W2 vector\
               and the other one */\
            Collective<t> W2_reshaped;\
            try\
            {\
                /* Forward Propagation: The forward function performs forward propagation and calculate the hidden layer\
                   activation and predicted probabilities using the current word pair (pair), embedding matrix (W1),\
                   output weights (W2), vocabulary (vocab), and data type (t). The result is stored in the fp variable.*/\
                fp = forward<t>(W1, W2, vocab, pair);\
                /* Backward Propagation: The backward function performs backward propagation and calculate the gradients\
                   with respect to the input and output layer weights using the forward propagation results (fp), word pair (pair),\
                   embedding matrix (W1), output weights (W2), vocabulary (vocab), and data type (t).\
                   The result is stored in the bp variable. */\
                bp = backward<t>(W1, W2, vocab, fp, pair);\
                /* Reshape W2 so tht it has the same shape as the other vector.\
                   Function reshape works when first vector is smaller in shape than the other vector */\
                W2_reshaped = Numcy::reshape(W2, bp.grad_weights_hidden_to_output);\
                /* Update Weights */\
                W1 -= bp.grad_weights_input_to_hidden * lr;\
                W2_reshaped -= bp.grad_weights_hidden_to_output * lr;\
                /* Update W2 */\
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)\
                {\
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getNumberOfColumns(); j++)\
                    {\
                        W2[i*W2.getShape().getNumberOfColumns() + j] = W2_reshaped[i*W2_reshaped.getShape().getNumberOfColumns() + j];\
                    }\
                }\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
            }\
            /* Loss Function: The Skip-gram model typically uses negative log-likelihood (NLL) as the loss function.\
               In NLL, lower values indicate better performance. */\
            el = el + (-1*log(fp.pb(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)));\
        }\
        std::cout<< "epoch_loss = (" << el << "), " << el/pairs.get_number_of_word_pairs() << std::endl;\
    }\
}\

#endif