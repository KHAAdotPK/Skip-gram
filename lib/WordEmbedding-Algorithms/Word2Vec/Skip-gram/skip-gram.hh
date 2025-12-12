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
    forward_propogation(void) : hidden_layer_vector(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), 
                                predicted_probabilities(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), 
                                intermediate_activation(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), 
                                /*hidden_layer_vector_negative_samples(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}) ,
                                predicted_probabilities_negative_samples(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}),
                                intermediate_activation_neative_samples(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}),*/
                                positive_samples_loss(0),
                                negative_samples_loss(0),
                                positive_negative_epoch_loss(0)
    {        
    }

    forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u, /*Collective<E>& h_ns, Collective<E>& y_pred_ns, Collective<E>& u_ns,*/ E psl, E nsl, E pnel) throw (ala_exception)    
    {
        try
        {        
            hidden_layer_vector = h;
            predicted_probabilities = y_pred;
            intermediate_activation = u;
            /*hidden_layer_vector_negative_samples = h_ns;
            predicted_probabilities_negative_samples = y_pred_ns;
            intermediate_activation_neative_samples = u_ns;*/

            positive_samples_loss = psl;
            negative_samples_loss = nsl;
            positive_negative_epoch_loss = pnel;
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
    }

    /*
        TODO, 
        Use of Initialization Lists: Utilize constructor initialization lists to initialize
        member variables rather than assigning them inside the constructor body. This improves efficiency and readability...
        implemented but still commented out from the implementation of function.
     */
    //forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u) : hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u)
    /*forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u)*/ /*: hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u) */
    /*{           
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
    }*/

    forward_propogation<E>(forward_propogation<E>& other) throw (ala_exception)
    {
        try
        {
            hidden_layer_vector = other.hidden_layer_vector;
            intermediate_activation = other.intermediate_activation;
            predicted_probabilities = other.predicted_probabilities;

            /*hidden_layer_vector_negative_samples = other.hidden_layer_vector_negative_samples;
            intermediate_activation_neative_samples = other.intermediate_activation_neative_samples;
            predicted_probabilities_negative_samples = other.predicted_probabilities_negative_samples;*/

            positive_samples_loss = other.positive_samples_loss;
            negative_samples_loss = other.negative_samples_loss;
            positive_negative_epoch_loss = other.positive_negative_epoch_loss;
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
    }

    /*forward_propogation<E>(forward_propogation<E>& other) 
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
    }*/

    forward_propogation<E>& operator= (forward_propogation<E>& other) throw (ala_exception)   
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
        hidden_layer_vector = Collective<E>{ptr, other.hidden_layer_vector.getShape()};

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
        predicted_probabilities = Collective<E>{ptr, other.predicted_probabilities.getShape()};

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
        intermediate_activation = Collective<E>{ptr, other.intermediate_activation.getShape()};
        
        /*try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.hidden_layer_vector_negative_samples.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.hidden_layer_vector_negative_samples.getShape().getN(); i++)
            {
                ptr[i] = other.hidden_layer_vector_negative_samples[i];
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
        hidden_layer_vector_negative_samples = Collective<E>{ptr, other.hidden_layer_vector_negative_samples.getShape().copy()};
        
        try
        {                
            ptr = cc_tokenizer::allocator<E>().allocate(other.predicted_probabilities_negative_samples.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.predicted_probabilities_negative_samples.getShape().getN(); i++)
            {
                ptr[i] = other.predicted_probabilities_negative_samples[i];
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
        predicted_probabilities_negative_samples = Collective<E>{ptr, other.predicted_probabilities_negative_samples.getShape().copy()};
        
        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(other.intermediate_activation_neative_samples.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.intermediate_activation_neative_samples.getShape().getN(); i++)
            {
                ptr[i] = other.intermediate_activation_neative_samples[i];
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
        intermediate_activation_neative_samples = Collective<E>{ptr, other.intermediate_activation_neative_samples.getShape().copy()};*/

        positive_samples_loss = other.positive_samples_loss;
        negative_samples_loss = other.negative_samples_loss;
        positive_negative_epoch_loss = other.positive_negative_epoch_loss;
        
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
        //std::cout<< "SHAPE OF Y_PRED = Rws = " << predicted_probabilities.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << ", " << predicted_probabilities.getShape().getNumberOfColumns() << std::endl;

        if (i >= predicted_probabilities.getShape().getN())
        {
            throw ala_exception("forward_propogation::pb() Error: Provided index value is out of bounds.");
        }

        // return predicted_probabilities[((i/predicted_probabilities.getShape().getNumberOfColumns())*predicted_probabilities.getShape().getNumberOfColumns() + i%predicted_probabilities.getShape().getNumberOfColumns())];
        return predicted_probabilities[i];
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
    friend backward_propogation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propogation<T>&, WORDPAIRS_PTR, bool, bool) throw (ala_exception);

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

        /*
            Negative samples....
        */
        /* h_negative_samples */
        //Collective<E> hidden_layer_vector_negative_samples;
        /* y_pred_negative_samples */
        //Collective<E> predicted_probabilities_negative_samples;
        /* u_negative_samples */
        //Collective<E> intermediate_activation_neative_samples;

        E positive_samples_loss;
        E negative_samples_loss;
        E positive_negative_epoch_loss;         
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

        //std::cout<< "iN BACKWARD_PROPOATION Constructor: " << grad_W1.getShape().getN() << ", " << grad_W2.getShape().getN() << ", " << grad_center_word.getShape().getN() << std::endl;

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
        grad_weights_input_to_hidden = Collective<E>{ptr, grad_W1.getShape()};

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
        grad_weights_hidden_to_output = Collective<E>{ptr, grad_W2.getShape()};

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
        grad_hidden_with_respect_to_center_word = Collective<E>{ptr, grad_hidden_with_respect_to_center_word.getShape()};
    }

    backward_propogation<E>& operator= (backward_propogation<E>& other)    
    { 
        if (this == &other)
        {
            return *this;
        }

        E* ptr = NULL;

        //std::cout<< "ASSIGNMENT OPERATOR OVERLOADED FOR BACKWARD PROPAGATION CALLED." << std::endl;


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
        grad_weights_input_to_hidden = Collective<E>{ptr, other.grad_weights_input_to_hidden.getShape()};

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
        grad_weights_hidden_to_output = Collective<E>{ptr, other.grad_weights_hidden_to_output.getShape()};

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
        grad_hidden_with_respect_to_center_word = Collective<E>{ptr, other.grad_hidden_with_respect_to_center_word.getShape()};

        //std::cout<< "ASSIGNMENT OPERATOR OVERLOADED FOR BACKWARD PROPAGATION CALLED." << std::endl;

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
    friend backward_propogation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propogation<T>&, WORDPAIRS_PTR, bool, bool) throw (ala_exception);
    
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

template <typename E = double>
void clip_gradients(Collective<E>& grad, AXIS axis = AXIS_NONE, E threshold = SKIP_GRAM_CLIP_GRADIENTS_DEFAULT_THRESHOLD) throw (ala_exception)
{
    if (grad.getShape().getN() == 0)
    {
        throw ala_exception("clip_gradients() Error: Gradient vector is empty.");
    }

    Collective<E> norm;

    switch(axis)
    {
        case AXIS_NONE:
        {                        
            try
            {
                norm = Numcy::LinAlg::norm(grad);
                if (norm.getShape().getN() != 1)
                {
                    throw ala_exception("clip_gradients() Error: \"norm\" vector has incorrect dimensions for \"AXIS_NONE\".");
                }

                if (norm[0] > threshold)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad.getShape().getN() /*norm.getShape().getN()*/; i++)
                    {
                        grad[i] = grad[i] * (threshold/norm[0]);
                    }
                }
            }
            catch(ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("clip_gradient() Error: ") + e.what());
            }            
        }
        break;

        case AXIS_COLUMN:
        {
            try
            {
                 norm = Numcy::LinAlg::norm(grad, AXIS_COLUMN);
                 if (norm.getShape().getN() != grad.getShape().getNumberOfColumns())
                 {
                     throw ala_exception("clip_gradients() Error: \"norm\" vector has incorrect dimensions for \"AXIS_COLUMN\".");
                 }
                                        
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < norm.getShape().getN(); i++)
                {
                    if (norm[i] > threshold)
                    {
                        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < grad.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                        {
                            grad[i + j*grad.getShape().getNumberOfColumns()] = grad[i + j*grad.getShape().getNumberOfColumns()] * (threshold/norm[i]);
                        }
                    }
                }
            }
            catch(ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("clip_gradients() -> ") + e.what());
            }
        }
        break;

        case AXIS_ROWS:
        try
        {
            norm = Numcy::LinAlg::norm(grad, AXIS_ROWS);
            if (norm.getShape().getN() != grad.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
            {
                throw ala_exception("clip_gradients() Error: \"norm\" vector has incorrect dimensions for \"AXIS_ROWS\".");
            }

            for(cc_tokenizer::string_character_traits<char>::size_type i = 0; i < norm.getShape().getN(); i++)
            {
                if (norm[i] > threshold)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < grad.getShape().getNumberOfColumns(); j++)
                    {
                        grad[i*grad.getShape().getNumberOfColumns() + j] = grad[i*grad.getShape().getNumberOfColumns() + j] * (threshold/norm[i]);
                    }
                }
            }
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("clip_gradients() -> ") + e.what());
        }
        break;

        default:
        {
            throw ala_exception("clip_gradients() Error: Invalid axis specified.");
        }
        break;
    }    
}

/**
 * @brief Constructs the negative sampling lookup table using the unigram^(3/4) noise distribution.
 *
 * This function builds a 100-million-entry (1e8) precomputed lookup table for O(1) negative
 * sample generation during Skip-gram and CBOW training with negative sampling — exactly
 * as implemented in the original Google Word2Vec C code (Mikolov et al., 2013).
 *
 * The noise distribution follows the recommendation from:
 *   "Distributed Representations of Words and Phrases and their Compositionality"
 *   (arXiv:1310.4546), Section 3.1:
 *     P(w) ∝ [frequency(w)]^(3/4)
 *
 * This subsampling of frequent words (raising to 0.75) is the single most important trick
 * that made Word2Vec embeddings high-quality and semantically meaningful. It ensures that
 * rare words are oversampled as negatives relative to their raw frequency, dramatically
 * improving analogical reasoning performance.
 *
 * The resulting table enables constant-time negative sample drawing via:
 *     negative_word = negative_sampling_table[rand() % 100000000]
 *
 * @pre Vocabulary must be fully built (head ≠ nullptr and frequencies populated).
 * @post negative_sampling_table contains 1e8 word indices distributed according to U^{3/4}.
 *
 * @throws ala_exception if vocabulary is empty or memory allocation fails.
 *
 * @note This function must be called once after corpus construction and before training.
 *       The table is reused across all epochs and both Skip-gram and CBOW models.
 *
 * @see generateNegativeSamples()
 * @see Tomas Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", 2013
 * @see Original Word2Vec source: https://code.google.com/archive/p/word2vec/
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
Collective<E> buildNegativeSamplesTable (CORPUS& vocab) throw (ala_exception)
{

    COMPOSITE_PTR composite_ptr = vocab.get_composite_ptr(1);

    if (composite_ptr == NULL || vocab.numberOfUniqueTokens() == 0)
    {
        throw ala_exception("buildNegativeSamplesTable(CORPUS&) Error: Vocabulary is empty.");
    }

    double* power = NULL; // Array to hold the powered frequencies 

    double z = 0.0; // Normalization constant
    E *negative_sampling_table = NULL;
    constexpr long long NEGATIVE_SAMPLING_TABLE_SIZE = 100'000'000LL; // Size of the negative sampling table 1e8 – original Word2Vec uses this size

    try
    {                
        power = cc_tokenizer::allocator<double>().allocate(vocab.numberOfUniqueTokens());
        negative_sampling_table = cc_tokenizer::allocator<E>().allocate(NEGATIVE_SAMPLING_TABLE_SIZE);

        // Phase 1: Compute freq^0.75 and Z
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; composite_ptr; composite_ptr = composite_ptr->next, ++i) 
        {
            power[i] = std::pow(composite_ptr->get_frequency(), 0.75); // Using power of 0.75 as per Mikolov et al.'s recommendation
            z += power[i];  
        }
        
        // Phase 2: Fill the negative sampling table
        long long i = 0;
        E wid = 0; // word index
        double cum /* :) */ = power[0] / z; // cumulative probability
        composite_ptr = vocab.get_composite_ptr(1);

        do
        {
            double p = power[wid] / z; // Probability of the current word
            while (i < NEGATIVE_SAMPLING_TABLE_SIZE && i / double(NEGATIVE_SAMPLING_TABLE_SIZE) < cum) {
                negative_sampling_table[i++] = wid;
            }

            cum += p;
            ++wid;            
        }
        while ((composite_ptr = composite_ptr->next) != NULL);

        while (i < NEGATIVE_SAMPLING_TABLE_SIZE) 
        {
            negative_sampling_table[i++] = vocab.numberOfUniqueTokens() - 1;
        }
    }
    catch (const std::bad_alloc& e)
    {
        // CRITICAL: Length constraint violation - system should terminate immediately
        // NO cleanup performed - this is a fatal error requiring process exit
        throw ala_exception(cc_tokenizer::String<char>("buildNegativeSamplesTable(CORPUS&) Error: ") + e.what());
    }
    catch (const std::length_error& e)
    {
        // CRITICAL: Memory allocation failure - system should terminate immediately
        // NO cleanup performed - this is a fatal error requiring process exit
        throw ala_exception(cc_tokenizer::String<char>("buildNegativeSamplesTable(CORPUS&) Error: ") + e.what());
    }
    /*catch (ala_exception& e)
    {
        // Propagate existing ala_exception with additional context
        // NO cleanup performed assuming this is also a critical error
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() -> ") + cc_tokenizer::String<char>(e.what()));
    }*/
  
    return Collective<E>{negative_sampling_table, DIMENSIONS{NEGATIVE_SAMPLING_TABLE_SIZE, 1, NULL, NULL}};
}

/**
 * @brief Generates a batch of negative samples using the precomputed unigram^(3/4) lookup table.
 *
 * This function draws `n` negative word indices in **O(n)** time using the 100-million-entry
 * negative sampling table built by `CORPUS::buildNegativeSamplingTable()`.
 *
 * It implements the **exact negative sampling strategy** from:
 *   Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality"
 *   (arXiv:1310.4546, 2013)
 *
 * Key properties (identical to original Google Word2Vec):
 * - Constant-time sampling via precomputed table
 * - Noise distribution: P(w) ∝ [frequency(w)]^(3/4)
 * - No rejection sampling
 * - Allows negative samples to be the center word (handled in training loop if needed)
 * - Allows duplicates (statistically correct and harmless)
 * - Thread-safe static RNG (Mersenne Twister) seeded from hardware
 *
 * This is the **fast path** that made Word2Vec scalable to billions of words.
 *
 * @tparam E Index type (usually size_t or vocab token ID type)
 * @param negative_sampling_table Reference to the 1e8-entry lookup table
 * @param n Number of negative samples to draw (typically 5–20)
 *
 * @return Collective<E> containing `n` word indices drawn from the U^(3/4) noise distribution
 *
 * @throws ala_exception on memory allocation failure (critical error)
 *
 * @note This function is used by both Skip-gram and CBOW with negative sampling.
 *       It is deliberately minimal and blazing fast — every cycle counts at scale.
 *
 * @warning The table **must** be built with unigram^0.75 weighting.
 *          Uniform or unsmoothed tables will produce garbage embeddings.
 *
 * @see CORPUS::buildNegativeSamplingTable()
 * @see Original Word2Vec source: https://code.google.com/archive/p/word2vec/
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
Collective<E> generateNegativeSamplesFromTable(Collective<E>& negative_sampling_table, E n) throw (ala_exception)
{
    static std::mt19937 rng(std::random_device{}()); // Random number generator
    static std::uniform_int_distribution<long long> dist(0, 99999999LL); // Distribution for sampling from the negative samples table

    E* samples = NULL; // Array to hold the negative samples

    try
    {
        samples = cc_tokenizer::allocator<E>().allocate(n);

        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)
        {
            long long random_index = dist(rng); // Get a random index
            samples[i] = negative_sampling_table[random_index]; // Sample from the negative samples table
        }

        return Collective<E>{samples, DIMENSIONS{n, 1, NULL, NULL}};
    }
    catch (const std::bad_alloc& e)
    {
        // CRITICAL: Length constraint violation - system should terminate immediately
        // NO cleanup performed - this is a fatal error requiring process exit
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples(Collective<E>&, E) Error: ") + e.what());
    }
    catch (const std::length_error& e)
    {
        // CRITICAL: Memory allocation failure - system should terminate immediately
        // NO cleanup performed - this is a fatal error requiring process exit
        throw ala_exception(cc_tokenizer::String<char>("buildNegativeSamplesTable(CORPUS&) Error: ") + e.what());
    }
    catch (ala_exception& e)
    {
        // Propagate existing ala_exception with additional context
        // NO cleanup performed assuming this is also a critical error
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples(Collective<E>&, E) -> ") + cc_tokenizer::String<char>(e.what()));
    }

    return Collective<E>{samples, DIMENSIONS{n, 0, NULL, NULL}};
}


/*
    In Skip-gram, we are predicting the context words from the central word (target).
    So, the negative samples should be words that are not the actual context words for the given target word 
    but could be drawn from the entire vocabulary.

    A better way to phrase it:
    We need to randomly select words from the vocabulary that are not the true context words when given a target word.
    These negative samples help the model learn to distinguish between actual context words and unrelated words.
 */
/**
 * @brief Generates negative samples for multiple target-context word pairs in a Skip-gram model.
 *
 * This function selects negative samples for each target-context word pair by randomly sampling 
 * words from the vocabulary that do not belong to the immediate context of the given target word.
 * Negative sampling helps the Skip-gram model distinguish between true context words 
 * and unrelated words, aiding in the training process.
 *
 * @tparam E The data type for indices, typically an integer type like `size_t` or `int`.
 *
 * @param word_pairs A vector of pointers to target-context word pairs (WORDPAIRS_PTR).
 *                   Each pair contains a target word and its associated context words.
 * @param n The number of negative samples to generate for each target-context word pair.
 *          Defaults to SKIP_GRAM_DEFAULT_NUMBER_OF_NEGATIVE_SAMPLES.
 *
 * @return A map (std::unordered_map) where:
 *         - The key is a pointer to a target-context word pair (WORDPAIRS_PTR).
 *         - The value is a vector of indices (E) representing negative samples 
 *           for the corresponding word pair.
 *
 * @throws ala_exception If memory allocation fails or any other runtime error occurs.
 *
 * Example usage:
 * @code
 * std::vector<WORDPAIRS_PTR> word_pairs = ...; // List of word pairs
 * auto negative_samples_map = generateNegativeSamplesBatch(word_pairs, 5);
 *
 * for (const auto& [pair, samples] : negative_samples_map) {
 *     std::cout << "Negative samples for pair: ";
 *     for (const auto& sample : samples) {
 *         std::cout << sample << " ";
 *     }
 *     std::cout << std::endl;
 * }
 * @endcode
 *
 * Notes:
 * - The function assumes that the vocabulary size is large enough to provide meaningful negative samples.
 * - Random sampling is uniform; for more sophisticated sampling, replace the random generator with a 
 *   frequency-based distribution (e.g., unigram table with power smoothing).
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
Collective<E> generateNegativeSamples(WORDPAIRS_PTR current_word_pair_ptr, CORPUS& vocab, E n = SKIP_GRAM_DEFAULT_NUMBER_OF_NEGATIVE_SAMPLES) throw (ala_exception)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<E> distrib(0, vocab.numberOfUniqueTokens() - 1);
        
    CONTEXTWORDS_PTR left_context = current_word_pair_ptr->getLeft();
    CONTEXTWORDS_PTR right_context = current_word_pair_ptr->getRight();

    cc_tokenizer::string_character_traits<char>::size_type* ptr = NULL; 

    /*
    std::vector<double> power = vocab.getWordProbabilities(); // Ensure word probabilities are initialized

    COMPOSITE_PTR composite_ptr = vocab.get_composite_ptr(1);

    int foo = 0;
    do {

        foo++;
    } 
    while ((composite_ptr = composite_ptr->next) != NULL);

    std::cout<< "Number of unique tokens in the vocabulary = " << vocab.numberOfUniqueTokens() << std::endl; 
    std::cout<< "foo = " << foo << std::endl;
     */
    
    try
    {                
        ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(n);
    }
    catch (const std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: ") + e.what());
    }
    catch (const std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: ") + e.what());
    }

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)
    {
        *(ptr + i) = static_cast<cc_tokenizer::string_character_traits<char>::size_type>(INDEX_NOT_FOUND_AT_VALUE);
    }
    
    // Generate negative samples
    E i = 0;
    for (; i < n;)
    {        
         // Randomly select a central word from the vocabulary, central words are all the unique words
        cc_tokenizer::string_character_traits<char>::size_type central_word_index = distrib(gen);

        if (current_word_pair_ptr->getCenterWord() != (central_word_index + INDEX_ORIGINATES_AT_VALUE))
        {                    
            cc_tokenizer::string_character_traits<char>::size_type j = 0;
            for (; j < SKIP_GRAM_CONTEXT_WINDOW_SIZE;)
            {
                if ((left_context->array[j] == (central_word_index + INDEX_ORIGINATES_AT_VALUE)) || (right_context->array[j] == (central_word_index + INDEX_ORIGINATES_AT_VALUE)))
                {
                    break;
                }

                j++;
            } 

            if (!(j < SKIP_GRAM_CONTEXT_WINDOW_SIZE))
            {
                *(ptr + i) = central_word_index /*+ INDEX_ORIGINATES_AT_VALUE*/;

                i++;
            }                
        }
    }

    return Collective<E>{ptr, DIMENSIONS{n, 1, NULL, NULL}};               
}

/*
    The following implementation faithfully implements the softmax function described in the article... https://arxiv.org/pdf/1411.2738 
 */
/*
    Page 7 of https://arxiv.org/pdf/1411.2738 
    -----------------------------------------
    - uc,j the net input (logit) for the j-th word in the output layer for the c-th context word. It is computed as:
        - uc,j = h . W2j (for just a single "uc,j"), where "h" is the embedding of the center word wI(i-th) and wj(j-th) column of the weight matrix W2.
    - exp(uc,j) the exponential of the logit, ensuring all values are positive.
    - [Sum(for all j from 1 to V) exp(uj)] the sum of exponentials of logits across all words in the vocabulary V normalizing the probabilities.
    - yc,j the softmax output, representing the probability of the j-th word being the context word wO,c given the center/target word wI.

    Why exponentials? 
    - Exponentials amplify the differences between logits, making higher scores dominate in the probability distribution. 
    - This helps the model focus on the most likely target words.

    The Equation.
    p(wc,j = wO,c | wI) = yc,j = exp(uc,j) / [Sum(for all j from 1 to V) exp(uj)] 
*/
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
        /*
            Max Value for Stability.
            - This step subtracts the maximum value max(a) from each input value "a" or u in the eqution from the article.
            - It's a standard numerical stability trick to prevent overflow or underflow when computing exp(u) for large or small values.
         */
        m = Numcy::max(a); // Max value of a
        a_m = Numcy::subtract(a, m); // a - max(a)

        /*
            Exponential of Adjusted Values.
            - Computes the exponential exp(a - max(a)), which corresponds to exp(uj) from the equation of article.
         */
        e_a_m = Numcy::exp(a_m); // exp(a - max(a))

        /*
            Sum of Exponentials.
            Computes the denominator of the equation from the article... [Sum(for all j from 1 to V) exp(uj)].
         */
        s_e_a_m = Numcy::sum(e_a_m); // sum(exp(a - max(a)))

        /*
            Normalization.
            - Divides the exponential of each adjusted input exp(uj) = exp(a - max(a)) by the sum of exponentials, producing the softmax probabilities.
         */
        /*
            m is max
            a_m, a minus m
            e_a_m, exp over a_m
            s_e_a_m, sum of e_a_m
         */  
        e_a_minus_max_divided_by_e_a_minus_max_sum = Numcy::divide(e_a_m, s_e_a_m); // Normalize   
    }
    catch(ala_exception& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + e.what());
    }
    
    return e_a_minus_max_divided_by_e_a_minus_max_sum;
} 

/*
    Performs a portion of the forward propagation step in a Skip-gram model.

    @param W1: Embedding matrix represented as a Collective object. This matrix stores word embeddings.
    @param W2: Output layer weight matrix represented as a Collective object. This matrix contains weights connecting hidden to output layers.
    @param negative_samples_indices: A Collective object containing indices of negative samples in the embedding matrix (W1).
    @param vocab: An instance of the corpus class that provides context about the vocabulary used in the model.
    @param pair: A pointer to a WordPair object, which contains the center word index and context word indices for the Skip-gram model.
    @param ns: A flag indicating whether to use negative sampling. 
               If true, negative sampling is applied; otherwise, a full softmax is used.
    @param verbose: A flag for debugging. When set to true, detailed debug information is printed to the screen.

    The function computes the following:
    - Extracts the embedding vector for the center word from W1 (hidden layer representation).
    - Performs a dot product of the hidden layer vector with W2 to obtain unnormalized scores (logits).
    - Applies either softmax (for full vocabulary prediction) or sigmoid (for negative sampling) to generate probabilities.
    - Handles negative samples if `ns` is set to true, ensuring correct gradient calculations for the Skip-gram model.
*/
template <typename T = double, typename E = cc_tokenizer::string_character_traits<char>::size_type>
forward_propogation<T> forward(Collective<T>& W1, Collective<T>& W2, Collective<E> negative_samples_indices, CORPUS_REF vocab, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("forward() Error: Index of center word is out of bounds of W1.");
    }

    Collective<T> h;    
    Collective<T> y_pred;    
    Collective<T> u;
        
    T positive_samples_loss = 0, negative_samples_loss = 0, positive_negative_epoch_loss = 0;
            
    try 
    {
        /*
            Page 7 of https://arxiv.org/pdf/1411.2738 
            -----------------------------------------
            - Neural networks often assume that inputs and outputs are column vectors, as this aligns with standard linear algebra conventions.
            - Thus, the "transpose" step may be a reminder to treat "h" as a column vector. 
            - But in this implementation, "h" is row vector 1xN and "W2" is NxV so both a vector and a martix are already aligned for valid computation.  There's no need to transpose "h" unnecessarily.
         */
        /*
            --------------------------------------------
            | For both negative and positive sampling. |
            --------------------------------------------
             
            Extract the corresponding word embedding from the weight matrix 𝑊1.
            Instead of directly using a one-hot input vector, this implementation uses a linked list of word pairs.
            Each pair provides the index of the center word, which serves to extract the relevant embedding from 𝑊1.
            The embedding for the center word is stored in the hidden layer vector h.
         */        
        h = W1.slice(W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE), W1.getShape().getNumberOfColumns());
                
        // Compute the logits u using the dot product of h (center word embedding) and W2 (output layer weights)
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
        /*std::cout<< "--> fp Dimensions of h =  Rows" << h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << h.getShape().getNumberOfColumns() << std::endl;
        std::cout<< "--> fp Dimensions of W2 = Rows " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W2.getShape().getNumberOfColumns() << std::endl;*/
                //u_positive_samples = Numcy::dot(h, W2);  
        /*std::cout<< "--> Dimensions of u = " << u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << u.getShape().getNumberOfColumns() << std::endl;*/

        if (negative_samples_indices.getShape().getN() == 0)
        {
            /*T* W2_sample_ptr = cc_tokenizer::allocator<T>().allocate(W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
            {
                W2_sample_ptr[i] = W2[i*W2.getShape().getNumberOfColumns() + (pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)];
            }
            Collective<T> W2_sample = Collective<T>{W2_sample_ptr, DIMENSIONS{1, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};

            cc_tokenizer::allocator<T>().deallocate(W2_sample_ptr, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());
            W2_sample_ptr = NULL;*/
            //u = Numcy::dot(h, W2_sample);

            //std::cout<< "Columns = " << h.getShape().getNumberOfColumns() << " ----- " << "Rows = " << h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            //std::cout<< "Columns = " << W2.getShape().getNumberOfColumns() << " ----- " << "Rows = " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;

            /*
                The dot product produces the logits (net inputs) for the softmax layer.
                The weight matrix "W2" (N x V) is shared across all context words for a given center word.

                Page 7 of https://arxiv.org/pdf/1411.2738 
                -----------------------------------------
                - In the skip-gram model, multiple "panels" represent different context words. For example:
                    - "c" represents the position of the context word relative to the center word.
                - All these panels share the same weight matrix "W2" meaning "W2" is reused for every context position.

                - "uc,j" is the net input to the j-th unit in the output layer for a specific context "c".  
                - "wj" is the j-th column of the weight matrix "W2"
                - "h" is the hidden layer representation (the word embedding of the input word).
                Mathematically ucj = h . W2j (for just a single "uc,j")

                The dot product computes the net inputs for all output units (not just a single "uc,j").
                - The result is a vector where each element represents the net input for a specific output word.
                - OR a vector of net inputs for all words in the vocabulary, which will later pass through a softmax function to compute probabilities.                
             */
            /* m is rows, likewise */
            /* m*n . n*p = m*p */
            /* h = 1xN and W2 = NxV hence u = 1xV */
            u = Numcy::dot(h, W2); 
            
            //std::cout<< "Columns = " << u.getShape().getNumberOfColumns() << " ----- " << "Rown = " << u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;

            /*
                When ns == false (no negative sampling), y_pred is computed using softmax(u),
                which is correct for the traditional skip-gram model without negative sampling.
             */
            /*
                After computing "u" for a center word, you apply the softmax function over "u" to convert it into probabilities for all words in the vocabulary.
             */
            /*
                The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
                The softmax function converts the raw scores into probabilities.

                Ensure that y_pred has valid probability values (between 0 and 1). The implementation is correct and that it normalizes the probabilities correctly, 
                i.e., all values should sum up to 1. If there's a bug in the softmax calculation, it might return incorrect values (like 0 or very small numbers).

                In `Skip-gram`, this output represents the likelihood of each word being one of the context words for the given center word.
            */
            y_pred = softmax(u);

            /*
                The model then computes the loss using these probabilities against the actual target word(s) in the context.
             */        
        }
        else if (negative_samples_indices.getShape().getN() > 0) 
        {   
            //T positive_negative_epoch_loss = 0;

            T* ptr = cc_tokenizer::allocator<T>().allocate(W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());

            //T* ptr = cc_tokenizer::allocator<T>().allocate((pair->getLeft()->size() + pair->getRight()->size())*W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());

            Collective<T> W2_positive_sample = Collective<T>{ptr, DIMENSIONS{1, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};
            
            //Collective<T> W2_positive_samples = Collective<T>{ptr, DIMENSIONS{pair->getLeft()->size() + pair->getRight()->size(), W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};

                /*cc_tokenizer::allocator<T>().deallocate(ptr, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());*/

            //cc_tokenizer::allocator<T>().deallocate(ptr, (pair->getLeft()->size() + pair->getRight()->size())*W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());

            ptr = NULL;

            //cc_tokenizer::string_character_traits<char>::size_type k = 0;        
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_CONTEXT_WINDOW_SIZE; i++)
            {
                /*if ((*(pair->getLeft()))[i] != INDEX_NOT_FOUND_AT_VALUE)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        W2_positive_samples[j*(pair->getLeft()->size() + pair->getRight()->size()) + k] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE];
                    }

                    k = k + 1;
                }*/
                
                Collective<T> u_positive_sample;

                if ((*(pair->getLeft()))[i] != INDEX_NOT_FOUND_AT_VALUE)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        W2_positive_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE];
                    }

                    u_positive_sample = Numcy::dot(h, W2_positive_sample);
                    
                    positive_samples_loss = positive_samples_loss + std::log(Numcy::sigmoid(u_positive_sample)[0])*(-1);
                }

                if ((*(pair->getRight()))[i] != INDEX_NOT_FOUND_AT_VALUE)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        W2_positive_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE];                        
                    }

                    u_positive_sample = Numcy::dot(h, W2_positive_sample); 
                    
                    positive_samples_loss = positive_samples_loss + std::log(Numcy::sigmoid(u_positive_sample)[0])*(-1);
                }
            }
            /*std::cout<< "--> fp Dimensions of h =  Rows" << h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << h.getShape().getNumberOfColumns() << std::endl;
            std::cout<< "--> fp Dimensions of W2_positive_samples = Rows " << W2_positive_samples.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W2_positive_samples.getShape().getNumberOfColumns() << std::endl;*/
            //u_positive_samples = Numcy::dot(h, W2_positive_samples);
            
            ptr = cc_tokenizer::allocator<T>().allocate(W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());            
            Collective<T> W2_negative_sample = Collective<T>{ptr, DIMENSIONS{1, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};                        
                /*cc_tokenizer::allocator<T>().deallocate(ptr, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());*/            
            ptr = NULL;
            
            //Collective<T> u_negative_sample;
            
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_samples_indices.getShape().getN(); i++)
            {
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                {
                        /*std::cout<< "negative_samples_indices[i]" << negative_samples_indices[i] << std::endl;*/

                    W2_negative_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + negative_samples_indices[i]];
                }

                Collective<T> u_negative_sample = Numcy::dot(h, W2_negative_sample);

                //std::cout<< "----> " << u_negative_sample[0] << std::endl;
                u_negative_sample = u_negative_sample*((T)-1);

                negative_samples_loss = negative_samples_loss + std::log(Numcy::sigmoid(u_negative_sample)[0])*(-1);
                
                //std::cout<< "--> fp Dimensions of u_negative_sample =  Rows" << u_negative_sample.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << u_negative_sample.getShape().getNumberOfColumns() << std::endl;
                //std::cout<< "----> " << u_negative_sample[0] << std::endl;
            }
            
            positive_negative_epoch_loss = positive_negative_epoch_loss + (positive_samples_loss + negative_samples_loss);
            
            //std::cout << "=-----------------------------------------> " << positive_negative_epoch_loss << std::endl;
        }
        else
        {
            throw ala_exception("The array containing indices of negative samples is empty.");
        }
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
        throw ala_exception(cc_tokenizer::String<char>("forward() -> ") + cc_tokenizer::String<char>(e.what()));
    }
            
    return forward_propogation<T>(h, y_pred, u, positive_samples_loss, negative_samples_loss, positive_negative_epoch_loss);
}

template <typename T = double, typename E = cc_tokenizer::string_character_traits<char>::size_type>
backward_propogation<T> backward_2(Collective<T>& W1, Collective<T>& W2, Collective<E> negative_samples_indices, CORPUS_REF vocab, forward_propogation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false, T learning_rate = SKIP_GRAM_DEFAULT_LEARNING_RATE) throw (ala_exception)
{
    // Bounds check for Center Word
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("backward() Error: Index of center word is out of bounds of W1.");
    }

    /* -------------------------------------------------------------------------
       Output Gradient Containers
       grad_W1: Shape (Vocab x Dimensions) -> Row-Major, Center Word Gradients
       grad_W2: Shape (Dimensions x Vocab) -> Row-Major, Context Word Gradients
       ------------------------------------------------------------------------- */
    Collective<T> grad_W1;
    Collective<T> grad_W2;

    try 
    {
        /* =========================================================================
           BRANCH 1: SOFTMAX (Full Vocabulary)
           Use this when negative_samples_indices is empty
           ========================================================================= */
        if (negative_samples_indices.getShape().getN() == 0)
        {            
            /* 1. Create One-Hot Vector for Context (Target) */
            // Shape: (1 x Vocab)
            Collective<T> oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});
                    
            // Mark Left Context Words
            for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
            {       
                if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
                {
                    oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
                }
            }
            // Mark Right Context Words
            for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
            {
                if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
                {
                    oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
                }        
            }

            /* 2. Calculate Prediction Error (grad_u) */
            // grad_u = y_pred - y_true
            // Shape: (1 x Vocab)
            Collective<T> grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);
            
            /* 3. Calculate Gradient for W2 (Context Weights) */
            // Math: grad_W2 = h^T * grad_u
            // h_transpose: (Dimensions x 1)
            // grad_u:      (1 x Vocab)
            // Result:      (Dimensions x Vocab)
            Collective<T> h_transpose = Numcy::transpose<T>(fp.hidden_layer_vector);
            grad_W2 = Numcy::dot(h_transpose, grad_u);
        
            /* 4. Calculate Gradient for W1 (Center Word) */
            // Math: grad_h = grad_u * W2^T
            // Result: (1 x Dimensions)
            Collective<T> W2_T = Numcy::transpose(W2);
            Collective<T> grad_h = Numcy::dot(grad_u, W2_T);

            /* 5. Initialize grad_W1 container */
            // Shape: (Vector_Size x Vocab) -- Note: Dimensions might differ based on your Numcy logic, 
            // but standard W1 is (Vocab x Dimensions).
            grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});

            // Update center word gradient row
            cc_tokenizer::string_character_traits<char>::size_type center_word_offset = (pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) * SKIP_GRAM_EMBEDDNG_VECTOR_SIZE;
            
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; i++)
            {
                grad_W1[center_word_offset + i] += grad_h[i];
            }
        }
        
        /* =========================================================================
           BRANCH 2: NEGATIVE SAMPLING (Sparse Update)
           Use this when negative_samples_indices has data
           ========================================================================= */
        else if (negative_samples_indices.getShape().getN() > 0)
        {   
            /* 1. Initialize Gradients with correct shapes (Matching Softmax Dimensions) */
            // grad_W1: (Vocab x EmbeddingSize) -> Although DIMENSIONS arg order depends on your Numcy impl.
            grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
            
            // grad_W2: (EmbeddingSize x Vocab)
            grad_W2 = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
                        
            /* 2. PERFORMANCE FIX: Allocate temp vector ONCE outside loops */
            T* ptr_temp = cc_tokenizer::allocator<T>().allocate(SKIP_GRAM_EMBEDDNG_VECTOR_SIZE);
            Collective<T> temp_vec_context = Collective<T>{ptr_temp, DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
            
            /* Helper: Offset for Center Word in W1 */
            cc_tokenizer::string_character_traits<char>::size_type center_word_idx = pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE;
            cc_tokenizer::string_character_traits<char>::size_type center_word_offset = center_word_idx * SKIP_GRAM_EMBEDDNG_VECTOR_SIZE;

            /* ---------------------------------------------------------------------
               A. Positive Samples Backprop
               --------------------------------------------------------------------- */
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_CONTEXT_WINDOW_SIZE; i++)
            {
                // Check Left and Right Context
                cc_tokenizer::string_character_traits<char>::size_type context_indices[2] = {
                    (*(pair->getLeft()))[i], 
                    (*(pair->getRight()))[i]
                };

                for(int ctx = 0; ctx < 2; ctx++) 
                {
                    cc_tokenizer::string_character_traits<char>::size_type target_idx = context_indices[ctx];

                    if (target_idx != INDEX_NOT_FOUND_AT_VALUE)
                    {
                        target_idx -= INDEX_ORIGINATES_AT_VALUE; // Normalize index

                        // Load W2 (Context Vector) into temp array [No Allocation Here!]
                        for (int d = 0; d < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; d++)
                        {
                            // Access W2 as (Dimensions x Vocab) -> row * num_cols + col
                            // Assuming W2 is (Dims x Vocab) row-major:
                            temp_vec_context[d] = W2[d * vocab.numberOfUniqueTokens() + target_idx];
                        }

                        // Calculate Error: (sigmoid(h . v_c) - 1)
                        Collective<T> u_val = Numcy::dot(fp.hidden_layer_vector, temp_vec_context);
                        T grad_scalar = Numcy::sigmoid(u_val)[0] - 1.0; 

                        // Accumulate Gradients
                        for (int d = 0; d < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; d++)
                        {
                            // Update W2 (Context): gradient = scalar * h
                            grad_W2[d * vocab.numberOfUniqueTokens() + target_idx] += grad_scalar * fp.hidden_layer_vector[d];

                            // Update W1 (Center): gradient = scalar * v_c
                            // MATH FIX: Using temp_vec_context instead of hidden_layer_vector
                            grad_W1[center_word_offset + d] += grad_scalar * temp_vec_context[d];
                        }
                    }
                }
            }

            /* ---------------------------------------------------------------------
               B. Negative Samples Backprop
               --------------------------------------------------------------------- */
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_samples_indices.getShape().getN(); i++)
            {                
                cc_tokenizer::string_character_traits<char>::size_type neg_idx = negative_samples_indices[i]; // Already normalized usually? Or need ORIGIN check?
                
                // Load Negative Context Vector
                for (int d = 0; d < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; d++)
                {
                    temp_vec_context[d] = W2[d * vocab.numberOfUniqueTokens() + neg_idx];
                }

                // Calculate Error: sigmoid(h . v_neg)
                Collective<T> u_val = Numcy::dot(fp.hidden_layer_vector, temp_vec_context); 
                T grad_scalar = Numcy::sigmoid(u_val)[0]; // Prediction - 0 (Label)

                // Accumulate Gradients
                for (int d = 0; d < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; d++)
                {
                    // Update W2 (Context)
                    grad_W2[d * vocab.numberOfUniqueTokens() + neg_idx] += grad_scalar * fp.hidden_layer_vector[d];

                    // Update W1 (Center)
                    // MATH FIX: Using temp_vec_context
                    grad_W1[center_word_offset + d] += grad_scalar * temp_vec_context[d];
                }
            }
            
            /* Clean up temporary pointer manually if your allocator requires it, 
               otherwise handled by Collective destructor if it owns the ptr */
            // ptr_temp = NULL; // Prevent double free if Collective handles it
        }
        else
        {            
            throw ala_exception("The array containing indices of negative samples is empty.");        
        }
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }

    // Return the gradients
    DIMENSIONS temp1 = DIMENSIONS{0, 0, NULL, NULL};
    Collective<T> temp2 = Collective<T>{NULL, temp1};       
    backward_propogation<T> ret = backward_propogation<T>{grad_W1, grad_W2, temp2};
    
    return ret;
}

template <typename T = double, typename E = cc_tokenizer::string_character_traits<char>::size_type>
backward_propogation<T> backward_1(Collective<T>& W1, Collective<T>& W2, Collective<E> negative_samples_indices, CORPUS_REF vocab, forward_propogation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false, T learning_rate = SKIP_GRAM_DEFAULT_LEARNING_RATE) throw (ala_exception)
{
    // 1. Safety Check
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("backward() Error: Index of center word is out of bounds of W1.");
    }

    // 2. Define Dimension Constants for clearer indexing
    const auto VECTOR_SIZE = SKIP_GRAM_EMBEDDNG_VECTOR_SIZE;
    const auto VOCAB_SIZE = vocab.numberOfUniqueTokens();
    const auto W2_COLS = W2.getShape().getNumberOfColumns(); // Should match VOCAB_SIZE if W2 is (D x V)

    // 3. Initialize Gradients with Zeros
    // grad_W1 shape: (Vocab Rows, Vector Cols) - Same as W1
    Collective<T> grad_W1 = Numcy::zeros(DIMENSIONS{VOCAB_SIZE, VECTOR_SIZE, NULL, NULL});
    
    // grad_W2 shape: (Vector Rows, Vocab Cols) - Same as W2
    Collective<T> grad_W2 = Numcy::zeros(DIMENSIONS{VECTOR_SIZE, VOCAB_SIZE, NULL, NULL});

    try 
    {
        // =========================================================
        // MODE A: SOFTMAX (No Negative Sampling)
        // =========================================================
        if (negative_samples_indices.getShape().getN() == 0)
        {            
            Collective<T> oneHot = Numcy::zeros(DIMENSIONS{VOCAB_SIZE, 1, NULL, NULL});
            
            // Build One-Hot Vector from Context
            for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
            {       
                if ((*(pair->getLeft()))[i] != INDEX_NOT_FOUND_AT_VALUE && ((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < VOCAB_SIZE)
                    oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;

                if ((*(pair->getRight()))[i] != INDEX_NOT_FOUND_AT_VALUE && ((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < VOCAB_SIZE)
                    oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;       
            }

            // Calculate Gradient of Score (Prediction - Target)
            Collective<T> grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);
            
            // Calculate grad_W2 = h_transpose * grad_u
            // This is an Outer Product: (D x 1) * (1 x V) -> (D x V)
            Collective<T> h_transpose = Numcy::transpose<T>(fp.hidden_layer_vector);
            Collective<T> calculated_grad_W2 = Numcy::dot(h_transpose, grad_u);

            // Accumulate into main grad_W2 holder
            // Assuming calculated_grad_W2 has linear layout compatible with grad_W2
            for(size_t k=0; k < (VECTOR_SIZE * VOCAB_SIZE); k++) {
                grad_W2[k] += calculated_grad_W2[k];
            }

            // Calculate grad_W1 (Center Word Update)
            // Error propagated back to hidden layer: grad_h = grad_u * W2_Transpose
            Collective<T> W2_T = Numcy::transpose(W2);
            Collective<T> grad_h = Numcy::dot(grad_u, W2_T);

            // Update only the row corresponding to the Center Word
            size_t center_word_idx = pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE;
            for (size_t i = 0; i < VECTOR_SIZE; i++)
            {
                grad_W1[center_word_idx * VECTOR_SIZE + i] += grad_h[i];
            }
        }
        // =========================================================
        // MODE B: NEGATIVE SAMPLING (Optimized)
        // =========================================================
        else 
        {   
            // PERFORMANCE: Allocate temporary vector ONCE outside the loop to avoid malloc/free spam
            T* raw_ptr = cc_tokenizer::allocator<T>().allocate(VECTOR_SIZE);
            Collective<T> temp_vec_context = Collective<T>{raw_ptr, DIMENSIONS{VECTOR_SIZE, 1, NULL, NULL}};

            // --- 1. POSITIVE SAMPLES ---
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_CONTEXT_WINDOW_SIZE; i++)
            {
                // Helper lambda or macro could clean this up, but keeping it explicit for C++98/03 compat if needed
                cc_tokenizer::string_character_traits<char>::size_type indices[2];
                indices[0] = (*(pair->getLeft()))[i];
                indices[1] = (*(pair->getRight()))[i];

                for(int side = 0; side < 2; side++) 
                {
                    auto target_idx = indices[side];
                    if (target_idx == INDEX_NOT_FOUND_AT_VALUE) continue;
                    
                    target_idx = target_idx - INDEX_ORIGINATES_AT_VALUE;
                    if (target_idx >= VOCAB_SIZE) continue; // Safety

                    // Load W2 column into temp vector (No Allocation)
                    for(int d = 0; d < VECTOR_SIZE; d++) {
                        temp_vec_context[d] = W2[d * W2_COLS + target_idx];
                    }

                    // Forward: u = h . v_c
                    Collective<T> u_res = Numcy::dot(fp.hidden_layer_vector, temp_vec_context);
                    
                    // Backward: Error = Sigmoid(u) - 1 (Because label is 1)
                    T error_scalar = Numcy::sigmoid(u_res)[0] - 1.0;

                    // Accumulate Gradients
                    // dL/dW2 = h * error
                    // dL/dW1 = v_c * error
                    size_t center_idx = pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE;
                    
                    for(int d = 0; d < VECTOR_SIZE; d++) 
                    {
                        // Update Context (W2) Gradient
                        grad_W2[d * W2_COLS + target_idx] += error_scalar * fp.hidden_layer_vector[d];

                        // Update Center (W1) Gradient [CORRECTED MATH]
                        grad_W1[center_idx * VECTOR_SIZE + d] += error_scalar * temp_vec_context[d];
                    }
                }
            }

            // --- 2. NEGATIVE SAMPLES ---
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_samples_indices.getShape().getN(); i++)
            {                
                auto neg_idx = negative_samples_indices[i];
                if (neg_idx >= VOCAB_SIZE) continue;

                // Load W2 column (No Allocation)
                for(int d = 0; d < VECTOR_SIZE; d++) {
                    temp_vec_context[d] = W2[d * W2_COLS + neg_idx];
                }

                // Forward: u = h . v_neg
                Collective<T> u_res = Numcy::dot(fp.hidden_layer_vector, temp_vec_context);
                
                // Backward: Error = Sigmoid(u) - 0 (Because label is 0)
                T error_scalar = Numcy::sigmoid(u_res)[0];

                // Accumulate Gradients
                size_t center_idx = pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE;

                for(int d = 0; d < VECTOR_SIZE; d++) 
                {
                    // Update Context (W2) Gradient
                    grad_W2[d * W2_COLS + neg_idx] += error_scalar * fp.hidden_layer_vector[d];

                    // Update Center (W1) Gradient [CORRECTED MATH]
                    grad_W1[center_idx * VECTOR_SIZE + d] += error_scalar * temp_vec_context[d];
                }
            }

            // Clean up the manual allocation
            // Note: If Collective destructor handles deallocation of its ptr, set raw_ptr to NULL first if needed.
            // Assuming standard allocator behavior:
            cc_tokenizer::allocator<T>().deallocate(raw_ptr, VECTOR_SIZE);
            //temp_vec_context.setPtr(NULL); // Prevent double free if Collective dtor tries to free
        }
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }

    // Return the gradients and a dummy/empty Collective for the 3rd argument if required by struct
    DIMENSIONS empty_dim = DIMENSIONS{0, 0, NULL, NULL};
    Collective<T> empty_coll = Collective<T>{NULL, empty_dim};       
    return backward_propogation<T>{grad_W1, grad_W2, empty_coll};
}



template <typename T = double, typename E = cc_tokenizer::string_character_traits<char>::size_type>
backward_propogation<T> backward_trial_and_error_does_not_word(Collective<T>& W1, Collective<T>& W2, Collective<E> negative_samples_indices, CORPUS_REF vocab, forward_propogation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false, T learning_rate = SKIP_GRAM_DEFAULT_LEARNING_RATE) throw (ala_exception)
{
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("backward() Error: Index of center word is out of bounds of W1.");
    }

    /* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    Collective<T> oneHot;
    /* 
        Gradient for the positive sample.
        The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency) 
     */
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

    Collective<T> grad_u_negative_samples;
    Collective<T> grad_W2_negative_samples;

    /* Neative sampling */
            
    try 
    {
        //std::cout<< "-> Columns = " << fp.hidden_layer_vector.getShape().getNumberOfColumns() << ", Rows = " << fp.hidden_layer_vector.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
        /*
            h_transpose has shape (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE rows, 1 column)
         */
        Collective<T> h_transpose = Numcy::transpose<T>(fp.hidden_layer_vector);

        /*std::cout<< "h(fp.hidden_layer_vector) Columns = " << fp.hidden_layer_vector.getShape().getNumberOfColumns() << ", Rows = " << fp.hidden_layer_vector.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
        
        if (negative_samples_indices.getShape().getN() == 0)
        {            
            /*
                Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
                This creates a zero-filled column vector with a length equal to the vocabulary size
             */
            oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/, 1, NULL, NULL});
                    
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
                if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/)
                {
                    oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
                }
            }
            for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
            {
                if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/)
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
                The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1 row, len(vocab columns) with redundency)

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
                Accumulates gradients first and applies them later in a separate step.
                ---------------------------------------------------------------------- 
                Take h transpose of  hidden_layer_vector(h) and the multiply it with 
                grad_u(gradient of intermediate_activation) and the resulting matrix will grad_W2.
                Before transpose_h has shape (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE rows, 1 column) and grad_u is (1 row, len((vocab with redundency) columns)

                grad_W2 has shape (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE rows, len((vocab with redundency) columns)
              */
             grad_W2 = Numcy::dot(h_transpose, grad_u);
             
             // Update gradients for positive samples
             //W2 = W2 + grad_W2;                          
                        
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
             /*
                So what you are saying is that take he transpose of  hidden_layer_vector(h) and the multiply it with 
                grad_u(gradient of intermediate_activation) and the resulting matrix will grad_W2 
             */
            /*Collective<T> h_transpose = Numcy::transpose<T>(fp.hidden_layer_vector);*/
            /*grad_W2 =  Numcy::dot(h_transpose, grad_u);*/

            //std::cout<< "Dimensions of h_transpose = " << h_transpose.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << h_transpose.getShape().getNumberOfColumns() << std::endl;
            //std::cout<< "Dimensions of h = " << fp.hidden_layer_vector.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << fp.hidden_layer_vector.getShape().getNumberOfColumns() << std::endl;

            /*std::cout<< "Dimensions of fp.intermediate_activation a.k.a u = " << fp.intermediate_activation.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << fp.intermediate_activation.getShape().getNumberOfColumns() << std::endl;
            std::cout<< "Dimensions of grad_u = " << grad_u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_u.getShape().getNumberOfColumns() << std::endl;*/
            //grad_W2 = Numcy::outer(fp.intermediate_activation, grad_u);
            /*std::cout<< "Dimensions of grad_W2 = " << grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_W2.getShape().getNumberOfColumns() << std::endl;*/
        
            // Update W1 for positive samples
            // ------------------------------
            W2_T = Numcy::transpose(W2);

            /*
                std::cout<< grad_u.getShape().getNumberOfColumns() << " ------- " << grad_u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                std::cout<< W2_T.getShape().getNumberOfColumns() << " ------- " << W2_T.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            */
            /*
                A (m, n), B = (n, p) then A*B is (m, p)
                The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1 row, len(vocab columns) with redundency)
                The shape of W2_T is (vocab.numberOfTokens() rows, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE columns)
                The shape of grad_h is (1 row, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE columns)
            */        
            grad_h = Numcy::dot(grad_u, W2_T);

            /*std::cout<< "grad_h shape Columns =" << grad_h.getShape().getNumberOfColumns() << " -------  Rows = " << grad_h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/

            /*
                std::cout<< grad_h.getShape().getNumberOfColumns() << " ------------------------- " << grad_h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            */
            /*
                Accumulates gradients first and applies them later in a separate step.
                ---------------------------------------------------------------------- 
             */
            grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, /*vocab.numberOfTokens()*/ vocab.numberOfUniqueTokens(), NULL, NULL});

            /*
                Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
                Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
            */
            // Update center word gradient for positive sample
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
            {
                grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
            }
        }
        else if (negative_samples_indices.getShape().getN() > 0)
        {   
            // Initialize gradient accumulators
            grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
            grad_W2 = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
                        
            // Backpropagation for positive sample
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_CONTEXT_WINDOW_SIZE; i++)
            {
                T* ptr = cc_tokenizer::allocator<T>().allocate(W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());
                Collective<T> W2_positive_sample = Collective<T>{ptr, DIMENSIONS{1, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};

                /*cc_tokenizer::allocator<T>().deallocate(ptr, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());*/

                ptr = NULL;

                Collective<T> u_positive_sample;
                Collective<T> grad_u_positive_sample;
                Collective<T> grad_W2_positive_sample;

                if ((*(pair->getLeft()))[i] != INDEX_NOT_FOUND_AT_VALUE)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        W2_positive_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE];
                    }

                    // Calculate Error for Positive Sample
                    u_positive_sample = Numcy::dot(fp.hidden_layer_vector, W2_positive_sample);                    
                    grad_u_positive_sample = Numcy::sigmoid(u_positive_sample) - 1;

                    // Calculate Gradient for W2 for Positive Sample
                    // dL/dW2 = h * error
                    grad_W2_positive_sample = Numcy::outer(fp.hidden_layer_vector, grad_u_positive_sample);

                    //std::cout<< "Dimensions of grad_W2_positive_sample (ROWS) = " << grad_W2_positive_sample.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_W2_positive_sample.getShape().getNumberOfColumns() << std::endl;
                    //std::cout<< "Dimensions of grad_u = " << grad_u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_u.getShape().getNumberOfColumns() << std::endl;
                    
                    //positive_samples_loss = positive_samples_loss + std::log(Numcy::sigmoid(u_positive_sample)[0])*(-1);

                    //W2[:, context_word_index] -= learning_rate * grad_W2_positive[:, 0]

                    // Accumulate Gradient for W2 for Positive Sample
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W2*/grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        // Commented this line to accumulate gradients first and apply them later
                        /*W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] - learning_rate*grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0];*/

                        // USE += (Add the negative gradient)                        
                        grad_W2[j*grad_W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] += /*learning_rate**/grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0]; 
                    }

                    // Calculate Gradient for W1 (Center Vector)
                    // ERROR WAS HERE: You used dot(grad, hidden_layer). 
                    // CORRECTION: It must be dot(grad, W2_context).
                    // Math: dL/dh = error * W2
                    // Note: grad_u is scalar (1x1), W2 is vector (Dx1). 
                    // You effectively just want to scale W2 by the error term.

                    Collective<T> grad_h_contribution = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL});

                    for(int k=0; k<SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; k++) {
                    // Scalar multiplication: Error * Context_Vector[k]
                        grad_h_contribution[k] = grad_u_positive_sample[0] * W2_positive_sample[k];
                    }

                    // 5. Accumulate Gradient for W1
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < grad_W1.getShape().getNumberOfColumns(); j++)
                    {
                        // USE += (Add the negative gradient)
                        grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] += grad_h_contribution[j];
                    }
                    //std::cout<< "Dimensions of Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector) ROWS = " << Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector).getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector).getShape().getNumberOfColumns() << std::endl;                    
                }

                if ((*(pair->getRight()))[i] != INDEX_NOT_FOUND_AT_VALUE)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        W2_positive_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE];                        
                    }

                    u_positive_sample = Numcy::dot(fp.hidden_layer_vector, W2_positive_sample); 

                    grad_u_positive_sample = Numcy::sigmoid(u_positive_sample) - 1;

                    grad_W2_positive_sample = Numcy::outer(fp.hidden_layer_vector, grad_u_positive_sample);

                    //std::cout<< "--Dimensions of grad_W2_positive_sample (ROWS) = " << grad_W2_positive_sample.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_W2_positive_sample.getShape().getNumberOfColumns() << std::endl;
                    
                    //positive_samples_loss = positive_samples_loss + std::log(Numcy::sigmoid(u_positive_sample)[0])*(-1);

                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W2*/grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        // Commented this line to accumulate gradients first and apply them later
                        /*W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] - learning_rate*grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0];*/

                        grad_W2[j*grad_W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] /*-=*/ += /*learning_rate**/grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0];
                    }

                    Collective<T> product = Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector);

                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W1*/grad_W1.getShape().getNumberOfColumns(); j++)
                    {
                        // Commented this line to accumulate gradients first and apply them later
                        /*W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] - learning_rate*product[j];*/

                        grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] -= /*learning_rate**/product[j];
                    }
                }
            }
            // Backpropagation for negative samples
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_samples_indices.getShape().getN(); i++)
            {                
                T* ptr = cc_tokenizer::allocator<T>().allocate(W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());            
                Collective<T> W2_negative_sample = Collective<T>{ptr, DIMENSIONS{1, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};                        
                    /*cc_tokenizer::allocator<T>().deallocate(ptr, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());*/            
                ptr = NULL;

                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                {
                    W2_negative_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + negative_samples_indices[i]];
                } 

                Collective<T> u_negative_sample = Numcy::dot(fp.hidden_layer_vector, W2_negative_sample); 

                /*u_negative_sample = u_negative_sample*((T)-1);*/
                Collective<T> grad_u_negative_sample = Numcy::sigmoid(u_negative_sample) /*- 1*/;

                Collective<T> grad_W2_negative_sample = Numcy::outer(fp.hidden_layer_vector, grad_u_negative_sample);

                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W2*/grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                {
                    // Commented this line to accumulate gradients first and apply them later
                    /*W2[j*W2.getShape().getNumberOfColumns() + i] = W2[j*W2.getShape().getNumberOfColumns() + i] - learning_rate*grad_W2_negative_sample[j*grad_W2_negative_sample.getShape().getNumberOfColumns() + 0];*/

                    grad_W2[j*grad_W2.getShape().getNumberOfColumns() + negative_samples_indices[i]] += /*learning_rate**/grad_W2_negative_sample[j*grad_W2_negative_sample.getShape().getNumberOfColumns() + 0];
                }

                Collective<T> product = Numcy::dot(grad_u_negative_sample, fp.hidden_layer_vector);

                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W1*/grad_W1.getShape().getNumberOfColumns(); j++)
                {
                    // Commented this line to accumulate gradients first and apply them later
                    /*W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] - learning_rate*product[j];*/

                    // Bug 2: The line below was incorrectly subtracting the product instead of adding it.
                    //grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] -= /*learning_rate**/product[j];

                    // Correct: Scaling the negative vector by the error signal
                    grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] += grad_u_negative_sample[0] * W2_negative_sample[j];
                }
            }
        }
        else
        {            
            throw ala_exception("The array containing indices of negative samples is empty.");        
        }
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)
     */ 
    //return backward_propogation<T>{grad_W1, grad_W2, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}};

   //T * ptrptrptr = cc_tokenizer::allocator<T>().allocate(10);
    //DIMENSIONS temp1 = DIMENSIONS{10, 1, NULL, NULL};

    DIMENSIONS temp1 = DIMENSIONS{0, 0, NULL, NULL};
    Collective<T> temp2 = Collective<T>{NULL, temp1};       
    backward_propogation<T> ret = backward_propogation<T>{grad_W1, grad_W2, temp2};
    
    //std::cout<< "AT THE END OF BACKWARD PROPAGATION FUNCTION" << std::endl;

    return ret;
}

template <typename T = double, typename E = cc_tokenizer::string_character_traits<char>::size_type>
backward_propogation<T> backward(Collective<T>& W1, Collective<T>& W2, Collective<E> negative_samples_indices, CORPUS_REF vocab, forward_propogation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false, T learning_rate = SKIP_GRAM_DEFAULT_LEARNING_RATE) throw (ala_exception)
{
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("backward() Error: Index of center word is out of bounds of W1.");
    }

    /* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    Collective<T> oneHot;
    /* 
        Gradient for the positive sample.
        The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency) 
     */
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

    Collective<T> grad_u_negative_samples;
    Collective<T> grad_W2_negative_samples;

    /* Neative sampling */
            
    try 
    {
        //std::cout<< "-> Columns = " << fp.hidden_layer_vector.getShape().getNumberOfColumns() << ", Rows = " << fp.hidden_layer_vector.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
        /*
            h_transpose has shape (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE rows, 1 column)
         */
        Collective<T> h_transpose = Numcy::transpose<T>(fp.hidden_layer_vector);

        /*std::cout<< "h(fp.hidden_layer_vector) Columns = " << fp.hidden_layer_vector.getShape().getNumberOfColumns() << ", Rows = " << fp.hidden_layer_vector.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
        
        if (negative_samples_indices.getShape().getN() == 0)
        {            
            /*
                Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
                This creates a zero-filled column vector with a length equal to the vocabulary size
             */
            oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/, 1, NULL, NULL});
                    
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
                if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/)
                {
                    oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
                }
            }
            for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
            {
                if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/)
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
                The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1 row, len(vocab columns) with redundency)

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
                Accumulates gradients first and applies them later in a separate step.
                ---------------------------------------------------------------------- 
                Take h transpose of  hidden_layer_vector(h) and the multiply it with 
                grad_u(gradient of intermediate_activation) and the resulting matrix will grad_W2.
                Before transpose_h has shape (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE rows, 1 column) and grad_u is (1 row, len((vocab with redundency) columns)

                grad_W2 has shape (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE rows, len((vocab with redundency) columns)
              */
             grad_W2 = Numcy::dot(h_transpose, grad_u);
             
             // Update gradients for positive samples
             //W2 = W2 + grad_W2;                          
                        
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
             /*
                So what you are saying is that take he transpose of  hidden_layer_vector(h) and the multiply it with 
                grad_u(gradient of intermediate_activation) and the resulting matrix will grad_W2 
             */
            /*Collective<T> h_transpose = Numcy::transpose<T>(fp.hidden_layer_vector);*/
            /*grad_W2 =  Numcy::dot(h_transpose, grad_u);*/

            //std::cout<< "Dimensions of h_transpose = " << h_transpose.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << h_transpose.getShape().getNumberOfColumns() << std::endl;
            //std::cout<< "Dimensions of h = " << fp.hidden_layer_vector.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << fp.hidden_layer_vector.getShape().getNumberOfColumns() << std::endl;

            /*std::cout<< "Dimensions of fp.intermediate_activation a.k.a u = " << fp.intermediate_activation.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << fp.intermediate_activation.getShape().getNumberOfColumns() << std::endl;
            std::cout<< "Dimensions of grad_u = " << grad_u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_u.getShape().getNumberOfColumns() << std::endl;*/
            //grad_W2 = Numcy::outer(fp.intermediate_activation, grad_u);
            /*std::cout<< "Dimensions of grad_W2 = " << grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_W2.getShape().getNumberOfColumns() << std::endl;*/
        
            // Update W1 for positive samples
            // ------------------------------
            W2_T = Numcy::transpose(W2);

            /*
                std::cout<< grad_u.getShape().getNumberOfColumns() << " ------- " << grad_u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                std::cout<< W2_T.getShape().getNumberOfColumns() << " ------- " << W2_T.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            */
            /*
                A (m, n), B = (n, p) then A*B is (m, p)
                The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1 row, len(vocab columns) with redundency)
                The shape of W2_T is (vocab.numberOfTokens() rows, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE columns)
                The shape of grad_h is (1 row, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE columns)
            */        
            grad_h = Numcy::dot(grad_u, W2_T);

            /*std::cout<< "grad_h shape Columns =" << grad_h.getShape().getNumberOfColumns() << " -------  Rows = " << grad_h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/

            /*
                std::cout<< grad_h.getShape().getNumberOfColumns() << " ------------------------- " << grad_h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            */
            /*
                Accumulates gradients first and applies them later in a separate step.
                ---------------------------------------------------------------------- 
             */
            grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, /*vocab.numberOfTokens()*/ vocab.numberOfUniqueTokens(), NULL, NULL});

            /*
                Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
                Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
            */
            // Update center word gradient for positive sample
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
            {
                grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
            }
        }
        else if (negative_samples_indices.getShape().getN() > 0)
        {   
            // Initialize gradient accumulators
            grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
            grad_W2 = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
                        
            // Backpropagation for positive sample
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_CONTEXT_WINDOW_SIZE; i++)
            {
                T* ptr = cc_tokenizer::allocator<T>().allocate(W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());
                Collective<T> W2_positive_sample = Collective<T>{ptr, DIMENSIONS{1, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};

                /*cc_tokenizer::allocator<T>().deallocate(ptr, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());*/

                ptr = NULL;

                Collective<T> u_positive_sample;
                Collective<T> grad_u_positive_sample;
                Collective<T> grad_W2_positive_sample;

                if ((*(pair->getLeft()))[i] != INDEX_NOT_FOUND_AT_VALUE)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        W2_positive_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE];
                    }

                    // Calculate Error for Positive Sample
                    u_positive_sample = Numcy::dot(fp.hidden_layer_vector, W2_positive_sample);                    
                    grad_u_positive_sample = Numcy::sigmoid(u_positive_sample) - 1;

                    // Calculate Gradient for W2 for Positive Sample
                    // dL/dW2 = h * error
                    grad_W2_positive_sample = Numcy::outer(fp.hidden_layer_vector, grad_u_positive_sample);

                    //std::cout<< "Dimensions of grad_W2_positive_sample (ROWS) = " << grad_W2_positive_sample.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_W2_positive_sample.getShape().getNumberOfColumns() << std::endl;
                    //std::cout<< "Dimensions of grad_u = " << grad_u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_u.getShape().getNumberOfColumns() << std::endl;
                    
                    //positive_samples_loss = positive_samples_loss + std::log(Numcy::sigmoid(u_positive_sample)[0])*(-1);

                    //W2[:, context_word_index] -= learning_rate * grad_W2_positive[:, 0]

                    // Accumulate Gradient for W2 for Positive Sample
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W2*/grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        // Commented this line to accumulate gradients first and apply them later
                        /*W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] - learning_rate*grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0];*/

                        // USE += (Add the negative gradient)                        
                        grad_W2[j*grad_W2.getShape().getNumberOfColumns() + (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] += /*learning_rate**/grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0]; 
                    }

                    // Calculate Gradient for W1 (Center Vector)
                    // ERROR WAS HERE: You used dot(grad, hidden_layer). 
                    // CORRECTION: It must be dot(grad, W2_context).
                    // Math: dL/dh = error * W2
                    // Note: grad_u is scalar (1x1), W2 is vector (Dx1). 
                    // You effectively just want to scale W2 by the error term.

                    Collective<T> product = Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector);

                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W1*/grad_W1.getShape().getNumberOfColumns(); j++)
                    {
                        // Commented this line to accumulate gradients first and apply them later
                        /*W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] - learning_rate*product[j];*/

                        /*if (!(grad_W1.getShape() == W1.getShape()))
                        {
                             std::cout<< "They both are not same" << std::endl;

                            std::cout<< "Dimensions of W1 = " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W1.getShape().getNumberOfColumns() << std::endl;
                            std::cout<< "Dimensions of grad_W1 = " << grad_W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_W1.getShape().getNumberOfColumns() << std::endl;
                        }*/
                    
                        grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] -= /*learning_rate**/product[j];
                    }

                    //std::cout<< "Dimensions of Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector) ROWS = " << Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector).getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector).getShape().getNumberOfColumns() << std::endl;                    
                }

                if ((*(pair->getRight()))[i] != INDEX_NOT_FOUND_AT_VALUE)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        W2_positive_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE];                        
                    }

                    u_positive_sample = Numcy::dot(fp.hidden_layer_vector, W2_positive_sample); 

                    grad_u_positive_sample = Numcy::sigmoid(u_positive_sample) - 1;

                    grad_W2_positive_sample = Numcy::outer(fp.hidden_layer_vector, grad_u_positive_sample);

                    //std::cout<< "--Dimensions of grad_W2_positive_sample (ROWS) = " << grad_W2_positive_sample.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << grad_W2_positive_sample.getShape().getNumberOfColumns() << std::endl;
                    
                    //positive_samples_loss = positive_samples_loss + std::log(Numcy::sigmoid(u_positive_sample)[0])*(-1);

                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W2*/grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                    {
                        // Commented this line to accumulate gradients first and apply them later
                        /*W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = W2[j*W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] - learning_rate*grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0];*/

                        grad_W2[j*grad_W2.getShape().getNumberOfColumns() + (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] /*-=*/ += /*learning_rate**/grad_W2_positive_sample[j*grad_W2_positive_sample.getShape().getNumberOfColumns() + 0];
                    }

                    Collective<T> product = Numcy::dot(grad_u_positive_sample, fp.hidden_layer_vector);

                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W1*/grad_W1.getShape().getNumberOfColumns(); j++)
                    {
                        // Commented this line to accumulate gradients first and apply them later
                        /*W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] - learning_rate*product[j];*/

                        grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] -= /*learning_rate**/product[j];
                    }
                }
            }
            // Backpropagation for negative samples
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_samples_indices.getShape().getN(); i++)
            {                
                T* ptr = cc_tokenizer::allocator<T>().allocate(W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());            
                Collective<T> W2_negative_sample = Collective<T>{ptr, DIMENSIONS{1, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};                        
                    /*cc_tokenizer::allocator<T>().deallocate(ptr, W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays());*/            
                ptr = NULL;

                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                {
                    W2_negative_sample[j] = W2[j*W2.getShape().getNumberOfColumns() + negative_samples_indices[i]];
                } 

                Collective<T> u_negative_sample = Numcy::dot(fp.hidden_layer_vector, W2_negative_sample); 

                /*u_negative_sample = u_negative_sample*((T)-1);*/
                Collective<T> grad_u_negative_sample = Numcy::sigmoid(u_negative_sample) /*- 1*/;

                Collective<T> grad_W2_negative_sample = Numcy::outer(fp.hidden_layer_vector, grad_u_negative_sample);

                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W2*/grad_W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++)
                {
                    // Commented this line to accumulate gradients first and apply them later
                    /*W2[j*W2.getShape().getNumberOfColumns() + i] = W2[j*W2.getShape().getNumberOfColumns() + i] - learning_rate*grad_W2_negative_sample[j*grad_W2_negative_sample.getShape().getNumberOfColumns() + 0];*/

                    grad_W2[j*grad_W2.getShape().getNumberOfColumns() + negative_samples_indices[i]] += /*learning_rate**/grad_W2_negative_sample[j*grad_W2_negative_sample.getShape().getNumberOfColumns() + 0];
                }

                Collective<T> product = Numcy::dot(grad_u_negative_sample, fp.hidden_layer_vector);

                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*W1*/grad_W1.getShape().getNumberOfColumns(); j++)
                {
                    // Commented this line to accumulate gradients first and apply them later
                    /*W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] - learning_rate*product[j];*/

                    // Bug 2: The line below was incorrectly subtracting the product instead of adding it.
                    //grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] -= /*learning_rate**/product[j];

                    grad_W1[grad_W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + j] += grad_u_negative_sample[0] * W2_negative_sample[j];
                }
            }
        }
        else
        {            
            throw ala_exception("The array containing indices of negative samples is empty.");        
        }
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)
     */ 
    //return backward_propogation<T>{grad_W1, grad_W2, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}};

   //T * ptrptrptr = cc_tokenizer::allocator<T>().allocate(10);
    //DIMENSIONS temp1 = DIMENSIONS{10, 1, NULL, NULL};

    DIMENSIONS temp1 = DIMENSIONS{0, 0, NULL, NULL};
    Collective<T> temp2 = Collective<T>{NULL, temp1};       
    backward_propogation<T> ret = backward_propogation<T>{grad_W1, grad_W2, temp2};
    
    //std::cout<< "AT THE END OF BACKWARD PROPAGATION FUNCTION" << std::endl;

    return ret;
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
    @data_parser, instance of class csv_parser
    @epoch, number of times the training loop would iterate
    @W1, embedding matrix. Each row in W1 is a unique word's embedding vector, representing its semantic relationship with other words
    @W2, output layer. Weights for predicting context words
    @el, epoch loss
    @el_previous,
    @vocab, instance of class corpus
    @pairs, inctance of class skip gram pairs. The target/center word and its context words
    @lr, learning rate. The learning rate controls the step size at each iteration of the optimization process
    @lr_decay, learning rate decay, also known as learning rate scheduling. If you want the learning rate to remain constant throughout training, set the learning rate decay factor to 1.
    @rs, regulirazation strength. To prevent the model over-learning from the data
    @t, data type. Used as argument to templated types and functions
    @stf, Stop Training Flag, when set to true all training loops are stoped    
    @nns, Number of Negative Samples. The number of negative samples to generate for each positive word pair during training
    @default_clip_gradients_threshold, default threshold value for gradient clipping to prevent exploding gradients
    @shuffle_target_context_pairs, when true shuffles the training data (word pairs) before each epoch to avoid biases in weight updates
    @verbose, when true puts more text on screen to help debug code    
 */
#define SKIP_GRAM_TRAINING_LOOP(data_parser, epoch, W1, W2, el, el_previous, vocab, pairs, lr, lr_decay, rs, t, stf, nns, default_clip_gradients_threshold, shuffle_target_context_pairs, verbose)\
{\
    cc_tokenizer::string_character_traits<char>::size_type patience = 0;\
    Collective<cc_tokenizer::string_character_traits<char>::size_type> negative_sampling_table = buildNegativeSamplesTable(vocab);\
    /* Epoch loop */\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch && !stf; i++)\
    {\
        /* Initializes the epoch loss to 0 before accumulating errors from word pairs */\
        el = 0;\
        if (verbose)\
        {\
            std::cout<< "Epoch# " << i << " of " << epoch << " epochs." << std::endl;\
        }\
        forward_propogation<t> fp;\
        backward_propogation<t> bp;\
        Collective<cc_tokenizer::string_character_traits<char>::size_type> negative_samples;\
        \
        data_parser.reset(LINES);\
        data_parser.reset(TOKENS);\
        while (data_parser.go_to_next_line() != cc_tokenizer::string_character_traits<char>::eof() && !stf)\
        {\
            cc_tokenizer::String<char> line(data_parser.get_current_line() + cc_tokenizer::String<char>("\n"));\
            cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> line_parser(line);\
            CORPUS vocab_line(line_parser);\
            PAIRS pairs_line(vocab_line);\
            /* Shuffle Word Pairs: Shuffles the training data (word pairs) before each epoch to avoid biases in weight updates */\
            if (shuffle_target_context_pairs)\
            {\
                Numcy::Random::shuffle<PAIRS>(pairs_line, pairs_line.get_number_of_word_pairs());\
            }\
            /* Iterates through each word pair in the training data  */\
            while (pairs_line.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof() && !stf)\
            {\
                /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
                WORDPAIRS_PTR pair = pairs_line.get_current_word_pair();\
                try\
                {\
                    /*Collective<cc_tokenizer::string_character_traits<char>::size_type>*/ negative_samples = generateNegativeSamplesFromTable(negative_sampling_table, nns);\
                    /*\
                        Forward Propagation: The forward function performs forward propagation and calculate the hidden layer\
                        activation and predicted probabilities using the current word pair (pair), embedding matrix (W1),\
                        output weights (W2), vocabulary (vocab), and data type (t). The result is stored in the fp variable.\
                     */\
                    fp = forward<t>(W1, W2, negative_samples, vocab, pair);\
                    if(nns && verbose)\
                    {\
                        std::cout<< "This pair positive samples loss = " << fp.positive_samples_loss << ", and Negative samples loss = " << fp.negative_samples_loss << std::endl;\
                        el = el + fp.positive_negative_epoch_loss;\
                    }\
                    /*\ Backward Propagation: The backward function performs backward propagation and calculate the gradients\
                        with respect to the input and output layer weights using the forward propagation results (fp), word pair (pair),\
                        embedding matrix (W1), output weights (W2), vocabulary (vocab), and data type (t).\
                        The result is stored in the bp variable.\
                     */\
                    bp = backward<t>(W1, W2, negative_samples, vocab, fp, pair, false, lr);\
                    if (default_clip_gradients_threshold > 0)\
                    {\
                        clip_gradients(bp.grad_weights_input_to_hidden, /*AXIS_ROWS*/ AXIS_NONE, default_clip_gradients_threshold);\
                        clip_gradients(bp.grad_weights_hidden_to_output, /*AXIS_COLUMN*/ AXIS_NONE, default_clip_gradients_threshold);\
                    }\
                    /*if (!nns)*/\
                    {\
                        /* Update Weights */\
                        if (rs == 0)\
                        {\
                            W1 -= bp.grad_weights_input_to_hidden * lr;\
                            W2 -= bp.grad_weights_hidden_to_output * lr;\
                        }\
                        else\
                        {\
                            /* Relationship between learning rate (lr) and regularization strength (rs) */\
                            /* ------------------------------------------------------------------------ */\
                            /* High learning rate (lr) often requires higher regularization strength (rs) to prevent overfitting or unstable updates during training. */\
                            /* A high learning rate leads to larger parameter updates, which can result in the model overshooting the optimal solution or overfitting the training data. */\
                            /* Increasing the regularization strength helps by penalizing large weights, thus stabilizing the training */\
                            /* Low learning rate (lr) generally allows for either no regularization or lower regularization strength because smaller updates reduce the risk of overfitting. */\
                            /* In such cases, the model converges more slowly, and heavy regularization may not be necessary */\
                            \
                            /* L2 regularization (also known as weight decay). */\
                            /* ----------------------------------------------- */\
                            /* The regularization strength (rs) controls how much penalty is applied. */\
                            /* The weights are updated by subtracting the learning rate (lr) scaled by the sum of the gradient and the regularization term. */\
                            /* Ensure that the regularization strength (rs) is not too large, as it might lead to excessively penalizing the weights and slow down convergence. */\
                            /* The regularization expression W1 * rs and W2 * rs are added to gradient added to the gradient to penalize large weights, which helps prevent overfitting. */\
                            Collective<t>  W1_rs = W1 * rs;\
                            W1 -= ((bp.grad_weights_input_to_hidden + W1_rs /*(W1 * rs)*/) * lr);\
                            Collective<t>  W2_rs = W2 * rs;\
                            W2 -= ((bp.grad_weights_hidden_to_output + W2_rs /*(W2 * rs)*/) * lr);\
                            \
                            /*W1 -= (bp.grad_weights_input_to_hidden * lr) + (W1 * rs * lr);*/\
                            /*W2 -= ((bp.grad_weights_hidden_to_output + (W2 * rs)) * lr);*/\
                            \
                            /*W2 -= (bp.grad_weights_hidden_to_output * lr) + (W2 * rs * lr);*/\
                        }\
                        if (!nns)\
                        {\
                            /* Loss Function: The Skip-gram model typically uses negative log-likelihood (NLL) as the loss function.\
                               In NLL, lower values indicate better performance. */\
                            el = el + (-1*log(fp.pb(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)));\
                        }\
                        /*cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().deallocate(negative_samples_ptr);*/\
                    }\
                }\
                catch(ala_exception& e)\
                {\
                    std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
                    stf = true;\
                }\
            }\
            if (!stf)\
            {\
                if (!nns)\
                {\
                    if (el_previous == 0 || el < el_previous)\
                    {\
                        std::cout<< "epoch_loss = (" << el << "), Average epoch_loss = " << el/pairs.get_number_of_word_pairs() << ". Reduction between consecutive epochs: " << el_previous - el << "." << std::endl;\
                        \
                        el_previous = el;\
                    }\
                    else\
                    {\
                        std::cout<< "Average epoch_loss is increasing... from " << el_previous/pairs.get_number_of_word_pairs() << " to " << el/pairs.get_number_of_word_pairs() << std::endl;\
                        if (patience < DEFAULT_TRAINING_LOOP_PATIENCE)\
                        {\
                            patience = patience + 1;\
                        }\
                        else\
                        {\
                            stf = true;\
                        }\
                    }\
                }\
                /* Negative sampling is enabled */\
                else\
                {\
                    std::cout<< "Total negative positive sampling epoch loss = (" << fp.positive_negative_epoch_loss << "), Average epoch loss = " <<  fp.positive_negative_epoch_loss/pairs.get_number_of_word_pairs() << ", (" << el << ", " << el/pairs.get_number_of_word_pairs() << ")." << std::endl;\
                }\
            }\
            /* Multiply the learning rate by a decay factor, after each epoch. Specially, when you are not using negative sampling, start with higher learning rate and gradually decrease it at the completion of each epoch. If you want the learning rate to remain constant throughout training, set the learning rate decay factor to 1 */\
            lr = lr * lr_decay;\
        }\
    }\
}\

#define SKIP_GRAM_TRAINING_LOOP_WORKING(epoch, W1, W2, el, el_previous, vocab, pairs, lr, lr_decay, rs, t, stf, nns, default_clip_gradients_threshold, shuffle_target_context_pairs, verbose)\
{\
    cc_tokenizer::string_character_traits<char>::size_type patience = 0;\
    Collective<cc_tokenizer::string_character_traits<char>::size_type> negative_sampling_table = buildNegativeSamplesTable(vocab);\
    /* Epoch loop */\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch && !stf; i++)\
    {\
        /* Initializes the epoch loss to 0 before accumulating errors from word pairs */\
        el = 0;\
        /* Conditional block that prints the current epoch number if verbose is True */\
        if (verbose)\
        {\
            std::cout<< "Epoch# " << i << " of " << epoch << " epochs." << std::endl;\
        }\
        forward_propogation<t> fp;\
        backward_propogation<t> bp;\
        Collective<cc_tokenizer::string_character_traits<char>::size_type> negative_samples;\
        /* Shuffle Word Pairs: Shuffles the training data (word pairs) before each epoch to avoid biases in weight updates */\
        if (shuffle_target_context_pairs)\
        {\
            Numcy::Random::shuffle<PAIRS>(pairs, pairs.get_number_of_word_pairs());\
        }\
        /* Iterates through each word pair in the training data  */\
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof() && !stf)\
        {\
            /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
            /*forward_propogation<t> fp;*/\
            /*backward_propogation<t> bp;*/\
            try\
            {\
                /*Collective<cc_tokenizer::string_character_traits<char>::size_type>*/ negative_samples = generateNegativeSamplesFromTable(negative_sampling_table, nns);\
                /* Forward Propagation: The forward function performs forward propagation and calculate the hidden layer\
                   activation and predicted probabilities using the current word pair (pair), embedding matrix (W1),\
                   output weights (W2), vocabulary (vocab), and data type (t). The result is stored in the fp variable.*/\
                fp = forward<t>(W1, W2, negative_samples, vocab, pair);\
                if(nns && verbose)\
                {\
                    std::cout<< "This pair positive samples loss = " << fp.positive_samples_loss << ", and Negative samples loss = " << fp.negative_samples_loss << std::endl;\
                    el = el + fp.positive_negative_epoch_loss;\
                }\
                /* Backward Propagation: The backward function performs backward propagation and calculate the gradients\
                   with respect to the input and output layer weights using the forward propagation results (fp), word pair (pair),\
                   embedding matrix (W1), output weights (W2), vocabulary (vocab), and data type (t).\
                   The result is stored in the bp variable. */\
                bp = backward<t>(W1, W2, negative_samples, vocab, fp, pair, false, lr);\
                if (default_clip_gradients_threshold > 0)\
                {\
                    clip_gradients(bp.grad_weights_input_to_hidden, /*AXIS_ROWS*/ AXIS_NONE, default_clip_gradients_threshold);\
                    clip_gradients(bp.grad_weights_hidden_to_output, /*AXIS_COLUMN*/ AXIS_NONE, default_clip_gradients_threshold);\
                }\
                /*std::cout<< "AFTER BACKWARD" << std::endl;*/\
                if (!nns)\
                {\
                    /* Update Weights */\
                    if (rs == 0)\
                    {\
                        W1 -= bp.grad_weights_input_to_hidden * lr;\
                        W2 -= bp.grad_weights_hidden_to_output * lr;\
                    }\
                    else\
                    {\
                        /* Relationship between learning rate (lr) and regularization strength (rs) */\
                        /* ------------------------------------------------------------------------ */\
                        /* High learning rate (lr) often requires higher regularization strength (rs) to prevent overfitting or unstable updates during training. */\
                        /* A high learning rate leads to larger parameter updates, which can result in the model overshooting the optimal solution or overfitting the training data. */\
                        /* Increasing the regularization strength helps by penalizing large weights, thus stabilizing the training */\
                        /* Low learning rate (lr) generally allows for either no regularization or lower regularization strength because smaller updates reduce the risk of overfitting. */\
                        /* In such cases, the model converges more slowly, and heavy regularization may not be necessary */\
                        \
                        /* L2 regularization (also known as weight decay). */\
                        /* ----------------------------------------------- */\
                        /* The regularization strength (rs) controls how much penalty is applied. */\
                        /* The weights are updated by subtracting the learning rate (lr) scaled by the sum of the gradient and the regularization term. */\
                        /* Ensure that the regularization strength (rs) is not too large, as it might lead to excessively penalizing the weights and slow down convergence. */\
                        /* The regularization expression W1 * rs and W2 * rs are added to gradient added to the gradient to penalize large weights, which helps prevent overfitting. */\
                        Collective<t>  W1_rs = W1 * rs;\
                        W1 -= ((bp.grad_weights_input_to_hidden + W1_rs /*(W1 * rs)*/) * lr);\
                        Collective<t>  W2_rs = W2 * rs;\
                        W2 -= ((bp.grad_weights_hidden_to_output + W2_rs /*(W2 * rs)*/) * lr);\
                        \
                        /*W1 -= (bp.grad_weights_input_to_hidden * lr) + (W1 * rs * lr);*/\
                        /*W2 -= ((bp.grad_weights_hidden_to_output + (W2 * rs)) * lr);*/\
                        \
                        /*W2 -= (bp.grad_weights_hidden_to_output * lr) + (W2 * rs * lr);*/\
                    }\
                    /* Loss Function: The Skip-gram model typically uses negative log-likelihood (NLL) as the loss function.\
                       In NLL, lower values indicate better performance. */\
                    el = el + (-1*log(fp.pb(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)));\
                    /*cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().deallocate(negative_samples_ptr);*/\
                }\
                /*if (nns)*/\
                /*{*/\
                    /*clip_gradients(W1, AXIS_ROWS);*/\
                    /*clip_gradients(W2, AXIS_COLUMN);*/\
                /*}*/\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
                stf = true;\
            }\
        }\
        if (!stf)\
        {\
            if (!nns)\
            {\
                if (el_previous == 0 || el < el_previous)\
                {\
                    std::cout<< "epoch_loss = (" << el << "), Average epoch_loss = " << el/pairs.get_number_of_word_pairs() << ". Reduction between consecutive epochs: " << el_previous - el << "." << std::endl;\
                    \
                    el_previous = el;\
                }\
                else\
                {\
                    std::cout<< "Average epoch_loss is increasing... from " << el_previous/pairs.get_number_of_word_pairs() << " to " << el/pairs.get_number_of_word_pairs() << std::endl;\
                    if (patience < DEFAULT_TRAINING_LOOP_PATIENCE)\
                    {\
                        patience = patience + 1;\
                    }\
                    else\
                    {\
                        stf = true;\
                    }\
                }\
            }\
            /* Negative sampling is enabled */\
            else\
            {\
                std::cout<< "Total negative positive sampling epoch loss = (" << fp.positive_negative_epoch_loss << "), Average epoch loss = " <<  fp.positive_negative_epoch_loss/pairs.get_number_of_word_pairs() << ", (" << el << ", " << el/pairs.get_number_of_word_pairs() << ")." << std::endl;\
            }\
        }\
        /* Multiply the learning rate by a decay factor, after each epoch. Specially, when you are not using negative sampling, start with higher learning rate and gradually decrease it at the completion of each epoch. If you want the learning rate to remain constant throughout training, set the learning rate decay factor to 1 */\
        lr = lr * lr_decay;\
    }\
}\

#endif