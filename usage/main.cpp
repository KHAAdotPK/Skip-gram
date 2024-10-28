/*
    usage/main.cpp
    Q@khaa.pk
 */

#include "main.hh"

int main(int argc, char* argv[])
{ 
    ARG arg_corpus, arg_epoch, arg_help, arg_lr, arg_rs, arg_verbose, arg_loop;
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
    cc_tokenizer::String<char> data;

    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    if (argc < 2)
    {        
        HELP(argsv_parser, arg_help, "help");                
        HELP_DUMP(argsv_parser, arg_help);

        return 0;                     
    }

    FIND_ARG(argv, argc, argsv_parser, "verbose", arg_verbose);
    FIND_ARG(argv, argc, argsv_parser, "corpus", arg_corpus);
    FIND_ARG(argv, argc, argsv_parser, "lr", arg_lr);
    FIND_ARG(argv, argc, argsv_parser, "rs", arg_rs);
    FIND_ARG(argv, argc, argsv_parser, "loop", arg_loop);

    if (arg_corpus.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_corpus);
        if (arg_corpus.argc)
        {            
            try 
            {
                data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + 1]);
                if (arg_verbose.i)
                {
                    std::cout<< "Corpus: " << argv[arg_corpus.i + 1] << std::endl;
                }
            }
            catch (ala_exception e)
            {
                std::cout<<e.what()<<std::endl;
                return -1;
            }            
        }
        else
        { 
            ARG arg_corpus_help;
            HELP(argsv_parser, arg_corpus_help, "--corpus");                
            HELP_DUMP(argsv_parser, arg_corpus_help);

            return 0; 
        }                
    }
    else
    {
        try
        {        
            data = cc_tokenizer::cooked_read<char>(SKIP_GRAM_DEFAULT_CORPUS_FILE);
            if (arg_verbose.i)
            {
                std::cout<< "Corpus: " << SKIP_GRAM_DEFAULT_CORPUS_FILE << std::endl;
            }
        }
        catch (ala_exception e)
        {
            std::cout<<e.what()<<std::endl;
            return -1;
        }
    }

    /*        
        In the context of training a machine learning model, an epoch is defined as a complete pass over the entire training dataset during training.
        One epoch is completed when the model has made one update to the weights based on each training sample in the dataset.
        In other words, during one epoch, the model has seen every example in the dataset once and has made one update to the model parameters for each example.

        The number of epochs to train for is typically set as a hyperparameter, and it depends on the specific problem and the size of the dataset. 
        One common approach is to monitor the performance of the model on a validation set during training, and stop training when the performance 
        on the validation set starts to degrade.
     */
    unsigned long default_epoch = SKIP_GRAM_DEFAULT_EPOCH;    
    FIND_ARG(argv, argc, argsv_parser, "e", arg_epoch);
    if (arg_epoch.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_epoch);

        if (arg_epoch.argc)
        {            
            default_epoch = atoi(argv[arg_epoch.i + 1]);            
        }
        else
        {
            ARG arg_epoch_help;
            HELP(argsv_parser, arg_epoch_help, "e");                
            HELP_DUMP(argsv_parser, arg_epoch_help);

            return 0;
        }                
    }

    double default_lr = SKIP_GRAM_DEFAULT_LEARNING_RATE;
    if (arg_lr.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_lr);

        if (arg_lr.argc)
        {
            default_lr = atof(argv[arg_lr.j]);
        }
    }

    double default_rs = SKIP_GRAM_REGULARIZATION_STRENGTH;
    if (arg_rs.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_rs);

        if (arg_rs.argc)
        {
            default_rs = atof(argv[arg_rs.j]);
        }
    }

    long default_loop = 1;
    if (arg_loop.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_loop);
        if (arg_loop.argc)
        {
            default_loop =  default_loop + atol(argv[arg_rs.j]);
        }
        else
        {   
            default_loop = default_loop + 1;
        }
    }
    
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> data_parser(data);
    class Corpus vocab(data_parser);
    PAIRS pairs(vocab/*, arg_verbose.i ? true : false*/);

    /*
        For the neural network itself, Skip-gram typically uses a simple architecture. 

        Each row in W1 represents the embedding vector for one specific center word in your vocabulary(so in W1 word redendency is not allowed).
        During training, the central word from a word pair is looked up in W1 to retrieve its embedding vector.
        The size of embedding vector is hyperparameter(SKIP_GRAM_EMBEDDING_VECTOR_SIZE). It could be between 100 to 300 per center word.

        Each row in W2 represents the weight vector for predicting a specific context word (considering both positive and negative samples).
        The embedding vector of the central word (from W1) is multiplied by W2 to get a score for each context word.

        Hence the skip-gram variant takes a target word and tries to predict the surrounding context words.

        Why Predict Context Words?
        1. By predicting context words based on the central word's embedding, Skip-gram learns to capture semantic relationships between words.
        2. Words that often appear together in similar contexts are likely to have similar embeddings.
     */
    /*
        * Skip-gram uses a shallow architecture with two weight matrices, W1 and W2.

        * W1: Embedding Matrix
          - Each row in W1 is a unique word's embedding vector, representing its semantic relationship with other words.
          - The size of this embedding vector (SKIP_GRAM_EMBEDDING_VECTOR_SIZE) is a hyperparameter, typically ranging from 100 to 300.

        * W2: Output Layer (weights for predicting context words)
          - Each row in W2 represents the weight vector for predicting a specific context word (considering both positive and negative samples).
          - The embedding vector of the central word (from W1) is multiplied by W2 to get a score for each context word.

        * By predicting surrounding context words based on the central word's embedding, Skip-gram learns to capture semantic relationships between words with similar contexts.
     */

    Collective<double> W1;
    Collective<double> W2;

    try 
    {
        W1 = Numcy::Random::randn(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
        W2 = Numcy::Random::randn(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;
    }

    double epoch_loss = 0.0;

    /*
               ---- 
            ----------  
           ----------------     
        ----------------------
        FOR DEBUGGING PURPOSES
        ----------------------
           ----------------
              ----------
                 ----          
     */
    /*
    try 
    {    
        Numcy::Random::shuffle<PAIRS>(pairs, pairs.get_number_of_word_pairs());
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
        {
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();

            forward_propogation<double> fp = forward(W1, W2, vocab, pair); 
            std::cout<< "---------------------------------------------------------------------------------" << std::endl;         
            backward_propogation<double> bp = backward(W1, W2, vocab, fp, pair, false);
        }
        std::cout<< std::endl;
    }
    catch (ala_exception &e)
    {
        std::cout<< e.what() << std::endl;
    }
     */
    /*
                 -------------- 
              --------------------  
           --------------------------     
        --------------------------------
        FOR DEBUGGING PURPOSES ENDS HERE
        --------------------------------
           --------------------------
              --------------------
                 --------------          
     */ 

    
    /* Start training. */
    for (long i = 0; i < default_loop; i++)
    {
        SKIP_GRAM_TRAINING_LOOP(default_epoch, W1, W2, epoch_loss, vocab, pairs, default_lr, default_rs, double, arg_verbose.i ? true : false);
    }

    /* 
        --------------------------------------
       ||  We need to store the weights now  ||
        --------------------------------------
     */
    
    std::cout<< "Trained input weights written to file: " << TRAINED_INPUT_WEIGHTS_FILE_NAME << std::endl;

    /*
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < vocab.numberOfUniqueTokens(); i++)
    {
        cc_tokenizer::String<char> line;    
       
        line = line + vocab[i + INDEX_ORIGINATES_AT_VALUE] + cc_tokenizer::String<char>(" ");

        cc_tokenizer::string_character_traits<char>::size_type j = 0;

        for (; j < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE;)
        {            
            cc_tokenizer::String<char> num(W1[i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + j]);

            j++;

            if (j < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
            {    
                line = line + num + cc_tokenizer::String<char>(" ");
            }
            else
            {
                line = line + num;
            }
        }
        
        line = line + cc_tokenizer::String<char>("\n");

        cc_tokenizer::cooked_write(cc_tokenizer::String<char>(TRAINED_INPUT_WEIGHTS_FILE_NAME), line);
    } 
     */

    WRITE_W1(W1, cc_tokenizer::String<char>(TRAINED_INPUT_WEIGHTS_FILE_NAME), vocab);

    std::cout<< "Trained output weights written to file: " << TRAINED_OUTPUT_WEIGHTS_FILE_NAME << std::endl;

    /*
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < vocab.numberOfUniqueTokens(); i++)
    {
        cc_tokenizer::String<char> line; 

        line = line + vocab[i + INDEX_ORIGINATES_AT_VALUE] + cc_tokenizer::String<char>(" ");

        cc_tokenizer::string_character_traits<char>::size_type j = 0;

        for (; j < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE;)
        {
            cc_tokenizer::String<char> num(W2[j*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i]);

            j++;

            if (j < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
            {    
                line = line + num + cc_tokenizer::String<char>(" ");
            }
            else
            {
                line = line + num;
            }
        }

        line = line + cc_tokenizer::String<char>("\n");

        cc_tokenizer::cooked_write(cc_tokenizer::String<char>(TRAINED_OUTPUT_WEIGHTS_FILE_NAME), line);    
    }
     */

    WRITE_W2(W2, cc_tokenizer::String<char>(TRAINED_OUTPUT_WEIGHTS_FILE_NAME), vocab);
               
    return 0;
}