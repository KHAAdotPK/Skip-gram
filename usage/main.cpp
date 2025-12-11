/*
    usage/main.cpp
    Q@khaa.pk
 */

#include "main.hh"

int main(int argc, char* argv[])
{ 
    ARG arg_corpus, arg_epoch, arg_help, arg_lr, arg_rs, arg_verbose, arg_loop, arg_input, arg_output, arg_ns, arg_show_pairs, arg_shuffle_target_context_pairs, arg_random_number_generator_seed, arg_save_initial_weights, arg_lr_decay, arg_clip_gradients_threshold;
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
    FIND_ARG(argv, argc, argsv_parser, "--input", arg_input);
    FIND_ARG(argv, argc, argsv_parser, "--output", arg_output);
    FIND_ARG(argv, argc, argsv_parser, "ns", arg_ns);
    FIND_ARG(argv, argc, argsv_parser, "show_pairs", arg_show_pairs);
    FIND_ARG(argv, argc, argsv_parser, "shuffle_target_context_pairs", arg_shuffle_target_context_pairs);
    FIND_ARG(argv, argc, argsv_parser, "--random_number_generator_seed", arg_random_number_generator_seed);
    FIND_ARG(argv, argc, argsv_parser, "save_initial_weights", arg_save_initial_weights);
    FIND_ARG(argv, argc, argsv_parser, "learning_rate_scheduling", arg_lr_decay);
    FIND_ARG(argv, argc, argsv_parser, "--clip_gradients_threshold", arg_clip_gradients_threshold);

    cc_tokenizer::string_character_traits<char>::size_type number_of_negative_samples = 0;
    if (arg_ns.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_ns);

        if (arg_ns.argc)
        {
            number_of_negative_samples = atoi(argv[arg_ns.i + 1]);
        }
        else
        {
            number_of_negative_samples = SKIP_GRAM_DEFAULT_NUMBER_OF_NEGATIVE_SAMPLES;
        }
    }

    double default_clip_gradients_threshold = 0.0;
    if (arg_clip_gradients_threshold.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_clip_gradients_threshold);

        if (arg_clip_gradients_threshold.argc)
        {
            default_clip_gradients_threshold = atof(argv[arg_clip_gradients_threshold.i + 1]);
        }
        else
        {
            default_clip_gradients_threshold = SKIP_GRAM_CLIP_GRADIENTS_DEFAULT_THRESHOLD;
        }
    }

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

    double default_lr_decay = 1.0; // Keep the learning rate constant throughout training
    if (arg_lr_decay.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_lr_decay);

        if (arg_lr_decay.argc)
        {
            default_lr_decay = atof(argv[arg_lr_decay.i + 1]);
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

    double default_random_number_generator_seed = 0;
    if (arg_random_number_generator_seed.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_random_number_generator_seed);

        if (arg_random_number_generator_seed.argc)
        {
            default_random_number_generator_seed = atof(argv[arg_random_number_generator_seed.i + 1]);
        }
        else
        {
            default_random_number_generator_seed = NUMCY_DEFAULT_RANDOM_NUMBER_GENERATOR_SEED;
        }
    }

    double default_lr = SKIP_GRAM_DEFAULT_LEARNING_RATE;
    if (arg_lr.i)
    {        
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_lr);

        if (arg_lr.argc)
        {
            default_lr = atof(argv[arg_lr.i + 1]);
        }
    }
        
    double default_rs = SKIP_GRAM_REGULARIZATION_STRENGTH;
    if (arg_rs.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_rs);

        if (arg_rs.argc)
        {
            default_rs = atof(argv[arg_rs.i + 1]);
        }
    }

    long default_loop = 1;
    if (arg_loop.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_loop);
        if (arg_loop.argc)
        {
            default_loop =  default_loop + atol(argv[arg_loop.i + 1]);
        }
        else
        {   
            default_loop = default_loop + 1;
        }
    }

    cc_tokenizer::String<char> W1OutPutFile(TRAINED_INPUT_WEIGHTS_FILE_NAME);
    cc_tokenizer::String<char> W2OutPutFile(TRAINED_OUTPUT_WEIGHTS_FILE_NAME);
    if (arg_output.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_output); 

        if (arg_output.argc > 1)
        {              
            W1OutPutFile = cc_tokenizer::String<char>(argv[arg_output.i + 1]);
            W2OutPutFile = cc_tokenizer::String<char>(argv[arg_output.i + 2]);
        }
        else
        {
            ARG arg_output_help;
            HELP(argsv_parser, arg_output_help, "output");                
            HELP_DUMP(argsv_parser, arg_output_help);

            return 0;
        }
    }
        
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> data_parser(data);
    class Corpus vocab(data_parser);
    PAIRS pairs(vocab, arg_show_pairs.i ? true : false);

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

    Collective<double> W_input_to_hidden /*= Collective<double>{NULL, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL}}*/;
    Collective<double> W_hidden_to_output /*= Collective<double>{NULL, DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}}*/;

    /*cc_tokenizer::String<char> W1InputFile;
    cc_tokenizer::String<char> W2InputFile;*/

    if (arg_input.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_input); 

        if (arg_input.argc > 1)
        {               
            W_input_to_hidden = Collective<double>{NULL, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/, NULL, NULL}};
            W_hidden_to_output = Collective<double>{NULL, DIMENSIONS{vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};

            std::cout<< "Dimensions of W1 = " << W_input_to_hidden.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W_input_to_hidden.getShape().getNumberOfColumns() << std::endl;
            std::cout<< "Dimensions of W2 = " << W_hidden_to_output.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W_hidden_to_output.getShape().getNumberOfColumns() << std::endl;

            READ_W_BIN(W_input_to_hidden, cc_tokenizer::String<char>(argv[arg_input.i + 1]), double);
            READ_W_BIN(W_hidden_to_output, cc_tokenizer::String<char>(argv[arg_input.i + 2]), double);

            //W1InputFile = cc_tokenizer::String<char>(argv[arg_input.i + 1]);
            //W2InputFile = cc_tokenizer::String<char>(argv[arg_input.i + 2]);

            /*cc_tokenizer::String<char>*/ //W1InputFile = cc_tokenizer::cooked_read<char>(cc_tokenizer::String<char>(argv[arg_input.i + 1]));
            /*cc_tokenizer::String<char>*/ //W2InputFile = cc_tokenizer::cooked_read<char>(cc_tokenizer::String<char>(argv[arg_input.i + 2]));

            //cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> w1trainedParser(W1InputFile);
            //cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> w2trainedParser(W2InputFile);

            /*READ_W1(w1trainedParser, W1);
            //READ_W2(w2trainedParser, W2);
            READ_W2_ChatGPT_With_W1(w2trainedParser, W1, W2);*/
            
            //std::cout<< "Dimensions of W1 = " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W1.getShape().getNumberOfColumns() << std::endl;
            //std::cout<< "Dimensions of W2 = " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W2.getShape().getNumberOfColumns() << std::endl;

            //return 0;
        }
        else
        {
            ARG arg_input_help;
            HELP(argsv_parser, arg_input_help, "--input");                
            HELP_DUMP(argsv_parser, arg_input_help);

            return 0;
        }
    }
    else 
    {
        try 
        {            
            /*
                The weights 𝑊1 and 𝑊2​ are initialized using random values drawn from a normal distribution, which is typical for training embeddings in skip-gram models. This approach prevents symmetry and allows gradients to flow during backpropagation.
             */
            W_input_to_hidden = Numcy::Random::randn<double>(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, /*vocab.numberOfTokens()*/ vocab.numberOfUniqueTokens(), NULL, NULL}, default_random_number_generator_seed);
            W_hidden_to_output = Numcy::Random::randn<double>(DIMENSIONS{vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}, default_random_number_generator_seed, AXIS_COLUMN);
            
            std::cout<< "Dimensions of W1 = " << W_input_to_hidden.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W_input_to_hidden.getShape().getNumberOfColumns() << std::endl;
            std::cout<< "Dimensions of W2 = " << W_hidden_to_output.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W_hidden_to_output.getShape().getNumberOfColumns() << std::endl;

            if (arg_save_initial_weights.i)
            {
                WRITE_W_BIN(W_input_to_hidden, cc_tokenizer::String<char>("W1_initial_weights.dat"), double);
                WRITE_W_BIN(W_hidden_to_output, cc_tokenizer::String<char>("W2_initial_weights.dat"), double);
            }            
        }
        catch (ala_exception& e)
        {
            std::cerr<< "main() Error: " << e.what() << std::endl;
        }
    }

    if (arg_save_initial_weights.i)
    {
        WRITE_W1_TO_TEXT_FILE(W_input_to_hidden, INITIAL_W1_WEIGHT_TXT_FILE, vocab);
        WRITE_W2_TO_TEXT_FILE(W_input_to_hidden, W_hidden_to_output, INITIAL_W2_WEIGHT_TXT_FILE, vocab);
    }    

    bool stop_training_flag = false;
    double epoch_loss = 0.0;
    double epoch_loss_previous = 0.0;
    
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
    for (long i = 0; i < default_loop && !stop_training_flag; i++)
    {
        SKIP_GRAM_TRAINING_LOOP(default_epoch, W_input_to_hidden, W_hidden_to_output, epoch_loss, epoch_loss_previous, vocab, pairs, default_lr, default_lr_decay, default_rs, double, stop_training_flag, /*arg_ns.i ? true : false*/ number_of_negative_samples, default_clip_gradients_threshold, arg_shuffle_target_context_pairs.i ? true : false, arg_verbose.i ? true : false);
    }
    
    /* 
        --------------------------------------
       ||  We need to store the weights now  ||
        --------------------------------------
     */

    std::cout<< "Trained input weights written to file: " << /*TRAINED_INPUT_WEIGHTS_FILE_NAME*/ W1OutPutFile.c_str() << std::endl;
                                 
    WRITE_W_BIN(W_input_to_hidden, W1OutPutFile.c_str(), double);
     
    std::cout<< "Trained output weights written to file: " << /*TRAINED_OUTPUT_WEIGHTS_FILE_NAME*/ W2OutPutFile.c_str() << std::endl;
            
    WRITE_W_BIN(W_hidden_to_output, W2OutPutFile.c_str(), double);
                   
    return 0;
}