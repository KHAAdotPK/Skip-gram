```C++
#ifdef  GO_FOR_L1_CODE_BLOCK\
            /* L1 regularization, WIP(Work in proess) */\
            Collective<t> summed;\
            Collective<t> Wx_signs;\
            try\
            {\
                /*Extract the corresponding word embedding from the weight matrix 𝑊1.\
                  Instead of directly using a one-hot input vector, this implementation uses a linked list of word pairs.\
                  Each pair provides the index of the center word, which serves to extract the relevant embedding from 𝑊1.\
                  The embedding for the center word is stored in the hidden layer vector h.*/\
                t* ptr = cc_tokenizer::allocator<t>().allocate(W1.getShape().getNumberOfColumns());\
                /*Loop through the columns of W1 to extract the embedding for the center word.*/\
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)\
                {\
                    ptr[i] = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];\
                }\
                /*Create a vector 'target_W1_row_signs' that contains the sign of each element in the selected row of the weight matrix W1.*/\
                /*Collective<t> */Wx_signs = Numcy::sign(Collective<t>{ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}});\
                /*Allocate memory for a single element of type 't'.\
                  This element will hold the regularization strength (rs), which is a hyperparameter used to tune the model.*/\
                t* multiplier = cc_tokenizer::allocator<t>().allocate(1);\
                *multiplier = rs;\
                Collective<t> regularization_strength = Collective<t>{multiplier, DIMENSIONS{1, 1, NULL, NULL}};\
                /*Compute the dot product between the 'target_W1_row_signs' vector and the 'regularization_strength'.\
                 'target_W1_row_signs' contains the signs of the elements in the selected row of the weight matrix W1.\
                 'regularization_strength' is a scalar value representing the regularization strength (rs).\
                 The result 'product' will be used to apply regularization to the model's parameters.*/\
                Collective<t> product = Numcy::dot(Wx_signs, regularization_strength);\
                /*std::cout<< product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << ", " << product.getShape().getNumberOfColumns() << std::endl;*/\
                /*std::cout<< bp.grad_weights_input_to_hidden.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << ", " << bp.grad_weights_input_to_hidden.getShape().getNumberOfColumns() << std::endl;*/\
                /*Compute the element-wise sum of the gradient matrix 'bp.grad_weights_input_to_hidden' and the 'product' matrix along the specified axis (rows).\
                  'bp.grad_weights_input_to_hidden' contains the gradients with respect to the weights between the input and hidden layers.\
                  'product' is the result of the dot product of 'target_W1_row_signs' and 'regularization_strength'.\
                  The 'AXIS_ROWS' parameter specifies that the sum operation is performed along the rows of the matrices.\
                  The result 'summed' will hold the updated gradient values, incorporating both the original gradients and the regularization term.*/\
                /*Collective<t>*/ summed = Numcy::sum(bp.grad_weights_input_to_hidden, product, AXIS_ROWS);\
                /*std::cout<< summed.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << ", " << summed.getShape().getNumberOfColumns() << std::endl;*/\
                /*std::cout<< bp.grad_weights_input_to_hidden.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << ", " << bp.grad_weights_input_to_hidden.getShape().getNumberOfColumns() << std::endl;*/\
                /*Update the gradient matrix 'bp.grad_weights_input_to_hidden' with the new values computed in 'summed'.\
                  Iterate over each element in the gradient matrix 'bp.grad_weights_input_to_hidden'.\
                  The loop runs from 0 to the total number of elements in 'grad_weights_input_to_hidden'.\
                  For each element, assign the corresponding value from the 'summed' matrix to 'bp.grad_weights_input_to_hidden'.\
                  This ensures that the gradient matrix now contains the updated gradients, including the regularization term.*/\
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < bp.grad_weights_input_to_hidden.getShape().getN(); i++)\
                {\
                    bp.grad_weights_input_to_hidden[i] = summed[i];\
                }\
                /*Collective<t> */Wx_signs = Numcy::sign(W2);\
                product = Numcy::dot(Wx_signs, regularization_strength);\
                summed = Numcy::sum(W2, product, AXIS_NONE);\
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getN(); i++)\
                {\
                    W2[i] = summed[i];\
                }\
            }\
            catch (std::length_error& e)\
            {\
                std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
            }\
            catch(std::bad_alloc& e)\
            {\
                std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
            }\
#endif\
```