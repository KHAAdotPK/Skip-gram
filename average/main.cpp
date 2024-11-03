#include "main.hh"

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cout<< "Required option missing... <corpus.txt W1trained.txt W2trained.txt averaged.txt>" << std::endl;

        return 0;
    }

    std::cout<< "Processing: " << argv[1] << std::endl;

    cc_tokenizer::String<char> data = cc_tokenizer::cooked_read<char>(argv[1]);
    cc_tokenizer::String<char> data1 = cc_tokenizer::cooked_read<char>(argv[2]);
    cc_tokenizer::String<char> data2 = cc_tokenizer::cooked_read<char>(argv[3]);

    CORPUS vocab(data);
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> parser1(data1);
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> parser2(data2);

    Collective<double> W1;
    Collective<double> W2;

    READ_W1(parser1, W1);
    READ_W2_ChatGPT(parser2, W2);

    std::cout<< "Dimensions of W1 = " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W1.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "Dimensions of W2 = " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W2.getShape().getNumberOfColumns() << std::endl;   

    std::cout<< "\nAveraging and upating file: " << argv[3] << std::endl; 

    try 
    {
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getNumberOfColumns(); i++) // 224
        {
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); j++) // 50
            {
                W2[j*W2.getShape().getNumberOfColumns() + i] = ((W1[i*W1.getShape().getNumberOfColumns() + j] + (W2[j*W2.getShape().getNumberOfColumns() + i] * 2)) / 2);
            }  
        }
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;
    }

    parser2.reset(LINES);
    parser2.reset(TOKENS);

    WRITE_W2(W2, cc_tokenizer::String<char>(argv[4]), vocab);
    
    return 0;
}