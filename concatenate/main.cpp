#include "main.hh"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout<< "Required option missing... <W1trained.txt W2trained.txt concatenate.txt>" << std::endl;

        return 0;
    }

    std::cout<< "Processing: " << argv[1] << std::endl;

    cc_tokenizer::String<char> data1 = cc_tokenizer::cooked_read<char>(argv[1]);
    cc_tokenizer::String<char> data2 = cc_tokenizer::cooked_read<char>(argv[2]);

    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> parser1(data1);
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> parser2(data2);

    /*if (parser1.get_total_number_of_lines() != parser2.get_total_number_of_lines()) 
    {
        std::cout<< "Incompatible shapes." << std::endl;

        std::cout<< parser1.get_total_number_of_lines() << " ----- " << parser2.get_total_number_of_lines() << std::endl;

        return 0;
    }*/

    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= parser1.get_total_number_of_lines(); i++)
    {
        cc_tokenizer::String<char> line = parser1.get_line_by_number(i);
        parser2.get_line_by_number(i);

        for (cc_tokenizer::string_character_traits<char>::size_type j = 2; j <= parser2.get_total_number_of_tokens(); j++)
        {
            line = line + cc_tokenizer::String<char>(" ") + parser2.get_token_by_number(j);
        }

        line = line + cc_tokenizer::String<char>("\n");

        cc_tokenizer::cooked_write<char>(cc_tokenizer::String<char>(argv[3]), line);
    }


    return 0;
}