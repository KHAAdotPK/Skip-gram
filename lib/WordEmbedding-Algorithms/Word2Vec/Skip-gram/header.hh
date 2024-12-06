/*
    lib/WordEmbedding-Algorithms/Word2Vec/skip-gram/header.hh
    Q@khaa.pk
 */

#ifndef WORDEMBEDDING_ALGORITHMS_WORD2VEC_SKIP_GRAM_HEADER_HH
#define WORDEMBEDDING_ALGORITHMS_WORD2VEC_SKIP_GRAM_HEADER_HH

#include "../../../../lib/Numcy/header.hh"
#include "../../../../lib/corpus/corpus.hh"

/*
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
struct negative_samples_collective
{
    E* ptr;
    E n;
};
typedef negative_samples_collective<cc_tokenizer::string_character_traits<char>::size_type> NEGATIVE_SAMPLES_COLLECTIVE; 
typedef NEGATIVE_SAMPLES_COLLECTIVE* NEGATIVE_SAMPLES_COLLECTIVE_PTR;
 */

#include "hyper-parameters.hh"
#include "../../../pairs/src/header.hh"
#include "skip-gram.hh"

#endif

