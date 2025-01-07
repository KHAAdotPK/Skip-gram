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

/*
    Early Stopping or Loss Threshold Tuning
    Since the loss reduction between epochs is consistently low but not zero, consider setting a slightly lower threshold (e.g., 5e-5) to allow the model to stop earlier if improvements become negligible.
 */
/*
    If the learning rate is relatively high, the epoch loss will decrease sharply, and the difference between two consecutive epoch losses will also be relatively high.
    On the other hand, if the learning rate is relatively low, the epoch loss will decrease gradually at a slower rate, resulting in a smaller difference between two consecutive epoch losses.
    We need to keep this in mind because, during training, the learning rate can decay or increase. In such situations, the difference between two consecutive epoch losses may suddenly increase or decrease.
*/
#ifndef SKIP_GRAM_LOSS_REDUCTION_THRESHOLD_BETWEEN_EPOCHS
#define SKIP_GRAM_LOSS_REDUCTION_THRESHOLD_BETWEEN_EPOCHS 1e-4
#endif 

#include "hyper-parameters.hh"
#include "../../../pairs/src/header.hh"
#include "skip-gram.hh"

#endif

