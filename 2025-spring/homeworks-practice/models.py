from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple
from collections import defaultdict

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        posteriors = []
        for sentence_pair in parallel_corpus:
            source_tokens = sentence_pair.source_tokens  
            target_tokens = sentence_pair.target_tokens

            if len(source_tokens) == 0 or len(target_tokens) == 0:
                posteriors.append(np.array([[]]))
                continue

            probs = self.translation_probs[np.ix_(source_tokens, target_tokens)]
            col_sums = probs.sum(axis=0)

            posterior = np.empty_like(probs)
            non_zero = col_sums > 0
          
            if np.any(non_zero):
                posterior[:, non_zero] = probs[:, non_zero] / col_sums[np.newaxis, non_zero]
            if np.any(~non_zero):
                posterior[:, ~non_zero] = 1.0 / len(source_tokens)

            posteriors.append(posterior)
        return posteriors

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        elbo = 0.0
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            source_tokens = sentence_pair.source_tokens
            target_tokens = sentence_pair.target_tokens

            if len(source_tokens) == 0 or len(target_tokens) == 0 or posterior.size == 0:
                continue

            src_len = len(source_tokens)
            log_src_len = np.log(src_len)
            probs = self.translation_probs[np.ix_(source_tokens, target_tokens)]
           
            mask = posterior > 0
            if np.any(mask):
                log_probs = np.log(np.maximum(probs, 1e-10)) - log_src_len
                elbo += np.sum(posterior[mask] * log_probs[mask])
                elbo -= np.sum(posterior[mask] * np.log(np.maximum(posterior[mask], 1e-10)))
        return elbo

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs.fill(0)  
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            if posterior.size == 0:
                continue

            source_tokens = sentence_pair.source_tokens
            target_tokens = sentence_pair.target_tokens
            src_idx, tgt_idx = np.meshgrid(source_tokens, target_tokens, indexing='ij')
            mask = posterior > 1e-10
            if np.any(mask):
                np.add.at(self.translation_probs, (src_idx[mask], tgt_idx[mask]), posterior[mask])

        row_sums = self.translation_probs.sum(axis=1)
        non_zero = row_sums > 0
        self.translation_probs[non_zero] /= row_sums[non_zero, np.newaxis]
        self.translation_probs[~non_zero] = 1.0 / self.num_target_words

        return self._compute_elbo(parallel_corpus, posteriors)


    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        result = []
        posteriors = self._e_step(sentences)
        for sentence_pair, posterior in zip(sentences, posteriors):
            tgt_len = len(sentence_pair.target_tokens)
            if posterior.size == 0:
                result.append([])
            else:
                best_indices = np.argmax(posterior, axis=0)
                alignment = list(zip(best_indices + 1, np.arange(tgt_len) + 1))
                result.append(alignment)
        return result


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        key = (src_length, tgt_length)
        if key not in self.alignment_probs:
            self.alignment_probs[key] = np.full((src_length, tgt_length), 1.0 / src_length, dtype=np.float32)
        return self.alignment_probs[key]

    def _e_step(self, parallel_corpus):
        posteriors = []
        for sentence_pair in parallel_corpus:
            source_tokens = sentence_pair.source_tokens  
            target_tokens = sentence_pair.target_tokens

            if len(source_tokens) == 0 or len(target_tokens) == 0:
                posteriors.append(np.array([[]]))
                continue

            src_len = len(source_tokens)
            tgt_len = len(target_tokens)
            phi = self._get_probs_for_lengths(src_len, tgt_len) 
            probs = self.translation_probs[np.ix_(source_tokens, target_tokens)]  
            joint = phi * probs
            col_sums = joint.sum(axis=0)
            posterior = np.empty_like(joint)
            non_zero = col_sums > 0
            if np.any(non_zero):
                posterior[:, non_zero] = joint[:, non_zero] / col_sums[np.newaxis, non_zero]
            if np.any(~non_zero):
                posterior[:, ~non_zero] = 1.0 / src_len
            posteriors.append(posterior)
        return posteriors

    def _compute_elbo(self, parallel_corpus, posteriors):
        elbo = 0.0
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            source_tokens = sentence_pair.source_tokens  
            target_tokens = sentence_pair.target_tokens
            if len(source_tokens) == 0 or len(target_tokens) == 0 or posterior.size == 0:
                continue

            src_len = len(source_tokens)
            tgt_len = len(target_tokens)
            phi = self._get_probs_for_lengths(src_len, tgt_len)  
            probs = self.translation_probs[np.ix_(source_tokens, target_tokens)]
            log_joint = np.log(np.maximum(phi, 1e-10)) + np.log(np.maximum(probs, 1e-10))
            elbo += np.sum(posterior * log_joint)
            elbo -= np.sum(posterior * np.log(np.maximum(posterior, 1e-10)))
        return elbo

    def _m_step(self, parallel_corpus, posteriors):
        self.translation_probs.fill(0)
        phi_counts = {}
        sentence_counts = {}
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            if posterior.size == 0:
                continue

            source_tokens = sentence_pair.source_tokens  
            target_tokens = sentence_pair.target_tokens
            src_len = len(source_tokens)
            tgt_len = len(target_tokens)
            src_idx, tgt_idx = np.meshgrid(source_tokens, target_tokens, indexing='ij')
            mask = posterior > 1e-10
            if np.any(mask):
                np.add.at(self.translation_probs, (src_idx[mask], tgt_idx[mask]), posterior[mask])
            
            key = (src_len, tgt_len)
            if key not in phi_counts:
                phi_counts[key] = np.zeros((src_len, tgt_len), dtype=np.float32)
                sentence_counts[key] = 0
            phi_counts[key] += posterior 
            sentence_counts[key] += 1

        row_sums = self.translation_probs.sum(axis=1)
        non_zero = row_sums > 0
        self.translation_probs[non_zero] /= row_sums[non_zero, np.newaxis]
        self.translation_probs[~non_zero] = 1.0 / self.num_target_words

        for key, count_matrix in phi_counts.items():
            src_len, tgt_len = key
            avg_phi = count_matrix / sentence_counts[key]
            col_sums = avg_phi.sum(axis=0)
            for i in range(tgt_len):
                if col_sums[i] > 0:
                    avg_phi[:, i] /= col_sums[i]
                else:
                    avg_phi[:, i] = 1.0 / src_len
            self.alignment_probs[key] = avg_phi

        return self._compute_elbo(parallel_corpus, posteriors)