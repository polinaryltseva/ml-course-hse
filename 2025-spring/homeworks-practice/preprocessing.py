from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    tree = open(filename).read().replace('&', '&amp;')
    root = ET.fromstring(tree)
    
    sentence_pairs = []
    alignments = []
    
    for s in root.findall('s'):

        english_elem = s.find('english')
        czech_elem = s.find('czech')
        english_text = english_elem.text.strip() if english_elem is not None and english_elem.text else ""
        english_tokens = english_text.split() if english_text else []
        czech_text = czech_elem.text.strip() if czech_elem is not None and czech_elem.text else ""
        czech_tokens = czech_text.split() if czech_text else []
        sentence_pairs.append(SentencePair(source=english_tokens, target=czech_tokens))
     
        sure_elem = s.find('sure')
        possible_elem = s.find('possible')
        sure_text = sure_elem.text.strip() if sure_elem is not None and sure_elem.text else ""
        possible_text = possible_elem.text.strip() if possible_elem is not None and possible_elem.text else ""
        
        sure = []
        if sure_text:
            for pair in sure_text.split():
                parts = pair.split('-')
                if len(parts) != 2:
                    continue 
                src, tgt = parts
                sure.append((int(src), int(tgt)))
                
        possible = []
        if possible_text:
            for pair in possible_text.split():
                parts = pair.split('-')
                if len(parts) != 2:
                    continue 
                src, tgt = parts
                possible.append((int(src), int(tgt)))
                
        alignments.append(LabeledAlignment(sure=sure, possible=possible))
    
    return sentence_pairs, alignments




def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_counter = Counter()
    target_counter = Counter()
    
    for pair in sentence_pairs:
        source_counter.update(pair.source)
        target_counter.update(pair.target)
    
    if freq_cutoff is not None:
        source_tokens = [token for token, _ in source_counter.most_common(freq_cutoff)]
        target_tokens = [token for token, _ in target_counter.most_common(freq_cutoff)]
    else:
        source_tokens = list(source_counter.keys())
        target_tokens = list(target_counter.keys())
    
    source_dict = {token: idx for idx, token in enumerate(source_tokens)}
    target_dict = {token: idx for idx, token in enumerate(target_tokens)}
    
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    for pair in sentence_pairs:
        source_indices = [source_dict[token] for token in pair.source if token in source_dict]
        target_indices = [target_dict[token] for token in pair.target if token in target_dict]

        if not source_indices or not target_indices:
            continue
        tokenized_sentence_pairs.append(TokenizedSentencePair(source_tokens=np.array(source_indices, dtype=np.int32),
            target_tokens=np.array(target_indices, dtype=np.int32)))
        
    return tokenized_sentence_pairs
