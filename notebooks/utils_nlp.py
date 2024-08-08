import tiktoken
import json
import re


def check_word_in_text(word, text):
    return word in text


def get_number_tokens(sentence, encoding_type = "cl100k_base"):
    enc = tiktoken.get_encoding(encoding_type)   
    return len(enc.encode(sentence))

