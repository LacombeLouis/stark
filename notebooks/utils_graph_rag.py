import itertools as it
from tqdm import tqdm

from utils_nlp import get_number_tokens
from utils_openai import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import math


LIST_NODE_TYPES = [
    'gene/protein', 'drug', 'effect/phenotype', 'disease',
    'biological_process', 'molecular_function', 'cellular_component',
    'exposure', 'pathway', 'anatomy'
]

def find_hybrid_entity(entity, graph, top_k=5, top_k_search=15, clean_results=True, return_ID=False, min_score=0.9, node_type=LIST_NODE_TYPES):
    entity = entity.replace("'", " ") # because neo4j struggles reading single quotes
    emb_ = get_embedding(entity)

    query = """
    CALL {
        CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) YIELD node, score 
        WITH collect({node:node, score:score}) AS nodes, max(score) AS vector_index_max_score 
        UNWIND nodes AS n 
        RETURN n.node AS node, (n.score / vector_index_max_score) AS score 
        UNION 
        CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) YIELD node, score 
        WITH collect({node:node, score:score}) AS nodes, max(score) AS ft_index_max_score 
        UNWIND nodes AS n 
        RETURN n.node AS node, (n.score / ft_index_max_score) AS score 
    } 
    WITH node, max(score) AS score
    RETURN score, node 
    ORDER BY score DESC
    LIMIT $top_k
    """

    params =  {
        'query_text': entity,  # Should match the value of 'entity'
        'top_k': top_k_search,
        'vector_index_name': 'entityEmbeddings',
        'fulltext_index_name': 'entityAndType',
        'query_vector': emb_,
        'min_score': min_score,
        'node_types': node_type
    }

    results = graph.query(query, params)

    if clean_results:
        results = clean_too_many_results(results, min_score)

    if results is None:
        return []
    else:
        if return_ID:
            return [int(item["node"]["id"]) for item in results[0]][:top_k]
        else:
            return [item["node"].__dict__['_element_id'] for item in results[0]][:top_k]


def clean_too_many_results(dict_, min_score):
    if dict_ is not None:
        # get min score from the results
        min_score_results = np.min([float(item["score"]) for item in dict_[0]])
        
        # remove entity that have a lower score than min_score
        results = [item for item in dict_[0] if float(item["score"]) >= min_score]

        if min_score_results > 0.99:
            results = None
        return results


def find_similar_entity(entity, graph, top_k=5, top_k_search=100, return_ID=False, min_score=0.9, node_type=LIST_NODE_TYPES):
    entity = entity.replace("'", " ") # because neo4j struggles reading single quotes
    emb_ = get_embedding(entity)

    query_template = """
    CALL db.index.vector.queryNodes($vector_index_name, $top_k_search, $query_vector)
    YIELD node, score
    WHERE score >= $min_score AND node.type IN $node_types
    RETURN score, node
    ORDER BY score DESC LIMIT $top_k
    """

    params =  {
        'query_text': entity,  # Should match the value of 'entity'
        'vector_index_name': 'entityEmbeddings',
        'query_vector': emb_,
        'min_score': min_score,
        'node_types': node_type,
        'top_k_search': top_k_search,
        'top_k': top_k
    }

    results = graph.query(query_template, params)

    if results is None:
        return []
    if return_ID:
        return [int(item["node"]["id"]) for item in results[0]]
    else:
        return [item["node"].__dict__['_element_id'] for item in results[0]]
    

def find_matching_nlp_entity(entity, graph, return_ID=False, lower_case=True, top_k=1000):
    entity = entity.replace("'", " ") # because neo4j struggles reading single quotes
    # query_template = "MATCH (e) WHERE trim(toLower(e.name)) CONTAINS trim(toLower($query_text)) RETURN e"
    # query_template = "MATCH (e) WHERE toLower(e.name) =~ '(?i).*\\\\b' + toLower($query_text) + '\\\\b.*' RETURN e"
    # query_template = "MATCH (e) WHERE toLower(e.name) = toLower($query_text) RETURN e"
    if lower_case:
        query_template = "MATCH (e) WHERE toLower(e.name) = toLower($query_text) RETURN e"
    else:
        query_template = "MATCH (e) WHERE e.name = $query_text RETURN e"

    params =  {
        'query_text': entity,  # Should match the value of 'entity'
    }

    results = graph.query(query_template, params)
    if return_ID:
        return [int(item['e']['id']) for item in results[0]][:top_k]
    else:
        return [item['e'].__dict__['_element_id'] for item in results[0]][:top_k]
    

def find_full_text_entity(entity, graph, top_k=10, return_ID=False):
    entity = entity.replace("'", " ") # because neo4j struggles reading single quotes
    query_template = """
    CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k})
    YIELD node, score
    RETURN node, score
    """
    params =  {
        'query_text': entity,
        'fulltext_index_name': 'entityAndType',
        'top_k': top_k
    }

    results = graph.query(query_template, params)

    # print("entity: ", entity)
    # for item in results[0]:
    #     print(item["node"])
    #     print(item["score"])
    # print("-"*50)

    if results is None:
        return []
    if return_ID:
        return [int(item["node"]["id"]) for item in results[0]]
    else:
        return [item[0].__dict__['_element_id'] for item in results[0]]



def find_matching_entity_name(elementId, graph):
    query_template = "MATCH (e) WHERE elementId(e) = '{elementId}' RETURN e.name"
    query = query_template.format(elementId=elementId)
    name_ = graph.query(query)
    return name_[0][0]['e.name']


def format_ids_output(output):
    """
    Get the ids from the output of the query

    Args:
        output (List[List[Dict]]) : The output of the query

    Returns:
        List[str] : The ids of the entities in the knowledge graph
    """
    return [entry['e'].__dict__['_element_id'] for entry in output[0]]


def format_paths(paths):
    output = ""
    for path in paths:
        output += f"{path}\n"
    return output


def get_combinations(data):
    # Get combinations of two items from different lists
    list_combinations = [
        (item1, item2)
        for (list1, list2) in it.combinations(data, 2)
        for item1 in list1
        for item2 in list2
    ]
    return list_combinations

def format_links(links):
    output = ""
    for link in links:
        output += f"{link}\n"
    return output


def delete_paths(paths, relations):
    return [path for path in paths if any(relation in path for relation in relations)]


def check_kg_ids(kg_ids):
    # Check if kg_ids has at least 2 elements
    is_valid_length = len(kg_ids) >= 2

    # # Check if each element in kg_ids is a list with at least 1 element
    # are_elements_valid = any(len(sublist) >= 1 for sublist in kg_ids)

    sum_elements = sum([len(sublist) for sublist in kg_ids])

    can_use_path = is_valid_length and sum_elements > 10
    return can_use_path


def limit_ids(list_ids, limit=100, show=False):
    if show:
        print("len of: ", [len(item) for item in list_ids])
    return [item for item in list_ids if len(item) < limit]


def delete_links(links, relations):
    return [link for link in links if any(relation in link for relation in relations)]


def split_text(text):
    return text.split(" -> ")


def max_score_ranking(texts, word_scores):
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
            
        max_score = np.max([word_scores.get(word, 0) for word in word_count])
        ranked_texts.append((text, max_score))
    return ranked_texts



def hybrid_weighted_ranking(texts, word_scores, alpha=0.5):
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)        
        total_words = np.sum(sum(word_count.values()))
    
        max_score = np.max([word_scores.get(word, 0) for word in word_count])
        weighted_sum = np.sum([word_count[word] * word_scores.get(word, 0) for word in word_count if word in word_scores])

        if total_words == 0:
            weighted_average = 0
        else:
            weighted_average = weighted_sum / total_words
        
        final_score = alpha * weighted_average + (1 - alpha) * max_score
        ranked_texts.append((text, final_score))
    return ranked_texts



def hybrid_ranking(texts, word_scores, alpha=0.5):
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
    
        max_score = np.max([word_scores.get(word, 0) for word in word_count])
        weighted_product = np.product([word_count[word] * word_scores.get(word, 0) for word in word_count if word in word_scores])
        
        final_score = alpha * weighted_product + (1 - alpha) * max_score
        ranked_texts.append((text, final_score))
    return ranked_texts


def compute_idf(word, all_texts):
    num_texts_with_word = sum(1 for text in all_texts if word in split_text(text))
    return math.log(len(all_texts) / (1 + num_texts_with_word))  # Add 1 to avoid division by zero


def tfidf_ranking(texts, word_scores):
    idf_scores = {}
    
    # Calculate IDF for each word
    for text in texts:
        for word in split_text(text):
            if word not in idf_scores:
                idf_scores[word] = compute_idf(word, texts)
    
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
        total_words = np.sum(sum(word_count.values()))

        tfidf_sum = np.sum([(word_count[word] / total_words) * idf_scores.get(word, 0) for word in word_count])
        relation_sum = np.sum([word_scores.get(word, 0) for word in word_scores])
        total_sum = (tfidf_sum)/len(words) + (relation_sum)/len(words)
        ranked_texts.append((text, total_sum))
    return ranked_texts


def tf_ranking(texts, word_scores):    
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
        total_words = np.sum(sum(word_count.values()))

        tfidf_sum = np.sum([(word_count[word] / total_words) for word in word_count])
        relation_sum = np.sum([word_scores.get(word, 0) for word in word_scores])
        total_sum = (tfidf_sum)/len(words) + (relation_sum)/len(words)
        ranked_texts.append((text, total_sum))
    return ranked_texts


def lists_to_dict(name_, score_):
    return {name_[i]: int(score_[i]) for i in range(len(name_))}


def replace_relations(connection_str):
    # Define a dictionary that maps relations to detailed sentences
    relation_to_sentence = {
        'ppi': 'has a physical interaction with',
        'carrier': 'binds and moves molecules across membranes, typically without energy',
        'enzyme': 'acts as a catalyst in a biochemical reaction involving',
        'target': 'is a specific target that interacts with',
        'transporter': 'actively moves substances across membranes, usually requiring energy',
        'contraindication': 'is a medical contraindication for',        
        'indication': 'is an indication for the treatment of',
        'off-label_use': 'is prescribed off-label for',
        'synergistic_interaction': 'has a synergistic interaction with',
        'associated_with': 'is associated with',
        'parent-child': 'is a parent node of',
        'phenotype_absent': 'does not exhibit the phenotype of',
        'phenotype_present': 'exhibits the phenotype of',
        'side_effect': 'is a side effect of',
        'interacts_with': 'interacts with',
        'linked_to': 'is functionally linked to',
        'expression_present': 'is expressed in',
        'expression_absent': 'is not expressed in'
    }
    # Iterate over the dictionary and replace relations with detailed sentences
    for relation, sentence in relation_to_sentence.items():
        connection_str = connection_str.replace(f" -> {relation} -> ", f" {sentence} ")

    # Capitalize the first letter of the connection string
    connection_str = connection_str[0].upper() + connection_str[1:]

    # Replace underscores with spaces
    connection_str = connection_str.replace("_", " ")

    # Add period at the end of the connection string
    connection_str = connection_str + "."
    return connection_str


def limit_paths(paths, dict_score, limit=6000, show=False):
    paths = tf_ranking(paths, dict_score)
    # paths = hybrid_ranking(paths, dict_score, alpha=0)

    paths.sort(key=lambda x: x[1], reverse=True)

    # remove score from the tuple
    paths = [path for path, _ in paths]

    # remove duplicates in paths
    paths = list(dict.fromkeys(paths))

    # # replace relations with detailed sentences
    # paths = [replace_relations(path) for path in paths]

    number_of_tokens_paths = [get_number_tokens(path) for path in paths]
    print("sum of tokens: ", sum(number_of_tokens_paths))

    # Calculate the cumulative sum using itertools.accumulate
    cumsum_list = list(it.accumulate(number_of_tokens_paths))

    # Find the first index where the cumulative sum exceeds 110,000
    cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > limit), len(paths))

    # Trim the paths list up to the cutoff index
    paths = paths[:cutoff_index]
    print("sum of tokens after: ", sum(number_of_tokens_paths[:cutoff_index]))
    return paths


def limit_links(links, dict_score, limit=6000, show=False):
    links = tf_ranking(links, dict_score)
    # links = hybrid_ranking(links, dict_score, alpha=0)

    links.sort(key=lambda x: x[1], reverse=True)

    links = [link for link, _ in links]

    # remove links that are the same while keeping the order
    links = list(dict.fromkeys(links))

    # # replace relations with detailed sentences
    # links = [replace_relations(link) for link in links]

    number_of_tokens_links = [get_number_tokens(link) for link in links]
    print("sum of tokens: ", sum(number_of_tokens_links))

    # Calculate the cumulative sum using itertools.accumulate
    cumsum_list = list(it.accumulate(number_of_tokens_links))

    # Find the first index where the cumulative sum exceeds 110,000
    cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > limit), len(links))

    # Trim the paths list up to the cutoff index
    links = links[:cutoff_index]

    print("sum of tokens after: ", sum(number_of_tokens_links[:cutoff_index]))
    return links