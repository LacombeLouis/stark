import itertools as it
from tqdm import tqdm

from utils_nlp import get_number_tokens
from utils_openai import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math


def find_similar_entity(entity, graph, k=5, return_ID=False, min_score=0.9):
    emb_ = get_embedding(entity)

    entity = entity.replace("'", " ")
    word = f"'{entity}'"
    query_template = """
    CALL db.index.vector.queryNodes('entityEmbeddings', 100, {emb_})
    YIELD node, score
    WHERE score >= {min_score}
    RETURN node, score
    ORDER BY score DESC LIMIT {k}
    """

    query = query_template.format(emb_=emb_, k=k, min_score=min_score)
    # print(query)
    results = graph.query(query)

    # for item in results[0]:
    #     print(item)

    if return_ID:
        return [int(item["node"]["id"]) for item in results[0]]
    else:
        return [item[0].__dict__['_element_id'] for item in results[0]]
    

def find_matching_nlp_entity(entity, graph, return_ID=False):
    # # query_template = "MATCH (e) WHERE trim(toLower(e.name)) CONTAINS trim(toLower({word})) RETURN e"
    # # query_template = "MATCH (e) WHERE toLower(e.name) =~ '(?i).*\\\\b' + toLower({word}) + '\\\\b.*' RETURN e"
    # query_template = "MATCH (e) WHERE toLower(e.name) = toLower({word}) RETURN e"
    entity = entity.replace("'", " ")
    word = f"'{entity}'"
    query_template = "MATCH (e) WHERE toLower(e.name) = toLower({word}) RETURN e"
    query = query_template.format(word=word)
    response = graph.query(query)
    if return_ID:
        return [int(item['e']['id']) for item in response[0]]
    else:
        return [item['e'].__dict__['_element_id'] for item in response[0]]
    

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


# def limit_paths_v2(question, paths, limit=250, show=False):
#     # paths = delete_paths(paths, relations)

#     emb_question = get_embedding(question)
#     emb_paths = [get_embedding(path) for path in tqdm(paths)]

#     similarity_scores = [cosine_similarity(emb_question, emb_path) for emb_path in tqdm(emb_paths)]

#     # sort paths by the similarity scores in descending order
#     paths = sorted(zip(paths, similarity_scores), key=lambda x: x[1], reverse=True)
#     paths = paths[:limit]
#     return paths


def check_kg_ids(kg_ids):
    # Check if kg_ids has at least 2 elements
    is_valid_length = len(kg_ids) >= 2

    # Check if each element in kg_ids is a list with at least 1 element
    are_elements_valid = any(len(sublist) >= 1 for sublist in kg_ids)

    sum_elements = sum([len(sublist) for sublist in kg_ids])

    can_use_path = is_valid_length and are_elements_valid and sum_elements > 5
    return can_use_path


def limit_ids(list_ids, limit=100, show=False):
    if show:
        print("len of: ", [len(item) for item in list_ids])
    return [item for item in list_ids if len(item) < limit]


def delete_links(links, relations):
    return [link for link in links if any(relation in link for relation in relations)]


def split_text(text):
    return text.split(" -> ")


def weighted_sum_ranking(texts, word_scores):
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
        
        weighted_sum = sum(word_count[word] * word_scores.get(word, 0) for word in word_count)
        
        ranked_texts.append((text, weighted_sum))
    
    # Sort texts by their weighted sum score in descending order
    ranked_texts.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_texts


def weighted_average_ranking(texts, word_scores):
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
        
        total_weighted_score = sum(word_count[word] * word_scores.get(word, 0) for word in word_count)
        total_words = sum(word_count.values())
        
        if total_words == 0:
            weighted_average = 0
        else:
            weighted_average = total_weighted_score / total_words
        
        ranked_texts.append((text, weighted_average))
    
    # Sort texts by their weighted average score in descending order
    ranked_texts.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_texts


def max_score_ranking(texts, word_scores):
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
        
        max_score = max((word_scores.get(word, 0) for word in word_count), default=0)
        
        ranked_texts.append((text, max_score))
    
    # Sort texts by their max score in descending order
    ranked_texts.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_texts



def hybrid_ranking(texts, word_scores, alpha=0.5):
    ranked_texts = []
    
    for text in texts:
        words = split_text(text)
        word_count = Counter(words)
        
        weighted_sum = sum(word_count[word] * word_scores.get(word, 0) for word in word_count)
        max_score = max((word_scores.get(word, 0) for word in word_count), default=0)
        
        final_score = alpha * weighted_sum + (1 - alpha) * max_score
        ranked_texts.append((text, final_score))
    
    # Sort texts by their final score in descending order
    ranked_texts.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_texts



def compute_idf(word, all_texts):
    num_texts_with_word = sum(1 for text in all_texts if word in split_text(text))
    return math.log(len(all_texts) / (1 + num_texts_with_word))  # Add 1 to avoid division by zero

def tfidf_ranking(texts, word_scores):
    idf_scores = {}
    
    # Calculate IDF for each word
    for text in texts:
        for word in text.split():
            if word not in idf_scores:
                idf_scores[word] = compute_idf(word, texts)
    
    ranked_texts = []
    
    for text in texts:
        words = text.split()
        word_count = Counter(words)
        total_words = len(words)
        
        tfidf_sum = sum((word_count[word] / total_words) * idf_scores.get(word, 0) * word_scores.get(word, 0) 
                        for word in word_count)
        
        ranked_texts.append((text, tfidf_sum))
    
    # Sort texts by their TF-IDF score in descending order
    ranked_texts.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_texts


def lists_to_dict(name_, score_):
    return {name_[i]: int(score_[i]) for i in range(len(name_))}


def replace_relations(connection_str):
    # Define a dictionary that maps relations to detailed sentences
    relation_to_sentence = {
        'ppi': 'has a physical interaction with',
        'carrier': 'is involved in the transport or movement of',
        'enzyme': 'acts as a catalyst in a biochemical reaction affecting',
        'target': 'is a specific target that interacts with or inhibits',
        'transporter': 'helps move substances, including',
        'contraindication': 'is a contraindication for',
        'indication': 'is an indication for the treatment of',
        'off-label_use': 'is prescribed off-label for',
        'synergistic_interaction': 'has a synergistic interaction with',
        'associated_with': 'is associated with',
        'parent-child': 'is a parent entity to',
        'phenotype_absent': 'does not exhibit the phenotype of',
        'phenotype_present': 'exhibits the phenotype of',
        'side_effect': 'is a side effect of',
        'interacts_with': 'interacts with',
        'linked_to': 'is linked to',
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


def limit_paths(paths, relations, scores, limit=6000, show=False):
    dict_path_score = lists_to_dict(relations, scores)

    # paths = delete_paths(paths, relations)
    if scores is not None:
        if show:
            for item in zip(relations, scores):
                print(item[0], " :", item[1])

        # path_scores = []
        # for path in paths:
        #     path_score_ = sum([score for relation, score in zip(relations, scores) if relation in path]) / len(path.split(" -> "))
        #     print(path_score_)
        #     path_scores.append(path_score_)

        # # sort paths the sum in path_scores in descending order
        # paths = sorted(zip(paths, path_scores), key=lambda x: x[1], reverse=True)

        # paths = hybrid_ranking(paths, dict_path_score, alpha=0.5)
        # paths = max_score_ranking(paths, dict_path_score)
        paths = tfidf_ranking(paths, dict_path_score)

        # remove score from the tuple
        paths = [path for path, _ in paths]

        # remove duplicates in paths
        paths = list(dict.fromkeys(paths))

        # replace relations with detailed sentences
        paths = [replace_relations(path) for path in paths]

        number_of_tokens_paths = [get_number_tokens(path) for path in paths]

        # Calculate the cumulative sum using itertools.accumulate
        cumsum_list = list(it.accumulate(number_of_tokens_paths))

        # Find the first index where the cumulative sum exceeds 110,000
        cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > limit), len(paths))

        # Trim the paths list up to the cutoff index
        paths = paths[:cutoff_index]
    else:
        paths = delete_paths(paths, relations)
        paths = paths[:150]
    return paths


def limit_links(links, relations, scores, limit=6000, show=False):
    dict_link_score = lists_to_dict(relations, scores)
    # links = delete_links(links, relations)
    if relations is not None and scores is not None:
        # for link in links:
        #     link_score_ = sum([score for relation, score in zip(relations, scores) if relation in link])
        #     link_scores.append(link_score_)
        
        # # sort links by the sum in link_scores in descending order
        # links = sorted(zip(links, link_scores), key=lambda x: x[1], reverse=True)

        # links = hybrid_ranking(links, dict_link_score, alpha=0.5)
        # links = max_score_ranking(links, dict_link_score)
        links = tfidf_ranking(links, dict_link_score)


        links = [link for link, _ in links]

        # remove links that are the same while keeping the order
        links = list(dict.fromkeys(links))

        # replace relations with detailed sentences
        links = [replace_relations(link) for link in links]

        number_of_tokens_links = [get_number_tokens(link) for link in links]

        # Calculate the cumulative sum using itertools.accumulate
        cumsum_list = list(it.accumulate(number_of_tokens_links))

        # Find the first index where the cumulative sum exceeds 110,000
        # cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > 110000), len(links))
        cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > limit), len(links))
        # cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > 30000), len(links))

        # Trim the paths list up to the cutoff index
        links = links[:cutoff_index]
    else:
        try:
            relations = relations.relations
        except:
            pass
        # delete the links that don't contain the relation
        links = delete_links(links, relations)
        links = links[:250]
    return links