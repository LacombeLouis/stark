import itertools as it
from tqdm import tqdm

from utils_nlp import get_number_tokens
from utils_openai import get_embedding
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_entity(entity, graph, k=5, return_ID=False):
    emb_ = get_embedding(entity)

    entity = entity.replace("'", " ")
    word = f"'{entity}'"
    query_template = """
    CALL db.index.vector.queryNodes('entityEmbeddings', 100, {emb_})
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC LIMIT 100
    """

    query = query_template.format(emb_=emb_, k=k)
    # print(query)
    results = graph.query(query)

    # for item in results[0]:
    #     print(item)

    if return_ID:
        return [int(item["node"]["id"]) for item in results[0]][:k]
    else:
        return [item[0].__dict__['_element_id'] for item in results[0]][:k]
    

def find_matching_nlp_entity(entity, graph):
    entity = entity.replace("'", " ")
    word = f"'{entity}'"
    query_template = "MATCH (e) WHERE toLower(e.name) = toLower({word}) RETURN e"
    query = query_template.format(word=word)
    response = graph.query(query)
    return [int(item['e']['id']) for item in response[0]]


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
        output += f"{path}\n \n"
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

def limit_paths(paths, relations, scores, limit=250, show=False):
    # paths = delete_paths(paths, relations)
    if scores is not None:
        if show:
            print("Relations: ", relations)
            print("Scores: ", scores)

        path_scores = []
        for path in paths:
            path_score_ = sum([score for relation, score in zip(relations, scores) if relation in path]) / len(path.split(" -> "))
            if not isinstance(path_score_, float):
                path_scores.append(0)
            else:
                path_scores.append(path_score_)

        # sort paths the sum in path_scores in descending order
        paths = sorted(zip(paths, path_scores), key=lambda x: x[1], reverse=True)

        # remove score from the tuple
        paths = [path for path, _ in paths]

        number_of_tokens_paths = [get_number_tokens(path) for path in paths]

        # Calculate the cumulative sum using itertools.accumulate
        cumsum_list = list(it.accumulate(number_of_tokens_paths))

        # Find the first index where the cumulative sum exceeds 110,000
        # cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > 110000), len(paths))
        cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > 60000), len(paths))
        # cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > 30000), len(paths))

        # Trim the paths list up to the cutoff index
        paths = paths[:cutoff_index]
    else:
        paths = delete_paths(paths, relations)
        paths = paths[:150]
    return paths


def check_kg_ids(kg_ids):
    # Check if kg_ids has at least 2 elements
    is_valid_length = len(kg_ids) >= 2

    # Check if each element in kg_ids is a list with at least 1 element
    are_elements_valid = any(len(sublist) >= 1 for sublist in kg_ids)

    can_use_path = is_valid_length and are_elements_valid
    return can_use_path


def limit_ids(list_ids, limit=100, show=False):
    if show:
        print("len of: ", [len(item) for item in list_ids])
    return [item for item in list_ids if len(item) < limit]


def delete_links(links, relations):
    return [link for link in links if any(relation in link for relation in relations)]


def limit_links(links, relations, scores, limit=500):
    # links = delete_links(links, relations)
    if relations is not None and scores is not None:
        link_scores = []
        for link in links:
            link_score_ = sum([score for relation, score in zip(relations, scores) if relation in link])
            if not isinstance(link_score_, float):
                link_scores.append(0)
            else:
                link_scores.append(link_score_)
        
        # sort links by the sum in link_scores in descending order
        links = sorted(zip(links, link_scores), key=lambda x: x[1], reverse=True)

        links = [link for link, _ in links]

        number_of_tokens_links = [get_number_tokens(link) for link in links]

        # Calculate the cumulative sum using itertools.accumulate
        cumsum_list = list(it.accumulate(number_of_tokens_links))

        # Find the first index where the cumulative sum exceeds 110,000
        # cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > 110000), len(links))
        cutoff_index = next((i for i, cumsum in enumerate(cumsum_list) if cumsum > 60000), len(links))
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
        links = links[:limit]
    return links