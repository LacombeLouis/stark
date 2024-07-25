import itertools as it


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
        output += f"{link}\n \n"
    return output


def delete_paths(paths, relations):
    return [path for path in paths if any(relation in path for relation in relations)]


def limit_paths(paths, relations, relation_score, limit=1000, show=False):
    if relation_score:


        relations_ = relations.relations
        scores_ = relations.scores

        paths = delete_paths(paths, relations_)

        if show:
            print("Relations: ", relations_)
            print("Scores: ", scores_)

        assert len(relations_) == len(scores_), "The number of relations and scores must be the same"

        path_scores = []
        for path in paths:
            path_score_ = sum([score for relation, score in zip(relations_, scores_) if relation in path])
            path_scores.append(path_score_)

        # sort paths the sum in path_scores in descending order
        paths = sorted(zip(paths, path_scores), key=lambda x: x[1], reverse=True)
        paths = paths[:limit]
    else:
        # delete the paths that don't contain the relation
        paths = delete_paths(paths, relations)
    return paths


def check_kg_ids(kg_ids):
    # Check if kg_ids has at least 2 elements
    is_valid_length = len(kg_ids) >= 2

    # Check if each element in kg_ids is a list with at least 1 element
    are_elements_valid = all(len(sublist) >= 1 for sublist in kg_ids)
    return is_valid_length and are_elements_valid


def limit_ids(list_ids, limit=350, show=False):
    if show:
        print("len of: ", [len(item) for item in list_ids])
    return [item for item in list_ids if len(item) < limit]