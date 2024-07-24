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
    print(len(data))
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


def limit_paths(paths, relations, limit=1000):
    # delete the paths that don't contain the relation
    paths = [path for path in paths if any(relation in path for relation in relations)]

    # order the path by the number of relations in the path
    paths = sorted(paths, key=lambda x: len([relation for relation in relations if relation in x]), reverse=True)
    paths = paths[:limit]
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