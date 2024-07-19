from utils_neo4j import Neo4jApp

from typing import List, Optional

from tqdm import tqdm
from utils_prompt_templates import entity_extraction, relations_extraction

import itertools as it

def ask_question(question: str, context: str = "", llm=None):
    # Combine context and question into a single prompt
    prompt = f"""
    You are a researcher working on a project to answer questions about a given context.
    You answer the questions by listing the entities that could answer the question.
    It's key to provide the most relevant entities to the question only given the context.
    The context provided is given as the links between entities in a graph, with the entities being the nodes and the links being the edges.

    Question: {question}
    Context: {context}\n
    """
    # Generate the response using a method from the OpenAI class
    response = llm.complete(prompt)

    return response.text


def filter_ids(output):
    # count the type of ids for each value of ids and then only keep the ones with the majority count
    # values = [item['e']['type'] for item in output[0]]
    # count = Counter(values)
    # most_common = count.most_common(1)[0]  # (value, count)

    # ids = [entry['e'].__dict__['_element_id'] for entry in output[0] if entry['e']['type'] == most_common[0]]
    ids = [entry['e'].__dict__['_element_id'] for entry in output[0]]
    # ids = [entry['e']['id'] for entry in output[0] if entry['e']['type'] == most_common[0]]
    return ids


def find_ids(entities, app: Neo4jApp):
    kg_ids = []
    # query_template = "MATCH (e) WHERE trim(toLower(e.name)) CONTAINS trim(toLower({word})) RETURN e"
    query_template = "MATCH (e) WHERE toLower(e.name) =~ '(?i).*\\\\b' + toLower({word}) + '\\\\b.*' RETURN e"
    # extract only entities that match what they should most likely be

    for entity in entities:
        word = f"'{entity}'"
        query = query_template.format(word=word)  
        # try:
        output = app.query(query)
        print(output)
        list_ids = filter_ids(output)
        kg_ids.append(list_ids)
        # except:
        #     print("No ID found for entity: ", entity)
    return kg_ids

def format_paths(paths, limit=1000):
    output = ""
    for path in paths:
        # if not len(path.split(" --> ")) > limit:
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


def get_paths(kg_ids, app: Neo4jApp, max_distance=50, show=True):
    paths_ = []
    print("We have more than one node")
    query_template = """
        MATCH (e1:Entity), (e2:Entity),
        p = shortestPath((e1)-[*..{max_distance}]-(e2))
        WHERE elementId(e1) = "{node_id1}" AND elementId(e2) = "{node_id2}"
        WITH [node IN nodes(p) | node.name] AS nodes, [rel IN relationships(p) | type(rel)] AS rels
        WITH apoc.coll.flatten(apoc.coll.zip(nodes, rels + [""])) AS elements
        RETURN apoc.text.join(elements[0..-1], " -> ") AS answer
    """
    # query_template = """
    #     MATCH (e1:Entity {{id: "{node_id1}"}}), (e1:Entity {{id: "{node_id2}"}}),
    #     p = shortestPath((e1)-[*..{max_distance}]-(e2))
    #     WITH [node IN nodes(p) | node.name] AS nodes, [rel IN relationships(p) | type(rel)] AS rels
    #     WITH apoc.coll.flatten(apoc.coll.zip(nodes, rels + [""])) AS elements
    #     RETURN apoc.text.join(elements[0..-1], " --> ") AS answer
    # """
    print(type(kg_ids))
    for item in kg_ids:
        print(item)
        print(type(item))
    all_combinations = get_combinations(kg_ids)

    for item in tqdm(all_combinations, total=len(all_combinations)):
        node_id1, node_id2 = item
        query = query_template.format(node_id1=node_id1, node_id2=node_id2, max_distance=max_distance)
        try:
            results = app.query(query)
            paths_.extend([results[0][0]["answer"]])
        except:
            if show:
                print("Error in query")
    return paths_



def format_links(links):
    output = ""
    for link in links:
        output += f"{link}\n \n"
    return output


def get_links(entity, app: Neo4jApp):
    list_output = []
    print(entity)
    query_template = """
    MATCH (e1:Entity)-[rel]->(connected)
    WHERE elementId(e1) = '{entity}'
    RETURN e1.name + ' -> ' + type(rel) + ' -> ' + connected.name AS answer
    """
    for item in entity:
        query = query_template.format(entity=item)
        response = app.query(query)
        list_output.extend([entry['answer'] for entry in response[0]])
    return list_output


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


def query_engine(question: str, app: Neo4jApp, llm=None):
    # getting entities
    entities = entity_extraction(question)
    print("Entities: ", entities)

    # getting relations
    relations = relations_extraction(question)

    # getting ids
    kg_ids = find_ids(entities, app)

    kg_ids = [list(set(ids)) for ids in kg_ids]
    
    print("KG IDs: ", kg_ids)
    print("length of KG IDs: ", len(kg_ids))
    print("len of: ", [len(item) for item in kg_ids])

    if len(kg_ids)>1:
        kg_ids = [item for item in kg_ids if len(item) < 350]

        print("len of: ", [len(item) for item in kg_ids])
        paths_ = get_paths(kg_ids, app)
        print("Paths before removing: ", len(paths_))
        if relations:
            paths_ = limit_paths(paths_, relations, limit=350)
        print("Paths after removing: ", len(paths_))
        context = format_paths(paths_)
    
    if not check_kg_ids(kg_ids):
        list_outputs = get_links(kg_ids[0], app)
        context = format_links(list_outputs)

    print("Question: ", question)
    print("Context: ", context)
    return ask_question(question, context, llm)
