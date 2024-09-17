from utils_neo4j import Neo4jApp
import os

from typing import List, Optional
from dotenv import load_dotenv

from tqdm import tqdm
from utils_prompt_templates import entity_extraction, question_pydantic, format_prompt_final_question
from utils_graph_rag import get_combinations, format_paths, format_links, limit_paths, check_kg_ids, limit_ids, limit_links, find_similar_entity, get_embedding, find_matching_nlp_entity, find_matching_entity_name, find_hybrid_entity, find_full_text_entity
from llama_index.llms.azure_openai import AzureOpenAI
from utils_openai import get_embedding_score
from utils_neo4j import Neo4jApp


load_dotenv()

limit_tokens = int(os.getenv("limit_tokens"))

HIGH_SIMILARITY_THRESHOLD = 0.92

# Note that we need to have some relations with an underscore --> check prepare_data_KG why
list_relations = [
    'ppi', 'carrier', 'enzyme', 'target', 'transporter',
    'contraindication', 'indication', 'off-label_use',
    'synergistic_interaction', 'associated_with', 'parent-child',
    'phenotype_absent', 'phenotype_present', 'side_effect',
    'interacts_with', 'linked_to', 'expression_present',
    'expression_absent'
]


list_relations_detailed = [
    "ppi (protein-protein interaction): This denotes a physical interaction between two proteins, often influencing cellular functions or processes, which may impact diseases or drug responses.",
    "carrier: This refers to a gene or protein responsible for transporting a drug or other substances within the body, affecting drug efficacy, distribution, and elimination.",
    "enzyme: This denotes a protein that catalyzes biochemical reactions, potentially modifying drugs by metabolizing them, and influencing their therapeutic effects or side effects.",
    "target: This indicates a specific gene or protein that a drug is designed to interact with, either to inhibit, activate, or modulate its activity, influencing the drug's therapeutic goal.",
    "transporter: This refers to a protein that helps move drugs or other substances across cell membranes, influencing absorption, distribution, and elimination.",
    "contraindication: This signifies a condition or factor that makes a particular treatment or drug unsafe for a patient due to the risk of harmful effects.",
    "indication: This refers to the medical condition or disease for which a drug is prescribed or recommended, based on its therapeutic efficacy.",
    "off-label use: This refers to the prescription of a drug for a condition or purpose not officially approved by regulatory authorities, often based on emerging research or clinical judgment.",
    "synergistic interaction: This describes a situation where two drugs work together to enhance or amplify each other's effects, leading to greater efficacy or increased potential risks.",
    "associated with: This relation indicates a correlation or link between two entities, such as a gene/protein and a disease, or a gene/protein and an effect or phenotype.",
    "parent-child: This describes a hierarchical relationship between two entities, such as a broader category of disease and its more specific subtypes or manifestations.",
    "phenotype absent: This indicates that a specific observable characteristic (phenotype) is not present in a given disease, condition, or biological entity.",
    "phenotype present: This signifies that a particular phenotype or observable characteristic is present in a given disease, condition, or biological entity.",
    "side effect: This refers to an unintended, often adverse, effect of a drug, which can range from mild discomfort to severe complications.",
    "interacts with: This denotes an interaction or influence between two entities, such as genes, proteins, cellular components, or biological processes, affecting their function or behavior.",
    "linked to: Similar to 'associated with,' this indicates a connection or correlation between two entities, such as a gene and a disease, based on observed data or studies.",
    "expression present: This signifies that a gene or protein is actively being produced or expressed in a particular tissue or anatomy, which may influence biological processes.",
    "expression absent: This indicates that a gene or protein is not being produced or expressed in a specific tissue or anatomy, potentially affecting related biological functions or diseases.",
]


list_relations_emb = [get_embedding(relation) for relation in list_relations_detailed]

# Note that we need to have some node types with an underscore --> check prepare_data_KG why
list_node_types = [
    'gene/protein', 'drug', 'effect/phenotype', 'disease',
    'biological_process', 'molecular_function', 'cellular_component',
    'exposure', 'pathway', 'anatomy'
]

list_node_types_detailed = [
    "gene/protein: A sequence of DNA that codes for a specific protein or performs regulatory functions. Proteins are molecular machines that perform diverse functions in the body, including as enzymes, transporters, and receptors. Genes/proteins can be associated with diseases, phenotypes, or drug interactions.",
    "drug: A chemical substance used in the diagnosis, treatment, or prevention of disease. Drugs can interact with various biological entities, such as genes, proteins, and pathways, to produce therapeutic effects or cause side effects.",
    "effect/phenotype: Observable characteristics or traits of an organism, which can be the result of genetic and environmental influences. In medical contexts, phenotypes often refer to symptoms or clinical features of diseases, which can be linked to underlying genetic causes or drug effects.",
    "disease: A pathological condition that affects an organism's structure or function, often resulting in characteristic clinical symptoms. Diseases can be associated with genes, proteins, and phenotypes, and can be targeted or affected by drugs.",
    "biological_process: A series of events or molecular activities carried out by cells, tissues, or organs to maintain life. These processes are often regulated by genes and proteins, and they can interact with drugs or diseases.",
    "molecular_function: The specific biochemical activity performed by a gene or protein, such as binding to other molecules, catalyzing reactions, or transporting substances. Molecular functions can influence diseases and interact with drugs or cellular components.",
    "cellular_component: The parts or structures of a cell, such as the nucleus, membrane, or mitochondria. Cellular components interact with genes, proteins, and drugs, playing key roles in health and disease.",
    "exposure: Contact with external factors such as chemicals, drugs, or environmental agents that may affect an organism's health. Exposure can modify drug effects, gene expression, or disease outcomes.",
    "pathway: A sequence of molecular events or interactions between genes, proteins, or other cellular components that leads to a specific outcome, such as cell growth or immune response. Pathways are central to understanding diseases and drug mechanisms.",
    "anatomy: The structure of the body or its parts, including organs and tissues. Genes, proteins, and drugs can be expressed or act in specific anatomical locations, affecting biological functions and disease processes.",
]

list_node_types_emb = [get_embedding(node_type) for node_type in list_node_types_detailed]


llm = AzureOpenAI(
    model=os.getenv("model"),
    deployment_name="gpt4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=os.getenv("temperature"),
)

class GraphRAG():
    def __init__(
        self,
        graph: Neo4jApp = None,
        llm = llm,
        add_similarity: bool = True,
        add_question_similarity: bool = True,
        force_path: Optional[bool] = None,
        show: bool = False
    ):
        self.graph = graph
        self.llm = llm
        self.add_similarity = add_similarity
        self.add_question_similarity = add_question_similarity
        self.force_path = force_path
        self.show = show


    def ask_question(
            self,
            question: str,
            context: str = "",
            use_path: bool = True
    ):
        """
        Prompt to be asked to the language model

        Args:
            question (str) : The question to be asked to the language model
            context (str) :The context to be provided to the language model
            llm (openai.ChatCompletion) : The language model to be used to generate the response

        Returns:
            str : The response generated by the language model
        """
        if self.show:
            prompt = format_prompt_final_question(question, context, use_path)
            print(prompt)

        try:
            response = question_pydantic(question, context, use_path)
        except Exception as e:
            response = [f"Error in response: {e}"]
        return response


    def find_ids(self, entities, search_depth_node_type=None):
        """
        Find the ids of the entities in the knowledge graph

        Args:
            entities (List[str]) : The entities as english word to find the ids of

        Returns:
            List[List[str]] : The ids of the entities in the knowledge graph
        """
        kg_ids = []
        for entity in entities:
            list_ids = []
            try:
                # nlp_entity = find_matching_nlp_entity(entity, self.graph)
                nlp_entity = find_full_text_entity(entity, self.graph, top_k=2) #before 3
                list_ids.extend(nlp_entity)
            except:
                if self.show:
                    print("No nlp entities found for entity: ", entity)
            if self.add_similarity:
                try:
                    if search_depth_node_type is not None:
                        entity_emb = get_embedding_score(entity, list_node_types_emb)

                        # Get indices of the top 5 values
                        top_indices = sorted(range(len(entity_emb)), key=lambda i: entity_emb[i], reverse=True)[:search_depth_node_type]

                        # Use these indices to get the corresponding node types
                        top_node_types = [list_node_types[i] for i in top_indices]
                        similar_entities = find_similar_entity(entity, graph=self.graph, top_k=3, min_score=HIGH_SIMILARITY_THRESHOLD, node_type=top_node_types)
                    else:
                        similar_entities = find_similar_entity(entity, graph=self.graph, top_k=3, min_score=HIGH_SIMILARITY_THRESHOLD)
                    list_ids.extend(similar_entities)
                except:
                    if self.show:
                        print("No similar entities found for entity: ", entity)
            list_ids = list(set(list_ids))
            kg_ids.append(list_ids)
        return kg_ids


    def get_paths(self, node_ids, add_node_type=False):
        """
        Get the paths between the entities in the knowledge graph

        Args:
            kg_ids (List[List[str]]) : The ids of the entities in the knowledge graph
            max_distance (int) : The maximum distance to search for paths

        Returns:
            List[str] : The paths between the entities in the knowledge graph
        """
        paths_ = []
        if add_node_type:
            query_template = """
            MATCH (e1:Entity), (e2:Entity),
            p = shortestPath((e1)-[*..50]-(e2))
            WHERE elementId(e1) = $node_id1 AND elementId(e2) = $node_id2
            WITH [node IN nodes(p) | node.name + " (" + node.type + ") "] AS nodes, [rel IN relationships(p) | type(rel)] AS rels
            WITH apoc.coll.flatten(apoc.coll.zip(nodes, rels + [""])) AS elements
            RETURN apoc.text.join(elements[0..-1], " -> ") AS answer
            """
        else:
            query_template = """
            MATCH (e1:Entity), (e2:Entity),
            p = shortestPath((e1)-[*..50]-(e2))
            WHERE elementId(e1) = $node_id1 AND elementId(e2) = $node_id2
            WITH [node IN nodes(p) | node.name] AS nodes, [rel IN relationships(p) | type(rel)] AS rels
            WITH apoc.coll.flatten(apoc.coll.zip(nodes, rels + [""])) AS elements
            RETURN apoc.text.join(elements[0..-1], " -> ") AS answer
            """

        all_combinations = get_combinations(node_ids)
        for item in tqdm(all_combinations, total=len(all_combinations), disable=not self.show):
            node_id1, node_id2 = item
            params =  {
                'node_id1': node_id1,
                'node_id2': node_id2,
            }

            try:
                results = self.graph.query(query_template, params)
                paths_.extend([results[0][0]["answer"]])
            except:
                if self.show:
                    print("Error in query")
        return paths_


    def get_links(self, node_ids, add_node_type=False):
        """
        Get the links of the entities in the knowledge graph

        Args:
            entity (List[str]) : The entities in the knowledge graph

        Returns:
            List[str] : The links of the entities in the knowledge graph
        """
        list_output = []

        if add_node_type:
            query_template = """
            MATCH (e1:Entity)-[rel]->(connected)
            WHERE elementId(e1) = $node_id
            RETURN e1.name + ' ('+ e1.type + ') ' + ' -> ' + type(rel) + ' -> ' + connected.name + ' ('+ connected.type + ') ' AS answer
            """
        else:
            query_template = """
            MATCH (e1:Entity)-[rel]->(connected)
            WHERE elementId(e1) = $node_id
            RETURN e1.name + ' -> ' + type(rel) + ' -> ' + connected.name AS answer
            """

        def is_list_of_lists(lst):
            return all(isinstance(i, list) for i in lst)

        if is_list_of_lists(node_ids):
            for list_ in node_ids:
                for item in list_:
                    params =  {'node_id': item}
                    try:
                        response = self.graph.query(query_template, params)
                        list_output.extend([entry['answer'] for entry in response[0]])
                    except:
                        if self.show:
                            print("Error in response")
        else:
            for item in node_ids:
                params =  {'node_id': item}
                try:
                    response = self.graph.query(query_template, params)
                    list_output.extend([entry['answer'] for entry in response[0]])
                except:
                    if self.show:
                        print("Error in response")
        return list_output


    def query_engine(self, question: str, return_context: bool = False):
        # Getting entities
        entities = entity_extraction(question)
        if self.show:
            print("Entities: ", entities)

        cosine_similarity = get_embedding_score(question, list_relations_emb)
        relations_scores = {
            "relations": list_relations,
            "scores": cosine_similarity
        }
        new_relations_scores = {}
        for i, item in enumerate(relations_scores["relations"]):
            new_relations_scores[item] = relations_scores["scores"][i]
        
        if self.show:
            len_relations = len(relations_scores["relations"])
            for i in range(len_relations):
                print(f"{relations_scores['relations'][i]}: {relations_scores['scores'][i]}")

        list_ids = self.find_ids(entities)

        if self.show:
            print("len of before limit and question similarity: ", [len(item) for item in list_ids])
        list_ids = limit_ids(list_ids, show=self.show)

        use_path = check_kg_ids(list_ids)
        path_should_be_used = check_kg_ids(list_ids)
        force_path = self.force_path
        if force_path is not None:
            use_path = force_path

        if self.add_question_similarity:
            questions_ids = find_similar_entity(question, graph=self.graph, top_k=15, min_score=0.88)

            for item in questions_ids[:5]:
                name_ = find_matching_entity_name(item, self.graph)
                synonyms_ = find_similar_entity(name_, graph=self.graph, top_k=2, min_score=HIGH_SIMILARITY_THRESHOLD) #before 5
                questions_ids.extend(synonyms_)
            list_ids.append(questions_ids)

        # If too little ids are found and we force path, then we need to open up the list_ids
        if path_should_be_used is False and force_path is True:
            list_ids = [item for sublist in list_ids for item in sublist]
            list_ids = list(set(list_ids))
            list_ids = [[item] for item in list_ids]

        # if self.show:
        print("len of after question: ", [len(item) for item in list_ids])

        print("Using path: ", use_path)
        if use_path:
            paths_ = self.get_paths(list_ids)
            paths_ = limit_paths(paths_, new_relations_scores, limit=limit_tokens, show=self.show)
            context = format_paths(paths_)
        else:
            list_outputs = self.get_links(list_ids)
            list_outputs = limit_links(list_outputs, new_relations_scores, limit=limit_tokens, show=self.show)
            context = format_links(list_outputs)


        if return_context:
            return self.ask_question(question, context, use_path), context
        else:
            return self.ask_question(question, context, use_path)
