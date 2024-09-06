from utils_neo4j import Neo4jApp
import os

from typing import List, Optional
from dotenv import load_dotenv

from tqdm import tqdm
from utils_prompt_templates import entity_extraction, question_pydantic, format_prompt_final_question, embedding_relation_extraction_score
from utils_graph_rag import get_combinations, format_paths, format_links, limit_paths, check_kg_ids, limit_ids, limit_links, find_similar_entity, get_embedding, find_matching_nlp_entity, find_matching_entity_name
from llama_index.llms.azure_openai import AzureOpenAI
from utils_neo4j import Neo4jApp
from utils_nlp import get_number_tokens


load_dotenv()

limit_tokens = int(os.getenv("limit_tokens"))

list_relations = ['ppi', 'carrier', 'enzyme', 'target', 'transporter',
'contraindication', 'indication', 'off-label_use',
'synergistic_interaction', 'associated_with', 'parent-child',
'phenotype_absent', 'phenotype_present', 'side_effect',
'interacts_with', 'linked_to', 'expression_present',
'expression_absent'
]


list_detailed_relations = [
    'ppi (protein-protein interaction): This denotes a physical interaction between two proteins',
    'carrier: This refers to a gene or protein that transports or is involved in the movement of a drug within the body',
    'enzyme: This denotes a protein that acts as a catalyst to bring about a specific biochemical reaction, often affecting drugs by metabolizing them',
    'target: This indicates a specific gene or protein that a drug is designed to interact with or inhibit',
    'transporter: This refers to a protein that helps move substances, including drugs, across cell membranes or within the body',
    'contraindication: This signifies a condition or factor that serves as a reason to withhold a certain medical treatment due to the harm it could cause the patient',
    'indication: This refers to the condition or disease for which a drug is prescribed or recommended as a treatment',
    'off-label use: This refers to the prescription of a drug for a purpose that is not approved by the regulatory authorities',
    'synergistic interaction: This describes a scenario where two drugs interact in a way that enhances or amplifies their effects',
    'associated with: This relation indicates a connection or link between two entities, such as a gene/protein and a disease, or a gene/protein and an effect/phenotype',
    'parent-child: This describes a hierarchical relationship between two entities, such as a broader disease category and its more specific subtypes',
    'phenotype absent: This indicates that a particular phenotype or observable characteristic is not present in an entity, such as a disease',
    'phenotype present: This signifies that a particular phenotype or observable characteristic is present in an entity, such as a disease',
    'side effect: This refers to an unintended and often adverse effect of a drug',
    'interacts with: This denotes an interaction between two entities, such as genes, proteins, cellular components, or biological processes',
    'linked to: Similar to “associated with,” this indicates a connection or correlation between two entities',
    'expression present: This signifies that a gene or protein is actively expressed in a particular anatomy or tissue',
    'expression absent: This indicates that a gene or protein is not expressed in a particular anatomy or tissue',
]

list_relations_emb = [get_embedding(relation) for relation in list_detailed_relations]

HIGH_SIMILARITY_THRESHOLD = 0.92

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
        force_link: Optional[bool] = None,
        show: bool = False
    ):
        self.graph = graph
        self.llm = llm
        self.add_similarity = add_similarity
        self.add_question_similarity = add_question_similarity
        self.force_link = force_link
        self.show = show


    def ask_question(
            self,
            question: str,
            context: str = "",
            use_links: bool = False
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

        prompt = format_prompt_final_question(question, context, use_links)

        # print("question: ", question)
        # print("number of tokens: ", get_number_tokens(context))
        if self.show:
            print(prompt)

        try:
            response = question_pydantic(question, context, use_links)
        except Exception as e:
            response = [f"Error in response: {e}"]
        return response


    def find_ids(self, entities):
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
            entity = entity.replace("'", " ")
            try:
                nlp_entity = find_matching_nlp_entity(entity, self.graph)
                list_ids.extend(nlp_entity)
            except:
                if self.show:
                    print("No nlp entities found for entity: ", entity)
            if self.add_similarity:
                try:
                    similar_entities = find_similar_entity(entity, graph=self.graph, k=5, min_score=HIGH_SIMILARITY_THRESHOLD)
                    list_ids.extend(similar_entities)
                    # print("list_ids: ", list_ids)
                except:
                    if self.show:
                        print("No similar entities found for entity: ", entity)

            kg_ids.append(list_ids)
        
        # print("KG IDs: ", kg_ids)
        return kg_ids


    def get_paths(self, kg_ids, max_distance=50):
        """
        Get the paths between the entities in the knowledge graph

        Args:
            kg_ids (List[List[str]]) : The ids of the entities in the knowledge graph
            max_distance (int) : The maximum distance to search for paths

        Returns:
            List[str] : The paths between the entities in the knowledge graph
        """
        paths_ = []
        query_template = """
            MATCH (e1:Entity), (e2:Entity),
            p = shortestPath((e1)-[*..{max_distance}]-(e2))
            WHERE elementId(e1) = "{node_id1}" AND elementId(e2) = "{node_id2}"
            WITH [node IN nodes(p) | node.name] AS nodes, [rel IN relationships(p) | type(rel)] AS rels
            WITH apoc.coll.flatten(apoc.coll.zip(nodes, rels + [""])) AS elements
            RETURN apoc.text.join(elements[0..-1], " -> ") AS answer
        """
        all_combinations = get_combinations(kg_ids)

        for item in tqdm(all_combinations, total=len(all_combinations), disable=not self.show):
            node_id1, node_id2 = item
            node_id1 = node_id1.replace("'", " ")
            node_id2 = node_id2.replace("'", " ")
            query = query_template.format(node_id1=node_id1, node_id2=node_id2, max_distance=max_distance)
            try:
                results = self.graph.query(query)
                paths_.extend([results[0][0]["answer"]])
            except:
                if self.show:
                    print("Error in query")
        return paths_


    def get_links(self, entity):
        """
        Get the links of the entities in the knowledge graph

        Args:
            entity (List[str]) : The entities in the knowledge graph

        Returns:
            List[str] : The links of the entities in the knowledge graph
        """
        list_output = []
        query_template = """
        MATCH (e1:Entity)-[rel]->(connected)
        WHERE elementId(e1) = '{entity}'
        RETURN e1.name + ' -> ' + type(rel) + ' -> ' + connected.name AS answer
        """
        def is_list_of_lists(lst):
            return all(isinstance(i, list) for i in lst)

        if is_list_of_lists(entity):
            for list_ in entity:
                for item in list_:
                    query = query_template.format(entity=item)
                    response = self.graph.query(query)
                    list_output.extend([entry['answer'] for entry in response[0]])
        else:
            for item in entity:
                query = query_template.format(entity=item)
                response = self.graph.query(query)
                list_output.extend([entry['answer'] for entry in response[0]])
        return list_output


    def query_engine(self, question: str, return_context: bool = False):
        # Getting entities
        entities = entity_extraction(question)
        if self.show: 
            print("Entities: ", entities)

        cosine_similarity = embedding_relation_extraction_score(question, list_relations_emb)
        relations = {
            "relations": list_relations,
            "scores": cosine_similarity
        }
        
        if self.show:
            print("Relations: ", relations)

        # Getting ids
        if self.show:
            print("Finding IDs")
        list_ids = self.find_ids(entities)

        print("len of before limit and question similarity: ", [len(item) for item in list_ids])
        list_ids = limit_ids(list_ids, show=self.show)

        use_links = not check_kg_ids(list_ids)
        force_link = self.force_link
        if force_link is not None:
            use_links = force_link

        if self.add_question_similarity:
            questions_ids = find_similar_entity(question, graph=self.graph, k=20)

            for item in questions_ids[:10]:
                name_ = find_matching_entity_name(item, self.graph)
                synonyms_ = find_similar_entity(name_, graph=self.graph, k=3, min_score=HIGH_SIMILARITY_THRESHOLD)
                questions_ids.extend(synonyms_)

            list_ids.append(questions_ids)

        if self.show:
            print("len of after question: ", [len(item) for item in list_ids])

        print("Using links: ", use_links)
        if use_links:
            if self.show:
                print("Getting links")
            list_outputs = self.get_links(list_ids)
            if self.show:
                print("Limiting links")
            if relations:
                list_outputs = limit_links(list_outputs, relations["relations"], relations["scores"], limit=limit_tokens, show=self.show)
            context = format_links(list_outputs)
        else:
            if self.show:
                print("Getting paths")
            paths_ = self.get_paths(list_ids)
            if relations:
                if self.show:
                    print("Limiting paths")
                paths_ = limit_paths(
                    paths_,
                    relations["relations"],
                    relations["scores"],
                    limit=limit_tokens,
                    show=self.show
                )
            context = format_paths(paths_)


        if return_context:
            return self.ask_question(question, context, use_links), context
        else:
            return self.ask_question(question, context, use_links)
