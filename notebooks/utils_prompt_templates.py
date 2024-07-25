from typing import List, Optional
from pydantic import BaseModel

from llama_index.program.openai import OpenAIPydanticProgram


prompt_final_question_ = """
You are a researcher working on a project to answer questions about a given context.
You answer the questions by listing the entities that could answer the question.
It's key to provide the most relevant entities to the question only given the context.
The context provided is given as the links between entities in a graph, with the entities being the nodes and the links being the edges.

For example, in the question:
Question: What is the side effect of drug X?
Context: X -> side effect -> Y
The answer is Y

Question: {question}
Context: \n{context}
"""


def format_prompt_final_question(question: str, context: str) -> str:
    return prompt_final_question_.format(question=question, context=context)


class FinalQuestion(BaseModel):
    """The final question to be asked to the language model"""
    answer: Optional[List[str]]


def question_pydantic(question: str, context: str) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=FinalQuestion, prompt_template_str=prompt_final_question_
    )
    result = pydantic_(question=question, context=context)
    return result.answer


class Entities(BaseModel):
    """List of named entities in the text such as names of people, organizations, concepts, and locations"""
    entities: Optional[List[str]]


prompt_template_entities = """
Extract all medical-related named entities such as names of diseases, medications,
medical procedures, anatomical terms, healthcare organizations, and healthcare professionals.

For example, in the question:
The protein encoded by HIF3A is associated with negative regulation of what?
The named entities are: HIF3A, protein, negative regulation

Do the same for the following text:
{text}
"""

def entity_extraction(text: str) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=Entities, prompt_template_str=prompt_template_entities
    )
    result = pydantic_(text=text)
    return result.entities


# def relation_extraction(text: str) -> List[str]:
#     possible_relations = [
#         'associated with',
#         'carrier',
#         'contraindication',
#         'enzyme',
#         'expression absent',
#         'expression present',
#         'indication',
#         'interacts with',
#         'linked to',
#         'off-label use',
#         'parent-child',
#         'phenotype absent',
#         'phenotype present',
#         'ppi',
#         'side effect',
#         'synergistic interaction',
#         'target',
#         'transporter'
#     ]

#     relevant_relations = []
#     for relation in possible_relations:
#         if relation in text.lower():
#             relevant_relations.append(relation)
#     return relevant_relations


class Relations_Score(BaseModel):
    """List of relevant relations in the text"""
    relations: Optional[List[str]]
    scores: Optional[List[float]]

# Define the prompt template
prompt_template_relations_score = """
Extract relevant medical-related relations from the following text.
The relations must be among the given list: 'associated with', 'carrier', 
'contraindication', 'enzyme', 'expression absent', 'expression present', 'indication', 
'interacts with', 'linked to', 'off-label use', 'parent-child', 'phenotype absent', 
'phenotype present', 'ppi', 'side effect', 'synergistic interaction', 'target', 
'transporter'.

For each relation, also provide a confidence score between 0 and 1.
The scores must be in the same order as the relations.
The higher the score, the more confident you are that the relation is correct.

For example, in the text:
The drug interacts with the enzyme and has a known side effect.
The relations are: interacts with, side effect
The scores are: 0.8, 0.6

Do the same for the following text:
{text}
"""


def relation_extraction_score(text: str) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=Relations_Score,
        prompt_template_str=prompt_template_relations_score
    )
    result = pydantic_(text=text)
    return result


class Relations(BaseModel):
    """List of relevant relations in the text"""
    relations: Optional[List[str]]


# Define the prompt template
prompt_template_relations = """
Extract relevant medical-related relations from the following text.
The relations must be among the given list: 'associated with', 'carrier', 
'contraindication', 'enzyme', 'expression absent', 'expression present', 'indication', 
'interacts with', 'linked to', 'off-label use', 'parent-child', 'phenotype absent', 
'phenotype present', 'ppi', 'side effect', 'synergistic interaction', 'target', 
'transporter'.

For example, in the text:
The drug interacts with the enzyme and has a known side effect.
The relations are: interacts with, side effect

Do the same for the following text:
{text}
"""


def relation_extraction(text: str) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=Relations,
        prompt_template_str=prompt_template_relations
    )
    result = pydantic_(text=text)
    return result.relations


# def extract_relevant_relations(text: str) -> List[str]:
#     possible_relations = [
#         'associated with',
#         'carrier',
#         'contraindication',
#         'enzyme',
#         'expression absent',
#         'expression present',
#         'indication',
#         'interacts with',
#         'linked to',
#         'off-label use',
#         'parent-child',
#         'phenotype absent',
#         'phenotype present',
#         'ppi',
#         'side effect',
#         'synergistic interaction',
#         'target',
#         'transporter'
#     ]
    
#     # Extract relations using the model
#     result = entity_extraction(text=text)
    
#     # Ensure to only keep relations that are in the possible_relations list
#     if result.relations:
#         relevant_relations = [relation for relation in result.relations if relation in possible_relations]
#         relevant_relations = [relation.replace(" ", "_") for relation in relevant_relations]
#         result = relevant_relations
#     else:
#         result = []
#     return result
