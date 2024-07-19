from typing import List, Optional
from pydantic import BaseModel

from llama_index.program.openai import OpenAIPydanticProgram


class Entities(BaseModel):
    """List of named entities in the text such as names of people, organizations, concepts, and locations"""
    names: Optional[List[str]]


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
    entity_extraction = OpenAIPydanticProgram.from_defaults(
        output_cls=Entities, prompt_template_str=prompt_template_entities
    )
    result = entity_extraction(text=text)
    return result.names


def relations_extraction(text: str) -> List[str]:
    possible_relations = [
        'associated with',
        'carrier',
        'contraindication',
        'enzyme',
        'expression absent',
        'expression present',
        'indication',
        'interacts with',
        'linked to',
        'off-label use',
        'parent-child',
        'phenotype absent',
        'phenotype present',
        'ppi',
        'side effect',
        'synergistic interaction',
        'target',
        'transporter'
    ]

    relevant_relations = []
    for relation in possible_relations:
        if relation in text.lower():
            relevant_relations.append(relation)
    return relevant_relations


# class Relations(BaseModel):
#     """List of relevant relations in the text"""
#     relations: Optional[List[str]]

# # Define the prompt template
# prompt_template_relations = """
# Extract relevant medical-related relations from the following text.
# The relations must be among the given list: 'associated with', 'carrier', 
# 'contraindication', 'enzyme', 'expression absent', 'expression present', 'indication', 
# 'interacts with', 'linked to', 'off-label use', 'parent-child', 'phenotype absent', 
# 'phenotype present', 'ppi', 'side effect', 'synergistic interaction', 'target', 
# 'transporter'.

# For example, in the text:
# The drug interacts with the enzyme and has a known side effect.
# The relations are: interacts with, side effect

# Do the same for the following text:
# {text}
# """

# # Define the OpenAIPydanticProgram for relation extraction
# entity_extraction = OpenAIPydanticProgram.from_defaults(
#     output_cls=Relations,
#     prompt_template_str=prompt_template_relations
# )

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

#     print("Relations:", result)
    
#     # Ensure to only keep relations that are in the possible_relations list
#     if result.relations:
#         relevant_relations = [relation for relation in result.relations if relation in possible_relations]
#         relevant_relations = [relation.replace(" ", "_") for relation in relevant_relations]
#         result = relevant_relations
#     else:
#         result = []
#     return result
