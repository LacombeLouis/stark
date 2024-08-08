from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.program.openai import OpenAIPydanticProgram


load_dotenv()

# llm = OpenAI(model=os.getenv("model"), temperature=os.getenv("temperature"))

load_dotenv()


# client = OpenAI()
    
llm = AzureOpenAI(
    model=os.getenv("model"),
    deployment_name="gpt4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=os.getenv("temperature"),
)

prompt_new_final_question_ = prompt_new_final_question_ = """
You are a virtual assistant specialized in aiding biomedical professionals by providing comprehensive, relevant, highly detailed, and well-justified answers to their questions. As an expert in biomedical sciences and healthcare, you will always respond with a professional tone and language.

Think step-by-step and thoroughly analyze all the context before answering.

Provide only the name of the entity or entities that answer the question.

Examples:
----------

Question:
Which medications, designed to target genes or proteins associated with the transport of long-chain fatty acids, enhance the duration of drug presence on the ocular surface?

Answer:
Carboxymethylcellulose

Question:
Which proteins are involved in the regulation of apoptosis in human cells?

Answer:
Bcl-2, Caspase-3, p53

Question:
What are the primary neurotransmitters involved in the regulation of mood?

Answer:
Serotonin, Dopamine, Norepinephrine

Question:
Which viruses are known to cause hemorrhagic fever?

Answer:
Ebola virus, Marburg virus, Dengue virus

Question: {question}
"""

class GPT4Question(BaseModel):
    """The final question to be asked to the language model"""
    answer: Optional[List[str]]


def question_pydantic_direct(question: str, llm=llm) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=GPT4Question, prompt_template_str=prompt_new_final_question_, llm=llm
    )
    result = pydantic_(question=question)
    return result.answer


prompt_final_path_question_ = """
You are a researcher working on a project to answer questions about a given context.
You answer the questions by listing maximum the 10 most relevant entities that could answer the question.
It's key to provide the most relevant entities to the question only given the context.
The context provided is given as the links between entities in a graph, with the entities being the nodes and the links being the edges.

These are the definitions of the relations:
- associated with: This relation indicates a connection or link between two entities, such as a gene/protein and a disease, or a gene/protein and an effect/phenotype.
- carrier: This refers to a gene or protein that transports or is involved in the movement of a drug within the body.
- contraindication: This signifies a condition or factor that serves as a reason to withhold a certain medical treatment due to the harm it could cause the patient.
- enzyme: This denotes a protein that acts as a catalyst to bring about a specific biochemical reaction, often affecting drugs by metabolizing them.
- expression absent: This indicates that a gene or protein is not expressed in a particular anatomy or tissue.
- expression present: This signifies that a gene or protein is actively expressed in a particular anatomy or tissue.
- indication: This refers to the condition or disease for which a drug is prescribed or recommended as a treatment.
- interacts with: This denotes an interaction between two entities, such as genes, proteins, cellular components, or biological processes.
- linked to: Similar to “associated with,” this indicates a connection or correlation between two entities.
- off-label use: This refers to the prescription of a drug for a purpose that is not approved by the regulatory authorities.
- parent-child: This describes a hierarchical relationship between two entities, such as a broader disease category and its more specific subtypes.
- phenotype absent: This indicates that a particular phenotype or observable characteristic is not present in an entity, such as a disease.
- phenotype present: This signifies that a particular phenotype or observable characteristic is present in an entity, such as a disease.
- ppi (protein-protein interaction): This denotes a physical interaction between two proteins.
- side effect: This refers to an unintended and often adverse effect of a drug.
- synergistic interaction: This describes a scenario where two drugs interact in a way that enhances or amplifies their effects.
- target: This indicates a specific gene or protein that a drug is designed to interact with or inhibit.
- transporter: This refers to a protein that helps move substances, including drugs, across cell membranes or within the body.

For example, in the question:
Question: What is the side effect of drug X compared to Y?
Context: X -> drug -> Z -> side effect -> Y
The answer is Z

Note: the answer cannot be the start or end node of the path.

Question: {question}
Context: \n{context}
"""


prompt_final_link_question_ = """
You are a researcher working on a project to answer questions about a given context.
You answer the questions by listing the entities that could answer the question.
It's key to provide the most relevant entities to the question only given the context.
The context provided is given as the links between entities in a graph, with the entities being the nodes and the links being the edges.

These are the definitions of the relations:
- associated with: This relation indicates a connection or link between two entities, such as a gene/protein and a disease, or a gene/protein and an effect/phenotype.
- carrier: This refers to a gene or protein that transports or is involved in the movement of a drug within the body.
- contraindication: This signifies a condition or factor that serves as a reason to withhold a certain medical treatment due to the harm it could cause the patient.
- enzyme: This denotes a protein that acts as a catalyst to bring about a specific biochemical reaction, often affecting drugs by metabolizing them.
- expression absent: This indicates that a gene or protein is not expressed in a particular anatomy or tissue.
- expression present: This signifies that a gene or protein is actively expressed in a particular anatomy or tissue.
- indication: This refers to the condition or disease for which a drug is prescribed or recommended as a treatment.
- interacts with: This denotes an interaction between two entities, such as genes, proteins, cellular components, or biological processes.
- linked to: Similar to “associated with,” this indicates a connection or correlation between two entities.
- off-label use: This refers to the prescription of a drug for a purpose that is not approved by the regulatory authorities.
- parent-child: This describes a hierarchical relationship between two entities, such as a broader disease category and its more specific subtypes.
- phenotype absent: This indicates that a particular phenotype or observable characteristic is not present in an entity, such as a disease.
- phenotype present: This signifies that a particular phenotype or observable characteristic is present in an entity, such as a disease.
- ppi (protein-protein interaction): This denotes a physical interaction between two proteins.
- side effect: This refers to an unintended and often adverse effect of a drug.
- synergistic interaction: This describes a scenario where two drugs interact in a way that enhances or amplifies their effects.
- target: This indicates a specific gene or protein that a drug is designed to interact with or inhibit.
- transporter: This refers to a protein that helps move substances, including drugs, across cell membranes or within the body.

Note: the answer cannot be the starting node.

Question: {question}
Context: \n{context}
"""


def format_prompt_final_question(question: str, context: str, use_links: bool) -> str:
    if use_links:
        return prompt_final_link_question_.format(question=question, context=context)
    else:
        return prompt_final_path_question_.format(question=question, context=context)


class FinalQuestion(BaseModel):
    """The final question to be asked to the language model"""
    answer: Optional[List[str]]


def question_pydantic(question: str, context: str, use_links: bool, llm=llm) -> List[str]:
    if use_links:
        prompt_ = prompt_final_link_question_
    else:
        prompt_ = prompt_final_path_question_
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=FinalQuestion, prompt_template_str=prompt_, llm=llm
    )
    result = pydantic_(question=question, context=context)
    return result.answer


prompt_final_after_normal_question_ = """
We have the question and answers from the previous step.
Give only the entities that answer the question.

Question: {question}\n
Answer: {answer}
"""

class FinalQuestionAfterNormal(BaseModel):
    """The final question to be asked to the language model"""
    answer: Optional[List[str]]


def question_after_normal_pydantic(question: str, answer: str, llm=llm) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=FinalQuestionAfterNormal, prompt_template_str=prompt_final_after_normal_question_, llm=llm
    )
    result = pydantic_(question=question, answer=answer)
    return result.answer


class Entities(BaseModel):
    """List of named entities in the text such as names of medications, diseases, etc."""
    entities: Optional[List[str]]


prompt_template_entities = """
Extract all medical-related named entities such as names of diseases, medications,
medical procedures, anatomical terms, medical conditions, medical associations, etc.

For example, in the question:
The protein encoded by HIF3A is associated with negative regulation of what?
The named entities are: HIF3A, protein, negative regulation

Which pharmacological treatments are recommended for managing diabetes mellitus and its associated complications such as diabetic nephropathy and retinopathy?
The named entities are: pharmacological treatments, diabetes mellitus, complications, diabetic nephropathy, retinopathy

Do the same for the following text:
{text}
"""

def entity_extraction(text: str, llm=llm) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=Entities, prompt_template_str=prompt_template_entities, llm=llm
    )
    result = pydantic_(text=text)
    return result.entities


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
The relations are: interacts with, side effect, target, associated with
The scores are: 0.8, 0.6, 0.3, 0.4

Do the same for the following text:
{text}
"""


def relation_extraction_score(text: str, llm=llm) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=Relations_Score,
        prompt_template_str=prompt_template_relations_score,
        llm=llm,
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


def relation_extraction(text: str, llm=llm) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=Relations,
        prompt_template_str=prompt_template_relations,
        llm=llm
    )
    result = pydantic_(text=text)
    return result.relations

