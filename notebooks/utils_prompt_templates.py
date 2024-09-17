from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import numpy as np
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.program.openai import OpenAIPydanticProgram
from utils_graph_rag import get_embedding


load_dotenv()

llm = AzureOpenAI(
    model=os.getenv("model"),
    deployment_name="gpt4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=os.getenv("temperature"),
)

prompt_llm_alone = """
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
        output_cls=GPT4Question, prompt_template_str=prompt_llm_alone, llm=llm
    )
    result = pydantic_(question=question)
    return result.answer


prompt_template_shortest_path = """ 
You are a virtual assistant designed to assist biomedical professionals by providing precise, relevant, and well-justified answers to their inquiries. You are an expert in biomedical sciences and healthcare.

### Guidelines:
1. **Graph-Driven Responses:** Base your answers solely on the relationships and entities provided in the context enclosed by ####. Do not use outside knowledge or assumptions.
2. **Step-by-Step Analysis:** Carefully analyze the context step by step, extracting all relevant information to form a precise list of entities.
3. **Output Format:** Provide as few entities as necessary to clearly and accurately answer the question. Use the format: ['Entity1', 'Entity2', ...].

### Step-by-Step Process:
1. **Identify Key Entities and Relationships:** Start by pinpointing the entities given in the question. Search for these entities and their direct relationships within the context.
2. **Trace Relevant Paths:** Follow the shortest and most logical paths connecting these entities through the given relationships, ensuring all connections are considered. Focus on paths that lead directly to an answer.
3. **Filter Pertinent Information:** Extract only the entities and relationships that are directly relevant to the question. Ignore unrelated paths and nodes. Pay special attention to nodes or paths that are repeated, as they may signify importance.
4. **Synthesize Information:** Combine the relevant entities and relationships to form a coherent answer that directly addresses the question.
5. **Verify Consistency:** Ensure the extracted information logically supports the final answer. Double-check that no critical connections are missed and that the answer is consistent with the given context.

### Context Information:
- **Structured Information:** Entity relationships are presented as a graph where nodes represent entities (e.g., genes, proteins, diseases) and edges represent relationships (e.g., interaction, association).
- **Data in context:** The context is the shortest path between relevant entities of the question starting with one entity and ending with another entity, with each entity connected by relations.
- **Relation Definitions:**
    - **associated with:** Connection or link between two entities (e.g., gene/protein and disease).
    - **carrier:** Gene or protein involved in drug transport.
    - **contraindication:** Condition withholding medical treatment.
    - **enzyme:** Protein acting as a catalyst for biochemical reactions.
    - **expression absent:** Gene/protein not expressed in a specific tissue.
    - **expression present:** Gene/protein actively expressed in a specific tissue.
    - **indication:** Disease for which a drug is prescribed.
    - **interacts with:** Interaction between entities (e.g., genes, proteins).
    - **linked to:** Connection or correlation between entities.
    - **off-label use:** Prescription of a drug for unapproved purposes.
    - **parent-child:** Hierarchical relationship between entities.
    - **phenotype absent:** Absence of a phenotype in an entity.
    - **phenotype present:** Presence of a phenotype in an entity.
    - **ppi (protein-protein interaction):** Physical interaction between proteins.
    - **side effect:** Unintended adverse effect of a drug.
    - **synergistic interaction:** Interaction enhancing drug effects.
    - **target:** Gene or protein interacting with a drug.
    - **transporter:** Protein aiding substance movement across cell membranes.

**Question:**
{question}

**Context:**
####
{context}
####

### Important Instructions:
- **Key Focus:** Pay particular attention to the direct relationships and repeated nodes, as they often indicate the most relevant connections.
- **Precision:** Avoid extraneous information. Only include entities directly answering the question.
- **Thorough Verification:** Ensure that all extracted information supports the final answer and aligns with the given context.
"""


prompt_template_link = """
You are a virtual assistant designed to assist biomedical professionals by providing precise, relevant, and well-justified answers to their inquiries. You are an expert in biomedical sciences and healthcare.

### Guidelines:
1. **Graph-Driven Responses:** Base your answers solely on the relationships and entities provided in the context enclosed by ####. Do not use outside knowledge or assumptions.
2. **Step-by-Step Analysis:** Carefully analyze the context step by step, extracting all relevant information to form a precise list of entities.
3. **Use of Relationships:** Pay special attention to the relationships (e.g., contraindication, indication, parent-child, phenotype present) between entities to derive the answer.
4. **Output Format:** Provide as few entities as necessary to clearly and accurately answer the question. Use the format: ['Entity1', 'Entity2', ...].

### Graph Context Information:
- **Structured Information:** The context is structured as a graph where:
   - Nodes represent biomedical entities (e.g., genes, proteins, diseases).
   - Edges represent relationships (e.g., association, interaction) between the entities.
   - The connection between nodes and edges is illustrated using a triplet format, which explicitly defines the relationship between a subject (node), predicate (relationship), and object (node).
- **Data in context:** You will be provided with links between relevant entities of a given question. Each link or connection between entities is described using the triplet format, where:
   - The first element of the triplet is the subject node (the starting entity).
   - The second element is the relationship (the edge connecting the entities).
   - The third element is the object node (the ending entity).
- **Relation Definitions:**
   - **associated with:** A connection or link between two entities (e.g., gene/protein and disease).
   - **carrier:** A gene or protein involved in drug transport.
   - **contraindication:** A condition in which treatment with a drug is inadvisable.
   - **enzyme:** A protein acting as a catalyst for biochemical reactions.
   - **expression absent:** The gene/protein is not expressed in a specific tissue.
   - **expression present:** The gene/protein is actively expressed in a specific tissue.
   - **indication:** A disease or condition for which a drug is prescribed.
   - **interacts with:** Physical or functional interaction between entities (e.g., genes, proteins).
   - **linked to:** A general connection or correlation between entities.
   - **off-label use:** Use of a drug for an unapproved purpose.
   - **parent-child:** A hierarchical relationship between entities.
   - **phenotype absent:** The absence of a particular phenotype in an entity.
   - **phenotype present:** The presence of a particular phenotype in an entity.
   - **ppi (protein-protein interaction):** A direct physical interaction between proteins.
   - **side effect:** An unintended adverse effect of a drug.
   - **synergistic interaction:** An interaction between drugs that enhances their effects.
   - **target:** A gene or protein that a drug interacts with.
   - **transporter:** A protein involved in the movement of substances across membranes.

As a biomedical expert, you are expected to be thorough and use as much relevant information as possible from the context. Work step by step to gather all pertinent information and compile it into an accurate list of entities as your final answer.

Question:
{question}

Context:
####
{context}
####
"""


def format_prompt_final_question(question: str, context: str, use_path: bool) -> str:
    """
    This function is not used directly by the LLM but is used to generate the prompt and save it
    """
    if use_path:
        return prompt_template_link.format(question=question, context=context)
    else:
        return prompt_template_shortest_path.format(question=question, context=context)


class FinalQuestion(BaseModel):
    """The final question to be asked to the language model"""
    answer: Optional[List[str]]


def question_pydantic(question: str, context: str, use_path: bool, llm=llm) -> List[str]:
    if use_path:
        prompt_ = prompt_template_shortest_path
    else:
        prompt_ = prompt_template_link
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=FinalQuestion, prompt_template_str=prompt_, llm=llm
    )
    result = pydantic_(question=question, context=context)
    return result.answer


class Entities(BaseModel):
    """List of named entities in the text such as names of medications, diseases, etc."""
    entities: Optional[List[str]]


prompt_template_entities = """
Extract all medical-related named entities from the following text. This includes:

- **Diseases/Medical Conditions**: Names of diseases, syndromes, disorders, and medical conditions (e.g., diabetes mellitus, retinopathy, heart failure, ...).
- **Medications/Drugs**: Any pharmaceutical treatments, medications, or drugs (e.g., insulin, metformin, ...).
- **Medical Procedures/Interventions**: Names of surgeries, treatments, or interventions (e.g., bypass surgery, chemotherapy, ...).
- **Anatomical Terms**: Body parts, organs, tissues, or other anatomical structures (e.g., kidney, retina, ...).
- **Medical Associations or Pathways**: Relationships or pathways associated with biological processes or conditions (e.g., protein, gene regulation, inflammation).
- **Complications**: Any associated conditions or secondary complications resulting from a disease or condition (e.g., diabetic nephropathy, neuropathy).
- **Other Medical Terms**: Any other relevant medical terminologies (e.g., pathway, gene, biomarkers, receptors, ... ).

For example, in the sentence:
- "The protein encoded by **HIF3A** is associated with **negative regulation** of what?"
  The named entities are: **HIF3A**, **protein**, **negative regulation**.

In the question:
- "Which **pharmacological treatments** are recommended for managing **diabetes mellitus** and its associated complications such as **diabetic nephropathy** and **retinopathy**?"
  The named entities are: **pharmacological treatments**, **diabetes mellitus**, **complications**, **diabetic nephropathy**, **retinopathy**.

Now, extract the named entities from the following text:
{text}
"""


def entity_extraction(text: str, llm=llm) -> List[str]:
    pydantic_ = OpenAIPydanticProgram.from_defaults(
        output_cls=Entities, prompt_template_str=prompt_template_entities, llm=llm
    )
    result = pydantic_(text=text)
    return result.entities
