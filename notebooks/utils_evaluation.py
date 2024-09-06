from tqdm import tqdm
import os
from utils_graph_rag import find_similar_entity, find_matching_nlp_entity
from stark_qa import load_qa, load_skb

from dotenv import load_dotenv

load_dotenv()

dataset_name = os.getenv("dataset_name")


skb = load_skb(dataset_name, download_processed=False, root='../')

def collect_responses(qa_dataset, graph_rag, dict_=None):
    if dict_ is None:
        dict_responses = {}
    else:
        dict_responses = dict_.copy()

    for item in tqdm(qa_dataset):
        question_ = item[1]
        question_ids_ = str(item[0])
        answer_id_ = item[2]
        
        # check if question_ids_ exists in the dictionary
        if question_ids_ not in dict_responses:
            dict_responses[question_ids_] = {}
            dict_responses[question_ids_]["question_"] = question_
            dict_responses[question_ids_]["answer_id_"] = answer_id_
            dict_responses[question_ids_]["answer_name"] = [skb.__getitem__(int(item)).name for item in answer_id_]

        answer_count = 0
        name = f"answer_{answer_count}"
        # check if name exists in the dictionary
        while name in dict_responses[question_ids_].keys():
            answer_count += 1
            name = f"answer_{answer_count}"

        response, context = graph_rag.query_engine(question_, return_context=True)

        dict_responses[question_ids_][name] = {
            "response": response,
            "context": context
        }
    return dict_responses

def collect_matching_entities(dict_responses, graph):
    for item in tqdm(dict_responses.keys()):
        # get list of answers in the dictionary if type is dict
        answers_ = [answer_ for answer_ in dict_responses[item].keys() if type(dict_responses[item][answer_]) == dict]

        for answer_ in answers_: 
            response = dict_responses[item][answer_]["response"]
            matching_ids = []
            # matching_ids_sim = []
            try:
                for entity in response:
                    matching_ids.append(find_matching_nlp_entity(entity, graph, return_ID=True))
            except:
                matching_ids = [["None"]]
            dict_responses[item][answer_]["matching_ids"] = matching_ids
            # try:
            #     for entity in response:
            #         matching_ids_sim.append(find_similar_entity(entity, graph, return_ID=True, k=1))
            # except:
            #     matching_ids_sim = [["None"]]
            # dict_responses[item][answer_]["matching_ids_sim"] = matching_ids_sim
    return dict_responses
