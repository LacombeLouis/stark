from tqdm import tqdm
import os
from utils_graph_rag import find_full_text_entity, find_similar_entity
from stark_qa import load_skb

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


def collect_matching_entities(dict_responses, graph, get_similar_entities=False):
    for item in tqdm(dict_responses.keys()):
        # get list of answers in the dictionary if type is dict
        answers_ = [answer_ for answer_ in dict_responses[item].keys() if type(dict_responses[item][answer_]) == dict]

        for answer_ in answers_: 
            response = dict_responses[item][answer_]["response"]
            matching_ids = []
            if response is None:
                print(f"no response for {item}")
                matching_ids = [["None"]]
            else:
                for entity in response:
                    try:
                        matched_ids = find_full_text_entity(entity, graph, top_k=1, return_ID=True)
                        matching_ids.append(matched_ids)
                    except:
                        matching_ids.append(["None"])
                dict_responses[item][answer_]["matching_ids"] = matching_ids
            if get_similar_entities:
                matching_ids_sim = []
                for entity in response:
                    try:
                        matched_ids = find_similar_entity(entity, graph, return_ID=True, top_k=1, clean_results=False)
                        matching_ids_sim.append(matched_ids)
                    except:
                        matching_ids_sim.append(["None"])
                dict_responses[item][answer_]["matching_ids_sim"] = matching_ids_sim
    return dict_responses
