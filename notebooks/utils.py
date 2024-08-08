import pickle

def save_dict_to_file(dictionary, filename):
    """
    Save a dictionary to a file using pickle.

    Parameters:
    dictionary (dict): The dictionary to save.
    filename (str): The path to the file where the dictionary should be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dictionary saved to {filename}")


def load_dict_from_file(filename):
    """
    Load a dictionary from a file using pickle.

    Parameters:
    filename (str): The path to the file from which to load the dictionary.

    Returns:
    dict: The loaded dictionary.
    """
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    print(f"Dictionary loaded from {filename}")
    return dictionary

def flatten(xss):
    return [x for xs in xss for x in xs]


def all_int_in_set(set_):
    return set([int(x) for x in set_])

def change_qa_dataset_to_tuple(qa_dataset):
    new_qa_dataset = []
    for item in qa_dataset:
        new_qa_dataset.append([item[1], item[0], item[2]])
    return new_qa_dataset

