import re
import sys
import csv
import torch
import pandas as pd
import torch.utils.data
from collections import Counter,defaultdict

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class InputExample(object):
    def __init__(self, guid=None, text=None, user=None, product=None, label=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.user = user
        self.product = product


class Data(torch.utils.data.Dataset):
    # Map-style datasets: __getitem__(), __len__()
    sort_key = None
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)


class SentenceProcessor(object):
    NAME = 'SENTENCE'

    def get_sentences(self):
        raise NotImplementedError

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            # % string formatting method (guid: 'train-0')
            guid = "%s-%s" % (type, i)
            # text = [sentence for sentence in split_sents(line[2])]
            # text = [[sentence] for sentence in generate_sents(line[2])]
            text = clean_document(line[2])
            # text = line[2]
            examples.append(
                InputExample(guid=guid, user=line[0], product=line[1], text=text, label=int(line[3]) - 1))
        return examples

    def _read_file(self, dataset):
        pd_reader = pd.read_csv(dataset, header=None, skiprows=0, encoding="utf-8", sep='\t\t', engine='python')
        documents = []
        for i in range(len(pd_reader[0])):
            # if i == 100:
            #     break
            # [ user, product, review, label]
            document = list([pd_reader[0][i], pd_reader[1][i], pd_reader[3][i], pd_reader[2][i]])
            documents.append(document)
        return documents

    def _create_sentences(self, *datasets):
        sentences = []
        for dataset in datasets:
            for id, document in enumerate(dataset):
                user = document[0]
                product = document[1]
                review = document[2]
                label = int(document[3]) - 1
                sentences.extend([InputExample(user=user, product=product, text=sentence, label=label) for
                                  sentence in generate_sents(clean_document(review))])
                # for s in generate_sents(clean_document(review)):
                #     f = open("temp.txt", 'a')
                #     f.write(s+'\n')
        return sentences

    def _creat_sent_doc(self, *datasets):
        import time
        documents = []
        for dataset in datasets:
            for id, document in enumerate(dataset):
                user = document[0]
                product = document[1]
                review = document[2]
                label = int(document[3]) - 1
                documents.append(InputExample(user=user, product=product, text=generate_sents(clean_document(review)), label=label))
                print(generate_sents(clean_document(review)))
                print(len(generate_sents(clean_document(review))))
                time.sleep(10)

        return documents


    def _get_attributes(self, *datasets):
        users = Counter()
        products = Counter()
        ATTR_MAP = {
            'user': int(0),
            'product': int(1)
        }
        for dataset in datasets:
            for document in dataset:
                # update new objects and counts
                users.update([document[ATTR_MAP["user"]]])
                products.update([document[ATTR_MAP["product"]]])
        # tuple is a collection which is ordered and unchangeable
        # tuple are used to store multiple items in a single variable
        return tuple([users, products])

    def _get_user_rating_history(self, dataset):
        user_rating_history = defaultdict(list)
        ATTR_MAP_2 = {
            'user': int(0),
            'label': int(3),
        }
        for document in dataset:
            usr_map = ATTR_MAP_2["user"]
            usr = document[usr_map]
            lab_map = ATTR_MAP_2["label"]
            lab = document[lab_map]
            user_rating_history[usr].append(lab)

        return tuple([user_rating_history])

class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


class UnknownUPVecCache(object):
    @classmethod
    def unk(cls, tensor):
        return tensor.uniform_(-0.25, 0.25)


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"sssss", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


# def split_sents(string):
#     string = re.sub(r"[!?]", " ", string)
#     string = re.sub(r"\.{2,}", " ", string)
#     sents = string.strip().split('.')
#     sents = [clean_string(sent) for sent in sents]
#     return filter(lambda x: len(x) > 0, sents)


def clean_document(document):
    string = re.sub(r"<sssss>", "", document)
    string = re.sub(r" n't", "n't", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\\'.`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def build_vocab(counter):
    from torchtext.vocab import Vocab
    vocab = Vocab(counter=counter, specials=[], vectors=None)
    return vocab

def generate_sents(docuemnt, max_length=230):
    if isinstance(docuemnt, list):
        docuemnt = docuemnt[0]
    string = re.sub(r"[!?]", " ", docuemnt)
    string = re.sub(r"\.{2,}", " ", string)
    sents = string.strip().split('.')
    sents = [clean_string(sent) for sent in sents]
    n_sents = []
    n_sent = []
    for sent in sents:
        n_sent.extend(sent)
        if len(n_sent) > max_length:
            n_sents.append(" ".join(n_sent))
            n_sent = []
            n_sent.extend(sent)
    n_sents.append(" ".join(n_sent))
    return n_sents

def _truncate_and_pad(tokens, max_length=510, pad_strategy="head"):
    """
    Truncates a sequence in place to the maximum length
    :param tokens:
    :param max_length:
    :param pad_strategy: "head", "tail", "both"
    :return:
    """
    total_length = len(tokens)
    if total_length > max_length:
        if pad_strategy == 'head':
            return ['[CLS]'] + tokens[:max_length] + ['[SEP]']
        if pad_strategy == 'tail':
            return ['[CLS]'] + tokens[-max_length:]+ ['[SEP]']
        if pad_strategy == 'both':
            return ['[CLS]'] + tokens[:128] + tokens[-max_length+128:] + ['[SEP]']
        return
    else:
        return ['[CLS]'] + tokens + ['[SEP]'] + ['<PAD>'] * (max_length-total_length)
