import logging
import pickle
import math
import json
import random

import numpy as np
import faiss as faiss
import numpy as np

from tqdm import tqdm
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.util import dot_score
from torch.utils.data import DataLoader


random.seed(42)
logging.getLogger().setLevel(logging.INFO)

# This is a clean implementation of a valilla dense retriever


class VectorIndex:
    def __init__(self, d):
        self.d = d
        self.vectors = []
        self.index = None

    def add(self, v):
        self.vectors.append(v)

    def build(self, use_gpu=False):
        self.vectors = np.array(self.vectors)

        faiss.normalize_L2(self.vectors)

        logging.info('Indexing {} vectors'.format(self.vectors.shape[0]))

        if self.vectors.shape[0] > 8000000:
            num_centroids = 8 * int(math.sqrt(math.pow(2, int(math.log(self.vectors.shape[0], 2)))))

            logging.info('Using {} centroids'.format(num_centroids))

            self.index = faiss.index_factory(self.d, "IVF{}_HNSW32,Flat".format(num_centroids))

            ngpu = faiss.get_num_gpus()
            if ngpu > 0 and use_gpu:
                logging.info('Using {} GPUs'.format(ngpu))

                index_ivf = faiss.extract_index_ivf(self.index)
                clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(self.d))
                index_ivf.clustering_index = clustering_index

            logging.info('Training index...')

            self.index.train(self.vectors)
        else:
            self.index = faiss.IndexFlatL2(self.d)
            if faiss.get_num_gpus() > 0 and use_gpu:
                self.index = faiss.index_cpu_to_all_gpus(self.index)

        logging.info('Adding vectors to index...')

        self.index.add(self.vectors)

    def load(self, path):
        self.index = faiss.read_index(path)

    def save(self, path):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)

    def save_vectors(self, path):
        pickle.dump(self.vectors, open(path, 'wb'), protocol=4)

    def search(self, vectors, k=1, probes=512):
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        faiss.normalize_L2(vectors)
        try:
            self.index.nprobe = probes
        except:
            pass
        distances, ids = self.index.search(vectors, k)
        similarities = [(2-d)/2 for d in distances]
        return ids, similarities


class InputExample:
    def __init__(self, texts: List[str],  label: Union[int, float] = 0):
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, text: {}".format(str(self.label), self.texts[0])


class DenseRetriever:
    def __init__(self, model_path=None, batch_size=256, use_gpu=True, dim=768):
        if model_path is not None:
            self.load_model(model_path)
        self.vector_index = VectorIndex(dim)
        self.batch_size = batch_size
        self.use_gpu = use_gpu

    def load_model(self, model_path=None):
        logging.info("Loading model weights...")

        word_embedding_model = Transformer(model_path)
        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=False,
                                pooling_mode_cls_token=True,
                                pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        logging.info("Loaded model weights.")

    def create_index_from_documents(self, documents):
        logging.info('Building index...')

        self.vector_index.vectors = self.model.encode(documents, batch_size=self.batch_size, show_progress_bar=True)
        self.vector_index.build(self.use_gpu)

        logging.info('Built index')

    def create_index_from_vectors(self, vectors_path):
        logging.info('Building index...')
        logging.info('Loading vectors...')
        self.vector_index.vectors = pickle.load(open(vectors_path, 'rb'))
        logging.info('Vectors loaded')
        self.vector_index.build(use_gpu=False)

        logging.info('Built index')

    def search(self, queries, topk=1000, probes=512, min_similarity=0):
        query_vectors = self.model.encode(queries, batch_size=self.batch_size)
        ids, similarities = self.vector_index.search(query_vectors, k=topk, probes=probes)
        results = []
        for j in range(len(ids)):
            results.append([
                (ids[j][i], similarities[j][i]) for i in range(len(ids[j])) if similarities[j][i] > min_similarity
            ])
        return results

    def load_index(self, path):
        self.vector_index.load(path)

    def save_index(self, index_path='', vectors_path=''):
        if vectors_path != '':
            self.vector_index.save_vectors(vectors_path)
        if index_path != '':
            self.vector_index.save(index_path)

    def train(self,
              train_examples,
              dev_examples,
              model_name="sentence-transformers/all-distilroberta-v1",
              output_path="weights",
              epochs=3,
              evaluation_steps=1000,
              warmup_steps=1000,
              batch_size=64
              ):
        self.load_model(model_name)

        logging.info("Loading dataset...")

        examples = []
        for ex in train_examples:
            if ex[2] == 1:
                examples.append(InputExample(texts=[ex[0], ex[1]], label=1))

        train_examples = examples
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = MultipleNegativesRankingLoss(self.model, similarity_fct=dot_score)

        dev_sents_a = [ex[0] for ex in dev_examples]
        dev_sents_b = [ex[1] for ex in dev_examples]
        dev_labels = [ex[2] for ex in dev_examples]
        evaluator = EmbeddingSimilarityEvaluator(dev_sents_a,
                                                 dev_sents_b,
                                                 dev_labels,
                                                 name='dev',
                                                 batch_size=batch_size)
        logging.info("Training model...")

        # Train the model
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=evaluator,
                       epochs=epochs,
                       save_best_model=False,
                       evaluation_steps=evaluation_steps,
                       checkpoint_save_steps=evaluation_steps,
                       checkpoint_save_total_limit=1,
                       warmup_steps=warmup_steps,
                       checkpoint_path=output_path,
                       output_path=output_path)


def remove_newlines_tabs(text):
    return text.replace("\n", " ").replace("\t", " ").replace("\r", "")


def limit_str_tokens(text, limit):
    tokens = text.split(" ")
    return " ".join(tokens[:limit])


def limit_turns_tokens(texts, limit):
    added_tokens = 0
    trunc_texts = []
    for text in reversed(texts):
        trunc_text_tokens = []
        tokens = text.split(" ")
        for token in tokens:
            if added_tokens == limit:
                break
            trunc_text_tokens.append(token)
            added_tokens += 1
        trunc_texts.append(" ".join(trunc_text_tokens))
    trunc_texts = [t for t in trunc_texts if t != ""]
    trunc_texts = reversed(trunc_texts)
    return " | ".join(trunc_texts)


def prepare_query(d, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100):
    last_k_turns = 0
    use_title = True
    use_content = True

    turns = [t["text"] for t in d["thread"]]

    if turns_max_tokens > 0:
        query_turns = limit_turns_tokens(turns[-last_k_turns:], turns_max_tokens)
    else:
        query_turns = " ".join(turns[-last_k_turns:])

    query_title = d["post"]["title"] if use_title else ""
    if title_max_tokens > 0:
        query_title = limit_str_tokens(query_title, title_max_tokens)

    query_content = d["post"]["text"] if use_content else ""
    if post_max_tokens > 0:
        query_content = limit_str_tokens(query_content, post_max_tokens)

    query = remove_newlines_tabs(" | ".join([query_title, query_content, query_turns]))

    return query


if __name__ == "__main__":
    print('loading data...')
    wiki_to_doc = {}
    with open("collection.jsonl") as f:
        for line in tqdm(f):
            d = json.loads(line.rstrip("\n"))
            wiki_to_doc[d["wiki"]] = d["wiki"].replace('_', ' ') + ': ' + d["contents"]

    train_examples = []
    with open("train.jsonl") as f:
        for line in tqdm(f):
            d = json.loads(line.rstrip("\n"))
            for wiki in d["wiki_links"]:
               train_examples.append((prepare_query(d), wiki_to_doc[wiki], 1.0))

    print(train_examples[0])
    print('-------------------')
    
    dev_examples = []
    wikis = list(wiki_to_doc.keys())
    with open("gold.jsonl") as f:
        for line in tqdm(f):
            d = json.loads(line.rstrip("\n"))
            for wiki in d["wiki_links"]:
                query = prepare_query(d)
                dev_examples.append((query, wiki_to_doc[wiki], 1.0))
                # sample 100 negative examples for the query so that the wiki doc is not in the wiki_links
                for i in range(100):
                    random_wiki = wikis[np.random.randint(len(wikis))]
                    while random_wiki in d["wiki_links"]:
                        random_wiki = wikis[np.random.randint(len(wikis))]
                    dev_examples.append((query, wiki_to_doc[random_wiki], 0.0))
    
    print(dev_examples[0])

    # training example
    de = DenseRetriever()
    de.train(train_examples=train_examples,
             dev_examples=dev_examples,
             output_path="weights_dr_v_64",
             model_name="distilbert/distilbert-base-uncased",
             epochs=10,
             evaluation_steps=2000,
             warmup_steps=1000,
             batch_size=64)

    # search example
    # de = DenseRetriever(model_path="weights")
    # de.build_index(["hello", "hi"])
    # v = de.model.encode(["hello there"])
    # r = de.index.search(vectors=v)
