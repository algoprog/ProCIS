import json
import os

from tqdm import tqdm
from trectools import TrecQrel, TrecRun, TrecEval
from model_lmgr import LMGR
from cpg_metric import calculate_npdcg
from model_dr import DenseRetriever, prepare_query
from model_bm25 import BM25
from proactive_classifier import BinaryClassifier

method = 'lmgr'
topk = 20

print(f'Running non-proactive evaluation for {method}...')

# load corpus
print("Loading corpus...")
articles_descriptions = []
articles_titles = []
wiki_to_id = {}
with open("collection.jsonl") as f:
    for line in tqdm(f):
        d = json.loads(line)
        articles_descriptions.append(d["wiki"].replace('_', ' ') + ': ' + d["contents"])
        articles_titles.append(d["wiki"])
        wiki_to_id[d["wiki"]] = len(articles_descriptions) - 1

if method == 'dr':
    index_path = 'index_dr_v_34K.pkl'
    print('loading model...')
    model = DenseRetriever('weights_dr_v_64/34K', dim=768)
    print('indexing...')
    if not os.path.exists(index_path):
        model.create_index_from_documents(articles_descriptions)
        model.save_index(vectors_path=index_path)
    else:
        model.create_index_from_vectors(vectors_path=index_path)
elif method == 'bm25':
    print('loading model...')
    model = BM25(load=True)
    print('indexing...')
    model.index_documents(articles_descriptions)
elif method == 'lmgr':
    model = LMGR()

print('loading golden set...')
# for non-proactive, using whole conversation as query
queries = [] # queries, one per conversation
rel_wikis = [] # wiki names for each query

# for proactive, using conversation histories as query
include_last_utterance = False
queries_proactive = [] # queries/histories, multiple per conversation
queries_proactive_acts = [] # binary labels, if 1 then search should run, otherwise return empty result list
rel_wikis_proactive = [] # wiki names for each conversation subquery

print('loading proactive classifier...')
classifier = BinaryClassifier()
classifier.load_model('best_model.pth')

print('loading data...')
with open('test_final.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if method == 'dr':
            query = prepare_query(d, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
        else:
            query = prepare_query(d, turns_max_tokens=0, title_max_tokens=0, post_max_tokens=0)
        if method == 'lmgr':
            query = d
        queries.append(query)

        wiki_links = [(wiki_to_id[annotation['wiki']], annotation['score']) for annotation in d['annotations']]
        rel_wikis.append(wiki_links)

        subqueries = []
        subqueries_acts = [] # a list of binary labels, if 1 then search should run, otherwise return empty result list
        subqueries_wikis = []
        for i in range(len(d['thread'])):
            d_sub = d.copy()
            if include_last_utterance:
                d_sub['thread'] = d['thread'][:i+1]
            else:
                d_sub['thread'] = d['thread'][:i]
            
            subquery = prepare_query(d_sub, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
            pred_label = classifier.predict(subquery)
            subqueries_acts.append(pred_label)
            
            if method != 'lmgr':
                if method == 'dr':
                    subquery = prepare_query(d_sub, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
                else:
                    subquery = prepare_query(d_sub, turns_max_tokens=0, title_max_tokens=0, post_max_tokens=0)
                subqueries.append(subquery)
            else:
                subqueries.append(d_sub)

            wiki_links = [(wiki_to_id[annotation['wiki']], annotation['score']) for annotation in d['thread'][i]['annotations']]
            subqueries_wikis.append(wiki_links)
        
        queries_proactive.append(subqueries)
        queries_proactive_acts.append(subqueries_acts)
        rel_wikis_proactive.append(subqueries_wikis)

# evaluation for non-proactive
print('running non-proactive evaluation...')
scores = model.search(queries, topk=topk) # list of lists of tuples (doc_id, score)

# create TrecQrel and TrecRun objects
qrel_data = [(i, wiki[0], wiki[1]) for i, wikis in enumerate(rel_wikis) for wiki in wikis]
# write to csv with headers query, docid, rel
with open('qrel.csv', 'w+') as f:
    # query docid rel
    for i, wiki in enumerate(qrel_data):
        f.write(' '.join(map(str, wiki)) + '\n')

run_data = [(i, wiki[0], rank, wiki[1]) for i, scores_ in enumerate(scores) for rank, wiki in enumerate(scores_)]
# write to csv with headers query, docid, rank
with open('run.csv', 'w+') as f:
    # query docid rank score
    for i, wiki in enumerate(run_data):
        f.write(' '.join(map(str, wiki)) + '\n')

qrel = TrecQrel('qrel.csv', qrels_header=['query', 'docid', 'rel'])
run = TrecRun()
run.read_run('run.csv', run_header=['query', 'docid', 'rank', 'score'])

# create TrecEval object
te = TrecEval(run, qrel)

# calculate metrics
ndcg_at_5 = te.get_ndcg(depth=5)
ndcg_at_10 = te.get_ndcg(depth=10)
ndcg_at_20 = te.get_ndcg(depth=20)
ndcg_at_100 = te.get_ndcg(depth=100)
ndcg_at_1000 = te.get_ndcg(depth=1000)
mrr = te.get_reciprocal_rank()
map_score = te.get_map()

def calculate_recall(retrieved, relevant):
    retrieved = [doc_id for doc_id, score in retrieved]
    relevant = [doc_id for doc_id, score in relevant]
    tp = len(set(retrieved) & set(relevant))
    fn = len(set(relevant) - set(retrieved))
    return tp / (tp + fn) if tp + fn > 0 else 0

# calculate recall for each query
r_at_5 = [calculate_recall(scores[i][:5], rel_wikis[i]) for i in range(len(rel_wikis))]
r_at_10 = [calculate_recall(scores[i][:10], rel_wikis[i]) for i in range(len(rel_wikis))]
r_at_20 = [calculate_recall(scores[i][:20], rel_wikis[i]) for i in range(len(rel_wikis))]
r_at_100 = [calculate_recall(scores[i][:100], rel_wikis[i]) for i in range(len(rel_wikis))]
r_at_1000 = [calculate_recall(scores[i][:1000], rel_wikis[i]) for i in range(len(rel_wikis))]

# calculate average recall
r_at_5_avg = sum(r_at_5) / len(r_at_5)
r_at_10_avg = sum(r_at_10) / len(r_at_10)
r_at_20_avg = sum(r_at_20) / len(r_at_20)
r_at_100_avg = sum(r_at_100) / len(r_at_100)
r_at_1000_avg = sum(r_at_1000) / len(r_at_1000)

print(f'NDCG@5: {ndcg_at_5}, NDCG@10: {ndcg_at_10}, NDCG@20: {ndcg_at_20}, NDCG@100: {ndcg_at_100}, NDCG@1000: {ndcg_at_1000}')
print(f'MRR: {mrr}, MAP: {map_score}')
print(f'R@5: {r_at_5_avg}, R@10: {r_at_10_avg}, R@20: {r_at_20_avg}, R@100: {r_at_100_avg}, R@1000: {r_at_1000_avg}')

print('---')

# evaluation for proactive
print('running proactive evaluation...')

npdcg = []
cuttoffs = [5, 10, 20, 100, 1000]

for conv_id, subqueries in enumerate(queries_proactive):
    scores = [model.search([subqueries[i]], topk=topk)[0] if queries_proactive_acts[conv_id][i] == 1 else [] for i in range(len(subqueries))]
    print(scores)
    retrieved = []
    for thread_id, subquery_scores in enumerate(scores):
        print(f"subquery_scores: {subquery_scores}")
        retrieved_docs = [doc_id for doc_id, score in subquery_scores]
        correct_docs = rel_wikis_proactive[conv_id][thread_id]
        retrieved.append({'retrieved_docs': retrieved_docs, 'correct_docs': correct_docs})
    npdcg.append(calculate_npdcg(retrieved, [5, 10, 20, 100, 1000]))

# calculate average npdcg per cutoff, calculate_npdcg returns dict
npdcg_avg = {c: sum([npdcg[i][c] for i in range(len(npdcg))]) / len(npdcg) for c in cuttoffs}
print(npdcg_avg)
