# ProCIS: A Benchmark for Proactive Retrieval in Conversations

[![arxiv](https://img.shields.io/badge/arXiv-2405.06460-b31b1b.svg)](https://arxiv.org/abs/2405.06460)

The field of conversational information seeking is changing how we interact with search engines through natural language interactions. Existing datasets and methods are mostly evaluating reactive conversational information seeking systems that solely provide response to every query from the user. We identify a gap in building and evaluating proactive conversational information seeking systems that can monitor a multi-party human conversation and proactively engage in the conversation at an opportune moment by retrieving useful resources and suggestions. In this paper, we introduce a large-scale dataset for proactive document retrieval that consists of over 2.8 million conversations. We conduct crowdsourcing experiments to obtain high-quality and relatively complete relevance judgments through depth-k pooling. We also collect annotations related to the parts of the conversation that are related to each document, enabling us to evaluate proactive retrieval systems. We introduce normalized proactive discounted cumulative gain (npDCG) for evaluating these systems, and further provide benchmark results for a wide range of models, including a novel model we developed for this task.

You can download the dataset from [here](https://archive.org/details/procis). The zip contains the data splits for training and evaluation and the Wikipedia corpus.

# Dataset

You can download the dataset [here](https://archive.org/details/procis). It's a 5GB compressed zip with all the required .jsonl files.

# Data format

ProCIS consists of four subsets: train, dev, future-dev, and test. The three subsets of train, dev, and test are split randomly, while the future-dev set only contains conversations that follow after the conversations in the training set chronologically. This split can be used for evaluating the generalization capabilities of retrieval models in potentially new emerging concepts and topics not seen during training. The test split was sampled from 100 unique random subreddits, all from posts with at least a Reddit score of 20 to ensure high quality. The test set has relevance judgements from crowdsourcing. Along with relevance judgements, we also collected evidence annotations to enable evaluation of proactive search systems. Each split is a jsonl file with the following format:

```jsonl
{
  "post": {
    "id": <post id>,
    "title": <title of the post>,
    "text": <text of the post>,
    "author": <username>,
    "date": <unix timestamp>,
    "subreddit": <subreddit name>,
    "subreddit_id": <subreddit id>,
    "score": <reddit score>,
    "num_comments": <number of comments>,
    "url": <reddit url of the post>,
    "thread": [{
      "id": "response id",
      "author": <username>,
      "text": "response text",
      "date": <unix timestamp>,
      "score": <reddit score>,
      "wiki_links": [<list of wikipedia page names mentioned>],
      "annotations": [  (list of crowdsourced annotations)
      {"wiki": <relevant wikipedia page name>,
       "score" <1 if partially relevant, 2 for relevant>,
       "evidence": [{"comment_id": <0 for post or title, 1+ for comments>,
                    "text": <supporting evidence for wikipedia article>},...]},...
      ]
    },...]
  },
  "wiki_links": [<list of all wikipedia page names mentioned in the thread>],
  "annotations": [<list of all annotations in the thread>]
}
```

The annotations field is only for the test split, the rest of the schema remains the same for all splits.

# Pre-trained Models
- [Proactive Binary Classifier (DeBERTa-v3-base)](https://huggingface.co/algoprog/DeBERTa-v3-base-ProCIS-Classifier)
- [Dense Retriever (ANCE-distilbert)](https://huggingface.co/algoprog/ANCE-distilbert-ProCIS)

# Baselines

In this repo you can reproduce the baseline results for BM25, vanilla dense retrieval (single-vector) and LMGR (see `baselines.py`). In the following 2 tables are our results for all the other models we tested.

## Reactive Retrieval

| **Model** | **nDCG@5** | **nDCG@20** | **nDCG@100** | **MRR** | **MAP** | **R@5** | **R@20** | **R@100** | **R@1K** |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.0654 | 0.0754 | 0.0969 | 0.1561 | 0.0395 | 0.0410 | 0.0687 | 0.1202 | 0.2266 |
| SPLADE | 0.1605 | 0.1578 | 0.1575 | 0.4752 | 0.0752 | 0.0946 | 0.1343 | 0.1432 | 0.2946 |
| ANCE | 0.1854 | 0.1912 | 0.2240 | 0.4902 | 0.0984 | 0.0989 | 0.1635 | 0.2517 | 0.4316 |
| ColBERT | 0.2091 | 0.2094 | 0.2383 | 0.5679 | 0.1113 | 0.1117 | 0.1778 | 0.2649 | 0.4564 |
| LMGR, k=1 | 0.2638 | 0.3678 | 0.3678 | 0.6187 | 0.2000 | 0.2116 | 0.4091 | 0.4091 | 0.4091 |
| LMGR, k=3 | 0.2714 | 0.3986 | 0.3986 | 0.6132 | 0.2198 | 0.2354 | 0.4614 | 0.4614 | 0.4614 |
| **LMGR, k=5** | **0.3408\*** | **0.4524\*** | - | **0.6300\*** | **0.2663\*** | **0.2853\*** | **0.5306\*** | - | - |

## Proactive Retrieval

| **Model** | **npDCG@5** | **npDCG@20** | **npDCG@100** |
|---|---|---|---|
| BM25 | 0.0229 | 0.0337 | 0.0405 |
| SPLADE | 0.1305 | 0.1440 | 0.1542 |
| ANCE | 0.1508 | 0.1792 | 0.2061 |
| **ColBERT** | **0.1719** | **0.1944** | **0.2172** |
| LMGR, k=1 | 0.0574 | 0.1445 | - |
| LMGR, k=3 | 0.0613 | 0.1527 | - |
| LMGR, k=5 | 0.0781 | 0.1840 | - |

# Dataset Statistics

|   | **train** | **dev** | **future-dev** | **test** |
|---|---|---|---|---|
| Total conversations | 2,830,107 | 4165 | 3385 | 100 |
| Total posts | 1,893,201 | 4165 | 3385 | 100 |
| Number of subreddits covered | 34,785 | 1563 | 1659 | 100 |
| Total unique users in the conversations | 2,284,841 | 10,896 | 7,920 | 309 |
| Avg. number of turns | 5.41 (± 7.81) | 4.91 (± 3.60) | 4.48 (± 3.30) | 4.49 (± 1.60) |
| Avg. number of words per conversation | 406.01 (± 774.67) | 359.19 (± 734.95) | 325.36 (± 609.58) | 173.85 (± 101.22) |
| Avg. number of words per turn | 70.54 (± 82.38) | 68.77 (± 74.80) | 72.55 (± 85.37) | 41.58 (± 26.49) |
| Avg. number of Wikipedia links per conversation | 1.71 (± 2.46) | 1.90 (± 3.03) | 1.15 (± 0.57) | 1.15 (± 0.46) |
| Avg. number of unique users per conversation | 3.17 (± 1.41) | 2.93 (± 1.16) | 2.88 (± 1.11) | 3.41 (± 1.39) |
| Avg. number of comments per user | 6.71 (± 462.74) | 1.88 (± 8.21) | 1.92 (± 12.93) | 1.45 (± 2.49) |

# Questions?
Open an issue or send an email to Chris Samarinas (csamarinas@umass.edu)

# Citation
You can use the following to cite our work:

```
@misc{Samarinas_2024_ProCIS,
      title={ProCIS: A Benchmark for Proactive Retrieval in Conversations}, 
      author={Chris Samarinas and Hamed Zamani},
      year={2024},
      eprint={2405.06460},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
