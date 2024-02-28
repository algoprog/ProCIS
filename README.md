# ProCIS: A Benchmark for Proactive Retrieval in Conversations

The field of conversational information seeking, which is rapidly gaining interest in both academia and industry, is changing how we interact with search engines through natural language interactions. Existing datasets and methods are mostly evaluating reactive conversational information seeking systems that solely provide response to every query from the user. We identify a gap in building and evaluating proactive conversational information seeking systems that can monitor a multi-party human conversation and proactively engage in the conversation at an opportune moment by retrieving useful resources and suggestions. In this paper, we introduce a large-scale dataset for proactive document retrieval that consists of over 2.8 million conversations. We conduct crowdsourcing experiments to obtain high-quality and relatively complete relevance judgments through depth-k pooling. We also collect annotations related to the parts of the conversation that are related to each document, enabling us to evaluate proactive retrieval systems. We introduce normalized proactive discounted cumulative gain (npDCG) for evaluating these systems, and further provide benchmark results for a wide range of models, including a novel model we developed for this task.

You can download the dataset from [here](https://archive.org/details/procis). The zip contains the data splits for training and evaluation and the Wikipedia corpus.

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
      "text": "Response text",
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

# Baselines

| **Model** | **npDCG@5** | **npDCG@20** | **npDCG@100** |
|---|---|---|---|
| BM25 | 0.0124 | 0.0177 | 0.0201 |
| SPLADE | 0.0707 | 0.0757 | 0.0766 |
| ANCE | 0.0817 | 0.0942 | 0.1024 |
| ColBERT | 0.0931 | 0.1022 | 0.1079 |
| LMGR, k=1 | 0.2059 | 0.3143 | - |
| LMGR, k=3 | 0.2201 | 0.3321 | - |
| **LMGR, k=5** | **0.2802*** | **0.4002*** | - |

# Citation

You can use the following to cite our work:

```
@article{Anonymous_2024_ProCIS,
  title   =  {ProCIS: A Benchmark for Proactive Retrieval in Conversations},
  author  =  {},
  journal =  {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year    =  {2024}
}
```
