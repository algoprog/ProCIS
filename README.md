# ProCIS

[Download Dataset](https://drive.google.com/file/d/1aXBDAgjdJ6E83ia5kndZLntnVqO9chIF/view?usp=sharing)

When running experiments exclude the last utterance from your queries.

We want [ColBERT](https://github.com/stanford-futuredata/ColBERT) baselines (zero-shot from MSMARCO + fine-tuned only on ProCIS) like these [here](https://docs.google.com/spreadsheets/d/1nbz21FptIWDYNeomtFdNd689vKNbPwKohQjSbT7tt7k/edit#gid=0). We also need the ranklists from the best fine-tuned ColBERT model (query id, doc id, rank) to use for annotation later.

For preparing the queries and qlrels files using the original dataset files, you can use `prepare_queries.py`
