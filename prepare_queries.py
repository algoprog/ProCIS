import json


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


last_k_turns = 0
use_title = False
use_content = False

turns_max_tokens = 0  # 110+90
title_max_tokens = 0  # 40
post_max_tokens = 0  # 90

q1 = "last-{}-turns".format(last_k_turns) if last_k_turns > 0 else "all-turns"
q2 = "title" if use_title else "no-title"
q3 = "content" if use_content else "no-content"
filename = "{}_{}_{}".format(q1, q2, q3)

splits = ["test", "test_time"]

qid = 0
for s in splits:
    oq = open("queries_{}_{}.tsv".format(s, filename), "w+")
    oqr = open("qrels_{}_{}.txt".format(s, filename), "w+")
    with open("{}.jsonl".format(s)) as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            turns = [t["text"] for t in d["thread"]][:-1]

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

            #query = remove_newlines_tabs(" | ".join([query_title, query_content, query_turns]))
            query = remove_newlines_tabs(" | ".join([query_title, query_content]))

            oq.write("{}\t{}\n".format(qid, query))

            for doc_id in d["doc_ids"]:
                oqr.write("{} 0 {} 1\n".format(qid, doc_id))

            qid += 1
