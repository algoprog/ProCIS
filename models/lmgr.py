import json
import os
import openai
import re

from model_dr import DenseRetriever
from tqdm import tqdm

def chatgpt_api(prompt, model_name='gpt-4-turbo-preview'):
    openai.api_key = "YOUR-API-KEY"
    openai.api_base = "https://api.openai.com/v1"
    response = openai.ChatCompletion.create(
        model=model_name,
        max_tokens=3000,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    used_tokens = response["usage"]["total_tokens"]
    return response["choices"][0]["message"]["content"].strip(), used_tokens

def custom_chatgpt_api(prompt):
    openai.api_key = "YOUR-API-KEY"
    openai.api_base = "https://api.deepinfra.com/v1/openai"
    gpt_model_name = 'openchat/openchat_3.5'
    response = openai.ChatCompletion.create(
        model=gpt_model_name,
        max_tokens=3000,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    used_tokens = response["usage"]["total_tokens"]
    return response["choices"][0]["message"]["content"].strip(), used_tokens


def remove_enumeration(s):
    return re.sub(r'^\d+\.\s', '', s).strip()


def parse_json(data):
    try:
        return json.loads(data)
    except:
        prompt = f"fix the json below, return only valid json in your response like this {{\"title\":..., \"description\":...}}, nothing else:\n\n{data}\n\n"
        response, used_tokens = chatgpt_api(prompt)
        return json.loads(response)


class LMGR:
    def __init__(self):
        self.load_corpus()
        self.load_retriever()

    def load_corpus(self):
        print("Loading corpus...")
        self.docs = []
        self.wikis = []
        with open("collection.jsonl") as f:
            for line in tqdm(f):
                d = json.loads(line)
                self.docs.append(d["wiki"] + ': ' + d["contents"][:60] + '...')
                self.wikis.append(d["wiki"])
    
    def load_retriever(self):
        print('loading model...')
        self.retriever = DenseRetriever('sentence-transformers/all-MiniLM-L6-v2', dim=384)
        print('indexing...')
        if os.path.exists('embeddings.pkl'):
            print('loading cached embeddings...')
            self.retriever.create_index_from_vectors('embeddings.pkl')
        else:
            print('creating embeddings...')
            self.retriever.create_index_from_documents(self.docs)
            self.retriever.save_index(vectors_path='embeddings.pkl')
    
    def search(self, conversations, topk=20, candidates=3, proactive=False):
        results = []
        for conversation in tqdm(conversations):
            comments = conversation["thread"]
            user_to_id = {}
            uid = 1
            for c in comments:
                if c["author"] not in user_to_id:
                    user_to_id[c["author"]] = uid
                    uid += 1
            reg_exp = r'\n+'
            comments_str = "\n".join([f"user {user_to_id[c['author']]}: {re.sub(reg_exp, ' ', c['text'])}" for c in comments])
            
            if proactive:
                prompt = f"\npost title: {conversation['post']['title']}\n\npost text: {conversation['post']['text']}\n\ncomments:\n{comments_str}\n\n" \
                f"based on the last comment, or the post if no comments shown, give up to {topk} wikipedia articles that provide " \
                f"useful information and answer potential questions, ambiguities or misunderstandings or they provide some relevant context, only show articles that add very useful context, you don't need to generate all {topk}, you might also give 0 results if not necesary, " \
                "ordered by relevance and use jsonl format, each line has a json with title and description, nothing else in your response:\n\n{\"title\":..., \"description\":...}\n{\"title\":..., \"description\":...}...\n\ndescription is the first sentence of the wikipedia article\n\n"
            else:
                prompt = f"\npost title: {conversation['post']['title']}\n\npost text: {conversation['post']['text']}\n\ncomments:\n{comments_str}\n\n" \
                f"based on the content of the post and the comments, give up to {topk} wikipedia articles that provide " \
                f"useful information and answer potential questions, ambiguities or misunderstandings from the conversation or they provide some relevant context, only show articles that add very useful context, you don't need to generate all {topk} if you are not confident, you might also give 0 results, " \
                "ordered by relevance and use jsonl format, each line has a json with title and description, nothing else in your response:\n\n{\"title\":..., \"description\":...}\n{\"title\":..., \"description\":...}...\n\ndescription is the first sentence of the wikipedia article\n\n"
                
            try:
                response, used_tokens = custom_chatgpt_api(prompt)
            except:
                response, used_tokens = chatgpt_api(prompt, model_name='gpt-3.5-turbo')

            response = response.replace('\n\n', '\n')
            generated_candidate_docs = response.split('\n')
            generated_candidate_docs = [parse_json(remove_enumeration(d)) for d in generated_candidate_docs if d]
            generated_candidate_docs = [f"{d['title']}: {d['description']}" for d in generated_candidate_docs]
            final_docs = []
            added_docs = set()
            for doc in tqdm(generated_candidate_docs):
                retrieved_candidates = self.retriever.search([doc], candidates)[0]
                retrieved_candidates_str = "\n".join([f"{i+1}. {self.docs[cand_doc[0]]}" for i, cand_doc in enumerate(retrieved_candidates)])
                prompt = f"Pick the candidate that describes the same concept with the given document, return only the number "\
                         f"in your response, nothing else, if none of them then return 0:\n\ndocument: {doc}\n\ncandidates: {retrieved_candidates_str}\n\n"
                try:
                    response, used_tokens = custom_chatgpt_api(prompt)
                except:
                    response, used_tokens = chatgpt_api(prompt, model_name='gpt-3.5-turbo')
                cand_id = 0
                try:
                    cand_id = int(re.sub(r'\D', '', response).strip())
                except:
                    pass
                
                if cand_id > len(retrieved_candidates) or cand_id < 0:
                    cand_id = 0
                
                if cand_id > 0:
                    retrieved_candidate = retrieved_candidates[cand_id-1]
                    if retrieved_candidate[0] not in added_docs:
                        added_docs.add(retrieved_candidate[0])
                        final_docs.append(retrieved_candidate)

            results.append(final_docs)

        return results

if __name__ == '__main__':
    lmgr = LMGR()
    with open('test.jsonl') as f:
        conversations = [json.loads(line) for line in f]
        results = lmgr.search(conversations[:1]) 
