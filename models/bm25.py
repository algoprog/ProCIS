import logging
import os
import tantivy
import re

from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

def clean_string(s):
    cleaned = re.sub(r'\W+', ' ', s)
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned.lower().strip()

class BM25:
    def __init__(self, path='index_bm25', load=False):
        if not os.path.exists(path):
            os.mkdir(path)
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("body", stored=False)
        schema_builder.add_unsigned_field("doc_id", stored=True)
        schema = schema_builder.build()
        self.index = tantivy.Index(schema, path=path, reuse=load)
        self.searcher = self.index.searcher()

    def index_documents(self, documents):
        logging.info('Building sparse index of {} docs...'.format(len(documents)))
        writer = self.index.writer()
        for i, doc in tqdm(enumerate(documents)):
            writer.add_document(tantivy.Document(
                body=[clean_string(doc)],
                doc_id=i
            ))
            if (i+1) % 100000 == 0:
                writer.commit()
        writer.commit()
        logging.info('Built sparse index')
        self.index.reload()
        self.searcher = self.index.searcher()

    def search(self, queries, topk=100):
        results = []
        for q in tqdm(queries, desc='searched'):
            docs = []
            query = self.index.parse_query(clean_string(q), ["body"])
            scores = self.searcher.search(query, topk).hits
            docs = [(self.searcher.doc(doc_id)['doc_id'][0], score)
                    for score, doc_id in scores]
            results.append(docs)

        return results


if __name__ == '__main__':
    bm25 = BM25()
    bm25.index_documents(['hello world', 'foo bar', 'baz qux'])
    print(bm25.search(['hello', 'world', 'foo', 'bar', 'baz', 'qux', 'quux', 'corge', 'grault', 'garply']))
