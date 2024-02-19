import numpy as np

from typing import List, Dict, Tuple

def calculate_npdcg(retrieved: List[Dict[str, List[Tuple[str, int]]]], cutoffs: List[int]) -> Dict[int, float]:
    npdcg_values = {}
    for cutoff in cutoffs:
        pdcg = calculate_pdcg(retrieved, cutoff)
        ipdcg = calculate_ipdcg(retrieved, cutoff)
        npdcg = pdcg / ipdcg if ipdcg != 0 else 0
        npdcg_values[cutoff] = npdcg
    return npdcg_values

def calculate_pdcg(retrieved: List[Dict[str, List[Tuple[str, int]]]], cutoff: int) -> float:
    pdcg = 0
    Z = 0
    ideal_position_l = {}
    labels = {}
    for i, utterance in enumerate(retrieved):
        for doc, label in utterance["correct_docs"]:
            if doc not in ideal_position_l and label > 0:
                ideal_position_l[doc] = i
                labels[doc] = label
    checked_docs = set()
    for i, utterance in enumerate(retrieved):
        retrieved_docs = utterance["retrieved_docs"][:cutoff]
        if len(retrieved_docs) > 0:
            Z += 1
            dcg = 0
            for j, doc in enumerate(retrieved_docs):
                if doc in ideal_position_l and i >= ideal_position_l[doc]:
                    if doc not in checked_docs:
                        checked_docs.add(doc)
                        dcg += labels[doc] / np.log2(2 + i - ideal_position_l[doc]) / np.log2(j + 2)
            pdcg += dcg
    return pdcg / Z if Z != 0 else 0


def calculate_ipdcg(retrieved: List[Dict[str, List[Tuple[str, int]]]], cutoff: int) -> float:
    ipdcg = 0
    Z = 0
    ideal_position_l = {}
    labels = {}
    for i, utterance in enumerate(retrieved):
        for doc, label in utterance["correct_docs"]:
            if doc not in ideal_position_l and label > 0:
                ideal_position_l[doc] = i
                labels[doc] = label
    checked_docs = set()
    for i, utterance in enumerate(retrieved):
        retrieved_docs = [doc for doc, _ in sorted(utterance["correct_docs"], key=lambda x: x[1], reverse=True)][:cutoff]
        if len(retrieved_docs) > 0:
            Z += 1
            dcg = 0
            for j, doc in enumerate(retrieved_docs):
                if doc not in checked_docs:
                    checked_docs.add(doc)
                    rel = labels[doc]
                    dcg += rel / np.log2(j + 2)
            ipdcg += dcg
    return ipdcg / Z if Z != 0 else 0


# retrieved = [
#     {
#         "retrieved_docs": [1, 2, 30, 4],
#         "correct_docs": [(1, 2), (2, 1), (14, 2)]
#     },
#     {
#         "retrieved_docs": [],
#         "correct_docs": [(9, 1)]
#     },
#     {
#         "retrieved_docs": [3, 7, 6, 9, 10],
#         "correct_docs": [(3, 2)]
#     },
#     {
#         "retrieved_docs": [12, 13, 14, 30],
#         "correct_docs": []
#     }
# ]

# retrieved = [
#     {
#         "retrieved_docs": [1],
#         "correct_docs": [(1, 1), (2, 1)]
#     },
#     {
#         "retrieved_docs": [3, 2, 4],
#         "correct_docs": [(3, 1), (4, 1)]
#     }
# ]

# cutoffs = [100]
# npdcg_values = calculate_npdcg(retrieved, cutoffs)
# print(npdcg_values)
