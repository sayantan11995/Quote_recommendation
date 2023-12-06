from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import random

def find_most_similar_documents_sbert(query, candidate_documents, device, k=5):
    # Load the Sentence-BERT model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    model.to(device)

    # Encode the query and candidate documents
    query_embedding = model.encode(query, convert_to_tensor=True)
    candidate_ids, candidate_texts = zip(*candidate_documents.items())
    candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

    # Compute cosine similarity between the query and candidate embeddings
    similarities = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0].tolist()

    # Create a list of (docid, similarity) pairs and sort them in descending order
    sim_ranking = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    # Get the top k most similar docids
    top_k_docids = [int(candidate_ids[idx]) for idx, _ in sim_ranking[:k]]

    return top_k_docids

def rank_paragraph_sbert(test_set, docid_to_text, ctxid_to_text, device, k=200):

    # text_to_docid = {contents: docid for docid, contents in  docid_to_text.items()}
        
    # corpus = list(text_to_docid.keys())

    print(f"Len docid_text: {len(docid_to_text)}")
    ranked_list = {}
    for i in range(len(test_set)):
        seq = test_set[i]
        qid, label, cands = seq[0], seq[1], seq[2]
        q_text = ctxid_to_text[str(qid)]

        # q_text = ctxid_to_text[qid]

        # cands = [x+1 for x in cands]
        # cands = label + create_candidates_from_list(cands, 200, label[0])

        candidate_docs = {docid: docid_to_text[str(docid)] for docid in cands}

        ranked_list[qid]  = find_most_similar_documents_sbert(q_text, candidate_docs, device, k)

    return ranked_list


def bm25_ranking(query, candidate_documents, k=5):
    # Extract document texts and ids
    candidate_ids, candidate_texts = zip(*candidate_documents.items())

    # Tokenize the documents
    tokenized_docs = [doc.split() for doc in candidate_texts]

    # Initialize the BM25 model
    bm25 = BM25Okapi(tokenized_docs)

    # Tokenize the query
    query_tokens = query.split()

    # Get BM25 scores for each document
    bm25_scores = bm25.get_scores(query_tokens)

    # Create a list of (docid, score) pairs and sort them in descending order
    score_ranking = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)

    # Get the top k docids
    top_k_docids = [candidate_ids[idx] for idx, _ in score_ranking[:k]]

    return top_k_docids

def rank_paragraph_bm25(test_set, docid_to_text, ctxid_to_text, k=200):
    print(list(docid_to_text.keys())[:10])
    bm25_ranked_list = {}
    for i in range(len(test_set)):
        seq = test_set[i]
        qid, label, cands = seq[0], seq[1], seq[2]
        # print(label)
        # print(cands[:100])
        q_text = ctxid_to_text[str(qid)]

        # cands = [x+1 for x in cands]
        # cands = label + create_candidates_from_list(cands, 200, label[0])

        candidate_docs = {docid: docid_to_text[str(docid)] for docid in cands}

        bm25_ranked_list[qid] = bm25_ranking(q_text, candidate_docs, k)

    return bm25_ranked_list


def create_candidates_from_list(lst, n, exclude_value):
    # Remove the excluded value from the list
    available_values = [value for value in lst if value != exclude_value]

    # Check if there are enough available values for sampling
    if len(available_values) < n:
        raise ValueError("Not enough available values for sampling.")

    # Randomly select n values from the available values
    selected_values = random.sample(available_values, n)

    return selected_values