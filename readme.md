## Paragraph Ranking

Dataset: [[ctx_id, [pos_doc_id], [neg_doc_id * (n-1), pos_doc_id]], ..] => dataset_n_neg.pkl
labels: {'ctx_id1': [pos_doc_id1, pos_doc_id2, pos_doc_id3], 'ctx_id2': [pos_doc_id1, pos_doc_id2], ..} => labels_n_neg.pkl
ctxid_to_text: {ctx_id1: ctx1, ctx_id2: ctx2, ..}
docid_to_text: {doc_id1: doc1, doc_id2: doc2, ..}


## Full system eval data

full_system_data_<book>: {ctxid: [quote_id, para_id]}