from pymilvus import connections, Collection, Index, IndexType

connections.connect()
collection_name = "med_qa_tw_en_bigbio"
collection = Collection(name=collection_name)

index_params = {
    "index_type":"IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}

index = Index(collection, field_name="text_embedding", index_params=index_params)