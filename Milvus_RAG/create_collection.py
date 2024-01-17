from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections

id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000, default_value="")
answer = FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=20000, default_value="")
text_embedding = FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=128)

schema = CollectionSchema(
    fields=[id_field, text, answer, text_embedding],
    description="Text search",
    enable_dynamic_field=True
)

collection_name = "med_qa_tw_en_bigbio"

connections.connect(host='localhost', port='19530')

collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)

