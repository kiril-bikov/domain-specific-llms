from pymilvus import connections, Collection, utility

connections.connect()

collection_name = "med_qa_tw_en_bigbio"

if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)

    collection.drop()
    print(f"Collection '{collection_name}' dropped successfully.")
else:
    print(f"Collection '{collection_name}' does not exist.")

