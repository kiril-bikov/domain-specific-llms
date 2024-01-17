from flask import Flask, request, jsonify
from pymilvus import Collection, connections
from embed import embed_input

app = Flask(__name__)
app.config['ALLOWED_HOSTS'] = '*'

conn = connections.connect(host="127.0.0.1", port=19530)

@app.route('/insert', methods=['POST'])
def insert_document():
    data = request.json

    collection_name = data['collection']
    collection = Collection(name=collection_name)
    collection.load()
    del data['collection']


    embeddings = embed_input([data['text']])

    data['text_embedding'] = embeddings[0][:128]
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        collection.insert(data)

        return jsonify({"message": "Document inserted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['GET'])
def search_similar_records():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    collection_name = data['collection']
    collection = Collection(name=collection_name)
    collection.load()

    number_documents = data.get('number_documents', 10)
    embeddings = embed_input([data['text']])

    query_embedding = embeddings.tolist()[0][:128]

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    search_results = collection.search(
        data=[query_embedding],
        anns_field="text_embedding",
        param=search_params,
        limit=number_documents,
        output_fields = ["text", "answer"],
        expr=None,
        consistency_level="Strong"
    )

    records = [
        {
			"id": hit.id,
            "text": hit.entity.get("text"),
            "answer": hit.entity.get("answer"),
            "distance": hit.distance
		} for hit in search_results[0]
    ]

    return jsonify(records)

if __name__ == '__main__':
    app.run(port=5003,debug=True)
