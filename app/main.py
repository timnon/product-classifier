from flask import Flask, request, jsonify, render_template
import os
import pickle
import json

from embeddings import get_vec


app = Flask(__name__)

mod = pickle.load(open('model.pkl', 'rb'))


@app.route('/classify', methods=['POST'])
def classify(n_results=3):
    data = request.get_json(force=True)
    title = data['title']
    result = dict()
    result['title'] = title

    # compute product type probabilities
    vec = get_vec(title)
    vec = vec.reshape(1,-1)
    probs = mod.predict_proba(vec)[0]

    probs = zip(mod.classes_,probs)
    probs = sorted(probs,key=lambda x: x[1],reverse=True)
    result['top_3_results'] = [ 
        {
            'product_type': item[0], 
            'score': item[1]
        } 
        for item in probs[:3] ]
    result['product_type'] = probs[0][0]
    return json.dumps(result)


# Basic GUI for testing 
@app.route("/", methods=['GET', 'POST'])
def gui():
	return render_template('template.html',text='',tags='no tags yet')


if __name__ == '__main__':
    app.run(debug=True, host=os.getenv('HOST'), port=os.getenv('PORT'))

    # curl -X POST -H "Content-Type: application/json" -d '{"title":"Aquarelle"}' http://localhost:5000/classify