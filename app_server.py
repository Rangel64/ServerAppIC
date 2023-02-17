from flask import Flask, request, jsonify
import json
import base64
import Estimador


app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])


def index():
    global reponse
    
    if(request.method == 'POST'):
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        decodeString = request_data['image']
        imageBytes = base64.b64decode((decodeString))
        numeroDeAnimais = Estimador.result(imageBytes)
        return jsonify({'response': str(numeroDeAnimais)})
        
if(__name__ == "__main__"):
    app.run(host = '192.168.100.218',debug=True)
    