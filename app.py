from flask import Flask,jsonify,render_template
import tensorflow
import numpy as np
from inltk.inltk import get_sentence_encoding

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('result.html')

@app.route('/classifier/<string:n>')
def classifier(n):
    def get_embeddings(str):
        v=get_sentence_encoding(str, 'hi-en')
        return v
    
    def fake_model(emb):
        new_model = tensorflow.keras.models.load_model('my_model')
        y=[]
        y.append(emb)
        y=np.array(y)
        y=y.reshape([-1,1,400])
        y=new_model.predict(y)
        if(y>=0.5):
            return str("Fake "+"("+str(y[0][0])+")")
        else:
            return str("Non-Fake "+"("+str(y[0][0])+")")
    
    def hate_model(emb):
        new_model = tensorflow.keras.models.load_model('hate_model')
        y=[]
        y.append(emb)
        y=np.array(y)
        y=y.reshape([-1,1,400])
        y=new_model.predict(y)
        if(y>=0.5):
            return str("Hate "+"("+str(y[0][0])+")")
        else:
            return str("Non-Hate "+"("+str(y[0][0])+")")

    def sentiment_model(emb):
        new_model = tensorflow.keras.models.load_model('sentiment_model')
        y=[]
        y.append(emb)
        y=np.array(y)
        y=y.reshape([-1,1,400])
        y=new_model.predict(y)
        if(y>=0.5):
            return str("Positive "+"("+str(y[0][0])+")")
        else:
            return str("Negative "+"("+str(y[0][0])+")")
    
    emb=get_embeddings(n)
    fake_output=fake_model(emb)
    hate_output=hate_model(emb)
    senti_output=sentiment_model(emb)

    results={
        'Fake':fake_output,
        'Hate':hate_output,
        'Sentiment':senti_output
    }
    return render_template('result.html',fake=fake_output,hate=hate_output,senti=senti_output)

    

if __name__ == "__main__":
    app.run(debug=True)