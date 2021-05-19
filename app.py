from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import regex
import joblib



app = Flask(__name__)

def clean_text(text):
    text = regex.sub("[^a-zA-Z]", " ", text)
    text = regex.sub(' +', ' ', text)
    text = regex.sub(r"\b[b]\b", "", text)
    return text

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv(r'Combined_News_DJIA.csv')
    data['Top23'].fillna(data['Top23'].median,inplace=True)
    data['Top24'].fillna(data['Top24'].median,inplace=True)
    data['Top25'].fillna(data['Top25'].median,inplace=True)

    def create_df(dataset):
        
        dataset = dataset.drop(columns=['Date', 'Label'])
        dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
        for col in dataset.columns:
            dataset[col] = dataset[col].str.lower()
            
        headlines = []
        for row in range(0, len(dataset.index)):
            headlines.append(' '.join(str(x) for x in dataset.iloc[row, 0:25]))
            
        df = pd.DataFrame(headlines, columns=['headlines'])
        df['label'] = data.Label
        df['date'] = data.Date
        
        return df

    def clean_text(text):
        text = regex.sub("[^a-zA-Z]", " ", text)
        text = regex.sub(' +', ' ', text)
        text = regex.sub(r"\b[b]\b", "", text)
        return text

    df = create_df(data)

    X = df.headlines

    X = X.apply(clean_text)

   def tokenize(text):
        text = regex.sub(r'[^\w\s]','',text)
        special_characters = ['!','"','#','$','%','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t']
        for i in special_characters : 
            text = text.replace(i, '')
        tokens = text.split()

        return tokens



    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words = 'english',ngram_range=(2,2))),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(n_estimators=100))])


    train = df[df['date'] < '20150101']
    test = df[df['date'] > '20141231']

    x_train = train.headlines
    y_train = train.label
    x_test = test.headlines
    y_test = test.label

    pipeline.fit(x_train, y_train)

    pipeline.score(x_test,y_test)

    #best_model = joblib.load("another_best.pkl")

    if request.method =='POST':
        message = request.form['message']

        message = clean_text(message)

        data = [message]

        p = pipeline.predict(data)


    return render_template('home.html',prediction = p)


if __name__ == '__main__':
    app.run(debug=True)
