import nltk 
from textblob import TextBlob
checked = True
while checked:
    text = request.args.get('msg')
    obj = TextBlob(text)
    sentiment, subjectivity = obj.sentiment
    print(obj.sentiment)
    if sentiment > 0:
        count = count+1
    elif sentiment < 0:
        count = count-1
    else:
        count = count+0

