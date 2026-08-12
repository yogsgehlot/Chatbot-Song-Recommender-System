from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from textblob import TextBlob
import webbrowser
count = 0

app = Flask(__name__)

english_bot = ChatBot("ChatterBot", storage_adaptor="chatterbot.storage.SQLStorageAdaptor")
trainer = ChatterBotCorpusTrainer(english_bot)

trainer.train("chatterbot.corpus.english")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global count
    userText = request.args.get('msg')
    obj = TextBlob(userText)
    sentiment, subjectivity = obj.sentiment
    print(obj.sentiment)
    if sentiment > 0:
        count = count + 1
    elif sentiment < 0:
        count = count - 1
    else:
        count = count + 0
    print("count", count)

    return str(english_bot.get_response(userText))


@app.route("/forward/", methods=['POST'])
def get_song_playlist():
    global count
    if count > 0:
        webbrowser.open_new("https://wynk.in/music/playlist/desi-hip-hop-hits/bb_1491457072469?q=hip+hop")
    elif count < 0:
        webbrowser.open_new("https://wynk.in/music/playlist/arijit-all-the-way/QVVESU9fQXJpaml0IFNpbmdoIC0gQWxsIFRoZSBXYXlfMjAxNF9zcmNoX2JzYl9oaQ?q=+")
    else:
        webbrowser.open_new("https://wynk.in/music/playlist/motivational-tunes-hindi/bb_1588754763579?q=motivation")


if __name__ == '__main__':
    app.run()
