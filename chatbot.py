from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import random
import json  
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model





# Token del bot
bot_token = "7529046823:AAG9kWyNZZlmSPDt4YsPz38SOMqQoFPfsKw"

lemmatizer = WordNetLemmatizer()

# Cargar intents.json
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Cargar el modelo y otros archivos necesarios
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence = sentence.lower()  # Convertir la oración a minúsculas
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
 
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    responses = []
    for intent in list_of_intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            break
    result = random.choice(responses) if responses else "soy joancito "
    return result

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    message = message.lower()  # Convertir el mensaje a minúsculas
    ints = predict_class(message)
    res = get_response(ints, intents)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=res)

def main():
    application = Application.builder().token(bot_token).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
    

if __name__ == '__main__':
    main()