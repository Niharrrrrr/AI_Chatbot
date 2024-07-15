from flask import Flask, request, jsonify, render_template
import google.generativeai as generativeai
import PyPDF2
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

generativeai.configure(api_key="AIzaSyD2kOL6PG2V4zYKWZqsNBZ0f-Y72Zu_2i0")

pdf_file = 'Corpus.pdf'
corpus_text = ''
with open(pdf_file, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        corpus_text += page.extract_text()

conversation_history = ["AI: Hi, how can I help you today?"]


stop_words = set(stopwords.words('english'))
corpus_tokens = word_tokenize(corpus_text)
corpus_tokens = [word.lower() for word in corpus_tokens if word.isalnum() and word.lower() not in stop_words]
corpus_text_cleaned = ' '.join(corpus_tokens)

vectorizer = TfidfVectorizer()
corpus_vector = vectorizer.fit_transform([corpus_text_cleaned])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response = generate_response(user_input)
    return jsonify({"response": response})

def generate_response(user_input):
    global conversation_history

    conversation_history.append(f"User: {user_input}")
    
    conversation_context = "\n".join(conversation_history)
    prompt = f"Context: {corpus_text}\nConversation so far:\n{conversation_context}\nAI:"
    model = generativeai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    bot_response = response.text.strip() 

    # If the response is out of corpus, provide a fallback response
    if is_out_of_corpus(bot_response):
        bot_response = "Please contact Jessup Cellars directly for more information."

    bot_response = bot_response.replace("**", "").strip()
    
    conversation_history.append(f"AI: {bot_response}")

    return bot_response

def is_out_of_corpus(response):
    response_tokens = word_tokenize(response)
    response_tokens = [word.lower() for word in response_tokens if word.isalnum() and word.lower() not in stop_words]
    response_text_cleaned = ' '.join(response_tokens)
    response_vector = vectorizer.transform([response_text_cleaned])

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(corpus_vector, response_vector).flatten()[0]

    threshold = 0.2  
    return similarity < threshold

if __name__ == '__main__':
    app.run(debug=True)
