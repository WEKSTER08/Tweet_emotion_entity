from flask import Flask, render_template, request
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        emotion_dict = {'Negative emotion': 0, 'Positive emotion': 1, 'No emotion toward brand or product': 2, "I can't tell": 3} 
        brand_dict = {'iPhone': 0, 'iPad or iPhone App': 1, 'iPad': 2, 'Google': 3, 'Android': 4, 'Apple': 5, 'Android App': 6, 'Other Google product or service': 7, 'Other Apple product or service': 8}
        def get_key(val,dct):
            for key, value in dct.items():
                if val == value:
                    return key
    
            return "key doesn't exist"
        def get_values(tweet_text, emotion_dict, brand_dict):
            emo_model = RobertaForSequenceClassification.from_pretrained('Emotion_model')  
            brand_model = RobertaForSequenceClassification.from_pretrained('Brand_model')  

            input_ids = tokenizer.encode(tweet_text, return_tensors='pt', truncation=True, padding=True)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                emo_outputs = emo_model(input_ids, attention_mask=attention_mask)
                brand_outputs = brand_model(input_ids, attention_mask=attention_mask)
                emotion_logits = emo_outputs.logits
                brand_logits = brand_outputs.logits

            emotion_probs = torch.argmax(emotion_logits, dim=1).squeeze().tolist()
            brand_probs = torch.argmax(brand_logits, dim=1).squeeze().tolist()

            return get_key(emotion_probs, emotion_dict), get_key(brand_probs, brand_dict)

        tweet_text = request.form.get('tweet_text')
        predicted_emotion, predicted_brand = get_values(tweet_text, emotion_dict, brand_dict)

        return render_template('index.html', emotion_prediction=predicted_emotion, brand_prediction=predicted_brand, tweet_text=tweet_text)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
