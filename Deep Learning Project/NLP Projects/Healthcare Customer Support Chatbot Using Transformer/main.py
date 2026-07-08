from fastapi import FastAPI, Request, Jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

app = FastAPI()

model = T5ForConditionalGeneration.from_pretrained("./chatbot_model")
tokenizer = T5Tokenizer.from_pretrained("./chatbot_model")

device = model.device

def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'<.*?>', '', text) 
    text = text.strip().lower()  
    return text

def chatbot(dialogue):
    dialogue = clean_text(dialogue) 
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=250)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs["input_ids"],
        max_length=250,
        num_beams=4,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/chat")
def chat():
    user_message = Request.json.get("message", "")
    if not user_message:
        return Jsonify({"error": "Message is Required"}), 400
    response = chatbot(user_message)
    return Jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)