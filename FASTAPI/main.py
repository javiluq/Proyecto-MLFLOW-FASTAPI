from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
from transformers import pipeline

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/Saluda")
def root(): 
    return {"Message": "Hola, este es mi primer proyecto con FastApi!"}

#############################################################


@app.post("/operacion_matematica")
def realizar_operacion(input: TextInput):
    try:
        resultado = eval(input.text, {"__builtins__": {}}, {})
        return {"resultado": resultado}
    except Exception as e:
        return {"error": str(e)}



################################################################

sentiment_analysis = pipeline("sentiment-analysis")

@app.post("/sentimiento")
def analyze_sentiment(input: TextInput):
    result = sentiment_analysis(input.text)
    return {"Sentiment": result[0]['label'], "Score": result[0]['score']}


################################################################

classifier = pipeline("zero-shot-classification")

options = ["política", "entretenimiento", "comida", "viajes"]

@app.post("/clasificador")
def classify_text(input: TextInput):
    result = classifier(input.text, candidate_labels=options)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return {
        "text": input.text,
        "categoria": top_label,
        "confianza": top_score
    }

#################################################################


text_generator = pipeline("text-generation", model="gpt2")

@app.post("/generate-text")
async def generate_text(prompt: str):
    generated_text = text_generator(prompt, max_length=100)
    return {"generated_text": generated_text[0]['generated_text']}

################################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)