from transformers import pipeline
from fastapi import FastAPI, Response
from pydantic import BaseModel

generator = pipeline('text-classification')

app = FastAPI()


class Body(BaseModel):
    text: str


@app.get('/')
def root():
    return Response('A self-documenting API to interact with a text classification')


@app.post('/generate')
def predict(body: Body):
    result = generator(body.text)
    return result
