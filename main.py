import json

import pyaudio
import speech_recognition as sr
import pyttsx3
from flask_cors import CORS

from google.cloud import texttospeech
import pygame
from datetime import date, datetime
from flask import Flask, request, jsonify

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import find_dotenv, load_dotenv
from langchain_google_vertexai import ChatVertexAI
from google.cloud import texttospeech

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

app = Flask(__name__)
cors = CORS(app, origins=["*"])

@app.route('/literature', methods=['POST'])
def literatureHelper():
  question = request.json['question']
  if question:
    output_parser = StrOutputParser()
    llm = ChatVertexAI(model="gemini-pro")
    prompt = ChatPromptTemplate.from_messages([
      ("system", f"Apartir de agora, voce é a Kate. Voce foi criada pelo Alexandre Diano. Sempre responda como um autor literario renomado. Sempre Baseie as suas respostas a artigos cientificos. Sempre Responda tudo com clareza e com exemplos. Sempre utilize respostas atualizadas. Sempre se inspire em grandes autores da literature mundial, e grandes professores de portugues."),
      ("user", "{input}")
    ])
    chain = prompt | llm | output_parser
    answer = chain.invoke({"input": question})

    return answer
  else:
    return jsonify({'error': 'invalid question'}), 400


@app.route('/medicine', methods=['POST'])
def medicineHelper():
  question = request.json['question']
  if question:
    output_parser = StrOutputParser()
    llm = ChatVertexAI(model="gemini-pro")
    prompt = ChatPromptTemplate.from_messages([
      ("system", f"Apartir de agora, voce é a Kate. Voce foi criada pelo Alexandre Diano. Sempre responda como um medico renomado. Sempre Baseie as suas respostas a artigos cientificos. Sempre Responda tudo com clareza e com exemplos. Sempre utilize respostas atualizadas."),
      ("user", "{input}")
    ])
    chain = prompt | llm | output_parser
    answer = chain.invoke({"input": question})

    return answer
  else:
    return jsonify({'error': 'invalid question'}), 400


if __name__ == "__main__":
  app.run(debug=True)
