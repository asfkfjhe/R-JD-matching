from fastapi import FastAPI, UploadFile, File
from joblib import load
from pydantic import BaseModel
import pandas as pd
from PyPDF2 import PdfReader
import yake



app = FastAPI()

# Load the trained model
clf = load('kneighbors_classifier.joblib')
word_vectorizer = load('tfidf_vectorizer.joblib')
le = load('label_encoder.joblib')


class Resume(BaseModel):
    text: str

import re

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def extract_skills_keywords(text):
    keyword_extractor = yake.KeywordExtractor()

    skill_keywords_extracted = keyword_extractor.extract_keywords(text)

    skill_keywords = [keyword for keyword, score in skill_keywords_extracted]

    return skill_keywords


@app.post("/predict/")
async def predict_category(file: UploadFile = File(...)):
    # Extract text from PDF file
    pdf_text = extract_text_from_pdf(file.file)
    # Clean the extracted text
    cleaned_resume = cleanResume(pdf_text)

    skills = extract_skills_keywords(cleaned_resume)
    skills_series = pd.Series(skills)


    
    # Transform the cleaned text into features using the TF-IDF vectorizer
    word_vector = word_vectorizer.transform(skills_series)
    
    # Make predictions using the loaded classifier model
    prediction = clf.predict(word_vector)
    predicted_category = le.inverse_transform(prediction)[0]
    
    return {"category": predicted_category}