from fastapi import FastAPI, UploadFile, File, Form
from PyPDF2 import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import yake
import torch
import nltk
from joblib import load
from pydantic import BaseModel
import pandas as pd




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
# job_descriptions_df = pd.read_csv("job_descriptions.csv")
# job_descriptions = job_descriptions_df["Job Description"].tolist()



def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text

nltk.download('stopwords')
 
SKILLS_DB = [
    'machine learning',
    'data science',
    'python',
    'word',
    'excel',
    'english',
    'react js', 
    'node.js', 
    'nest js',
    'express', 
    'nest.js',  
    'git', 
    'postgresql',
    'firebase', 
    'redis', 
    'mongodb', 
    'docker', 
    'cloud services', 
    'chakra ui',
]

 
 
def extract_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)
 
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
 
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
 
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
 
    # we create a set to keep the results in.
    found_skills = set()
 
    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)
 
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)
 
    return found_skills
 
def extract_skills_keywords(text):
    keyword_extractor = yake.KeywordExtractor()

    skill_keywords_extracted = keyword_extractor.extract_keywords(text)

    skill_keywords = [keyword for keyword, score in skill_keywords_extracted]

    return skill_keywords
 

from transformers import DistilBertTokenizer, DistilBertModel

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to preprocess and get embeddings for text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()  # Convert to numpy array

@app.post("/calculate_similarity/")
async def calculate_similarity(job_description: str = Form(...), resume: UploadFile = File(...)):
    try:
        resume_text = extract_text_from_pdf(resume.file)
        
        preprocessed_job_description = preprocess_text(job_description)
        preprocessed_resume = preprocess_text(resume_text)
        
        job_description_keywords = list(extract_skills_keywords(preprocessed_job_description))
        resume_keywords = list(extract_skills_keywords(preprocessed_resume))
        
        # Check if either job description or resume keywords are empty
        if not job_description_keywords or not resume_keywords:
            return {"error": "No skills found in the input text."}

        job_description_embeddings = get_embeddings(job_description_keywords)
        resume_embeddings = get_embeddings(resume_keywords)

        similarity_score = cosine_similarity(job_description_embeddings, resume_embeddings)[0][0]
        
        return {"similarity_score": float(similarity_score)}  # Convert to Python float
        
    except Exception as e:
        return {"error": str(e)}

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