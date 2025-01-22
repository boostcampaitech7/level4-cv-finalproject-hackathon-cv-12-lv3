import pytextrank
import spacy
import requests

import sys
import os
sys.path.insert(0, os.path.abspath('/Users/haneol/Documents/Coding/level4-cv-finalproject-hackathon-cv-12-lv3/'))
from api.api_classes import ChatCompletionAPI
from config.config import API_CONFIG

def extractive_summarization(preprocessed_text, lang):
    if lang=="en":
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    elif lang=="ko":
        spacy.cli.download("ko_core_news_sm")
        nlp = spacy.load("ko_core_news_sm")
    nlp.add_pipe("textrank")
    doc = nlp(preprocessed_text)
    extracted_sentences = []
    tr = doc._.textrank
    for sent in tr.summary(limit_phrases=50, limit_sentences=30):
        extracted_sentences.append(sent.text)
    return extracted_sentences


def abstractive_summarization(extracted_sentences):
    extracted_text = " ".join(extracted_sentences)

    # chat_api = ChatCompletionAPI(
    #     host = API_CONFIG["host"],
    #     api_key = API_CONFIG["api_key"],
    #     request_id=API_CONFIG["request_id"],
    # )
    chat_api = ChatCompletionAPI(
        host='clovastudio.stream.ntruss.com',
        api_key='Bearer nv-b9d7d7c3b9374204a2b486e3dcc18a9aER2R',
        request_id='3e9deb64e18447d48d8ee7562c8eb4f6'
    )
    messages = [
        {"role": "system", "content": 
"""
You are a highly skilled AI research assistant tasked with summarizing academic papers for researchers. Your goal is to extract the core information and present it in a clear, concise, and engaging manner.

For the given research paper, please provide the following:
1. Catchy Summary: Condense the paper's main contribution into a single, eye-catching sentence with less than 200 characters. This should be easily understandable and pique the reader's interest.
2. TL;DR:  Provide a comprehensive summary of the paper in 2 paragraphs. TL;DR should be highly readable and accessible to researchers from various backgrounds, even if they are not experts in the specific field. Focus on clarity and avoid technical jargon as much as possible. Explain key concepts, methods, and findings in a way that is easy to grasp. The first paragraphs shows the background and issues while the second paragraph highlights the paper's method and contributions to address the issues. Each paragraph should be written in 500 characters. 
3. Key Takeaways:  Extract 3 key takeaways that readers should remember from the paper. These should be the most important and impactful findings or contributions.
4. Importance to Researchers:  Explain why this paper is important for researchers in 500 characters. Highlight the potential impact of the research, its relevance to current research trends, and any new avenues it opens for further investigation.

Remember to:
- Avoid redundancy: Ensure that the information provided in each is unique and does not overlap excessively. (i.e. if you already mentioned the project name from the TL;DR, do not mention it again in the other sections)
- Focus on the main idea: Prioritize the core contributions and findings of the paper, ensuring that readers can grasp the main idea effectively.
- Maintain a professional and objective tone: Present the information in a neutral and unbiased manner.
- Use **bold** to highlight the important parts of the text in the sections "TL;DR" and "Importance to Researchers.
- Answer in Korean.
"""
},
        {"role": "user", "content": extracted_text}
    ]
    res = chat_api.get_completion(messages)
    return res

if __name__ == "__main__":
    # Example Usage (replace with your actual preprocessed text)
    from pathlib import Path
    file_path = Path('/Users/haneol/Documents/Coding/multilevel-summary.txt')
    preprocessed_text = file_path.read_text()
    print(preprocessed_text)
    extracted_summary = extractive_summarization(preprocessed_text, lang="en")
    abstractive_summary = abstractive_summarization(extracted_summary)

    if abstractive_summary:
        print("Extractive Summary:")
        print("\n".join(extracted_summary))
        print("\nAbstractive Summary:")
        print(abstractive_summary)
