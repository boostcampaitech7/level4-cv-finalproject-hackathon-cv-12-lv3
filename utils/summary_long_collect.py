from .cited_by_extraction import get_cited_by_papers
from .summary_short import extractive_summarization, abstractive_summarization


def summarize_long_collect(text, title, image, table):
    # Extract relevant information using related work extraction techniques
    # This could involve named entity recognition, topic modeling, or other methods
    extractive_summary = extractive_summarization(text)
    abstractive_summary = abstractive_summarization(extractive_summary)
    # Extract related works from the text
    cited_by_papers = get_cited_by_papers(title)
    related_works = [paper.get('title') for paper in cited_by_papers]
    # Create a timeline image based on the reference provided

    return abstractive_summary, related_works, image, table
