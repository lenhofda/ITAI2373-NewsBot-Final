# NewsBot Intelligence System 2.0

**ITAI 2373 – Final Project**

## Project Overview

For my final project, I continued working on the NewsBot Intelligence System that I started during the midterm. The goal was to build on what I had already completed and add more advanced natural language processing (NLP) features.

I used the BBC News dataset and Python in Google Colab. My project focuses on finding topics in news articles, summarizing information, searching for related articles, translating text, and allowing users to ask questions through a conversational interface.

## Project Features

### Module 9 – Topic Modeling

I used two topic modeling methods, LDA and NMF, to identify patterns in the news articles.

- LDA identified five topics using word counts.
- NMF identified five topics using TF-IDF.
- I reviewed the most important words for each topic.
- I compared the results to see how well the models identified the news categories.

### Module 10 – Language Models

In this module, I worked with pretrained language models to better understand and summarize news articles.

- Used DistilBART to summarize articles.
- Used SentenceTransformers for semantic search.
- Tested query expansion to improve searches.
- Explored ways to provide additional context for news content.

### Module 11 – Multilingual Processing

I explored how NewsBot could work with articles in different languages.

- Used language detection to identify languages.
- Translated English text into Spanish using MarianMT.
- Compared multilingual sentence embeddings.
- Explored named entities and cultural context.

### Module 12 – Conversational AI

For the final module, I worked on combining the different features into a conversational NewsBot.

- Identified user questions and their intent.
- Parsed questions to understand what users wanted.
- Tested conversation memory.
- Generated responses based on user requests.
- Connected semantic search with the conversational interface.
- Demonstrated summarization and translation.

I completed three functional tests of the conversational interface.

## Technologies Used

I used Python and Google Colab throughout the project, along with several NLP and machine learning libraries.

These included pandas, NumPy, scikit-learn, NLTK, spaCy, Hugging Face Transformers, SentenceTransformers, Matplotlib, and Plotly.

## Project Results

My topic modeling results showed:

- LDA topic label agreement: 90.20%
- NMF topic label agreement: 90.54%
- Multilingual embedding cosine similarity: 0.8928
- Three conversational interface tests completed successfully

The topic label agreement results measure how closely the discovered topics matched the news categories in my evaluation. They do not represent the overall accuracy of NewsBot.

## GitHub Repository Organization

I organized my GitHub repository into folders to make the project easier to follow.

- `notebooks/` – My completed Google Colab notebook
- `src/data_processing/` – Text cleaning and preprocessing
- `src/analysis/` – LDA and NMF topic modeling
- `src/language_models/` – Language model components
- `src/multilingual/` – Multilingual processing
- `src/conversation/` – Conversational interface
- `src/utils/` – Utility functions
- `config/` – Configuration
- `tests/` – Testing
- `data/` – Dataset information
- `docs/` – Technical documentation
- `reports/` – Project reports

The main project code is in my Google Colab notebook. I have also started organizing the code into separate Python files in GitHub.

## How to Run the Project

1. Open the NewsBot notebook from the `notebooks` folder in Google Colab.
2. Install the required Python libraries listed in `requirements.txt`.
3. Make sure the BBC News dataset is available.
4. Run the notebook cells in order.
5. Review the results for topic modeling, summarization, multilingual processing, and conversational AI.

## Challenges and Limitations

One challenge was bringing together the different NLP techniques into one system. Each module worked on a different part of the problem, and I needed to understand how they could work together.

The project is still a prototype. Some summarization and translation examples use a selected article rather than automatically following every user search. The separate Python modules also need additional testing.

## Future Improvements

If I continued developing NewsBot, I would focus on connecting all the features more completely, improving the conversational interface, adding automated testing, and making the system easier for someone else to use.

## Conclusion

This project helped me better understand how different NLP methods can work together. I started with the NewsBot system from my midterm and expanded it to include topic modeling, language models, multilingual processing, and conversational AI.

I also gained more experience using Google Colab, organizing Python code, and preparing a GitHub repository for a larger project.
