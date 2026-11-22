# NewsBot Technical Documentation
## Project Overview

For my ITAI 2373 final project, I expanded the NewsBot Intelligence System from my midterm. I used Python and Google Colab to build and test additional natural language processing (NLP) features using the BBC News dataset.

The main goal was to combine several NLP methods into a system that could analyze news articles, identify topics, summarize information, translate text, and respond to user questions.

## System Architecture

NewsBot is organized into four main components.

### 1. Advanced Content Analysis – Module 9

I used LDA and NMF topic modeling to identify common themes in the BBC News articles.

Both models were configured to identify five topics. I compared the most important words and evaluated how closely the discovered topics matched the news categories.

The LDA topic label agreement was 90.20%, while NMF achieved 90.54%.

### 2. Language Understanding – Module 10

I used pretrained language models to summarize and search news articles.

DistilBART was used for summarization, and SentenceTransformers was used to create embeddings for semantic search.

I also explored query expansion and content enhancement.

### 3. Multilingual Processing – Module 11

I tested language detection, English-to-Spanish translation, and multilingual sentence embeddings.

I used MarianMT for translation and explored how named entities and cultural context could affect multilingual analysis.

### 4. Conversational Interface – Module 12

I developed a prototype conversational interface to demonstrate how users could interact with NewsBot.

The interface includes intent classification, query parsing, response generation, and a conversation memory demonstration.

I also connected semantic search to the interface and tested summarization and translation features.

Three functional tests were completed successfully.

## Data Processing

The project uses the BBC News dataset.

I cleaned and prepared article text before applying NLP methods. The preprocessing included removing unnecessary characters, tokenizing words, removing stopwords, and lemmatizing text.

The preprocessing functions are also being organized into the `src/data_processing/` folder.

## Software and Dependencies

The project was developed using Python in Google Colab.

The main libraries include:

- pandas and NumPy for data processing
- scikit-learn for topic modeling
- NLTK and spaCy for text processing
- Hugging Face Transformers for language models
- SentenceTransformers for semantic search
- langdetect for language identification
- Matplotlib and Plotly for visualization

The repository includes a `requirements.txt` file listing the main Python packages.

## Installation and Running the Project

1. Open the notebook in the GitHub `notebooks/` folder using Google Colab.
2. Install the required Python packages.
3. Make the BBC News dataset available to the notebook.
4. Run the notebook cells in sequence.
5. Review the results and visualizations for each module.

The Google Colab notebook is currently the primary implementation of NewsBot.

## Configuration

The project uses model settings defined within the Colab notebook, including the number of topics and other parameters.

The repository contains a `config/` folder for organizing configuration files as the project develops.

## Main Functions and Components

The project includes the following functions and components:

- `clean_text(text)` – Cleans article text before processing.
- `preprocess_text(text, remove_stopwords=True, lemmatize=True)` – Tokenizes and prepares text for analysis.
- `run_lda(df)` – Runs LDA topic modeling.
- `run_nmf(df)` – Runs NMF topic modeling.
- DistilBART – Generates article summaries.
- SentenceTransformers – Creates embeddings for semantic search.
- MarianMT – Translates English text into Spanish.

The first four functions have also been organized into separate Python source files. The language model and conversational components are primarily demonstrated in the notebook.

## Testing and Evaluation

I compared LDA and NMF topic modeling results and reviewed the generated topics.

I also tested article summarization, multilingual embeddings, and conversational responses.

The multilingual embedding comparison produced a cosine similarity of 0.8928 in the example tested.

Three conversational interface functional tests completed successfully.

## Challenges and Limitations

One challenge was integrating different NLP methods into one project. Each module had its own requirements, and combining them required additional testing.

NewsBot is currently a prototype. Some summarization and translation demonstrations use a predefined article rather than automatically using the article selected by a user.

The separate Python source modules also require additional development and testing before they can operate as a complete standalone application.

## Future Improvements

I would like to improve the integration between search, summarization, translation, and the conversational interface.

Other improvements include automated testing, more language support, better configuration management, and a user-friendly interface.

## Conclusion

This project helped me understand how several NLP methods can work together to analyze news articles. It also gave me experience organizing a larger Python project and documenting the work in GitHub.
