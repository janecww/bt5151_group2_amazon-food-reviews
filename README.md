# Roadmap of Codespace for BT5151 Group 2 Project Proposal 
All the software code and data are included in this folder. Please find the following directions for navigating through the code and data files.

## Code
### Overall Sentiment Models
All the code files that are used to implement sentiment analysis models are included in this folder.

- `Baseline_Models_and_Distilbert.ipynb` : Includes EDAs of the dataset, data preprocessing with TfidfVectorizer, baseline models (i.e., Logistic Regression, Naive Bayes Multinomial), Distilbert Transformers
- ` Distilroberta Model.ipynb`: Includes the distilled version of the RoBERTa-base model
- `Multi-head attention.ipynb`: Includes the multi-head attention transformer model using Keras package. Since it is our best model, shap value analysis is also conducted for the model result
- `Roberta_Model.ipynb`: Includes the RoBERTa model
- `Tokenizer as Feature Extractor.ipynb`: Includes data preprocessing with distilbert tokenizer and a logistic regression model
- `distilbertroberta_tokenizer.py`: Includes the exclusive tokenizer for the distilbertroberta model
- `roberta_tokenizer.py`: Includes the exclusive tokenizer for the roberta model
  
### Aspect-based Sentiment Analysis
This folder contains the code work for aspect-based sentiment analysis

- `PyABSA.ipynb`: Includes the aspect-based sentiment analysis performed by PyABSA package
- `Spacy_Aspect_Classifier.ipynb`: Includes the aspect-based sentiment analysis performed by Spacy Aspect Classifier



## Data
- `
