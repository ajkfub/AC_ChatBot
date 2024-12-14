# Accounting ChatBot
---

# Accounting Chatbot using Llama3

## Overview
---
This project fine-tunes the Llama3 model to create a specialized chatbot for accounting inquiries. Using a dataset of responses from accounting professionals, the model is trained to generate accurate and contextually relevant answers.

The selected open-source model is Llama3, which leverages advanced architectures for enhanced language understanding. Through fine-tuning, the model significantly improves its grasp of accounting terminology and user queries, delivering clear and informative responses.

The fine-tuned model is available on Hugging Face: https://huggingface.co/SamHunghaha/AC_ChatBot
### Project Motivation
This project leverages cutting-edge large language models and the latest fine-tuning techniques to create a customized chatbot trained on a dataset of accounting-related dialogues and inquiries. Designed specifically for accounting professionals, it delivers detailed and nuanced content tailored to the needs of the field.

### Features

- **Natural Language Processing**: Leverage the capabilities of the Llama3 model for understanding and generating human-like responses.
- **Accounting-Specific Knowledge**: The chatbot is fine-tuned with accounting data to ensure accuracy and relevance in responses.
- **Data Visualization**: Integration with data visualization tools to present financial data interactively.

## Operation Principles
---

### Fine-tuning
Large pre-trained language models possess fundamental capabilities for generating human-like responses. By fine-tuning these models with specific textual data, they can enhance their ability to mimic various aspects such as tone, style, information, and word usage. It is important to note that fine-tuning does not equip the model with language abilities from scratch; rather, it deepens its understanding of localized textual information and patterns based on its original pre-trained capabilities.

### Data Sources

We scrape data from the following websites:

- [IFRS](https://www.ifrs.org/)
- [AccountingCoach](https://www.accountingcoach.com/)
- [ReadyRatios](https://www.readyratios.com/)

### Data Processing

The collected data consists of informative textual content. Our goal is to transform this data into prompt and response pairs suitable for fine-tuning.

### Methodology

To facilitate this transformation, we leverage the [T5 Base Question Generator](https://huggingface.co/iarfmoose/t5-base-question-generator) available on Hugging Face. This question generator model helps us formulate a dataset specifically tailored for our fine-tuning process.

### Base Model
Meta developed and released the Meta Llama 3 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks. Further, in developing these models, we took great care to optimize helpfulness and safety.

## File Structure
---
- src/ : Python code
    - DataCleaner.py: Contains functions for cleaning and preprocessing raw data to ensure quality and consistency before finetuning.
    - DataGenerator.py: Implements logic to generate prompt and response pairs from cleaned data for model fine-tuning.
    - DataVisualizer.py: Provides visualization tools to analyze and present data insights through various graphical representations.
    - Extractor.py: Extracts and web-scrap text data from various website.
    - ModelFinetune.py: Contains the code for fine-tuning the LLM model using the generated dataset of prompt and response pairs.
    - Prediction.py: Implements functionality for making predictions on the next question based on the fine-tuned model and input data.
    - Vectorizer.py: Provides methods for transforming textual data into numerical representations suitable for model training and predictions.
    - run.py: Contains scripts to execute the overall workflow of the project, integrating all functionalities from data extraction to prediction.
- data/ : Raw data obtained from data fetching, stored as .csv or .json
- dataset/ : Processed training data

## Usage Instructions
--- 
### Hardware Requirements
This project utilizes the A100 GPU in Google Colab with > 200 Ram. The local machine requires about 100GB of RAM for smooth inference.

### Environment Setup
Run the following shell script to set up and configure the environment using Anaconda and download all necessary dependencies according to requirements.txt.

```
conda create -n ac_chatbot python=3.9
conda activate ac_chatbot
pip install -r requirements.txt
```

### Running ac_chatbot

```
# Mode - Prompt
python src/run.py --mode "prompt" --prompt "What is Net Income"
# Mode - Data
python src/run.py --mode "data" --stock_code "AAPL" --item "Net Income" --freq "A"
```