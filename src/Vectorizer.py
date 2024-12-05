# Import required packages

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
wordnet = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
# create the transform
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
from nltk.probability import FreqDist
from nltk import ngrams

def cleansing(text, wordnet):
    
    '''To clean the documents'''
    
    # Remove space after "IFRS" if next string is a number
    text_ifrs = re.sub(r'(IFRS)\s+(\d+)', r'\1\2', text)
    # Remove space after "IAS" if next string is a number
    text_ias= re.sub(r'(IAS)\s+(\d+)', r'\1\2', text_ifrs)
    # Remove squared brackets and text between them
    text_no_square = re.sub(r'\[[^\]]*\]', '', text_ias)
    # Remove round brackets and text between them
    text_no_round = re.sub(r'\([^\)]*\)', '', text_no_square)
    #Convert to lower case
    text_lower_case = text_no_round.lower()
    # Remove punctuations
    text_no_punctuation = re.sub(r'[^\w\s]', '', text_lower_case)
    # Remove whitespace
    text_no_wspace = text_no_punctuation.strip()
    # Remove numbers
    text_no_num = re.sub(r'\s\d+\s', '', text_no_wspace)
    # Remove text with single character
    text_non_single = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text_no_num)
    # word tokenize
    text_tokenize = word_tokenize(text_non_single)
    # (1) Cleaning text of stopwords and (2) lemmatization
    text_no_stop = []
    for token in text_tokenize:
        # if tokenized string is not stop words, append lemmatized text
        if not token in (stopwords.words('english') + ['shall']):
            text_no_stop.append(wordnet.lemmatize(token))
                   
    # rearrange cleaned words according to the sentance they belong to
    text_processed = ' '.join(text_no_stop)
    #Remove the word 'paragraph'
    text_processed = re.sub(r'\b{}\b'.format('paragraph'), '', text_processed)
    
    return text_processed

def munti_gram(cleaned_txt, gram):
    '''Generate bag of words for n-gram analysis'''
    mulgram_lst = []
    
    for sentence in cleaned_txt:
        mulgram = list(ngrams(sentence.split(), gram))
        mulgram_lst_temp = []
        mulgram_lst_temp_pos = []
        
        for item in mulgram:
            mulgram_lst_temp.append(' '.join(item))
            mulgram_lst_temp_pos.append(list(item))
        mulgram_lst.append(mulgram_lst_temp)
        
    mulgram_lst_ttl = []
    for i in mulgram_lst:
        mulgram_lst_ttl += i
        
    return mulgram_lst_ttl

def noun_phrase (mulgram_lst_ttl_pos):
    '''Identify noun phrases based on Part of Speech (PoS) tagging'''
    tagged_words = nltk.pos_tag(mulgram_lst_ttl_pos)
        
    #Extract Noun Phrase from text :
    grammar = "NP:{<DT>?<JJ>*<NN>}" 
    #Creating a parser :
    parser = nltk.RegexpParser(grammar)
    #Parsing text :
    output = parser.parse(tagged_words)
    
    return output

def vectorizer (cleaned_txt):
    '''Vectorizes the statements'''
    # create the transform
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    # Tokenize and build vocab
    vectorizer.fit(cleaned_txt)
                   
    # Vectorize each paragraph in the text list
    vector_list = [(vectorizer.transform([x])).toarray() for x in cleaned_txt]
    # Result of vectorization
    return vector_list

# Extract files to be prepared
ready_ratio = pd.read_csv('data/processed_data/ready_ratio.csv')
accountingcoach = pd.read_csv('data/processed_data/accountingcoach.csv')
IFRS = pd.read_csv('data/processed_data/IFRS.csv')

# Generate the dictionsry of dataframes
file_dict = {'ready_ratio': ready_ratio, 'accountingcoach': accountingcoach, 'IFRS': IFRS}

# Extract questions and answer for preparation
for name, table in file_dict.items():
    question_lst = list(table['question'])
    answer_lst = list(table['answer'])
    res_file_cleanse = pd.DataFrame()
    res_file_vectorized = pd.DataFrame()
    
    qa_dict = {'question': question_lst, 'answer': answer_lst}
    
    # Prepare data for further analysis by columns
    for record, details in qa_dict.items():
        # Cleanse data
        cleaned_rcord = [cleansing(str(text), wordnet) for text in details]
        res_file_cleanse[record] = cleaned_rcord
        
        # Perform frequency analysis
        monogram_lst_ttl = munti_gram(cleaned_rcord, 1)
        bigram_lst_ttl = munti_gram(cleaned_rcord, 2)
        trigram_lst_ttl = munti_gram(cleaned_rcord, 3)

        dist_dict = {'monogram':FreqDist(monogram_lst_ttl), 'bigram':FreqDist(bigram_lst_ttl), 'trigram':FreqDist(trigram_lst_ttl)}
        for key, val in dist_dict.items():
            print(val.most_common(5))
            val.plot(20)
        
        # Vectorize string
        vec_lst = [str(list(x[0])) for x in vectorizer (cleaned_rcord)]
        res_file_vectorized[record] = vec_lst
    
    # Upload cleansed and vectorized data in the format of csv files 
    res_file_cleanse.to_csv(f"data/processed_data/{name}_cleansed.csv")
    json_string = res_file_vectorized.to_json()
    with open(f"data/processed_data/{name}_vectorized.txt", 'w') as f:
        f.write(str(json_string))
        
        f.close
    
    # Perform Part od Speech (PoS) tagging
    noun_phrase = noun_phrase (monogram_lst_ttl)
    
    # Upload noun phrases with PoS tagging as txt file
    with open(f"data/processed_data/{name}_pos.txt", 'w') as f:
        f.write(str(noun_phrase))
        
        f.close
