import os
import math
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from natsort import natsorted
import nltk

nltk.download('punkt')

# First phase


# Read 10 files (.txt)  >>>>>>>>>>>
documents = []

file_names = natsorted([file for file in os.listdir("files") if file.endswith('.txt')])


def read_files(file):
    if 'txt' in file:
        with open(f'files/' + file, 'r') as f:
            return f.read()


print("read 10 files.txt :\n")
print(file_names, "\n")


####Apply tokenization & stemmeing
def preprocess_document(doc):
    tokenized_doc = word_tokenize(doc)

    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    # Apply stemming to each token
    stemmed_tokens = [stemmer.stem(token) for token in tokenized_doc]

    return stemmed_tokens


# Read files and preprocess each document

for file in natsorted(os.listdir('files')):
    content = read_files(file)
    preprocessed_doc = preprocess_document(content)

    documents.append(preprocessed_doc)

# Display the preprocessed documents>>>>>
print("tokenization & stemming :\n")
for doc in documents:
    print(", ".join(doc))
print("\n")

# second phase


fileno = 1

# Initialize the dictionary.
pos_index = {}

# For every file.
for file_name in file_names:
    # Read file contents.
    with open(f'files/{file_name}', 'r') as f:
        doc = f.read()

    # Preprocess doc.
    final_token_list = preprocess_document(doc)

    # For position and term in the tokens.
    for pos, term in enumerate(final_token_list):
        # If term already exists in the positional index dictionary.
        if term in pos_index:

            pos_index[term][0] = pos_index[term][0] + 1

            # Check if the term has existed in that DocID before.
            if fileno in pos_index[term][1]:
                pos_index[term][1][fileno].append(pos)
            else:
                pos_index[term][1][fileno] = [pos]

        # If term does not exist in the positional index dictionary
        else:

            pos_index[term] = []

            pos_index[term].append(1)

            pos_index[term].append({})

            pos_index[term][1][fileno] = [pos]

    fileno += 1

# >>>> print the positional index
print("The posistional index\n")
for term, info in pos_index.items():
    total_frequency, doc_positions = info

    print(f'{term}: [{total_frequency}, {doc_positions}]', )


#######


def put_query(q, display=1):
    lis = [[] for _ in range(10)]
    q = preprocess_document(q)

    for i, term in enumerate(q):
        if term in pos_index.keys():
            for key in pos_index[term][1].keys():
                if lis[key - 1] != []:
                    if lis[key - 1][-1] == pos_index[term][1][key][0] - 1:
                        lis[key - 1].append(pos_index[term][1][key][0])
                    elif i > 0 and lis[key - 1][-1] + 1 == pos_index[term][1][key][0]:
                        lis[key - 1][-1] = pos_index[term][1][key][0]
                else:
                    lis[key - 1].append(pos_index[term][1][key][0])

    positions = []
    if display == 1:
        for pos, lst in enumerate(lis, start=1):
            if len(lst) == len(q):
                positions.append('document ' + str(pos))
        return positions
    else:
        for pos, lst in enumerate(lis, start=1):
            if len(lst) == len(q):
                positions.append('doc' + str(pos))
        return positions


##   [Term frequency]
documents = []
files = os.listdir('files')
for file in range(1, 11):
    documents.append(" ".join(preprocess_document(read_files(str(file) + '.txt'))))

# Get all unique terms in the order they appear in the original documents
all_terms = [term for doc in documents for term in doc.split()]


def get_tf(document):
    wordDict = dict.fromkeys(all_terms, 0)
    for word in document.split():
        wordDict[word] += 1
    return wordDict


tf = pd.DataFrame(get_tf(documents[0]).values(), index=get_tf(documents[0]).keys())
for i in range(1, len(documents)):
    tf[i] = get_tf(documents[i]).values()
tf.columns = ['doc' + str(i) for i in range(1, 11)]

# print the term frequency
print("\n<<<< Term frequency>>>\n")
print(tf, "\n")


# compute W tf(1+ log tf)

def weighted_tf(x):
    if x > 0:
        return math.log10(x) + 1
    return 0


w_tf = tf.copy()
for i in range(0, len(documents)):
    w_tf['doc' + str(i + 1)] = tf['doc' + str(i + 1)].apply(weighted_tf)

## print the idf
print("display the Weighted_tf\n")
print(w_tf, "\n")

# compute tdf.idf

tdf = pd.DataFrame(columns=['df', 'idf'])
for i in range(len(tf)):
    in_term = w_tf.iloc[i].values.sum()

    tdf.loc[i, 'df'] = in_term

    tdf.loc[i, 'idf'] = math.log10(10 / (float(in_term)))

tdf.index = w_tf.index

## print df&idf
print("calculate df&idf\n")
print(tdf, "\n")

# tf*idf


tf_idf = w_tf.multiply(tdf['idf'], axis=0)

print(tf_idf)


# normailzed tf.idf
def get_doc_len(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x ** 2).sum())


doc_len = pd.DataFrame()
for col in tf_idf.columns:
    doc_len.loc[0, col + '_length'] = get_doc_len(col)

##compute document length
print("calculate document length\n")
print(doc_len, "\n")


def get_norm_tf_idf(col, x):
    try:
        return x / doc_len[col + '_length'].values[0]
    except:
        return 0


#### normailzation
norm_tf_idf = pd.DataFrame()
for col in tf_idf.columns:
    norm_tf_idf[col] = tf_idf[col].apply(lambda x: get_norm_tf_idf(col, x))

print("calculate normalized tf.idf \n")
print(norm_tf_idf, "\n")


#
def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0


# insert query >>>>>>>


def insert_query(q):
    docs_found = put_query(q, 2)
    if docs_found == []:
        return "Not Fount"

    new_q = preprocess_document(q)

    query = pd.DataFrame(index=norm_tf_idf.index)
    query['tf'] = [1 if x in new_q else 0 for x in list(norm_tf_idf.index)]
    query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
    product = norm_tf_idf.multiply(query['w_tf'], axis=0)
    query['idf'] = tdf['idf'] * query['w_tf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['normalized'] = 0
    for i in range(len(query)):
        query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values ** 2))
    print('Query Details')
    print(query.loc[new_q])
    product2 = product.multiply(query['normalized'], axis=0)
    # Calculate similarity scores
    scores = {}
    for col in put_query(q, 2):
        scores[col] = product2[col].sum()
    product_result = product2[list(scores.keys())].loc[new_q]
    print()
    print('Product (query*matched doc)')
    print(product_result)
    print()
    print('product sum')
    print(product_result.sum())
    print()
    print('Query Length')
    q_len = math.sqrt(sum([x ** 2 for x in query['idf'].loc[new_q]]))
    print(q_len)
    print()
    print('Cosine Simliarity')
    print(product_result.sum())
    print()
    # Sort scores in descending order
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Print ranked documents
    print('Returned docs')
    for typle in sorted_scores:
        print(typle[0], end=" ")


while True:
    user_query = input("Enter your query (): ")

    # Process the user's query

    insert_query(user_query)
