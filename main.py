from gensim import corpora
from gensim.models import LsiModel, Word2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import similarities
import json
from os.path import exists
import time

tag = ""

def preprocess_data(doc_set):

    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set('for a of the and to in'.split())
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        texts.append(tokens)
    return texts


def prepare_corpus(doc_clean):

    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary, doc_term_matrix


def create_gensim_lsa_model(doc_clean, number_of_topics, words):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    return lsamodel, dictionary, doc_term_matrix


desc_table_disorders = {}  # Holds {conceptID: term} for all disorder values
desc_table_disorders_flipped = {}  # Holds {term: conceptID} for all disorder values
desc_table_findings = {}  # Holds {conceptID: term} for all finding values
desc_table_findings_flipped = {}  # Holds {term: conceptID} for all finding values
rel_table = {}  # Holds the {conceptID: attribute frequencies} of all terms
con_table = set()  # Holds all active conceptIDs

desc_table_temp = {}


# Getting path to snapshots:

desc_file = "sct2_Description_Snapshot-en_INT_20210731.txt"
rel_file = "sct2_Relationship_Snapshot_INT_20210731.txt"
con_file = "sct2_Concept_Snapshot_INT_20210731.txt"

print("Initializing data...")

start = time.time()

print("Active concepts...")


with open(con_file, encoding="utf-8") as f:  # Save only active IDs to set
    for line in f:
        temp = line.split('\t')
        if temp[2] == '1':  # Only extract conceptIDs that are active
            con_table.add(temp[0])
    f.close()

print("Active concepts loaded!")

print("Active relationships...")

if exists("rel_table.json"):  # Check if relationship json file exists
    print("Found rel_table.json, Loading...")
    with open('rel_table.json') as f:
        data = f.read()
    js = json.loads(data)
    rel_table = js
    print("rel_table.json loaded!")
else:
    print("Building rel_table.json...")  # Load relationship information
    with open(rel_file, encoding="utf-8") as f:
        for line in f:
            temp = line.split('\t')
            if temp[2] == '1' and temp[7] != "116680003":  # Append only active non-hierarchical values to relationship dictionary
                if temp[4] in rel_table:
                    rel_table[temp[4]] += 1  # If it already exists, increment the frequency
                else:
                    rel_table[temp[4]] = 1  # Else, initialize as 1
        f.close()

    js = json.dumps(rel_table)
    f = open("rel_table.json", "w")
    f.write(js)  # Save the json file
    f.close()
    print("rel_table.json saved!")

print("Data loading complete!")

end = time.time()

print("\n\nInitialization completed after " + str(end-start) + " seconds!\n\n")


# Initialize 2 lists: one for disorders and one for findings
disordersList = list(desc_table_disorders.values())
findingsList = list(desc_table_findings.values())

threshold = 0.70


t = input("Enter term to compare: ")  # user enters the term to compare
t = t.lower()
tag = input("Enter the tag: ")
tag = tag.lstrip("("). rstrip(")").lower()
print("Building Model for", t, ". This won't take long!")

start = time.time()

with open(desc_file, encoding="utf-8") as f:
    for line in f:
        temp = line.split('\t')
        if temp[2] == '1' and temp[6] == "900000000000003001":  # Only pull descriptions that are active and typeID shown
            if "("+tag+")" in temp[7] and temp[4] in con_table:
                desc_table_temp[temp[4]] = temp[7].lower().replace(" ("+tag+")", "")  # Append conceptID and disorder term to disorders dictionary
    f.close()

documents_list = list(desc_table_temp.values())
num_list = list(desc_table_temp.keys())

texts = preprocess_data(documents_list)  # create and process the texts

# Building the model for the term
lsa_model, dictionary, doc_term_matrix = create_gensim_lsa_model(texts, 300, 5)
Word2Vec_model = Word2Vec(texts, vector_size=100, window=5, min_count=1)
Word2Vec_sims = Word2Vec_model.wv.most_similar(texts[1], topn=10)
lsa_index = similarities.MatrixSimilarity(lsa_model[doc_term_matrix])


# Gets similarities
def get_sims(i, lsa_index=lsa_index):
    lsa_sims = lsa_index[lsa_model[doc_term_matrix[i]]]
    return lsa_sims, i


def lsa(lsa_sims):

    # Build an output text file or overwrite the existing file
    if exists("output.txt"):  # Clear the output file if it exists
        file = open("output.txt", "r+")
        file.truncate(0)
        file.close()
    file = open("output.txt", "a")

    found = set()
    lsa_sims = sorted(enumerate(lsa_sims), key=lambda item: -item[1])
    display = ""
    for doc_position, doc_score in lsa_sims:  # iterate through all matches
        if doc_score > threshold:  # First check if it is above threshold

            # Check for duplicates, if it's in the relationship table, and the attributes aren't equal
            if num_list[doc_position] not in found and num_list[doc_position] in rel_table and num_list[i] in rel_table and rel_table[num_list[doc_position]] != rel_table[num_list[i]]:
                found.add(str(num_list[doc_position]))
                display += num_list[i] + " " + documents_list[i] + "\n" + num_list[doc_position] + " " + documents_list[doc_position] + "\nSimilarity Score: " + str(doc_score) + "\n\n"
        else:
            break

    file.write(display)  # Write results to text file
    file.close()

# Gets index of term to search
def find_index_from_document(document_original = "rip"):
    for i in range(len(documents_list)):
        if documents_list[i] == document_original:
            return i
    return -2


# Final similarities determined here
lsa_sims, i = get_sims(find_index_from_document(t))
lsa(lsa_sims)

end = time.time()
print("Similarity calculation complete after " + str(end-start) + " seconds!\nCheck results in output.txt")

