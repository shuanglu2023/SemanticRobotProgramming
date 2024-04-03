import spacy
import crosslingual_coreference
DEVICE = -1 # Number of the GPU, -1 if want to use CPU

# Add coreference resolution model
coref = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
coref.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": DEVICE})

def corref(input_text):
    coref_text = coref(input_text)._.resolved_text
    return coref_text