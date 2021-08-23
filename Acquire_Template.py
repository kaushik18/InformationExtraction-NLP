# Acquire Template - Acquire(Organization,Organization,Date)

# Basic Imports
import re

# SpaCy Imports
import neuralcoref
from nltk.corpus.reader import dependency    
import spacy
from spacy.pipeline import EntityRuler

# Initialization of the Spacy NLP Pipeline 
nlp = spacy.load('en_core_web_sm')

# Entity Ruler to add Patterns
ruler = EntityRuler(nlp)

# Patterns to Identify Acquire Templates
patterns = [{"label": "ACQUIRE", "pattern": "acquired by"}, {"label": "ACQUIRE", "pattern": "acquired"},{"label": "ACQUIRE", "pattern": "acquire"},]

# Adding the Patterns to the EntityRuler
ruler.add_patterns(patterns)

# Adding the Pipe to the NLP Pipeline
nlp.add_pipe(ruler)

# Adding the Neural Coreference Resolution to the Pipeline
neuralcoref.add_to_pipe(nlp)

# Extraction Function
def acquire_template_extraction(doc,ner_list,dp_list):

    # Blank Template Format for Acquire
    template = {"Organization 1": "", "Organization 2": "", "Date": ""}

    # Initialization of the list of Templates
    list_of_templates = []

    # Initialization of the Temp_Date - Incase only one date is present in the Sentence
    temp_date=None

    # Looping through the NLP Pipeline Document - Active Case
    for head in doc:
        # To count the number of Date in the sentence NER 
        count_date = sum((x == 'DATE') for x in ner_list.values())
        
        # Assign if only one Date
        if(count_date == 1 and head.ent_type_=="DATE"):
            temp_date = head
        
        # Comparison for Acquire Entity
        if(head.ent_type_=="ACQUIRE"):
            
            # Looping through the Children in the Dependency Parse Tree
            for token in head.children:

                # If the Noun Subject is the dependency of the current child for Orgaization 1
                if (token.dep_ == "nsubj"):
                    template["Organization 1"] = token
                    
                # if the current child has a dobj dependency for Orgaization 2
                if (token.dep_ == "dobj"):

                    # if the dobj is a Noun for Orgaization 2
                    if (token.pos_=="NOUN"):
                        template["Organization 2"] =  token
                    
                    # To find Compund names for Orgaization 2
                    else:
                        name = str(token)
                        # Check the Children of the Current Word
                        for child in token.children:            
                            if(child.dep_=="compound" and child.pos_=="PROPN"):
                                name= str(child)+ " "+ name
                        
                        # Comparison with NER to find out Organization for Orgaization 2
                        for index,x in enumerate(ner_list):
                            if(x.label_ == "ORG" and x.text == name):
                                template["Organization 2"] = name
                                break
                
                # Checking for Single Date present in the Sentence
                if(temp_date != None):
                    template["Date"] = temp_date
                
                # If more than one date then check if pobj and the head word is "In"
                elif(head.dep_=="pobj" and head.head.text == "In"):
                    template["Date"] = token
                
                # More than one date but cannot find through dependency parsing
                else:
                    # Looping through the NER and checking if the Date part of template is filled or not
                    for index,x in enumerate(ner_list):
                        if(x.label_ == "DATE" and len(template["Date"])==0):
                            template["Date"] = x.text
                
                # Template Selection if both Organizations are found as the minimum base case
                if (len(template["Organization 1"]) > 0 and len(template["Organization 2"]) > 0 ):
                    list_of_templates.append(template)

                    # Reinitialization for new sentences
                    template = {"Organization 1": "", "Organization 2": "", "Date": ""}
    
    # Looping through the NLP Pipeline Document - Passive Case
    for head in doc:

        # To count the number of Date in the sentence NER 
        count_date = sum((x == 'DATE') for x in ner_list.values())

        # Assign if only one Date
        if(count_date == 1 and head.ent_type_=="DATE"):
            temp_date = head
        
        # Comparison for Acquire Entity
        if(head.ent_type_=="ACQUIRE"):
            
            # Looping through the Children in the Dependency Parse Tree
            for token in head.children:

                # If the Noun Subject Passive is the dependency of the current child ajd the POS Tag is a Noun for Orgaization 2
                if((token.pos_=="NOUN" and token.dep_ == "nsubjpass")):
                    template["Organization 2"] = token
                
                # if the dependency parse is only a noun subject Pasive voice for Orgaization 2
                elif ( token.dep_ == "nsubjpass"):
                    template["Organization 2"] = token
                
                # Direct Object for Organizaion 1
                if ( token.dep_ == "pobj" ):
                    template["Organization 1"] = token

                # Checking for Single Date present in the Sentence
                if(temp_date != None):
                        template["Date"] = temp_date
                
                # If more than one date then check if pobj and the head word is "In"
                elif(head.dep_=="pobj" and head.head.text == "In"):
                        template["Date"] = token
                
                # More than one date but cannot find through dependency parsing
                else:

                    # Looping through the NER and checking if the Date part of template is filled or not
                    for index,x in enumerate(ner_list):
                        if(x.label_ == "DATE" and len(template["Date"])==0):
                            template["Date"] = x.text

                # Template Selection if both Organizations are found as the minimum base case
                if (len(template["Organization 1"]) > 0 and len(template["Organization 2"]) > 0 ):
                    list_of_templates.append(template)

                    # Reinitialization for new sentences
                    template = {"Organization 1": "", "Organization 2": "", "Date": ""}

    return list_of_templates

# Function to check for Acquire Phrases using RE
def acquire_relation_check(sentence):
    if(re.search("(acquire|acquires|acquired by)", sentence)):
        return True
    return False

# Filter for Sentences that are potential Acquire Template Candidates
def acquire_template_sentence_check(sentence,ner_sentence,dp_sentence):

    count_org = sum((x == 'ORG') for x in ner_sentence.values())
    count_date = sum((x == 'DATE') for x in ner_sentence.values())
    if(count_org >=2 and count_date>=1):
        return sentence

    return None

# Main Program to call the Acquire Template
def getAquire(sentences,ners_list,dependency_parse_list):
    
    # Initialization of the final list of templates
    final_part=[]

    # Filtered Sentence List
    selected_sentences_list = []
    
    # Filtered Dependency Parse List
    selected_sentences_dependency_parse_structure = []
    
    # Filtered NER List List
    selected_sentences_ner = []

    # Looping through Sentneces One by one to Filter
    for index,sentence in enumerate(sentences):
        output_sentence = acquire_template_sentence_check(sentence,ners_list[index],dependency_parse_list[index])
        
        if output_sentence is None:
            continue
        else:
            selected_sentences_ner.append(ners_list[index])
            selected_sentences_dependency_parse_structure.append(dependency_parse_list[index])
            selected_sentences_list.append(output_sentence)

    # Looping through Selected Sentences for Template Extraction 
    for index,sentence_text in enumerate(selected_sentences_list):
        try:
            sentence = nlp(sentence_text)

            # ACQUIRE Template Extraction Call
            acquire_output = acquire_template_extraction(sentence,selected_sentences_ner[index],selected_sentences_dependency_parse_structure[index])

            # If no Template then Continue
            if(acquire_output != []):
                for i in acquire_output:

                    # Formatting of the Acquire Template
                    temp_dict={}
                    temp_dict["template"]="ACQUIRE"
                    temp_dict["sentences"]=[]
                    temp_dict["sentences"].append(sentence.text)
                    temp_dict["arguments"]={}
                    temp_dict["arguments"]["1"]=i["Organization 1"].text
                    temp_dict["arguments"]["2"]=i["Organization 2"].text

                    # Empty Date Values Cases
                    if(len(i['Date']) == 0):
                        temp_dict["arguments"]["3"]=""
                    else:
                        temp_dict["arguments"]["3"]=i["Date"].text
                   
                    # Final Template appended to list to be returned to the calling function
                    final_part.append(temp_dict)
        except:
            continue
    
    return final_part
    
