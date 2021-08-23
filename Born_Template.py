# NLP - Information Extraction Templates

# Basic Imports
import re

# SpaCy Imports
import neuralcoref    
import spacy
from spacy.pipeline import EntityRuler

from Part_Template_LOC import location_pattern

# Initialization of the Spacy NLP Pipeline 
nlp = spacy.load('en_core_web_sm')

# Entity Ruler to add Patterns
ruler = EntityRuler(nlp)

# Patterns to Identify Born Templates
patterns = [{"label": "BORN", "pattern": "founded by"}, {"label": "BORN", "pattern": "founded in"}, {"label": "BORN", "pattern": "founded on"}, {"label": "BORN", "pattern": "born on"},{"label": "BORN", "pattern": "founder of"},]

# Adding the Patterns to the EntityRuler
ruler.add_patterns(patterns)

# Adding the Pipe to the NLP Pipeline
nlp.add_pipe(ruler)

# Adding the Neural Coreference Resolution to the Pipeline
neuralcoref.add_to_pipe(nlp)


def born_template_extraction(sentence,doc,ner_list,dp_list):
    
    # Blank Template Format for Born
    template = {"Parameter_1": "", "Date": "", "Location": ""}
    
    # Initialization of the list of Templates
    list_of_templates = []

    # Initialization of the Temp_Date - Incase only one date is present in the Sentence
    temp_date=None

    # Looping through the NLP Pipeline Document
    for head in doc:
        
        # Looping through the Children in the Dependency Parse Tree
        for index, token in enumerate(head.children):

            # Keeping track of current and previous tokens in Dependency Parse Tree
            if index == 0:
                cur_token = [token.dep_,token.head.text,token.text,token.pos_]
                prev_token = [token.dep_,token.head.text,token.text,token.pos_]
            else:
                prev_token = cur_token
                cur_token = [token.dep_,token.head.text,token.text,token.pos_]

            # Comparison for Born Entity
            if(head.ent_type_=="BORN"):

                # Location Count
                count_loc = sum((x == 'GPE') for x in ner_list.values())

                # Looping through NER
                for index,x in enumerate(ner_list):
                    
                    # Keeping track of current and previous tokens in NER List
                    if index == 0:
                        prev_token = [x.label_,x.text]
                        cur_token = [x.label_,x.text]
                    else:
                        prev_token = cur_token
                        cur_token = [x.label_,x.text]

                    # Pattern Matching through RE and comparison with NER for first parameter of template
                    if(re.search("(founded by|born on|founder of|founded on|founded in)", cur_token[1])):
                        if(prev_token[0]=="ORG" or prev_token[0]=="PERSON"):
                            template["Parameter_1"] = prev_token[1]
                    
                    # Date Parameter 
                    if(x.label_ == "DATE" and len(template["Date"])==0):
                        template["Date"] = x.text
                    
                    # Locaton Parameter
                    if(count_loc > 1 and x.label_ == "GPE"):
                        loc = list(location_pattern(sentence))
                        # Condition if returned locations are more than one
                        if(len(loc)>0):
                            template["Location"] = str(loc[0][0])+", "+ str(loc[0][1])
                    
                    # If only one location in the NER Sentence
                    elif(count_loc == 1 and x.label_ == "GPE"):
                        template["Location"] = x.text
    
    # Template Selection if both Parameter and Date are found as the minimum base case
    if (len(template["Parameter_1"]) > 0 and len(template["Date"]) > 0):
        list_of_templates.append(template)
        
        # Reinitialization for new sentences
        template = {"Parameter_1": "", "Date": "", "Location": ""}

    return list_of_templates

# Main Program to call the Born Template
def getBorn(sentences,ners_list,dependency_parse_list):
    
    # Initialization of the final list of templates
    final_part=[]
    
    # Filtered Sentence List
    selected_sentences_list = sentences

    # Filtered Dependency Parse List
    selected_sentences_dependency_parse_structure = dependency_parse_list

    # Filtered NER List List
    selected_sentences_ner = ners_list

    # Looping through Selected Sentences for Template Extraction 
    for index,sentence_text in enumerate(selected_sentences_list):
        try:
            # Passing Sentence through Pipeline
            sentence = nlp(sentence_text)
            
            # BORN Template Extraction Call
            ans = born_template_extraction(sentence_text,sentence,selected_sentences_ner[index],selected_sentences_dependency_parse_structure[index])

            # If no Template then Continue
            if(ans!=[]):
                for i in ans:

                    # Formatting of the Born Template
                    temp_dict={}
                    temp_dict["template"]="BORN"
                    temp_dict["sentences"]=[]
                    temp_dict["sentences"].append(sentence.text)
                    temp_dict["arguments"]={}
                    temp_dict["arguments"]["1"]=i["Parameter_1"]
                    temp_dict["arguments"]["2"]=i["Date"]
                    temp_dict["arguments"]["3"]=i["Location"]
                    
                    # Final Template appended to list to be returned to the calling function
                    final_part.append(temp_dict)
        except:
            continue
        
    return final_part
    
