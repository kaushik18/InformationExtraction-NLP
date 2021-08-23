# NLTK Imports
from nltk import Tree
import re

# SpaCy Imports
import neuralcoref    
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load('en_core_web_sm')
# Entity Ruler to add Patterns
ruler = EntityRuler(nlp)

# Patterns to Identify Born Templates
patterns = [{"label": "PART_OF", "pattern": "part of"}]

# Adding the Patterns to the EntityRuler
ruler.add_patterns(patterns)

# Adding the Pipe to the NLP Pipeline
nlp.add_pipe(ruler)


def part_of_relation_check(sentence):
    if(re.search("part of", sentence)):
        return True
    return False

def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)
    
def organization_pattern(text,ners,dp):

    doc = nlp(text)

    template = {"Organization 1": "", "Organization 2": ""}
    list_of_templates = []
  
    temp_org = None
    for entity in doc:

        if(entity.ent_type_ == "PART_OF"):

            for index,x in enumerate(ners):
                
                if(x.label_ == "ORG"):
                    temp_org = x.text
                
                if(template["Organization 1"] != ""):
                    if(x.label_ == "ORG"):
                        template["Organization 2"] = x.text

                # Keeping track of current and previous tokens in NER List
                if index == 0:
                    prev_token = [x.label_,x.text]
                    cur_token = [x.label_,x.text]
                else:
                    prev_token = cur_token
                    cur_token = [x.label_,x.text]
                # Pattern Matching through RE and comparison with NER for first parameter of template
                if(re.search("(part of)", cur_token[1])):
                    if(prev_token[0]=="ORG"):
                        template["Organization 1"] = prev_token[1]
                    else:
                        template["Organization 1"] = temp_org

    # print(template)
    # Template Selection if both Parameter and Date are found as the minimum base case
    if (len(template["Organization 1"]) > 0 and len(template["Organization 2"]) > 0):
        list_of_templates.append(template)
        
        # Reinitialization for new sentences
        template = {"Organization 1": "", "Organization 2": ""}
    # print(list_of_templates)
    return list_of_templates

def part_template_sentence_check(sentence,ner_sentence):

    count = sum((x == 'ORG') for x in ner_sentence.values())

    if(count>=2):
        result = part_of_relation_check(sentence)
        if(result): 
            return sentence

    return None

def display_tree(sentence):

    doc = nlp(sentence)
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

def part_home(sentence,ners,dp):
    combined_output=set()
    combined_output.update(organization_pattern(sentence,ners,dp))
    combined_output = list(combined_output)
    return combined_output

def getPartOrg(sentences,ners_list,dependency_parse_list):
    
    final_part = []
    selected_sentences_list = []
    selected_sentences_dependency_parse_structure = []
    selected_sentences_ners = []

    for index,sentence in enumerate(sentences):
        output_sentence = part_template_sentence_check(sentence,ners_list[index])
        if output_sentence is None:
            continue
        else:
            selected_sentences_dependency_parse_structure.append(dependency_parse_list[index])
            selected_sentences_ners.append(ners_list[index])
            selected_sentences_list.append(output_sentence)

    for index,sentence in enumerate(selected_sentences_list):
        try:

            part_output = organization_pattern(sentence,selected_sentences_ners[index],selected_sentences_dependency_parse_structure[index])

            if(part_output != []):
                # print(part_output)
                for j in part_output:
                    temp_dict ={}
                    temp_dict["template"] ="PART_OF"
                    temp_dict["sentences"] = []
                    temp_dict["sentences"].append(sentence)
                    temp_dict["arguments"] = {}
                    temp_dict["arguments"]["1"] = j["Organization 1"]
                    temp_dict["arguments"]["2"] = j["Organization 2"]
                    final_part.append(temp_dict)
        
        except:
            continue    

   
    return final_part