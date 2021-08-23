
import os
import errno
import json

# Template Import
from Part_Template_LOC import getPart
from Acquire_Template import getAquire
from Part_Template_ORG import getPartOrg
from Born_Template import getBorn

# ------------------------------------------------------------------------------------------------------------------------------------ #
# ------------------- Task 2 - Extract Information Templates using Heuristic, or Statistical or Both Methods ------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------ #
def Template_Extraction(sentences,dependency_parse_tree_list,ners_list,file_name):

    # To obtain the File Name from the path
    base = os.path.basename(file_name)

    print('\nStarting Task 2 - Extract Information Templates using Heuristic, or Statistical or Both Methods\n')
    print('Three Templates: \n1. Part(Location, Location) or Part(Organization, Organization)\n2. Acquire(Organization, Organization, Date)\n3. Born(Person/Organization, Date, Location)\n')

    # # Individual Template Extraction
    
    # Part(Organization,Organization)
    output_part_template_org = getPartOrg(sentences,ners_list,dependency_parse_tree_list)

    # Part(Location,Location)
    output_part_template = getPart(sentences,ners_list)

    # Acquire(Organization, Organization, Date)
    output_acquire_template = getAquire(sentences,ners_list,dependency_parse_tree_list)

    # Born(Person/Organization, Date, Location)
    output_born_template = getBorn(sentences,ners_list,dependency_parse_tree_list)

    # Final Output for Template
    final_output_dictionary={}

    # Document Name in the Template
    final_output_dictionary["document"]=base

    # Extraction Section of the Template - Initialization
    final_output_dictionary["extraction"]=[]

    # Looping through Acquire Template throughout the document
    for acquire_templates in output_acquire_template:
        final_output_dictionary['extraction'].append(acquire_templates)
    
    # Looping through Born Template throughout the document
    for born_templates in output_born_template:
        final_output_dictionary['extraction'].append(born_templates)

    # Looping through Part(Location,Location) Template throughout the document
    for part_templates in output_part_template:
        final_output_dictionary['extraction'].append(part_templates)

    # Looping through Part(Organization,Organization) Template throughout the document
    for part_templates_org in output_part_template_org:
        final_output_dictionary['extraction'].append(part_templates_org)


    # Create the Features Folder with TextFile Folder
    try:
        os.makedirs('Output_JSONs/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    # Getting the File Name from the Path
    output_file_name = os.path.splitext(base)[0]
    
    # Creating the JSON File to store the extracted templates
    json_output_file_name = "Output_" + output_file_name + ".json"
    json_object_output = json.loads(json.dumps(final_output_dictionary))
    final_json_data = json.dumps(json_object_output, indent=2)

    # Output JSON Folder
    output_file = open('Output_JSONs/'+json_output_file_name, "w")
    output_file.write(final_json_data)
    output_file.close()
    
    # Information to Display after Extraction
    print('Output JSON for "' + base + '" created in the Output_JSONs Folder - File Name: ' + json_output_file_name)
    print("\n-----------------------------------------------------------------------------------------------------------")