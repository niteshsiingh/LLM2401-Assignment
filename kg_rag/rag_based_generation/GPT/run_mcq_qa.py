'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys
import json


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
JSON_PROMPT = system_prompts["JSON_PROMPT"]

CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


MODE = "3"
### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ### 

def jsonlize_context(input_text):
    associations = []
    sentences = input_text.split('. ')
    
    for sentence in sentences:
        if "Disease" in sentence and "associates Gene" in sentence:
            parts = sentence.split('associates Gene')
            disease = parts[0].strip().replace("Disease", "").strip()
            gene = parts[1].strip()
            associations.append({"Disease": disease, "Gene": gene})
        elif "Variant" in sentence and "associates Disease" in sentence:
            parts = sentence.split('associates Disease')
            variant = parts[0].strip().replace("Variant", "").strip()
            disease = parts[1].strip()
            associations.append({"Disease": disease, "Variant": variant})
    return json.dumps(associations, indent=4)

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    
    for index, row in tqdm(question_df.iterrows(), total=306):
        try: 
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ### 
                context = retrieve_context(
                    question, vectorstore, embedding_function_for_context_retrieval, node_context_df,
                    CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                ### MODE 1: jsonlize the context from KG search ### 
                ### Please implement the first strategy here    ###
                context = retrieve_context(
                    question, vectorstore, embedding_function_for_context_retrieval, node_context_df,
                    CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                json_output = jsonlize_context(context)
                enriched_prompt = "Context: "+ json_output + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "2":
                ### MODE 2: Add the prior domain knowledge      ### 
                ### Please implement the second strategy here   ###
                context = retrieve_context(
                    question, vectorstore, embedding_function_for_context_retrieval, node_context_df,
                    CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                
                # prior_domain_knowledge = """
                # -	Disease-Gene Associations: Many diseases, like amyloidosis and erythroleukemia, are tied to specific gene sets and variants that impact protein production or cellular pathways, influencing disease progression and manifestation. Understanding these associations aids in mapping broader disease networks.
	            # -	Pathway-Level Impacts: Diseases often share pathway disruptions. For instance, inflammatory pathways are not only implicated in conditions like serum amyloid A amyloidosis but also play a role in chronic conditions with overlapping genetic markers. Such shared pathways can highlight potential common therapeutic targets.
	            # -	Mutational Impacts Across Diseases: Genetic mutations, especially those affecting bone development or blood cell differentiation, show variable expression across diseases. The same mutation can sometimes present in multiple conditions but lead to distinct phenotypes, offering a nuanced understanding of genotype-phenotype relationships.
	            # -	Network Interactions: Diseases connected through the same genes often show similar progression patterns. Network analysis of such connections can illuminate unknown or less-studied relationships between conditions, offering a pathway to better therapeutic mapping and understanding secondary disease manifestations.
	            # -	Comorbidity Patterns and Genetic Overlaps: In the context of a broader network like Spoke, observing comorbid diseases sharing genetic markers or pathways can reveal higher-level insights into disease clustering and predisposition, suggesting areas for early diagnosis and intervention.
                # """ #74.84%
                # prior_domain_knowledge = """
                # - Some genes directly cause diseases (e.g., mutations in the CFTR gene lead to Cystic Fibrosis), while others serve as risk factors without direct causation (e.g., APOE variants increase Alzheimer’s risk).
                # - Diseases often share genes involved in core biological pathways. For example, BRCA1 mutations link breast and ovarian cancers due to their role in DNA repair.
                # - Gene expression changes, epigenetic modifications, and regulatory elements also contribute to disease processes, impacting gene function without altering DNA sequences.
                # - Genetic differences can influence drug metabolism, efficacy, and toxicity, which are important for personalizing medical treatments.
                # """ 
                # 74.18%
                prior_domain_knowledge = """
                - Symptom and Provenance information is useless.
                - Similar Diseases tend to have similar gene associations.
                - Diseases with similar genetic profiles may respond similarly to treatments.
                - Gene expression levels can indicate disease severity and progression.
                - Diseases in the same family often share genetic risk factors.
                """ #77.12%, 77.45%, 77.78%, 78.10%
                # - Some genes have protective effects against certain diseases.
                enriched_prompt = "Context: "+ context + "\n" +  "Prior Domain Knowledge: "+ prior_domain_knowledge + "\n"+ "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            if MODE == "3":
                ### MODE 3: Combine MODE 1 & 2                  ### 
                ### Please implement the third strategy here    ###
                context = retrieve_context(
                    question, vectorstore, embedding_function_for_context_retrieval, node_context_df,
                    CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                json_output = jsonlize_context(context)
                # prior_domain_knowledge = """
                # - Serum Amyloid A Amyloidosis: A disease involving the abnormal accumulation of serum amyloid proteins, often linked with chronic inflammatory disorders, which can lead to organ damage.
                # - Erythroleukemia: A rare and aggressive form of acute myeloid leukemia characterized by the abnormal proliferation of immature red and white blood cells, often associated with genetic mutations affecting blood cell development.
                # - Spondylometaphyseal Dysplasia (Sedaghatian Type): A severe genetic skeletal disorder caused by specific mutations, marked by spinal and metaphyseal anomalies along with other systemic complications.
                # - Cerebrooculofacioskeletal Syndrome 2: A complex, multi-system genetic disorder that involves neurological, ocular, facial, and skeletal abnormalities, often with significant developmental implications.
                # - Bilateral Frontoparietal Polymicrogyria: A condition characterized by abnormal cortical development, leading to multiple small brain folds, and commonly associated with developmental delays and epilepsy.
                # """
                # prior_domain_knowledge = """
                # -	Disease-Gene Associations: Many diseases, like amyloidosis and erythroleukemia, are tied to specific gene sets and variants that impact protein production or cellular pathways, influencing disease progression and manifestation. Understanding these associations aids in mapping broader disease networks.
	            # -	Pathway-Level Impacts: Diseases often share pathway disruptions. For instance, inflammatory pathways are not only implicated in conditions like serum amyloid A amyloidosis but also play a role in chronic conditions with overlapping genetic markers. Such shared pathways can highlight potential common therapeutic targets.
	            # -	Mutational Impacts Across Diseases: Genetic mutations, especially those affecting bone development or blood cell differentiation, show variable expression across diseases. The same mutation can sometimes present in multiple conditions but lead to distinct phenotypes, offering a nuanced understanding of genotype-phenotype relationships.
	            # -	Network Interactions: Diseases connected through the same genes often show similar progression patterns. Network analysis of such connections can illuminate unknown or less-studied relationships between conditions, offering a pathway to better therapeutic mapping and understanding secondary disease manifestations.
	            # -	Comorbidity Patterns and Genetic Overlaps: In the context of a broader network like Spoke, observing comorbid diseases sharing genetic markers or pathways can reveal higher-level insights into disease clustering and predisposition, suggesting areas for early diagnosis and intervention.
                # """
                # prior_domain_knowledge = """
                # - Symptom and Provenance information is useless.
                # - Similar Diseases tend to have similar gene associations.
                # - Diseases with similar genetic profiles may respond similarly to treatments.
                # - Gene expression levels can indicate disease severity and progression.
                # - Diseases in the same family (e.g., autoimmune diseases) often share genetic risk factors.
                # - Some genes have protective effects against certain diseases.
                # """
                #  #78.43%
                prior_domain_knowledge = """
                - Symptom and Provenance information is useless.
                - Similar Diseases tend to have similar gene associations.
                - Diseases with similar genetic profiles may respond similarly to treatments.
                - Gene expression levels can indicate disease severity and progression.
                - Diseases in the same family often share genetic risk factors.
                """
                # json_output["Prior Domain Knowledge: "+ prior_domain_knowledge]
                enriched_prompt = "Context: "+ json_output + "\n" + "Prior Domain Knowledge: "+ prior_domain_knowledge + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            answer_list.append((row["text"], row["correct_node"], output))
        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))


    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    answer_df.to_csv(output_file, index=False, header=True) 
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))

        
        
if __name__ == "__main__":
    main()


