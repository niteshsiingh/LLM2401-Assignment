## Instructions to Reproduce the Assignment

1. **Set Up the Environment:**
  - **Step 1:** Clone the repository.
    ```sh
    git clone https://github.com/niteshsiingh/LLM2401-Assignment.git
    cd LLM2401-Assignment
    ```
  - **Step 2:** Create a virtual environment.
    ```sh
    conda create -n kg_rag python=3.10.12
    conda activate kg_rag
    ```
  - **Step 3:** Install the required dependencies.
    ```sh
    pip install -r requirements.txt
    ```
  - **Step 4:** Run the setup script to create the disease vector database for KG-RAG.
    ```sh
    python3 -m kg_rag.run_setup
    ```

2. **Update Google API Key:**
  - Add your Google API key in the `gpt_config.env` file.

3. **Replicate KG-RAG with Gemini:**
  - All the prior knowledge is commented, with their accuracies noted beside them. You can uncomment and proceed similarly.
  - Adjust the mode number in `kg_rag/rag_based_generation/GPT/run_mcq_qa.py` as required.
  - Execute the script `sh run_gemini.sh` to replicate KG-RAG using gemini-1.5-flash.

4. **Evaluate the Model:**
  - Change the model output path in `data/assignment_results/evaluate_gemini.py` and run the script to evaluate the model variants.
  - Use the command `python3 -m data.assignment_results.evaluate_gemini` to evaluate the model.


## Instructions for the Assignment

**1. Set up the environment following Steps 1-4 in the original README.**

**2. Update your Google API key in `gpt_config.env`.**

**3. Replicate KG-RAG with gemini-1.5-flash via `sh run_gemini.sh`.**

**4. Evaluate the model via `python data/assignment_results/evaluate_gemini.py`**

**5. Implement three enhancement strategies in `kg_rag/rag_based_generation/GPT/run_mcq_qa.py`.**

**6. Evaluate these model variants by changing the model output path in the file `data/assignment_results/evaluate_gemini.py` and running it.**





## Original README

<p align="center">
  <img src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/0b2f5b42-761e-4d5b-8d6f-77c8b965f017" width="450">
</p>




## Table of Contents
[What is KG-RAG](https://github.com/C-anwoy/LLM2401-Assignment#what-is-kg-rag)

[Example use case of KG-RAG](https://github.com/C-anwoy/LLM2401-Assignment#example-use-case-of-kg-rag)
 - [Prompting GPT without KG-RAG](https://github.com/C-anwoy/LLM2401-Assignment#without-kg-rag)  
 - [Prompting GPT with KG-RAG](https://github.com/C-anwoy/LLM2401-Assignment#with-kg-rag)
 - [Example notebook for KG-RAG with GPT](https://github.com/C-anwoy/LLM2401-Assignment/blob/main/notebooks/kg_rag_based_gpt_prompts.ipynb)

[How to run KG-RAG](https://github.com/C-anwoy/LLM2401-Assignment#how-to-run-kg-rag)
 - [Step 1: Clone the repo](https://github.com/C-anwoy/LLM2401-Assignment#step-1-clone-the-repo)
 - [Step 2: Create a virtual environment](https://github.com/C-anwoy/LLM2401-Assignment#step-2-create-a-virtual-environment)
 - [Step 3: Install dependencies](https://github.com/C-anwoy/LLM2401-Assignment#step-3-install-dependencies)
 - [Step 4: Update config.yaml](https://github.com/C-anwoy/LLM2401-Assignment#step-4-update-configyaml)
 - [Step 5: Run the setup script](https://github.com/C-anwoy/LLM2401-Assignment#step-5-run-the-setup-script)
 - [Step 6: Run KG-RAG from your terminal](https://github.com/C-anwoy/LLM2401-Assignment#step-6-run-kg-rag-from-your-terminal)
    - [Using GPT](https://github.com/C-anwoy/LLM2401-Assignment#using-gpt)
    - [Using GPT interactive mode](https://github.com/C-anwoy/LLM2401-Assignment/blob/main/README.md#using-gpt-interactive-mode)
    - [Using Llama](https://github.com/C-anwoy/LLM2401-Assignment#using-llama)
    - [Using Llama interactive mode](https://github.com/C-anwoy/LLM2401-Assignment/blob/main/README.md#using-llama-interactive-mode)
  - [Command line arguments for KG-RAG](https://github.com/C-anwoy/LLM2401-Assignment?tab=readme-ov-file#command-line-arguments-for-kg-rag)
  
[BiomixQA: Benchmark dataset](https://github.com/C-anwoy/LLM2401-Assignment/tree/main?tab=readme-ov-file#biomixqa-benchmark-dataset)

[Citation](https://github.com/C-anwoy/LLM2401-Assignment/blob/main/README.md#citation)


## What is KG-RAG?

KG-RAG stands for Knowledge Graph-based Retrieval Augmented Generation.

### Start by watching the video of KG-RAG

<video src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/86e5b8a3-eb58-4648-95a4-271e9c69b4ed" controls="controls" style="max-width: 730px;">
</video>

It is a task agnostic framework that combines the explicit knowledge of a Knowledge Graph (KG) with the implicit knowledge of a Large Language Model (LLM). Here is the [arXiv preprint](https://arxiv.org/abs/2311.17330) of the work.

Here, we utilize a massive biomedical KG called [SPOKE](https://spoke.ucsf.edu/) as the provider for the biomedical context. SPOKE has incorporated over 40 biomedical knowledge repositories from diverse domains, each focusing on biomedical concept like genes, proteins, drugs, compounds, diseases, and their established connections. SPOKE consists of more than 27 million nodes of 21 different types and 53 million edges of 55 types [[Ref](https://doi.org/10.1093/bioinformatics/btad080)]


The main feature of KG-RAG is that it extracts "prompt-aware context" from SPOKE KG, which is defined as: 

**the minimal context sufficient enough to respond to the user prompt.** 

Hence, this framework empowers a general-purpose LLM by incorporating an optimized domain-specific 'prompt-aware context' from a biomedical KG.

## Example use case of KG-RAG
Following snippet shows the news from FDA [website](https://www.fda.gov/drugs/news-events-human-drugs/fda-approves-treatment-weight-management-patients-bardet-biedl-syndrome-aged-6-or-older) about the drug **"setmelanotide"** approved by FDA for weight management in patients with *Bardet-Biedl Syndrome*

<img src="https://github.com/BaranziniLab/KG_RAG/assets/42702311/fc4d0b8d-0edb-461d-86c5-9d0d191bd97d" width="600" height="350">

### Ask GPT-4 about the above drug:

### WITHOUT KG-RAG

*Note: This example was run using KG-RAG v0.3.0. We are prompting GPT from the terminal, NOT from the chatGPT browser. Temperature parameter is set to 0 for all the analysis. Refer [this](https://github.com/C-anwoy/LLM2401-Assignment/blob/main/config.yaml) yaml file for parameter setting*

<video src="https://github.com/C-anwoy/LLM2401-Assignment/assets/42702311/dbabb812-2a8a-48b6-9785-55b983cb61a4" controls="controls" style="max-width: 730px;">
</video>

### WITH KG-RAG

*Note: This example was run using KG-RAG v0.3.0. Temperature parameter is set to 0 for all the analysis. Refer [this](https://github.com/BaranziniLab/KG_RAG/blob/main/config.yaml) yaml file for parameter setting*

<video src="https://github.com/C-anwoy/LLM2401-Assignment/assets/42702311/acd08954-a496-4a61-a3b1-8fc4e647b2aa" controls="controls" style="max-width: 730px;">
</video>

You can see that, KG-RAG was able to give the correct information about the FDA approved [drug](https://www.fda.gov/drugs/news-events-human-drugs/fda-approves-treatment-weight-management-patients-bardet-biedl-syndrome-aged-6-or-older).

## How to run KG-RAG

**Note: At the moment, KG-RAG is specifically designed for running prompts related to Diseases. We are actively working on improving its versatility.**

### Step 1: Clone the repo

Clone this repository. All Biomedical data used in the paper are uploaded to this repository, hence you don't have to download that separately.

### Step 2: Create a virtual environment
Note: Scripts in this repository were run using python 3.10.9
```
conda create -n kg_rag python=3.10.9
conda activate kg_rag
cd KG_RAG
```

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

### Step 4: Run the setup script
Note: Make sure you are in KG_RAG folder. 

Running the setup script will create disease vector database for KG-RAG

```
python -m kg_rag.run_setup
```


## Citation

```
@article{soman2023biomedical,
  title={Biomedical knowledge graph-enhanced prompt generation for large language models},
  author={Soman, Karthik and Rose, Peter W and Morris, John H and Akbas, Rabia E and Smith, Brett and Peetoom, Braian and Villouta-Reyes, Catalina and Cerono, Gabriel and Shi, Yongmei and Rizk-Jackson, Angela and others},
  journal={arXiv preprint arXiv:2311.17330},
  year={2023}
}
```














