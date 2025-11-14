# Main points RAG

**Algorithm designed for global and accurate retrieval augmented generation (RAG)**

The algorithm has the following steps:

1. Extract points of information from a paper that are relevant to answer a query.
2. Identify any chunks in the paper that are semantically close to the extrated points.
3. Select only the chunks with semantic distance under a threshold.
4. Build context information from the selected chunks.
5. Give the LLM the context information to answer the query.
6. Check if the answer has unsupported claims.
7. Fix the answer with any identified unsupported claims.
8. Repeat until there are no usupported claims.
9. Return the answer and retrieved chunks.

## Global context

The LLM is initially provided with the full document text to extract points of information present in the paper and that are relevant to the query (step 1). The individual points may not be semantically similar to the query, but they are needed information that is likely present in the paper. 

This sets a contrast with standard RAG since information that is not directly similar to the query can be logically combined to generate a nuanced answer. Other advanced-RAG algorithms also divide the query into sub-queries. The difference here is that the sub-queries (points) are confined to information that must be explicitly present in the paper. This then allows to quantitatively verify if a point or sub-query is valid through similarity metrics. 

## Accurate retrieval

Each extracted point is compared with literal chunks from the paper (step 2). Only matching chunks are selected (step 3). This mitigates hallucinations by considering only points that are closely related to explicit text in the paper.

## Caveats

An LLM needs to be able to process the entire document. This limits the document size to text smaller than the maximum LLM context length. This is, thus, intended for standard papers that can be processed by current LLMs (~100K context length). 

Only points with text close to text explicitly appearing in the paper are considered. This could limit the ability of the LLM to answer some queries. The verification/fix steps (6 and 7) are intended to mitigate hallucinations when the LLM finds itself with less than enough information to answer a query.

# Example usage

## setup

- Install ollama by running on a terminal the following command:

curl -fsSL https://ollama.com/install.sh | sh

- Pull an LLM. For instance gpt-oss:

ollama pull gpt-oss

- Recreate the conda environment:

conda create -f environment.yml --name deepresearch

- Activate the environment:

conda activate deepresearch

## get a paper text


```python
with open('paper_text.txt', 'r') as fp:
    paper_text = fp.read()

print(paper_text[:500], '\n\n...\n\n', paper_text[-500:])
```

     Cell Cycle
    ISSN: 1538-4101 (Print) 1551-4005 (Online) Journal homepage: www.tandfonline.com/journals/kccy20
    Acetic acid eﬀects on aging in budding yeast: Are
    they relevant to aging in higher eukaryotes?
    William C. Burhans & Martin Weinberger
    To cite this article: William C. Burhans & Martin Weinberger (2009) Acetic acid eﬀects on aging
    in budding yeast: Are they relevant to aging in higher eukaryotes?, Cell Cycle, 8:14, 2300-2302,
    DOI: 10.4161/cc.8.14.8852
    To link to this article:  https://doi. 
    
    ...
    
      8. Chu IM, et al. Nat Rev Cancer 2008; 8:253-67.
     9. Beales IL, et al. BMC Cancer 2007; 7:97.
     10. Huang WC, et al. Curr Biol 2008; 18:781-5.
     11. Burhans WC, et al. Nucleic Acids Res 2007; 35:7545-56.
     12. Miyauchi H, et al. EMBO J 2004; 23:212-20.
     13. Nogueira V, et al. Cancer Cell 2008; 14:458-70.
     14. Halazonetis TD, et al. Science 2008; 319:1352-5.
     15. Brunet A, et al. Science 2004; 303:2011-5.
     16. Wang F, et al. Aging Cell 2007; 6:505-14.
     17. Jones RG, et al. Mol Cell 2005; 18:283-93.


## run


```python
import main_points_rag


answer, retrieved_chunks = main_points_rag.run(
    query='What mechanisms by which acetic acid contributes to aging in yeast can be extrapolated to human aging?',
    paper_text=paper_text,
    model='gpt-oss',
    num_predict=4000,
    num_tries=5,
    threshold=0.4,
    num_chunks_per_point=1,
    window_size=3,
    max_iterations=3,
    verbose=True,
)
```

    Your task is to extract the most relevant points of information from a paper in order to answer a query. The points must closely correspond to literal excerpts from the paper. Both the paper and the query are given below:
    
    <paper>
     Cell Cycle
    ISSN: 1538-4101 (Print) 1551-4005 (Online) Journal homepage: www.tandfonline.com/journals/kccy20
    Acetic acid eﬀects on aging in budding yeast: Are
    they relevant to aging in higher eukaryotes?
    William C. Burhans & Martin Weinberger
    To cite this article: Will 
    
    ...
    
      be extrapolated to human aging?
    </query>
    
    Provide your answer in the following format:
    
    <thoughts>
    Your reasoning on what pieces of information are needed to answer the query. Are any of those pieces of information present in the paper? Which ones?
    </thoughts>
    
    <points>
    A list of one-sentence distinct points of information from the paper (if any) that are relevant to the query. In as much as possible, the points must be independent from each other. Example:
    - point 1.
    - point 2.
    ...
    </points>
    
    
    <thoughts>
    The query asks for mechanisms by which acetic acid contributes to aging in yeast that can be extrapolated to human aging.  
    To answer, we need to identify statements in the paper that:  
    1. describe how acetic acid affects yeast aging (oxidative stress, growth‑signaling activation, replication stress, G1 arrest inhibition, etc.);  
    2. link those yeast mechanisms to analogous processes in mammals (low‑pH activation of AKT/RAS, oxidative‑stress suppression, replication stress, G1 arrest suppression, etc.).  
    All relevant points must be literal excerpts from the paper, each forming an independent sentence or phrase.  
    The paper contains several such excerpts that directly compare yeast and mammalian mechanisms, so the answer can be constructed from those sentences.
    
    <points>
    - “Thus, the accumulation of acetic acid in stationary phase induces oxidative stress, a factor previously implicated in chronological aging of yeast and aging in other organisms as well.”  
    - “Buffering medium also increased the frequency with which cells arrest growth in G1 when they enter stationary phase.”  
    - “Mutational inactivation of conserved growth signaling pathways (deletion of genes encoding the AKT homologue Sch9 or the RAS homologue Ras2) confer resistance to acetic acid toxicity in stationary phase cultures.”  
    - “Deletion of SCH9 induces the superoxide dismutase Sod2 in stationary phase cells.”  
    - “Acetic acid‑induced growth signals that promote entry into S phase, but in the absence of nutrients or the regulatory mechanisms required for the synthesis of dNTPs and efficient DNA replication. This is a recipe for replication stress.”  
    - “Low pH also stimulates growth signaling pathways in mammals. This includes, for example, activation by low pH of AKT and cellular proliferation in Barrett’s esophagus.”  
    - “Extracellular acidification—which is detected in many solid tumors—also has been linked to proliferation of medulloblastoma cells by triggering the formation of IP3, a potent activator of AKT, and
    Your task is to answer a query based only on provided information. Both the query and information are given below.
    
    <query>
    What mechanisms by which acetic acid contributes to aging in yeast can be extrapolated to human aging?
    </query>
    
    <information>
    Abrogation of this checkpoint leads to growth arrest in S phase followed by apoptosis. 17 In summary, Burtner et al. raise an important question about the relevance of acetic acid effects in the yeast chronological aging model to aging in higher euk 
    
    ...
    
     nd other age-related diseases, including cancer.
    </information>
    
    Provide your answer in the following format:
    
    <thoughts>
    Your reasoning on what information persent in the provided information can be used to answer the query, if at all possible.
    </thoughts>
    
    <is-answerable>
    Is there enough information to answer the query? write down "Yes" or "No"
    <is-answerable>
    
    <answer>
    Your answer to the query based only on the provided information. Leave empty if no enough information was provided.
    </answer>
    <thoughts>
    The passage describes how acetic acid accumulation in yeast stationary phase cultures activates conserved growth signaling pathways (e.g., Sch9/Sic1 homologues, AKT, RAS, cAMP), increases oxidative stress, and induces replication stress by forcing cells into S‑phase without adequate nutrients or dNTP synthesis. It also notes that low extracellular pH can stimulate similar signaling in mammals, leading to mitochondrial signaling, inflammation, and activation of AKT/MEK‑ERK pathways. These mechanisms are linked to age‑related diseases such as cancer and neurodegeneration. Therefore, the information allows us to extrapolate that in humans, chronic low‑pH or acetic‑acid‑like metabolic conditions could similarly drive growth signaling, oxidative damage, and replication stress, contributing to aging and age‑related pathologies.
    </thoughts>
    
    <is-answerable>
    Yes
    </is-answerable>
    
    <answer>
    In yeast, the accumulation of acetic acid during stationary phase activates conserved growth‑signaling pathways (e.g., Sch9/Sic1, the AKT/RAS/cAMP axis), drives cells into S‑phase even when nutrients are scarce, and thereby creates **replication stress**.  Acetic acid also lowers extracellular pH, which in yeast and in mammals stimulates mitochondrial signaling and increases **oxidative stress**.  These two stresses—replication and oxidative—are known to accelerate cellular senescence and apoptosis in yeast.
    
    Because the same signaling molecules (AKT, RAS, cAMP, IGF‑1 pathways) and stress responses are conserved in mammals, the mechanisms observed in yeast can be extrapolated to human aging in the following ways:
    
    1. **Chronic low‑pH or acetic‑acid‑like metabolic states** can continuously stimulate growth‑signaling pathways in human cells, promoting cell cycle entry and proliferation in an environment that may lack sufficient nutrients or DNA‑repair capacity, leading to replication stress.
    
    2. **Activation of AKT/MEK‑ERK and IGF‑1 signaling** by low extracellular pH or metabolic acids can increase mitochondrial ROS production, thereby elevating oxidative stress, a hallmark of aging and age‑related diseases.
    
    3. **Replication stress** induced by inappropriate cell‑cycle progression can cause DNA damage, genomic instability, and trigger senescence or apoptosis, contributing to tissue dysfunction and age‑related pathologies such as cancer and neurodegeneration.
    
    4. **Inflammatory responses** triggered by extracellular acidification (as seen in tumors) may further amplify growth signaling and oxidative damage, creating a feed‑forward loop that accelerates aging processes.
    
    Thus, the yeast model suggests that persistent metabolic acidification and the resulting activation of growth‑signaling pathways, coupled with oxidative and replication stress, are conserved mechanisms that can contribute to aging in humans.
    Your task is to verify whether an assistant's answer to a query is accurately based only the explicitly provided context information. The context information, query, and answer are given below.
    
    <context-information>
    Abrogation of this checkpoint leads to growth arrest in S phase followed by apoptosis. 17 In summary, Burtner et al. raise an important question about the relevance of acetic acid effects in the yeast chronological aging model to aging in higher eukaryotes. indicate that in yeast, a 
    
    ...
    
     at:
    
    <thoughts>
    Your thoughts regarding whether the answer is accurately derived from the context information. Are there any important points in the response that are not supported by the context information?
    </thoughts>
    
    <is-supported>
    Is the assistant's answer to the query supported by the context information? write down Yes or No
    <is-supported>
    
    <unsupported-claims>
    Write here any unsupported claims made by the assistant. Leave empty if there are no unsupported claims.
    </unsupported-claims>
    
    
    <thoughts>
    The assistant’s answer largely reflects the mechanisms described in the context: acetic acid accumulation in yeast activates conserved growth‑signaling pathways (Sch9/Sic1, AKT/RAS/cAMP), drives cells into S‑phase under nutrient‑depleted conditions, and induces replication stress; it also lowers extracellular pH, stimulating mitochondrial signaling and oxidative stress. The context explicitly notes that low pH stimulates growth signaling in mammals, that extracellular acidification in tumors activates IP3, AKT, and MEK/ERK pathways, and that conserved IGF‑1/AKT/RAS signaling contributes to aging and age‑related diseases. The assistant’s extrapolation to human aging—linking chronic low‑pH or metabolic acidification to sustained growth signaling, replication stress, oxidative stress, and inflammation—is a reasonable inference from these points. No claim in the answer introduces information that is not at least hinted at in the provided text. Therefore the answer is supported by the context, with no clearly unsupported statements.
    </thoughts>
    
    <is-supported>
    Yes
    </is-supported>
    
    <unsupported-claims>
    </unsupported-claims>


## Retrieved chunks


```python
for chunk in retrieved_chunks: print(chunk, '\n\n')
```

    Abrogation of this checkpoint leads to growth arrest in S phase followed by apoptosis. 17 In summary, Burtner et al. raise an important question about the relevance of acetic acid effects in the yeast chronological aging model to aging in higher eukaryotes. indicate that in yeast, accumulation of acetic acid in stationary phase cultures stimulates highly conserved growth signaling pathways and increases oxidative stress and replication stress, all of which have been implicated in aging and/or age-related diseases in more complex organisms. Low pH also stimulates growth signaling pathways in mammals. Although the reduced production of acetic acid identified by Burtner et al. The remarkable parallels between regulation of chronological aging in yeast and of aging in more complex organisms suggest that conserved growth signaling pathways impact aging in all eukaryotes via dual effects on oxida- tive and replication stress. 
    
    
    This conclusion is supported by the recent discovery of a distinct popu- lation of “non-quiescent” stationary phase cells that, in addition to its enrichment for budded and apoptotic cells, exhibits elevated expres- sion of genes encoding proteins that respond to replication stress. 2 A role for acetic acid in the induction of growth signaling path- ways and replication stress in yeast is not surprising. 6 Thus, despite the absence of glucose (which has been completely consumed), nutrient-depleted stationary phase cells are continuously subjected to acetic acid-induced growth signals that promote entry into S phase, but in the absence of nutrients or the regu- latory mechanisms required for the synthesis of dNTPs and efficient DNA replication. This is a recipe for replication stress. is related in part to reduced growth signaling that promotes the more frequent growth arrest in G 1 we detected earlier in these strains. 4 This would protect against acetic acid-induced replication stress. 
    
    
    Cell Cycle 2301 of the mammalian homologues of Sch9 and Sic1 (AKT and p27), however, this is likely to be the case—AKT inhibits G 1 arrest in mammalian cells by inhibiting the activity of p27. 8 Are the CLS-shortening effects of acetic acid in yeast relevant to aging in higher eukaryotes? Although acetic acid is not a specific physiological trigger of growth signaling pathways in more complex organisms, induction of mito- genic signaling by low pH has been reported in higher eukaryotes, where it contributes to a variety of patho - logical states related to aging, such as inflammation and 9 Extracellular acidification—which is detected in many solid tumors—also has been linked to proliferation of medulloblastoma cells by triggering the formation of IP3, a potent activator of AKT, and activating the MEK/ERK mitogen signaling pathway. Even more relevant is the fact that growth signaling through conserved insulin-like growth factor (IGF-1) path- ways regulated by AKT, RAS and cAMP and other molecules is a well-established factor in aging in all eukaryotes. Growth signaling through these pathways also contributes to neurodegenerative and other age-related diseases, including cancer. 
    
    


## Answer


```python
print(answer)
```

    In yeast, the accumulation of acetic acid during stationary phase activates conserved growth‑signaling pathways (e.g., Sch9/Sic1, the AKT/RAS/cAMP axis), drives cells into S‑phase even when nutrients are scarce, and thereby creates **replication stress**.  Acetic acid also lowers extracellular pH, which in yeast and in mammals stimulates mitochondrial signaling and increases **oxidative stress**.  These two stresses—replication and oxidative—are known to accelerate cellular senescence and apoptosis in yeast.
    
    Because the same signaling molecules (AKT, RAS, cAMP, IGF‑1 pathways) and stress responses are conserved in mammals, the mechanisms observed in yeast can be extrapolated to human aging in the following ways:
    
    1. **Chronic low‑pH or acetic‑acid‑like metabolic states** can continuously stimulate growth‑signaling pathways in human cells, promoting cell cycle entry and proliferation in an environment that may lack sufficient nutrients or DNA‑repair capacity, leading to replication stress.
    
    2. **Activation of AKT/MEK‑ERK and IGF‑1 signaling** by low extracellular pH or metabolic acids can increase mitochondrial ROS production, thereby elevating oxidative stress, a hallmark of aging and age‑related diseases.
    
    3. **Replication stress** induced by inappropriate cell‑cycle progression can cause DNA damage, genomic instability, and trigger senescence or apoptosis, contributing to tissue dysfunction and age‑related pathologies such as cancer and neurodegeneration.
    
    4. **Inflammatory responses** triggered by extracellular acidification (as seen in tumors) may further amplify growth signaling and oxidative damage, creating a feed‑forward loop that accelerates aging processes.
    
    Thus, the yeast model suggests that persistent metabolic acidification and the resulting activation of growth‑signaling pathways, coupled with oxidative and replication stress, are conserved mechanisms that can contribute to aging in humans.



```python

```

The answer could be directly checked against the also provided retrieved paper chunks.


```python

```
