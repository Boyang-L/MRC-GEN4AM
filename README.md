# MRC-GEN4AM
Code release for paper [Argument mining as a multi-hop generative machine reading comprehension task](https://aclanthology.org/2023.findings-emnlp.724/)

# Prerequisites
python                    3.9.0   
argparse                  1.4.0  
dgl                       1.0.1   
numpy                     1.24.2   
scikit-learn              1.2.2  
torch                     2.0.0  
transformers              4.27.4  
tqdm                      4.65.0  

# Usage
The code for different task are in different directory. For example, rct_acc contains the code for the ACC task on the AbstRCT dataset.  
To conduct the experiments, please run the batch_train.sh in scripts file. For the PE dataset, you should prepare the data using the code in *datapreprocessing*

# Citation
@inproceedings{liu-etal-2023-argument,
    title = "Argument mining as a multi-hop generative machine reading comprehension task",
    author = "Liu, Boyang  and
      Schlegel, Viktor  and
      Batista-Navarro, Riza  and
      Ananiadou, Sophia",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.724",
    doi = "10.18653/v1/2023.findings-emnlp.724",
    pages = "10846--10858",
    abstract = "Argument mining (AM) is a natural language processing task that aims to generate an argumentative graph given an unstructured argumentative text. An argumentative graph that consists of argumentative components and argumentative relations contains completed information of an argument and exhibits the logic of an argument. As the argument structure of an argumentative text can be regarded as an answer to a {``}why{''} question, the whole argument structure is therefore similar to the {``}chain of thought{''} concept, i.e., the sequence of ideas that lead to a specific conclusion for a given argument (Wei et al., 2022). For argumentative texts in the same specific genre, the {``}chain of thought{''} of such texts is usually similar, i.e., in a student essay, there is usually a major claim supported by several claims, and then a number of premises which are related to the claims are included (Eger et al., 2017). In this paper, we propose a new perspective which transfers the argument mining task into a multi-hop reading comprehension task, allowing the model to learn the argument structure as a {``}chain of thought{''}. We perform a comprehensive evaluation of our approach on two AM benchmarks and find that we surpass SOTA results. A detailed analysis shows that specifically the {``}chain of thought{''} information is helpful for the argument mining task.",
}
