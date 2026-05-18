# cURL Examples for the Scientific Paper Summarization API

This document provides practical `curl` examples to demonstrate the core features of the API.

*These examples use `jq` to pretty-print the JSON output. You can install it via `brew install jq` or `sudo apt-get install jq`, or simply remove `| jq` from the commands.*

---

## 1. Checking API Status

Before making requests, you can check the service's health and see which summarization prompts are available.

#### Health Check
Verify that the API is running and can connect to the AI model.

```bash
curl "http://localhost:8000/health" | jq
```

#### List Available Prompts
Get a list of all supported summarization strategies (`prompt_key` values).

```bash
curl "http://localhost:8000/prompts" | jq
```

---

## 2. Generating Summaries

The primary endpoint is `/summarize/`. The API automatically chooses the best summarization strategy based on the number of papers you provide.

### Example A: Quick Summary (Fewer than 6 Papers)

When you provide a small number of papers, the API defaults to generating a concise summary.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "med_img_01",
        "title": "Deep Learning for Medical Image Analysis",
        "abstract": "We present a novel deep learning approach for automated medical image analysis, achieving state-of-the-art performance on multiple diagnostic tasks with 92% accuracy."
      },
      {
        "id": "ethics_01",
        "title": "Ethical Considerations in AI-Assisted Diagnosis",
        "abstract": "This paper examines the ethical implications of using artificial intelligence in medical diagnosis, proposing guidelines for responsible implementation in clinical settings."
      },
      {
        "id": "federated_01",
        "title": "Federated Learning for Privacy-Preserving Medical AI",
        "abstract": "We introduce a federated learning framework that enables collaborative training of medical AI models while preserving patient privacy and complying with healthcare regulations."
      }
    ],
    "topic_name": "AI in Medical Diagnostics"
  }' | jq
```

### Example B: Detailed Literature Review (6+ Papers)

When you provide six or more papers, the API automatically switches to the `lit_review` prompt to generate a more comprehensive, multi-paragraph synthesis suitable for academic work.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "fin_ml_01",
        "title": "Machine Learning in Finance: A Survey",
        "abstract": "This paper provides a comprehensive survey of machine learning algorithms applied to algorithmic trading, portfolio management, and risk assessment, highlighting key trends and challenges."
      },
      {
        "id": "fin_dl_02",
        "title": "Deep Learning for Credit Scoring and Default Prediction",
        "abstract": "Our research shows that deep neural networks can improve the accuracy of credit risk evaluation by up to 15% compared to traditional logistic regression models."
      },
      {
        "id": "fin_nlp_03",
        "title": "Sentiment Analysis of Financial News Using Natural Language Processing",
        "abstract": "We developed NLP techniques to extract market sentiment from financial news articles, demonstrating a strong correlation between sentiment scores and subsequent market movements."
      },
      {
        "id": "fin_block_04",
        "title": "Applications of Blockchain Technology in Modern Banking",
        "abstract": "This study explores the application of blockchain and distributed ledger technology for creating secure, transparent, and efficient financial transaction systems."
      },
      {
        "id": "fin_quant_05",
        "title": "The Threat of Quantum Computing to Financial Cryptography",
        "abstract": "An analysis of how quantum algorithms like Shor’s algorithm threaten the security of current encryption methods used to protect financial systems and digital assets."
      },
      {
        "id": "fin_eth_06",
        "title": "A Framework for Ethical AI in Financial Decision Making",
        "abstract": "We examine the ethical implications of AI-driven financial products and propose a new framework for ensuring fairness, transparency, and accountability in automated financial services."
      }
    ],
    "topic_name": "The Impact of AI and Technology on Modern Finance"
  }' | jq
```

---

## 3. Manually Selecting a Strategy

You can override the automatic selection by explicitly setting a `prompt_key`. This is useful if you want a literature review for a small number of papers, or a brief summary for a large set.

This example forces the use of the `two_paragraph` strategy for only two papers.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "neuro_01",
        "title": "Neural Networks for Advanced Time Series Prediction",
        "abstract": "We propose a novel neural network architecture for time series forecasting that effectively combines LSTM layers with attention mechanisms to capture complex temporal dependencies."
      },
      {
        "id": "dl_comp_02",
        "title": "A Comparative Analysis of Deep Learning Models for Sequential Data",
        "abstract": "This study performs a comparative analysis of various deep learning approaches, including RNNs, LSTMs, and Transformers, for sequential data processing to identify optimal architectures for different tasks."
      }
    ],
    "topic_name": "Deep Learning for Time Series",
    "prompt_key": "two_paragraph"
  }' | jq
```

### Example C: Scholar Profile Summary

For scholar-profile pages, pass the author's name as `topic_name`, include richer paper metadata when available, and choose an explicit scholar prompt. Do not use bare `prompt_key: "scholar"`; use `scholar-overview` or `scholar-narrative`.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "23021531",
        "title": "DIANA-TarBase v8: a decade-long collection of experimentally supported miRNA–gene interactions",
        "abstract": "DIANA-TarBase v8 (http://www.microrna.gr/tarbase) is a reference database devoted to the indexing of experimentally supported microRNA (miRNA) targets. Its eighth version is the first database indexing >1 million entries, corresponding to ~670 000 unique miRNA-target pairs. The interactions are supported by >33 experimental methodologies, applied to ~600 cell types/tissues under ~451 experimental conditions. It integrates information on cell-type specific miRNA-gene regulation, while hundreds of thousands of miRNA-binding locations are reported. TarBase is coming of age, with more than a decade of continuous support in the non-coding RNA field. A new module has been implemented that enables the browsing of interactions through different filtering combinations. It permits easy retrieval of positive and negative miRNA targets per species, methodology, cell type and tissue. An incorporated ranking system is utilized for the display of interactions based on the robustness of their supporting methodologies. Statistics, pie-charts and interactive bar-plots depicting the database content are available through a dedicated result page. An intuitive interface is introduced, providing a user-friendly application with flexible options to different queries.",
        "year": "2017",
        "authors": "Dimitra Karagkouni; Maria D. Paraskevopoulou; Serafeim Chatzopoulos; Ioannis S. Vlachos; Spyros Tastsoglou; Ilias Kanellos; Dimitris Papadimitriou; Ioannis Kavakiotis; Sofia Maniou; Giorgos Skoufos; Thanasis Vergoulis; Theodore Dalamagas 0001; Artemis G. Hatzigeorgiou",
        "topics": [
          "MicroRNA in disease regulation",
          "Cancer-related molecular mechanisms research",
          "RNA modifications and cancer"
        ],
        "contribution_roles": [
          "Conceptualization",
          "Data curation",
          "Funding acquisition",
          "Investigation",
          "Methodology",
          "Writing - review and editing"
        ]
      },
      {
        "id": "23022082",
        "title": "DIANA-mirExTra v2.0: Uncovering microRNAs and transcription factors with crucial roles in NGS expression data",
        "abstract": "Differential expression analysis (DEA) is one of the main instruments utilized for revealing molecular mechanisms in pathological and physiological conditions. DIANA-mirExTra v2.0 (http://www.microrna.gr/mirextrav2) performs a combined DEA of mRNAs and microRNAs (miRNAs) to uncover miRNAs and transcription factors (TFs) playing important regulatory roles between two investigated states. The web server uses as input miRNA/RNA-Seq read count data sets that can be uploaded for analysis. Users can combine their data with 350 small-RNA-Seq and 65 RNA-Seq in-house analyzed libraries which are provided by DIANA-mirExTra v2.0. The web server utilizes miRNA:mRNA, TF:mRNA and TF:miRNA interactions derived from extensive experimental data sets. More than 450 000 miRNA interactions and 2 000 000 TF binding sites from specific or high-throughput techniques have been incorporated, while accurate miRNA TSS annotation is obtained from microTSS experimental/in silico framework. These comprehensive data sets enable users to perform analyses based solely on experimentally supported information and to uncover central regulators within sequencing data: miRNAs controlling mRNAs and TFs regulating mRNA or miRNA expression. The server also supports predicted miRNA:gene interactions from DIANA-microT-CDS for 4 species (human, mouse, nematode and fruit fly). DIANA-mirExTra v2.0 has an intuitive user interface and is freely available to all users without any login requirement.",
        "year": "2016",
        "authors": "Ioannis S. Vlachos; Thanasis Vergoulis; Maria D. Paraskevopoulou; Filopoimin Lykokanellos; Georgios K. Georgakilas; Penny Georgiou; Serafeim Chatzopoulos; Dimitra Karagkouni; Foteini Christodoulou; Theodore Dalamagas 0001; Artemis G. Hatzigeorgiou",
        "topics": [
          "MicroRNA in disease regulation",
          "RNA modifications and cancer",
          "CRISPR and Genetic Engineering"
        ],
        "contribution_roles": [
          "Conceptualization",
          "Data curation",
          "Software",
          "Supervision"
        ]
      },
      {
        "id": "62164348",
        "title": "BIP! Finder",
        "abstract": "Due to the rapidly increasing number of scientific articles, finding valuable work for further research has become tedious and time consuming. To alleviate this issue, search engines have used citation-based article impact ranking. However, most engines rely on very simplistic impact measures (usually the citation count) and make the problematic assumption that there is a one-size-fits-all impact measure. To address these problems, we present BIP! Finder, a search engine that facilitates the identification of valuable articles by exploiting two different impact measures, each capturing a different aspect of the article impact. In addition, BIP! Finder provides many useful features (article comparison, intuitive visualisations, article bookmarking mechanism, etc.) making it a powerful addition to the researcher's toolbox.",
        "year": "2019",
        "authors": "Thanasis Vergoulis; Serafeim Chatzopoulos; Ilias Kanellos; Panagiotis Deligiannis; Christos Tryfonopoulos; Theodore Dalamagas 0001",
        "topics": [
          "Scientific Computing and Data Management"
        ],
        "contribution_roles": [
          "Methodology"
        ]
      },
      {
        "id": "98501046",
        "title": "VeTo: Expert Set Expansion in Academia",
        "abstract": "Expanding a set of known domain experts with new individuals, that have similar expertise, is a problem with many practical applications (e.g., adding new members to a conference program committee). In this work, we study this problem in the context of academic experts and we introduce VeTo, a novel method to effectively deal with it by exploiting scholarly knowledge graphs. In particular, VeTo expands the given set of experts by identifying researchers that share similar publishing habits with them, based on a graph analysis approach. Our experiments show that VeTo is more effective than existing techniques that can be applied to deal with the same problem.",
        "year": "2020",
        "authors": "Thanasis Vergoulis; Serafeim Chatzopoulos; Theodore Dalamagas 0001; Christos Tryfonopoulos",
        "topics": [
          "Expert finding and Q&A systems",
          "Advanced Graph Neural Networks"
        ],
        "contribution_roles": []
      },
      {
        "id": "60515736",
        "title": "BIP4COVID19: Releasing impact measures for articles relevant to COVID-19",
        "abstract": "Since the beginning of the 2019-20 coronavirus pandemic, a large number of relevant articles has been published or become available in preprint servers. These articles, along with earlier related literature, compose a valuable knowledge base affecting contemporary research studies, or even government actions to limit the spread of the disease and treatment decisions taken by physicians. However, the number of such articles is increasing at an intense rate making the exploration of the relevant literature and the identification of useful knowledge in it challenging. In this work, we describe BIP4COVID19, an open dataset compiled to facilitate the coronavirus-related literature exploration, by providing various indicators of scientific impact for the relevant articles. Additionally, we provide a publicly accessible Web interface on top of our data, allowing the exploration of the publications based on the computed indicators.",
        "year": "2020",
        "authors": "Vergoulis, Thanasis; Kanellos, Ilias; Chatzopoulos, Serafeim; Karidi, Danae Pla; Dalamagas, Theodore",
        "topics": [
          "Academic Publishing and Open Access",
          "COVID-19 diagnosis using AI",
          "scientometrics and bibliometrics research"
        ],
        "contribution_roles": []
      }
    ],
    "topic_name": "Serafeim Chatzopoulos",
    "prompt_key": "scholar-overview"
  }' | jq
```

This scholar example demonstrates these optional metadata fields per paper:

- `year`
- `authors`
- `topics`
- `contribution_roles`
