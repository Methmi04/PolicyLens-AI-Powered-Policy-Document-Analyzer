## PolicyLens — Policy Document Analyzer 

PolicyLens is an AI-powered web application designed to analyze complex policy documents, extract key insights.

It is built to work with **any policy or structured document**, making it a flexible tool for policy analysis, research, and decision-making.



##  Features

###  1. Document Upload
- Upload PDF or TXT documents
- Automatic text extraction

###  2. NLP-Based Summarisation
- Extracts:
  -  Goals / Objectives
  -  Key Strategies / Actions
  -  Overall Direction
- Uses:
  - TF-IDF Vectorization
  - Cosine Similarity
  - TextRank Algorithm

###  3. Scenario-Based Adaptation
- Generate alternative policy drafts based on different scenarios
- Easily extendable for custom use cases

###  4. AI-Powered Draft Generation
- Uses large language models
- Produces structured outputs:
  - Goals
  - Strategies
  - Direction

###  5. Fallback Mechanism
- Template-based generation if API is unavailable
- Ensures system reliability



## System Workflow

Document Upload  
→ Text Extraction  
→ Text Cleaning  
→ NLP Summarisation  
→ Scenario Selection  
→ Draft Generation  
→ Output



## Technologies Used

- **Backend:** Flask (Python)
- **NLP:** Scikit-learn (TF-IDF, Cosine Similarity)
- **PDF Processing:** PyPDF2
- **Frontend:** HTML, CSS, JavaScript




