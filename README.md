# AI-Powered Task Management System

## Project Overview
This project aims to design and develop an intelligent task management system that leverages Natural Language Processing (NLP) and Machine Learning (ML) techniques to automatically classify, prioritize, and assign tasks to users based on their behavior, deadlines, and workloads.

---

## Features
- **Task Classification:** Classifies tasks into categories using NLP and ML models (Naive Bayes, SVM).
- **Priority Prediction:** Predicts task priority using ensemble models like Random Forest and XGBoost.
- **Workload Balancing:** Assigns tasks intelligently by considering user workloads and deadlines.
- **Interactive Dashboard:** Provides summary insights and performance metrics (planned for final release).

---

## Project Timeline

| Week                      | Tasks                                                                                          |
|---------------------------|------------------------------------------------------------------------------------------------|
| Week 1                    | - Collect task datasets (e.g., Trello, Jira APIs, synthetic data).                             |
|                           | - Perform exploratory data analysis (EDA) and data cleaning.                                  |
|                           | - Apply NLP preprocessing (tokenization, stemming, stopword removal).                         |
| Week 2                    | - Feature extraction using TF-IDF and word embeddings (Word2Vec/BERT).                        |
|                           | - Implement task classification using Naive Bayes and SVM.                                   |
|                           | - Set up version control with GitHub.                                                        |
|                           | - Evaluate models using accuracy, precision, recall metrics.                                 |
| Mid-Project Review (End of Week 2) | - Completed cleaned and preprocessed dataset.                                  |
|                           | - Trained and evaluated task classifiers (Naive Bayes/SVM).                                  |
|                           | - Completed EDA visualizations.                                                              |
| Week 3                    | - Develop priority prediction model (Random Forest/XGBoost).                                 |
|                           | - Integrate workload balancing logic (heuristic or ML-based).                                |
|                           | - Perform hyperparameter tuning with GridSearchCV.                                           |
| Week 4                    | - Finalize models for classification and priority prediction.                                |
|                           | - Design dashboard mockup or generate output summary.                                        |
|                           | - Compile performance metrics, visualizations, and results.                                  |
| Final Review (End of Week 4) | - Completed and finalized all models.                                                  |
|                           | - Prepared summary dashboard or mockup.                                                     |
|                           | - Prepared final report and documentation.                                                  |

---

## Technologies Used
- Programming Languages: Python
- NLP Libraries: NLTK, spaCy, Transformers (for BERT embeddings)
- ML Frameworks: scikit-learn, XGBoost
- Data Visualization: Matplotlib, Seaborn
- Version Control: Git & GitHub
- APIs: Trello API, Jira API (for dataset collection)

---

## Getting Started

### Prerequisites
- Python 3.x
- Install required libraries:  
```bash
pip install -r requirements.txt
/ai-task-manager
│
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── src/                    # Source code (preprocessing, models, utils)
├── outputs/                # Model outputs, visualizations, reports
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── LICENSE

