# COVID-19 Case Prediction Using Machine Learning

**Abdulhamid Alaboud · Ahmed Alaglan**    
Department of Computer Science  
Jazan University, Saudi Arabia

## Overview
This project implements a supervised machine learning pipeline for classifying COVID-19 infection status using Logistic Regression and Random Forest models. The system achieves 99.69% accuracy with perfect recall, providing reliable infection classification based on global epidemiological data.

## Key Features
- Two-stage hybrid classification model (Logistic Regression + Random Forest)
- Comprehensive exploratory data analysis with visualizations
- CPIS (Composite Pandemic Impact Score) metric for regional comparison
- Feature importance analysis and correlation studies
- Global case analysis across WHO regions

## Results
- **Accuracy:** 99.69%
- **Recall:** 100%
- **Precision:** 98.96%
- **F1-Score:** 99.48%

## Dataset
**Source:** Corona Virus Report on Kaggle  
🔗 [https://www.kaggle.com/datasets/imdevskp/corona-virus-report](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Beginners)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abdulhamid-Alaboud/Covid19-Case-Prediction/blob/main/Covid_19_Case_Prediction.ipynb)

1. Click the "Open In Colab" button above
2. Upload your dataset to Colab's file system
3. Run all cells (Runtime → Run all)

### Option 2: Local Jupyter Environment
1. Download the dataset from Kaggle (free account required)
2. Place `covid_19_clean_complete.csv` in the `data/` folder
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `jupyter notebook Covid_19_Case_Prediction.ipynb`

## Methodology
1. **Data Preprocessing:** Missing value handling, normalization, and label encoding
2. **Exploratory Analysis:** Monthly trends, correlation heatmaps, and geographic distribution
3. **Feature Engineering:** CPIS metric development for regional severity scoring
4. **Model Development:** Two-stage hybrid pipeline with comparative evaluation

## Repository Structure
```
Covid19-Case-Prediction/
├── Covid_19_Case_Prediction.ipynb  # Main analysis notebook
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── .gitignore                      # Git exclusion rules
└── data/                           # Dataset directory
    └── .gitkeep                    # Maintains folder structure
```

## Requirements
```txt
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

## Citation
```bibtex
@software{alaboud2024covid,
  title = {COVID-19 Case Prediction Using Machine Learning},
  author = {Alaboud, Abdulhamid and Alaglan, Ahmed},
  year = {2024},
  url = {https://github.com/Abdulhamid-Alaboud/Covid19-Case-Prediction}
}
```

## License
MIT License - See LICENSE file for details.

## Contact
For questions or collaborations, open an issue in this repository.
```
