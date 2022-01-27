# CMPT459 Group Project

Hazem Hisham, Harry Preet Singh,     Jiongyu Zhu

## About

Predicting the outcome group of a COVID-19 case. 

## File Structure
```bash
data-mining-459
├── README.md
├── envrionment.yml
└── milestone1
    └── code
        ├── data
        │   ├── cases_2021_train.csv
        │   ├── cases_2021_test.csv
        │   └── location_2021.csv
        ├── plots
        │   ├── task1.3
        │   └── task1.x
        ├── results
        │   ├── cases_2021_test_processed.csv
        │   ├── cases_2021_test_processed_features.csv
        │   ├── cases_2021_train_processed.csv
        │   ├── cases_2021_train_processed_features.csv
        │   └── location_2021_processed.csv
        └── src
            ├── eda.ipynb
            ├── helper1.py
            ├── helper2.ipynb
            └── main.py
```

## Setup 

1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Run the anaconda command prompt through start menu
3. Navigate to the project directory.
4. Create your environment from the provided environment file : ```conda env create -f environment.yml```
5. Activate environment: ```conda activate cmpt459```



## Datasets

Download datasets from the [course repo](https://github.com/shumanpng/CMPT459-D100-SPRING2022/tree/main/dataset) into ```/milestone1/code/data```

## Running src files
```src/eda.ipynb → Cell → Run All``` and also ```python src/main.py``` 






