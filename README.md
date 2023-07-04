# Synthetic-data-thesis
Case study which investigates the opportunities of synthetic data at governmental institution UWV. Currently, they face privacy challenges concerning their client data.
For now, this is solved by anonimising the data. However, this comes with its risks and might not be useful. Therefore, synthetic data generation is considered and implemented as another anonymization technique to discover its benefits and disadvantages for our client data. In this way, the privacy of our clients is still maintained while analyzing their information using several AI models.

# Methodology
## Part 1: Generating synthetic data from a use case WW naar Bijstand using company's IT server

The use case has originally nearly 2 million observations where its information is collected from the years 2019 to 2022 (full years). All of our clients, who were part of our unemployment benefit population, belonged to the data. For each client, however, it differs how many observations are present. This is highly depended on how long someone's receives such benefit. 

## Part 2: Generating synthetic data from four public datasets using LISA.
Four example datasets:
1. Iris as an initial experiment dataset.
2. Covertype: around half million observations.
3. Adult
4. Bank
5. Metro

# Experimental set-up
Two seperate researches will run in parallel and apply the following techniques:
- Decision tree based models: Adversarial Random Forest & CART
- Deep generative models: CTGAN, TVAE & TABDDPM

The evaluation procedure is similar for both and contains the following metrics:
- Applying Machine Learning model to both original and synthetic data and compare performance.
- Comparison based on statistical properties such as distribution, KStest, etcera.

# Functionalities

## How does the pipeline code work?
The pipeline works in two steps:
1. First, for each dataset, it is important that
  ``` bash
  python Pipeline_models.py [dataset] [model] [nr_trials] [outliers]
  ```
2. Second
  ``` bash
  python Evaluation_procedure.py [dataset] [metrics|visuals]
  ```
  ``` bash
  python Evaluation_procedure.py [dataset] 
  ```
## How to add your own data?

## How to structure your own repository?

# Requirements

## What type of data can be used?
Tabular data consisting of numerical and categorical variables only. These may contain strings, numbers and characters. The pipeline already prepares the dataset for each model for you and works with missing values as well. 

## In what format is the data permitted?
The data can only be added as .csv files to the repository, where each row can either be comma separated or semicolon. 


## How much CPU is needed?
The pipeline was tested on LISA (Dutch super computer). (how much CPU was granted for me during research?)

# Dependencies
These are enclosed in generative_models.yml file which can be installed once a new virtual environment is hosted.
The following steps must be followed:
1. First, you can create a new virtual conda environmant like:
  
2. Second, you can load the requirements file:
  
