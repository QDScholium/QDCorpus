'''
A simple script to retrieve the arXiv dataset from Kaggle using the KaggleHub API.
Ideally, this should only be ran once, as the dataset is large and will take a while to download.
'''


import kagglehub

# Download latest version
path = kagglehub.dataset_download("Cornell-University/arxiv")
print("Path to dataset files:", path)


