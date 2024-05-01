# Neural Nets Project


> A blog post about this project can be found [here](https://medium.com/@acweingrad/robots-in-the-kitchen-neuralnourishment-6be279f3c125)


## Local Development Setup
1. Clone this repository locally
2. `cd` into that directory and install the project requirements with `pip -r requirements.txt`
3. Download a zip of the [RecipeNLG dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg)

    **Using a browser**
    1. Use the link above to download the `.zip` file locally
    2. Move the `.zip` file into this repository's base directory

    **Using the CLI**
    1. Configure the kaggle CLI and generate a Kaggle token, if needed (instructions [here](https://www.kaggle.com/docs/api#authentication))
    2. Run `kaggle datasets download paultimothymooney/recipenlg` to download the dataset into this repository's base directory

4. Run `unzip recipenlg.zip -d RecipeNLG/ && rm recipenlg.zip` to unzip the dataset, move it into an untracked directory, and delete the `.zip` file
6. Start contributing to the project

## About `RecipeNLG/`
This directory contains the raw content from the [RecipeNLG dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg). This includes a 2.29GB `.csv` file with the content of +2.2 million recipes, along with the web scraping scripts and named entity recognition tasks that produced this corpus. The abbreviated structure is as follows:
```
RecipeNLG/
├── RecipeNLG_code/
│   ├── eval/             # Notebooks for evaluating the dataset
│   ├── generation/       # Text generation and language model fine-tuning
│   ├── ner/              # Notebooks for named entity recognition
│   ├── recipes_spider/   # Spiders for crawling recipe websites
│   ├── scraping-scripts/ # Data cleaning, analysis, and preprocessing
|   ├── README.md
│   └── requirements.txt
├── RecipeNLG_dataset.csv
├── RecipeNLG_license.png
└── RecipeNLG_paper.pdf
```
We do not track this folder with git due to its large size and static nature.
