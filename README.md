# Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

This project implements a comprehensive machine learning pipeline on the Heart Disease UCI dataset. The pipeline consists of data preprocessing, feature selection, dimensionality reduction, model training, evaluation, and deployment. Logistic Regression, Decision Trees, Random Forest, and SVM are used for classification, and K-Means and Hierarchical Clustering for unsupervised learning. Furthermore, a Streamlit UI is built for user interaction and deployed via Ngrok, and the project is hosted on GitHub.

## Requirements

- Python 3.11

## Install Python and create environment

1. Download and install Python 3.11 from [here](https://www.python.org/downloads/release/python-3119/)
2. Create a new environment using the following command:

```bash
$ python -m venv myenv
```

3. Activate the environment:

```bash
$ source myenv\Scripts\activate
```

4. Install the required packages using the following command:

```bash
$ pip install -r requirements.txt
```

## Run the project

1. Run the following command to start the Streamlit app:

```bash
$ streamlit run ui/app.py --server.port 8501
```

2. Open [http://localhost:8501](http://localhost:8501) in your web browser.
