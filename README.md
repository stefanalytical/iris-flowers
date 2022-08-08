# [Iris Flowers - Machine Learning](https://github.com/stefanalytical/iris-flowers.git)
 
This is my first project using machine learning.
 
## About
 
I chose the iris flowers dataset because it is considered a great starting point for any data scientist looking to take on their first machine learning project. It is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper _The use of multiple measurements in taxonomic problems_ as an example of linear discriminant analysis.
<br />
In this project I:

1. Installed SciPy as it as the most useful package for machine learning.
2. Loaded the dataset and used statistical summaries and data visualizations to understand it.
3. Created 5 ML models and picked the optimal one while ensuring the accuracy is reliable.

## ML Models

1. **Logistic Regression 'LR'**. Simple linear algorithm. (Estimated accuracy score of 94.2%)
2. **Linear Discriminant Analysis 'LDA'**. Simple linear algorithm. (Estimated accuracy score of 97.5%)
3. **Classification and Regression Trees 'CART'**. Nonlinear algorithm. (Estimated accuracy score of 93.3%)
4. **Gaussian Naive Bayes. Nonlinear algorithm 'NB'**. (Estimated accuracy score of 95%)
5. **Support Vector Machines. Nonlinear algorithm 'SVM'**. (Estimated accuracy score of 98.3%)

## Install and Run

This project was created using ScipPy and Python 3.10. If you need to download or update SciPy, you can do so on their [website](https://scipy.org/install/). If you need to download or update Python, you can do so [here](https://www.python.org/downloads/).

1. First, clone the repository: [iris-flowers](https://github.com/stefanalytical/iris-flowers.git) and save it to your machine using the command prompt. Navigate to the [iris-flowers](https://github.com/stefanalytical/iris-flowers.git) directory.

2. A virtual environment is required to run the program. Creation and activation will depend on the system you are using. Using the command prompt:

**To create the virtual environment:**

`python -m venv venv` or `python3 -m venv venv` <br />
or <br />
`python -m venv env` or `python3 -m venv env`

**To activate the virtual environment:**

Windows: <br />
<br />`venv/Scripts/Activate` <br />

Mac: <br />
<br />`source venv/bin/activate` <br />
or <br />
`source env/bin/activate`

3. Once the virtual environment is activated, please install the project's dependencies found in the requirements.txt folder.

```bash
pip install -r requirements.txt
```
or
```bash
pip3 install -r requirements.txt
```
4. Run the file.
```bash
main.py
```

## Inspiration

This project was created by following a tutorial by Jason Brownlee, PhD., author of _Machine Learning Mastery With Python_.
