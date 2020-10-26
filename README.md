# auto_mpg_prediction
This is a End-To-End Projects on Auto Mpg Prediction.

# Steps
1. Dataset: data is collected from UCI ML Repository using url- http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
2. EDA: for exploratory data analysis 6 steps were followed
  -Check for Data type of column
  - Check for null values.
  - Check for outliers
  - Look for the category distribution in categorical columns
  - Plot for correlation
  - Look for new variables
2. Data Preparation: for this process 4 steps were followed
  - Handling Categorical Attribute -> one hot encoding
  - Data cleaning -> Imputer
  - Attribute Addition -> Adding custom transformation (acceleration on power & acceleration on cylinder)
  - Setting up Data Transformation Pipeline for numerical and categorical column.
3. Selection and Training Models
  - Select and Train a few Algorithm (Linear Regression, Decision Tree, Random Forest, SVM regressor)
  - Evaluate using Mean Squared Error
  - Model Evaluation using Cross Validation
  - Hyperparameter Tuning using GridSearchCV
  - Evaluate the Final System on test data
  - Saving the Model
 4. Deploying Trained Model
  - Start a Flask Project.
  - Set up a dedicated environment with dependencies installed using pip.
    Packages to install:
    -pandas
    -numpy
    -sklean
    -flask
    -matplotlib
    -gunicorn
    -seaborn
  - Create a quick flask application to test a simple endpoint.
  - Define a function to that accepts data from the POST request and return the prediction using a helper module.
  - Test the endpoint using requests package.
  

![Screenshot](/readme_resource/s1.png)
![Screenshot](/readme_resource/s2.png)

