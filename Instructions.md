
# CS-175-Project
UCI CS 175 Project\
This file is the same as README.MD

This version is for *Submission*\
This version is tested on: Ubuntu 16.04, Windows
\

We put up a test with according data. Use `python3 main.py`\
This test used a saved model.It use 30-moment predictions over 1000 minutes of test data, the same as Figure 4 in the report.

modules and environments needed:

Keras\
keras-tcn\
pandas\
numpy\
statsmodels\
tensorflow >= 2.0\
sklearn\
matplotlib\
and maybe other environement

(optional) if you want to train your own model:

To train a new model, set loadModel to None and specify the model type w/ model = 1 or model = 2\
`model = 1` is the first architecture\
`model = 2` for the second architecture\
specify the number of moments that the model is to use as well as that is required for the model to be made and uncomment Tcn.train()
