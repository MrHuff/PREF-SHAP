# Explaining Preferences with Shapley Values
Accompanying code for the paper "Explaining Preferences with Shapley Values".

To recreate an experiment, use the "debug_train_model.py" to train a model.

Here there are 2 models to chose from:

1. SGD_krr (GPM) or SGD_ukrr (C-GPM)
2. SGD_krr_pgp (UPM) or SGD_ukrr_pgp (context UPM)

As an easy-to-run example, one could consider the dataset "chameleon", which can be preprocessed by running the load_experiments.py

One then needs to run the following:

1. debug_train_model.py with SGD_krr on the chameleon dataset
2. shap_pipeline.py on the saved model from 1.
3. post_process_plots.py where one selects the saved Shapley values from 2. 


