# Explainable_Active_Learning

Install LIME:

1. git clone https://github.com/marcotcr/lime.git
2. cd lime
3. Either run: python setup.py or pip install lime

Run the notebooks:

1. cd lime/doc/notebooks/
2. Copy the following notebooks to this path:
    a. Run the cells of Random_Active_Learning_based_predictions.ipynb for "Random Query Generator"
    b. Run the cells of Explanation_Based_experiments.ipynb for "Explanation Based Query Generator" and uncertainity based query sampling
    c. Run the cells of Explanataion_based_Algorithm_2.ipynb for "Implementation of lime based Algorithm".



=================================================================================
For running on the server:

1. pip install shap
2. Unzip mnist_AL.zip
3. Place SHAP_RANDOM_VS_UNCERTAINITY.py inside mnist_AL.
4. Run SHAP_RANDOM_VS_UNCERTAINITY.py
5. The accuracy and shap plots will be saved inside the same folder as the script
