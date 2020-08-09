
# ALEX : Explainable_Active_Learning

This repository contains the official code for the paper : # "ALEX: Active Learning Based Enhancement of a Model's EXplainability"

Since obtaining manual annotations on data is an expensive and time-consuming process, an active learning (AL) algorithm seeks to construct an effective classifier with a minimal number of labeled examples in a bootstrapping manner. After starting with a seed set of labeled instances, an AL algorithm predicts the most relevant batch of instances that should be annotated next in order to improve on the model effectiveness. While heuristics that work well in practice, among others, include those of selecting points for which the current classifier model is most uncertain of, there has been no empirical investigation to see if these heuristics lead to models that are more interpretable to humans. In the era of data-driven learning, this is an important research direction to pursue. This paper describes our work-in-progress towards developing an AL selection function (ALEX) that in addition to model effectiveness also seeks to improve on the interpretability of a model during the bootstrapping steps. Concretely speaking, our proposed selection function trains an 'explainer' model in addition to the classifier model, and favours those instances where a different part of the data is used, on an average, to explain the predicted class. Initial experiments exhibited encouraging trends in showing that such a heuristic can lead to developing more effective and more explainable end-to-end data-driven classifiers. 


Instructions to run :

Requires python 3.6

```markdown
      pip install -r requirements.txt
      Run python AL_%dataset_name.py
```
