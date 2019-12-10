# correctness_bert
A model that uses BERT to evaluate the semantic correctness between SQL and natural language

Trained by giving pairs of (SQL, NL) from WikiSQL and a label of 0(incorrect) or 1(correct).

Dataset is made by labeling the original (SQL, NL) pairs in WikiSQL as correct and shuffled (SQL, NL) pairs, or correct (SQL, NL) pairs with the column names in SQL shuffled as incorrect pairs.

Trained on BERT
