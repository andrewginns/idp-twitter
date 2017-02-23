# idp-twitter
Author classification through data mining. 
Written in python using .csv files of tweets downloaded through the tweepy python module.
Optional:
- Modify 'Create_Vectors.py' and 'cos_sim' to change settings:
  - Enabling/disabling GUI
  - GUI Tweaks
  - Using different weighting measures for words in vectors

Usage:
1) Modify Create_vectors.py so that the vocabulary, authors and test csv's are correct
2) In the project directory open the terminal and run 'python Create_Vectors.py'

3ai) Run cos_sim with TF_IDF=0 and cosine =0

3aii) Run cos_sim with TF_IDF=0 and cosine =1

3bi) Run cos_sim with TF_IDF=1 and cosine =0

3bii) Run cos_sim with TF_IDF=1 and cosine =1

3c) Run Create_SVD
3ci) Run knn_sim.py

