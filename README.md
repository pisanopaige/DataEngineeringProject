The title of this project is "Detecting and Plotting the Number of Faulty Credit Card Transactions Against Legitimized Transactions Using Machine Learning".

The objective of this project is to undestand what patterns exist that differentiate fraud from legitimate transactions and determine the frequency of fraudulent 
transactions detected compared to those of legitimate transactions. The goal we set out to accomplish through the course of this project is identifying key 
patterns of fraud to explore how data-driven models can aid in the enhancement of transaction security and furthermore utilize data visualizations to highlight
these trends in a more digestable format.

This dataset includes both training and testing sets with over 500,000 simulated transactions across two years, each datapoint includes various features related
to each transaction, such as the amount, time, location, merchant details, and credit card owner information. Additionally, each datapoint is labeled as either 
legitimate or fraudulent, using 0 or 1, respectively. The format of this dataset is ideal for using supervised learning techniques to train a model. 

This project will use the Batch – ML – Visualization pipeline to detect fraudulent transactions. The data ingestion portion of this piplines requires Kaggle API 
to download the dataset. The transformation portion of this pipline will require Pandas and NumPy to clean, scale, and manipulate the data. The machine learning 
portion of this pipeline requires a Random Forest Classifer which is best-suited for imbalanced datasets. And the visualization portion requires Tableau. To
orchestrate and execute the entire pipeline, Apache Airflow is used.

The data set is good quality with respects to the goals of our project. The volume of transactions the data set possesses is adequate for our machine learning 
efforts. The variety of entries is adequate because while it is simulated data, it accurately reflects real world transactions. The only downside of this data is 
the lack of fraudulent transactions within the dataset, the proportion of fraudulent to legitimate transactions is not equal and therefore presents more of 
challenge when training the Ml algorithm to detect.

