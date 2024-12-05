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

The data set was cleaned and preprocessed to provide the highest efficiency for our machine learning model. The transform.py script cleans the data by eliminating 
null or duplicate values. The process of standardization was also taken so all features were weighted equally to prevent large variations in the data. As previously 
referenced, Kaggle API was used to access the dataset. Our final result was an accurate identification of trends that allow differentiation between fraudulent and 
legitimate transactions. 

In order to scale this project up we would need a larger dataset that encompasses a broader variety of transaction features, including more diverse merchant categories, geographic regions, and user demographics, to better capture the complexity of real-world credit card transactions. A significantly higher proportion of fraudulent transactions would also be needed to address the inherent class imbalance observed in this fraud detection. A dataset with less class imbalance would enable the model to learn more nuanced patterns and improve its ability to identify subtle fraudulent behaviors. Additionally, using a dataset composed of real-world credit card fraud cases would enhance the model's applicability and performance, which would allow for this project to have a further reach. Real-world data often contains noise, anomalies, and contextual variations that are difficult to replicate synthetically but ultimately are needed when creating a model capable of generalizing effectively. Overall, our project demonstrated many of the tools and approaches used in previous labs so while our project was a success, it cannot be categorized as innovative. We did not face any major technical issues or limitations while creating the pipeline. The next steps for this project would be to curate a larger dataset that will allow for a more accurate machine learning model.


**Visualizations from Tableau**

<img width="302" alt="image" src="https://github.com/user-attachments/assets/df8b781c-e72a-478d-a3f5-ff4f77dd7fa2">



<img width="614" alt="image" src="https://github.com/user-attachments/assets/b07b5ead-94ef-4e03-b78d-8969d55847f5">


**Infographic**


<img width="592" alt="image" src="https://github.com/user-attachments/assets/680b0318-e5c9-4947-bc95-e603bcd241f0">


