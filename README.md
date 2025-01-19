### PROJECT TITLE: Customer Segmentation & Recommendation System
> ${\color{mediumblue}Author\space :\space Atirek\space Khare}$

> **EXECUTIVE SUMMARY**
> 
> This CAPSTONE project aims to develop a Customer Segmentation & Recommendations system using multiple algorithms and strategies available in Machine Learning. It compares the performance of the algorithms to
> identify the best-performing algorithm for the subset of the dataset, generate the customer segments, and make recommendations to customers. The project utilizes different datasets for segmentation and recommendation. The recommendation project evaluated the neural model to generate the recommendation. The data for the project is downloaded from [UCI](https://archive.ics.uci.edu/) and [Kaggle](https://www.kaggle.com/) and a subset of data is used
> to execute the machine learning algorithms. The project utilizes the CRISP-DM methodology defined in the diagram and covers all its phases.  
> 
>${\color{crimson}Used \space CRISP-DM \space model \space to \space achieve \space the \space business \space goals}$
>
> ![image](https://github.com/user-attachments/assets/937f6f2a-b9e1-41e8-8396-193a2c46b57a)

> **PROBLEM STATEMENT**
> The problem statement is to analyze the customer data using machine learning models to build customer segmentation and generate recommendations. The segmentation and recommendation use two different sources of datasets, and available machine models are executed and compared for the result. 

> **BUSINESS UNDERSTANDING**
> 
> ${\color{mediumblue}Customer \space\ Segmentation}$
The online sales data is available from retailers and is available at the [UCI website](https://archive.ics.uci.edu/dataset/352/online+retail) for analysis. The business expects to use the RFM methodology, data mining techniques, and machine learning algorithms to derive meaningful customer segmentation, better understand customer purchase behavior, and identify the characteristics of customers in each segment.
>
> ${\color{mediumblue}Recommendation \space\ System}$
The movie data is available from Netflix and is available at the [Kaggle website](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) as part Netflix prize. The business expects to use the [SURPRISE](https://surprise.readthedocs.io/en/stable/index.html)/ and other recommendation methodologies such as Neural to develop a customer recommendation system. Compare various algorithms and systems must be able to recommend movies that the customer has not watched earlier. Also, the algorithm should be tuned with the available parameters to minimize the error. 
  
> **BUSINESS GOAL**
>
> The Customer Segmentation goal is to organize customers in similar groups, better understand individual customers in each cluster, and identify the customers at risk. This way businesses can have a customer-centric focus & approach to target individual customers.
> 
> The Recommendation System aims to provide movie recommendations to the customers. The system shall provide top movie recommendations to customers, and identify the top nearest neighbors for customer & movie, and compare different algorithms, and select the best algorithm for recommendation with the least error. 

> **TECH STACK**
> 
> $${\color{mediumblue}Language}$$ `Python`
> 
> $${\color{mediumblue}Packages}$$
> `Pandas`, `Numpy`, `Seaborn`, `Plotly`, `Matplotlib`, `Column Transformer`, `TargetEncoder`, `StandardScaler`, `Pipeline`, `train_test_split`, `KMeans`, `Silhouette_samples', 'Silhouette_score`, `accuracy_score`, `GridSearchCV`, `Cross_Validate`, `Surprise Dataset`, `SVD`, `SVDpp`, `NMF`, `SlopeOne`, `KNNBasic`, `KNNWithMeans`, `KNNBaseline`, `CoClustering`, `NormalPredictor`, `BaselineOnly`, `Keras`, `Keras.Layers`, `Keras.Losses`, `Keras.Metrics`, `Tensorflow`
>
> $${\color{mediumblue}Repository-path}$$ [CustomerSegmentation](customerClustering.ipynb)  [RecommendationSystem](recommendationSystem.ipynb)
>
> $${\color{mediumblue}Datafile-path}$$ [CustomerSegmentation](OnlineRetail.zip)  [RecommendationSystem-Combined_data1](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)  [RecommendationSystem-MovieTitle](movie_titles.csv) 
> 
> ****Note**** The RecommendationSystem project uses the combined_data1 file. This file is to be directly downloaded from the Netflix page on Kaggle. And converted using the first step of the project (which is currently commented on in the notebook) to a data file that can be utilized by the project.

 
> **DATA UNDERSTANDING & PREPARATION**
>
> ***Data Acquisition***
> 
> ${\color{mediumblue}Customer \space\ Segmentation}$
> The sales data is downloaded from [UCI website](https://archive.ics.uci.edu/dataset/352/online+retail), it is a transactional data set that contains all the transactions occurring between 01/12/2010 and
> 09/12/2011 for a retailer and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers. 
> 
> The original data attributes are:
> 
> |Feature Name | Description                                | Feature Type  |
> |-------------|--------------------------------------------|---------------|
> |InvoiceNo    | A unique number for a transaction          | Categorical   |
> |StockCode    | Product number for an Item                 | Categorical   |                          
> |Description  | Product Name                               | Categorical   |
> |Quantity     | Product quantity in each transaction       | Integer       |
> |InvoiceDate  | Day & Time when transaction was generated  | Date          |
> |UnitPrice    | Price of unit product                      | Continuous    |
> |CustomerID   | ID of customer                             | Categorical   |
> |Country      | Country where customer resides             | Categorical   |
>
> The data attributes added after analysis & feature engineering are:
> 
> |Feature Name 	  | Description                                                               | Feature Type  |
> |----------------|---------------------------------------------------------------------------|---------------|
> |Recency      	  | Define how recently the customer made a purchase                          | Integer       |
> |Frequency    	  | Define how often customers make purchases                                 | Integer       |
> |Monetary     	  | Define the amount the customer has spent                                  | Float         |
> |Recency Score	  | Quantile-based discretization on a scale of 1-5 based on Recency value    | Integer       |
> |Frequency Score | Quantile-based discretization on a scale of 1-5 based on Frequency value  | Integer       |
> |Monetary Score  | Quantile-based discretization on a scale of 1-5 based on Frequency value  | Integer       |
> |RFM Segment     | ID of customer                                                            | Object        |
> |Customer Type   | Country where customer resides                                            | Object        |
>
> ${\color{mediumblue}Recommendation \space\ System}$
> 
> The movie rating data is downloaded from [Kaggle website](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data), The movie rating files contain over 100 million ratings from 480 thousand randomly
> chosen anonymous customers and over 17 thousand movie titles. The data were collected between October 1998 and December 2005 and reflect the distribution of all ratings received during this period. The dataset
> was trimmed to support the local execution of algorithms, and all predictions and comparisons/ tuning are done on the  trimmed dataset, this may compromise on error measure/ accuracy of the algorithms. The
> rating data is provided by Netflix, the dataset contains all the movie ratings ranging from 1-5 provided by users. Also, a separate file is provided which contains the movie title/ID for a user.
>
> The consolidated file data attributes are:
> 
> |Feature Name  | Description                                                | Feature Type  |
> |------------- |------------------------------------------------------------|---------------|
> |MovieID       | A unique number for a Movie                                | Integer       |
> |CustomerID    | A unique number given to identify a customer               | Integer       |
> |Title         | Movie title                                                | Categorical   |
> |YearOfRelease | Movie release date                                         | Date          |
> |Ratings       | Movie ratings on a five-star (integral) scale from 1 to 5  | Integer       |
> |Rating Date   | Movie rating date by customer                              | Date          |

> ***Feature Analysis***
>   
> - All Features in Customer Segmentation & Recommendation datasets are being analyzed for missing values and missing values are filled in with the mean values strategy.
> - The customer segmentation dataset is scaled using a Standard scaler pipeline
>  
>   ![Screenshot 2025-01-18 105115](https://github.com/user-attachments/assets/5717ecb7-4461-418d-9408-9b48d7b1e3ea)

> - The overall missing value count is very low compared to the data volume.
> - Analysed the features in the dataset using a correlation matrix & Pair plots.
> - Outlier Analysis & treatment for the Customer Segmentation dataset. 
> - Created new features for Customer Segmentation using RFM Methodology.
> - Reduce the data volume to execute the algorithms locally.
>
> - Pair Plot Analysis of New Features in Customer Segmentation.
>   - The graph shows the linear relationship of data between the frequency of visits and monetary value.
>   - Monetary and recency do not hold any definitive relationship of data.
>   - The RFM features show the majorly left concentrated data points. 
>     
> ![Screenshot 2024-12-30 135951](https://github.com/user-attachments/assets/f857d5f5-2d39-45c8-a217-67eee03dda9f)
> 
> - Conducted the outlier analysis for features. 
>   - The outliers appear more in monetary and frequency plots. 
>   - The outliers are treated by considering the features data in the first & third quartiles (25% - 75%).
>   
> ![Screenshot 2024-12-30 135909](https://github.com/user-attachments/assets/cf53ca9f-377e-4af5-bc79-0aa4f5e5984b)
> 
> - Conducted the correlation matrix analysis.
>   - The dataset is scaled before the generation of the correlation matrix.
>   - The majority of features do not show tight coupling.
>   - The frequency and monetary score are dependent.     
> 
> ![Screenshot 2025-01-18 112030](https://github.com/user-attachments/assets/4bcffafe-ef66-4700-ba5b-0564289dde4a)

> **MODELING**
>
> ${\color{mediumblue}Customer \space\ Segmentation}$
> 
> The unsupervised machine learning techniques were used to group the customer into meaningful segments. After data analysis, new features were added to the dataset and multiple algorithms were tried on the new feature to group the customer. The results from all the algorithms show that data can be grouped into 5 meaningful clusters.  
>
> ***1. Kmeans Analysis:***
> 
>  - Elbow method to Identify the number of optimum clusters:
>    - The graph shows bend at 2 & 5 clusters.
>    - Based on the graph there are 5 clusters to be considered for analysis.  
>  
> ![Screenshot 2024-12-30 150348](https://github.com/user-attachments/assets/c10f314c-ceef-4466-8ea5-ba7f650c3a79)
>
>  - Plotted the Customer Clusters by Kmeans
>    - There are 5 distinct groups formed by the K-means algorithm.
>    - The clusters are grouped on the customer's recent visits to the website.
>    - The customer visits and monetary group data are in a linear relation.
>   
> ![Screenshot 2024-12-30 150411](https://github.com/user-attachments/assets/664b8dba-3563-4b35-9560-5ca77a7a423e)
>
> ![Screenshot 2025-01-18 130310](https://github.com/user-attachments/assets/b4904bc3-3ec4-4661-862f-60065b63fba8)
![Screenshot 2025-01-18 130558](https://github.com/user-attachments/assets/a130c552-a45b-4074-aef9-2e7b202ade53)
>
> ***2. Silhouette Analsysi & Score***
>
>  - The Silhouette score is decreasing as the number of clusters is increasing.
>    - For n_clusters = 2 The average silhouette_score is : 0.6387
>    - For n_clusters = 3 The average silhouette_score is : 0.5611
>    - For n_clusters = 4 The average silhouette_score is : 0.4793
>    - For n_clusters = 5 The average silhouette_score is : 0.4423
>    - For n_clusters = 6 The average silhouette_score is : 0.4185    
>  - After the fifth cluster the outliers start to appear.
>  - Selecting 5 clusters seems to be the optimum choice for this dataset.
>    
> ![Screenshot 2024-12-30 150505](https://github.com/user-attachments/assets/63fe7af1-018b-43d5-ab39-5149bb121eee)
>
> ***3. DBScan***
>
> - Selected the optimum EPS value using nearest neighbor 
>   - The plot shows an optimum EPS value between 0.1 & 0.2
>   - The DBScan at 0.118 resulted in 5 customer clusters.
>   - The plot results in similar segments as other algorithms have generated. Sharing the monetary & frequency plot.
>
> <p>
>  <img src="https://github.com/user-attachments/assets/eebfeade-c2c8-4ad4-95be-1f3bca3a5376" hspace="5" width="350" height="350" >
>  <img src="https://github.com/user-attachments/assets/0f7fe1b8-dd5c-44cc-a1f3-bab6a7d924bc" hspace="5" width="350" height="350" >
> </p>    
>
> 
> ${\color{mediumblue}Recommendation \space\ System}$
> 
> |Algorithms                        | Description                                                                    |
> |----------------------------------|--------------------------------------------------------------------------------|
> |SVD                               | Matrix factorization-based algorithms                                          |
> |SVD++                             | Matrix factorization-based algorithms considering implicit ratings             |                           
> |NFM                               | Collaborative filtering algorithm based on non-negative Matrix Factorization.  |                                                     |
> |Slope One                         | Collaborative filtering algorithm.                                             | 
> |Co-Clustering                     | Collaborative filtering algorithm based on co-clustering.                      |   
> |KNN Nearest Neighbor              | K-nearest neighbors  inspired algorithm.                                       | 
> |Normal Predictor                  | Predict random rating based on the normal distribution of the training set.    |
> |Neural Network(Tensor/ Keras API) | Use dense layers and weights to process the relationship between user/ movies. | 


> **MODEL EVALUATION**
>
> ${\color{mediumblue}Customer \space\ Segmentation}$
> 
> - Customer Cluster Wise RFM Score:
>   - Cluster 1 has the highest number of customers. This cluster's customers are frequent visitors and contribute high monetary value.
>   - Cluster 2 & 3 customers need attention, as they have a low recent score even though they have contributed more to monetary value. 
> 
> ![Screenshot 2024-12-30 150819](https://github.com/user-attachments/assets/299e42a2-80c1-44e5-8698-03675ac76218)
>
> - Customer cluster-wise count & peak selling time
>   - The number of customers in the first segment is the highest, and these customer's contributions are the highest as well.
>   - The number of customers in segment two is comparatively lowest.
>   - The peak selling time for the website across all the clusters is 11:00 Am-2:00 Pm
>     
> <p>
>  <img src="https://github.com/user-attachments/assets/1cbfccda-7d61-4c71-bd1d-2fc56db3c379" hspace="5" width="350" height="350" >
>  <img src="https://github.com/user-attachments/assets/0d9f21d2-8742-46f9-afe1-c0812ac2bd3b" hspace="5" width="350" height="350" >
> </p>
>
> 
>
> Customer cluster-wise Top selling month 
> ![Screenshot 2024-12-30 150918](https://github.com/user-attachments/assets/91092ece-b315-4856-b3c4-1cf9061b8574)
>
> Tree MAP for Customer Segments based on RFM score
> ![Screenshot 2024-12-30 150958](https://github.com/user-attachments/assets/adc1ab42-0a79-401b-b0c6-823fabeb338b)
>
> ${\color{mediumblue}Recommendation \space\ System}$ 
> The SURPRISE modules were used to execute and fine-tune customer recommendation algorithms. 
> 
> Comparision of Scores and execution time post hyperparameter tuning: 
> ![Screenshot 2024-12-30 153220](https://github.com/user-attachments/assets/5756418f-af13-408c-ae63-e834e048b6f7)
> ![Screenshot 2024-12-30 153251](https://github.com/user-attachments/assets/1523d3cd-9175-4147-9a85-550557335601)
> ![Screenshot 2024-12-30 153334](https://github.com/user-attachments/assets/97c67457-7830-40ae-a8bb-044086582b66)
> ![Screenshot 2024-12-30 153402](https://github.com/user-attachments/assets/b367b787-65a0-4c6f-9190-eae422954364)
> 
> Top 3 Movie Recommendations for the Users.
> 
> ![Screenshot 2024-12-30 153425](https://github.com/user-attachments/assets/6ca59fe1-31bf-441c-8790-a7e3f31ada7b)
>
> Nearest Neighbour based on Movie and Customer
>  
> ![Screenshot 2024-12-30 153449](https://github.com/user-attachments/assets/a6a75223-c8a8-41bc-b96f-0d3024f28bcc)
> ![Screenshot 2024-12-30 153512](https://github.com/user-attachments/assets/b501aa9f-5601-4dd4-b5d2-5b9557abd50a)
>
>
>**CONCLUSION**
>
>
> **REFERENCES**    
> - [UCI website](https://archive.ics.uci.edu/dataset/352/online+retail)
> - [SURPRISE](https://surprise.readthedocs.io/en/stable/index.html)
> - [KAGGLE](https://www.kaggle.com/) 
