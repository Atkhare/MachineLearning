
# Customer Segmentation & Recommendation System

> **OVERVIEW**
> 
> This CAPSTONE project aims to develop a Customer Segmentation & Recommendations system using multiple algorithms and strategies available in Machine Learning. It compares the performance of the algorithms to
> identify the best-performing algorithm for the subset of data. The data for the project is downloaded from [UCI](https://archive.ics.uci.edu/) and [Kaggle](https://www.kaggle.com/) and a subset of data is used
> to execute the machine learning algorithms. The project utilizes the CRISP-DM methodology defined in the diagram and covers all its phases.  
>
>${\color{crimson}Used \space CRISP-DM \space model \space to \space achieve \space the \space business \space goals}$
>
> ![image](https://github.com/user-attachments/assets/937f6f2a-b9e1-41e8-8396-193a2c46b57a)

> **BUSINESS UNDERSTANDING**
> 
> CUSTOMER SEGMENTATION
The online sales data is available from retailers and is available at the [UCI website](https://archive.ics.uci.edu/dataset/352/online+retail) for analysis. The business expects to use the RFM methodology, data mining techniques, and machine learning algorithms to derive meaningful customer segmentation, better understand customer purchase behavior, and identify the characteristics of customers in each segment.
>
> RECOMMENDATION SYSTEM
The movie data is available from Netflix and is available at the [Kaggle website](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) as part Netflix prize. The business expects to use the [SURPRISE](https://surprise.readthedocs.io/en/stable/index.html)/ and other recommendation methodologies to develop a customer recommendation system. Compare various algorithms and systems must be able to recommend movies that the customer has not watched earlier. Also, tune the algorithm with the available parameters to minimize the error. 
  
> **BUSINESS GOAL**
>
> The Customer Segmentation goal is to organize customers in similar groups, better understand individual customers in each cluster, and identify the customers at risk. This way businesses can have a customer-centric focus & approach to target individual customers.
> 
> The Recommendation System's goal is to provide movie recommendations to the customers. The system shall provide top movie recommendations to customers, and identify the top nearest neighbors for customer & movie, and compare different algorithms, and select the best algorithm for recommendation with the least error. 

> **TECH STACK**
>> $${\color{mediumblue}Language}$$ `Python`
>> 
>> $${\color{mediumblue}Packages}$$
>> `Pandas`, `Numpy`, `Seaborn`, `Plotly`, `Matplotlib`, `Column Transformer`, `TargetEncoder`, `StandardScaler`, `Pipeline`, `train_test_split`, `KMeans`, `Silhouette_samples', 'Silhouette_score`, `accuracy_score`, `GridSearchCV`, `Cross_Validate`, `Surprise Dataset`, `SVD`, `SVDpp`, `NMF`, `SlopeOne`, `KNNBasic`, `KNNWithMeans`, `KNNBaseline`, `CoClustering`, `NormalPredictor`, `BaselineOnly`
>>
>> $${\color{mediumblue}Repository-path}$$ [Code Repo](model_comparision.ipynb)
>>
>> $${\color{mediumblue}Datafile-path}$$ [Data file](bank-additional-full.csv)
>> 

> **DATA UNDERSTANDING & PREPARATION**
> 
> ${\color{mediumblue}CUSTOMER \space\ SEGMENTATION}$
> The sales data is downloaded from [UCI website](https://archive.ics.uci.edu/dataset/352/online+retail), it is a transactional data set that contains all the transactions occurring between 01/12/2010 and
> 09/12/2011 for a retailer and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
> 
> The original data attributes are:
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
> The data attributes added as part of Analysis & Feature Engineering are:
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
> ${\color{mediumblue}RECOMMENDATION \space\ SYSTEM}$
> The movie rating data is downloaded from [Kaggle website](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data), The movie rating files contain over 100 million ratings from 480 thousand randomly
> chosen, anonymous customers over 17 thousand movie titles. The data were collected between October 1998 and December 2005 and reflect the distribution of all ratings received during this period. The dataset
> was trimmed to support the local execution of algorithms, and all predictions and comparisons/ tuning are done on the  trimmed dataset, this may compromise on error measure/ accuracy of the algorithms. The
> rating data is provided by Netflix, the dataset contains all the movie ratings ranging from 1-5 provided by users. Also, a separate file is provided which contains the movie title/ID for a user.
>
> The consolidated file data attributes are:
> |Feature Name  | Description                                                | Feature Type  |
> |------------- |------------------------------------------------------------|---------------|
> |MovieID       | A unique number for a Movie                                | Integer       |
> |CustomerID    | A unique number given to identify a customer               | Integer       |
> |Title         | Movie title                                                | Categorical   |
> |YearOfRelease | Movie release date                                         | Date          |
> |Ratings       | Movie ratings on a five-star (integral) scale from 1 to 5  | Integer       |
> |Rating Date   | Movie rating date by customer                              | Date          |

> **FEATURE ANALYSIS**
> -  
> - All Features in Customer Segmentation & Recommendation datasets are being analyzed for missing values and filled with mean values.   
> - Analysed the features of Customer Segmentation using a correlation matrix.
> - Outlier Analysis & treatment for the Customer Segmentation dataset.
> - Created new features for Customer Segmentation using RFM Methodology
> - 
>
> ![image](https://github.com/user-attachments/assets/44f7288a-fbb8-4edb-8322-46424511a2ae)

 
> **MODELING**
> This is a classifier problem statement. To start with first have to baseline the model to collect the `ModelTrainingTime`, `TrainingAccuracyScore`, and `TestAccuracyScore`.
> 
> - $${\color{mediumblue}Baseline \space Model}$$ 
> The baseline matrices are captured by executing & evaluating the following ML algorithms `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, and `SVC`.
>
>${\color{crimson}Baseline \space model \space score}$
>
> ![image](https://github.com/user-attachments/assets/a1970a2b-ae22-4d7c-bb37-f5478cf2731a)
>
> - $${\color{mediumblue}Model \space performance \space tuning}$$
> The model is now tuned on various hyperparameters to obtain the best scoring. The `GridSearchCV` cross-validator is used for optimal model performance tuning.
>
>${\color{crimson}Tuned \space model \space score}$ 
>
>   ![image](https://github.com/user-attachments/assets/3872bf67-d053-43eb-ad18-abad359688c7) 

>**MODEL EVALUATION**
The Model performance is evaluated based on training time, test & train score & ROC curve area. The scores are captured by executing the model on various hyperparameters and then selecting the best parameter.
>
>${\color{crimson}ROC \space Curve \space Analysis}$
>
> ![image](https://github.com/user-attachments/assets/d65279a1-3192-49a4-b312-6597c3d8c253)     \
>
>${\color{crimson}Model \space Scores \space Comparison}$
>
>![image](https://github.com/user-attachments/assets/3a48d1b7-22c9-487a-b8e9-080e4e5c8012)

>**CONCLUSION**
> The SVC algorithm has scored better performance and accuracy in terms of prediction. The drawback with SVC is the time it takes to execute when compared to other machine learning algorithms, however, still acceptable within seconds for this model. The analysis of the area covered under the ROC curve also shows a good score for SVC which means the model will comparatively correctly rank a randomly chosen positive example. The SVC is the first choice for this classification scenario.  



   




