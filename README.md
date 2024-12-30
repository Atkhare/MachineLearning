
# Customer Segmentation & Recommendation System

> **OVERVIEW** 
This CAPSTONE project aims to develop a Customer Segmentation & Recommendations system using multiple algorithms and strategies available in Machine Learning. It compares the performance of the algorithms to identify the best-performing algorithm for the subset of data. The data for the project is downloaded from [UCI](https://archive.ics.uci.edu/) and [Kaggle](https://www.kaggle.com/) and a subset of data is used to execute the machine learning algorithms. The project utilizes the CRISP-DM methodology defined in the diagram and covers all its phases.  
>
>${\color{crimson}Used \space CRISP-DM \space model \space to \space achieve \space the \space business \space goals}$

![image](https://github.com/user-attachments/assets/937f6f2a-b9e1-41e8-8396-193a2c46b57a)

> **BUSINESS UNDERSTANDING**
The business goal is to find a model that can explain the success of a contact with a client, i.e., if the client subscribes to the term deposit. The classification model can increase campaign efficiency by identifying the main characteristics that affect success, helping in better management of the available resources (e.g., human effort, phone calls, and time), and selecting a high-quality and affordable set of potential buying customers.

> **TECH STACK**
>> $${\color{mediumblue}Language}$$ `python`
>> 
>> $${\color{mediumblue}Packages}$$
>> `Pandas`, `Numpy`, `Seaborn`, `Plotly`, `Matplotlib`, `Simple Imputer`, `Column Transformer`, `TargetEncoder`, `LabelEncoder`, `StandardScaler`, `Pipeline`, `OneHotEncode`, `Ordinal Encoder`, `PCA`, `RandomForestClassifier`, `train_test_split`, `DummyClassifier`, `classification_report`, `accuracy_score`, `roc_curve`, `roc_auc_score`, `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `SVC`, `mutual_info_regression`, `GridSearchCV`
>>
>> $${\color{mediumblue}Repository-path}$$ [Code Repo](model_comparision.ipynb)
>>
>> $${\color{mediumblue}Datafile-path}$$ [Data file](bank-additional-full.csv)
>> 

> **DATA UNDERSTANDING**
The dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns. The article accompanying the dataset [here](CRISP-DM-BANK.pdf) is being referred to for more information on the data and its features.
> To find the effectiveness of marketing campaigns, Portugese bank reached out to customers from their contact centers and collected the customer perspective. This collected data along with the internal customer data within the bank was used to prepare the dataset features for analysis. The classification goal was to determine if a client would subscribe to a term deposit. The team used 3 iterations of a campaign to fine-tune the predictive model

> **DATA PREPARATION**
The data is first analyzed on the missing or NAN elements, then the data aggregation is done to analyze the trend & unique components column and row-wise. The data is being imputed for any missing/ unknown values. Also, the data encoding is done for categorical features. The final Dataframe is scaled using `StandardScaler' and split into Test & Train datasets. 

> **FEATURE ANALYSIS**
> - Analyzed imbalance in the Target class from the provided Dataframe. 
> - The correlation between dataset features is being analyzed by generating the `Correlation Matrix Heatmap` over a scaled Dataframe.
> - Executed `PCA` to analyze the scope for dimension reduction.
> - Scored feature importance using `RandomForestClassifier` and `mutual_info_regression`
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



   




