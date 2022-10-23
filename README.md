# Data Analytics.

In this project, I was assigned to analyze according to what economic-social factors countries are ranked with the highest levels of overall happiness and how these have changed over time. In general analyze what patterns make a country more or less happy. 

To begin these analyses there were steps to take which are:

1- NOSQL:

Create two APIs through two separate Azure Functions. Both need to take as input the year and return values appropriately saved with a JSON structure (NoSQL).
One for the first dataset World Happiness Reports.
And the second for Freedom Index.

- Example API:

"2015": 
[
	{
	"Country": "Switzerland",
      "Region": "Western Europe",
	"Happiness Rank": 1,
	"Happiness Score": 7.587,
	"Standard Error": 0.03411,
	"Economy (GDP per Capita)": 1.39651,
	"Family": 1.34951,
	"Health (Life Expectancy)": 0.94143,
	"Freedom": 0.66557,
	"Trust (Government Corruption)": 0.41978,
	"Generosity": 0.29678,
	"Dystopia Residual": 2.51738
	}
,



The response must contain a JSON that to the key corresponding to the year, contains a list of JSONs, one for each country considered.

2- Python and Data Engineering:

Extracting and cleaning and transforming the data into a form that is then more conveniently usable for visualization to set up a data analysis at-through which to extract insights useful for the proposed objective. 

the steps to be addressed were: 

-Management of Missing Values
-Verification and Management of Data types
-Normalization of Data

Here as you have seen in the Project.py file there are Object OOPs, I chose to implement these 3 classes that let's say ease the process a bit the work in this project, and it was also to get familiar with OOPs.

3- Predictive Analytics:

I had to choose a machine learning model to predict a column that you think is most appropriate with respect to the context and objectives set forth, the models I chose were:

- Linear Regression.
- KNeighborsClassifier.
- DecisionTreeClassifier.
- RandomForestClassifier.
- GradientBoostingClassifier.
- AdaBoostClassifier.
- XGBClassifier.





