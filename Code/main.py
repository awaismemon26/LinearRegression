## This Linear Regression model class is implemented using Sklearn package - which basically provides
## already implemented algorithm models.

# %%
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Standardization using Sklearn (Z-Score Normalization)
from sklearn.model_selection import train_test_split  # For Splitting training and test dataset
from sklearn.linear_model import LinearRegression # Linear Regression Model Class

# %% Loading Dataset
dataset = pd.read_csv('/Users/awaismemon/Documents/ML/LinearRegression/petrol_consumption.csv')
print(dataset.columns)
# %% Selecting Columns from DataSet
featureCol = ['Petrol_tax', 'Average_income', 'Paved_Highways',  'Population_Driver_licence(%)'] # input("Enter Column Name [FEATURE]: ") # Feature --
labelCol = 'Petrol_Consumption' # input("Enter Column Name [LABEL]: ")  # Label -- Target    

# %% 
# X = dataset.iloc[:, 3].values  # Matrix of Features    ---- dataset.iloc[:, :-1].values  -- This should be Matrix
# Y = dataset.iloc[:, 8].values # Matrix of Labels            ---- dataset.iloc[:, 3].values  -- This should be Vector

X = dataset[featureCol] # -- Independent Variables 
Y = dataset[labelCol] # -- Dependent Variable 

print(X.shape)
print(Y.shape)

# %% Splitting Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# # %% Feature Scaling
# #  
# # -- Standardize data
# # The values for each attribute now have a mean value of 0 and a standard deviation of 1.
# scale_X = StandardScaler()
# X_train = scale_X.fit_transform(X_train)
# X_test = scale_X.transform(X_test)

# scale_Y = StandardScaler()
# Y_train = scale_Y.fit_transform(Y_train)
# # ## 

#%% Fitting Linear Regression to Training Set
model = LinearRegression()
model.fit(X_train, Y_train) 


#%% In case of Multivariable linear regression, the model has to find the most
# optimal coeffients for all the attributes. 
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient']) 
print(coeff_df)
# %% Predicting the Test set results
model_pred = model.predict(X_test)

# %% Visualising the Training Set Results
# plt.scatter(X_train, Y_train, color='red')
# plt.plot(X_train, model.predict(X_train), color='blue')
# plt.title('Petrol Consumption')
# plt.show()
predictedValue = pd.DataFrame({'Actual': Y_test, 'Predicted':model_pred})
predictedValue
# %%
