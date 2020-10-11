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
dataset = pd.read_csv('/Users/awaismemon/Documents/Datasets/california_housing_prices.csv')
# %% Selecting Columns from DataSet
featureCol = ['total_rooms'] # input("Enter Column Name [FEATURE]: ") # Feature --
labelCol = ['median_house_value'] # input("Enter Column Name [LABEL]: ")  # Label -- Target    

# %% 
# X = dataset.iloc[:, 3].values  # Matrix of Features    ---- dataset.iloc[:, :-1].values  -- This should be Matrix
# Y = dataset.iloc[:, 8].values # Matrix of Labels            ---- dataset.iloc[:, 3].values  -- This should be Vector

X = dataset[featureCol] # -- Independent Variables 
Y = dataset[labelCol] # -- Dependent Variable 
print(X.shape)
print(Y.shape)
# %% Splitting Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# %% Feature Scaling
#  
# -- Standardize data
# The values for each attribute now have a mean value of 0 and a standard deviation of 1.
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

scale_Y = StandardScaler()
Y_train = scale_Y.fit_transform(Y_train)
# ## 

#%% Fitting Linear Regression to Training Set
model = LinearRegression()
model.fit(X_train, Y_train) 
 

# %% Predicting the Test set results
model_pred = model.predict(X_test)

# %% Visualising the Training Set Results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title('Housing Prices')
plt.xlabel('Total Rooms')
plt.ylabel('Prices')
plt.show()