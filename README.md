# Linear_Regression_A-Z

This repository contains the code and resources for a machine learning project that aims to predict the prices of used cars based on various features. The project involves analyzing a dataset of used cars, performing data preprocessing, exploratory data analysis, and training a linear regression model. Additionally, Lasso and Ridge regression techniques are applied for regularization and improved model performance.

## Dataset

The dataset used in this project includes information about used cars, such as car details, location, year, kilometers driven, fuel type, transmission, owner type, mileage, engine, power, seats, new price, and price.

The dataset can be downloaded from Kaggle https://www.kaggle.com/code/dscodingp19/cars4u-project-using-linear-regression/input.

## Code Structure

The code in this repository follows the following steps:

### 1.Data Preprocessing:

Replace 0.0 values with NaN in the dataset.
Fix data types, converting object columns to categorical and numerical columns.
Group the cars by model and brand to create new features.
Treat missing values in the Price column by calculating the median price per brand and per brand's model and replacing the missing values accordingly.

### 2.Exploratory Data Analysis (EDA):

Perform data visualization, including histograms and box plots, to understand the distribution and characteristics of the data.
Explore the relationships between variables and identify any patterns or trends.
Feature Engineering:

Group the cars by regions and create a new feature representing the price tier.
Use heat maps to analyze the correlation between features and identify important relationships.

### 3.Outlier Treatment:

Apply outlier treatment techniques, such as the high whisker and lower whisker methods, to handle extreme values in the dataset.
Data Splitting and Model Training:

Split the data into training and testing sets.
Train a linear regression model on the training data.
Evaluate the performance of the model using various metrics, including mean squared error (MSE), R-squared, and visualizations such as residual plots.

### 4.Lasso and Ridge Regression:

Apply Lasso and Ridge regression techniques to the data to improve model performance and handle multicollinearity.
Select optimal hyperparameters using cross-validation.

### 5.Assumptions of Linear Regression:

Check the five assumptions of linear regression, including linearity of variables, independence of residuals, constant variance (homoscedasticity), normality of residuals, and absence of multicollinearity.


## Results and Conclusion

### Assumptions

1.NO MULTICOLLINEARITY.
We calculated the Variance Inflation Factor (VIF) for each independent variable in order to assess multicollinearity. The VIF values indicate the presence of multicollinearity, with higher values indicating a stronger correlation between variables.

![image](https://github.com/Shyam123-bot/Linear_Regression_A-Z/assets/61462986/c8c73af0-765e-469e-b71e-76eaa63facf8)

The VIF values showed a high degree of correlation between the "Engine" and "Power" variables. To address this issue, we made the decision to remove these two variables from the analysis.After removing the "Engine" and "Power" variables, we performed linear regression using the Ordinary Least Squares (OLS) method. The updated model exhibited an improved R-squared score, indicating a better fit to the data. This improvement suggests that removing the highly correlated variables of "Engine" and "Power" due to collinearity enhanced the predictive power of the linear regression model.

2.MEAN OF RESIDUALS IS 0.

![image](https://github.com/Shyam123-bot/Linear_Regression_A-Z/assets/61462986/1ab76cf0-7cf5-4a27-b01d-81b455ebd97d)

The mean of the residuals in the linear regression model was calculated to be approximately -1.68e-12. This indicates that, on average, the model's predictions closely align with the observed values. The near-zero mean residual suggests that the model accurately captures the underlying relationships between the independent variables and the dependent variable, without any significant bias in its predictions. Overall, this indicates the model's reliability and effectiveness in predicting the target variable based on the provided features.

3.TEST FOR LINEARITY

![image](https://github.com/Shyam123-bot/Linear_Regression_A-Z/assets/61462986/2907d01d-b18a-41d3-9976-6327676bf9a1)

The scatter plot of residuals and fitted values shows no discernible pattern, indicating that the assumption of linearity is satisfied. The absence of any noticeable trend or pattern suggests that the linear regression model adequately captures the relationship between the dependent variable and the independent variables. This supports the validity of using linear regression for predicting the target variable based on the provided features.

4.TEST FOR NORMALITY

![image](https://github.com/Shyam123-bot/Linear_Regression_A-Z/assets/61462986/9ce54184-1a1a-4e1f-b691-b94703d5c0fe)

![image](https://github.com/Shyam123-bot/Linear_Regression_A-Z/assets/61462986/a2e1272b-a4cf-4340-a547-548a2b1154e3)

The test for normality was conducted using a histogram and a Q-Q plot. The histogram of the residuals displayed a bell-shaped curve, indicating that the residuals approximately follow a normal distribution. Additionally, the Q-Q plot showed that the points closely aligned with a straight line, suggesting that the residuals conform to a normal distribution. These results confirm that the assumption of normality is satisfied in this project, indicating that the residuals of the linear regression model are normally distributed.

5.TEST FOR HOMOSCEDASTICITY

![image](https://github.com/Shyam123-bot/Linear_Regression_A-Z/assets/61462986/4c2d26cd-1def-4810-8a2d-5cca691a9fee)

The test for homoscedasticity was conducted using the Goldfeld-Quandt test. The result showed a p-value of 0.202, which is greater than the significance level of 0.05. This indicates that we fail to reject the null hypothesis, suggesting that the residuals are equal (homoscedastic) across all independent variables. In simpler terms, the variability of the residuals remains relatively constant regardless of the values of the independent variables.
This finding supports the assumption of homoscedasticity in the linear regression model. It means that the model's performance is not significantly affected by varying levels of the independent variables. The consistent variance of the residuals enhances the reliability and interpretability of the regression results.


### Linear Regression Model:

Mean Absolute Error (MAE): 0.890
Root Mean Squared Error (RMSE): 1.215
R-squared (R2) Score: 0.954

### Ridge Regression:

Best Parameters: {'alpha': 1e-15}
Best Score (negative mean squared error): -1.519
Further analysis with Ridge regression:

Best Parameters: {'alpha': 1e-08}
Best Score (negative mean squared error): -1.495

### Conclusion:

Based on the analysis, the linear regression model achieved good predictive performance with an MAE of 0.890, indicating that, on average, the model's predictions deviate by approximately 0.890 from the actual values. The RMSE value of 1.215 signifies the average magnitude of the prediction errors, providing an overall measure of the model's accuracy. The R2 score of 0.954 indicates that around 95.4% of the variance in the dependent variable can be explained by the independent variables.

The Ridge regression model, with the best parameter value of alpha=1e-15, demonstrated a slightly better performance in terms of the negative mean squared error (-1.519). Upon further analysis, using alpha=1e-08 resulted in a slightly improved score of -1.495.

Overall, the results suggest that both the linear regression model and the Ridge regression model are effective in predicting the target variable. The high R2 score indicates a strong relationship between the independent variables and the dependent variable. Further improvements in performance could be explored by optimizing other hyperparameters or considering alternative regression techniques.

