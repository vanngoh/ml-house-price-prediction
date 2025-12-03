import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame

# x = df.drop(columns=['MedHouseVal'])
'''
Longitude (~-0.45): Strong negative impact. In California (your dataset), moving east (increasing longitude) generally means moving away from the coast, which reduces property values significantly.
Latitude (~-0.42): Strong negative impact. Moving north (increasing latitude) in California tends to correlate with lower prices, possibly reflecting the premium of southern California locations.
AveRooms (~0.10): Average rooms per household shows a modest positive effect. More total rooms generally indicates larger, more valuable properties.
MedInc (~0.40): Median income has a significant positive impact. Areas with higher median incomes tend to have more expensive housing, reflecting purchasing power and neighborhood desirability.
AveBedrms (~0.85): The strongest positive predictor. More bedrooms per household correlates strongly with higher house values. This makes intuitive sense as larger homes with more bedrooms typically cost more.
'''
x = df[['Longitude', 'Latitude', 'AveRooms', 'MedInc', 'AveBedrms']]
y = df['MedHouseVal']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Plot the results and feature importance side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted scatter plot
axes[0].scatter(y_test, y_pred)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Actual vs Predicted')

# Feature importance bar plot
feature_importance = model.coef_
axes[1].barh(x.columns, feature_importance)
axes[1].set_xlabel('Feature Importance')
axes[1].set_ylabel('Features')
axes[1].set_title('Feature Importance')

plt.show()
