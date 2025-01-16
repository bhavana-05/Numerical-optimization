import numpy as np
import pandas as pd


#use pandas to load the real estate dataset
df = pd.read_csv("real_estate_dataset.csv")

#get the number of samples and features
n_samples, n_features = df.shape

#display the first 5 rows of the dataset
#df.head()

#print the number of samples and features
print("number of samples, number of features: ", n_samples, n_features)

#save the features/columns of the dataset
columns = df.columns
np.savetxt("real_estate_dataset_columns.txt", columns, fmt="%s")

#use the square feet, garage size, location score, and distance to center as features
X = df[['Square_Feet','Garage_Size','Location_Score','Distance_to_Center']].values

#use the price column as the target
y = df['Price']

# check the target column
#print(y)

#print X.shape and X.type
print(f'shape of X: {X.shape}\n')
print("type of X:", X.dtype)

#get the samples and features of X
n_samples, n_features = X.shape

# Build the linear model to predict the price
# make an array of coefs with n_features+1 elements and initialize it to 1
# n+1 because the additional column is for the bias term
coefs = np.ones(n_features+1)


# bias = expected value of target variable, 
# when the prediction is exactly equal to bias

# predict the price of each sample
predictions_bydefn = X @ coefs[1:] + coefs[0]

#append a column of ones to the input matrix ,X
X = np.hstack([np.ones((n_samples, 1)), X])

#predict the samples for each samples in X
predictions = X@coefs

# check whether the predictions in both cases are the same
is_same = np.allclose(predictions, predictions_bydefn)
print("Are the predictions the same?", is_same)


#calculate the error between the prediction and the target
error = y - predictions

#calcule the relative error
rel_error = error / y

#calculate the mean of square of errors using a loop
loss_loop = 0
for i in range(n_samples):
    loss_loop += error[i]**2

loss_loop /= n_samples


#calculate the mean of square of errors using matrix operations
loss_matrix = np.transpose(error) @ error / n_samples

#compare the two methods of calculating the mean of square of errors
is_diff = np.allclose(loss_loop, loss_matrix)
print('Are the two methods of calculating the mean of square of errors the same?', is_diff)


#print the size of errors and its L2 norm
print('Size of errors:', error.shape)
print('L2 norm of errors:', np.linalg.norm(error))
print('L2 norm of relative errors:', np.linalg.norm(rel_error))

# Objective function: f(coefs) = 1/n_samples * \sum_{i=1}^{n_samples} (y_i - (coefs[0] + X_i @ coefs[1:]))^2

#what is the optimization problem we are solving?
# I want to find the coefficients that minimize the loss function
# This problem is called least squares problem

# what is a solution
# solution is set of coefficients that minimize the loss function
# so for a set of coefficients to be solution, the gradient of the loss function(objective) evaluated with respect to the coefficients should be zero.
# to be minimum, the hessian of the loss function evaluated with respect to the coefficients should be positive definite
# in that n+1 dimensional sapce, for a  set to be solution, the gradient of the loss function evaluated with respect to the coefficients should be zero. 
# and the hessian of the loss function evaluated with respect to the coefficients should be positive definite

# we can solve by setting the gradient of the loss function to zero and solving for the coefficients
# write the loss matrix in terms of data and coefficients

loss_matrix = (y-X@coefs).T@(y-X@coefs) / n_samples

# calculate the gradient of the loss function with respect to the coefficients
grad_matrix = -2/n_samples*X.T@(y-X@coefs)

# set the gradient to zero and solve for the coefficients
#  X.T @ y = X.T @ X @ coefs
# X.T @ X @ coefs = X.T @ y
# coefs = (X.T @ X)^-1 @ X.T @ y

coefs = np.linalg.inv(X.T@X)@X.T@y

# save the coefficients to a file
np.savetxt("coefs.csv", coefs, delimiter=",")

# calculate the gradient using the new coefficients
predictions_model = X@coefs

#calculate the error using the new coefficients
errors_model = y - predictions_model

#print the L2 norm of the error_model
print('L2 norm of error_model:', np.linalg.norm(errors_model))

#calculate and print the relative error using the optimal coefs
relative_error_model = errors_model / y
print("L2 norm of the errors using the new coefficients: ", np.linalg.norm(errors_model))
print("Relative error of L2 norm using the new coefficients: ", np.linalg.norm(relative_error_model))

#use all the features in the dataset and build the linear model
X = df.drop('Price', axis=1).values
y = df['Price'].values

#get the number of features and samples in X
n_samples, n_features = X.shape
print("number of samples, number of features: ", n_samples, n_features)

#solve the model using normal equations
X = np.hstack([np.ones((n_samples, 1)), X])
coefs = np.linalg.inv(X.T@X)@X.T@y

#calculate the predictions using the new coefficients
predictions = X@coefs

#calculate the error using the new coefficients
errors_all = y - predictions
rel_error_all = errors_all / y
print("L2 norm of the errors using all the features: ", np.linalg.norm(errors_all))
print("Relative error of L2 norm using all the features: ", np.linalg.norm(rel_error_all))

#save the coefficients to a file
np.savetxt("coefs_all.csv", coefs, delimiter=",")

#calculate the rank of X^T @ X
rank_xtx = np.linalg.matrix_rank(X.T@X)
print("Rank of X^T @ X: ", rank_xtx)

#solve the normal equation using QR decomposition
Q, R = np.linalg.qr(X)

print("Q shape: ", Q.shape)
print("R shape: ", R.shape)

#save R to a file
np.savetxt("R.csv", R, delimiter=",")

sol = Q.T @ Q #identity matrix
np.savetxt("sol.csv", sol, delimiter=",")

# X = QR
# X^T @ X = R^T @ Q^T @ Q @ R = R^T @ R
# X^T @ y = R^T @ Q^T @ y
# R @ coefs = Q^T @ y

b = Q.T @ y

print("b shape: ", b.shape)
print("R shape: ", R.shape)

# solve for coefs using back substitution
coefs_qr = np.zeros(n_features+1)

for i in range(n_features, -1, -1):
    coefs_qr[i] = b[i]
    for j in range(i+1, n_features+1):
        coefs_qr[i] -= R[i, j]*coefs_qr[j]
    coefs_qr[i] /= R[i, i]

#calculate the predictions using the new coefficients
predictions_qr = X@coefs_qr

#calculate the error using the new coefficients 
errors_qr = y - predictions_qr  
rel_error_qr = errors_qr / y
print("L2 norm of the errors using QR decomposition: ", np.linalg.norm(errors_qr))
print("Relative error of L2 norm using QR decomposition: ", np.linalg.norm(rel_error_qr))

#save the coefficients to a file
np.savetxt("coefs_qr.csv", coefs_qr, delimiter=",")

# solve the normal equation using SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)

#eigen values of X^T @ X are the squares of the singular values of X
# eigen decomposition of square matrix
# A = V @ D @ V.T
# A ^ -1 = V @ D^-1 @ V.T      
# X * ceefs = y
# A = X.T @ X

# normal equation is X.T @ X @ coefs = X.T @ y
# coefs = (X.T @ X)^-1 @ X.T @ y
# Xdagger = (X.T @ X)^-1 @ X.T
                                          
# find the inverse of X in least squares sense
# pseudo inverse of X is Xdagger = V @ D^-1 @ U.T

# to complete : calculate the coefficients_SVD using psudo inverse of X

# Compute the pseudoinverse of X
# Reciprocal of singular values (for inverse of a diagonal matrix)
S_inv = np.diag(1 / S)  
X_dagger = Vt.T @ S_inv @ U.T  

# Calculate coefficients using the pseudoinverse
coefs_svd = X_dagger @ y

# Calculate predictions using the SVD coefficients
predictions_svd = X @ coefs_svd

# Calculate the errors using the SVD coefficients
errors_svd = y - predictions_svd
rel_error_svd = errors_svd / y
print("L2 norm of the errors using SVD decomposition: ", np.linalg.norm(errors_svd))
print("Relative error of L2 norm using SVD decomposition: ", np.linalg.norm(rel_error_svd))

# Save the coefficients to a file
np.savetxt("coefs_svd.csv", coefs_svd, delimiter=",")



