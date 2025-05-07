from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

iris = load_iris()
dataset1 = StandardScaler().fit_transform(iris.data)
pca=PCA(n_components=2)
principal_components=pca.fit_transform(dataset1)
inverse_transform_data=pca.inverse_transform(principal_components)
mse=mean_squared_error(dataset1,inverse_transform_data)
print("Mean Squared Error for PCA & Inverse PCA is ",mse)

from sklearn.neural_network import MLPRegressor
autoencoder=MLPRegressor(hidden_layer_sizes=(2024,2,2024),activation='relu',solver='adam',max_iter=10000000)
autoencoder.fit(dataset1,dataset1)
reconstructed_data=autoencoder.predict(dataset1)
mse_auto=mean_squared_error(dataset1,reconstructed_data)
print("auto mse",mse_auto)