import pandas as pd
import numpy as np

# Cargar el conjunto de datos
data = pd.read_csv("c:\\employee.csv")

# Verifica si hay datos faltantes y decide cómo tratarlos
data.isnull().sum()

# Aplicar codificación one-hot a las variables categóricas
data = pd.get_dummies(data, columns=["Education", "City", "Gender", "EverBenched"], drop_first=True)

# Supongamos que las etiquetas de clase son binarias (LeaveOrNot)
test_size = 0.2  # Proporción para el conjunto de prueba

# Dividir el conjunto de datos en entrenamiento y prueba manualmente
train_indices = []
test_indices = []

for class_label in [0, 1]:
    class_data = data[data["LeaveOrNot"] == class_label]
    num_samples = len(class_data)
    num_test_samples = int(test_size * num_samples)

    # Obtener los índices de prueba
    test_indices_class = np.random.choice(class_data.index, num_test_samples, replace=False)
    test_indices.extend(test_indices_class)

    # Los índices de entrenamiento son los que no están en los índices de prueba
    train_indices_class = [idx for idx in class_data.index if idx not in test_indices_class]
    train_indices.extend(train_indices_class)

# Crear conjuntos de entrenamiento y prueba
X_train_class = data.iloc[train_indices].drop("LeaveOrNot", axis=1)
y_train_class = data.iloc[train_indices]["LeaveOrNot"]
X_test_class = data.iloc[test_indices].drop("LeaveOrNot", axis=1)
y_test_class = data.iloc[test_indices]["LeaveOrNot"]

# Normalizar las características si es necesario (ya normalizadas)
mean = X_train_class.mean()
std = X_train_class.std()
X_train_class = (X_train_class - mean) / std
X_test_class = (X_test_class - mean) / std

# Agregar una columna de unos para el término independiente b0
X_train_class = np.column_stack((np.ones(len(X_train_class)), X_train_class))
X_test_class = np.column_stack((np.ones(len(X_test_class)), X_test_class))

# Inicializar los coeficientes (b0, b1, b2, etc.)
coefficients = np.zeros(X_train_class.shape[1])

# Definir la tasa de aprendizaje y el número de iteraciones
learning_rate = 0.01
num_iterations = 10000

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Entrenar el modelo de regresión logística
for _ in range(num_iterations):
    z = np.dot(X_train_class, coefficients)
    predictions = sigmoid(z)
    gradient = np.dot(X_train_class.T, (predictions - y_train_class)) / len(y_train_class)
    coefficients -= learning_rate * gradient

# Realizar predicciones en el conjunto de prueba
z = np.dot(X_test_class, coefficients)
predictions = sigmoid(z)
y_pred = (predictions >= 0.5).astype(int)

# Calcular la precisión del modelo
accuracy = np.mean(y_pred == y_test_class)
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))
