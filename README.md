# RegresionLogistica

Este proyecto implementa un modelo de Regresión Logística en Python. El código se encarga de predecir la variable binaria 'LeaveOrNot' en un conjunto de datos de empleados. A continuación, se describen los pasos clave del código:

## Pasos del Proyecto

1. **Importación de Bibliotecas:**

   El código comienza importando las bibliotecas necesarias, que incluyen `pandas` y `numpy`, para el manejo de datos y cálculos matemáticos.

2. **Carga del Conjunto de Datos:**

   Se carga un conjunto de datos desde un archivo CSV llamado "employee.csv" y se almacena en un DataFrame de pandas llamado "data".

3. **Verificación de Datos Faltantes:**

   Se verifica la existencia de datos faltantes en el conjunto de datos utilizando `data.isnull().sum()`. Este paso es crucial para determinar si es necesario realizar imputaciones o tratar los valores faltantes de alguna manera.

4. **Codificación One-Hot:**

   Se aplica la codificación one-hot a algunas de las variables categóricas en el conjunto de datos utilizando `pd.get_dummies()`. Esto permite convertir variables categóricas en variables numéricas para su posterior procesamiento.

5. **División del Conjunto de Datos:**

   La proporción para el conjunto de prueba (`test_size`) se define y se utiliza para dividir manualmente el conjunto de datos en conjuntos de entrenamiento y prueba. Se realizan selecciones aleatorias de subconjuntos de datos para el conjunto de prueba y el resto se utiliza como conjunto de entrenamiento.

6. **Normalización de Características:**

   Si es necesario, se normalizan las características utilizando la media y la desviación estándar del conjunto de entrenamiento. La normalización puede ayudar al rendimiento del modelo.

7. **Intercepto y Coeficientes:**

   Se agrega una columna de unos al conjunto de entrenamiento y al conjunto de prueba para representar el término independiente (intercepto) en el modelo de regresión logística. Además, se inicializan los coeficientes a cero.

8. **Entrenamiento del Modelo:**

   El código utiliza el descenso de gradiente para entrenar el modelo de regresión logística. En cada iteración, se calculan las predicciones, el gradiente y se actualizan los coeficientes. Esto se repite un número específico de veces, definido por `num_iterations`.

9. **Predicciones y Precisión:**

   Finalmente, se utilizan los coeficientes entrenados para realizar predicciones en el conjunto de prueba. La precisión del modelo se calcula y se imprime en la consola.

## Uso del Proyecto

- Clona o descarga este repositorio en tu máquina local.

- Asegúrate de tener Python 3.x instalado y las bibliotecas requeridas (pandas y numpy).

- Ejecuta el script `RegresionLogistica.py` para entrenar el modelo y obtener la precisión del mismo.

## Contribuciones

Contribuciones son bienvenidas. Si encuentras errores, deseas mejorar el código o agregar nuevas características, no dudes en abrir un problema o enviar una solicitud de extracción.

