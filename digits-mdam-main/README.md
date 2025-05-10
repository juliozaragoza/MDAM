Objetivo es mostrar si hay algun modelo que funcione igual de bien con 100 datos que con 10 000 (tamaño original)

---

### Orden de la presentación

1. Explicar que nuestro objetivo es buscar el modelo que con un dataset pequeño tenga un rendimiento parecido a los modelos con datasets grandes.
2. Mostrar todas las gráficas usando `train_size=0.7` y comprobar que los modelos tienen muy buen rendimiento.
3. Mostrar algunos de los elementos en los que todos los modelos se equivocan.
4. Mostrar gráfica de como varía el rendimiento de cada modelo mientras de reduce el `train_size`. Los datos no están balanceados pero se hace la media.
5. Explicar los 2 tipos de preprocesamiento de datos que se van a realizar para comprobar si mejora o empeora el rendimiento.
6. Empezar a realizar estudio sobre `train_size=0.01`  y ambas formas de preprocesar (fijar en la pantalla los resultados de algún modelo como referencia).
7. Calcular cuantas predicciones realiza mal cada modelo y ver si son las mismas entre modelos.
8. Mostrar todas las gráficas usando `train_size=0.01` y decir que estás gráficas son mejores para el estudio.
9. Explicar cada gráfica del punto 4 junto con sus gemelas de los otros tipos de preprocesamiento. Cada gráfica tiene que ser hecha a partir de la media de varias repeticiones.
10. Cuando se acaben de explicar todas las gráficas, mostrar si las malas predicciones de cada modelo son las mismas o no y en que se diferencian.
11. Probar con datos propios distintos y similares a los que fallan los modelos y ver que pasa.
12. Concluir que modelo es el que:

    1. Aprende mas rápido.
    2. Falla más.
    3. Acierta más.
    4. Mejor modelo en general.


---



1. Comprobamos rendimiento con dataset grande.
2. Vemos donde fallan todos los modelos.
3. Análisis sobre como varia el rendimiento segun el tamaño del dataset.
   1. MLP va mal -> Vamos a mejorarlo
4. Volvemos a realizar análisis de rendimiento segun tamaño, también hacemos zoom.
   1. MLP es una mierda para conjuntos de datos pequeño.
5. Como KNN es el mejor, realizamos un análisis mas profundo
   1. Con k=3 tiene un poco mejor de rendimento.
   2. Todos los demás parámetros no valen para nada.
6. Aplicamos treshold en nuestro mejor modelo.
   1. Creamos la tercera clase "unknow".
   2. El rendimiento mejora significativamente.



Conclusiones sobre nuestro conjunto de datos específico.

* La clase 0 tiene mayor proporcion de error.
* Hay algunos numeros del dataset dificiles de entender hasta para el ser humano.
* MLP va mal en conjuntos de datos pequeño.
* En KNN el único valor importante es K.
* Variar los parámetros de linear y svm no afecta apenas en el rendimiento.
*
