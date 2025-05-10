from os import name
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.model_selection import train_test_split


# Load the database
mat_file =  "BigDigits.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

taska = False

data = mat["data"]      # read feature vectors
labs = mat["labs"] - 1  # read labels 1..10

allNlabs = np.unique(labs) # all labs 0 .. 9

classsiz = ()
for c in allNlabs:
    classsiz = classsiz + (np.size(np.nonzero(labs==c)),)  
print ('\n%% Class labels are: %s' % (allNlabs,) )   
print ('%% Class frequencies are: %s' % (classsiz,))


# Let's say my digit is ...
myDigit = 5

otherDigits  = np.setdiff1d(allNlabs,myDigit)
other3Digits = np.random.permutation(otherDigits)[:3]

if taska:
    others = other3Digits
else:
    others = otherDigits

print ('class 1 = %s' % myDigit)
print ('class 2 = %s' % others)

# To construct a 2-class dataset you can use the same matrix
# data and change the vector of labels

aux = labs
classone = np.in1d(labs,myDigit)
classtwo = np.in1d(labs,others)
aux[classone] = 0  # class one
aux[classtwo] = 1  # class two

# Features
X = data[np.logical_or(classone,classtwo)]
# (unchanged) labels
y = aux[np.logical_or(classone,classtwo)]


# Show some digits

hwmny = 20
some1 = np.random.permutation(np.where(y==0)[0])[:hwmny]
some2 = np.random.permutation(np.where(y==1)[0])[:hwmny]

img1 = np.reshape(X[some1,:],(28*hwmny,28)).T
plt.figure(figsize=(10,3))
plt.imshow(img1, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Show digits in class one (0) = '+str(myDigit) )
plt.show()


img2 = np.reshape(X[some2,:],(28*hwmny,28)).T
plt.figure(figsize=(10,3))
plt.imshow(img2, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Show digits in class two (1) = '+str(others) )
plt.show()


train_quantity = 500

train_size = train_quantity / len(labs)  # variable para el train y el test

X_train, X_test, y_train, y_test = train_test_split(data, labs, train_size=train_size)

class Linear:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_f1(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return f1_score(y_test, y_pred, average="macro")

    def roc_curve(self, X_test, y_test):
        y_prob = self.model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        return fpr, tpr, thresholds

    def predict(self, X_test):
        return self.model.predict(X_test)


model = Linear()

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo
f1 = model.evaluate_f1(X_test, y_test)
print("F1 Score:", f1)

# Obtener la curva ROC
fpr_linear, tpr_linear, thresholds_linear = model.roc_curve(X_test, y_test)
roc_auc_linear = auc(fpr_linear, tpr_linear)




class QDA:
    def __init__(self):
        self.model = QuadraticDiscriminantAnalysis()

    def fit(self, X_train, y_train):
        """Entrena el modelo QDA con los datos de entrenamiento."""
        self.model.fit(X_train, y_train)

    def evaluate_f1(self, X_test, y_test):
        """Calcula el F1 Score para las predicciones del modelo."""
        qda_pred = self.model.predict(X_test)
        qda_pred = np.where(np.array(qda_pred) <= 0.5, 0, 1)  # Umbral de 0.5
        return f1_score(y_test, qda_pred, average="macro")

    def roc_curve(self, X_test, y_test):
        """Calcula la curva ROC y devuelve FPR, TPR y umbrales."""
        y_scores = self.model.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        return fpr, tpr, thresholds

    def predict(self, X_test):
        """Realiza predicciones con el modelo entrenado."""
        return self.model.predict(X_test)

# Crear el modelo QDA
qda_model = QDA()

# Entrenar el modelo
qda_model.fit(X_train, y_train)

# Evaluar el modelo con F1 Score
f1 = qda_model.evaluate_f1(X_test, y_test)
print("F1 Score:", f1)

# Obtener la curva ROC
fpr_qda, tpr_qda, thresholds_qda = qda_model.roc_curve(X_test, y_test)
roc_auc_qda = auc(fpr_qda, tpr_qda)


# Graficar ambas curvas ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_linear, tpr_linear, label=f"Linear - AUC {roc_auc_linear:.2f}", color="blue")
plt.plot(fpr_qda, tpr_qda, label=f"QDA - AUC {roc_auc_qda:.2f}", color="red")

# Configurar la gráfica
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparación de Curvas ROC")
plt.legend(loc="lower right")
plt.grid()
plt.show()