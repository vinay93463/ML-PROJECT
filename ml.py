# ===============================
# Comparative Performance Analysis
# K-Means vs Supervised Classifiers
# On Iris Dataset
# ===============================

# 1. Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 2. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("Dataset Shape:", X.shape)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. K-MEANS CLUSTERING


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

kmeans_labels = kmeans.labels_

# Evaluation for K-Means
ari_score = adjusted_rand_score(y, kmeans_labels)
sil_score = silhouette_score(X, kmeans_labels)

print("\n--- K-Means Results ---")
print("Adjusted Rand Index (ARI):", ari_score)
print("Silhouette Score:", sil_score)
# 5. KNN CLASSIFIER


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)

print("\n--- KNN Results ---")
print("Accuracy:", knn_acc)
print(classification_report(y_test, knn_pred, target_names=target_names))


# 6. SVM CLASSIFIER


svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print("\n--- SVM Results ---")
print("Accuracy:", svm_acc)
print(classification_report(y_test, svm_pred, target_names=target_names))


# 7. DECISION TREE CLASSIFIER


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

print("\n--- Decision Tree Results ---")
print("Accuracy:", dt_acc)
print(classification_report(y_test, dt_pred, target_names=target_names))

# 8. CONFUSION MATRICES


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(confusion_matrix(y_test, knn_pred),
            annot=True, fmt='d', cmap='Blues',
            ax=axes[0])
axes[0].set_title("KNN Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, svm_pred),
            annot=True, fmt='d', cmap='Greens',
            ax=axes[1])
axes[1].set_title("SVM Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, dt_pred),
            annot=True, fmt='d', cmap='Reds',
            ax=axes[2])
axes[2].set_title("Decision Tree Confusion Matrix")

plt.tight_layout()
plt.show()

# 9. ACCURACY COMPARISON GRAPH


models = ['KNN', 'SVM', 'Decision Tree']
accuracies = [knn_acc, svm_acc, dt_acc]

plt.figure(figsize=(6,5))
plt.bar(models, accuracies)
plt.title("Accuracy Comparison of Supervised Models")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.05)
plt.show()


# 10. SUMMARY TABLE


results = pd.DataFrame({
    "Model": ["K-Means (ARI)", "KNN", "SVM", "Decision Tree"],
    "Score": [ari_score, knn_acc, svm_acc, dt_acc]
})

print("\nFinal Comparison Table:")
print(results)