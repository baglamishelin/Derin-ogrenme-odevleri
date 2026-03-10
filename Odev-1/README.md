# Breast Cancer Classification using Random Forest

## 📌 Project Description
This project focuses on classifying breast tumors as benign (0) or malignant (1) using a machine learning approach. The task is a binary classification problem.

The Random Forest algorithm was used to perform the classification and evaluate the model performance.

---

## 🧠 Selected Algorithm
Random Forest was chosen because:
- It performs well on high-dimensional datasets
- It can model non-linear relationships
- It is resistant to overfitting
- It provides strong performance in medical classification problems

---

## ⚙️ Implementation Steps
1. Dataset loaded in Google Colab.
2. Unnecessary columns removed.
3. Target variable encoded (Benign = 0, Malignant = 1).
4. Dataset split into training (%80) and testing (%20).
5. Features scaled using StandardScaler.
6. Random Forest model trained on training data.
7. Predictions made on test data.
8. Model evaluated using classification metrics.

---

## 📊 Model Performance

- **Accuracy:** 0.9737  
- **Precision:** 1.00  
- **Recall:** 0.9286  
- **F1-Score:** 0.963  

The results show that the model achieved high classification performance. The precision score of 1.00 indicates that all predicted malignant cases were correctly classified. The recall score shows that most malignant tumors were successfully detected.

---

## 🛠 Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- Google Colab

---

## 👩‍💻 Author
Helin Bağlamış 


