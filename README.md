#  Breast Cancer Classification using Support Vector Machines (SVM)

##  Objective
To build a classification model using Support Vector Machines for binary classification of breast cancer as **Malignant (M)** or **Benign (B)** based on extracted features from tumor images.

---

##  Tools & Libraries Used
- Python 
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (SVC, GridSearchCV, PCA, etc.)

---

##  Dataset
The dataset contains 30 numeric features (e.g., radius, texture, area) computed from digitized images of breast mass. The target variable `diagnosis` includes:
- M = Malignant
- B = Benign

### Sample Columns:
- `radius_mean`, `texture_mean`, `smoothness_mean`, ...
- `diagnosis` (target)

---

##  Steps Performed

###  1. Data Preparation
- Removed the `id` column.
- Encoded `diagnosis` (M → 1, B → 0).
- Standardized the feature values.

###  2. Train-Test Split
- 80% for training and 20% for testing using `train_test_split`.

###  3. Model Training
- Trained two SVM classifiers:
  - **Linear Kernel**
  - **RBF (Radial Basis Function) Kernel**

###  4. Hyperparameter Tuning
- Used `GridSearchCV` to optimize:
  - `C` values: [0.1, 1, 10, 100]
  - `gamma` values: ['scale', 0.01, 0.1, 1]

###  5. Evaluation
- Confusion matrix and classification report used.
- Cross-validation (CV = 5) accuracy measured.

###  6. Visualization
- Reduced data to 2 dimensions using PCA.
- Plotted decision boundary using `matplotlib`.

---

##  Results
- **Best Parameters** from GridSearch: e.g., `{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}`
- **Cross-validation Accuracy**: ~95–98% depending on parameters
- Decision boundary shows clear separation between malignant and benign samples after PCA.

---

##  What You Learn

-  How SVM maximizes margin between classes.
-  How kernel trick allows nonlinear classification.
-  How to tune `C` (regularization) and `gamma` (kernel width).
-  How to visualize decision boundaries in 2D space.

---

##  How to Run This Notebook
1. Install dependencies: `pip install numpy pandas matplotlib scikit-learn`
2. Load the dataset CSV.
3. Run each cell in order to train, evaluate, and visualize the SVM model.

---

