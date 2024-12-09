# Contributor: Benjamin D. Simon and Katie Merriman
# Email: air@nih.gov
# Nov 26, 2024
#
# By downloading or otherwise receiving the SOFTWARE, RECIPIENT may 
# use and/or redistribute the SOFTWARE, with or without modification, 
# subject to RECIPIENT’s agreement to the following terms:
# 
# 1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS 
# OF CANINE OR HUMAN SUBJECTS.  RECIPIENT is responsible for 
# compliance with all laws and regulations applicable to the use 
# of the SOFTWARE.
# 
# 2. The SOFTWARE that is distributed pursuant to this Agreement 
# has been created by United States Government employees. In 
# accordance with Title 17 of the United States Code, section 105, 
# the SOFTWARE is not subject to copyright protection in the 
# United States.  Other than copyright, all rights, title and 
# interest in the SOFTWARE shall remain with the PROVIDER.   
# 
# 3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and 
# the name of the author of the SOFTWARE in all written publications 
# containing any data or information regarding or resulting from use 
# of the SOFTWARE. 
# 
# 4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT 
# ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS 
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.  
# 
# 5.	RECIPIENT agrees not to use any trademarks, service marks, trade names, 
# logos or product names of NCI or NIH to endorse or promote products derived 
# from the SOFTWARE without specific, prior and written permission.
# 
# 6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its 
# own copyright statement to its modifications or derivative works of the SOFTWARE 
# and may provide additional or different license terms and conditions in its 
# sublicenses of modifications or derivative works of the SOFTWARE provided that 
# RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies 
# with the conditions stated in this Agreement. Whenever Recipient distributes or 
# redistributes the SOFTWARE, a copy of this Agreement must be included with 
# each copy of the SOFTWARE.



# Data Processing
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn import tree
import csv
from numpy import mean
from numpy import std

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedKFold

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')




### Import training data saved with features
dfTrain = pd.read_csv("./SortedPatientData_thresh2_train.csv")

### Select features for training

### Just Distance Features

#X_train = dfTrain[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]


### Just Area Features

#X_train = dfTrain[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY"]]

### All Features

X_train = dfTrain[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY",
                 "dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]



X_train.sample(5, random_state=44)
y_train = dfTrain["GroundTruth_EPE"]


### Import Test data saved with features
dfVal = pd.read_csv("SortedPatientData_thresh2_test.csv")

### Select equivalent features for testing

### Just Distance Features

#X_test = dfVal[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]


### Just Area Features

#X_test = dfVal[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY"]]

### All Features

X_test = dfVal[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY",
                 "dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]

y_test = dfVal["GroundTruth_EPE"]

# Input trial information
trial = 'EPE grade, thresh = , features = '

rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)

### Optional Cross Validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(rf_model, X_train, y_train, scoring='balanced_accuracy', cv=cv)
print('Cross Validation: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# find out predictions generated for test set by model
predictions = rf_model.predict(X_test)
print(predictions)


### Path to file path to save predictions
file = open("./EPEgradePredictions.csv", 'a+', newline='')
# writing the data into the file
with file:
    write = csv.writer(file)
    write.writerows([[trial], predictions])

file.close()



# Look at the probabilities for each class generated by model
pred_probs = rf_model.predict_proba(X_test)
pred_classes = rf_model.classes_


# find out how important each feature was to prediction
importances = rf_model.feature_importances_
columns = X_train.columns
i=0
while i < len(columns):
    print(f" The importance of feature '{columns[i]}' is {round(importances[i] * 100, 2)}%.")
    i += 1


# Export the first three decision trees from the forest
'''
for i in range(3):
    export_graphviz(rf_model.estimators_[i], out_file='tree.dot',
                    feature_names=X_train.columns,
                    filled=True,
                    #max_depth=6,
                    impurity=False,
                    proportion=True)

    import pydot

    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')

    # Display in jupyter notebook
    from IPython.display import Image

    Image(filename='tree.png')

    from matplotlib import image as mpimg
    tree_image = mpimg.imread("tree.png")
    plt.imshow(tree_image)
    plt.show()

'''


## USE FOR BINARY
'''
fpr, tpr, thresholds = roc_curve(y_test, pred_probs[:,1], pos_label=1)
roc_auc = roc_auc_score(y_test, pred_probs[:,1])
roc_auc
# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# roc curve for tpr = fpr
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
'''

## USE FOR MULTI-CLASS

y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

colors = ['blue', 'green', 'darkorange', 'red']
aucs = []

for i in range(n_classes):
  fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred_probs[:, i])
  plt.plot(fpr[i], tpr[i], color=colors[i], lw=2)
  print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
  aucs.append(auc(fpr[i], tpr[i]))


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(['EPE 0 - AUC: ' + str(round(aucs[0], 3)), 'EPE 1 - AUC: ' + str(round(aucs[1], 3)), 'EPE 2 - AUC: ' +
            str(round(aucs[2], 3)), 'EPE 3 - AUC: ' + str(round(aucs[3], 3))], loc="lower right")
plt.show()

import scikitplot as skplt
skplt.metrics.plot_roc_curve(y_test, pred_probs)
plt.show()

plt.pause(1)
