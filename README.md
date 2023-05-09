# Housing Prices
Kaggle Housing Prices Competition
This repo is my attempt to walk through the famous Kaggle Housing Prices Competition.  I'll break it down into the following:
1) EDA Jupyter Notebook
2) Data Transformation Pipeline (py files) - including all the necessary transformers  and custom classes to transform the training/testing dataset
3) Model Notebooks. Initially I've been training a simple regression model but will expand this out to other algorithms
4) Model and Feature Analysis Notebook

All of these files will be coming soon.


## TransformerClasses.py File ##
This file contains 4 classes that are used to transform the categorical features in the Housing Prices datasets.  They are generalized in a way that they could be used for other types of data but given that they were devised to help with creating an sklearn pipeline, they are based used for this dataset/purpose.  To make use of these classes you can simply import the file:

```import TransformerClasses as tc```

Alternatively, you can import each class separately:
```from TransformerClasses import Cat2Val
from TransformerClasses import Cat2Dummies
from TransformerClasses import CombinationOHETransformer
from TransformerClasses import SelectiveScaler```


All 4 classes have the same basic methods: fit, transform and fit_transform and were developed according to the needs for addressing specific columns in different ways.

### Cat2Val Class ###
Cat2Val was birthed from the MiscFeatures and MiscVal columns.  Both columns are used to convey a feature of a house and the associated value of that feature. Since the features are related to one-another and are dependent upon specific values from one column to understand the other,  I created the Cat2Val class which converts the unique categorical values in the MiscFeatures column into columns and then associate the values in the MiscVal colun to those features.

### Cat2Dummies Class ###
Cat2Dummies is functionally the same as pd.get_dummies except for the fact that it saves some unnecessary coding steps when I want to consolidate some of the dummy columns into one.  This was the case with the Roof Style and Roof Material columns.  Each had a category that was heavily utilized and a few features that weren't.  The Cat2Dummies allowed me to put all of the less frequently occurring categories together so that I didn't have a lot of unnecessary and very empty columns created from the dummy process.

### CombinationOHETransformer Class ###
The Condition1 and Condition2 Columns are functionality the same column but since each row represents a home, splitting them into 2 columns allowed for the preservation of a one to 2 (many) relationship.  This class combines the two columns together such that there is one unique set of "dummy" columns corresponding to the values both of the original columns can take.  Additionally, users can specify whether they want the resulting columns to contain dummy values (1 - present / 0 - not present) or utilize separate aggregate columns to associate to each column.  It's very similiar to the Cat2Val class with the exception that this is combining functionaly similar columns.

### SelectiveScaler Class ###
This is really a wrapper for the sklearn scalers simply providing more control over what columns are scaled. I didn't want to go to great lengths to prevent the binary columns (which shouldn't be scaled) from being scaled with columns that have greater range of data. 


