import pandas as pd
import numpy as np
from RetailChurnTemplateUtility import *

def azureml_main(df1=None, df2=None):
    key_column = 'UserId'
    IsDevelopment = True

    # Feature Engineering
    churnUtil = RetailChurnTemplateUtility()
    
    # Return value must be of a sequence of pandas.DataFrame
    return churnUtil.calculateAverages(df1,
                                       df2,
                                       key_column,
                                       uniquable_columns=df2.columns, 
                                       summable_columns=df1.columns)
