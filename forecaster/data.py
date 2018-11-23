

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import seaborn as sns
from statsmodels import api as sm





class Data(pd.DataFrame):

    def __init__(self,data):

        super().__init__(data)





class 