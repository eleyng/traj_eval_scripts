# Import seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn_image as isns
isns.set_context(mode="poster", fontfamily="serif")
from matplotlib.font_manager import get_font_names

# Instructions: indicate the path to the data file, and the columns in the csv file that correspond to the columns of interest for datas
data_csv_file = 'scripts/data.csv' 
dataframe = pd.read_csv()
sns.violinplot(data=dataframe,  split=True, x='Condition', y='Percentage of Trials', hue='Planner Type', inner='quart', cut=0)
plt.legend(loc='upper left')
# plt.title('Confusion Matrices across Subjects per Planner Type')
plt.show()