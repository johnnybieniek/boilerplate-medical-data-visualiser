import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
def bmi(weight, height):
  height /= 100
  return weight/(height**2)

# Add 'overweight' column
df['overweight'] = df.apply(lambda row: 1 if bmi(row['weight'],row['height']) > 25 else 0, axis = 1)
# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

norm = {1:0,2:1,3:1}
df['gluc'].replace(norm,inplace=True)
df['cholesterol'].replace(norm,inplace=True)
df.apply(pd.to_numeric, errors='ignore')


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active','overweight'], id_vars = 'cardio')



    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.DataFrame(df_cat.groupby(['variable', 'value', 'cardio'])['value'].count()).rename(columns={'value': 'total'}).reset_index()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(data = df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True



    # Set up the matplotlib figure
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, vmin=-0.10, vmax=0.27, linewidths=.5)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig