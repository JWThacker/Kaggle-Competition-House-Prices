# imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.lines as mlines
from numpy import mean

"""Plot frequency plot for a single categorical feature

Positinal Arguments:
    feature: name of the feature as a string
    df: dataframe containing the data for the feature as a Pandas.DataFrame

Keyword Arguments:
    color: fill color for the bars at each level
    horizontal: if True, plot the categorical feature on the y-axis.
    order: the order to plot the levels of the feature
    palette: color the bars of the frequency plot according to a palette
             (for ordinal categorical features)
"""
def freq_plot(feature, df, color=None, horizontal=False, order=None, palette=None):
    if horizontal:
        g = sns.catplot(y=feature, data=df, kind='count', height=7, aspect=1, color=color,
                        order=order, palette=palette)
        g.fig.suptitle('Distribution of ' + feature, y=1.05)
        g.set(ylabel='Frequency');
    else:
        g = sns.catplot(x=feature, data=df, kind='count', height=7, aspect=1, color=color,
                        order=order, palette=palette)
        g.fig.suptitle('Distribution of ' + feature, y=1.05)
        g.set(ylabel='Frequency');
      

"""Plot heatmap for a correlation matrix

Positinal Arguments:
    corr_matrix: correlation matrix as a Pandas.DataFrame

Keyword Arguments:
    annot: if True, include correlation values centered within the squares of the heatmap
"""
def corr_heatmap(corr_matrix, annot=False):
    plt.figure(figsize=[12, 9])
    g = sns.heatmap(corr_matrix, annot=annot, vmin=-1, vmax=1, cmap='vlag')
    g.set_title('Pearson Correlation Heatmap for Numerical Features');

"""Plot boxplot between the levels of a categorical variable and numerical variable

Positinal Arguments:
    x: name of the variable on the x-axis
    y: name of the variable on the y-axis
    data: the Pandas.DataFrame containing x and y

Keyword Arguments:
    color: the color of the boxes if the categorical feature is nominal
    horizontal: if True, plot the categorical feature on the y-axis.
    order: the order to plot the levels of the feature
    palette: color the bars of the frequency plot according to a palette
             (for ordinal categorical features)
"""
def boxplots(x, y, data, color=None, horizontal=False, order=None, palette=None):
    if horizontal:
        g = sns.catplot(y=x, x=y, data=data, color=color,
                    height=7, aspect=1, kind='box', order=order, palette=palette);
        g.fig.suptitle(f'Boxplot of {y} by {x}',
                       y=1.05)
    else:
        g = sns.catplot(x=x, y=y, data=data, color=color,
                    height=7, aspect=1, kind='box', order=order, palette=palette);
        g.fig.suptitle(f'Boxplot of {y} by {x}',
                       y=1.05)

"""Plot pointplots between a categorical feature and numerical feature

Positinal Arguments:
    x: name of the feature on the x-axis
    y: name of the feature on the y-axis
    data: the Pandas.DataFrame containing x and y 
    hue: the color of the points and line

Keyword Arguments:
    scale: the size to make the points
"""
def pointplots(x, y, data, hue, scale=1):
    g = sns.catplot(x=x, y=y, scale=scale, data=data, hue=hue, kind='point',
                    esitmator=mean, join=False, ci=None, dodge=True,
                    height=7, aspect=1)
    g.fig.suptitle(f'Mean {y} by {x} and colored by {hue}')
