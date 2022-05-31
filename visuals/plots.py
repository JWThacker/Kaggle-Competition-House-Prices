import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.lines as mlines
from numpy import mean

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
      

def corr_heatmap(corr_matrix, annot=False):
    plt.figure(figsize=[12, 9])
    g = sns.heatmap(corr_matrix, annot=annot, vmin=-1, vmax=1, cmap='vlag')
    g.set_title('Pearson Correlation Heatmap for Numerical Features');

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

def pointplots(x, y, data, hue, scale=1):
    g = sns.catplot(x=x, y=y, scale=scale, data=data, hue=hue, kind='point',
                    esitmator=mean, join=False, ci=None, dodge=True,
                    height=7, aspect=1)
    g.fig.suptitle(f'Mean {y} by {x} and colored by {hue}')
