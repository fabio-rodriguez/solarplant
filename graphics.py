
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

TITLESIZE = 10
LBLSIZE = 7


def initialize():
    large = 22; med = 16; small = 12
    params = {'axes.titlesize': large,
            'legend.fontsize': med,
            'figure.figsize': (16, 10),
            'axes.labelsize': med,
            'axes.titlesize': med,
            'xtick.labelsize': med,
            'ytick.labelsize': med,
            'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    
    mpl.rc('xtick', labelsize=LBLSIZE) 
    mpl.rc('ytick', labelsize=LBLSIZE)


def example1():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/economics.csv")

    x = df['date']
    y1 = df['psavert']
    y2 = df['unemploy']

    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1,1,figsize=(10,5), dpi= 80)
    ax1.plot(x, y1, color='tab:red')

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue')

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel('Year', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel('Personal Savings Rate', color='tab:red', fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
    ax1.grid(alpha=.4)

    # ax2 (right Y axis)
    ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(np.arange(0, len(x), 60))
    ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
    ax2.set_title("Personal Savings Rate vs Unemployed: Plotting in Secondary Y Axis", fontsize=22)
    fig.tight_layout()
    plt.show()


def example2():
    df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')

    # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot('date', 'value', data=df, color='tab:red')

    # Decoration
    plt.ylim(50, 750)
    xtick_location = df.index.tolist()[::12]
    xtick_labels = [x[-4:] for x in df.date.tolist()[::12]]
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.grid(axis='both', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.show()


def test_graph(Xs, Y, Ypred, title):

    # Draw Plot
    # fig = plt.figure(figsize=(16,10), dpi= 80)
    # plt.plot(Xs, Y, color='tab:blue', label="Real value")
    # plt.plot(Xs, Ypred, color='tab:red', label="Prediction")
    
    fig = plt.figure()

    plt.subplot(221)
    plt.plot( Xs, Y, color='tab:blue', linestyle='-')
    plt.plot( Xs, Ypred, color='tab:red', linestyle='--')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title(title, fontsize=22)
    plt.grid(axis='both', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend()

    plt.subplot(222)
    plt.plot( Xs, Y, color='tab:blue', linestyle='-')
    plt.plot( Xs, Ypred, color='tab:red', linestyle='--')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title(title, fontsize=22)
    plt.grid(axis='both', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend()
    
    plt.subplot(223)
    plt.plot( Xs, Y, color='tab:blue', linestyle='-')
    plt.plot( Xs, Ypred, color='tab:red', linestyle='--')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title(title, fontsize=22)
    plt.grid(axis='both', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend()
    
    plt.subplot(224)
    plt.plot( Xs, Y, color='tab:blue', linestyle='-')
    plt.plot( Xs, Ypred, color='tab:red', linestyle='--')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title(title, fontsize=22)
    plt.grid(axis='both', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.legend()
    
    # Decoration
    #plt.ylim(50, 750)
    #xtick_location = df.index.tolist()[::12]
    #xtick_labels = [x[-4:] for x in df.date.tolist()[::12]]
    #plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    
    fig.tight_layout()
    plt.show()


def test_graph2(Xs, Y, Ypred, title):

    
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
    
    axes[0].plot(Xs, Y, color='tab:blue', linestyle='-')
    axes[0].plot(Xs, Ypred, color='tab:red', linestyle='--')
    axes[0].title.set_text('1')
    axes[0].set_ylabel('Exp 1')

    axes[1].plot(Xs, Y, color='tab:blue', linestyle='-')
    axes[1].plot(Xs, Ypred, color='tab:red', linestyle='--')
    axes[1].set_ylabel('Exp 2')
  
    axes[2].plot(Xs, Y, color='tab:blue', linestyle='-')
    axes[2].plot(Xs, Ypred, color='tab:red', linestyle='--')
    axes[2].set_ylabel('Exp 3')

    for i in range(3):
        axes[i].grid(axis='both', alpha=.3)
    
    axes[0].legend()

    fig.tight_layout()
    plt.show()


def test_graph3(Xs, Y, Ypred, title):

    fig, axes = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=False)
    fig.set_size_inches(18, 3)

    for j in range(5):    
        axes[0][j].plot(Xs, Y, color='tab:blue', linestyle='-', label='Real Values')
        axes[0][j].plot(Xs, Ypred, color='tab:red', linestyle='--', label='Predictions')
        axes[0][j].title.set_text(f'{j}')
        axes[0][j].title.set_size(TITLESIZE)
        if j== 0:
            axes[0][j].set_ylabel('Exp 1', fontsize=TITLESIZE)

        axes[1][j].plot(Xs, Y, color='tab:blue', linestyle='-')
        axes[1][j].plot(Xs, Ypred, color='tab:red', linestyle='--')
        if j== 0:
            axes[1][j].set_ylabel('Exp 2', fontsize=TITLESIZE)
    
        axes[2][j].plot(Xs, Y, color='tab:blue', linestyle='-')
        axes[2][j].plot(Xs, Ypred, color='tab:red', linestyle='--')
        if j== 0:
            axes[2][j].set_ylabel('Exp 3', fontsize=TITLESIZE)

        for i in range(3):
            axes[i][j].grid(axis='both', alpha=.3)

        if j== 0:
            #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
            axes[0][j].legend(loc='upper left', shadow=True, fontsize='x-small')

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':

    initialize()

    # example1()
    
    Y = np.random.uniform(-50,0,1000)
    Ypred = np.random.uniform(0,50,1000)
    Xs = list(range(len(Y)))
    title = "Air Passengers Traffic (1949 - 1969)"

    test_graph3(Xs, Y, Ypred, title)