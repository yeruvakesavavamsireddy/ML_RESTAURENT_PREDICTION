import matplotlib.pyplot as plt

def plot_distribution(df, column):
    plt.figure()
    df[column].hist()
    plt.title(f"Distribution of {column}")
    plt.savefig(f"outputs/{column}_distribution.png")
    plt.close()