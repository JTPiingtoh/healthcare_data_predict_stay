import numpy as np
import matplotlib.pyplot as plt

def generate_score_plot(scores: dict, 
                        metric: str, 
                        k_values: list,
                        dataset_name: str, 
                        ax):
    '''
    Takes model scores as a dict, and produces a grouped bar plot
    based on k values.
    '''

# plotting the averages
    x = np.arange(len(k_values))
    width = 0.25
    multiplier = 0

    for model, score in scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset,np.average(score, axis=1), width, label=model)
        ax.bar_label(rects, padding=3,fmt="%.2f")
        multiplier +=1

    ax.set_title(f"{dataset_name}")
    ax.legend(loc='lower left', ncols=3)
    ax.set_ylim(0,1)
    ax.set_xticks(x + width, k_values)
    ax.set_xlabel("K")
    ax.set_ylabel(f"{metric}")