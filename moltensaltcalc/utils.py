import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(data_dict, xlabel, ylabel, title, legend_labels=None):
    """
    Plot comparison of different simulation results.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with data to plot (key: label, value: (x, y))
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    legend_labels : list
        Labels for legend
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    
    for i, (label, (x, y)) in enumerate(data_dict.items()):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        
        plt.scatter(x, y, label=label, marker=marker, color=color, s=80)
        plt.plot(x, y, linestyle='--', color=color, alpha=0.5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_results(results, filename="results.json"):
    """
    Save analysis results to JSON file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    filename : str
        Output filename
    """
    import json
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    print(f"Results saved to {filename}")


def load_results(filename="results.json"):
    """
    Load analysis results from JSON file.
    
    Parameters:
    -----------
    filename : str
        Input filename
    
    Returns:
    --------
    dict
        Loaded results
    """
    import json
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results