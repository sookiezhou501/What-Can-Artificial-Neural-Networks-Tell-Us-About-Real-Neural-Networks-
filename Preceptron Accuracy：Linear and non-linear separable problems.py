import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 32,
    'axes.labelsize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 26,
    'figure.titlesize': 36
})

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')


class Perceptron:
    """Rosenblatt Perceptron Implementation"""
    
    def __init__(self, learning_rate=1.0, max_epochs=500):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.convergence_epoch = None
        
    def activate(self, x):
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activate(linear_output)
    
    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if np.array_equal(np.unique(y), [0, 1]):
            y = np.where(y == 0, -1, 1)
        
        for epoch in range(self.max_epochs):
            errors = 0
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                if y_i * linear_output <= 0:
                    self.weights += self.lr * y_i * x_i
                    self.bias += self.lr * y_i
                    errors += 1
            
            if errors == 0:
                self.convergence_epoch = epoch
                return True
        
        return False
    
    def score(self, X, y):
        predictions = self.predict(X)
        if np.array_equal(np.unique(y), [0, 1]):
            y = np.where(y == 0, -1, 1)
        return np.mean(predictions == y)


def generate_linear_separable_data(n_samples=200, noise=0.15, separation=3.0):
    X1 = np.random.randn(n_samples // 2, 2) + np.array([separation/2, separation/2])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-separation/2, -separation/2])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
    X += np.random.randn(n_samples, 2) * noise
    return X, y


def generate_xor_data(n_samples=200, separation=2.5):
    X1 = np.random.randn(n_samples // 4, 2) + np.array([separation/2, separation/2])
    X2 = np.random.randn(n_samples // 4, 2) + np.array([-separation/2, -separation/2])
    X3 = np.random.randn(n_samples // 4, 2) + np.array([-separation/2, separation/2])
    X4 = np.random.randn(n_samples // 4, 2) + np.array([separation/2, -separation/2])
    X = np.vstack([X1, X2, X3, X4])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
    return X, y


def generate_circular_data(n_samples=200, radius=3.0, noise=0.2):
    theta1 = np.random.uniform(0, 2*np.pi, n_samples // 2)
    r1 = np.random.uniform(0, radius/2, n_samples // 2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    
    theta2 = np.random.uniform(0, 2*np.pi, n_samples // 2)
    r2 = np.random.uniform(radius/2, radius, n_samples // 2)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
    X += np.random.randn(n_samples, 2) * noise
    return X, y


def generate_checkerboard_data(n_samples=400, grid_size=2):
    X = np.random.uniform(-grid_size, grid_size, (n_samples, 2))
    y = np.ones(n_samples)
    for i in range(n_samples):
        if (np.floor(X[i, 0] + grid_size) + np.floor(X[i, 1] + grid_size)) % 2 == 0:
            y[i] = -1
    return X, y


def generate_spiral_data(n_samples=200, noise=0.2):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 3 * np.pi
    
    x1 = np.column_stack([-theta * np.cos(theta) + np.random.randn(n) * noise,
                          theta * np.sin(theta) + np.random.randn(n) * noise])
    x2 = np.column_stack([theta * np.cos(theta) + np.random.randn(n) * noise,
                          -theta * np.sin(theta) + np.random.randn(n) * noise])
    
    X = np.vstack([x1, x2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y


def run_multiple_experiments(X, y, n_experiments=50, max_epochs=500):
    convergence_epochs = []
    final_accuracies = []
    
    for i in range(n_experiments):
        perceptron = Perceptron(learning_rate=1.0, max_epochs=max_epochs)
        converged = perceptron.fit(X, y)
        
        if perceptron.convergence_epoch is not None:
            convergence_epochs.append(perceptron.convergence_epoch)
        else:
            convergence_epochs.append(max_epochs)
        
        accuracy = perceptron.score(X, y)
        final_accuracies.append(accuracy)
    
    return convergence_epochs, final_accuracies


def plot_accuracy_comparison(all_results, save_path='perceptron_accuracy_comparison.png'):
    dataset_names = list(all_results.keys())
    accuracy_data = [all_results[name]['accuracies'] for name in dataset_names]
    

    short_names = ['Linear', 'XOR', 'Circular', 'Checker', 'Spiral']
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#e67e22']
    
    fig, ax = plt.subplots(figsize=(18, 12))
    

    bp = ax.boxplot(accuracy_data, labels=short_names, patch_artist=True,
                    widths=0.65, showmeans=True, meanline=True,
                    meanprops={'linestyle': '-', 'color': 'black', 'linewidth': 3,
                              'marker': 'D', 'markerfacecolor': 'black', 
                              'markersize': 14, 'markeredgecolor': 'black'})
    

    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
        box.set_edgecolor('black')
        box.set_linewidth(2.5)
    

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(2.5)
        whisker.set_linestyle('-')
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(2.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(3.5)
    

    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', linewidth=3, alpha=0.8, 
               label='Perfect Accuracy (100%)')
    ax.axhline(y=0.5, color='#e74c3c', linestyle=':', linewidth=3, alpha=0.8, 
               label='Random Guessing (50%)')
    
    ax.set_ylabel('Final Accuracy', fontsize=32, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=32, fontweight='bold')
    ax.set_title('Perceptron Accuracy: Linearly vs Non-linearly Separable Problems',
                 fontsize=36, fontweight='bold', pad=20)
    

    ax.tick_params(axis='x', rotation=0, labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    
    ax.legend(loc='lower right', fontsize=26, framealpha=0.9)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    

    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], fontsize=22)
    

    for i, (name, data) in enumerate(all_results.items()):
        median_val = np.median(data['accuracies'])
        mean_val = np.mean(data['accuracies'])
        

        q1 = np.percentile(data['accuracies'], 25)
        q3 = np.percentile(data['accuracies'], 75)
        
 
        iqr = q3 - q1
        upper_whisker = min(np.max(data['accuracies']), q3 + 1.5 * iqr)
        lower_whisker = max(np.min(data['accuracies']), q1 - 1.5 * iqr)
        

        if median_val > 0.8:

            text_y_median = upper_whisker + 0.04
            text_y_mean = upper_whisker - 0.03
            va_median = 'bottom'
            va_mean = 'top'
        elif median_val < 0.6:

            text_y_median = lower_whisker - 0.06
            text_y_mean = lower_whisker - 0.12
            va_median = 'top'
            va_mean = 'bottom'
        else:

            text_y_median = upper_whisker + 0.04
            text_y_mean = upper_whisker - 0.03
            va_median = 'bottom'
            va_mean = 'top'
        
 
        ax.text(i + 1, text_y_median, f'Median: {median_val:.1%}', 
                ha='center', va=va_median, fontsize=22, fontweight='bold')
        ax.text(i + 1, text_y_mean, f'Mean: {mean_val:.1%}', 
                ha='center', va=va_mean, fontsize=22, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    return fig


def main():
    print("=" * 60)
    print("PERCEPTRON ACCURACY COMPARISON EXPERIMENT")
    print("=" * 60)
    
    N_EXPERIMENTS = 50
    N_SAMPLES = 300
    MAX_EPOCHS = 500
    
    all_results = {}
    
    datasets = [
        {'name': 'Linearly Separable', 
         'generator': generate_linear_separable_data,
         'params': {'n_samples': N_SAMPLES, 'noise': 0.15, 'separation': 3.0}},
        {'name': 'XOR Problem', 
         'generator': generate_xor_data,
         'params': {'n_samples': N_SAMPLES, 'separation': 2.5}},
        {'name': 'Circular Boundary', 
         'generator': generate_circular_data,
         'params': {'n_samples': N_SAMPLES, 'radius': 3.0, 'noise': 0.2}},
        {'name': 'Checkerboard', 
         'generator': generate_checkerboard_data,
         'params': {'n_samples': N_SAMPLES, 'grid_size': 2}},
        {'name': 'Spiral', 
         'generator': generate_spiral_data,
         'params': {'n_samples': N_SAMPLES, 'noise': 0.2}}
    ]
    
    for dataset in datasets:
        print(f"Running {N_EXPERIMENTS} experiments on {dataset['name']}...")
        
        X, y = dataset['generator'](**dataset['params'])
        convergence_epochs, accuracies = run_multiple_experiments(
            X, y, n_experiments=N_EXPERIMENTS, max_epochs=MAX_EPOCHS
        )
        
        all_results[dataset['name']] = {
            'convergence_epochs': convergence_epochs,
            'accuracies': accuracies
        }
        
        conv_rate = np.sum(np.array(convergence_epochs) < MAX_EPOCHS) / N_EXPERIMENTS * 100
        print(f"  - Convergence Rate: {conv_rate:.1f}%")
        print(f"  - Mean Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
        print()
    
    print("=" * 60)
    print("Generating Accuracy Comparison Figure...")
    print("=" * 60)
    
    fig = plot_accuracy_comparison(all_results, save_path='perceptron_accuracy_comparison.png')
    
    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    results = main()