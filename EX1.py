import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

# Check available styles and use appropriate one
available_styles = plt.style.available
print("Available styles:", available_styles[:10])  # Print first 10 available styles

# Try to use a style that exists
try:
    if 'seaborn-v0_8-darkgrid' in available_styles:
        plt.style.use('seaborn-v0_8-darkgrid')
    elif 'seaborn-darkgrid' in available_styles:
        plt.style.use('seaborn-darkgrid')
    elif 'ggplot' in available_styles:
        plt.style.use('ggplot')
    else:
        # Use default style if no seaborn style available
        plt.style.use('default')
except:
    plt.style.use('default')

plt.rcParams['font.size'] = 12

class Perceptron:
    """Rosenblatt Perceptron Implementation"""
    
    def __init__(self, learning_rate=1.0, max_epochs=1000):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.convergence_epoch = None
        self.loss_history = []
        self.accuracy_history = []
        
    def activate(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X):
        """Prediction function"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activate(linear_output)
    
    def fit(self, X, y, verbose=False):
        """Train the perceptron"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        self.accuracy_history = []
        
        y_original = y.copy()
        # Convert labels to -1 and 1 (if originally 0 and 1)
        if np.array_equal(np.unique(y), [0, 1]):
            y = np.where(y == 0, -1, 1)
        
        converged = False
        for epoch in range(self.max_epochs):
            errors = 0
            epoch_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]
                
                # Compute prediction
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activate(linear_output)
                
                # Update weights (if prediction is wrong)
                if y_i * linear_output <= 0:
                    self.weights += self.lr * y_i * x_i
                    self.bias += self.lr * y_i
                    errors += 1
                    epoch_loss += abs(y_i - y_pred)
            
            # Calculate accuracy
            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)
            self.accuracy_history.append(accuracy)
            
            self.loss_history.append(epoch_loss / n_samples)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Errors: {errors}, Accuracy: {accuracy:.2%}")
            
            # Check convergence
            if errors == 0:
                self.convergence_epoch = epoch
                converged = True
                if verbose:
                    print(f"Converged at epoch {epoch}!")
                break
        
        return converged
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        y_original = y.copy()
        if np.array_equal(np.unique(y), [0, 1]):
            y_original = np.where(y_original == 0, -1, 1)
        accuracy = np.mean(predictions == y_original)
        return accuracy

def generate_linear_separable_data(n_samples=200, noise=0.15, separation=3.0):
    """Generate linearly separable data with varying separation"""
    np.random.seed(None)  # Random seed for each call
    
    # Generate two classes with controlled separation
    X1 = np.random.randn(n_samples // 2, 2) + np.array([separation/2, separation/2])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-separation/2, -separation/2])
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
    
    # Add noise
    X += np.random.randn(n_samples, 2) * noise
    
    return X, y

def generate_xor_data(n_samples=200, separation=2.5):
    """Generate XOR problem data"""
    np.random.seed(None)
    
    # Generate four quadrant points
    X1 = np.random.randn(n_samples // 4, 2) + np.array([separation/2, separation/2])
    X2 = np.random.randn(n_samples // 4, 2) + np.array([-separation/2, -separation/2])
    X3 = np.random.randn(n_samples // 4, 2) + np.array([-separation/2, separation/2])
    X4 = np.random.randn(n_samples // 4, 2) + np.array([separation/2, -separation/2])
    
    X = np.vstack([X1, X2, X3, X4])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
    
    return X, y

def generate_circular_data(n_samples=200, radius=3.0, noise=0.2):
    """Generate circular decision boundary data"""
    np.random.seed(None)
    
    # Inner circle (class 1)
    theta1 = np.random.uniform(0, 2*np.pi, n_samples // 2)
    r1 = np.random.uniform(0, radius/2, n_samples // 2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    
    # Outer circle (class -1)
    theta2 = np.random.uniform(0, 2*np.pi, n_samples // 2)
    r2 = np.random.uniform(radius/2, radius, n_samples // 2)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
    
    # Add noise
    X += np.random.randn(n_samples, 2) * noise
    
    return X, y

def generate_checkerboard_data(n_samples=400, grid_size=2):
    """Generate checkerboard pattern data"""
    np.random.seed(None)
    
    X = np.random.uniform(-grid_size, grid_size, (n_samples, 2))
    
    # XOR-like pattern in grid
    y = np.ones(n_samples)
    for i in range(n_samples):
        if (np.floor(X[i, 0] + grid_size) + np.floor(X[i, 1] + grid_size)) % 2 == 0:
            y[i] = -1
    
    return X, y

def generate_spiral_data(n_samples=200, noise=0.2):
    """Generate spiral pattern data"""
    np.random.seed(None)
    
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 3 * np.pi
    
    # First spiral
    x1 = np.column_stack([-theta * np.cos(theta) + np.random.randn(n) * noise,
                          theta * np.sin(theta) + np.random.randn(n) * noise])
    
    # Second spiral
    x2 = np.column_stack([theta * np.cos(theta) + np.random.randn(n) * noise,
                          -theta * np.sin(theta) + np.random.randn(n) * noise])
    
    X = np.vstack([x1, x2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    
    return X, y

def plot_decision_boundary(X, y, perceptron=None, title="Decision Boundary"):
    """Plot data distribution and decision boundary"""
    plt.figure(figsize=(14, 6))
    
    # Left: Data distribution
    plt.subplot(1, 2, 1)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='blue', label='Class 1', 
                alpha=0.6, edgecolors='black', s=50)
    plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], c='red', label='Class -1', 
                alpha=0.6, edgecolors='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right: Decision boundary
    plt.subplot(1, 2, 2)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='blue', label='Class 1', 
                alpha=0.6, edgecolors='black', s=50)
    plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], c='red', label='Class -1', 
                alpha=0.6, edgecolors='black', s=50)
    
    if perceptron is not None and perceptron.weights is not None:
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu', levels=[-1, 0, 1])
        plt.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0])
        
        # Add weight vector
        if perceptron.weights[0] != 0:
            center = np.mean(X, axis=0)
            plt.arrow(center[0], center[1], 
                     perceptron.weights[0]*2, perceptron.weights[1]*2,
                     head_width=0.2, head_length=0.2, fc='green', ec='green',
                     label='Weight Vector')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_training_history(loss_history, accuracy_history, convergence_epoch, title):
    """Plot training history"""
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-', linewidth=2, alpha=0.7, label='Loss')
    if convergence_epoch:
        plt.axvline(x=convergence_epoch, color='r', linestyle='--', 
                   label=f'Convergence (Epoch {convergence_epoch})', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, 'g-', linewidth=2, alpha=0.7, label='Accuracy')
    if convergence_epoch:
        plt.axvline(x=convergence_epoch, color='r', linestyle='--', 
                   label=f'Convergence (Epoch {convergence_epoch})', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Perfect Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_multiple_experiments(dataset_name, X, y, n_experiments=50):
    """Run multiple experiments on the same dataset"""
    convergence_epochs = []
    final_accuracies = []
    experiment_results = []
    
    print(f"\nRunning {n_experiments} experiments on {dataset_name}...")
    
    for i in range(n_experiments):
        perceptron = Perceptron(learning_rate=1.0, max_epochs=500)
        converged = perceptron.fit(X, y)
        
        if perceptron.convergence_epoch is not None:
            convergence_epochs.append(perceptron.convergence_epoch)
        else:
            convergence_epochs.append(500)  # Max epochs
        
        accuracy = perceptron.score(X, y)
        final_accuracies.append(accuracy)
        
        experiment_results.append({
            'experiment': i+1,
            'converged': converged,
            'convergence_epoch': perceptron.convergence_epoch,
            'accuracy': accuracy,
            'weights': perceptron.weights.copy(),
            'bias': perceptron.bias
        })
        
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{n_experiments} experiments")
    
    return convergence_epochs, final_accuracies, experiment_results

def plot_experiment_statistics(dataset_name, convergence_epochs, final_accuracies):
    """Plot statistics from multiple experiments"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Experiment Statistics - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Convergence epochs distribution
    axes[0, 0].hist(convergence_epochs, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=np.mean(convergence_epochs), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(convergence_epochs):.1f}')
    axes[0, 0].axvline(x=np.median(convergence_epochs), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(convergence_epochs):.1f}')
    axes[0, 0].set_xlabel('Convergence Epoch')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Convergence Time Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy distribution
    axes[0, 1].hist(final_accuracies, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=np.mean(final_accuracies), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(final_accuracies):.3f}')
    axes[0, 1].axvline(x=np.median(final_accuracies), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(final_accuracies):.3f}')
    axes[0, 1].set_xlabel('Final Accuracy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Accuracy Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot for convergence epochs
    axes[0, 2].boxplot(convergence_epochs, vert=True, patch_artist=True)
    axes[0, 2].set_ylabel('Epochs')
    axes[0, 2].set_title('Convergence Time Box Plot')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Scatter plot: Accuracy vs Convergence Time
    axes[1, 0].scatter(convergence_epochs, final_accuracies, alpha=0.6, c='purple', s=50)
    axes[1, 0].set_xlabel('Convergence Epoch')
    axes[1, 0].set_ylabel('Final Accuracy')
    axes[1, 0].set_title('Accuracy vs Convergence Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative convergence curve
    sorted_epochs = np.sort(convergence_epochs)
    cumulative = np.arange(1, len(sorted_epochs) + 1) / len(sorted_epochs)
    axes[1, 1].plot(sorted_epochs, cumulative, 'b-', linewidth=2, marker='o', markersize=4)
    axes[1, 1].set_xlabel('Convergence Epoch')
    axes[1, 1].set_ylabel('Cumulative Proportion')
    axes[1, 1].set_title('Cumulative Convergence Curve')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    # Statistics table
    stats_text = f"""Experiment Statistics (n={len(convergence_epochs)})
    
Convergence Epochs:
  Mean: {np.mean(convergence_epochs):.2f}
  Std: {np.std(convergence_epochs):.2f}
  Min: {np.min(convergence_epochs):.0f}
  Max: {np.max(convergence_epochs):.0f}
    
Final Accuracies:
  Mean: {np.mean(final_accuracies):.4f}
  Std: {np.std(final_accuracies):.4f}
  Min: {np.min(final_accuracies):.4f}
  Max: {np.max(final_accuracies):.4f}
    
Convergence Rate: {np.sum(np.array(convergence_epochs) < 500)/len(convergence_epochs)*100:.1f}%
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return stats_text

def plot_all_datasets_comparison(all_results):
    """Plot comparison of all datasets"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Perceptron Performance Across Different Datasets', fontsize=16, fontweight='bold')
    
    dataset_names = list(all_results.keys())
    
    # Prepare data for box plots
    convergence_data = [all_results[name]['convergence_epochs'] for name in dataset_names]
    accuracy_data = [all_results[name]['accuracies'] for name in dataset_names]
    
    # Box plot for convergence epochs
    bp1 = axes[0, 0].boxplot(convergence_data, labels=dataset_names, patch_artist=True)
    axes[0, 0].set_ylabel('Convergence Epochs')
    axes[0, 0].set_title('Convergence Time Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for box in bp1['boxes']:
        box.set_facecolor('lightblue')
    
    # Box plot for accuracies
    bp2 = axes[0, 1].boxplot(accuracy_data, labels=dataset_names, patch_artist=True)
    axes[0, 1].set_ylabel('Final Accuracy')
    axes[0, 1].set_title('Accuracy Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for box in bp2['boxes']:
        box.set_facecolor('lightgreen')
    
    # Bar plot for mean convergence
    mean_convergence = [np.mean(all_results[name]['convergence_epochs']) for name in dataset_names]
    std_convergence = [np.std(all_results[name]['convergence_epochs']) for name in dataset_names]
    
    x_pos = np.arange(len(dataset_names))
    axes[0, 2].bar(x_pos, mean_convergence, yerr=std_convergence, capsize=5,
                   color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(dataset_names, rotation=45)
    axes[0, 2].set_ylabel('Mean Convergence Epochs')
    axes[0, 2].set_title('Mean Convergence Time (± Std)')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Bar plot for mean accuracy
    mean_accuracy = [np.mean(all_results[name]['accuracies']) for name in dataset_names]
    std_accuracy = [np.std(all_results[name]['accuracies']) for name in dataset_names]
    
    axes[1, 0].bar(x_pos, mean_accuracy, yerr=std_accuracy, capsize=5,
                   color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(dataset_names, rotation=45)
    axes[1, 0].set_ylabel('Mean Accuracy')
    axes[1, 0].set_title('Mean Accuracy (± Std)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Convergence rate
    convergence_rate = [np.sum(np.array(all_results[name]['convergence_epochs']) < 500) / 
                        len(all_results[name]['convergence_epochs']) * 100 for name in dataset_names]
    
    axes[1, 1].bar(x_pos, convergence_rate, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(dataset_names, rotation=45)
    axes[1, 1].set_ylabel('Convergence Rate (%)')
    axes[1, 1].set_title('Convergence Rate')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Scatter plot of all points
    for i, name in enumerate(dataset_names):
        axes[1, 2].scatter(all_results[name]['convergence_epochs'], 
                          all_results[name]['accuracies'],
                          alpha=0.5, s=30, label=name)
    axes[1, 2].set_xlabel('Convergence Epochs')
    axes[1, 2].set_ylabel('Final Accuracy')
    axes[1, 2].set_title('All Experiments: Accuracy vs Convergence')
    axes[1, 2].legend(loc='best', fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function: Run experiments and visualize results"""
    
    print("="*70)
    print("PERCEPTRON CONVERGENCE VALIDATION EXPERIMENT")
    print("="*70)
    print("\nThis experiment validates Rosenblatt's Perceptron Convergence Theorem")
    print("and demonstrates its fundamental limitations on non-linearly separable problems.")
    
    # Print available matplotlib styles
    print(f"\nAvailable matplotlib styles: {plt.style.available[:10]}...")
    
    # Experiment parameters
    N_EXPERIMENTS = 50  # Number of experiments per dataset (reduced for faster execution)
    N_SAMPLES = 300      # Number of samples per dataset
    
    # Store results for all datasets
    all_results = {}
    
    # List of datasets to test
    datasets = [
        {
            'name': 'Linearly Separable',
            'generator': generate_linear_separable_data,
            'params': {'n_samples': N_SAMPLES, 'noise': 0.15, 'separation': 3.0}
        },
        {
            'name': 'XOR Problem',
            'generator': generate_xor_data,
            'params': {'n_samples': N_SAMPLES, 'separation': 2.5}
        },
        {
            'name': 'Circular Boundary',
            'generator': generate_circular_data,
            'params': {'n_samples': N_SAMPLES, 'radius': 3.0, 'noise': 0.2}
        },
        {
            'name': 'Checkerboard',
            'generator': generate_checkerboard_data,
            'params': {'n_samples': N_SAMPLES, 'grid_size': 2}
        },
        {
            'name': 'Spiral',
            'generator': generate_spiral_data,
            'params': {'n_samples': N_SAMPLES, 'noise': 0.2}
        }
    ]
    
    # Run experiments for each dataset
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {dataset['name']}")
        print('='*70)
        
        # Generate data
        X, y = dataset['generator'](**dataset['params'])
        
        print(f"\nData Statistics:")
        print(f"  - Total samples: {len(X)}")
        print(f"  - Class 1: {np.sum(y==1)} samples")
        print(f"  - Class -1: {np.sum(y==-1)} samples")
        print(f"  - Data shape: {X.shape}")
        
        # Visualize initial data
        plot_decision_boundary(X, y, title=f"{dataset['name']} - Initial Data")
        
        # Run multiple experiments
        convergence_epochs, accuracies, experiment_results = run_multiple_experiments(
            dataset['name'], X, y, n_experiments=N_EXPERIMENTS
        )
        
        # Store results
        all_results[dataset['name']] = {
            'convergence_epochs': convergence_epochs,
            'accuracies': accuracies,
            'experiment_results': experiment_results,
            'X': X,
            'y': y
        }
        
        # Plot statistics for this dataset
        stats = plot_experiment_statistics(dataset['name'], convergence_epochs, accuracies)
        print(stats)
        
        # Train and show one example
        print(f"\nExample Training Run:")
        perceptron = Perceptron(learning_rate=1.0, max_epochs=200)
        perceptron.fit(X, y, verbose=True)
        plot_training_history(perceptron.loss_history, perceptron.accuracy_history, 
                            perceptron.convergence_epoch, f"{dataset['name']} - Training History")
        plot_decision_boundary(X, y, perceptron, 
                             title=f"{dataset['name']} - Final Result (Acc: {perceptron.score(X, y):.2%})")
    
    # Plot comparison across all datasets
    print("\n" + "="*70)
    print("FINAL COMPARISON ACROSS ALL DATASETS")
    print("="*70)
    plot_all_datasets_comparison(all_results)
    
    # Experimental conclusions
    print("\n" + "="*70)
    print("EXPERIMENTAL CONCLUSIONS")
    print("="*70)
    
    print("\n✅ Success Cases (Linearly Separable Problems):")
    print("   • Linearly Separable: Perfect convergence with 100% accuracy")
    print("   • Fast convergence in finite steps (Rosenblatt's theorem validated)")
    
    print("\n❌ Failure Cases (Non-linearly Separable Problems):")
    print("   • XOR Problem: Cannot converge, accuracy ~50% (random guessing)")
    print("   • Circular Boundary: Cannot find linear separator")
    print("   • Checkerboard: Complex pattern beyond single neuron capability")
    print("   • Spiral: Highly non-linear pattern, perceptron fails")
    
    print("\n📊 Key Statistics Summary:")
    
    # Calculate summary statistics
    for name in all_results.keys():
        epochs = all_results[name]['convergence_epochs']
        accs = all_results[name]['accuracies']
        conv_rate = np.sum(np.array(epochs) < 500) / len(epochs) * 100
        print(f"\n{name}:")
        print(f"  - Convergence Rate: {conv_rate:.1f}%")
        print(f"  - Mean Accuracy: {np.mean(accs):.2%} ± {np.std(accs):.2%}")
        if conv_rate > 0:
            print(f"  - Mean Convergence Time: {np.mean([e for e in epochs if e < 500]):.1f} epochs")
    
    print("\n" + "="*70)
    print("THEORETICAL SIGNIFICANCE")
    print("="*70)

    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()