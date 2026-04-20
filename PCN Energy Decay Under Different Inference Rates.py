# -*- coding: utf-8 -*-
"""
Simulation 2: PCN Inference Dynamics Validation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)


plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'figure.titlesize': 28
})


plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class PredictiveCodingNetwork:

    
    def __init__(self, layer_dims: List[int], activation: str = 'tanh'):
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims)
        
        # 选择激活函数及其导数
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x)**2
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: self.activation(x) * (1 - self.activation(x))
        else:
            raise ValueError(f"Unknown activation: {activation}")
        

        self.weights = []
        for i in range(self.n_layers - 1):
            scale = np.sqrt(2.0 / (self.layer_dims[i] + self.layer_dims[i+1]))
            w = np.random.randn(self.layer_dims[i], self.layer_dims[i+1]) * scale
            self.weights.append(w)
        

        self.inference_history = {'energies': []}
        
    def forward_prediction(self, activities: List[np.ndarray]) -> List[np.ndarray]:

        predictions = []
        for l in range(1, self.n_layers):
            mu = self.activation(np.dot(activities[l-1], self.weights[l-1]))
            predictions.append(mu)
        return predictions
    
    def compute_errors(self, activities: List[np.ndarray], 
                       predictions: List[np.ndarray]) -> List[np.ndarray]:
 
        errors = []
        for l in range(1, self.n_layers):
            eps = activities[l] - predictions[l-1]
            errors.append(eps)
        return errors
    
    def compute_energy(self, errors: List[np.ndarray]) -> float:

        return 0.5 * sum(np.sum(eps ** 2) for eps in errors)
    
    def learning_step(self, activities: List[np.ndarray], learning_rate: float):

        predictions = self.forward_prediction(activities)
        errors = self.compute_errors(activities, predictions)
        
        for l in range(self.n_layers - 1):
            a_l = activities[l]
            eps_l1 = errors[l]
            z = np.dot(a_l, self.weights[l])
            f_prime = self.activation_derivative(z)
            grad = np.dot(a_l.T, eps_l1 * f_prime)
            self.weights[l] -= learning_rate * grad / a_l.shape[0]
    
    def inference_step(self, activities: List[np.ndarray], 
                       inference_rate: float) -> List[np.ndarray]:
  
        new_activities = [activities[0].copy()]
        
        predictions = self.forward_prediction(activities)
        errors = self.compute_errors(activities, predictions)
        
        for l in range(1, self.n_layers - 1):
            eps_l = errors[l-1]
            grad = eps_l.copy()
            
            if l < self.n_layers - 1 and l < len(errors):
                eps_l1 = errors[l]
                W_l = self.weights[l]
                a_l = activities[l]
                z = np.dot(a_l, W_l)
                f_prime = self.activation_derivative(z)
                weighted_error = eps_l1 * f_prime
                back_term = np.dot(weighted_error, W_l.T)
                grad = grad - back_term
            
            delta_a = -inference_rate * grad
            new_a = activities[l] + delta_a
            new_activities.append(new_a)
        
        if len(activities) > len(new_activities):
            new_activities.append(activities[-1].copy())
        
        return new_activities
    
    def inference_to_convergence(self, input_data: np.ndarray, target_data: Optional[np.ndarray] = None,
                                  inference_rate: float = 0.1, max_iters: int = 100,
                                  tol: float = 1e-6, track_history: bool = True) -> Tuple[List[np.ndarray], Dict]:

        activities = [input_data.copy()]
        
        for l in range(1, self.n_layers):
            if l < self.n_layers - 1 or target_data is None:
                mu = self.activation(np.dot(activities[l-1], self.weights[l-1]))
                activities.append(mu.copy())
            else:
                activities.append(target_data.copy())
        
        if track_history:
            self.inference_history = {'energies': []}
        
        for iteration in range(max_iters):
            if track_history:
                predictions = self.forward_prediction(activities)
                errors = self.compute_errors(activities, predictions)
                energy = self.compute_energy(errors)
                self.inference_history['energies'].append(energy)
            
            new_activities = self.inference_step(activities, inference_rate)
            
            diffs = []
            for l in range(1, self.n_layers - 1):
                diff = np.linalg.norm(new_activities[l] - activities[l])
                diffs.append(diff)
            
            max_diff = max(diffs) if diffs else 0
            activities = new_activities
            
            if max_diff < tol:
                break
        
        info = {
            'n_iterations': iteration + 1,
            'final_energy': self.inference_history['energies'][-1] if track_history and self.inference_history['energies'] else None,
            'converged': max_diff < tol
        }
        
        return activities, info


def generate_mnist_like_data(n_samples: int = 500, n_features: int = 784, n_classes: int = 10):

    X = np.random.randn(n_samples, n_features)
    X = X / (np.sqrt(np.sum(X**2, axis=1, keepdims=True)) + 1e-8)
    y = np.random.randint(0, n_classes, size=n_samples)
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1
    return X, y_onehot


def train_test_split(X, y, test_size: float = 0.2):

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_pcn(model: PredictiveCodingNetwork, X_train: np.ndarray, y_train: np.ndarray,
              inference_rate: float = 0.1, learning_rate: float = 0.005,
              n_epochs: int = 2, batch_size: int = 16):

    n_samples = X_train.shape[0]
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            activities, info = model.inference_to_convergence(
                X_batch, y_batch, inference_rate=inference_rate, max_iters=15, track_history=False
            )
            model.learning_step(activities, learning_rate)
            
            predictions = model.forward_prediction(activities)
            errors = model.compute_errors(activities, predictions)
            energy = model.compute_energy(errors)
            epoch_loss += energy
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

def test_inference_convergence(model: PredictiveCodingNetwork, 
                                X_test: np.ndarray, y_test: np.ndarray,
                                inference_rates: List[float]):

    results = {
        'inference_rates': inference_rates,
        'convergence_trajectories': [],
        'iterations_to_converge': [],
        'final_energies': [],
        'initial_energies': [],
        'converged': []
    }
    

    X_sample = X_test[:1].copy()
    y_sample = y_test[:1].copy()
    
    for gamma in inference_rates:
        print(f"Testing γ = {gamma}...")
        

        fresh_model = PredictiveCodingNetwork(model.layer_dims, activation='tanh')
        fresh_model.weights = [w.copy() for w in model.weights]  
        
        fresh_model.inference_history = {'energies': []}
        activities, info = fresh_model.inference_to_convergence(
            X_sample, y_sample, inference_rate=gamma, max_iters=100, tol=1e-4, track_history=True
        )
        energies = fresh_model.inference_history['energies']
        

        print(f"  Length of energies: {len(energies)}")
        print(f"  First 5 energies: {energies[:5]}")
        print(f"  Last 5 energies: {energies[-5:]}")
        print(f"  Final energy from info: {info['final_energy']:.6f}")
        
        results['convergence_trajectories'].append(energies)
        results['iterations_to_converge'].append(info['n_iterations'])
        results['final_energies'].append(info['final_energy'])
        results['initial_energies'].append(energies[0] if energies else None)
        results['converged'].append(info['converged'])
        
        print(f"  Initial energy: {energies[0]:.6f}")
        print(f"  Final energy: {info['final_energy']:.6f}")
        print(f"  Converged in {info['n_iterations']} steps")
        print(f"  Energy reduction: {(energies[0] - info['final_energy'])/energies[0]*100:.2f}%\n")
    
    return results

def print_quantitative_results(results: Dict):

    print("\n" + "=" * 80)
    print("QUANTITATIVE RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n{:<12} {:<15} {:<15} {:<15} {:<20}".format(
        "γ", "Initial Energy", "Final Energy", "Steps", "Reduction (%)"))
    print("-" * 80)
    
    for i, gamma in enumerate(results['inference_rates']):
        initial = results['initial_energies'][i]
        final = results['final_energies'][i]
        steps = results['iterations_to_converge'][i]
        reduction = (initial - final) / initial * 100
        
        print("{:<12} {:<15.6f} {:<15.6f} {:<15} {:<20.2f}".format(
            gamma, initial, final, steps, reduction))
    
    print("-" * 80)
    

    print("\nAdditional Statistics:")
    fastest_idx = np.argmin(results['iterations_to_converge'])
    slowest_idx = np.argmax(results['iterations_to_converge'])
    best_energy_idx = np.argmin(results['final_energies'])
    largest_reduction_idx = np.argmax([(results['initial_energies'][i] - results['final_energies'][i])/results['initial_energies'][i]*100 for i in range(len(results['inference_rates']))])
    
    print(f"  Fastest convergence: γ = {results['inference_rates'][fastest_idx]} ({results['iterations_to_converge'][fastest_idx]} steps)")
    print(f"  Slowest convergence: γ = {results['inference_rates'][slowest_idx]} ({results['iterations_to_converge'][slowest_idx]} steps)")
    print(f"  Best final energy: γ = {results['inference_rates'][best_energy_idx]} ({results['final_energies'][best_energy_idx]:.6f})")
    print(f"  Largest energy reduction: γ = {results['inference_rates'][largest_reduction_idx]}")

def plot_energy_decay(results: Dict, max_steps: int = 30, save_path: str = 'pcn_energy_decay.png'):

    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    linewidth = 3
    markersize = 10
    
    for i, gamma in enumerate(results['inference_rates']):
        trajectory = results['convergence_trajectories'][i]
        n_steps = results['iterations_to_converge'][i]
        
        if trajectory:
            steps_to_show = min(max_steps, len(trajectory))
            x_axis = np.arange(steps_to_show)
            

            label = f'γ = {gamma} ({n_steps} steps)'
            if gamma == 0.5 and n_steps >= 100:
                label = f'γ = {gamma} ({n_steps} steps, not converged)'
            
            ax.plot(x_axis, trajectory[:steps_to_show], 
                   marker=markers[i], 
                   markevery=max(1, steps_to_show // 10),
                   color=colors[i],
                   linestyle=linestyles[i],
                   label=label, 
                   linewidth=linewidth, 
                   markersize=markersize)
    
    ax.set_xlim([0, max_steps])
    ax.set_ylim([0.25, 0.70])
    
    ax.set_xlabel('Inference Step', fontsize=26, fontweight='bold')
    ax.set_ylabel('Energy E', fontsize=26, fontweight='bold')
    ax.set_title(f'PCN Energy Decay Under Different Inference Rates',
                 fontsize=28, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    
    ax.tick_params(axis='both', labelsize=22)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    return fig

def main():

    
    print("=" * 60)
    print("PCN Inference Dynamics Validation")
    print("=" * 60)
    print("\nThis experiment validates the convergence properties")
    print("of Predictive Coding Networks as described in van Zwol et al. (2024).\n")
    

    print("1. Generating synthetic data...")
    X, y = generate_mnist_like_data(n_samples=500, n_features=784, n_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    

    print("\n2. Creating 3-layer PCN...")
    layer_dims = [784, 64, 10]
    model = PredictiveCodingNetwork(layer_dims, activation='tanh')
    print(f"   Architecture: {layer_dims}")
    

    print("\n3. Training PCN...")
    train_pcn(model, X_train, y_train, inference_rate=0.1, learning_rate=0.005, n_epochs=2)
    

    print("\n4. Testing convergence for different inference rates...")
    inference_rates = [0.05, 0.1, 0.2, 0.5]
    results = test_inference_convergence(model, X_test, y_test, inference_rates)
    

    print_quantitative_results(results)
    

    print("\n5. Generating energy decay figure (first 30 steps)...")
    fig = plot_energy_decay(results, max_steps=25, save_path='pcn_energy_decay.png')
    

    print("\n" + "=" * 60)
    print("EXPERIMENTAL CONCLUSIONS")
    print("=" * 60)

    
    return results


if __name__ == "__main__":
    results = main()