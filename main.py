from micrograd.engine import Value
import itertools
import random 
import math

class PolynomialSolver:
    def __init__(self, coeffs):
        # 1. Initialize roots across the expected range (e.g., 0 to 6)
        # We add a tiny bit of separation so they don't start identical
        n = len(coeffs) - 1
        self.roots = [Value(random.uniform(0.5, 5.5)) for _ in range(n)]
        self.coeffs = coeffs
        # Track velocity for momentum
        self.velocities = [0.0] * n

    @property
    def n(self):
        return len(self.roots)

    # the Vieta's formula method allows us to one-shot the coefficients, but it is O(n choose k summed over all k) = O(2^n) time complexity,
    # which is fine for small n, but we can also do it in O(n^2) time with a more efficient method.
    def forward(self):
        coeffs = []
        
        # k represents the number of roots multiplied together
        for k in range(self.n + 1):
            # 1. Get all combinations of choosing k roots
            combos = itertools.combinations(self.roots, k)
            # 2. Sum the products of these combinations
            total_sum = sum(math.prod(c) for c in combos)
            # 3. Vieta's formula dictates the sign is (-1)^k
            sign = 1 if k % 2 == 0 else -1
            
            coeffs.append(sign * total_sum)
            
        return coeffs
    
    def forward_efficient(self):
        coeffs = [Value(1.0)]
        for r in self.roots:
            new_coeffs = []
            new_coeffs.append(coeffs[0]) 
            for i in range(1, len(coeffs)):
                new_coeffs.append(coeffs[i] - (r * coeffs[i-1]))
            new_coeffs.append(-r * coeffs[-1])
            coeffs = new_coeffs
        return coeffs

    def optimise(self, n_epochs=1000, lr=0.001, momentum=0.9, en_history=False):
        current_lr = lr
        self.history = []
        
        for epoch in range(1, n_epochs + 1):

            if epoch == 1000:
                current_lr = lr * 0.1
            if epoch == 1800:
                current_lr = lr * 0.01
                momentum = 0.5

            for r in self.roots:
                r.grad = 0.0
                
            # use efficient version
            pred_coeffs = self.forward_efficient()

            mse = sum((a - b) * (a - b) for a, b in zip(self.coeffs, pred_coeffs)) / len(self.coeffs)
            mse.backward()

            max_norm = 5.0
            total_norm = math.sqrt(sum(r.grad ** 2 for r in self.roots)) + 1e-6
            if total_norm > max_norm:
                clip_coef = max_norm / total_norm
                for r in self.roots:
                    r.grad *= clip_coef

            for i in range(self.n):
                self.velocities[i] = momentum * self.velocities[i] + current_lr * self.roots[i].grad
                self.roots[i].data -= self.velocities[i]
            
            if en_history and epoch % 10 == 0:
                self.history.append(sorted(r.data for r in self.roots))
    
        return self.roots

class DurandKernerSolver:
    def __init__(self, coeffs):
        # coeffs should be listed from highest degree to lowest, e.g., [1, -15, 85, ...]
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1

    def evaluate_polynomial(self, x):
        result = 0
        for coeff in self.coeffs:
            result = result * x + coeff
        return result

    def solve(self, max_iterations=100, tolerance=1e-10):
        roots = []
        for i in range(self.degree):
            seed = complex(0.4, 0.9) ** i
            roots.append(seed)

        for iteration in range(max_iterations):
            next_roots = list(roots)
            max_change = 0

            for i in range(self.degree):
                p_val = self.evaluate_polynomial(roots[i])

                denominator = 1.0
                for j in range(self.degree):
                    if i != j:
                        denominator *= (roots[i] - roots[j])

                adjustment = p_val / denominator
                next_roots[i] = roots[i] - adjustment

                max_change = max(max_change, abs(adjustment))

            roots = next_roots

            if max_change < tolerance:
                break

        roots_processed = [round(r.real, 6) if abs(r.imag) < 1e-6 else r for r in roots]
        return sorted(roots_processed, key=lambda x: (x.real, x.imag) if isinstance(x, complex) else (x, 0))


coeffs = [1, -15, 85, -225, 274, -120]

# Find a polynomial with complex roots:
# (x - (1 + 2j))(x - (1 - 2j)) is a good pick - DOTS means the coefficients will be real
# x^2 - 2x + 5
# coeffs = [1, -2, 5]

ps = PolynomialSolver(coeffs)
ps.optimise(2000, lr=0.001, momentum=0.9, en_history=True)

# save
with open("roots_history.txt", "w") as f:
    for epoch_roots in ps.history:
        f.write(",".join(f"{r:.6f}" for r in epoch_roots) + "\n")

ps_roots = sorted(r.data for r in ps.roots)

# Roots found by Durand-Kerner method:
dk = DurandKernerSolver(coeffs)
dk_roots = dk.solve()

print("Roots found by PolynomialSolver:", ps_roots)
print("Roots found by Durand-Kerner method:", dk_roots)