#!/usr/bin/env python3
"""
Module Generator Script

This script automatically generates all module files for the ML Mastery curriculum.
It creates: README.md, theory.md, examples.py, exercises.py, solutions.py, and problems.md
for all 19 modules.

Usage:
    uv run generate_modules.py
"""

import os
from pathlib import Path

# Module configurations
MODULES = [
    {
        "id": "00",
        "name": "python-fundamentals",
        "title": "Python Fundamentals",
        "description": "NumPy, Pandas, visualization, and vectorization",
        "skip": True  # Already created
    },
    {
        "id": "01",
        "name": "linear-algebra",
        "title": "Linear Algebra for ML",
        "description": "Vectors, matrices, eigenvalues, SVD"
    },
    {
        "id": "02",
        "name": "calculus",
        "title": "Calculus for ML",
        "description": "Derivatives, gradients, chain rule, optimization"
    },
    {
        "id": "03",
        "name": "probability",
        "title": "Probability Theory",
        "description": "Distributions, conditional probability, Bayes theorem"
    },
    {
        "id": "04",
        "name": "statistics",
        "title": "Statistics for ML",
        "description": "Hypothesis testing, confidence intervals, MLE"
    },
    {
        "id": "05",
        "name": "optimization",
        "title": "Optimization Theory",
        "description": "Gradient descent, convex optimization, constraints"
    },
    {
        "id": "06",
        "name": "ml-foundations",
        "title": "ML Foundations",
        "description": "Bias-variance, overfitting, regularization, validation"
    },
    {
        "id": "07",
        "name": "supervised-learning",
        "title": "Supervised Learning",
        "description": "Linear/logistic regression, decision trees, SVMs, ensembles"
    },
    {
        "id": "08",
        "name": "unsupervised-learning",
        "title": "Unsupervised Learning",
        "description": "K-means, PCA, DBSCAN, hierarchical clustering"
    },
    {
        "id": "09",
        "name": "feature-engineering",
        "title": "Feature Engineering",
        "description": "Scaling, encoding, selection, dimensionality reduction"
    },
    {
        "id": "10",
        "name": "model-evaluation",
        "title": "Model Evaluation",
        "description": "Cross-validation, metrics, ROC curves, calibration"
    },
    {
        "id": "11",
        "name": "deep-learning",
        "title": "Deep Learning Fundamentals",
        "description": "Neural networks, backpropagation, activation functions"
    },
    {
        "id": "12",
        "name": "neural-networks-from-scratch",
        "title": "Neural Networks from Scratch",
        "description": "Build complete neural network with NumPy only"
    },
    {
        "id": "13",
        "name": "pytorch-masterclass",
        "title": "PyTorch Masterclass",
        "description": "PyTorch essentials, training loops, GPU utilization"
    },
    {
        "id": "14",
        "name": "nlp",
        "title": "Natural Language Processing",
        "description": "Word embeddings, RNNs, Transformers, BERT"
    },
    {
        "id": "15",  
        "name": "computer-vision",
        "title": "Computer Vision",
        "description": "CNNs, ResNets, object detection, segmentation"
    },
    {
        "id": "16",
        "name": "mlops",
        "title": "MLOps & Production ML",
        "description": "Pipelines, versioning, monitoring, deployment"
    },
    {
        "id": "17",
        "name": "real-world-projects",
        "title": "Real-World ML Projects",
        "description": "5 complete end-to-end projects"
    },
    {
        "id": "18",
        "name": "advanced-topics",
        "title": "Advanced Topics",
        "description": "GANs, RL, AutoML, interpretability"
    }
]


def create_readme(module):
    """Generate README.md for a module."""
    return f"""# Module {module['id']}: {module['title']}

## 🎯 Learning Objectives

By the end of this module, you will master:
- {module['description']}
- Practical implementation in Python
- Real-world applications
- Common pitfalls and best practices

## 📚 Prerequisites

- Completion of all previous modules
- Strong Python and NumPy skills
- Understanding of fundamental ML concepts

## 🗺️ Study Plan

### Day 1-2: Theory & Examples
1. Read `theory.md` thoroughly
2. Run `examples.py` and understand each example
3. Take notes on key concepts

### Day 3-4: Practice
1. Complete all exercises in `exercises.py`
2. Solve Easy problems (1-10) in `problems.md`
3. Compare with `solutions.py`

### Day 5: Advanced
1. Solve Medium problems (1-10)
2. Attempt Hard problems (1-5)
3. Build mini-project

## ✅ Mastery Checklist

- [ ] Understand all theoretical concepts
- [ ] Can explain concepts to others
- [ ] Completed all exercises
- [ ] Solved all Easy problems
- [ ] Solved 7+ Medium problems
- [ ] Solved 3+ Hard problems
- [ ] Built related mini-project

## 🚀 Quick Start

```bash
# View examples
uv run {module['id']}-{module['name']}/examples.py

# Do exercises
uv run {module['id']}-{module['name']}/exercises.py

# Check solutions
uv run {module['id']}-{module['name']}/solutions.py
```

## 📖 Files in This Module

- `README.md` - This file
- `theory.md` - Complete theoretical foundation
- `examples.py` - Working code examples
- `exercises.py` - Hands-on practice
- `solutions.py` - Complete solutions
- `problems.md` - 25 curated problems (10 Easy, 10 Medium, 5 Hard)

## 🎓 Success Tips

1. Don't rush - understanding > speed
2. Code everything yourself
3. Solve problems before checking solutions
4. Build intuition through experimentation
5. Connect concepts to previous modules

**Ready? Let's master {module['title']}!** 🚀
"""


def create_theory(module):
    """Generate theory.md skeleton for a module."""
    return f"""# {module['title']} - Theory

## Introduction

This module covers {module['description']}.

## Core Concepts

### 1. Fundamental Principles

[To be filled with detailed theory]

### 2. Mathematical Foundations

[Include equations, proofs, and intuitions]

### 3. Algorithms and Methods

[Step-by-step algorithm descriptions]

### 4. Implementation Considerations

[Practical tips for coding]

## Key Equations

[Important formulas with explanations]

## Worked Examples

[Step-by-step examples showing theory in action]

## Common Pitfalls

1. **Pitfall 1**: [Description and how to avoid]
2. **Pitfall 2**: [Description and how to avoid]
3. **Pitfall 3**: [Description and how to avoid]

## Real-World Applications

[How this module's concepts apply to real ML problems]

## Connections to Other Topics

- **Previous**: How this builds on earlier modules
- **Next**: How this prepares for future modules
- **Related**: Connections to other ML areas

## Further Reading

- [Resource 1]
- [Resource 2]
- [Resource 3]

## Summary

Key takeaways from this module:
1. [Key point 1]
2. [Key point 2]
3. [Key point 3]

---

**Next**: Practice with `examples.py` 🚀
"""


def create_examples_py(module):
    """Generate examples.py skeleton."""
    return f'''#!/usr/bin/env python3
"""
{module['title']} - Examples

Fully working examples demonstrating:
- {module['description']}
- Practical implementations
- Visual demonstrations

Run: uv run {module['id']}-{module['name']}/examples.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)


def print_section(title):
    """Print formatted section header."""
    print(f"\\n{{'='*60}}")
    print(f"{{title:^60}}")
    print(f"{{'='*60}}\\n")


def example_1():
    """First example demonstrating core concept."""
    print_section("Example 1: [Concept Name]")
    
    # Implementation here
    print("Implementation to be added")
    

def example_2():
    """Second example showing practical use."""
    print_section("Example 2: [Concept Name]")
    
    # Implementation here
    print("Implementation to be added")


def example_3():
    """Third example with visualization."""
    print_section("Example 3: [Concept Name]")
    
    # Implementation here
    print("Implementation to be added")


def main():
    """Run all examples."""
    print("\\n" + "="*60)
    print(f" {{module['title']}} - Examples ".center(60, "="))
    print("="*60)
    
    example_1()
    example_2()
    example_3()
    
    print("\\n" + "="*60)
    print(" Examples Complete! ".center(60, "="))
    print("="*60)
    print("\\nNext: Complete exercises.py\\n")


if __name__ == "__main__":
    main()
'''


def create_exercises_py(module):
    """Generate exercises.py skeleton."""
    return f'''#!/usr/bin/env python3
"""
{module['title']} - Exercises

Fill in the TODOs to complete these exercises.
Test your understanding of {module['description']}.

Run: uv run {module['id']}-{module['name']}/exercises.py
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: [Description]
    
    TODO: Implement this function
    """
    # TODO: Your code here
    pass


def exercise_2():
    """
    Exercise 2: [Description]
    
    TODO: Implement this function
    """
    # TODO: Your code here
    pass


def test_exercises():
    """Test all exercises."""
    print("Testing Exercise 1...")
    # Test code here
    
    print("Testing Exercise 2...")
    # Test code here
    
    print("\\n✅ All tests passed!")


if __name__ == "__main__":
    test_exercises()
'''


def create_solutions_py(module):
    """Generate solutions.py skeleton."""
    return f'''#!/usr/bin/env python3
"""
{module['title']} - Solutions

Complete solutions to all exercises with detailed explanations.

Run: uv run {module['id']}-{module['name']}/solutions.py
"""

import numpy as np


def print_solution(number, title):
    """Print solution header."""
    print(f"\\n{{'-'*60}}")
    print(f"Solution {{number}}: {{title}}")
    print(f"{{'-'*60}}\\n")


def solution_1():
    """
    Solution to Exercise 1
    
    Explanation: [Step-by-step reasoning]
    """
    print_solution(1, "[Exercise Title]")
    
    # Implementation
    print("Implementation with detailed comments")


def main():
    """Run all solutions."""
    print("="*60)
    print(f" {{module['title']}} - Solutions ".center(60))
    print("="*60)
    
    solution_1()
    
    print("\\n" + "="*60)
    print("All solutions demonstrated!")
    print("="*60)


if __name__ == "__main__":
    main()
'''


def create_problems_md(module):
    """Generate problems.md with 25 problems."""
    return f"""# {module['title']} - Problems

Practice problems to build ML engineering intuition.

## Problem Guidelines

- **Easy**: Concept checks, basic calculations, simple coding
- **Medium**: Multi-step reasoning, implementations, applied problems
- **Hard**: Engineering thinking, debugging, optimization, design

---

## ⭐ Easy Problems (10)

### Problem 1: [Title]
**Difficulty**: Easy | **Type**: Conceptual

[Problem description]

**Expected Output**: [What you should get]

---

### Problem 2: [Title]
**Difficulty**: Easy | **Type**: Numerical

[Problem description]

---

### Problem 3: [Title]
**Difficulty**: Easy | **Type**: Coding

[Problem description]

---

### Problem 4: [Title]
**Difficulty**: Easy | **Type**: Conceptual

[Problem description]

---

### Problem 5: [Title]
**Difficulty**: Easy | **Type**: Numerical

[Problem description]

---

### Problem 6: [Title]
**Difficulty**: Easy | **Type**: Coding

[Problem description]

---

### Problem 7: [Title]
**Difficulty**: Easy | **Type**: Conceptual

[Problem description]

---

### Problem 8: [Title]
**Difficulty**: Easy | **Type**: Numerical

[Problem description]

---

### Problem 9: [Title]
**Difficulty**: Easy | **Type**: Coding

[Problem description]

---

### Problem 10: [Title]
**Difficulty**: Easy | **Type**: Conceptual

[Problem description]

---

## ⭐⭐ Medium Problems (10)

### Problem 11: [Title]
**Difficulty**: Medium | **Type**: Implementation

[Detailed problem requiring multi-step solution]

---

### Problem 12: [Title]
**Difficulty**: Medium | **Type**: Applied

[Real-world inspired problem]

---

### Problem 13: [Title]
**Difficulty**: Medium | **Type**: Algorithm

[Implement an algorithm from scratch]

---

### Problem 14: [Title]
**Difficulty**: Medium | **Type**: Analysis

[Analyze behavior, compare methods]

---

### Problem 15: [Title]
**Difficulty**: Medium | **Type**: Visualization

[Create informative plots]

---

### Problem 16: [Title]
**Difficulty**: Medium | **Type**: Implementation

[Build a component]

---

### Problem 17: [Title]
**Difficulty**: Medium | **Type**: Debugging

[Find and fix issues]

---

### Problem 18: [Title]
**Difficulty**: Medium | **Type**: Applied

[Solve practical problem]

---

### Problem 19: [Title]
**Difficulty**: Medium | **Type**: Optimization

[Improve performance]

---

### Problem 20: [Title]
**Difficulty**: Medium | **Type**: Design

[Design a solution approach]

---

## ⭐⭐⭐ Hard Problems (5)

### Problem 21: [Title]
**Difficulty**: Hard | **Type**: Engineering

[Complex real-world scenario requiring ML engineer thinking]

**Hint**: [Nudge in right direction]

**Expected Insights**:
- [What you should discover]
- [Key learning point]

---

### Problem 22: [Title]
**Difficulty**: Hard | **Type**: Deep Dive

[Requires deep understanding and implementation]

**Hint**: [Helpful guidance]

**Expected Insights**:
- [Important realization]
- [Connection to broader concepts]

---

### Problem 23: [Title]
**Difficulty**: Hard | **Type**: Research

[Open-ended problem requiring experimentation]

**Hint**: [Starting point]

**Expected Insights**:
- [Discovery from exploration]

---

### Problem 24: [Title]
**Difficulty**: Hard | **Type**: System Design

[Design and implement complete system]

**Hint**: [Architecture guidance]

**Expected Insights**:
- [Design principles learned]

---

### Problem 25: [Title]
**Difficulty**: Hard | **Type**: Challenge

[Difficult problem testing mastery]

**Hint**: [Subtle hint]

**Expected Insights**:
- [Deep understanding demonstrated]

---

## Solutions

Solutions are available in `solutions.py` with detailed explanations.

**Remember**: Attempt problems before checking solutions!

---

**Completed all 25 problems?** → You've mastered this module! 🎉
"""


def generate_module(module):
    """Generate all files for a module."""
    if module.get('skip'):
        print(f"Skipping {module['id']}-{module['name']} (already exists)")
        return
    
    module_dir = Path(f"/Users/sree/DEV/machine-learning-mastery/{module['id']}-{module['name']}")
    module_dir.mkdir(exist_ok=True)
    
    print(f"Generating {module['id']}-{module['name']}...")
    
    # Create each file
    files = {
        'README.md': create_readme(module),
        'theory.md': create_theory(module),
        'examples.py': create_examples_py(module),
        'exercises.py': create_exercises_py(module),
        'solutions.py': create_solutions_py(module),
        'problems.md': create_problems_md(module),
    }
    
    for filename, content in files.items():
        filepath = module_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
    
    # Make Python files executable
    for pyfile in ['examples.py', 'exercises.py', 'solutions.py']:
        (module_dir / pyfile).chmod(0o755)
    
    print(f"  ✓ Created {len(files)} files")


def main():
    """Generate all modules."""
    print("="*60)
    print(" ML Mastery - Module Generator ".center(60))
    print("="*60)
    print()
    
    for module in MODULES:
        generate_module(module)
    
    print()
    print("="*60)
    print(" Module Generation Complete! ".center(60))
    print("="*60)
    print()
    print(f"Generated {len([m for m in MODULES if not m.get('skip')])} modules")
    print("Each module contains:")
    print("  - README.md")
    print("  - theory.md")
    print("  - examples.py")
    print("  - exercises.py")
    print("  - solutions.py")
    print("  - problems.md (10 Easy + 10 Medium + 5 Hard)")
    print()
    print("Next steps:")
    print("1. Review generated modules")
    print("2. Fill in detailed content for each module")
    print("3. Run: uv sync")
    print("4. Start learning with 00-python-fundamentals/")
    print()


if __name__ == "__main__":
    main()
