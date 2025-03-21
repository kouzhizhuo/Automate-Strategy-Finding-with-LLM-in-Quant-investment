# Grail Investment Strategy Framework

A comprehensive framework for investment strategy development using Large Language Models (LLM), multi-agent systems, and deep learning techniques.

## Overview

The Grail framework implements a three-component approach to investment strategy development:

1. **Seed Alphas Factory**: Uses LLM to filter and categorize multimodal documents, constructing initial alpha factors
2. **Multi-agent Decision-making**: Implements a system of agents with distinct risk preferences to evaluate and select alpha factors
3. **Weight Optimization**: Employs deep learning techniques to optimize the weights of selected alpha factors

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/grail.git
cd grail
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
grail/
├── __init__.py
├── seed_alphas_factory.py    # LLM-based document processing
├── multi_agent_system.py     # Multi-agent evaluation system
├── weight_optimization.py    # Deep learning weight optimization
└── main.py                   # Main framework integration
```

## Usage

### Basic Usage

```python
from grail.main import GrailFramework

# Initialize the framework
framework = GrailFramework()

# Run the complete pipeline
results = framework.run_pipeline(
    documents=your_documents,
    alpha_data=your_alpha_data,
    train_labels=your_train_labels,
    val_data=your_val_data,
    val_labels=your_val_labels
)
```

### Component-wise Usage

```python
# Process documents
processed_alphas = framework.process_documents(documents)

# Evaluate alphas
evaluation_results = framework.evaluate_alphas(processed_alphas)

# Optimize weights
optimization_results = framework.optimize_weights(
    alpha_data=your_alpha_data,
    train_labels=your_train_labels
)
```

## Configuration

### Agent Configuration

```python
from grail.multi_agent_system import AgentConfig, RiskPreference

agent_configs = [
    AgentConfig(RiskPreference.CONSERVATIVE, 0.7, 0.3),
    AgentConfig(RiskPreference.MODERATE, 0.6, 0.5),
    AgentConfig(RiskPreference.AGGRESSIVE, 0.5, 0.7)
]
```

### Model Configuration

```python
from grail.weight_optimization import ModelConfig

model_config = ModelConfig(
    input_size=100,
    hidden_size=64,
    output_size=10,
    learning_rate=0.001,
    batch_size=32,
    num_epochs=100
)
```

## Features

- **LLM Integration**: Leverages state-of-the-art language models for document processing
- **Multi-agent System**: Implements diverse risk preferences for robust alpha evaluation
- **Deep Learning**: Uses a two-layer neural network for weight optimization
- **Flexible Configuration**: Customizable parameters for all components
- **Comprehensive Pipeline**: End-to-end processing from document analysis to weight optimization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
