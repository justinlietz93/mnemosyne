# Project Prometheus

Project Prometheus aims to build the Fully Unified Model (FUM) by expanding PrometheusAI's pillars. Core pillars (Mnemosyne, Antenor, Aegis, Kernel) are stable. This README provides an overview and usage examples, with focus on the newly implemented Delphi Oracle pillar.

## Overview

- **Mnemosyne**: Memory system.
- **Antenor**: Web tools.
- **Aegis**: Safety layer.
- **Kernel**: Central reasoning agent.
- **Delphi Oracle**: Predictive modeling and simulation engine.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure Ollama is installed and models are pulled (e.g., gpt-oss:120b, gemma3:4b, nomic-embed-text).

## Usage

### Running the Kernel

```bash
python kernel.py
```

Enter queries in the interactive loop.

### Delphi Oracle Usage Examples

Delphi is integrated into the Kernel. For standalone usage:

```python
from src.delphi_oracle.delphi import Delphi
from mnemosyne_core import Mnemosyne
from aegis_layer import Aegis

mnemosyne = Mnemosyne(db_path="./mvm_db", collection_name="mnemosyne_core", model="nmxbai-embed-large")
aegis = Aegis()
delphi = Delphi(mnemosyne, aegis)

# Example: Consequence Analysis
result = delphi.predict({
    'type': 'consequence_analysis',
    'data': 'Sample action'
})
print(result)

# Example: Forecasting
result = delphi.predict({
    'type': 'forecasting',
    'data': [1, 2, 3, 4, 5]
})
print(result)

# Example: Hypothesis Testing
result = delphi.predict({
    'type': 'hypothesis_testing',
    'hypothesis': 'Sample hypothesis',
    'data': [1, 2, 3]
})
print(result)
```

## Key Components of Delphi

- **Prediction Core**: Handles consequence analysis, forecasting, hypothesis testing.
- **Simulation Engine**: Supports parallel simulations for scalability.
- **Data Integrator**: Merges data from Mnemosyne, Antenor, synthetic sources.
- **Risk Assessor**: Validates predictions using Aegis.

For more details, see the architecture plan in the task description.
