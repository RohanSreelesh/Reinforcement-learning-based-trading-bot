# Reinforced Learning Project

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![python](https://img.shields.io/badge/python-3.11-3776AB?logo=python)](https://www.python.org/downloads/release/python-3110/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Discord](https://img.shields.io/badge/discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/KY7HGvfw)

This project explores various reinforced learning techniques on stock trading with the help of Gymnasium framework.

## Getting Started

### Prerequisites

- Python 3.11
- Poetry (for dependency management)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/RohanSreelesh/4900_RL.git
   cd 4900_RL
   ```

2. **Install Poetry**

   If you haven't installed Poetry yet, you can do so by following the instructions [here](https://python-poetry.org/docs/#installation).

3. **Install Dependencies via Poetry**

   ```bash
   poetry install
   ```

### Algorithms

#### Random Action

```bash
poetry run basic
```

#### Q-Learning with Linear Function Approximation

```bash
poetry run q_with_function_approximation
```

#### Dynamic Q-Learning with Linear Function Approximation

```bash
poetry run dyna_q
```

#### SARSA

```bash
poetry run sarsa
```

#### Proximal Policy Optimization

```bash
poetry run ppo
```

#### Deep Q Networks
Due to configuration issues and RAM + GPU needed to train neural networks, this algorithm only runs on Google Collab.
 - Import the deep_q.ipynb into collab and run each code block.
 - The Training() Function generates a model which can then be imported or stored for future use.
 - The demo() function imports the model and runs a single episode on the NASDAQ_TEST dataset to verify performance

## Features

- Reinforced learning techniques.
- Integration with Gymnasium for creating custom trading scenarios.
- Visualization of trading strategies and account balance using matplotlib.
- Ability to simulate and visualize buy, sell, and hold actions over time.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
