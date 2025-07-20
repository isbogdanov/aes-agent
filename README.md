# Automated Essay Scoring (AES) with Multi-Armed Bandits

This project implements an Automated Essay Scoring (AES) system that uses Large Language Models (LLMs) to grade essays. It features a Multi-Armed Bandit (MAB) controller to dynamically select the most cost-effective grading strategy, balancing accuracy with API costs.

## Key Features

- **LLM-Powered Grading**: Utilizes LLMs to score essays based on standard criteria (e.g., IELTS Task 2).
- **Multi-Armed Bandit (MAB) Optimization**: Employs an Epsilon-Greedy MAB to intelligently choose between different prompting strategies ("arms"), learning over time which one provides the best reward (a balance of accuracy and cost).
- **Multiple Prompting Strategies**: Includes various "recipes" for grading, such as multi-step chain-of-thought with and without examples, and single-step grading.
- **Cost & Performance Analysis**: Logs detailed data on each grading operation, including token usage, latency, and estimated cost, allowing for in-depth analysis of different models and strategies.
- **Extensible LLM Connector**: Integrates with various LLM providers (like Ollama and OpenRouter) through a modular `llm_connector` submodule.

## Grading Criteria

The system grades essays based on the official public assessment criteria for IELTS Writing Task 2. The four criteria evaluated are:

-   Task Response
-   Coherence and Cohesion
-   Lexical Resource
-   Grammatical Range and Accuracy

These criteria are sourced from the following official IELTS resources:

-   [Understanding IELTS Scoring](https://ielts.org/organisations/ielts-for-organisations/understanding-ielts-scoring)
-   [IELTS Writing Key Assessment Criteria (PDF)](https://s3.eu-west-2.amazonaws.com/ielts-web-static/production/Guides/ielts-writing-key-assessment-criteria.pdf)

## Dataset

The project uses the `ielts_writing_dataset_task2.csv` file, which contains essays from the IELTS Writing Task 2. This data was sourced from the [IELTS Writing Scored Essays Dataset on Kaggle](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset/). For this project, only the Task 2 essays portion of the original dataset was used.

## Core Components

- `grader/`: Contains the core logic for the AES agent, including the `EssayGraderAgent` and the different `Recipe` implementations for grading strategies.
- `scripts/`: Holds the main execution scripts:
    - `run_mab.py`: Runs the MAB experiment to find the optimal grading strategy over a number of steps.
    - `run_grid_search.py`: Exhaustively tests all available grading strategies on a set of essays to establish baseline performance.
    - `analyze_log.py`: Generates comparative analysis reports and plots from MAB and grid search log files.
- `settings.py`: Centralized configuration for file paths.
- `datasets/`: Stores the essay dataset used for training and evaluation.

## How It Works

1.  **Grid Search (Baseline)**: The `run_grid_search.py` script runs all available grading recipes on a number of essays. This provides a baseline understanding of each recipe's performance (accuracy, cost, latency) in a controlled manner.
2.  **Multi-Armed Bandit (Optimization)**: The `run_mab.py` script uses the MAB controller to explore and exploit the different grading recipes. For each essay, it chooses an arm (a recipe), grades the essay, and calculates a "reward" based on the accuracy of the score and the cost incurred. Over time, it learns to favor the recipe that yields the highest average reward.
3.  **Analysis**: The `analyze_log.py` script takes the log files from the MAB and grid search runs and produces detailed tables and plots comparing the performance and efficiency of the different strategies.

## Usage

### Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Configure the LLM connector by copying `utils/llm_connector/connector/connector_settings.py.template` to `utils/llm_connector/connector/connector_settings.py` and adding your API keys.

### Running Experiments

**1. Run a Grid Search:**

To evaluate all recipes on the first 100 essays using a specific model and basic prompts:

```bash
python scripts/run_grid_search.py --num-essays 100 --workers 10 --provider-name openrouter --model-name meta-llama/llama-4-maverick --basic-prompts
```

**2. Run a Multi-Armed Bandit Experiment:**

To run the MAB for 500 steps with 10 workers, a 20% exploration rate, and a small token penalty:

```bash
python scripts/run_mab.py --steps 500 --workers 10 --epsilon 0.2 --token-penalty 0.00001 --provider-name openrouter --model-name meta-llama/llama-4-maverick --basic-prompts
```

**3. Analyze the Results:**

To compare the logs from the MAB and grid search runs:

```bash
python scripts/analyze_log.py logs/mab_log_*.csv logs/grid_log_*.csv
```

This will generate a `_plots` directory with PNG images comparing the performance of the different approaches.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details. 