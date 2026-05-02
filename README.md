# README

## Usage

1. Unzip the `dataset.zip` and `crates.zip` files in the `data` directory.

2. Setup the following environment variables:

```bash
export OPENAI_API_KEY=sk-proj-1234567890
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=minimax-m2.7
```

3. Run the agent with the following command:

```bash
python -m src.agent.main --provider openai
```

For more usage, please run `python -m src.agent.main --help`.
