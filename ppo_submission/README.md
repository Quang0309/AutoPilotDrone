# PPO training and results

- `PPO_Results.ipynb`: saved stage results, learning curves and independent validation. **Run All** displays the records; training is disabled by default.
- `train_ppo.py`: Easy → Medium → Hard curriculum runner.
- `runtime.py`: submission-local PID reset and depth/state extractor.
- `models/`: one validated PPO checkpoint per stage.
- `ppo_results/`: evaluation, validation and progress records needed by the notebook.

From the repository root:

```bash
pip install -e .
pip install -r ppo_submission/requirements.txt
python ppo_submission/evaluate_ppo_models.py --difficulty hard --episodes 5 --seed 1002000
```

Add `--gui` to display the simulation. For fresh training:

```bash
python ppo_submission/train_ppo.py
```

New training writes to `ppo_submission/outputs/ppo_curriculum/`. Add `--resume` to continue that run. The notebook also has an optional training cell.

## Recorded run: 23 October 2026

| Stage | Training steps | Final screening return | Independent validation |
| --- | ---: | ---: | --- |
| Easy | 500,000 | +51.87 | 100/100, zero collisions |
| Medium | 560,000 | +42.73 | 100/100, zero collisions |
| Hard | 311,296 | +51.27 | 100/100, zero collisions |
