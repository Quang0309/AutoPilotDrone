# Trained SAC models

The selected checkpoint for each curriculum level is included:

| Level | Transitions | Evaluation | Success | Collision |
|---|---:|---:|---:|---:|
| Easy | 40,000 | 100 episodes | 100% | 0% |
| Medium | 300,000 | 100 episodes | 93% | 5% |
| Hard | 150,000 | 100 episodes | 94% | 6% |

Model hashes:

- Easy: `a3079f9d91f196816403fd7af714d10d7d0b1435b3c39936815a7bc0cec8fb8a`
- Medium: `de76aa4ea67812e4c0357d90aa3022a94df226fe9ef14ccb7ba562ddc604f6ed`
- Hard: `f8fcf7531ed92adcdd845c1670fa9bafd1cb844013c898b66b9e99e5ac3ac097`

Example:

```bash
python sac_submission/evaluate_sac_models.py --difficulty hard --episodes 5
```

Add `--gui` to display the PyBullet simulation.
