# SAC training and results

Main files:

- `SAC_Training.ipynb`: SAC setup and Easy, Medium and Hard training code.
- `SAC_Results.ipynb`: saved results, training curves and final policy video.
- `reward_modify/runtime.py`: environment wrappers and SAC feature extractor.
- `evaluate_sac_models.py`: evaluates the saved models without training.
- `models/`: best SAC models for Easy, Medium and Hard.

From the repository folder, install the project and notebook packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r sac_submission/requirements.txt
```

To test the supplied model:

```bash
python sac_submission/evaluate_sac_models.py --difficulty hard --episodes 5
```

To show the simulation window:

```bash
python sac_submission/evaluate_sac_models.py --difficulty hard --episodes 5 --gui
```

Training is disabled by default in `SAC_Training.ipynb`. Set `TRAIN_FULL = True` only when a full new training run is needed.
