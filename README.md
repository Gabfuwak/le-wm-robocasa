# LeWM × RoboCasa

Fork of [LeWorldModel](https://github.com/lucas-maes/le-wm) adapted to train on RoboCasa pick-and-place demonstrations. Adds proprio fusion to the JEPA encoder and a data converter from LeRobot format to LeWM's HDF5 schema.

## Setup

**Requirements**: Python 3.10, `uv`, `swig` (`sudo apt-get install -y swig`)

```bash
git clone --recurse-submodules <this-repo>
cd le-wm-robocasa

uv venv --python=3.10
source .venv/bin/activate

uv pip install "stable-worldmodel[train,env]"
uv pip install "gymnasium>=1.0.0"
pip install -e deps/robosuite
pip install -e deps/robocasa
```

## Data

Install data using:

```bash
python download_data.py
```

And follow the instructions the script outputs to convert the dataset from lerobot format to HDF5.

```bash
python scripts/robocasa_to_lewm.py --dataset-path [path_to dataset]
```

The converted HDF5 dataset (`robocasa_pickplace_train.h5`) should be placed under `$STABLEWM_HOME`:

```bash
export STABLEWM_HOME=/path/to/your/data
```

## Training

Set your W&B entity/project in `config/train/lewm.yaml` before running:

```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

**Smoke run (to verify pipeline):**
```bash
python train.py data=robocasa_pickplace img_size=96 patch_size=8 \
  loader.batch_size=64 trainer.max_epochs=2 wandb.enabled=False
```

**Full training run (at least L4 recommended, leWM paper settings):**
```bash
python train.py data=robocasa_pickplace trainer.max_epochs=100
```

Training resumes automatically from the last checkpoint if interrupted. Checkpoints are saved to `$STABLEWM_HOME`.
