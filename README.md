# mazedyn

Install from scratch (on linux)

```
pip install conda-lock
make conda-linux-64.lock
conda create --name mazedyn-b --file conda-linux-64.lock
conda activate mazedyn-b
make poetry.lock
```