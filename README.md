# AGREE

Source code for AGREE.

```text
AGREE_OpenSource/
├── configs/              # dataset hyper-parameters
├── data/                 # dataset placeholder and instructions
├── results/              # generated at runtime
├── scripts/              # training entrypoint
└── src/agree/            # core package
```

Required packages are listed in `requirements.txt`.

**Attributed Graph Data**

```bash
python scripts/train_agree.py --dataset acm --graph_data_root ./data/graph_data --no_log
```

**Mixed-Type Raw Data**

Prepare graph data from mixed-type raw data, then train AGREE:

```bash
python scripts/prepare_mixed_data.py \
  --dataset breast \
  --raw_data_root ./data/mixed_raw \
  --output_root ./data/graph_data

python scripts/train_agree.py \
  --dataset breast \
  --graph_data_root ./data/graph_data \
  --no_log
```
