very simple kernelbot api for efficient use of kernel competition data downloaded locally from HF in training llms

Installation:


```bash
pip install git+https://github.com/AnshKetchum/hypercorn.git
```

microdocs: 

Quickstart code. To test, go to https://huggingface.co/datasets/GPUMODE/kernelbot-data, clone the repo.

```bash
git clone https://huggingface.co/datasets/GPUMODE/kernelbot-data
cd kernelbot-data
git lfs pull
```

Then, find the `submissions.parquet` file and point it to the `CompetitionDataset` object.

```python
api = CompetitionDataset('/path/to/submissions.parquet')

submissions = api.sample_submissions(batch_size = 10)

for sub in submissions:
    print(sub)
```