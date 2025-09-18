# https://huggingface.co/datasets/randall-lab/revisitop
import datasets
from aiohttp import ClientTimeout

dataset_name = "randall-lab/revisitop"
timeout_period = 500000  # very long timeout to prevent timeouts
storage_options = {"client_kwargs": {"timeout": ClientTimeout(total=timeout_period)}}

# These are the config names defined in the script
dataset_configs = ["roxford5k", "rparis6k", "oxfordparis"]  # "revisitop1m" is large and may take a long time to load

# Load query split for evaluation
for i, config_name in enumerate(dataset_configs, start=1):
    # Load query images
    query_dataset = datasets.load_dataset(
        path=dataset_name,
        name=config_name,
        split="qimlist",
        trust_remote_code=True,
        storage_options=storage_options,
    )

    # Load database images
    db_dataset = datasets.load_dataset(
        path=dataset_name,  
        name=config_name,
        split="imlist",
        trust_remote_code=True,
        storage_options=storage_options,
    )


    # Example query image
    query_example = query_dataset[0]
