import os
from random import shuffle
import numpy as np
import torch
from pathlib import Path, PurePosixPath, PurePath
from copy import deepcopy

from miditok import TSD, TokSequence, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset
from tqdm import tqdm

hf_token ="TOKEN"

# # Recreates it from the configuration saved on the hub

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 16,
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "num_tempos": 16,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

midi_paths = Path("./maestro_dataset")

# augment first and then tokenize
tokenizer = TSD(config)
augment_dataset(
    midi_paths,
    pitch_offsets= list(range(-12,13)),
    velocity_offsets=[-4, 4],
    duration_offsets=[-0.5, 0.5],
)
p = os.path.abspath("./maestro_dataset")
midi_paths = list(Path(p).glob("**/*.mid*"))
tokenizer.train(vocab_size=30000, files_paths=midi_paths)
tokenizer.save_params(Path("tokenizer_augment/tokenizer.json"))
tokenizer.push_to_hub("jwb23/music-lm", private=True, token=hf_token)


# Now split files
p = os.path.abspath("./maestro_dataset")
midi_paths = list(Path(p).glob("**/*.mid*"))
total_num_files = len(midi_paths)
num_files_valid = round(total_num_files * 0.15)
# num_files_test = round(total_num_files * 0.15)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
# midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
midi_paths_train = midi_paths[num_files_valid:]


for files_paths, subset_name in (
    (midi_paths_train, "train"), (midi_paths_valid, "valid")
):

    # Split the MIDIs into chunks of sizes approximately about 1024 tokens
    subset_chunks_dir = Path(f"maestro_dataset_{subset_name}")
    split_files_for_training(
        files_paths=files_paths,
        tokenizer=tokenizer,
        save_dir=subset_chunks_dir,
        max_seq_len=1024,
        num_overlap_bars=2,
    )


print("gathering training data...")

train_paths = list(Path("maestro_dataset_train").glob("**/*.mid*"))
train_data = []
for path in tqdm(train_paths):
   ids = tokenizer.encode(path)[0].ids
   train_data += ids


print("gathering validation data...")
val_paths = list(Path("maestro_dataset_valid").glob("**/*.mid*"))
val_data = []
for path in tqdm(val_paths):
   ids = tokenizer.encode(path)[0].ids
   val_data += ids

# export to bin files
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print(f"train.bin has {len(train_ids)} tokens")
print(f"val.bin has {len(val_ids)} tokens")
# train.bin has 301,966 tokens
# val.bin has 36,059 tokens