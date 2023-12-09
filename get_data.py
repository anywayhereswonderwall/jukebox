import pickle
import glob
import pathlib
from encode import midi_parser
import torch
'''
Handles all operations related to getting data.
'''

DATA_DIR = pathlib.Path("data/maestro-v3.0.0")
NUM_FILES = None  # number of files to parse, None gets all files
PICKLE_DIR = pathlib.Path("pickles/seq.pkl")
PRINT_INTERVAL = 50


def pickle_seq(seq, path=PICKLE_DIR):
    """
    Params:
    seq: a sequence of MIDI encoded events (list of ints)
    path: path to pickle file
    Returns:
    None
    """
    '''
    Pickles a sequence of MIDI events. Since parsing MIDI files is slow, we
    pickle the parsed MIDI files so that we can load them quickly later.
    '''
    with open(path, 'wb') as f:
        pickle.dump(seq, f)


def unpickle_seq(path=PICKLE_DIR):
    """
    Unpickles a sequence of MIDI events.
    """
    with open(path, 'rb') as f:
        seq = pickle.load(f)
    return seq


def midis_to_list(data_dir=DATA_DIR, num_files=NUM_FILES):
    """

    """
    filenames = glob.glob(str(data_dir / "**/*.mid*"))
    seqs = []  # long list of events of all performances included in filenames

    for i, filename in enumerate(filenames[:num_files]):
        seq, _ = midi_parser(filename)
        seqs.extend(seq)
        if i % PRINT_INTERVAL == 0:
            print(f"Finished parsing {i} files")
    return seqs


def get_batch(data, batch_size, block_size):
    """
    Params:
    seq: a sequence of MIDI encoded events (list of ints)
    batch_size: number of sequences in batch
    block_size: length of each sequence in batch
    Returns:
    batch: a batch of sequences of MIDI encoded events (list of ints)
    """
    '''
    Gets a batch of sequences from a sequence of MIDI events.
    '''
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x, y