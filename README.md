# Task for text-classification
USAGE:
fast-record classifier [OPTIONS] --path <PATH>

OPTIONS:

    -h, --help
            Print help information

        --max-vocab-size <MAX_VOCAB_SIZE>
            max vocabulary size for build record, only effective when the with-vocab is not set
            [default: 10000]

    -o, --output-path <OUTPUT_PATH>
            output path of record [aliases: output]

    -p, --path <PATH>
            path of classifier dataset [aliases: input]

        --padding <PADDING>
            padding special token of vocabulary [default: <PAD>] [aliases: pad-token]

    -s, --separator <SEPARATOR>
            separator between sentence and label [default: "\t"] [aliases: delimiter]

        --sequence-length <SEQUENCE_LENGTH>
            max sequence length for sentence [default: 32]

        --stopwords-file <STOPWORDS_FILE>
            stopwords file for build vocabulary, only effective when the with-vocab is not set
            [aliases: stopwords]

        --unknown <UNKNOWN>
            unknown special token of vocabulary [default: <UNK>] [aliases: unk-token]

    -V, --version
            Print version information

        --with-label-id
            origin sample data use label_id instead of label

        --with-vocab
            with user vocabulary for classifier dataset

```python
"""read classification records"""
import os
import pyarrow as pa
import pandas as pd


def read_ipc_file(path) -> pd.DataFrame:
    with pa.OSFile(path, 'rb') as sink:
        with pa.ipc.open_file(sink) as reader:
            num_record_batches = reader.num_record_batches
            batches = []
            for i in range(num_record_batches):
                batch = reader.get_batch(i)
                batches.append(batch.to_pandas())
    return pd.concat(batches)


base_path = "<example dataset path>"
train_file = os.path.join(base_path, 'train.records.ipc')
train_df = read_ipc_file(train_file)
dev_file = os.path.join(base_path, 'dev.records.ipc')
dev_df = read_ipc_file(dev_file)
test_file = os.path.join(base_path, 'test.records.ipc')
test_df = read_ipc_file(test_file)
```