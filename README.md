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


def read_classifier():
    base_path = "../data/THUCNews"
    for filename in os.listdir(base_path):
        if not filename.endswith("ipc"):
            continue
        file_path = os.path.join(base_path, filename)
        df = read_ipc_file(file_path)
        max_len = int(df.shape[1] - 1)
        s_cols = [f'word_{i}' for i in range(max_len)]
        word_ids = df[s_cols].values
        label_ids = df['class'].values
        print(type(word_ids), word_ids.shape)
        print(type(label_ids), label_ids.shape)
```

# Task for text-similarity
USAGE:
fast-record similarity [OPTIONS] --path <PATH>

OPTIONS:

    -h, --help
            Print help information

        --label-sep <LABEL_SEP>
            separator between text and label [default: "\t"] [aliases: s2]

        --max-vocab-size <MAX_VOCAB_SIZE>
            max vocabulary size for build record, only effective when the with-vocab is not set
            [default: 10000]

    -o, --output-path <OUTPUT_PATH>
            output path of record [aliases: output]

    -p, --path <PATH>
            path of similarity dataset [aliases: input]

        --padding <PADDING>
            padding special token of vocabulary [default: <PAD>] [aliases: pad-token]

        --sent-sep <SENT_SEP>
            separator between text_a and text_b [default: "\t"] [aliases: s1]

        --sequence-length <SEQUENCE_LENGTH>
            max sequence length for sentence [default: 32]

        --stopwords-file <STOPWORDS_FILE>
            stopwords file for build vocabulary, only effective when the with-vocab is not set
            [aliases: stopwords]

        --unknown <UNKNOWN>
            [default: <UNK>] [aliases: unk-token]

    -V, --version
            Print version information

        --with-bool
            similarity with boolean value

        --with-lang-en
            with en language

        --with-vocab
            with user vocabulary for classifier dataset
```python
"""read similarity records"""
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


def read_similarity():
    base_path = "../data/ChineseSNLI"
    for filename in os.listdir(base_path):
        if not filename.endswith("ipc"):
            continue
        file_path = os.path.join(base_path, filename)
        df = read_ipc_file(file_path)
        max_len = (df.shape[1] - 1) // 2
        a_cols = [f'text_a_{i}' for i in range(max_len)]
        text_a_ids = df[a_cols].values
        b_cols = [f'text_b_{i}' for i in range(max_len)]
        text_b_ids = df[b_cols].values
        label_ids = df['label'].values
        print(type(text_a_ids), text_a_ids.shape)
        print(type(text_b_ids), text_b_ids.shape)
        print(type(label_ids), label_ids.shape)
```

# Task type for text-sequence-tagging

USAGE:
fast-record tagging [OPTIONS] --path <PATH>

OPTIONS:

    -h, --help
            Print help information

        --max-vocab-size <MAX_VOCAB_SIZE>
            max vocabulary size for build record, only effective when the with-vocab is not set
            [default: 10000]

    -o, --output-path <OUTPUT_PATH>
            output path of record [aliases: output]

    -p, --path <PATH>
            path of tagging dataset [aliases: input]

        --padding <PADDING>
            padding special token of vocabulary [default: <PAD>] [aliases: PAD]

        --padding-tag <PADDING_TAG>
            padding tag [default: None]

    -s, --separator <SEPARATOR>
            separator between word and tag [default: "\t"] [aliases: delimiter]

        --sequence-length <SEQUENCE_LENGTH>
            max sequence length for sentence [default: 32]

        --stopwords-file <STOPWORDS_FILE>
            stopwords file for build vocabulary, only effective when the with-vocab is not set
            [aliases: stopwords]

        --unknown <UNKNOWN>
            [default: <UNK>] [aliases: UNK]

    -V, --version
            Print version information

        --with-vocab
            with user vocabulary for classifier dataset

```python
"""record sequence-tagging records"""
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


def read_tagging():
    base_path = "../data/MSRA"
    for filename in os.listdir(base_path):
        if not filename.endswith("ipc"):
            continue
        file_path = os.path.join(base_path, filename)
        df = read_ipc_file(file_path)
        max_len = df.shape[1] // 2
        w_cols = [f'word_{i}' for i in range(max_len)]
        t_cols = [f'tag_{i}' for i in range(max_len)]
        word_ids = df[w_cols].values
        tag_ids = df[t_cols].values
        print(type(word_ids), word_ids.shape)
        print(type(tag_ids), tag_ids.shape)
```