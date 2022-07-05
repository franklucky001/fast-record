use clap::Args;

/// similarity args structure
#[derive(Args, Debug)]
pub struct SimilarityArgs{
    /// path of similarity dataset
    #[clap(long, short, visible_alias="input")]
    path: String,
    /// output path of record
    #[clap(long, short, visible_alias="output")]
    output_path: Option<String>,
    /// with user vocabulary for classifier dataset
    #[clap(long)]
    with_vocab: bool,
    /// max vocabulary size for build record, only effective when the with-vocab is not set
    #[clap(long, default_value = "10000")]
    max_vocab_size: usize,
    /// stopwords file for build vocabulary, only effective when the with-vocab is not set
    #[clap(long, visible_alias="stopwords")]
    stopwords_file: Option<String>,
    /// similarity with boolean value
    #[clap(long)]
    with_bool: bool,
    /// separator between text_a and text_b
    #[clap(long, visible_alias = "s1", default_value = "\t")]
    sent_sep: String,
    #[clap(long, visible_alias = "s2", default_value = "\t")]
    /// separator between text and label
    label_sep: String,
    #[clap(long, visible_alias = "unk-token", default_value = "<UNK>")]
    unknown: String,
    /// padding special token of vocabulary
    #[clap(long, visible_alias = "pad-token", default_value = "<PAD>")]
    padding: String,
}