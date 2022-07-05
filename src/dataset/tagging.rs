use clap::Args;

/// tagging args structure
#[derive(Args, Debug)]
pub struct TaggingArgs{
    /// path of tagging dataset
    #[clap(long, short, visible_alias="input")]
    path: String,
    /// output path of record
    #[clap(long, short, visible_alias="output")]
    output_path: Option<String>,
    /// with user vocabulary for classifier dataset
    #[clap(long)]
    with_vocab: bool,
    /// separator between word and tag
    #[clap(long, short, visible_alias="delimiter", default_value = "\t")]
    separator: String,
    #[clap(long, visible_alias = "UNK", default_value = "<UNK>")]
    unknown: String,
    /// padding special token of vocabulary
    #[clap(long, visible_alias = "PAD", default_value = "<PAD>")]
    padding: String,
}