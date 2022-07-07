mod dataset;
use clap::{Parser, Subcommand};
use dataset::{ClassifierArgs, ClassifierBuilder, SimilarityArgs, TaggingArgs};
use crate::dataset::{SimilarityBuilder, TaggingBuilder};
use crate::dataset::IDataset;

/// Record builder for NLP task
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
struct FastRecord{
    /// Task type supported for build records
    #[clap(subcommand)]
    command: Option<Command>,
}

/// All of subcommand
#[derive(Subcommand)]
enum Command{
    /// Task type for text classifier
    Classifier(ClassifierArgs),
    /// Task type for text-pair similarity
    Similarity(SimilarityArgs),
    /// Task type for text-sequence tagging
    Tagging(TaggingArgs),
    /// help for `fast-record'
    Help
}

fn main() {
    let fast_record = FastRecord::parse();
    match &fast_record.command.unwrap_or_else(||{
        println!("fast-record must with subcommand, use `fast-record help` get the usage");
        Command::Help
    }){
        Command::Classifier(args) => {
            println!("classifier dataset args: {:?}", args);
            let mut builder = ClassifierBuilder::new(args);
            builder.build();
        },
        Command::Similarity(args) => {
            println!("similarity dataset args: {:?}", args);
            let mut builder = SimilarityBuilder::new(args);
            builder.build();
        },
        Command::Tagging(args) => {
            println!("tagging dataset args: {:?}", args);
            let mut builder = TaggingBuilder::new(args);
            builder.build();
        },
        Command::Help => ()
    }
    println!("finished record!");
}
