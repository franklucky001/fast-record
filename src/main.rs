mod dataset;
use clap::{Parser, Subcommand};
use dataset::{ClassifierArgs, ClassifierBuilder, SimilarityArgs, TaggingArgs};

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
        },
        Command::Tagging(args) => {
            print!("tagging dataset args: {:?}", args);
        },
        Command::Help => ()
    }
    println!("finished record!");
}
