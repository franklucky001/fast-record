mod classifier;
mod similarity;
mod tagging;
mod traits;

pub use classifier::{ClassifierArgs, ClassifierBuilder};
pub use similarity::{SimilarityArgs, SimilarityBuilder};
pub use tagging::{TaggingArgs, TaggingBuilder};
pub use traits::IDataset;