use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, Write, BufReader, BufWriter};
use std::path::Path;
use arrow::ipc::writer::FileWriter;
use std::sync::Arc;
use arrow::array::{ArrayRef, UInt8Array, UInt32Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::{RecordBatch};
use rayon::prelude::*;
use clap::Args;
use crate::dataset::traits::IDataset;
use indicatif::ProgressBar;

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
    /// max vocabulary size for build record, only effective when the with-vocab is not set
    #[clap(long, default_value = "10000")]
    max_vocab_size: usize,
    /// max sequence length for sentence
    #[clap(long, default_value = "32")]
    sequence_length: usize,
    /// stopwords file for build vocabulary, only effective when the with-vocab is not set
    #[clap(long, visible_alias="stopwords")]
    stopwords_file: Option<String>,
    // /// with en language
    // #[clap(long)]
    // with_lang_en: bool,
    /// separator between word and tag
    #[clap(long, short, visible_alias="delimiter", default_value = "\t")]
    separator: String,
    #[clap(long, visible_alias = "UNK", default_value = "<UNK>")]
    unknown: String,
    /// padding special token of vocabulary
    #[clap(long, visible_alias = "PAD", default_value = "<PAD>")]
    padding: String,
    /// padding tag
    #[clap(long, default_value = "None")]
    padding_tag: String,
}

pub(crate) struct TaggingSample{
    tokens: Vec<String>,
    tags: Vec<String>
}

impl TaggingSample {
    pub(crate) fn new(tokens: Vec<String>, tags: Vec<String>) -> Self{
        Self{
            tokens,
            tags
        }
    }
}

pub(crate) struct TaggingRecord{
    token_ids: Vec<usize>,
    tag_ids: Vec<usize>
}

impl TaggingRecord {
    pub(crate) fn new(mut token_ids: Vec<usize>, mut tag_ids: Vec<usize>, max_length: usize) -> Self{
        if token_ids.len() > max_length{
            panic!("max length is less then current length {} !", token_ids.len());
        }else if token_ids.len() < max_length{
            let length = token_ids.len();
            token_ids.append(&mut vec![0usize; max_length - length]);
            tag_ids.append(&mut vec![0usize; max_length - length]);
        }
        Self{
            token_ids,
            tag_ids
        }
    }
}

pub struct TaggingBuilder<'a>{
    args: & 'a TaggingArgs,
    vocab: HashMap<String, usize>,
    tags: HashMap<String, usize>,
    stopwords: HashSet<String>,
}

impl<'a> TaggingBuilder<'a> {
    pub fn new(args: &'a TaggingArgs) -> Self{
        Self{
            args,
            vocab: HashMap::new(),
            tags: HashMap::new(),
            stopwords: HashSet::new(),
        }
    }
}

impl <'a> IDataset<TaggingSample, TaggingRecord> for TaggingBuilder<'a> {
    fn init(&mut self, train_samples: & Vec<TaggingSample>){
        if let Some(stopwords_file) = &self.args.stopwords_file{
            println!("reader stopwords file from {}", stopwords_file);
            let stopwords_reader = BufReader::new(File::open(stopwords_file).expect("open stopwords file failed"));
            stopwords_reader
                .lines()
                .filter_map(Result::ok)
                .for_each(|word|{
                    self.stopwords.insert(word);
                })
        }
        self.vocab.insert(self.args.padding.to_owned(), 0);
        let mut vocab = HashSet::new();
        let mut tags = HashSet::new();
        train_samples
            .iter()
            .for_each(|sample|{
                sample.tokens
                    .iter()
                    .for_each(|token|{
                        vocab.insert(token.to_string());
                    });
                sample.tags.iter().for_each(|tag|{
                    tags.insert(tag.to_string());
                })
            });
        vocab
            .into_iter()
            .filter(|token|!self.stopwords.contains(token))
            .enumerate()
            .for_each(|(i, word)|{self.vocab.insert(word, i + 1);});
        self.tags.insert(self.args.padding_tag.to_owned(), 0);
        tags.into_iter().enumerate().for_each(|(i, tag)|{self.tags.insert(tag, i + 1);});
        let len = self.vocab.len();
        self.vocab.insert(self.args.unknown.to_owned(), len);
    }

    fn read_dataset(&self, file: & str) -> Vec<TaggingSample>{
        let base_path = Path::new(&self.args.path);
        let data_file = base_path.join(file);
        let data_reader = BufReader::new(File::open(data_file).unwrap());
        let lines = data_reader
            .lines()
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        lines
            .split(|s|s.is_empty())
            .map(|group|{
                let (tokens, tags): (Vec<_>, Vec<_>) = group
                    .into_iter()
                    .map(|line|line
                        .split_once(&self.args.separator)
                        .map(|item|(item.0.to_string(), item.1.to_string())).unwrap()
                    )
                    .unzip();
                TaggingSample::new(tokens, tags)
            })
            .collect()
    }

    fn build_dataset(&self, samples: Vec<TaggingSample>) -> Vec<TaggingRecord>{
        let max_length = self.args.sequence_length;
        let unk_id = self.vocab.get(&self.args.unknown).unwrap();
        let pb = ProgressBar::new(samples.len() as u64);
        let records = samples
            .into_par_iter()
            .map(|sample|{
                pb.inc(1);
                let word_ids = sample.tokens
                    .into_iter()
                    .map(|token|self.vocab
                        .get(&token)
                        .map(|it|*it)
                        .unwrap_or(*unk_id)
                    ).collect::<Vec<_>>();
                let tag_ids = sample.tags
                    .into_iter()
                    .map(|tag|self.tags
                        .get(&tag)
                        .map(|it| *it)
                        .unwrap_or(0)
                    ).collect::<Vec<_>>();
                TaggingRecord::new(word_ids, tag_ids, max_length)
            })
            .collect();
        pb.finish_with_message("done");
        records
    }
    fn save_vocab(&self){
        let output_path = self.get_output_path();
        let vocab_file = File::create(output_path.join("vocab.txt")).expect("create vocab file failed");
        let mut writer = BufWriter::new(vocab_file);
        for (word, idx) in &self.vocab{
            writeln!(&mut writer, "{}\t{}", idx, word).expect("write vocab line failed");
        }
    }

    fn save_dataset(&self, records: Vec<TaggingRecord>, record_file: & str){
        let output_path = self.get_output_path();
        let max_length = self.args.sequence_length;
        let mut fields = Vec::new();
        for k in 0..max_length{
            let field = Field::new(&format!("word_{}", k), DataType::UInt32, false);
            fields.push(field);
            let field = Field::new(&format!("tag_{}", k), DataType::UInt8, false);
            fields.push(field);
        }
        let schema = Arc::new(Schema::new(fields));
        let record_file = File::create(output_path.join(record_file)).expect(&format!("create record file {} failed", record_file));
        let mut writer = FileWriter::try_new(record_file, &schema).expect("create file writer failed");
        for chunk in records.chunks(100){
            let mut values = Vec::new();
            for i in 0..max_length{
                let series = chunk
                    .iter()
                    .map(|record|record.token_ids[i] as u32)
                    .collect::<Vec<u32>>();
                values.push(Arc::new(UInt32Array::from(series)) as ArrayRef);
                let series = chunk
                    .iter()
                    .map(|record|record.tag_ids[i] as u8)
                    .collect::<Vec<u8>>();
                values.push(Arc::new(UInt8Array::from(series)) as ArrayRef);
            }
            let batch = RecordBatch::try_new(schema.clone(), values).expect("build batch error");
            writer.write(&batch).expect("write record error");
        }
        writer.finish().expect("finish write records error");
    }
    fn get_output_path(&self) -> &Path {
        match &self.args.output_path{
            None => Path::new(&self.args.path),
            Some(output_path) => Path::new(output_path)
        }
    }
}