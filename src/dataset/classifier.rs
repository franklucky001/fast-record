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

/// classifier args structure
#[derive(Args, Debug)]
pub struct ClassifierArgs{
    /// path of classifier dataset
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
    /// origin sample data use label_id instead of label
    #[clap(long)]
    with_label_id: bool,
    /// with en language
    #[clap(long)]
    with_lang_en: bool,
    /// separator between sentence and label
    #[clap(long, short, visible_alias="delimiter", default_value = "\t")]
    separator: String,
    /// unknown special token of vocabulary
    #[clap(long, visible_alias = "unk-token", default_value = "<UNK>")]
    unknown: String,
    /// padding special token of vocabulary
    #[clap(long, visible_alias = "pad-token", default_value = "<PAD>")]
    padding: String,
}

pub(crate) struct ClassifierRecord {
    word_ids: Vec<usize>,
    label_id: usize,
}

impl ClassifierRecord {
    pub fn new(mut word_ids: Vec<usize>, label_id: usize, max_length: usize) -> Self{
        if word_ids.len() > max_length{
            let _ = word_ids.split_off(max_length);
        }else if word_ids.len() < max_length {
            let len = word_ids.len();
            word_ids.append(&mut vec![0usize; max_length - len]);
        }
        Self{
            word_ids,
            label_id
        }
    }
}

pub(crate) struct ClassifierSample(String, String);

impl ClassifierSample {
    pub fn new(sent: & str, label: & str) -> Self{
        Self(sent.to_string(), label.to_string())
    }
}

pub struct ClassifierBuilder<'a>{
    args:&'a ClassifierArgs,
    vocab: HashMap<String, usize>,
    classes: HashMap<String, usize>,
    stopwords: HashSet<String>,
}

impl <'a>ClassifierBuilder<'a> {
    pub fn new(args: &'a ClassifierArgs) ->Self{
        Self{
            args,
            vocab: HashMap::new(),
            classes: HashMap::new(),
            stopwords: HashSet::new(),
        }
    }
}

impl<'a> IDataset<ClassifierSample, ClassifierRecord> for ClassifierBuilder<'a> {

    fn init(&mut self, train_samples: & Vec<ClassifierSample>){
        let base_path = Path::new(&self.args.path);
        let classes_file = base_path.join("class.txt");
        let class_reader = BufReader::new(File::open(classes_file).unwrap());
        class_reader
            .lines()
            .filter_map(Result::ok)
            .enumerate()
            .for_each(|(i, label)|{
                self.classes.insert(label, i);
            });
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
        train_samples
            .iter()
            .for_each(|item|if self.args.with_lang_en{
                item.0
                    .split(' ')
                    .for_each(|word|{vocab.insert(word.to_string());})
            }else {
                item.0
                    .chars()
                    .for_each(|ch|{vocab.insert(ch.to_string());})
            }
            );
        vocab
            .into_iter()
            .filter(|word|!self.stopwords.contains(word))
            .enumerate()
            .for_each(|(i, word)|{self.vocab.insert(word, i + 1);});
        let len = self.vocab.len();
        self.vocab.insert(self.args.unknown.to_owned(), len);
    }

    fn read_dataset(&self, file: &str) -> Vec<ClassifierSample> {
        let base_path = Path::new(&self.args.path);
        let data_file = base_path.join(file);
        let data_reader = BufReader::new(File::open(data_file).unwrap());
        data_reader
            .lines()
            .par_bridge()
            .filter_map(Result::ok)
            .map(|line|line
                .split_once(&self.args.separator)
                .map(|item|ClassifierSample::new(item.0, item.1)).unwrap()
            )
            .collect()
    }

    fn build_dataset(&self, samples: Vec<ClassifierSample>) -> Vec<ClassifierRecord> {
        let max_length = self.args.sequence_length;
        let unk_id = self.vocab.get(&self.args.unknown).unwrap();
        let pb = ProgressBar::new(samples.len() as u64);
        let records = samples
            .into_par_iter()
            .map(|sample|{
                pb.inc(1);
                if self.args.with_lang_en{
                    let word_ids = sample.0
                        .split(' ')
                        .map(|word| self.vocab
                            .get(word).map(|it| *it)
                            .unwrap_or(*unk_id)
                        ).collect::<Vec<_>>();
                    (word_ids, sample.1)
                }else {
                    let word_ids = sample.0
                        .chars()
                        .map(|ch| self.vocab
                            .get(&ch.to_string()).map(|it| *it)
                            .unwrap_or(*unk_id)
                        ).collect::<Vec<_>>();
                    (word_ids, sample.1)
                }
            })
            .map(|(word_ids, label)|{
                if self.args.with_label_id{
                    ClassifierRecord::new(word_ids, label.parse().unwrap(), max_length)
                }else {
                    let label_id = self.classes.get(&label).unwrap();
                    ClassifierRecord::new(word_ids, *label_id, max_length)
                }
            }).collect::<Vec<_>>();
        pb.finish_with_message("done");
        records
    }
    fn save_dataset(&self, records: Vec<ClassifierRecord>, record_file: &str) {
        let output_path = self.get_output_path();
        let max_length = self.args.sequence_length;
        let mut fields = Vec::new();
        for k in 0..max_length{
            let field = Field::new(&format!("word_{}", k), DataType::UInt32, false);
            fields.push(field);
        }
        let field = Field::new("class", DataType::UInt8, false);
        fields.push(field);
        let schema = Arc::new(Schema::new(fields));
        let record_file = File::create(output_path.join(record_file)).expect(&format!("create record file {} failed", record_file));
        let mut writer = FileWriter::try_new(record_file, &schema).expect("create file writer failed");
        for chunk in records.chunks(100){
            let mut values = Vec::new();
            for i in 0..max_length{
                let series = chunk
                    .iter()
                    .map(|item|item.word_ids[i] as u32)
                    .collect::<Vec<u32>>();
                values.push(Arc::new(UInt32Array::from(series)) as ArrayRef);
            }
            let label_ids = chunk
                .iter()
                .map(|item|item.label_id as u8)
                .collect::<Vec<u8>>();
            values.push(Arc::new(UInt8Array::from(label_ids)) as ArrayRef);
            let batch = RecordBatch::try_new(schema.clone(), values).expect("build batch error");
            writer.write(&batch).expect("write record error");
        }
        writer.finish().expect("finish write records error");
    }
    fn save_vocab(&self){
        let output_path = self.get_output_path();
        let vocab_file = File::create(output_path.join("vocab.txt")).expect("create vocab file failed");
        let mut writer = BufWriter::new(vocab_file);
        for (word, idx) in &self.vocab{
            writeln!(&mut writer, "{}\t{}", idx, word).expect("write vocab line failed");
        }
    }
    fn get_output_path(&self) -> &Path {
        match &self.args.output_path{
            None => Path::new(&self.args.path),
            Some(output_path) => Path::new(output_path)
        }
    }
}