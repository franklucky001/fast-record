use std::collections::{HashMap};
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

struct ClassifierRecord {
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

struct ClassifierSample(String, String);

impl ClassifierSample {
    pub fn new(sent: & str, label: & str) -> Self{
        Self(sent.to_string(), label.to_string())
    }
}

pub struct ClassifierBuilder<'a>{
    args:&'a ClassifierArgs,
    vocab: HashMap<String, usize>,
    classes: HashMap<String, usize>,
}

impl <'a>ClassifierBuilder<'a> {
    pub fn new(args: &'a ClassifierArgs) ->Self{
        Self{
            args,
            vocab: HashMap::new(),
            classes: HashMap::new(),
        }
    }

    pub fn build(&mut self) {
        let base_path = Path::new(&self.args.path);
        let output_path = match &self.args.output_path{
            None => base_path,
            Some(output_path) => Path::new(output_path)
        };
        let classes_file = base_path.join("class.txt");
        let class_reader = BufReader::new(File::open(classes_file).unwrap());
        let train_samples = self.read_dataset(base_path, "train.txt");
        println!("total {} samples of train data", train_samples.len());
        self.init(class_reader, &train_samples);
        let train_records = self.build_dataset(train_samples);
        let dev_samples = self.read_dataset(base_path, "dev.txt");
        println!("total {} samples of dev data", dev_samples.len());
        let dev_records = self.build_dataset(dev_samples);
        let test_samples = self.read_dataset(base_path, "test.txt");
        println!("total {} samples of test data", test_samples.len());
        let test_records = self.build_dataset(test_samples);
        self.save_vocab(output_path);
        self.save_dataset(train_records, output_path, "train.records.ipc");
        self.save_dataset(dev_records, output_path, "dev.records.ipc");
        self.save_dataset(test_records, output_path, "test.records.ipc");
    }
    fn init(&mut self, class_reader: BufReader<File>, train_samples: & Vec<ClassifierSample>){
        class_reader
            .lines()
            .filter_map(Result::ok)
            .map(|line|line)
            .enumerate()
            .for_each(|(i, label)|{
                self.classes.insert(label, i);
            });
        self.vocab.insert(self.args.padding.to_owned(), 0);
        train_samples
            .iter()
            .for_each(|item|item.0
                .chars()
                .enumerate()
                .for_each(|(i, ch)|{self.vocab.insert(ch.to_string(), i+1);})
            );
        let len = self.vocab.len();
        self.vocab.insert(self.args.unknown.to_owned(), len);
    }
    fn read_dataset(&self, base_path:& Path, file: & str) -> Vec<ClassifierSample>{
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
    fn build_dataset(&self, samples: Vec<ClassifierSample>) -> Vec<ClassifierRecord>{
        let max_length = self.args.sequence_length;
        let unk_id = self.vocab.get(&self.args.unknown).unwrap();
        samples
            .into_par_iter()
            .map(|sample|{
                let word_ids = sample.0
                    .chars()
                    .map(|ch|self.vocab
                        .get(&ch.to_string()).map(|it| * it)
                        .unwrap_or(*unk_id)
                    ).collect::<Vec<_>>();
                (word_ids, sample.1)
            })
            .map(|(word_ids, label)|{
                if self.args.with_label_id{
                    ClassifierRecord::new(word_ids, label.parse().unwrap(), max_length)
                }else {
                    let label_id = self.classes.get(&label).unwrap();
                    ClassifierRecord::new(word_ids, *label_id, max_length)
                }
            }).collect::<Vec<_>>()
    }
    fn save_vocab(&self, output_path: &Path){
        let vocab_file = File::create(output_path.join("vocab.txt")).expect("create vocab file failed");
        let mut writer = BufWriter::new(vocab_file);
        for (word, idx) in &self.vocab{
            writeln!(&mut writer, "{}\t{}", idx, word).expect("write vocab line failed");
        }
    }
    fn save_dataset(&self,
                    records: Vec<ClassifierRecord>,
                    output_path: &Path,
                    file: & str){
        let max_length = self.args.sequence_length;
        let mut fields = Vec::new();
        for k in 0..max_length{
            let field = Field::new(&format!("id_{}", k), DataType::UInt32, false);
            fields.push(field);
        }
        let field = Field::new("label", DataType::UInt8, false);
        fields.push(field);
        let schema = Arc::new(Schema::new(fields));
        let record_file = File::create(output_path.join(file)).expect(&format!("create record file {} failed", file));
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
        writer.finish().expect("finish write error");
    }
}