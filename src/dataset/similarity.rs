use std::collections::{HashSet, HashMap};
use std::fs::File;
use std::io::{BufRead, Write, BufReader, BufWriter};
use std::path::Path;
use arrow::ipc::writer::FileWriter;
use std::sync::Arc;
use arrow::array::{ArrayRef, UInt8Array, UInt32Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::{RecordBatch};
use rayon::prelude::*;
use indicatif::ProgressBar;
use crate::dataset::traits::IDataset;
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
    /// max sequence length for sentence
    #[clap(long, default_value = "32")]
    sequence_length: usize,
    /// stopwords file for build vocabulary, only effective when the with-vocab is not set
    #[clap(long, visible_alias="stopwords")]
    stopwords_file: Option<String>,
    /// similarity with boolean value
    #[clap(long)]
    with_bool: bool,
    /// with en language
    #[clap(long)]
    with_lang_en: bool,
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

pub(crate) struct  SimilarityRecord{
    front_word_ids: Vec<usize>,
    back_word_ids: Vec<usize>,
    label: u8
}

impl SimilarityRecord {
    pub fn new(mut front_word_ids: Vec<usize>, mut back_word_ids: Vec<usize>, label: u8, max_length: usize) -> Self{
        let front_length = front_word_ids.len();
        let back_length = back_word_ids.len();
        if front_length > max_length{
            let _ = front_word_ids.split_off(max_length);
        }else if front_length < max_length{
            front_word_ids.append(&mut vec![0usize; max_length - front_length]);
        }
        if back_length > max_length{
            let _ = back_word_ids.split_off(max_length);
        }else if back_length < max_length {
            back_word_ids.append(&mut vec![0usize; max_length - back_length]);
        }
        Self{
            front_word_ids,
            back_word_ids,
            label
        }
    }
}
pub(crate) struct  SimilaritySample(String, String, u8);

impl SimilaritySample{
    pub fn new(sent_a: & str, sent_b: & str, label: u8) -> Self{
        Self(sent_a.to_string(), sent_b.to_string(), label)
    }
}

pub struct SimilarityBuilder<'a>{
    args: &'a SimilarityArgs,
    vocab: HashMap<String, usize>,
    stopwords: HashSet<String>,
}

impl<'a> SimilarityBuilder<'a> {
    pub fn new(args: &'a SimilarityArgs) -> Self{
        Self{
            args,
            vocab: HashMap::new(),
            stopwords: HashSet::new(),
        }
    }
}

impl<'a> IDataset<SimilaritySample, SimilarityRecord> for SimilarityBuilder<'a>  {
    fn init(&mut self, train_samples: & Vec<SimilaritySample>){
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
            .for_each(|sample|if self.args.with_lang_en{
                sample.0
                    .split(' ')
                    .for_each(|word|{vocab.insert(word.to_string());});
                sample.1
                    .split(' ')
                    .for_each(|word|{vocab.insert(word.to_string());});
            }else {
                sample.0
                    .chars()
                    .for_each(|ch|{vocab.insert(ch.to_string());});
                sample.1
                    .chars()
                    .for_each(|ch|{vocab.insert(ch.to_string());});
            });
        vocab
            .into_iter()
            .filter(|word|!self.stopwords.contains(word))
            .enumerate()
            .for_each(|(i, word)|{self.vocab.insert(word, i + 1);});
        let len = self.vocab.len();
        self.vocab.insert(self.args.unknown.to_owned(), len);
    }

    fn read_dataset(&self, file: & str) -> Vec<SimilaritySample>{
        let base_path = Path::new(&self.args.path);
        let data_file = base_path.join(file);
        let data_reader = BufReader::new(File::open(data_file).unwrap());
        data_reader
            .lines()
            .par_bridge()
            .filter_map(Result::ok)
            .map(|line| line
                .split_once(&self.args.label_sep)
                .map(|(context, label)|(context.to_string(), label.to_string()))
                .map(|(context, label)|{
                    context
                        .split_once(&self.args.sent_sep)
                        .map(|item|{
                            let  label_id;
                            if self.args.with_bool{
                                let tag: bool = label.parse().unwrap();
                                label_id = tag as u8;
                            }else {
                                label_id = label.parse().unwrap();
                            }
                            SimilaritySample::new(item.0, item.1, label_id)
                        }).unwrap()
                }).unwrap()
            )
            .collect()
    }
    fn build_dataset(&self, samples: Vec<SimilaritySample>) -> Vec<SimilarityRecord>{
        let max_length = self.args.sequence_length;
        let unk_id = self.vocab.get(&self.args.unknown).unwrap();
        let pb = ProgressBar::new(samples.len() as u64);
        let records = samples
            .into_par_iter()
            .map(|sample|{
                pb.inc(1);
                if self.args.with_lang_en{
                    let text_a_ids = sample.0
                        .split(' ')
                        .map(|word|self.vocab
                            .get(word)
                            .map(|it|*it)
                            .unwrap_or(*unk_id)
                        ).collect::<Vec<_>>();
                    let text_b_ids = sample.1
                        .split(' ')
                        .map(|word|self.vocab
                            .get(word)
                            .map(|it|*it)
                            .unwrap_or(*unk_id)
                        ).collect::<Vec<_>>();
                    (text_a_ids, text_b_ids, sample.2)
                }else {
                    let text_a_ids = sample.0
                        .chars()
                        .map(|ch|self.vocab
                            .get(&ch.to_string())
                            .map(|it|*it)
                            .unwrap_or(*unk_id)
                        ).collect::<Vec<_>>();
                    let text_b_ids = sample.1
                        .chars()
                        .map(|ch|self.vocab
                            .get(&ch.to_string())
                            .map(|it|*it)
                            .unwrap_or(*unk_id)
                        ).collect::<Vec<_>>();
                    (text_a_ids, text_b_ids, sample.2)
                }
            })
            .map(|(text_a_ids, text_b_ids, label)|{
                SimilarityRecord::new(text_a_ids, text_b_ids, label, max_length)
            }).collect();
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
    fn save_dataset(&self, records: Vec<SimilarityRecord>, record_file: & str){
        let output_path = self.get_output_path();
        let max_length = self.args.sequence_length;
        let mut fields = Vec::new();
        for k in 0..max_length{
            let field = Field::new(&format!("text_a_{}", k), DataType::UInt32, false);
            fields.push(field);
        }
        for k in 0..max_length{
            let field = Field::new(&format!("text_b_{}", k), DataType::UInt32, false);
            fields.push(field);
        }
        let field = Field::new("label", DataType::UInt8, false);
        fields.push(field);
        let schema = Arc::new(Schema::new(fields));
        let record_file = File::create(output_path.join(record_file)).expect(&format!("create record file {} failed", record_file));
        let mut writer = FileWriter::try_new(record_file, &schema).expect("create file writer failed");
        for chunk in records.chunks(100){
            let mut values = Vec::new();
            for i in 0..max_length{
                let series_a = chunk
                    .iter()
                    .map(|item|item.front_word_ids[i] as u32)
                    .collect::<Vec<u32>>();
                values.push(Arc::new(UInt32Array::from(series_a)) as ArrayRef);
            }
            for i in 0..max_length{
                let series_b = chunk
                    .iter()
                    .map(|item|item.back_word_ids[i] as u32)
                    .collect::<Vec<u32>>();
                values.push(Arc::new(UInt32Array::from(series_b)) as ArrayRef);
            }
            let label_ids = chunk
                .iter()
                .map(|item|item.label)
                .collect::<Vec<u8>>();
            values.push(Arc::new(UInt8Array::from(label_ids)) as ArrayRef);
            let batch = RecordBatch::try_new(schema.clone(), values).expect("build batch error");
            writer.write(&batch).expect("write record error");
        }
        writer.finish().expect("finished write records error");
    }
    fn get_output_path(&self) -> &Path {
        match &self.args.output_path{
            None => Path::new(&self.args.path),
            Some(output_path) => Path::new(output_path)
        }
    }
}