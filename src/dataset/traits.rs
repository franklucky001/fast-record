use std::path::Path;

pub trait IDataset<S, R>{
    fn init(&mut self, samples: & Vec<S>);
    fn read_dataset(&self, file: & str) -> Vec<S>;
    fn build_dataset(&self, samples: Vec<S>) -> Vec<R>;
    fn save_vocab(&self);
    fn save_dataset(&self, records: Vec<R>, record_file: & str);
    fn get_output_path(&self) -> &Path;

    fn build(&mut self) {
        let train_samples = self.read_dataset("train.txt");
        println!("Processing train data...");
        self.init(&train_samples);
        let train_records = self.build_dataset(train_samples);
        let dev_samples = self.read_dataset("dev.txt");
        println!("Processing dev data...");
        let dev_records = self.build_dataset(dev_samples);
        let test_samples = self.read_dataset( "test.txt");
        println!("Processing test data...");
        let test_records = self.build_dataset(test_samples);
        self.save_vocab();
        self.save_dataset(train_records,  "train.records.ipc");
        self.save_dataset(dev_records, "dev.records.ipc");
        self.save_dataset(test_records,  "test.records.ipc");
    }
}