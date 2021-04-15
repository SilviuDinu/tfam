const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');

const ISIC_IMAGES_PATH = path.join(__dirname, '../../data/image-data/images');
let processedImageNames = [];

fs.readdirSync(ISIC_IMAGES_PATH).forEach(file => {
  if(file.includes('.jpg')) {
    processedImageNames.push(file.split('.')[0]);
  }
});

const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const trainDataWriter = createCsvWriter({
  path: path.join(__dirname, '../../data/csv-data/train.csv'),
  header: [
    { id: '_id', title: '_id' },
    { id: 'name', title: 'name' },
    { id: 'meta.clinical.age_approx', title: 'Age' },
    { id: 'meta.clinical.benign_malignant', title: 'Benign/Malign' },
    { id: 'meta.clinical.diagnosis', title: 'Diagnosis' },
    { id: 'meta.clinical.sex', title: 'Sex' },
    { id: 'meta.acquisition.acquisition_day', title: 'Acquisition day' },
    { id: 'meta.acquisition.color_tint', title: 'Color tint' },
    { id: 'meta.clinical.mel_type', title: 'Melanoma type' },
    { id: 'meta.clinical.diagnosis_confirm_type', title: 'Diagnosis confirm type' },
  ],
});

const testDataWriter = createCsvWriter({
  path: path.join(__dirname, '../../data/csv-data/test.csv'),
  header: [
    { id: '_id', title: '_id' },
    { id: 'name', title: 'name' },
    { id: 'meta.clinical.age_approx', title: 'Age' },
    { id: 'meta.clinical.benign_malignant', title: 'Benign/Malign' },
    { id: 'meta.clinical.diagnosis', title: 'Diagnosis' },
    { id: 'meta.clinical.sex', title: 'Sex' },
    { id: 'meta.acquisition.acquisition_day', title: 'Acquisition day' },
    { id: 'meta.acquisition.color_tint', title: 'Color tint' },
    { id: 'meta.clinical.mel_type', title: 'Melanoma type' },
    { id: 'meta.clinical.diagnosis_confirm_type', title: 'Diagnosis confirm type' },
  ],
});

const filteredData = [];

const metadata = path.join(__dirname, '../../data/csv-data/metadata.csv');
console.log('Processing items...');

fs.createReadStream(metadata)
  .pipe(csv())
  .on('data', async row => {
    if (processedImageNames.includes(row.name) && hasMalignantInfo(row)) {
      filteredData.push(row);
    }
  })
  .on('end', async () => {
    console.log(`${filteredData.length} items ready`);
    if (filteredData.length > 0) {
      const train = filteredData.splice(0, parseInt((filteredData.length * 80) / 100));
      const test = filteredData;

      Promise
        .all([writeTrainData(train), writeTestData(test)])
        .then(() => console.log('All done!'))
        .catch(err => console.error(err));

    } else {
      console.log('No items to write. Aborting...');
    }
  });

writeTrainData = async data => {
  console.log(`Writing ${data.length} elements to train.csv...`);
  const trainPromise = trainDataWriter.writeRecords(data);
  return await trainPromise;
}

writeTestData = async data => {
  console.log(`Writing ${data.length} elements to test.csv...`);
  const testPromise = testDataWriter.writeRecords(data);
  return await testPromise;
}

const hasMalignantInfo = row => {
  return (
    !!row['meta.clinical.benign_malignant'] &&
    (row['meta.clinical.benign_malignant'] === 'benign'
    || row['meta.clinical.benign_malignant'] === 'malignant')
  );
};
