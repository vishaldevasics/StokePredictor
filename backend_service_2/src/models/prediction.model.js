import mongoose from 'mongoose';

const predictionSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  gender: String,
  age: Number,
  hypertension: Number,
  heartDisease: Number,
  everMarried: String,
  workType: String,
  residenceType: String,
  avgGlucoseLevel: Number,
  bmi: Number,
  smokingStatus: String,
  prediction: Number
}, {
  timestamps: true
});

export const Prediction = mongoose.model('Prediction', predictionSchema);
