require('dotenv').config({ path: '../.env' });
const express = require('express');
const cors = require('cors');
const path = require('path');
const analysisRoutes = require('./routes/analysis');

const app = express();
const PORT = process.env.PORT || 3002;

app.use(cors());
app.use(express.json());

// Serve uploaded originals and job outputs
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/jobs', express.static(path.join(__dirname, 'jobs')));

app.use('/api/analysis', analysisRoutes);

app.listen(PORT, () => {
  console.log(`Pitcher Review API running on http://localhost:${PORT}`);
});
