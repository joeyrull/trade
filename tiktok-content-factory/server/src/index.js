require('dotenv').config();
const express = require('express');
const cors = require('cors');
const discoveryRouter = require('./routes/discovery');
const curationRouter = require('./routes/curation');
const editorRouter = require('./routes/editor');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.use('/api/discovery', discoveryRouter);
app.use('/api/curation', curationRouter);
app.use('/api/editor', editorRouter);

app.get('/api/health', (req, res) => res.json({ status: 'ok', version: '1.0.0' }));

app.listen(PORT, () => {
  console.log(`TikTok Content Factory server running on http://localhost:${PORT}`);
});
