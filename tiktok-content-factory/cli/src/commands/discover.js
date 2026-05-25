const axios = require('axios');

function parseArgs(args) {
  const opts = { source: 'both', type: 'photo', page: 1 };
  let query = '';
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--source') opts.source = args[++i];
    else if (args[i] === '--type') opts.type = args[++i];
    else if (args[i] === '--page') opts.page = parseInt(args[++i]);
    else if (!args[i].startsWith('--')) query = args[i];
  }
  return { query, opts };
}

async function searchUnsplash(query, opts) {
  const { data } = await axios.get('https://api.unsplash.com/search/photos', {
    headers: { Authorization: `Client-ID ${process.env.UNSPLASH_ACCESS_KEY}` },
    params: { query, page: opts.page, per_page: 10, orientation: 'portrait' },
  });
  return data.results.map(p => ({
    id: p.id,
    source: 'unsplash',
    type: 'photo',
    url: p.urls.regular,
    description: p.description || p.alt_description || '(no description)',
    credit: p.user.name,
  }));
}

async function searchPexels(query, opts) {
  const endpoint = opts.type === 'video' ? '/videos/search' : '/v1/search';
  const { data } = await axios.get(`https://api.pexels.com${endpoint}`, {
    headers: { Authorization: process.env.PEXELS_API_KEY },
    params: { query, page: opts.page, per_page: 10, orientation: 'portrait' },
  });
  const items = opts.type === 'video' ? data.videos : data.photos;
  return items.map(p => ({
    id: p.id,
    source: 'pexels',
    type: opts.type,
    url: opts.type === 'video' ? (p.video_files?.[0]?.link || '') : p.src.large,
    description: p.alt || '(no description)',
    credit: opts.type === 'video' ? p.user?.name : p.photographer,
  }));
}

module.exports = async function discover(args) {
  const { query, opts } = parseArgs(args);

  if (!query) {
    console.log('Usage: tcf discover <query> [--source unsplash|pexels|both] [--type photo|video]');
    return;
  }

  console.log(`\nSearching "${query}" (source: ${opts.source}, type: ${opts.type})...\n`);

  let results = [];

  if (opts.source === 'unsplash' || opts.source === 'both') {
    if (!process.env.UNSPLASH_ACCESS_KEY) {
      console.warn('⚠  UNSPLASH_ACCESS_KEY not set, skipping Unsplash');
    } else {
      const photos = await searchUnsplash(query, opts);
      results = results.concat(photos);
    }
  }

  if (opts.source === 'pexels' || opts.source === 'both') {
    if (!process.env.PEXELS_API_KEY) {
      console.warn('⚠  PEXELS_API_KEY not set, skipping Pexels');
    } else {
      const items = await searchPexels(query, opts);
      results = results.concat(items);
    }
  }

  if (!results.length) {
    console.log('No results found.');
    return;
  }

  results.forEach((item, i) => {
    console.log(`[${i + 1}] ${item.source.toUpperCase()} ${item.type.toUpperCase()}`);
    console.log(`    ${item.description.slice(0, 80)}`);
    console.log(`    Credit: ${item.credit}`);
    console.log(`    URL: ${item.url}\n`);
  });

  console.log(`Found ${results.length} results.`);
};
