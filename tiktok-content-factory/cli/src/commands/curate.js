const fs = require('fs').promises;
const path = require('path');
const { randomUUID } = require('crypto');

const STORAGE = path.join(process.env.STORAGE_PATH || './data', 'collections.json');

async function load() {
  try {
    return JSON.parse(await fs.readFile(STORAGE, 'utf8'));
  } catch {
    return [];
  }
}

async function save(collections) {
  await fs.mkdir(path.dirname(STORAGE), { recursive: true });
  await fs.writeFile(STORAGE, JSON.stringify(collections, null, 2));
}

function parseArgs(args) {
  const opts = {};
  const positional = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--vibe') opts.vibe = args[++i];
    else positional.push(args[i]);
  }
  return { positional, opts };
}

module.exports = async function curate(args) {
  const { positional, opts } = parseArgs(args);
  const [subcommand, ...rest] = positional;

  if (!subcommand || subcommand === 'list') {
    const collections = await load();
    if (!collections.length) {
      console.log('No collections yet. Create one with: tcf curate create <name>');
      return;
    }
    console.log(`\nCollections (${collections.length}):\n`);
    collections.forEach(c => {
      const vibe = c.vibe ? ` [${c.vibe}]` : '';
      console.log(`  • ${c.name}${vibe} — ${c.items.length} items  (${c.id})`);
    });
    return;
  }

  if (subcommand === 'create') {
    const name = rest.join(' ');
    if (!name) {
      console.log('Usage: tcf curate create <name> [--vibe live-it|intense]');
      return;
    }
    const collections = await load();
    const collection = {
      id: randomUUID(),
      name,
      vibe: opts.vibe || null,
      items: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    collections.push(collection);
    await save(collections);
    console.log(`\nCreated collection "${name}" (id: ${collection.id})`);
    if (opts.vibe) console.log(`Vibe: ${opts.vibe}`);
    return;
  }

  if (subcommand === 'delete') {
    const id = rest[0];
    if (!id) { console.log('Usage: tcf curate delete <id>'); return; }
    const collections = await load();
    const filtered = collections.filter(c => c.id !== id);
    if (filtered.length === collections.length) {
      console.log(`Collection not found: ${id}`);
      return;
    }
    await save(filtered);
    console.log(`Deleted collection ${id}`);
    return;
  }

  console.log('Usage: tcf curate [list|create|delete]');
};
