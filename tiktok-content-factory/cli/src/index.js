#!/usr/bin/env node
require('dotenv').config();

const [,, command, ...args] = process.argv;

const COMMANDS = {
  discover: require('./commands/discover'),
  curate: require('./commands/curate'),
  edit: require('./commands/edit'),
};

function usage() {
  console.log(`
TikTok Content Factory CLI

Usage: tcf <command> [options]

Commands:
  discover <query>          Search photos/videos (--source unsplash|pexels|both, --type photo|video)
  curate list               List all collections
  curate create <name>      Create a new collection (--vibe live-it|intense)
  edit <imageUrl|path>      Apply vibe to an image (--vibe live-it|intense, --out ./output.jpg)

Examples:
  tcf discover "sunset beach" --source pexels --type video
  tcf curate create "Summer Vibes" --vibe live-it
  tcf edit https://example.com/img.jpg --vibe intense --out ./result.jpg
`);
}

if (!command || command === '--help' || command === '-h') {
  usage();
  process.exit(0);
}

const handler = COMMANDS[command];
if (!handler) {
  console.error(`Unknown command: ${command}`);
  usage();
  process.exit(1);
}

handler(args).catch(err => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
