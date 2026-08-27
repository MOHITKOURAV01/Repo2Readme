# Troubleshooting

## The run finished but the file was not written

The output path is now checked before any API call, so this should surface as
an immediate exit code 2 with a message naming the problem — a path that is a
directory, a parent that cannot be created, or a file that is not writable.

If a write fails for some other reason (a full disk, a filesystem going away
mid-run), the generated README is printed to stdout so the run is not wasted,
and the exit code is 1. Redirect it if you want to keep it:

```bash
repo2readme run --local . --output /mnt/full/README.md > README.md
```

An existing README is never truncated before the new one is complete: the write
goes to a temporary file and is renamed into place. Use `--backup` to keep a
`.bak` copy of what was replaced.

## "Missing API key" or repeated key prompts

An exported key is used before the saved one, so setting the variable for the
provider you are using is enough:

```bash
export GROQ_API_KEY="your_groq_api_key"
export GOOGLE_API_KEY="your_google_api_key"
```

`repo2readme providers` prints the variable each provider reads. If you are
still asked for a key, check that it is exported in the shell that runs
`repo2readme` and that the value is not empty — an unfilled `.env` line
(`GROQ_API_KEY=`) counts as no key at all.

Or reset the saved keys and re-enter them:

```bash
repo2readme reset
repo2readme run --local ./my-project
```

## "there is no terminal to prompt on"

The run needs a key it could not find, and stdin is not a terminal — a CI job,
a cron entry, or a piped command. Nothing can be asked for in that situation,
so the run stops immediately rather than blocking. Set the environment variable
named in the message, or run the tool once from a terminal to save the key.

## Command not found: repo2readme

Make sure the package installed correctly and your Python scripts directory is on your `PATH`:

```bash
pip install repo2readme
python -m repo2readme --help
```

## Token estimate looks too high / request is too large

Use `--dry-run` first to check exactly which files will be sent, then narrow things down with `--exclude` or `--max-file-size-kb`:

```bash
repo2readme run --local ./my-project --dry-run
repo2readme run --local ./my-project --exclude "tests/*" --max-file-size-kb 100
```

## Rate limit errors (HTTP 429) on large repositories

Transient failures — rate limits, timeouts, dropped connections and malformed
JSON responses — are retried automatically with exponential backoff. When the
provider sends a `Retry-After` header or a "try again in 6.7s" hint, that value
is used instead of the computed delay.

The defaults are 2 retries (3 attempts) starting at a 1 second delay. Tune them
with environment variables:

```bash
# free tiers with strict per-minute limits
export REPO2README_MAX_RETRIES=5
export REPO2README_RETRY_BASE_DELAY=2

# fail fast, no retries at all
export REPO2README_MAX_RETRIES=0
```

Lowering `--max-workers` also helps, since fewer parallel requests means fewer
rate limits to retry in the first place:

```bash
repo2readme run --local ./my-project --max-workers 2
```

Authentication failures, unsupported providers and context-length errors are
never retried — retrying cannot fix them, and failing immediately tells you
what is actually wrong.

## Generated README looks incomplete or low quality

The tool iterates internally until the reviewer agent scores the draft 8.5+ or a max iteration count is hit. If quality is still off, check that your repo's key files (entry points, config, core logic) aren't being filtered out — run with `--dry-run` to confirm they're in the "Files to be processed" list.

## Errors cloning a GitHub URL

Confirm the repository is public (or that you have access), and that the URL is a valid `https://github.com/<owner>/<repo>` link.

## Still stuck?

Open an issue: https://github.com/agsaru/Repo2Readme/issues

## Getting more detail out of a failing run

Run with `-v` (info) or `-vv` (debug), or capture a full debug log to a file:

```bash
repo2readme run --local ./my-project -vv
repo2readme run --local ./my-project --log-file repo2readme.log
```

The log file always records `DEBUG` regardless of console verbosity, so it is
the right thing to attach to a bug report along with `repo2readme --version`.
See [Logging and verbosity](./logging.md).
