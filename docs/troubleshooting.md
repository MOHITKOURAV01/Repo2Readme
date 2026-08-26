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

## The redirected README is missing square brackets, or wrapped mid-sentence

This is fixed. Without `--output`, the README used to be printed through the
console's markup renderer, which reads `[...]` as a style tag and renders to a
fixed width. Redirecting it therefore produced a file with the brackets deleted
and every long line broken in half:

```bash
repo2readme run --local . > README.md
```

The README is now written to stdout verbatim, so the redirect gives you exactly
what the model produced — table of contents links, `[!NOTE]` callouts, array
indices and all. The same applies to file paths that contain brackets, such as
a Next.js `src/[id]/page.tsx` route, which used to disappear from the
`--dry-run` listing.

The commentary moved with it. The token estimate, the progress bars, the
confirmation prompt and "Saved to ..." were also on stdout, so they ended up in
the redirected file too. Stdout now carries the product and stderr carries
everything the CLI says about the run, so you can redirect one and still watch
the other:

```bash
repo2readme run --local . > README.md      # progress still on the terminal
repo2readme run --local . 2> run.log       # or the other way round
```

`--dry-run` is the exception, and deliberately: its report *is* what the command
produces, so it stays on stdout and can be redirected as usual.

## "Missing API key" or repeated key prompts

Your keys may not be saved correctly. Try setting them as environment variables instead:

```bash
export GROQ_API_KEY="your_groq_api_key"
export GOOGLE_API_KEY="your_google_api_key"
```

Or reset and re-enter them:

```bash
repo2readme reset
repo2readme run --local ./my-project
```

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

## The README has a table of contents that links to nothing

The structural checks — table of contents anchors, placeholder images, a single
top-level heading — now run against every draft while the review loop is still
running, and what they find is handed to the next generation round as concrete
instructions ("the link `#configuration` matches no heading"). Between two
drafts the reviewer scored about the same, the one with fewer structural
problems is kept.

Anything still reported at the end is what the loop could not fix, or what it
never had a chance to: the loop stops at three iterations, and stops early if a
review fails. Run with `-v` to see the per-draft counts:

```bash
repo2readme run --local . -v
```

## Errors cloning a GitHub URL

Confirm the repository is public (or that you have access), and that the URL is a valid `https://github.com/<owner>/<repo>` link.

## A CI job goes green even though the run failed

This is fixed. Failures used to end with status `0`, so a step that regenerated
a README stayed green when the clone had failed. See
[Exit codes](./cli-reference.md#exit-codes) for what each status means now:
`1` is a run that could not be completed, `2` is a command line that was wrong,
and `0` really does mean the README was generated (or that you declined a
prompt).

## Re-running produces a different README from an unchanged repository

This is fixed. File summaries used to be collected in whichever order the
worker threads finished, and that list goes straight into the generation
prompt — so two runs over the same repository, with the same provider and model
and even a fully warm cache, asked the model a differently ordered question and
got a different answer back.

Summaries and the directory roll-up now come back in repository order,
independent of `--max-workers` and of how long any one file took. Re-running on
an unchanged repository should now be a no-op diff, so `repo2readme` can be used
to regenerate a checked-in README.

Remaining sources of variation are the model itself (most providers sample) and
anything that changed in the repository.

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
