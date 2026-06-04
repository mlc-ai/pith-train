"""
Build tokenized corpus.

Tokenizes a directory of raw text files (JSONL, optionally zstd-compressed) into
a packed NumPy format suitable for language model training.

Parallelism is two-level: launch() spawns several leader processes, each owning
a pool of workers. Leaders pull files from a shared queue (largest first for
load balancing); workers within a pool tokenize individual documents. Splitting
by file first keeps queue contention low (a handful of leaders instead of
hundreds of workers competing for tasks) and overlaps I/O across files.

Each input file produces a single .bin file with two concatenated NumPy ararys:
  - tokens -- flat array of token IDs (smallest uint dtype that fits)
  - splits -- cumulative document lengths so boundaries can be recovered

A .lock sentinel makes the process resumable: files whose .bin output exists
without a leftover lock are skipped on restart.
"""

import atexit
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing import Pool, Process, Queue
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zstandard as zstd
from transformers import AutoTokenizer

from pithtrain.config import SlottedDefault
from pithtrain.modules.logging import LoggingCfg, LoggingCtx, StdoutLogger, logging_context


@dataclass(init=False, slots=True)
class TokenizeCorpusCfg(SlottedDefault):
    """User-provided settings for tokenize_corpus."""

    tokenizer_name: str
    """Hugging Face tokenizer model name."""

    source_path: Path
    """Directory containing raw text files to tokenize."""

    output_path: Path
    """Directory where tokenized output will be saved."""

    num_workers: int = field(default_factory=lambda: max(os.cpu_count() - 1, 1))
    """Number of parallel tokenization workers."""

    logging: LoggingCfg = field(default_factory=LoggingCfg)
    """Logging configuration."""


@dataclass(init=False, slots=True)
class TokenizeCorpusCtx(SlottedDefault):
    """Runtime context for tokenize_corpus."""

    logging: LoggingCtx = field(default_factory=LoggingCtx)
    """Active logging context."""


def read_file(path: Path):
    """Yield text from a file based on its extension."""
    match "".join(path.suffixes):
        case ".jsonl.zst" | ".jsonl.zstd":
            with zstd.open(path, mode="rt") as f:
                for line in f:
                    data = json.loads(line)
                    yield data["text"]
        case ".jsonl":
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    yield data["text"]
        case suffix:
            raise ValueError("unsupported format: %s" % suffix)


class Worker:
    """
    Worker for turning raw text into tokens. State is stored as class attributes
    because multiprocessing.Pool workers share the class, not an instance. At the
    end of each document, we append EOS so the model can learn the boundaries.
    """

    tokenizer: AutoTokenizer
    bestdtype: np.unsignedinteger

    def __init__(self, tokenizer_name: str) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        Worker.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        assert hasattr(Worker.tokenizer, "vocab_size")
        assert isinstance(Worker.tokenizer.vocab_size, int)
        assert hasattr(Worker.tokenizer, "eos_token_id")
        assert isinstance(Worker.tokenizer.eos_token_id, int)
        # Determine the smallest possible dtype for the token IDs.
        vocab_size = Worker.tokenizer.vocab_size
        Worker.bestdtype = np.uint64
        if vocab_size < np.iinfo(np.uint32).max:
            Worker.bestdtype = np.uint32
        if vocab_size < np.iinfo(np.uint16).max:
            Worker.bestdtype = np.uint16
        if vocab_size < np.iinfo(np.uint8).max:
            Worker.bestdtype = np.uint8
        # We are building the dataset so length limit does not apply.
        Worker.tokenizer.model_max_length = np.iinfo(np.uint64).max

    @staticmethod
    def encode(text: str) -> Tuple[np.ndarray, int]:
        encoded = Worker.tokenizer.encode(text)
        encoded.append(Worker.tokenizer.eos_token_id)
        return np.array(encoded, dtype=Worker.bestdtype), len(text)


class Writer:
    """
    Writer for saving tokenized sequences as a single packed .bin file.
    The file contains two concatenated .npy arrays: tokens then splits.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tokens: List[np.ndarray] = []
        self.offset: int = 0
        self.splits: List[int] = []

    def append(self, tokens: np.ndarray) -> None:
        self.tokens.append(tokens)
        self.offset += tokens.shape[0]
        self.splits.append(self.offset)

    def flush(self) -> None:
        tokens = np.concatenate(self.tokens, axis=0)
        splits = np.array(self.splits, dtype=np.uint64)
        with open(self.path, "wb") as f:
            np.save(f, tokens)
            np.save(f, splits)
        self.tokens.clear()
        self.splits.clear()


def leader(queue: Queue, psize: int, cfg: TokenizeCorpusCfg, npath: int):
    """
    Leader for a file group. Pulls files from the shared queue and tokenizes
    each file using a dedicated pool of wf workers.
    """
    stdout = StdoutLogger("pithtrain", logging.INFO)
    pool = Pool(psize, Worker, (cfg.tokenizer_name,))
    atexit.register(pool.terminate)
    try:
        while True:
            # Get the next file to tokenize.
            item = queue.get()
            if item is None:
                break
            i, path = item
            assert isinstance(i, int) and isinstance(path, Path)
            dest = path.relative_to(cfg.source_path)
            dest = Path(cfg.output_path, dest.parent, dest.name + ".bin")
            lock = Path(str(dest) + ".lock")
            if not lock.exists() and dest.exists():
                stdout.info("[%d/%d] %s | skip" % (i, npath, path.name))
                continue
            # Tokenize each document in the file.
            dest.parent.mkdir(parents=True, exist_ok=True)
            lock.touch()
            writer, cnt = Writer(dest), 0
            t0, prv = time.time(), time.time()
            iterable = pool.imap_unordered(Worker.encode, read_file(path), chunksize=64)
            for j, (tokens, nbytes) in enumerate(iterable, 1):
                writer.append(tokens)
                cnt, nxt = cnt + nbytes, time.time()
                # Report the throughput every three seconds.
                if nxt - prv > 3.0:
                    prv, throughput = nxt, cnt / (nxt - t0) / 1e6
                    stdout.info(
                        "[%d/%d] %s | %d records, %.1f MB/s" % (i, npath, path.name, j, throughput)
                    )
            # Remove the lock to mark completion after flushing.
            writer.flush()
            lock.unlink()
    finally:
        pool.terminate()


def launch(cfg: TokenizeCorpusCfg):
    """Launch the tokenization process."""
    with ExitStack() as stack:
        ctx = TokenizeCorpusCtx()
        stack.enter_context(logging_context(cfg, ctx))
        stdout = ctx.logging.stdout
        stdout.info("launch(cfg=%s)" % cfg)
        # Get all the files to tokenize, with hidden folders ignored.
        files: List[Path] = []
        for f in cfg.source_path.rglob("*"):
            if not f.is_file():
                continue
            if any(p.startswith(".") for p in f.relative_to(cfg.source_path).parts):
                continue
            files.append(f)
        # Assign psize workers per pool, with at least 1 pool.
        psize = 24
        npool = max(cfg.num_workers // psize, 1)
        psize = cfg.num_workers // npool
        # Feed files into a shared queue; large files first for load balancing.
        files.sort(key=lambda f: f.stat().st_size, reverse=True)
        mp = multiprocessing.get_context("spawn")
        queue = mp.Queue()
        for i, path in enumerate(files, 1):
            queue.put((i, path))
        for _ in range(npool):
            queue.put(None)
        # Cache the tokenizer upfront to avoid ratelimit with parallel workers.
        tokenizer_path = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, tokenizer_path, ignore_errors=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        tokenizer.save_pretrained(tokenizer_path)
        cfg.tokenizer_name = tokenizer_path
        leaders: List[Process] = []
        atexit.register(lambda: [p.terminate() for p in leaders])
        for _ in range(npool):
            p = mp.Process(target=leader, args=(queue, psize, cfg, len(files)))
            p.start()
            leaders.append(p)
        for p in leaders:
            p.join()
