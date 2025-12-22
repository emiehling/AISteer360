import pickle
import random
import struct
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from queue import Queue
from typing import Any


class ReplayBuffer(ABC):
    """Abstract base class for replay buffers.

    Implementations must support:
    - Adding examples (potentially with reservoir sampling)
    - Random sampling for batch construction
    - Persistence across training interruptions (optional)

    """

    @abstractmethod
    def __len__(self) -> int:
        """Return current number of stored examples."""
        pass

    @abstractmethod
    def add(self, example: dict[str, Any]) -> None:
        """Add a single example to the buffer."""
        pass

    def add_many(self, examples: Sequence[dict[str, Any]]) -> None:
        """Add multiple examples. Default implementation calls add() in loop."""
        for example in examples:
            self.add(example)

    @abstractmethod
    def sample(self, num_samples: int) -> list[dict[str, Any]]:
        """Sample num_samples examples uniformly at random."""
        pass

    def save_state(self, path: str | Path) -> None:
        """Optional: persist buffer state for resumption."""
        raise NotImplementedError("This buffer does not support persistence.")

    def load_state(self, path: str | Path) -> None:
        """Optional: restore buffer state from disk."""
        raise NotImplementedError("This buffer does not support persistence.")


class InMemoryReplayBuffer(ReplayBuffer):
    """Simple in-memory replay buffer with reservoir or FIFO replacement.

    Suitable for small experiments with models up to ~2B parameters.
    Memory usage: O(capacity * avg_example_size).

    Args:
        capacity: Maximum number of examples to store.
        reservoir: If True, use reservoir sampling (uniform over all seen examples).
            If False, use FIFO replacement (keeps most recent examples).

    Note:
        For large-scale continual pre-training (>2B parameters, >100K buffer size), consider using DiskReplayBuffer or
        the paper's Megatron-compatible implementation at https://github.com/chandar-lab/continual-pretraining
    """

    def __init__(self, capacity: int, reservoir: bool = True) -> None:
        self.capacity = int(capacity)
        self.reservoir = reservoir
        self._examples: list[dict[str, Any]] = []
        self._total_seen: int = 0

    def __len__(self) -> int:
        return len(self._examples)

    def add(self, example: dict[str, Any]) -> None:
        if self.capacity == 0:
            return

        self._total_seen += 1

        if len(self._examples) < self.capacity:
            self._examples.append(example)
            return

        if self.reservoir:
            # reservoir sampling: uniform probability over all seen examples
            replacement_idx = random.randint(0, self._total_seen - 1)
            if replacement_idx < self.capacity:
                self._examples[replacement_idx] = example
        else:
            # FIFO: always replace oldest
            self._examples.pop(0)
            self._examples.append(example)

    def sample(self, num_samples: int) -> list[dict[str, Any]]:
        if not self._examples or num_samples <= 0:
            return []
        num_samples = min(int(num_samples), len(self._examples))
        return random.sample(self._examples, num_samples)

    def save_state(self, path: str | Path) -> None:
        """Persist buffer to disk using pickle."""
        state = {
            "capacity": self.capacity,
            "reservoir": self.reservoir,
            "examples": self._examples,
            "total_seen": self._total_seen,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str | Path) -> None:
        """Restore buffer from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.capacity = state["capacity"]
        self.reservoir = state["reservoir"]
        self._examples = state["examples"]
        self._total_seen = state["total_seen"]


class DiskReplayBuffer(ReplayBuffer):
    """Disk-backed replay buffer with async prefetching.

    Stores tokenized examples on disk to support large buffer sizes without using up RAM. Uses a background thread to
    prefetch samples into a cache queue.

    This is a simplified implementation suitable for single-node HuggingFace training. For multi-node Megatron/NeoX
    training, use the paper's implementation: https://github.com/chandar-lab/continual-pretraining

    Args:
        capacity: Maximum number of examples to store.
        storage_dir: Directory for buffer files. If None, uses a temp directory.
        cache_size: Number of examples to prefetch into memory.
        reservoir: If True, use reservoir sampling; otherwise FIFO.

    Storage format:
        - index.bin: Fixed-size records mapping example_id -> (file_offset, length)
        - data.bin: Concatenated pickled examples
        - metadata.json: Buffer state for resumption
    """

    _HEADER_SIZE = 16  # 8 bytes offset + 8 bytes length per example

    def __init__(
            self,
            capacity: int,
            storage_dir: str | Path | None = None,
            cache_size: int = 1000,
            reservoir: bool = True,
    ) -> None:
        self.capacity = int(capacity)
        self.reservoir = reservoir
        self.cache_size = cache_size

        # storage setup
        if storage_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.storage_dir = Path(self._temp_dir.name)
        else:
            self._temp_dir = None
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self.storage_dir / "index.bin"
        self._data_path = self.storage_dir / "data.bin"

        # initialize files
        self._index_file = open(self._index_path, "w+b")
        self._data_file = open(self._data_path, "w+b")

        # state
        self._size = 0
        self._total_seen = 0
        self._data_offset = 0

        # prefetch infrastructure
        self._cache_queue: Queue[dict[str, Any]] = Queue(maxsize=cache_size)
        self._prefetch_thread: threading.Thread | None = None
        self._stop_prefetch = threading.Event()
        self._prefetch_lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    def __del__(self) -> None:
        self._stop_prefetch.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)
        self._index_file.close()
        self._data_file.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def add(self, example: dict[str, Any]) -> None:
        if self.capacity == 0:
            return

        self._total_seen += 1

        # serialize example
        data = pickle.dumps(example)
        data_len = len(data)

        with self._prefetch_lock:
            if self._size < self.capacity:
                # still filling: append
                slot = self._size
                self._size += 1
            elif self.reservoir:
                # reservoir sampling
                slot_candidate = random.randint(0, self._total_seen - 1)
                if slot_candidate >= self.capacity:
                    return  # don't add this example
                slot = slot_candidate
            else:
                # FIFO: overwrite oldest (slot 0)
                # For simplicity, overwrite slot (total_seen % capacity)
                slot = (self._total_seen - 1) % self.capacity

            # write
            self._data_file.seek(0, 2)  # Seek to end
            offset = self._data_file.tell()
            self._data_file.write(data)
            self._data_file.flush()

            # update
            self._index_file.seek(slot * self._HEADER_SIZE)
            self._index_file.write(struct.pack("<QQ", offset, data_len))
            self._index_file.flush()

    def _read_example(self, slot: int) -> dict[str, Any]:
        """Read a single example from disk by slot index."""
        self._index_file.seek(slot * self._HEADER_SIZE)
        offset, length = struct.unpack("<QQ", self._index_file.read(self._HEADER_SIZE))

        self._data_file.seek(offset)
        data = self._data_file.read(length)
        return pickle.loads(data)

    def sample(self, num_samples: int) -> list[dict[str, Any]]:
        if self._size == 0 or num_samples <= 0:
            return []

        num_samples = min(int(num_samples), self._size)
        indices = random.sample(range(self._size), num_samples)

        with self._prefetch_lock:
            return [self._read_example(idx) for idx in indices]

    def start_prefetch(self) -> None:
        """Start background prefetch thread."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return

        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def _prefetch_worker(self) -> None:
        """Background worker that keeps cache queue filled."""
        while not self._stop_prefetch.is_set():
            if self._cache_queue.full() or self._size == 0:
                self._stop_prefetch.wait(timeout=0.1)
                continue

            try:
                with self._prefetch_lock:
                    idx = random.randint(0, self._size - 1)
                    example = self._read_example(idx)
                self._cache_queue.put(example, timeout=0.1)
            except Exception:
                pass  # Queue full or other error, retry

    def sample_from_cache(self, num_samples: int) -> list[dict[str, Any]]:
        """Sample from prefetch cache (faster but may have duplicates)."""
        samples = []
        for _ in range(num_samples):
            if self._cache_queue.empty():
                break
            try:
                samples.append(self._cache_queue.get_nowait())
            except Exception:
                break

        # fall back to direct sampling
        remaining = num_samples - len(samples)
        if remaining > 0:
            samples.extend(self.sample(remaining))

        return samples

    def save_state(self, path: str | Path) -> None:
        """Save buffer metadata for resumption."""
        import json
        metadata = {
            "capacity": self.capacity,
            "reservoir": self.reservoir,
            "size": self._size,
            "total_seen": self._total_seen,
            "data_offset": self._data_offset,
            "storage_dir": str(self.storage_dir),
        }
        with open(path, "w") as f:
            json.dump(metadata, f)

    def load_state(self, path: str | Path) -> None:
        """Restore buffer metadata. Assumes data files are intact."""
        import json
        with open(path, "r") as f:
            metadata = json.load(f)
        self._size = metadata["size"]
        self._total_seen = metadata["total_seen"]
        self._data_offset = metadata["data_offset"]
