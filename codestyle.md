# Python Code Style Guide

> Inspired by [tinygrad](https://github.com/tinygrad/tinygrad) coding practices

## Philosophy

This guide follows **systems Python** principles: performance-conscious, explicit, and minimal abstraction. Code should be dense but readable, functional over object-oriented, and optimized for maintainability at scale.

**Core Tenets:**
- Dense > verbose
- Functions > classes > methods
- Explicit > implicit
- Fail fast > defensive programming
- Single source of truth > duplication

---

## 1. Import Style

### Dense, Flat Organization

**Do:** Comma-separate standard library imports on single lines
```python
import time, math, itertools, functools, struct, sys, inspect, pathlib
from typing import Callable, ClassVar, Sequence, cast, Literal, Generic
```

**Don't:** One import per line (wastes vertical space)
```python
import time
import math
import itertools
import functools
```

### Import Order
1. Standard library (comma-separated)
2. Third-party packages (comma-separated by package)
3. Local modules (explicit imports, no `import *`)

```python
import time, math, itertools
from typing import Callable, ClassVar
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import argfix, flatten, prod
```

---

## 2. Type Hints

### Modern Syntax (Python 3.10+)

**Always specify return types:**
```python
def get_shape(x) -> tuple[int, ...]:
    ...

def empty(*shape, device:str|tuple[str, ...]|None=None) -> Tensor:
    ...
```

**Use `|` for unions (not `Union[]`):**
```python
# Good
def process(x: int|float) -> str|None:
    ...

# Bad
from typing import Union, Optional
def process(x: Union[int, float]) -> Optional[str]:
    ...
```

**Complex nested types fully specified:**
```python
def canonicalize_device(device: str|tuple[str, ...]|None) -> str|tuple[str, ...]:
    ...
```

---

## 3. Naming Conventions

### Underscore Prefix = Internal

```python
# Module-level private helpers
def _fromnp(x) -> UOp: ...
def _frompy(x, dtype) -> UOp: ...

# Public module-level
def get_shape(x) -> tuple[int, ...]: ...

# Private instance methods
def _apply_uop(self, fxn): ...

# Public instance methods
def realize(self): ...
```

**Rules:**
- `_function`: Internal implementation, not public API
- `function`: Public API
- **Never use `__double_underscore__`** (except magic methods like `__init__`)
- Descriptive names > abbreviations (`_apply_broadcasted_uop`, not `_apply_bcast`)

---

## 4. Code Organization

### Functions > Classes

**Prefer module-level functions for stateless operations:**
```python
# Good: Pure function
def get_shape(x) -> tuple[int, ...]:
    if not hasattr(x, "__len__"): return ()
    return (len(x),) + get_shape(x[0])

# Bad: Unnecessary class
class ShapeUtils:
    @staticmethod
    def get_shape(x) -> tuple[int, ...]:
        ...
```

**Classes only for stateful objects:**
```python
class Tensor:
    """Represents a multi-dimensional array with state."""
    __slots__ = "uop", "requires_grad", "grad"

    def __init__(self, data, device=None):
        self.uop = data  # State
        self.requires_grad = False
```

### No Utility Classes

**Don't create:**
- `TensorUtils`
- `Helpers`
- `Manager` / `Handler` / `Provider` classes

**Instead:** Use module-level functions

---

## 5. Memory Efficiency

### Use `__slots__` for High-Volume Objects

```python
class Tensor:
    __slots__ = "uop", "requires_grad", "grad"

    def __init__(self, data):
        self.uop = data
        self.requires_grad = False
        self.grad = None
```

**Benefits:**
- No `__dict__` → saves memory
- Faster attribute access
- Prevents typos (AttributeError on wrong name)

### Delete Over Modify for In-Place Operations

```python
# Prefer creating new objects over in-place modification
def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned data, don't modify input."""
    return df.filter(pl.col("price") > 0)

# Not: def clean_data(df): df.drop(...); return df
```

**For pandas:** Use `.copy()` explicitly when mutating to avoid SettingWithCopyWarning

---

## 6. Error Handling

### Assertions for Invariants

**Fail fast, don't recover:**
```python
def assign(self, x):
    assert self.shape == x.shape, f"shape mismatch {self.shape} != {x.shape}"
    assert self.device == x.device, f"device mismatch {self.device} != {x.device}"
    assert self.dtype == x.dtype, f"dtype mismatch {self.dtype} != {x.dtype}"
    return self.replace(x)
```

**Rules:**
- Use assertions for internal invariants (not user input validation)
- Include f-strings in assertions for debugging
- No `try/except` around assertions
- Let the program crash on violations

**Don't use assertions for:**
- User input validation (use `raise ValueError`)
- External API errors (use explicit exceptions)

**Minimal try/except - Only for converting external errors:**
```python
# ✅ Good: Assertions for invariants
def process(df: pl.DataFrame) -> pl.DataFrame:
    assert len(df) > 0, f"empty dataframe"
    assert "price" in df.columns, f"missing price column"
    return df.filter(pl.col("price") > 0)

# ❌ Bad: Defensive try/except
def bad_example(path: str) -> pl.DataFrame:
    try:
        df = pl.read_csv(path)
    except:
        return pl.DataFrame()  # Silent failure - bad!

# ✅ Acceptable: Convert external errors to domain errors (still fail!)
def load_data(path: str) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except FileNotFoundError as e:
        raise ValueError(f"Data file missing: {path}") from e
    # Let other errors crash (encoding issues, schema problems, etc.)
```

---

## 7. Documentation

### Public API Only

**Document:**
```python
def empty(*shape, device:str|None=None, dtype:DTypeLike|None=None) -> Tensor:
    """
    Creates an empty tensor with the given shape.

    Example:
        >>> t = Tensor.empty(2, 3)
        >>> print(t.shape)
        (2, 3)
    """
    ...
```

**Don't document:**
```python
def _fromnp(x: 'np.ndarray') -> UOp:
    # No docstring - private helper, code is self-documenting
    ret = UOp.new_buffer("NPY", x.size, _from_np_dtype(x.dtype))
    ret.buffer.allocate(x)
    return ret.reshape(x.shape)
```

---

## 8. Properties and Methods

### Properties for Delegation and Computed Values

```python
@property
def device(self) -> str|tuple[str, ...]:
    return self.uop.device

@property
def shape(self) -> tuple[sint, ...]:
    return self.uop.shape

@property
def dtype(self) -> DType:
    return self.uop.dtype
```

**Use properties when:**
- Delegating to internal state
- Computing cheap read-only values (O(1) operations)
- Providing clean API for attribute access

**Don't use properties when:**
- Expensive computation (use explicit method)
- Side effects (use explicit method)
- Setting values (use explicit methods or `property.setter`)

**Example - Don't hide expensive operations:**
```python
# ❌ Bad: Expensive computation as property
@property
def correlation_matrix(self) -> np.ndarray:
    return np.corrcoef(self.data)  # O(n²) - looks free, isn't

# ✅ Good: Explicit method signals cost
def compute_correlation_matrix(self) -> np.ndarray:
    return np.corrcoef(self.data)
```

---

## 9. State Management

### Context Managers for Scoped State

```python
class Tensor:
    training: ClassVar[bool] = False

    class train(ContextDecorator):
        def __init__(self, mode: bool = True):
            self.mode = mode

        def __enter__(self):
            self.prev, Tensor.training = Tensor.training, self.mode

        def __exit__(self, exc_type, exc_value, traceback):
            Tensor.training = self.prev

# Usage
with Tensor.train():
    # training mode
    ...
# restored to previous mode
```

**Always restore state on exit**

### Minimal Global State

```python
# Only when truly necessary
all_tensors: dict[weakref.ref[Tensor], None] = {}

class Tensor:
    training: ClassVar[bool] = False  # Class-level shared state
```

**Prefer:**
- Passing state explicitly as arguments
- Class variables (`ClassVar`) over instance variables for shared state
- Module-level only when truly global

---

## 10. Functional Patterns

### Higher-Order Functions > Design Patterns

```python
def _apply_uop(self, fxn: Callable, *x: Tensor) -> Tensor:
    new_uop = fxn(*[t.uop for t in (self,)+x])
    return Tensor(new_uop, device=new_uop.device)

def _apply_broadcasted_uop(self, fxn: Callable, x: Tensor|ConstType) -> Tensor:
    lhs, rhs = self._broadcasted(x)
    return lhs._apply_uop(fxn, rhs)
```

**Pass functions as arguments, not inheritance hierarchies**

### Type Dispatch with isinstance Chains

```python
def __init__(self, data, ...):
    if isinstance(data, UOp):
        ...
    elif data is None:
        data = UOp.const(...)
    elif isinstance(data, get_args(ConstType)):
        data = UOp.const(...)
    elif isinstance(data, bytes):
        data = _frompy(data, dtypes.uint8)
    elif isinstance(data, (list, tuple)):
        data = _frompy(data, dtype)
    # Convert all inputs to canonical form (UOp)
```

**Converge all paths to single representation**

---

## 11. Code Clarity

### Delete Transformed Variables

```python
def __init__(self, data, device=None, dtype=None):
    _dtype = to_dtype(dtype) if dtype is not None else None
    _device = canonicalize_device(device)
    del device, dtype  # Prevent accidental use of wrong variable

    # Use _dtype and _device from here on
    ...
```

**Prevents bugs from using pre-transformation variables**

**For data operations (Polars/DuckDB/Pandas):**
```python
# Polars: Lazy evaluation with explicit transformation
def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    clean = (df
        .filter(pl.col("price") > 0)
        .with_columns(pl.col("timestamp").cast(pl.Datetime))
    )
    del df  # Prevent using uncleaned version
    return clean

# DuckDB: Dense SQL operations
def aggregate_trades(path: str) -> pl.DataFrame:
    return duckdb.query("""
        SELECT symbol, date_trunc('day', timestamp) as day,
               avg(price) as avg_price, sum(volume) as total_vol
        FROM read_csv(?)
        WHERE price > 0
        GROUP BY symbol, day
    """, params=[path]).pl()

# Pandas: Explicit .copy() when mutating
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()  # Explicit copy prevents SettingWithCopyWarning
    result["returns"] = result["price"].pct_change()
    return result
```

**Prefer Polars (lazy evaluation) or DuckDB (SQL) for data operations. Use pandas when ecosystem requires it.**

### Inline Comments Explain "Why"

```python
# Python has a non moving GC, so this should be okay
def __hash__(self):
    return id(self)

# NOTE: this can be in three states. False and None: no gradient, True: gradient
self.requires_grad: bool|None = requires_grad
```

**Comment decisions and rationale, not what the code does**

---

## 12. Class Design

### Magic Methods

```python
def __repr__(self):
    return f"<Tensor {self.uop.device} {self.shape} {self.dtype}>"

def __hash__(self):
    return id(self)  # Python has non-moving GC

def __bool__(self):
    raise TypeError("__bool__ on Tensor is not defined")

def __len__(self):
    if not self.shape:
        raise TypeError("len() of a 0-d tensor")
    return self.shape[0]
```

**Explicitly disable misleading operations** (like `__bool__`)

---

## Best Practices Checklist

### Code Organization
- [ ] Module-level functions for stateless operations
- [ ] Classes only for stateful objects
- [ ] No utility classes (`*Utils`, `*Helper`)
- [ ] `__slots__` for high-volume objects

### Type Safety
- [ ] All function signatures have return type hints
- [ ] Use `|` for unions, not `Union[]`
- [ ] Complex types fully specified
- [ ] Forward references quoted

### Naming
- [ ] `_prefix` for internal/private
- [ ] No `__double_underscore__` (except magic methods)
- [ ] Descriptive names over abbreviations

### Documentation
- [ ] Docstrings on public API only
- [ ] Executable examples in docstrings
- [ ] Inline comments explain "why", not "what"

### Error Handling
- [ ] Assertions for invariants with f-string messages
- [ ] Fail fast, don't recover
- [ ] Explicit exceptions for user-facing errors

### Functional Style
- [ ] Higher-order functions over design patterns
- [ ] Pass functions as arguments
- [ ] Compose small functions into larger ones

### Memory
- [ ] `__slots__` for high-volume objects
- [ ] Dict-as-set for fast membership (`dict[T, None]`)
- [ ] Context managers restore state

### Style
- [ ] Dense imports (comma-separated)
- [ ] Delete transformed variables
- [ ] Properties for delegation/computed values (not expensive ops)
- [ ] Minimal global state

### Debugging
- [ ] Use `.inspect()` for Polars pipelines
- [ ] Use `tap()` helper for function pipelines
- [ ] Decorator pattern for repeated debugging
- [ ] Keep inspection as removable side effect

---

## Anti-Patterns to Avoid

❌ **Defensive try/except with silent failures**
```python
# Bad
try:
    result = self.compute()
except Exception as e:
    logger.error(f"Failed: {e}")
    return default_value  # Silent failure, hard to debug
```

✅ **Assertions for invariants, convert external errors**
```python
# Good
assert self.is_valid(), f"Invalid state: {self.state}"
result = self.compute()

# Acceptable: Convert external errors to domain errors
try:
    data = load_file(path)
except FileNotFoundError as e:
    raise ValueError(f"Missing file: {path}") from e
```

---

❌ **Verbose one-import-per-line**
```python
# Bad
import sys
import math
import time
```

✅ **Dense comma-separated imports**
```python
# Good
import sys, math, time
```

---

❌ **Utility classes**
```python
# Bad
class TensorUtils:
    @staticmethod
    def get_shape(x): ...
```

✅ **Module-level functions**
```python
# Good
def get_shape(x) -> tuple[int, ...]: ...
```

---

❌ **Inheritance hierarchies**
```python
# Bad
class Operation(ABC):
    @abstractmethod
    def apply(self, x): ...

class AddOperation(Operation):
    def apply(self, x): ...
```

✅ **Functions and composition**
```python
# Good
def apply_operation(op: Callable, x):
    return op(x)

add = lambda x, y: x + y
```

---

## Example: Applying This Style

```python
# Good: Tinygrad-style code
import math, functools
from typing import Callable

def _validate_shape(shape: tuple[int, ...]) -> None:
    assert all(x > 0 for x in shape), f"invalid shape: {shape}"

def compute_size(shape: tuple[int, ...]) -> int:
    """Returns the total number of elements."""
    _validate_shape(shape)
    return functools.reduce(lambda a, b: a * b, shape, 1)

class Array:
    __slots__ = "shape", "data"

    def __init__(self, shape: tuple[int, ...]):
        _validate_shape(shape)
        self.shape = shape
        self.data = [0] * compute_size(shape)

    @property
    def size(self) -> int:
        return len(self.data)
```

**Key characteristics:**
- Dense imports
- Type hints throughout
- Module-level helper `_validate_shape`
- `__slots__` for efficiency
- Assertions for invariants
- Property for computed value
- Function composition (`compute_size`)

---

## 13. Debugging Data Pipelines

### Inspection Patterns

**For Polars - Use built-in `.inspect()`:**
```python
result = (df
    .filter(pl.col("price") > 0)
    # .inspect()  # Uncomment to see data here
    .with_columns(pl.col("price").log().alias("log_price"))
    .filter(pl.col("volume") > 100)
)
```

**For function pipelines - Simple `tap()` helper:**
```python
def tap(x, msg: str = ""):
    """Debug helper: print data shape/schema and pass through."""
    if hasattr(x, 'schema'): print(f"{msg}: {x.schema}")
    elif hasattr(x, 'shape'): print(f"{msg}: {x.shape}")
    else: print(f"{msg}: {len(x)} items")
    return x

# Use anywhere, comment out when done
result = tap(process(tap(load(), "loaded")), "processed")
```

**For repeated debugging - Decorator with global flag:**
```python
class Debug:
    enabled = False

def traced(func):
    """Trace function inputs/outputs when debugging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if Debug.enabled:
            print(f"{func.__name__}: {result.shape if hasattr(result, 'shape') else type(result)}")
        return result
    return wrapper

@traced
def remove_nulls(df): ...

@traced
def filter_prices(df): ...

# Toggle debugging globally
Debug.enabled = True
result = filter_prices(remove_nulls(df))
Debug.enabled = False
```

**Key principle:** Don't build observability into pipeline structure. Keep it as a side effect you can easily add/remove.

---

## When to Break These Rules

1. **External API compatibility** - Match the style of the library you're extending
2. **Team conventions** - Consistency > personal preference
3. **Performance profiling** - If measurements show a different approach is faster
4. **Readability** - If breaking a rule genuinely improves clarity for your domain

**But:** Document why you're deviating from the standard.
