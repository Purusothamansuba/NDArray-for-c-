# NDArray

A lightweight, header-only N-dimensional array for C++ (inspired by NumPy).

This repository contains a single header `NDArray.hpp` that implements a template-based N-dimensional array with basic features such as creation, indexing, reshape, transpose, broadcasting arithmetic, slicing, element-wise math functions, and convenience factories like `zeros`, `ones`, `arange`, `linspace`, and `full`.

---

## Highlights

* Header-only: single file `NDArray.hpp` — drop it into your project.
* Template-based: works with numeric types (`int`, `float`, `double`, etc.).
* Arbitrary dimensions via `std::vector<size_t>` shapes.
* Python-like indexing: `a(i, j, k, ...)`.
* NumPy-like broadcasting for element-wise arithmetic.
* Slicing (returns copies), reshape, transpose, swapaxes.
* Element-wise math: `sin`, `cos`, `exp`, `log`, `sqrt`, `pow`, `abs`.
* Convenience factories: `zeros`, `ones`, `arange`, `linspace`, `full`.
* Pretty printing with `print_pretty()`.

---

## Requirements

* C++11 or later.

---

## Quick start

1. Copy `NDArray.hpp` into your project directory.
2. Include it where needed:

```cpp
#include "NDArray.hpp"
#include <iostream>

int main() {
    NDArray<double> a({2, 3}); // 2x3 array of doubles initialized to 0

    // Assign with Python-style indexing
    a(0,0) = 1.0; a(0,1) = 2.0; a(0,2) = 3.0;
    a(1,0) = 4.0; a(1,1) = 5.0; a(1,2) = 6.0;

    a.print_pretty();

    NDArray<double> b({3}); // shape (3)
    b(0) = 10; b(1) = 20; b(2) = 30;

    NDArray<double> c = a + b; // broadcasting (2,3) + (3) -> (2,3)
    c.print_pretty();

    return 0;
}
```

Compile and run:

```bash
g++ -std=c++11 -O2 example.cpp -o example
./example
```

---

## API Summary

> All members belong to `template<typename T> class NDArray` (see header for full signatures).

### Constructors & assignment

* `NDArray()` — default
* `NDArray(const std::vector<size_t>& shape_, T init_val = T())` — create and initialize
* Copy & move constructors and assignment are supported

### Indexing

* `T& operator()(Args... args)` — variadic indexing
* `const T& operator()(Args... args) const`
* `T& operator[](const std::vector<size_t>& indices)`
* `const T& operator[](const std::vector<size_t>& indices) const`

### Info

* `const std::vector<size_t>& get_shape() const`
* `size_t size() const` — total elements

### Reshape & permutations

* `void reshape(const std::vector<size_t>& new_shape)`
* `NDArray<T> flatten() const`
* `NDArray<T> transpose() const` — reverse axes
* `NDArray<T> transpose(const std::vector<size_t>& perm) const`
* `NDArray<T> swapaxes(size_t axis1, size_t axis2) const`

### Slicing

* `NDArray<T> slice(size_t start, size_t end) const` — along first axis
* `NDArray<T> slice(const std::vector<size_t>& start, const std::vector<size_t>& end) const`
* `NDArray<T> slice(const std::vector<size_t>& start, const std::vector<size_t>& end, const std::vector<size_t>& step) const`
* `NDArray<T> slice_all() const` — copy of whole array

### Arithmetic & broadcasting

* `NDArray<T> add(const NDArray<T>& other) const`
* `NDArray<T> multiply(const NDArray<T>& other) const`
* Operators: `operator+`, `operator-`, `operator*`, `operator/` (element-wise with broadcasting)
* Scalar ops: `operator*(const T& val)`, `operator+(const T& val)`

### Element-wise math

* `sin()`, `cos()`, `exp()`, `log()`, `sqrt()`, `pow(T exponent)`, `abs()`
* `template<typename Func> NDArray<T> apply(Func f) const`

### Utilities

* `void print() const` — flat print
* `void print_pretty() const` — structured nested print
* `std::string to_string() const`

### Factories

* `static NDArray<T> zeros(const std::vector<size_t>& shape)`
* `static NDArray<T> ones(const std::vector<size_t>& shape)`
* `static NDArray<T> arange(T start, T end, T step = 1)` — 1D array
* `static NDArray<T> linspace(T start, T end, size_t num)`
* `static NDArray<T> full(const std::vector<size_t>& shape, T value)`

---

## Limitations & Notes

* `T` should support required arithmetic and `std::` math functions. Non-floating types may not work correctly for some functions.
* Broadcasting follows NumPy-like rules. Incompatible shapes throw `std::runtime_error("Shapes not broadcastable")`.
* Index bounds: the implementation validates the number of indices but does not check each index for range individually — out-of-range indices may trigger undefined behavior or exceptions depending on the environment. Consider adding explicit bounds checks if you need safety.
* Slicing returns a copy (not a view). This simplifies the implementation but can increase memory usage.
* The implementation is focused on clarity and correctness rather than maximum performance (no SIMD, no multi-threading or BLAS integration).

---

## Examples

### Broadcasting & math

```cpp
#include "NDArray.hpp"
#include <iostream>

int main() {
    NDArray<double> a = NDArray<double>::full({2,2}, 1.5);
    NDArray<double> v = NDArray<double>::arange(0.0, 2.0); // [0,1]

    NDArray<double> sum = a + v; // broadcast

    std::cout << "a:\n"; a.print_pretty();
    std::cout << "v:\n"; v.print_pretty();
    std::cout << "a + v:\n"; sum.print_pretty();

    auto r = (sum * 2.0).exp();
    r.print_pretty();

    return 0;
}
```

---

## Tests

Create a `test_ndarray.cpp` with simple assertions or prints to verify:

* Shapes and sizes
* Indexing correctness
* Broadcasting behavior
* Reshape/transpose correctness

Compile:

```bash
g++ -std=c++11 -O2 test_ndarray.cpp -o test_ndarray
./test_ndarray
```

---

## Contributing / TODO

Suggestions for improvements:

* Add view-based slicing (no-copy views)
* Add in-place operators `+=`, `*=` with broadcasting
* Provide iterators for range-based loops
* Optimize loops for cache locality and performance
* Add reduction ops (`sum`, `mean`, `max`, `min`)
* Add configurable bounds-checking mode
* Add unit tests (Catch2 / GoogleTest)

If you make improvements, consider opening a PR or sharing benchmarks.

---

## License

Recommended: MIT License

```
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Footer

NDArray — Tiny header-only N-dimensional array (C++11+). If you want, I can also:

* generate a ready-to-commit `README.md` file you can download,
* create a short `example.cpp` and `test_ndarray.cpp` files,
* or add a `CMakeLists.txt` for easy building and testing.
