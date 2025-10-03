#include "NDArray.hpp"
#include <iostream>
#include <cmath>

int main() {
    // ----------------------------
    // 1. Creating Arrays
    // ----------------------------
    NDArray<double> a = NDArray<double>::arange(0, 10);   // 1D array: 0..9
    NDArray<double> b = NDArray<double>::ones({10});       // 1D array of ones
    NDArray<double> c = a + b;                             // Broadcasting addition

    std::cout << "Array a:" << std::endl;
    a.print_pretty();
    std::cout << "\nArray b:" << std::endl;
    b.print_pretty();
    std::cout << "\nArray c = a + b:" << std::endl;
    c.print_pretty();

    // ----------------------------
    // 2. Mathematical Functions
    // ----------------------------
    NDArray<double> d = c.sin();
    std::cout << "\nSin(c):" << std::endl;
    d.print_pretty();

    NDArray<double> e = c.exp();
    std::cout << "\nExp(c):" << std::endl;
    e.print_pretty();

    // ----------------------------
    // 3. Multi-dimensional Arrays
    // ----------------------------
    NDArray<int> mat({2, 3}, 5); // 2x3 array filled with 5
    std::cout << "\nMatrix mat (2x3 filled with 5):" << std::endl;
    mat.print_pretty();

    // 2D slicing
    NDArray<int> slice = mat.slice({0, 1}, {2, 3});
    std::cout << "\nSlice of mat (rows 0-2, cols 1-3):" << std::endl;
    slice.print_pretty();

    // Transpose
    NDArray<int> transposed = mat.transpose();
    std::cout << "\nTransposed mat:" << std::endl;
    transposed.print_pretty();

    // Swap axes
    NDArray<int> swapped = mat.swapaxes(0, 1);
    std::cout << "\nSwapped axes 0 and 1:" << std::endl;
    swapped.print_pretty();

    // ----------------------------
    // 4. Scalar operations
    // ----------------------------
    NDArray<double> f = c * 2.0 + 3.0;
    std::cout << "\nc * 2.0 + 3.0:" << std::endl;
    f.print_pretty();

    // ----------------------------
    // 5. linspace
    // ----------------------------
    NDArray<double> x = NDArray<double>::linspace(0.0, 1.0, 5);
    NDArray<double> y = NDArray<double>::full({5}, 2.0);
    NDArray<double> result = (x * 2.0) + y;

    std::cout << "\nLinspace x:" << std::endl;
    x.print_pretty();
    std::cout << "\nFull y:" << std::endl;
    y.print_pretty();
    std::cout << "\nResult = x*2 + y:" << std::endl;
    result.print_pretty();

    return 0;
}
