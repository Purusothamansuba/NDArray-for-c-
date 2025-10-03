#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>

template<typename T>
class NDArray {
private:
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    // Compute linear index from multi-dimensional indices
    size_t compute_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size())
            throw std::out_of_range("Incorrect number of indices");
        size_t idx = 0;
        for (size_t i = 0; i < indices.size(); ++i)
            idx += strides[i] * indices[i];
        return idx;
    }

    // Update strides based on current shape
    void update_strides() {
        strides.resize(shape.size());
        if (shape.empty()) return;
        strides[shape.size()-1] = 1;
        for (int i = (int)shape.size()-2; i >= 0; --i)
            strides[i] = strides[i+1] * shape[i+1];
    }

    // Calculate total size from shape
    size_t total_size(const std::vector<size_t>& s) const {
        return std::accumulate(s.begin(), s.end(), size_t(1), std::multiplies<size_t>());
    }

    // Broadcast shapes to a common shape
    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, 
                                               const std::vector<size_t>& b) {
        size_t n = std::max(a.size(), b.size());
        std::vector<size_t> result(n, 1);
        
        for (size_t i = 0; i < n; ++i) {
            size_t a_dim = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
            size_t b_dim = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
            
            if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
                throw std::runtime_error("Shapes not broadcastable");
            
            result[i] = std::max(a_dim, b_dim);
        }
        return result;
    }

    // Compute index with broadcasting (optimized)
    size_t broadcast_index(const std::vector<size_t>& result_idx) const {
        size_t idx = 0;
        size_t offset = shape.size() < result_idx.size() ? result_idx.size() - shape.size() : 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            size_t dim_index = result_idx[i + offset];
            if (shape[i] == 1) dim_index = 0;
            idx += strides[i] * dim_index;
        }
        return idx;
    }

    // Optimized broadcasting: precompute stride info
    struct BroadcastInfo {
        std::vector<size_t> strides_a;
        std::vector<size_t> strides_b;
        std::vector<size_t> result_shape;
    };

    BroadcastInfo prepare_broadcast(const NDArray<T>& other) const {
        BroadcastInfo info;
        info.result_shape = broadcast_shape(this->shape, other.shape);
        size_t n = info.result_shape.size();
        
        info.strides_a.resize(n, 0);
        info.strides_b.resize(n, 0);
        
        for (size_t i = 0; i < n; ++i) {
            size_t a_dim = (i < n - shape.size()) ? 1 : shape[i - (n - shape.size())];
            size_t b_dim = (i < n - other.shape.size()) ? 1 : other.shape[i - (n - other.shape.size())];
            
            if (a_dim > 1 && i >= n - shape.size()) {
                info.strides_a[i] = strides[i - (n - shape.size())];
            }
            if (b_dim > 1 && i >= n - other.shape.size()) {
                info.strides_b[i] = other.strides[i - (n - other.shape.size())];
            }
        }
        return info;
    }

    // Generic element-wise operation with broadcasting
    template<typename Func>
    NDArray<T> elementwise(const NDArray<T>& other, Func f) const {
        auto info = prepare_broadcast(other);
        NDArray<T> result(info.result_shape);
        
        std::vector<size_t> idx(info.result_shape.size(), 0);
        for (size_t i = 0; i < result.data.size(); ++i) {
            size_t idx_a = 0, idx_b = 0;
            for (size_t j = 0; j < info.result_shape.size(); ++j) {
                idx_a += info.strides_a[j] * idx[j];
                idx_b += info.strides_b[j] * idx[j];
            }
            
            result.data[i] = f(data[idx_a], other.data[idx_b]);
            
            // Increment idx
            for (int j = (int)idx.size()-1; j >= 0; --j) {
                idx[j]++;
                if (idx[j] < info.result_shape[j]) break;
                idx[j] = 0;
            }
        }
        return result;
    }

    // Recursive helper for pretty printing
    void print_recursive(std::ostream& os, size_t dim, size_t& offset, int indent) const {
        if (dim == shape.size() - 1) {
            os << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i > 0) os << ", ";
                os << std::setw(8) << std::setprecision(4) << data[offset++];
            }
            os << "]";
        } else {
            os << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i > 0) {
                    os << ",\n" << std::string(indent + 1, ' ');
                }
                print_recursive(os, dim + 1, offset, indent + 1);
            }
            os << "]";
        }
    }

public:
    // Default constructor
    NDArray() = default;

    // Constructor with shape and optional initial value
    NDArray(const std::vector<size_t>& shape_, T init_val = T()) 
        : shape(shape_) 
    {
        data.resize(total_size(shape), init_val);
        update_strides();
    }

    // Copy constructor
    NDArray(const NDArray<T>& other) = default;

    // Move constructor
    NDArray(NDArray<T>&& other) noexcept
        : data(std::move(other.data))
        , shape(std::move(other.shape))
        , strides(std::move(other.strides)) {}

    // Copy assignment
    NDArray<T>& operator=(const NDArray<T>& other) = default;

    // Move assignment
    NDArray<T>& operator=(NDArray<T>&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            shape = std::move(other.shape);
            strides = std::move(other.strides);
        }
        return *this;
    }

    // Variadic operator() for Python-style indexing
    template<typename... Args>
    T& operator()(Args... args) {
        std::vector<size_t> indices{static_cast<size_t>(args)...};
        return data[compute_index(indices)];
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        std::vector<size_t> indices{static_cast<size_t>(args)...};
        return data[compute_index(indices)];
    }

    // Overload for vector indexing (used internally)
    T& operator[](const std::vector<size_t>& indices) {
        return data[compute_index(indices)];
    }

    const T& operator[](const std::vector<size_t>& indices) const {
        return data[compute_index(indices)];
    }

    // Get shape
    const std::vector<size_t>& get_shape() const { return shape; }

    // Get total number of elements
    size_t size() const { return data.size(); }

    // Reshape array
    void reshape(const std::vector<size_t>& new_shape) {
        if (total_size(new_shape) != data.size())
            throw std::runtime_error("Total size must remain unchanged");
        shape = new_shape;
        update_strides();
    }

    // Flatten to 1D array
    NDArray<T> flatten() const {
        NDArray<T> result({data.size()});
        result.data = data;
        return result;
    }

    // Transpose (reverse all dimensions)
    NDArray<T> transpose() const {
        std::vector<size_t> perm(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            perm[i] = shape.size() - 1 - i;
        }
        return transpose(perm);
    }

    // Transpose with custom permutation
    NDArray<T> transpose(const std::vector<size_t>& perm) const {
        if (perm.size() != shape.size())
            throw std::runtime_error("Permutation size must match number of dimensions");
        
        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            if (perm[i] >= shape.size())
                throw std::out_of_range("Invalid permutation index");
            new_shape[i] = shape[perm[i]];
        }
        
        NDArray<T> result(new_shape);
        std::vector<size_t> idx(shape.size(), 0);
        
        for (size_t i = 0; i < data.size(); ++i) {
            std::vector<size_t> new_idx(shape.size());
            for (size_t j = 0; j < shape.size(); ++j) {
                new_idx[j] = idx[perm[j]];
            }
            result[new_idx] = data[compute_index(idx)];
            
            // Increment idx
            for (int j = (int)idx.size()-1; j >= 0; --j) {
                idx[j]++;
                if (idx[j] < shape[j]) break;
                idx[j] = 0;
            }
        }
        return result;
    }

    // Swap two axes
    NDArray<T> swapaxes(size_t axis1, size_t axis2) const {
        if (axis1 >= shape.size() || axis2 >= shape.size())
            throw std::out_of_range("Axis out of range");
        
        std::vector<size_t> perm(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            perm[i] = i;
        }
        std::swap(perm[axis1], perm[axis2]);
        return transpose(perm);
    }

    // Simple slice (along first axis)
    NDArray<T> slice(size_t start, size_t end) const {
        if (shape.empty()) 
            throw std::runtime_error("Cannot slice scalar");
        if (start >= end || end > shape[0]) 
            throw std::out_of_range("Slice out of range");
        
        NDArray<T> result = *this;
        size_t slice_size = 1;
        for (size_t i = 1; i < shape.size(); ++i) 
            slice_size *= shape[i];
        
        result.data = std::vector<T>(data.begin() + start*slice_size, 
                                     data.begin() + end*slice_size);
        result.shape[0] = end - start;
        result.update_strides();
        return result;
    }

    // Multi-dimensional slice without step
    NDArray<T> slice(const std::vector<size_t>& start, 
                    const std::vector<size_t>& end) const {
        if (start.size() != shape.size() || end.size() != shape.size())
            throw std::runtime_error("Slice dimensions mismatch");
        
        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            if (end[i] <= start[i] || end[i] > shape[i])
                throw std::runtime_error("Slice indices out of range");
            new_shape[i] = end[i] - start[i];
        }
        
        NDArray<T> result(new_shape);
        std::vector<size_t> idx(new_shape.size(), 0);
        
        for (size_t i = 0; i < result.data.size(); ++i) {
            size_t original_idx = 0;
            for (size_t j = 0; j < new_shape.size(); ++j)
                original_idx += strides[j] * (idx[j] + start[j]);
            result.data[i] = data[original_idx];
            
            // Increment idx
            for (int j = (int)idx.size()-1; j >= 0; --j) {
                idx[j]++;
                if (idx[j] < new_shape[j]) break;
                idx[j] = 0;
            }
        }
        return result;
    }

    // Multi-dimensional slice with step
    NDArray<T> slice(const std::vector<size_t>& start,
                    const std::vector<size_t>& end,
                    const std::vector<size_t>& step) const {
        if (start.size() != shape.size() || end.size() != shape.size() || 
            step.size() != shape.size())
            throw std::runtime_error("Slice dimensions mismatch");

        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            if (end[i] <= start[i] || end[i] > shape[i] || step[i] == 0)
                throw std::runtime_error("Slice indices out of range");
            new_shape[i] = (end[i] - start[i] + step[i] - 1) / step[i];
        }

        NDArray<T> result(new_shape);
        std::vector<size_t> idx(new_shape.size(), 0);
        
        for (size_t i = 0; i < result.data.size(); ++i) {
            size_t original_idx = 0;
            for (size_t j = 0; j < new_shape.size(); ++j)
                original_idx += strides[j] * (start[j] + idx[j] * step[j]);
            result.data[i] = data[original_idx];

            // Increment idx
            for (int j = (int)idx.size()-1; j >= 0; --j) {
                idx[j]++;
                if (idx[j] < new_shape[j]) break;
                idx[j] = 0;
            }
        }
        return result;
    }

    // Slice all (copy)
    NDArray<T> slice_all() const {
        std::vector<size_t> start(shape.size(), 0);
        std::vector<size_t> end = shape;
        std::vector<size_t> step(shape.size(), 1);
        return slice(start, end, step);
    }

    // Broadcasted addition
    NDArray<T> add(const NDArray<T>& other) const {
        auto info = prepare_broadcast(other);
        NDArray<T> result(info.result_shape);
        
        std::vector<size_t> idx(info.result_shape.size(), 0);
        for (size_t i = 0; i < result.data.size(); ++i) {
            size_t idx_a = 0, idx_b = 0;
            for (size_t j = 0; j < info.result_shape.size(); ++j) {
                idx_a += info.strides_a[j] * idx[j];
                idx_b += info.strides_b[j] * idx[j];
            }
            result.data[i] = data[idx_a] + other.data[idx_b];
            
            // Increment idx
            for (int j = (int)idx.size() - 1; j >= 0; --j) {
                idx[j]++;
                if (idx[j] < info.result_shape[j]) break;
                idx[j] = 0;
            }
        }
        return result;
    }

    // Broadcasted multiplication
    NDArray<T> multiply(const NDArray<T>& other) const {
        auto info = prepare_broadcast(other);
        NDArray<T> result(info.result_shape);
        
        std::vector<size_t> idx(info.result_shape.size(), 0);
        for (size_t i = 0; i < result.data.size(); ++i) {
            size_t idx_a = 0, idx_b = 0;
            for (size_t j = 0; j < info.result_shape.size(); ++j) {
                idx_a += info.strides_a[j] * idx[j];
                idx_b += info.strides_b[j] * idx[j];
            }
            result.data[i] = data[idx_a] * other.data[idx_b];
            
            // Increment idx
            for (int j = (int)idx.size() - 1; j >= 0; --j) {
                idx[j]++;
                if (idx[j] < info.result_shape[j]) break;
                idx[j] = 0;
            }
        }
        return result;
    }

    // Element-wise operators with broadcasting
    NDArray<T> operator+(const NDArray<T>& other) const {
        return add(other);
    }

    NDArray<T> operator-(const NDArray<T>& other) const {
        return elementwise(other, [](T a, T b) { return a - b; });
    }

    NDArray<T> operator*(const NDArray<T>& other) const {
        return multiply(other);
    }

    NDArray<T> operator/(const NDArray<T>& other) const {
        return elementwise(other, [](T a, T b) { return a / b; });
    }

    // Scalar multiplication
    NDArray<T> operator*(const T& val) const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] * val;
        return result;
    }

    // Scalar addition
    NDArray<T> operator+(const T& val) const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] + val;
        return result;
    }

    // Mathematical functions (element-wise)
    NDArray<T> sin() const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::sin(data[i]);
        return result;
    }

    NDArray<T> cos() const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::cos(data[i]);
        return result;
    }

    NDArray<T> exp() const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::exp(data[i]);
        return result;
    }

    NDArray<T> log() const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::log(data[i]);
        return result;
    }

    NDArray<T> sqrt() const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::sqrt(data[i]);
        return result;
    }

    NDArray<T> pow(T exponent) const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::pow(data[i], exponent);
        return result;
    }

    NDArray<T> abs() const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = std::abs(data[i]);
        return result;
    }

    // Apply custom function element-wise
    template<typename Func>
    NDArray<T> apply(Func f) const {
        NDArray<T> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = f(data[i]);
        return result;
    }

    // Print flat array (simple)
    void print() const {
        for (auto& val : data)
            std::cout << val << " ";
        std::cout << "\n";
    }

    // Pretty print with structure
    void print_pretty() const {
        if (shape.empty()) {
            std::cout << "[]" << std::endl;
            return;
        }
        size_t offset = 0;
        print_recursive(std::cout, 0, offset, 0);
        std::cout << std::endl;
        
        std::cout << "shape: (";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << ")" << std::endl;
    }

    // Convert to string
    std::string to_string() const {
        std::ostringstream oss;
        if (shape.empty()) {
            oss << "[]";
            return oss.str();
        }
        size_t offset = 0;
        print_recursive(oss, 0, offset, 0);
        return oss.str();
    }

    // Convenience static factory functions
    static NDArray<T> zeros(const std::vector<size_t>& shape) {
        return NDArray<T>(shape, T(0));
    }

    static NDArray<T> ones(const std::vector<size_t>& shape) {
        return NDArray<T>(shape, T(1));
    }

    static NDArray<T> arange(T start, T end, T step = 1) {
        std::vector<T> v;
        for (T i = start; i < end; i += step) 
            v.push_back(i);
        NDArray<T> arr({v.size()});
        arr.data = v;
        arr.update_strides();
        return arr;
    }

    static NDArray<T> linspace(T start, T end, size_t num) {
        if (num == 0) return NDArray<T>({0});
        if (num == 1) {
            NDArray<T> arr({1});
            arr.data[0] = start;
            return arr;
        }
        
        NDArray<T> arr({num});
        T step = (end - start) / (num - 1);
        for (size_t i = 0; i < num; ++i) {
            arr.data[i] = start + i * step;
        }
        return arr;
    }

    static NDArray<T> full(const std::vector<size_t>& shape, T value) {
        return NDArray<T>(shape, value);
    }
};

#endif