#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <limits>

template<typename T, std::size_t Alignment = 16>
class aligned_allocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using pointer = T*;
    using const_pointer = const T*;

    using reference = T&;
    using const_reference = const T&;

    aligned_allocator() noexcept = default;

    template<typename U>
    constexpr aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    ~aligned_allocator() noexcept = default;

    pointer address(reference r) const noexcept { return &r; }
    const_pointer address(const_reference r) const noexcept { return &r; }

    [[nodiscard]] pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_alloc();
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(n * sizeof(value_type), Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(value_type)) != 0)
            throw std::bad_alloc();
#endif
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
#if defined(_WIN32)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    void construct(pointer p, const value_type& value) {
        new(p) value_type(value);
    }

    void destroy(pointer p) { p->~value_type(); }

    size_type max_size() const noexcept {
        return size_type(-1) / sizeof(value_type);
    }

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    bool operator==(const aligned_allocator& other) const noexcept { return true; }
    bool operator!=(const aligned_allocator& other) const noexcept { return !(*this == other); }
};

#endif // ALIGNED_ALLOCATOR_H

//#ifndef ALIGNED_ALLOCATOR_H
//#define ALIGNED_ALLOCATOR_H
//
///**
// * adapted from
// * https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
// */
//#include <cstdlib>
//#include <cstddef>
//#include <cstdio>
//
//template<typename T, std::size_t N = 16>
//class alligned_allocator {
//public:
//  typedef T value_type;
//  typedef std::size_t size_type;
//  typedef std::ptrdiff_t difference_type;
//
//  typedef T *pointer;
//  typedef const T *const_pointer;
//
//  typedef T &reference;
//  typedef const T &const_reference;
//
//public:
//  inline alligned_allocator() noexcept = default;
//
//  template<typename T2>
//  inline explicit alligned_allocator(
//          const alligned_allocator<T2, N> &) noexcept {}
//
//  inline ~alligned_allocator() noexcept = default;
//
//  inline pointer adress(reference r) { return &r; }
//
//  inline const_pointer adress(const_reference r) const { return &r; }
//
//  inline pointer allocate(size_type n) {
//#if defined(_WIN32)
//    return (pointer) _aligned_malloc(n * sizeof(value_type), N);
//#else
//    void *m = nullptr;
//    if (posix_memalign(&m, N, n * sizeof(value_type))) {
//      puts("cannot allocate that much memory!");
//      exit(1); // couldn't allocate enough memory
//    }
//    return (pointer)m;
//#endif
//  }
//
//  inline void deallocate(pointer p, size_type) {
//#if defined(_WIN32)
//    _aligned_free(p);
//#else
//    free(p);
//#endif
//  }
//
//  inline void construct(pointer p, const value_type &wert) {
//    new(p) value_type(wert);
//  }
//
//  inline void destroy(pointer p) { p->~value_type(); }
//
//  inline size_type max_size() const noexcept {
//    return size_type(-1) / sizeof(value_type);
//  }
//
//  template<typename T2>
//  struct rebind {
//    typedef alligned_allocator<T2, N> other;
//  };
//
//  bool operator!=(const alligned_allocator<T, N> &other) const {
//    return !(*this == other);
//  }
//
//  // Returns true if and only if storage allocated from *this
//  // can be deallocated from other, and vice versa.
//  // Always returns true for stateless allocators.
//  bool operator==(const alligned_allocator<T, N> &other) const { return true; }
//};
//
//#endif