#ifndef ITERATORS_HH__
#define ITERATORS_HH__

template<typename _Iterator>
struct iterator_traits
{
	typedef typename _Iterator::value_type        value_type;
	typedef typename _Iterator::difference_type   difference_type;
	typedef typename _Iterator::pointer           pointer;
	typedef typename _Iterator::reference         reference;
};

/// Partial specialization for pointer types.
template<typename _Tp>
struct iterator_traits<_Tp*>
{
	typedef _Tp                         value_type;
	typedef ptrdiff_t                   difference_type;
	typedef _Tp*                        pointer;
	typedef _Tp&                        reference;
};

/// Partial specialization for const pointer types.
template<typename _Tp>
struct iterator_traits<const _Tp*>
{
	typedef _Tp                         value_type;
	typedef ptrdiff_t                   difference_type;
	typedef const _Tp*                  pointer;
	typedef const _Tp&                  reference;
};


template <typename iterator_T, typename T>
struct iterator_base_t
{
	typedef T self_type;
	typedef T value_type;
	typedef T& reference;
	typedef T* pointer;

	typedef std::forward_iterator_tag iterator_category;
	typedef int difference_type;

	iterator_base_t() = default;
	iterator_base_t(int index):fIndex(index){}

	__device__ __host__
	iterator_T operator+(int offset)
	{
		iterator_T next =  *static_cast<iterator_T*>(this);
		next.fIndex += offset;
		return next;
	}

	__device__ __host__
	iterator_T operator-(int offset)
	{
		iterator_T next =  *static_cast<iterator_T*>(this);
		next.fIndex -= offset;
		return next;
	}

	__device__ __host__
	iterator_T operator+=(int offset)
	{
		fIndex+=offset;
		return *static_cast<iterator_T*>(this);
	}

	__device__ __host__
	bool operator!=(iterator_T other)
	{
		return fIndex != other.fIndex;
	}

	__device__ __host__
	iterator_T operator++()
	{
		fIndex++;
		return *static_cast<iterator_T*>(this);
	}

	__device__ __host__
	iterator_T operator++(int junk)
	{
		iterator_T old = *static_cast<iterator_T*>(this);
		fIndex++;
		return old;
	}

	int fIndex = 0;
};

template <typename iterator_T, typename T>
struct iterator_t : public iterator_base_t<iterator_T, T>
{
	typedef iterator_base_t<iterator_T, T> base_iterator_t;
	iterator_t() = default;
	iterator_t(int index): base_iterator_t(index){}

	__device__ __host__
	T operator[](int index) const
	{
		return static_cast<const iterator_T&>(*this)(base_iterator_t::fIndex + index );
	}

	__device__ __host__
	T operator*() const
	{
		return (*this)[0];
	}

};


template <typename T>
struct counting_iterator_t : public iterator_t<counting_iterator_t<T>, T>
{
public:
	counting_iterator_t() = default;
	counting_iterator_t(int index) : iterator_t<counting_iterator_t<T>, T>(index) {}
	__device__ __host__
	T operator()(int index) const {
		return T(index);
	}
};

#endif

