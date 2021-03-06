#pragma once

#include <array>
#include <cassert>
#include <memory>
#include <type_traits>

namespace clang
{
    namespace type_erasure
    {
        namespace detail
        {
            template <class T>
            void delete_data(void* data) noexcept
            {
                assert(data);
                delete static_cast<T*>(data);
            }

            template <class T>
            void* copy_data(void* data)
            {
                return data ? new T( *static_cast<T*>(data) ) : nullptr;
            }

            template <class T>
            std::shared_ptr<void> copy_data(const std::shared_ptr<void>& data)
            {
                return data ? std::make_shared<T>(*static_cast<T*>(data.get())) : nullptr;
            }

            template< class T, class Buffer >
            void* copy_into_buffer( void* data, Buffer& buffer ) noexcept( std::is_nothrow_copy_constructible<T>::value )
            {
                assert(data);
                new (&buffer) T( *static_cast<T*>( data ) );
                return &buffer;
            }

            template< class T, class Buffer >
            std::shared_ptr<void> copy_into_buffer(const std::shared_ptr<void>& data, Buffer& buffer) noexcept( std::is_nothrow_copy_constructible<T>::value )
            {
                assert(data);
                new (&buffer) T( *static_cast<T*>( data.get() ) );
                return std::shared_ptr<T>( std::shared_ptr<T>(), static_cast<T*>( static_cast<void*>( &buffer ) ) );
            }

            inline const char* char_ptr( const void* ptr ) noexcept
            {
                assert(ptr);
                return static_cast<const char*>( ptr );
            }

            template < class Buffer >
            bool is_heap_allocated (void* data, const Buffer& buffer) noexcept
            {
                // treat nullptr as heap-allocated
                if(!data)
                    return true;
                return data < static_cast<const void*>(&buffer) ||
                        static_cast<const void*>( char_ptr(&buffer) + sizeof(buffer) ) <= data;
            }

            template <class T>
            struct IsReferenceWrapper : std::false_type
            {};

            template <class T>
            struct IsReferenceWrapper< std::reference_wrapper<T> > : std::true_type
            {};


            template < class T >
            struct remove_reference_wrapper
            {
                using type = T;
            };

            template < class T >
            struct remove_reference_wrapper< std::reference_wrapper< T > >
            {
                using type = T;
            };

            template < class T >
            using remove_reference_wrapper_t = typename remove_reference_wrapper< T >::type;
        }


        template <class Derived>
        class Accessor
        {
        public:
            constexpr explicit Accessor(std::size_t tile_id = 0,
                                        bool contains_ref_wrapper = false) noexcept
                : contains_reference_wrapper(contains_ref_wrapper),
                  id(tile_id)
            {}

            template <class T>
            T& get() noexcept
            {
                auto data = static_cast<Derived*>(this)->write( );
                assert(data);
                if(contains_reference_wrapper)
                    return static_cast<std::reference_wrapper<T>*>(data)->get();
                return *static_cast<T*>(data);
            }

            template <class T>
            const T& get() const noexcept
            {
                const auto data = static_cast<const Derived*>(this)->read( );
                assert(data);
                if(contains_reference_wrapper)
                    return static_cast<const std::reference_wrapper<T>*>(data)->get();
                return *static_cast<const T*>(data);
            }

            template < class T >
            T* target() noexcept
            {
                auto data = static_cast<Derived*>(this)->write( );
                assert(data);
                if( data && id == typeid( T ).hash_code() )
                    return static_cast<T*>( data );
                return nullptr;
            }

            template < class T >
            const T* target( ) const noexcept
            {
                auto data = static_cast<const Derived*>(this)->read( );
                assert(data);
                if( data && id == typeid( T ).hash_code() )
                    return static_cast<const T*>( data );
                return nullptr;
            }

            explicit operator bool() const noexcept
            {
                return static_cast<const Derived*>(this)->read( ) != nullptr;
            }

        private:
            bool contains_reference_wrapper = false;
            std::size_t id = 0;
        };


        class Storage : public Accessor<Storage>
        {
        public:
            constexpr Storage() noexcept = default;

            template <class T,
                      std::enable_if_t<!std::is_base_of<Storage, std::decay_t<T> >::value>* = nullptr>
            explicit Storage(T&& value)
                : Accessor<Storage>(typeid(std::decay_t<T>).hash_code(),
                                    detail::IsReferenceWrapper< std::decay_t<T> >::value),
                  del(&detail::delete_data< std::decay_t<T> >),
                  copy_data(&detail::copy_data< std::decay_t<T> >),
                  data(new std::decay_t<T>(std::forward<T>(value)))
            {}

            template <class T,
                      std::enable_if_t<!std::is_base_of<Storage, std::decay_t<T> >::value>* = nullptr>
            Storage& operator=(T&& value)
            {
                return *this = Storage(std::forward<T>(value));
            }

            ~Storage()
            {
                reset();
            }

            Storage(const Storage& other)
                : Accessor<Storage>(other),
                  del(other.del),
                  copy_data(other.copy_data),
                  data(other.data == nullptr ? nullptr : other.copy())
            {}

            Storage(Storage&& other) noexcept
                : Accessor<Storage>(other),
                  del(other.del),
                  copy_data(other.copy_data),
                  data(other.data)
            {
                other.data = nullptr;
            }

            Storage& operator=(const Storage& other)
            {
                reset();
                Accessor<Storage>::operator=(other);
                del = other.del;
                copy_data = other.copy_data;
                data = (other.data == nullptr ? nullptr : other.copy());
                return *this;
            }

            Storage& operator=(Storage&& other) noexcept
            {
                reset();
                Accessor<Storage>::operator=(other);
                del = other.del;
                copy_data = other.copy_data;
                data = other.data;
                other.data = nullptr;
                return *this;
            }

        private:
            friend class Accessor<Storage>;

            void reset() noexcept
            {
                if(data)
                    del(data);
            }

            void* read() const noexcept
            {
                return data;
            }

            void* write() noexcept
            {
                return read();
            }

            void* copy() const
            {
                assert(data);
                return copy_data(data);
            }

            using delete_fn = void(*)(void*);
            using copy_fn = void*(*)(void*);
            delete_fn del = nullptr;
            copy_fn copy_data = nullptr;
            void* data = nullptr;
        };


        class COWStorage : public Accessor<COWStorage>
        {
        public:
            constexpr COWStorage() noexcept = default;

            template <class T,
                      std::enable_if_t<!std::is_base_of<COWStorage, std::decay_t<T> >::value>* = nullptr>
            explicit COWStorage(T&& value)
                : Accessor<COWStorage>(typeid(std::decay_t<T>).hash_code(),
                                       detail::IsReferenceWrapper< std::decay_t<T> >::value),
                  copy_data(&detail::copy_data< std::decay_t<T> >),
                  data(std::make_shared< std::decay_t<T> >(std::forward<T>(value)))
            {}

            template <class T,
                      std::enable_if_t<!std::is_base_of<COWStorage, std::decay_t<T> >::value>* = nullptr>
            COWStorage& operator=(T&& value)
            {
                return *this = COWStorage(std::forward<T>(value));
            }

        private:
            friend class Accessor<COWStorage>;

            void* read() const noexcept
            {
                return data.get();
            }

            void* write()
            {
                if(!data.unique())
                    data = copy_data(data);
                return read();
            }

            using copy_fn = std::shared_ptr<void>(*)(const std::shared_ptr<void>&);
            copy_fn copy_data = nullptr;
            std::shared_ptr<void> data = nullptr;
        };


        class NonCopyableCOWStorage : public Accessor<NonCopyableCOWStorage>
        {
        public:
            constexpr NonCopyableCOWStorage() noexcept = default;

            template <class T,
                      std::enable_if_t<!std::is_base_of<NonCopyableCOWStorage, std::decay_t<T> >::value>* = nullptr>
            explicit NonCopyableCOWStorage(T&& value)
                : Accessor<NonCopyableCOWStorage>(typeid(std::decay_t<T>).hash_code(),
                                       detail::IsReferenceWrapper< std::decay_t<T> >::value),
                  data(std::make_shared< std::decay_t<T> >(std::forward<T>(value)))
            {}

            template <class T,
                      std::enable_if_t<!std::is_base_of<NonCopyableCOWStorage, std::decay_t<T> >::value>* = nullptr>
            NonCopyableCOWStorage& operator=(T&& value)
            {
                return *this = NonCopyableCOWStorage(std::forward<T>(value));
            }

        private:
            friend class Accessor<NonCopyableCOWStorage>;

            void* read() const noexcept
            {
                return data.get();
            }

            void* write() noexcept
            {
                return read();
            }

            std::shared_ptr<void> data = nullptr;
        };


        template <int buffer_size>
        class SBOStorage : public Accessor< SBOStorage<buffer_size> >
        {
            using Buffer = std::array<char,buffer_size>;

            struct FunctionTable
            {
                using delete_fn = void(*)(void*);
                using copy_fn = void*(*)(void*);
                using buffer_copy_fn = void*(*)(void*, Buffer&);

                delete_fn del = nullptr;
                copy_fn copy = nullptr;
                buffer_copy_fn copy_into = nullptr;
            };

        public:

            constexpr SBOStorage() noexcept = default;

            template <class T,
                      std::enable_if_t<!std::is_base_of<SBOStorage, std::decay_t<T> >::value>* = nullptr>
            explicit SBOStorage(T&& value)
            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
                : Accessor< SBOStorage<buffer_size> >(typeid(std::decay_t<T>).hash_code(),
                                                      detail::IsReferenceWrapper< std::decay_t<T> >::value),
                  function_table{&detail::delete_data< std::decay_t<T> >,
                                 &detail::copy_data< std::decay_t<T> >,
                                 &detail::copy_into_buffer<std::decay_t<T>, Buffer>}
            {
                if( sizeof(std::decay_t<T>) <= sizeof(Buffer))
                {
                    new(&buffer) std::decay_t<T>(std::forward<T>(value));
                    data = &buffer;
                }
                else
                    data = new std::decay_t<T>(std::forward<T>(value));
            }

            template <class T,
                      std::enable_if_t<!std::is_base_of<SBOStorage, std::decay_t<T> >::value>* = nullptr>
            SBOStorage& operator=(T&& value)
            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
            {
                return *this = SBOStorage(std::forward<T>(value));
            }

            ~SBOStorage()
            {
                reset();
            }

            SBOStorage(const SBOStorage& other)
                : Accessor< SBOStorage<buffer_size> >(other),
                  function_table(other.function_table)
            {
                data = other.copy_into(buffer);
            }

            SBOStorage(SBOStorage&& other) noexcept
                : Accessor< SBOStorage<buffer_size> >(other),
                  function_table(other.function_table)
            {
                if(!other.data)
                    return;

                if(detail::is_heap_allocated(other.data, other.buffer))
                    data = other.data;
                else
                {
                    buffer = other.buffer;
                    data = &buffer;
                }

                other.data = nullptr;
            }

            SBOStorage& operator=(const SBOStorage& other)
            {
                reset();
                Accessor< SBOStorage<buffer_size> >::operator=(other);
                function_table = other.function_table;
                data = other.copy_into(buffer);
                return *this;
            }

            SBOStorage& operator=(SBOStorage&& other) noexcept
            {
                reset();
                if(!other.data)
                {
                    data = nullptr;
                    return *this;
                }
                Accessor< SBOStorage<buffer_size> >::operator=(other);
                function_table = other.function_table;
                if(detail::is_heap_allocated(other.data, other.buffer))
                    data = other.data;
                else
                {
                    buffer = other.buffer;
                    data = &buffer;
                }
                other.data = nullptr;
                return *this;
            }

        private:
            friend class Accessor< SBOStorage<buffer_size> >;

            void reset() noexcept
            {
                if(data && detail::is_heap_allocated(data, buffer))
                    function_table.del(data);
            }

            void* read() const noexcept
            {
                return data;
            }

            void* write()
            {
                return read();
            }

            void* copy_into(Buffer& other_buffer) const
            {
                if(!data)
                    return nullptr;
                if(detail::is_heap_allocated(data, buffer))
                    return function_table.copy(data);
                return function_table.copy_into(data, other_buffer);
            }

            FunctionTable function_table;
            void* data = nullptr;
            Buffer buffer;
        };


        template <int buffer_size>
        class NonCopyableSBOStorage : public Accessor< NonCopyableSBOStorage<buffer_size> >
        {
            using Buffer = std::array<char,buffer_size>;

            struct FunctionTable
            {
                using delete_fn = void(*)(void*);
                delete_fn del = nullptr;
            };

        public:

            constexpr NonCopyableSBOStorage() noexcept = default;

            template <class T,
                      std::enable_if_t<!std::is_base_of<NonCopyableSBOStorage, std::decay_t<T> >::value>* = nullptr>
            explicit NonCopyableSBOStorage(T&& value)
            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
                : Accessor< NonCopyableSBOStorage<buffer_size> >(typeid(std::decay_t<T>).hash_code(),
                                                      detail::IsReferenceWrapper< std::decay_t<T> >::value),
                  function_table{&detail::delete_data< std::decay_t<T> >}
            {
                if( sizeof(std::decay_t<T>) <= sizeof(Buffer))
                {
                    new(&buffer) std::decay_t<T>(std::forward<T>(value));
                    data = &buffer;
                }
                else
                    data = new std::decay_t<T>(std::forward<T>(value));
            }

            template <class T,
                      std::enable_if_t<!std::is_base_of<NonCopyableSBOStorage, std::decay_t<T> >::value>* = nullptr>
            NonCopyableSBOStorage& operator=(T&& value)
            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
            {
                return *this = NonCopyableSBOStorage(std::forward<T>(value));
            }

            ~NonCopyableSBOStorage()
            {
                reset();
            }

            NonCopyableSBOStorage(NonCopyableSBOStorage&& other) noexcept
                : Accessor< NonCopyableSBOStorage<buffer_size> >(other),
                  function_table(other.function_table)
            {
                if(!other.data)
                    return;

                if(detail::is_heap_allocated(other.data, other.buffer))
                    data = other.data;
                else
                {
                    buffer = other.buffer;
                    data = &buffer;
                }

                other.data = nullptr;
            }

            NonCopyableSBOStorage& operator=(NonCopyableSBOStorage&& other) noexcept
            {
                reset();
                if(!other.data)
                {
                    data = nullptr;
                    return *this;
                }
                Accessor< NonCopyableSBOStorage<buffer_size> >::operator=(other);
                function_table = other.function_table;
                if(detail::is_heap_allocated(other.data, other.buffer))
                    data = other.data;
                else
                {
                    buffer = other.buffer;
                    data = &buffer;
                }
                other.data = nullptr;
                return *this;
            }

        private:
            friend class Accessor< NonCopyableSBOStorage<buffer_size> >;

            void reset() noexcept
            {
                if(data && detail::is_heap_allocated(data, buffer))
                    function_table.del(data);
            }

            void* read() const noexcept
            {
                return data;
            }

            void* write()
            {
                return read();
            }

            FunctionTable function_table;
            void* data = nullptr;
            Buffer buffer;
        };


        template <int buffer_size>
        class SBOCOWStorage : public Accessor< SBOCOWStorage<buffer_size> >
        {
            static const constexpr bool always_copy = false;
            static const constexpr bool move_heap_allocated = true;
            using Buffer = std::array<char,buffer_size>;

            struct FunctionTable
            {
                using copy_fn = std::shared_ptr<void>(*)(const std::shared_ptr<void>&);
                using buffer_copy_fn = std::shared_ptr<void>(*)(const std::shared_ptr<void>&, Buffer&);

                copy_fn copy = nullptr;
                buffer_copy_fn copy_into = nullptr;
            };

        public:

            constexpr SBOCOWStorage() noexcept = default;

            template <class T,
                      std::enable_if_t<!std::is_base_of<SBOCOWStorage, std::decay_t<T> >::value>* = nullptr>
            explicit SBOCOWStorage(T&& value)
            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
                : Accessor< SBOCOWStorage<buffer_size> >(typeid(std::decay_t<T>).hash_code(),
                                                         detail::IsReferenceWrapper< std::decay_t<T> >::value),
                  function_table{&detail::copy_data< std::decay_t<T> >,
                                 &detail::copy_into_buffer<std::decay_t<T>, Buffer>}
            {
                if( sizeof(std::decay_t<T>) <= sizeof(Buffer))
                {
                    new(&buffer) std::decay_t<T>(std::forward<T>(value));
                    data = std::shared_ptr< std::decay_t<T> >(
                               std::shared_ptr< std::decay_t<T> >(),
                               static_cast<std::decay_t<T>*>(static_cast<void*>(&buffer))
                               );
                }
                else
                    data = std::make_shared< std::decay_t<T> >(std::forward<T>(value));
            }

            template <class T,
                      std::enable_if_t<!std::is_base_of<SBOCOWStorage, std::decay_t<T> >::value>* = nullptr>
            SBOCOWStorage& operator=(T&& value)
            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
            {
                return *this = SBOCOWStorage(std::forward<T>(value));
            }

            SBOCOWStorage(const SBOCOWStorage& other)
                : Accessor< SBOCOWStorage<buffer_size> >(other),
                  function_table(other.function_table)
            {
                if(!other.data)
                    return;
                data = other.copy(buffer);
            }

            SBOCOWStorage(SBOCOWStorage&& other) noexcept
                : Accessor< SBOCOWStorage<buffer_size> >(other),
                  function_table(other.function_table)
            {
                if(!other.data)
                    return;
                data = std::move(other).move_if_heap_allocated(buffer);
                other.data = nullptr;
            }

            SBOCOWStorage& operator=(const SBOCOWStorage& other)
            {
                if(!other.data)
                {
                    data = nullptr;
                    return *this;
                }
                Accessor< SBOCOWStorage<buffer_size> >::operator=(other);
                function_table = other.function_table;

                data = other.copy(buffer);
                return *this;
            }

            SBOCOWStorage& operator=(SBOCOWStorage&& other) noexcept
            {
                if(!other.data)
                {
                    data = nullptr;
                    return *this;
                }
                Accessor< SBOCOWStorage<buffer_size> >::operator=(other);
                function_table = other.function_table;
                data = std::move(other).move_if_heap_allocated(buffer);
                other.data = nullptr;
                return *this;
            }

        private:
            friend class Accessor< SBOCOWStorage<buffer_size> >;

            void* read() const noexcept
            {
                return data.get();
            }

            void* write()
            {
                if(!data.unique() && detail::is_heap_allocated(data.get(), buffer))
                    data = function_table.copy(data);
                return read();
            }

            std::shared_ptr<void> copy(Buffer& other_buffer) const
            {
                if(detail::is_heap_allocated(data.get(), buffer))
                    return data;
                else
                    return function_table.copy_into(data, other_buffer);
            }

            std::shared_ptr<void> move_if_heap_allocated(Buffer& other_buffer) const &&
            {
                if(detail::is_heap_allocated(data.get(), buffer))
                    return std::move(data);
                else
                    return function_table.copy_into(data, other_buffer);
            }

            FunctionTable function_table;
            std::shared_ptr<void> data = nullptr;
            Buffer buffer;
        };


//        template <int buffer_size>
//        class NonCopyableSBOCOWStorage : public Accessor< NonCopyableSBOCOWStorage<buffer_size> >
//        {
//            static const constexpr bool always_copy = false;
//            static const constexpr bool move_heap_allocated = true;
//            using Buffer = std::array<char,buffer_size>;
//        public:

//            constexpr NonCopyableSBOCOWStorage() noexcept = default;

//            template <class T,
//                      std::enable_if_t<!std::is_base_of<NonCopyableSBOCOWStorage, std::decay_t<T> >::value>* = nullptr>
//            explicit NonCopyableSBOCOWStorage(T&& value)
//            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
//                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
//                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
//                : Accessor< NonCopyableSBOCOWStorage<buffer_size> >(typeid(std::decay_t<T>).hash_code(),
//                                                         detail::IsReferenceWrapper< std::decay_t<T> >::value),
//                  function_table{&detail::copy_data< std::decay_t<T> >,
//                                 &detail::copy_into_buffer<std::decay_t<T>, Buffer>}
//            {
//                if( sizeof(std::decay_t<T>) <= sizeof(Buffer))
//                {
//                    new(&buffer) std::decay_t<T>(std::forward<T>(value));
//                    data = std::shared_ptr< std::decay_t<T> >(
//                               std::shared_ptr< std::decay_t<T> >(),
//                               static_cast<std::decay_t<T>*>(static_cast<void*>(&buffer))
//                               );
//                }
//                else
//                    data = std::make_shared< std::decay_t<T> >(std::forward<T>(value));
//            }

//            template <class T,
//                      std::enable_if_t<!std::is_base_of<NonCopyableSBOCOWStorage, std::decay_t<T> >::value>* = nullptr>
//            NonCopyableSBOCOWStorage& operator=(T&& value)
//            noexcept( sizeof(std::decay_t<T>) <= sizeof(Buffer) &&
//                      ( (std::is_rvalue_reference<T>::value && std::is_nothrow_move_constructible<std::decay_t<T>>::value) ||
//                        (std::is_lvalue_reference<T>::value && std::is_nothrow_copy_constructible<std::decay_t<T>>::value) ) )
//            {
//                return *this = NonCopyableSBOCOWStorage(std::forward<T>(value));
//            }

//            NonCopyableSBOCOWStorage(NonCopyableSBOCOWStorage&& other) noexcept
//                : Accessor< NonCopyableSBOCOWStorage<buffer_size> >(other),
//                  function_table(other.function_table)
//            {
//                if(!other.data)
//                    return;
//                data = std::move(other).move_if_heap_allocated(buffer);
//                other.data = nullptr;
//            }

//            NonCopyableSBOCOWStorage& operator=(NonCopyableSBOCOWStorage&& other) noexcept
//            {
//                if(!other.data)
//                {
//                    data = nullptr;
//                    return *this;
//                }
//                Accessor< NonCopyableSBOCOWStorage<buffer_size> >::operator=(other);
//                function_table = other.function_table;
//                data = std::move(other).move_if_heap_allocated(buffer);
//                other.data = nullptr;
//                return *this;
//            }

//        private:
//            friend class Accessor< NonCopyableSBOCOWStorage<buffer_size> >;

//            void* read() const noexcept
//            {
//                return data.get();
//            }

//            void* write() noexcept
//            {
//                return read();
//            }

//            std::shared_ptr<void> move_if_heap_allocated(Buffer& other_buffer) const &&
//            {
//                if(detail::is_heap_allocated(data.get(), buffer))
//                    return std::move(data);
//                else
//                    return function_table.copy_into(data, other_buffer);
//            }

//            FunctionTable function_table;
//            std::shared_ptr<void> data = nullptr;
//            Buffer buffer;
//        };
    }
}
