#ifndef STATICLINKEDHASH_H
#define STATICLINKEDHASH_H

#include<unordered_map>

namespace psi{ namespace forte{
template<
        class Key,
        class T,
        class Hash = std::hash<Key>,
        class KeyEqual = std::equal_to<Key>,
        class Allocator = std::allocator< std::pair< const Key, T> >
> class StaticLinkedHash
{
public:
    StaticLinkedHash();
    ~StaticLinkedHash();
};
}}

#endif // STATICLINKEDHASH_H
