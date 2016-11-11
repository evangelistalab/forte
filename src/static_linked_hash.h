#ifndef STATICLINKEDHASH_H
#define STATICLINKEDHASH_H

#include<unordered_set>

namespace psi{ namespace forte{
template<
        class Key,
        class Hash = std::hash<Key>,
        class KeyEqual = std::equal_to<Key>,
        class Allocator = std::allocator<Key>
> class StaticLinkedHashSet
{
public:
    StaticLinkedHashSet();
    ~StaticLinkedHashSet();
};
}}

#endif // STATICLINKEDHASH_H
