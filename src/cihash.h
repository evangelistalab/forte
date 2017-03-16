/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _cihash_h_
#define _cihash_h_
#include <tuple>
#include <vector>

template <class Key, class Hash = std::hash<Key>> class CIHash {
  private:
    template <class V> struct CINode {
        V value;
        size_t next;
    };
    std::vector<CINode<Key>> vec;
    std::vector<size_t> begin_index;
    const size_t MIN_NUM_BUCKET = 1 << 0;
    float max_load = 1.0;
    size_t current_max_load_size = 1;
    const size_t MAX_SIZE = std::min(vec.max_size(), begin_index.max_size());
    size_t num_bucket;
    size_t current_size;

    std::tuple<size_t, size_t, size_t> find_detail_by_key(const Key& key) const;
    void double_buckets();
    void half_buckets();
    inline void update_current_max_load_size();

  public:
    class iterator;
    explicit CIHash();
    explicit CIHash(size_t count);
    explicit CIHash(const std::vector<Key>& other);
    explicit CIHash(size_t count, const std::vector<Key>& other);
    explicit CIHash(const CIHash<Key, Hash>& other);
    /*- Element access -*/
    const Key& operator[](size_t pos) const;
    size_t find(const Key& key) const;
    /*- Iterators -*/
    const iterator begin() { return this->vec.begin(); }
    const iterator end() { return this->vec.end(); }
    /*- Capacity -*/
    size_t size() const;
    size_t max_size() const;
    size_t capacity() const;
    /*- Modifiers -*/
    void clear();
    size_t add(const Key& key);
    std::pair<size_t, size_t> erase_by_key(const Key& key);
    std::pair<size_t, size_t> erase_by_index(size_t index);
    std::vector<size_t> merge(const CIHash<Key, Hash>& source);
    std::vector<size_t> merge(const std::vector<Key>& source);
    /*- Bucket interface -*/
    size_t bucket_count() const;
    size_t max_bucket_count() const;
    size_t bucket_size(size_t n) const;
    size_t bucket(const Key& key) const;
    /*- Hash policy -*/
    float load_factor() const;
    float max_load_factor() const;
    void max_load_factor(float ml);
    void reserve(size_t count);
    void shrink_to_fit();
    std::vector<size_t> optimize();

    class iterator : public std::vector<CINode<Key>>::iterator {
      public:
        iterator(typename std::vector<CINode<Key>>::iterator it)
            : std::vector<CINode<Key>>::iterator(it) {}

        const Key& operator*() const {
            return std::vector<CINode<Key>>::iterator::operator*().value;
        }

        const Key* const operator->() const {
            return &(std::vector<CINode<Key>>::iterator::operator*().value);
        }
    };
};

#endif // _cihash_h_
