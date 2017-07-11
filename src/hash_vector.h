/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _hash_vec_h_
#define _hash_vec_h_
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <functional>
#include <iostream>

template <class Key, class Hash = std::hash<Key>> class HashVector {
  private:
    template <class V> struct CINode {
        V value;
        size_t next;
    };
    std::vector<CINode<Key>> vec;
    std::vector<size_t> begin_index;
    static const size_t MIN_NUM_BUCKET = 1 << 0;
    float max_load = 1.0;
    size_t current_max_load_size = 1;
    size_t num_bucket;
    size_t current_size;

    std::tuple<size_t, size_t, size_t> find_detail_by_key(const Key& key) const;
    void double_buckets();
    void half_buckets();
    inline void update_current_max_load_size();

  public:
    class iterator;
    explicit HashVector();
    explicit HashVector(size_t count);
    explicit HashVector(const std::vector<Key>& other);
    template <class Hash_2> explicit HashVector(const std::unordered_set<Key, Hash_2>& other);

    /*- Element access -*/
    static const size_t npos = SIZE_MAX;
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
    template <class Hash_2> std::vector<size_t> merge(const HashVector<Key, Hash_2>& source);
    std::vector<size_t> merge(const std::vector<Key>& source);
    template <class Hash_2> void merge(const std::unordered_set<Key, Hash_2>& source);
    void swap(HashVector<Key, Hash>& other);

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

    /*- Convertors -*/
    std::vector<Key> toVector();
    std::unordered_set<Key, Hash> toUnordered_set();

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

template <class Key, class Hash> const size_t HashVector<Key, Hash>::npos;

namespace std {
template <class Key, class Hash> void swap(HashVector<Key, Hash>& a, HashVector<Key, Hash>& b) {
    a.swap(b);
}
}

template <class Key, class Hash> HashVector<Key, Hash>::HashVector() { this->clear(); }

template <class Key, class Hash> HashVector<Key, Hash>::HashVector(size_t count) {
    this->clear();
    this->reserve(count);
}

template <class Key, class Hash> HashVector<Key, Hash>::HashVector(const std::vector<Key>& other) {
    this->clear();
    this->reserve(other.size());
    for (Key k : other) {
        this->add(k);
    }
}

template <class Key, class Hash>
template <class Hash_2>
HashVector<Key, Hash>::HashVector(const std::unordered_set<Key, Hash_2>& other) {
    this->clear();
    this->reserve(other.size());
    for (Key k : other) {
        this->add(k);
    }
}

template <class Key, class Hash>
std::tuple<size_t, size_t, size_t> HashVector<Key, Hash>::find_detail_by_key(const Key& key) const {
    size_t bucket_index = Hash()(key) % num_bucket;
    if (this->begin_index[bucket_index] != npos) {
        size_t pre_index = npos;
        size_t temp_index = this->begin_index[bucket_index];
        while (temp_index != npos) {
            if (this->vec[temp_index].value == key) {
                return std::make_tuple(bucket_index, pre_index, temp_index);
            }
            pre_index = temp_index;
            temp_index = this->vec[temp_index].next;
        }
        return std::make_tuple(bucket_index, pre_index, npos);
    }
    return std::make_tuple(bucket_index, npos, npos);
}

template <class Key, class Hash> void HashVector<Key, Hash>::double_buckets() {
    size_t new_num_bucket = num_bucket << 1;
    if (begin_index.capacity() < new_num_bucket)
        begin_index.reserve(new_num_bucket);
    begin_index.insert(begin_index.end(), num_bucket, npos);
    if (current_size >= num_bucket) {
        for (size_t bucket_index = 0; bucket_index < num_bucket; ++bucket_index) {
            size_t pre_index = npos;
            size_t temp_index = this->begin_index[bucket_index];
            size_t new_bucket_index = bucket_index + num_bucket;
            size_t new_temp_index = npos;
            while (temp_index != npos) {
                size_t temp_bucket_index = Hash()(this->vec[temp_index].value) % new_num_bucket;
                size_t next_index = this->vec[temp_index].next;
                if (temp_bucket_index != bucket_index) {
                    if (new_temp_index == npos) {
                        this->begin_index[new_bucket_index] = temp_index;
                    } else {
                        this->vec[new_temp_index].next = temp_index;
                    }
                    new_temp_index = temp_index;
                    this->vec[temp_index].next = npos;
                    if (pre_index == npos) {
                        this->begin_index[bucket_index] = next_index;
                    } else {
                        this->vec[pre_index].next = next_index;
                    }
                } else {
                    pre_index = temp_index;
                }
                temp_index = next_index;
            }
        }
    } else {
        for (size_t i = 0; i < current_size; ++i) {
            size_t hash_value = Hash()(this->vec[i].value);
            size_t old_bucket_index = hash_value % num_bucket;
            size_t new_bucket_index = hash_value % new_num_bucket;
            if (new_bucket_index != old_bucket_index &&
                this->begin_index[new_bucket_index] == npos) {
                size_t pre_index = npos;
                size_t temp_index = this->begin_index[old_bucket_index];
                size_t new_temp_index = npos;
                while (temp_index != npos) {
                    size_t temp_bucket_index = Hash()(this->vec[temp_index].value) % new_num_bucket;
                    size_t next_index = this->vec[temp_index].next;
                    if (temp_bucket_index != old_bucket_index) {
                        if (new_temp_index == npos) {
                            this->begin_index[new_bucket_index] = temp_index;
                        } else {
                            this->vec[new_temp_index].next = temp_index;
                        }
                        new_temp_index = temp_index;
                        this->vec[temp_index].next = npos;
                        if (pre_index == npos) {
                            this->begin_index[old_bucket_index] = next_index;
                        } else {
                            this->vec[pre_index].next = next_index;
                        }
                    } else {
                        pre_index = temp_index;
                    }
                    temp_index = next_index;
                }
            }
        }
    }
    this->num_bucket = new_num_bucket;
    update_current_max_load_size();
}

template <class Key, class Hash> void HashVector<Key, Hash>::half_buckets() {
    size_t new_num_bucket = num_bucket >> 1;
    if (new_num_bucket < MIN_NUM_BUCKET)
        return;
    if (current_size >= new_num_bucket) {
        for (size_t bucket_index = 0; bucket_index < new_num_bucket; ++bucket_index) {
            size_t appending_index = this->begin_index[bucket_index + new_num_bucket];
            if (appending_index == npos)
                continue;
            size_t temp_index = this->begin_index[bucket_index];
            if (temp_index == npos) {
                this->begin_index[bucket_index] = appending_index;
                continue;
            }
            while (this->vec[temp_index].next != npos) {
                temp_index = this->vec[temp_index].next;
            }
            this->vec[temp_index].next = appending_index;
        }
    } else {
        for (size_t i = 0; i < current_size; ++i) {
            size_t old_bucket_index = Hash()(this->vec[i].value) % num_bucket;
            if (old_bucket_index >= new_num_bucket && this->begin_index[old_bucket_index] != npos) {
                size_t appending_index = this->begin_index[old_bucket_index];
                this->begin_index[old_bucket_index] = npos;
                size_t new_bucket_index = old_bucket_index - new_num_bucket;
                size_t temp_index = this->begin_index[new_bucket_index];
                if (temp_index == npos) {
                    this->begin_index[new_bucket_index] = appending_index;
                    continue;
                }
                while (this->vec[temp_index].next != npos) {
                    temp_index = this->vec[temp_index].next;
                }
                this->vec[temp_index].next = appending_index;
            }
        }
    }
    this->num_bucket = new_num_bucket;
    update_current_max_load_size();
    this->begin_index.resize(num_bucket);
}

template <class Key, class Hash> inline void HashVector<Key, Hash>::update_current_max_load_size() {
    this->current_max_load_size = max_load * num_bucket;
}

template <class Key, class Hash> const Key& HashVector<Key, Hash>::operator[](size_t pos) const {
    return this->vec[pos].value;
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::find(const Key& key) const {
    size_t bucket_index = Hash()(key) % num_bucket;
    if (this->begin_index[bucket_index] != npos) {
        size_t temp_index = this->begin_index[bucket_index];
        while (temp_index != npos) {
            if (this->vec[temp_index].value == key) {
                return temp_index;
            }
            temp_index = this->vec[temp_index].next;
        }
    }
    return npos;
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::size() const {
    return this->vec.size();
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::max_size() const {
    size_t vec_max_size = vec.max_size();
    return vec_max_size < npos ? vec_max_size : npos - 1;
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::capacity() const {
    return this->vec.capacity();
}

template <class Key, class Hash> void HashVector<Key, Hash>::clear() {
    this->vec.clear();
    this->begin_index.clear();
    this->current_size = 0;
    this->num_bucket = MIN_NUM_BUCKET;
    update_current_max_load_size();
    begin_index.reserve(num_bucket);
    begin_index.insert(begin_index.begin(), num_bucket, npos);
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::add(const Key& key) {
    size_t key_bucket_index, key_pre_index, key_index;
    std::tie(key_bucket_index, key_pre_index, key_index) = find_detail_by_key(key);
    if (key_index != npos)
        return key_index;
    this->vec.push_back({key, npos});
    if (key_pre_index != npos)
        this->vec[key_pre_index].next = current_size;
    else
        this->begin_index[key_bucket_index] = current_size;
    ++current_size;
    if (current_size > current_max_load_size)
        double_buckets();
    return current_size - 1;
}

template <class Key, class Hash>
std::pair<size_t, size_t> HashVector<Key, Hash>::erase_by_key(const Key& key) {
    size_t key_bucket_index, key_pre_index, key_index;
    size_t last_bucket_index, last_pre_index, last_index;
    std::tie(key_bucket_index, key_pre_index, key_index) = find_detail_by_key(key);
    if (key_index == npos)
        return std::make_pair(npos, npos);
    std::tie(last_bucket_index, last_pre_index, last_index) =
        find_detail_by_key(this->operator[](current_size - 1));
    if (key_pre_index == npos) {
        this->begin_index[key_bucket_index] = this->vec[key_index].next;
    } else {
        this->vec[key_pre_index].next = this->vec[key_index].next;
    }
    if (key_index == last_index) {
        this->vec.pop_back();
        this->current_size--;
        return std::make_pair(last_index, npos);
    }
    if (key_index == last_pre_index) {
        last_pre_index = key_pre_index;
    }
    this->vec[key_index] = this->vec[last_index];
    if (last_pre_index == npos) {
        this->begin_index[last_bucket_index] = key_index;
    } else {
        this->vec[last_pre_index].next = key_index;
    }
    this->vec.pop_back();
    this->current_size--;
    return std::make_pair(last_index, key_index);
}

template <class Key, class Hash>
std::pair<size_t, size_t> HashVector<Key, Hash>::erase_by_index(size_t index) {
    return erase_by_key(this->vec.operator[](index).value);
}

template <class Key, class Hash>
template <class Hash_2>
std::vector<size_t> HashVector<Key, Hash>::merge(const HashVector<Key, Hash_2>& source) {
    std::vector<size_t> cur_index;
    size_t merge_size = source.size();
    cur_index.reserve(merge_size);
    for (size_t i = 0; i < merge_size; ++i) {
        cur_index.push_back(this->add(source[i]));
    }
    return cur_index;
}

template <class Key, class Hash>
std::vector<size_t> HashVector<Key, Hash>::merge(const std::vector<Key>& source) {
    std::vector<size_t> cur_index;
    size_t merge_size = source.size();
    cur_index.reserve(merge_size);
    for (size_t i = 0; i < merge_size; ++i) {
        cur_index.push_back(this->add(source[i]));
    }
    return cur_index;
}

template <class Key, class Hash>
template <class Hash_2>
void HashVector<Key, Hash>::merge(const std::unordered_set<Key, Hash_2>& source) {
    for (Key k : source) {
        this->add(k);
    }
}

template <class Key, class Hash> void HashVector<Key, Hash>::swap(HashVector<Key, Hash>& other) {
    vec.swap(other.vec);
    begin_index.swap(other.begin_index);
    std::swap(this->max_load, other.max_load);
    std::swap(this->current_max_load_size, other.current_max_load_size);
    std::swap(this->num_bucket, other.num_bucket);
    std::swap(this->current_size, other.current_size);
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::bucket_count() const {
    return this->num_bucket;
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::max_bucket_count() const {
    size_t begin_index_max_size = begin_index.max_size();
    return begin_index_max_size < npos ? begin_index_max_size : npos - 1;
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::bucket_size(size_t n) const {
    if (n >= this->num_bucket)
        return npos;
    size_t temp_index = this->begin_index[n];
    size_t count = 0;
    while (temp_index != npos) {
        ++count;
        temp_index = this->vec[temp_index].next;
    }
    return count;
}

template <class Key, class Hash> size_t HashVector<Key, Hash>::bucket(const Key& key) const {
    size_t key_bucket_index, key_pre_index, key_index;
    std::tie(key_bucket_index, key_pre_index, key_index) = find_detail_by_key(key);
    if (key_index == npos)
        return npos;
    else
        return key_bucket_index;
}

template <class Key, class Hash> float HashVector<Key, Hash>::load_factor() const {
    return ((float)current_size) / num_bucket;
}

template <class Key, class Hash> float HashVector<Key, Hash>::max_load_factor() const {
    return this->max_load;
}

template <class Key, class Hash> void HashVector<Key, Hash>::max_load_factor(float ml) {
    this->max_load = ml;
    update_current_max_load_size();
    while (current_size > current_max_load_size)
        double_buckets();
}

template <class Key, class Hash> void HashVector<Key, Hash>::reserve(size_t count) {
    if (count <= current_size)
        return;
    this->vec.reserve(count);
    while (count > current_max_load_size)
        double_buckets();
}

template <class Key, class Hash> void HashVector<Key, Hash>::shrink_to_fit() {
    this->vec.shrink_to_fit();
    while ((num_bucket >> 1) >= MIN_NUM_BUCKET && current_size <= (current_max_load_size >> 1))
        half_buckets();
    this->begin_index.shrink_to_fit();
}

template <class Key, class Hash> std::vector<size_t> HashVector<Key, Hash>::optimize() {
    std::vector<size_t> cur_index(current_size, npos);
    std::vector<CINode<Key>> old_vec;
    this->vec.swap(old_vec);
    this->vec.reserve(old_vec.capacity());
    size_t temp_index = 0;
    for (size_t bucket_index = 0; bucket_index < num_bucket; ++bucket_index) {
        size_t old_temp_index = this->begin_index[bucket_index];
        while (old_temp_index != npos) {
            this->vec.push_back(old_vec[old_temp_index]);
            cur_index[old_temp_index] = temp_index++;
            old_temp_index = old_vec[old_temp_index].next;
        }
    }
    size_t pre_bucket_index = Hash()(this->vec[0].value) % num_bucket;
    this->begin_index[pre_bucket_index] = 0;
    for (size_t i = 1; i < current_size; ++i) {
        size_t temp_bucket_index = Hash()(this->vec[i].value) % num_bucket;
        if (temp_bucket_index != pre_bucket_index) {
            this->vec[i - 1].next = npos;
            this->begin_index[temp_bucket_index] = i;
            pre_bucket_index = temp_bucket_index;
        } else {
            this->vec[i - 1].next = i;
        }
    }
    this->vec[current_size - 1].next = npos;
    return cur_index;
}

template <class Key, class Hash> std::vector<Key> HashVector<Key, Hash>::toVector() {
    std::vector<Key> keys;
    keys.reserve(current_size);
    for (Key k : (*this)) {
        keys.push_back(k);
    }
    return keys;
}

template <class Key, class Hash>
std::unordered_set<Key, Hash> HashVector<Key, Hash>::toUnordered_set() {
    std::unordered_set<Key, Hash> uSet;
    uSet.reserve(current_size);
    for (Key k : (*this)) {
        uSet.insert(k);
    }
    return uSet;
}

template <class Key, class Hash, class Value>
HashVector<Key, Hash>
hashVector_from_unordered_map(const std::unordered_map<Key, Value, Hash>& umap) {
    HashVector<Key, Hash> hvec(umap.size());
    for (std::pair<Key, Value> kv : umap) {
        hvec.add(kv.first);
    }
    return hvec;
}

template <class Key, class Hash, class Value>
HashVector<Key, Hash>
hashVector_from_unordered_map(const std::unordered_map<Key, Value, Hash>& umap,
                              std::vector<Value>& values) {
    HashVector<Key, Hash> hvec(umap.size());
    values.reserve(umap.size());
    for (std::pair<Key, Value> kv : umap) {
        hvec.add(kv.first);
        values.push_back(kv.second);
    }
    return hvec;
}

template <class Key, class Hash, class Value>
std::unordered_map<Key, Value, Hash>
unordered_map_from_hashVector(const HashVector<Key, Hash>& hvec, const std::vector<Value>& values) {
    std::unordered_map<Key, Value, Hash> uMap;
    size_t current_size = hvec.size();
    uMap.reserve(current_size);
    for (size_t i = 0; i < current_size; ++i) {
        uMap[hvec[i]] = values[i];
    }
    return uMap;
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values,
           const HashVector<Key, Hash>& source, const std::vector<Value>& src_values) {
    size_t original_size = hvec.size();
    size_t merge_size = source.size();
    for (size_t i = 0; i < merge_size; ++i) {
        size_t index = hvec.add(source[i]);
        if (index < original_size) {
            values[index] = src_values[i];
        } else {
            values.push_back(src_values[i]);
        }
    }
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values,
           const HashVector<Key, Hash>& source, const std::vector<Value>& src_values,
           const std::function<Value(Value, Value)>& f, const Value& default_value,
           bool change_this) {
    size_t original_size = hvec.size();
    size_t merge_size = source.size();
    if (change_this) {
        std::vector<bool> intersects(original_size, false);
        for (size_t i = 0; i < merge_size; ++i) {
            size_t index = hvec.add(source[i]);
            if (index < original_size) {
                values[index] = f(values[index], src_values[i]);
                intersects[index] = true;
            } else {
                values.push_back(f(default_value, src_values[i]));
            }
        }
        for (size_t i = 0; i < original_size; ++i) {
            if (!intersects[i])
                values[i] = f(values[i], default_value);
        }
    } else {
        for (size_t i = 0; i < merge_size; ++i) {
            size_t index = hvec.add(source[i]);
            if (index < original_size) {
                values[index] = f(values[index], src_values[i]);
            } else {
                values.push_back(f(default_value, src_values[i]));
            }
        }
    }
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values, const std::vector<Key>& source,
           const std::vector<Value>& src_values) {
    size_t original_size = hvec.size();
    size_t merge_size = source.size();
    for (size_t i = 0; i < merge_size; ++i) {
        size_t index = hvec.add(source[i]);
        if (index < original_size) {
            values[index] = src_values[i];
        } else {
            values.push_back(src_values[i]);
        }
    }
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values, const std::vector<Key>& source,
           const std::vector<Value>& src_values, const std::function<Value(Value, Value)>& f,
           const Value& default_value, bool change_this) {
    size_t original_size = hvec.size();
    size_t merge_size = source.size();
    if (change_this) {
        std::vector<bool> intersects(original_size, false);
        for (size_t i = 0; i < merge_size; ++i) {
            size_t index = hvec.add(source[i]);
            if (index < original_size) {
                values[index] = f(values[index], src_values[i]);
                intersects[index] = true;
            } else {
                values.push_back(f(default_value, src_values[i]));
            }
        }
        for (size_t i = 0; i < original_size; ++i) {
            if (!intersects[i])
                values[i] = f(values[i], default_value);
        }
    } else {
        for (size_t i = 0; i < merge_size; ++i) {
            size_t index = hvec.add(source[i]);
            if (index < original_size) {
                values[index] = f(values[index], src_values[i]);
            } else {
                values.push_back(f(default_value, src_values[i]));
            }
        }
    }
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values,
           const std::vector<std::pair<Key, Value>>& source) {
    size_t original_size = hvec.size();
    for (std::pair<Key, Value> kv : source) {
        size_t index = hvec.add(kv.first);
        if (index < original_size) {
            values[index] = kv.second;
        } else {
            values.push_back(kv.second);
        }
    }
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values,
           const std::vector<std::pair<Key, Value>>& source,
           const std::function<Value(Value, Value)>& f, const Value& default_value,
           bool change_this) {
    size_t original_size = hvec.size();
    if (change_this) {
        std::vector<bool> intersects(original_size, false);
        for (std::pair<Key, Value> kv : source) {
            size_t index = hvec.add(kv.first);
            if (index < original_size) {
                values[index] = f(values[index], kv.second);
                intersects[index] = true;
            } else {
                values.push_back(f(default_value, kv.second));
            }
        }
        for (size_t i = 0; i < original_size; ++i) {
            if (!intersects[i])
                values[i] = f(values[i], default_value);
        }
    } else {
        for (std::pair<Key, Value> kv : source) {
            size_t index = hvec.add(kv.first);
            if (index < original_size) {
                values[index] = f(values[index], kv.second);
            } else {
                values.push_back(f(default_value, kv.second));
            }
        }
    }
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values,
           const std::unordered_map<Key, Value, Hash>& source) {
    size_t original_size = hvec.size();
    for (std::pair<Key, Value> kv : source) {
        size_t index = hvec.add(kv.first);
        if (index < original_size) {
            values[index] = kv.second;
        } else {
            values.push_back(kv.second);
        }
    }
}

template <class Key, class Hash, class Value>
void merge(HashVector<Key, Hash>& hvec, std::vector<Value>& values,
           const std::unordered_map<Key, Value, Hash>& source,
           const std::function<Value(Value, Value)>& f, const Value& default_value,
           bool change_this) {
    size_t original_size = hvec.size();
    if (change_this) {
        std::vector<bool> intersects(original_size, false);
        for (std::pair<Key, Value> kv : source) {
            size_t index = hvec.add(kv.first);
            if (index < original_size) {
                values[index] = f(values[index], kv.second);
                intersects[index] = true;
            } else {
                values.push_back(f(default_value, kv.second));
            }
        }
        for (size_t i = 0; i < original_size; ++i) {
            if (!intersects[i])
                values[i] = f(values[i], default_value);
        }
    } else {
        for (std::pair<Key, Value> kv : source) {
            size_t index = hvec.add(kv.first);
            if (index < original_size) {
                values[index] = f(values[index], kv.second);
            } else {
                values.push_back(f(default_value, kv.second));
            }
        }
    }
}

#endif // _hash_vec_h_
