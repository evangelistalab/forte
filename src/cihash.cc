#include "cihash.h"

template<class Key, class Hash>
CIHash<Key, Hash>::CIHash()
{
    this->clear();
}

template<class Key, class Hash>
CIHash<Key, Hash>::CIHash( size_t count ) {
    this->clear();
    this->reserve(count);
}

template<class Key, class Hash>
CIHash<Key, Hash>::CIHash( const std::vector<Key> &other ) {
    this->clear();
    this->reserve(other.size());
    for (Key k : other) {
        this->add(k);
    }
}

template<class Key, class Hash>
CIHash<Key, Hash>::CIHash( size_t count, const std::vector<Key> &other ) {
    this->clear();
    this->reserve(other.size());
    this->reserve(count);
    for (Key k : other) {
        this->add(k);
    }
}

template<class Key, class Hash>
CIHash<Key, Hash>::CIHash( const CIHash<Key, Hash>& other ) :
    vec(other.vec),
    begin_index(other.begin_index),
    max_load(other.max_load),
    num_bucket(other.num_bucket),
    current_size(other.current_size)
{
}

template<class Key, class Hash>
std::tuple<size_t, size_t, size_t> CIHash<Key, Hash>::find_detail_by_key( const Key& key ) const {
    size_t bucket_index = Hash()(key) % num_bucket;
    if (this->begin_index[bucket_index] != MAX_SIZE) {
        size_t pre_index = MAX_SIZE;
        size_t temp_index = this->begin_index[bucket_index];
        while (temp_index != MAX_SIZE) {
            if (this->vec[temp_index].value == key) {
                return std::make_tuple(bucket_index, pre_index, temp_index);
            }
            pre_index = temp_index;
            temp_index = this->vec[temp_index].next;
        }
        return std::make_tuple(bucket_index, pre_index, MAX_SIZE);
    }
    return std::make_tuple(bucket_index, MAX_SIZE, MAX_SIZE);
}

template<class Key, class Hash>
void CIHash<Key, Hash>::double_buckets() {
    size_t new_num_bucket = num_bucket << 1;
    if (begin_index.capacity() < new_num_bucket)
        begin_index.reserve(new_num_bucket);
    begin_index.insert(begin_index.end(), num_bucket, MAX_SIZE);
    for (size_t bucket_index = 0; bucket_index < num_bucket; ++bucket_index) {
        size_t pre_index = MAX_SIZE;
        size_t temp_index = this->begin_index[bucket_index];
        size_t new_bucket_index = bucket_index + num_bucket;
        size_t new_temp_index = MAX_SIZE;
        while (temp_index != MAX_SIZE) {
            size_t temp_bucket_index = Hash()(this->vec[temp_index].value) % new_num_bucket;
            size_t next_index = this->vec[temp_index].next;
            if (temp_bucket_index != bucket_index) {
                if (new_temp_index == MAX_SIZE) {
                    this->begin_index[new_bucket_index] = temp_index;
                    new_temp_index = temp_index;
                } else {
                    this->vec[new_temp_index].next = temp_index;
                }
                this->vec[temp_index].next = MAX_SIZE;
                if (pre_index == MAX_SIZE) {
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
    this->num_bucket = new_num_bucket;
}

template<class Key, class Hash>
void CIHash<Key, Hash>::half_buckets() {
    size_t new_num_bucket = num_bucket >> 1;
    for (size_t bucket_index = 0; bucket_index < new_num_bucket; ++bucket_index) {
        size_t appending_index = this->begin_index[bucket_index + new_num_bucket];
        if (appending_index == MAX_SIZE)
            continue;
        size_t temp_index = this->begin_index[bucket_index];
        if (temp_index == MAX_SIZE) {
            this->begin_index[bucket_index] = appending_index;
            continue;
        }
        while (this->vec[temp_index].next != MAX_SIZE) {
            temp_index = this->vec[temp_index].next;
        }
        this->vec[temp_index].next = appending_index;
    }
    this->num_bucket = new_num_bucket;
    this->begin_index.erase(begin_index.begin() + num_bucket, begin_index.end());
}

template<class Key, class Hash>
const Key& CIHash<Key, Hash>::operator[]( size_t pos ) const {
    return this->vec[pos].value;
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::find( const Key& key ) const {
    size_t bucket_index = Hash()(key) % num_bucket;
    if (this->begin_index[bucket_index] != MAX_SIZE) {
        size_t temp_index = this->begin_index[bucket_index];
        while (temp_index != MAX_SIZE) {
            if (this->vec[temp_index].value == key) {
                return temp_index;
            }
            temp_index = this->vec[temp_index].next;
        }
    }
    return MAX_SIZE;
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::size() const {
    return this->vec.size();
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::max_size() const {
    return MAX_SIZE - 1;
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::capacity() const {
    return this->vec.capacity();
}

template<class Key, class Hash>
void CIHash<Key, Hash>::clear() {
    this->vec.clear();
    this->begin_index.clear();
    this->current_size = 0;
    this->num_bucket = MIN_NUM_BUCKET;
    begin_index.reserve(num_bucket);
    begin_index.insert(begin_index.begin(), num_bucket, MAX_SIZE);
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::add(const Key &key ) {
    size_t key_bucket_index, key_pre_index, key_index;
    std::tie(key_bucket_index, key_pre_index, key_index) = find_detail_by_key(key);
    if (key_index != MAX_SIZE)
        return key_index;
    this->vec.push_back({key, MAX_SIZE});
    if (key_pre_index != MAX_SIZE)
        this->vec[key_pre_index].next = current_size;
    else
        this->begin_index[key_bucket_index] = current_size;
    ++current_size;
    if (load_factor() > max_load)
        double_buckets();
    return current_size - 1;
}

template<class Key, class Hash>
std::pair<size_t, size_t> CIHash<Key, Hash>::erase_by_key( const Key& key ) {
    size_t key_bucket_index, key_pre_index, key_index;
    size_t last_bucket_index, last_pre_index, last_index;
    std::tie(key_bucket_index, key_pre_index, key_index) = find_detail_by_key(key);
    if (key_index == MAX_SIZE) return std::make_pair(MAX_SIZE, MAX_SIZE);
    std::tie(last_bucket_index, last_pre_index, last_index) = find_detail_by_key(this->operator [](current_size -1));
    if (key_pre_index == MAX_SIZE) {
        this->begin_index[key_bucket_index] = this->vec[key_index].next;
    } else {
        this->vec[key_pre_index].next = this->vec[key_index].next;
    }
    if (key_index == last_index) {
        this->vec.pop_back();
        this->current_size--;
        return std::make_pair(last_index, MAX_SIZE);
    }
    if (key_index == last_pre_index) {
        last_pre_index = key_pre_index;
    }
    this->vec[key_index] = this->vec[last_index];
    if (last_pre_index == MAX_SIZE) {
        this->begin_index[last_bucket_index] = key_index;
    } else {
        this->vec[last_pre_index].next = key_index;
    }
    this->vec.pop_back();
    this->current_size--;
    return std::make_pair(last_index, key_index);
}

template<class Key, class Hash>
std::pair<size_t, size_t> CIHash<Key, Hash>::erase_by_index( size_t index ) {
    return erase_by_key(this->vec.operator [](index).value);
}

template<class Key, class Hash>
std::vector<size_t> CIHash<Key, Hash>::merge( const CIHash<Key, Hash>& source ) {
    std::vector<size_t> cur_index;
    size_t merge_size = source.size();
    cur_index.reserve(merge_size);
    for (size_t i = 0; i < merge_size; ++i) {
        cur_index.push_back(this->add(source[i]));
    }
    return cur_index;
}

template<class Key, class Hash>
std::vector<size_t> CIHash<Key, Hash>::merge( const std::vector<Key>& source ) {
    std::vector<size_t> cur_index;
    size_t merge_size = source.size();
    cur_index.reserve(merge_size);
    for (size_t i = 0; i < merge_size; ++i) {
        cur_index.push_back(this->add(source[i]));
    }
    return cur_index;
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::bucket_count() const {
    return this->num_bucket;
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::max_bucket_count() const {
    return MAX_SIZE - 1;
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::bucket_size( size_t n ) const {
    if (n >= this->num_bucket)
        return MAX_SIZE;
//    if (this->begin_index[n] == MAX_SIZE)
//        return 0;
    size_t temp_index = this->begin_index[n];
    size_t count = 0;
    while (temp_index != MAX_SIZE) {
        ++count;
        temp_index = this->vec[temp_index].next;
    }
    return count;
}

template<class Key, class Hash>
size_t CIHash<Key, Hash>::bucket( const Key& key ) const {
    size_t key_bucket_index, key_pre_index, key_index;
    std::tie(key_bucket_index, key_pre_index, key_index) = find_detail_by_key(key);
    if (key_index == MAX_SIZE)
        return MAX_SIZE;
    else
        return key_bucket_index;
}

template<class Key, class Hash>
float CIHash<Key, Hash>::load_factor() const {
    return ((float) current_size)/num_bucket;
}

template<class Key, class Hash>
float CIHash<Key, Hash>::max_load_factor() const {
    return this->max_load;
}

template<class Key, class Hash>
void CIHash<Key, Hash>::max_load_factor( float ml ) {
    this->max_load = ml;
    while (load_factor() > max_load)
        double_buckets();
}

template<class Key, class Hash>
void CIHash<Key, Hash>::reserve( size_t count ) {
    if (count <= current_size)
        return;
    this->vec.reserve(count);
    while (((float) count)/num_bucket > max_load)
        double_buckets();
}

template<class Key, class Hash>
void CIHash<Key, Hash>::shrink_to_fit() {
    this->vec.shrink_to_fit();
    while (((float) current_size)/num_bucket <= 0.5 * max_load)
        half_buckets();
}

template<class Key, class Hash>
std::vector<size_t> CIHash<Key, Hash>::optimize() {
    std::vector<size_t> cur_index(current_size, MAX_SIZE);
    std::vector<CINode<Key> > old_vec;
    this->vec.swap(old_vec);
    this->vec.reserve(old_vec.capacity());
    size_t temp_index = 0;
    for (size_t bucket_index = 0; bucket_index < num_bucket; ++bucket_index) {
        size_t old_temp_index = this->begin_index[bucket_index];
        while (old_temp_index != MAX_SIZE) {
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
            this->vec[i-1].next = MAX_SIZE;
            this->begin_index[temp_bucket_index] = i;
            pre_bucket_index = temp_bucket_index;
        } else {
            this->vec[i-1].next = i;
        }
    }
    this->vec[current_size-1].next = MAX_SIZE;
    return cur_index;
}
