use std::collections::{hash_map, HashMap};
use std::collections::hash_map::Iter;
use std::hash::Hash;
use enum_map::{EnumArray, EnumMap};

pub trait SMap<'a, K: 'a + Copy + Clone, V: 'a>{
    type Iter: Iterator<Item=(K, &'a V)>;

    //fn compose<F: Fn(K) -> V>() -> Self;
    //fn compose_const
    fn get_value(&self, key: K) -> Option<&V>;
    fn get_mut_value(&mut self, key: K) -> Option<&mut V>;
    fn val(&self, key: K) -> &V{
        self.get_value(key).unwrap()
    }
    fn val_mut(&mut self, key: K) -> &mut V{
        self.get_mut_value(key).unwrap()
    }

    fn iter(&'a self) -> Self::Iter;
    fn all<F: Fn(&V) -> bool> (&'a self, f: F) -> bool{
        self.iter().fold(true, move |acc, (_,v)|{
            acc && f(v)
        })
    }
    fn any<F: Fn(&V) -> bool> (&'a self, f: F) -> bool{
        self.iter().fold(false, move |acc, (_,v)|{
            acc || f(v)
        })
    }

    fn find<F: FnOnce(&V) -> bool + Copy>(&'a self, f:F) -> Option<K>{
        self.iter().find(|(_, v)| f(v)).and_then(|(k, _)| Some(k))

    }
}

pub struct EnumMapRefIterator<'a, K:'a + EnumArray<V>, V:'a> {
    enum_iter: enum_map::Iter<'a, K, V>,
}
impl<'a, K:'a + EnumArray<V>, V:'a> Iterator for EnumMapRefIterator<'a, K, V>{
    type Item = (K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.enum_iter.next()
    }
}

impl<'a, K: 'a + EnumArray<V> + Clone + Copy, V: 'a> SMap<'a, K, V> for EnumMap<K,V>{
    //type Iter = EnumMapRefIterator<'a, K, V>;
    type  Iter = enum_map::Iter<'a, K, V>;

    fn get_value(&self, key: K) -> Option<&V> {
        Some(&self[key.to_owned()])
    }



    fn get_mut_value(&mut self, key: K) -> Option<&mut V> {
        Some(&mut self[key.clone()])
    }

    fn val(&self, key: K) -> &V{
        &self[key.clone()]
    }

    fn val_mut(&mut self, key: K) -> &mut V {
        &mut self[key.clone()]
    }

    fn iter(&'a self) -> Self::Iter {
        //EnumMap::iter(&self)
        self.iter()
    }
}

pub struct HashMapOwnedKeyIterator<'a, K: Hash + Eq + Copy + Clone, V>{
    inner_iter: hash_map::Iter<'a, K, V>
}

impl<'a, K: Hash + Eq + Copy + Clone, V> From<hash_map::Iter<'a, K, V>> for HashMapOwnedKeyIterator<'a, K, V>{
    fn from(value: Iter<'a, K, V>) -> Self {
        Self{inner_iter: value}
    }
}

impl<'a, K: Hash + Eq + Copy + Clone, V> Iterator for HashMapOwnedKeyIterator<'a, K, V>{
    type Item = (K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter.next().map(|(k,v)| (*k, v))
    }
}

impl<'a, K: 'a + Copy + Clone, V: 'a> SMap<'a, K, V> for HashMap<K,V>
where K: Hash + Eq{
    type Iter = HashMapOwnedKeyIterator<'a, K, V>;

    fn get_value(&self, key: K) -> Option<&V> {
        self.get(&key)
    }

    fn get_mut_value(&mut self, key: K) -> Option<&mut V> {
        self.get_mut(&key)
    }

    fn iter(&'a self) -> Self::Iter {
        HashMapOwnedKeyIterator::from(self.iter())
    }
}

#[cfg(test)]
mod test{
/*
    enum Side{
        North,
        East,
        South,
        West
    }

 */
}