use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

#[derive(Debug, Clone, Copy)]
struct Entry {
    bytes: u64,
    last_touch: u64,
}

/// Byte-budgeted least-recently-used accounting policy.
///
/// The caller owns cached values and removes the keys returned by [`Self::touch`].
/// Keeping accounting separate lets memory policy stay in `rnb-memory` while the
/// owning subsystem retains its value types and synchronization strategy.
pub struct ByteLruPolicy<K>
where
    K: Clone + Eq + Hash + Ord,
{
    max_bytes: u64,
    resident_bytes: u64,
    clock: u64,
    entries: HashMap<K, Entry>,
    oldest: BinaryHeap<Reverse<(u64, K)>>,
}

impl<K> ByteLruPolicy<K>
where
    K: Clone + Eq + Hash + Ord,
{
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            resident_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            oldest: BinaryHeap::new(),
        }
    }

    pub const fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn touch(&mut self, key: K, bytes: u64) -> Vec<K> {
        self.clock = self.clock.wrapping_add(1);
        let last_touch = self.clock;
        if let Some(entry) = self.entries.get_mut(&key) {
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(entry.bytes)
                .saturating_add(bytes);
            *entry = Entry { bytes, last_touch };
        } else {
            self.resident_bytes = self.resident_bytes.saturating_add(bytes);
            self.entries
                .insert(key.clone(), Entry { bytes, last_touch });
        }
        self.oldest.push(Reverse((last_touch, key)));

        let mut evicted = Vec::new();
        while self.resident_bytes > self.max_bytes {
            let Some(key) = self.pop_oldest() else {
                break;
            };
            evicted.push(key);
        }
        evicted
    }

    pub fn pop_oldest(&mut self) -> Option<K> {
        loop {
            let Reverse((candidate_touch, candidate_key)) = self.oldest.pop()?;
            let Some(entry) = self.entries.get(&candidate_key).copied() else {
                continue;
            };
            if entry.last_touch != candidate_touch {
                continue;
            }
            self.entries.remove(&candidate_key);
            self.resident_bytes = self.resident_bytes.saturating_sub(entry.bytes);
            return Some(candidate_key);
        }
    }

    pub fn remove(&mut self, key: &K) -> bool {
        let Some(entry) = self.entries.remove(key) else {
            return false;
        };
        self.resident_bytes = self.resident_bytes.saturating_sub(entry.bytes);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_keys_until_total_fits() {
        let mut policy = ByteLruPolicy::new(10);
        assert!(policy.touch(1_u32, 4).is_empty());
        assert!(policy.touch(2_u32, 4).is_empty());
        assert!(policy.touch(1_u32, 4).is_empty());

        assert_eq!(policy.touch(3_u32, 5), vec![2]);
        assert_eq!(policy.resident_bytes(), 9);
    }

    #[test]
    fn updates_accounting_and_removes_entries() {
        let mut policy = ByteLruPolicy::new(10);
        policy.touch(1_u32, 8);
        policy.touch(1_u32, 3);
        assert_eq!(policy.resident_bytes(), 3);
        assert!(policy.remove(&1));
        assert_eq!(policy.resident_bytes(), 0);
        assert!(!policy.remove(&1));
        policy.touch(2, 2);
        policy.touch(3, 2);
        assert_eq!(policy.pop_oldest(), Some(2));
        assert_eq!(policy.resident_bytes(), 2);
    }
}
