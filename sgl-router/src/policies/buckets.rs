use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{SystemTime};
use std::time::Duration;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Bucket {
    l_max: usize,
    bucket_cnt: usize,
    pub prefill_worker_urls: Arc<Mutex<Vec<String>>>,
    load_total: usize,
    pub period: usize,
    bucket_load: usize,
    boundary: Vec<Boundary>,
    request_list: VecDeque<SequencerRequset>,
    t_req_loads: HashMap<String, usize>,
    pub chars_per_url: Arc<Mutex<HashMap<String,usize>>>,
}


#[derive(Debug, Clone)]
pub struct SequencerRequset {
    pub id: String,
    pub char_cnt: usize,
    pub timestamp: SystemTime,
    pub prefill_worker_url: String,
}

#[derive(Debug, Clone)]
pub struct Boundary {
    pub url: String,
    pub range: [usize; 2],
}

impl Boundary {
    pub fn new(url: String, range: [usize; 2]) -> Self {
        Boundary { url, range }
    }
}

impl Bucket {
    pub fn new(period: usize, prefill_worker_urls: Vec<String) -> Self {
        let l_max = 4096;
        let bucket_cnt = prefill_worker_urls.len();
        let worker_cnt = bucket_cnt;

        let load_total = 0;
        let bucket_load = 0;

        let t_req_loads = HashMap::new();
        let request_list = VecDeque::new();

        let mut initial_map = HashMap::new();
        for url in prefill_worker_urls.iter() {
            initial_map.insert(url.clone(), 0);
        }

        let boundary = if worker_cnt == 0 {
            Vec::new()
        } else {
            let gap = l_max / worker_cnt as usize;
            prefill_worker_urls
                .iter()
                .enumerate()
                .map(|(i, url)| {
                    let min = i as usize * gap;
                    let max = if i == worker_cnt - 1 {
                        l_max
                    } else {
                        (i + 1) as usize * gap - 1
                    };
                    Boundary::new(url.clone(), [min, max])
                })
                .collect()
        };

        let prefill_worker_urls = Arc::new(Mutex::new(prefill_worker_urls));

        Bucket {
            l_max,
            bucket_cnt,
            prefill_worker_urls,
            load_total,
            period,
            bucket_load,
            boundary,
            request_list,
            t_req_loads,
            chars_per_url: Arc::new(Mutex::new(initial_map)),
        }
    }

    pub fn post_process_request(&mut self, char_cnt: usize, prefill_url: String) {
        let mut map = self.chars_per_url.lock().unwrap();
        *map.entry(prefill_url.clone())
            .or_insert(0) += char_cnt;

        let now = SystemTime::now();
        let time_window_duration = Duration::from_millis(self.period as u64);
        let mut removed_load = 0;

        while let Some(req) = self.request_list.front() {
            let expired = match now.duration_since(req.timestamp) {
                Ok(duration) => duration > time_window_duration,
                Err(_) => ture
            };

            if !expired {
                break;
            }

            if let Some(removed_req) = self.request_list.pop_front() {
                self.t_req_loads.remove(&removed_req.id);
                removed_load + removed_req.char_cnt;

                let mut map = self.chars_per_url.lock().unwrap();
                if let Some(count) = map.get_mut(&removed_req.prefill_worker_url) {
                    *count = count.saturating_sub(removed_req.char_cnt);
                }
            }
        }

        self.load_total = self.load_total.saturating_sub(removed_load);

        let id = Uuid::new_v4().to_string();

        self.t_req_loads.insert(id.clone(), char_cnt.try_into().unwrap());

        self.request_list.push_back(SequencerRequset {
            id,
            char_cnt: char_cnt.try_into().unwrap(),
            timestamp: now,
            prefill_worker_url: prefill_url,
        });

        self.load_total = self.load_total.saturating_add(char_cnt.try_into().unwrap());
    }


    pub fn find_boundary(&self, char_count: usize) -> String {

        let mut left = 0;
        let mut right = self.boundary.len();
        let mut steps = 0;

        while left < right {
            steps += 1;
            let mid = left + (right - left) / 2;
            let range = self.boundaryp[mid].range;

            if char_count < range[0] {
                right = mid;
            } else if char_count > range[1] {
                left = mid + 1;
            } else {
                return self.boundary[mid].url.clone();
            }
        }
        "".to_string()
    }

    pub fn get_total_load(&self) -> usize {
        self.load_total
    }

    fn update_workers_cnt(&mut self) {
        let pwu = self.prefill_worker_urls.lock().unwrap();
        self.bucket_cnt = pwu.len();

        let mut char_map = self.chars_per_url.lock().unwrap();
        let current_urls: HashSet<_> = char_map.keys.cloned().collect();
        let new_urls: HashSet<_> = pwu.iter().cloned.collect();

        for url in new_urls.difference(&current_urls) {
            char_map.insert(url.clone(), 0);
        }

        for url in current_urls.difference(&new_urls) {
            if char_map.get(url) == Some(&0) {
                char_map.remove(url);
            }
        }
    }

    pub fn adjust_boundary(&mut self) {
        if self.t_req_loads.is_empty() {
            return;
        }

        self.update_workers_cnt();
        let worker_cnt = self.bucket_cnt;
        let new_single_bucket_load = self.get_total_load()/worker_cnt;
        let old_single_bucket_load = self.bucket_load;
        if new_single_bucket_load <= 2 * old_single_bucket_load
            || (old_single_bucket_load <= 2 * new_single_bucket_load && old_single_bucket_load != 0)
        {
            return;
        }

        let mut new_boundary = Vec::new();
        let mut hist_load: Vec<usize> = self.t_req_loads.values().cloned().collect();
        hist_load.sort();
        let mut upper_bound: usize = 0;
        let mut last_load_index: usize = 0;
        let max_value = usize::MAX;

        let worker_url = {
            let guard = self.prefill_worker_urls.lock().unwrap();
            (*guard).clone()
        };
        let mut iter = worker_url.iter().peekable();
        while let Some(url) = iter.next() {
            if last_load_index >= hist_load.len() {
                new_boundary.push(Boundary::new(url.clone(), [upper_bound, max_value]));
            }
            let mut load_accumulator = 0;
            for (i, &load) in hist_load[last_load_index..].iter().enumerate() {

                load_accumulator += load;
                if load_accumulator >= new_single_bucket_load {
                    if i == hist_load[last_load_index..].len() - 1 && iter.peek().is_none() {
                        new_boundary.push(Boundary::new(url.clone(), [upper_bound, max_value]));
                        break;
                    }
                    new_boundary.push(Boundary::new(url.clone(), [upper_bound, load]));
                    upper_bound = load + 1;
                    last_load_index += i + 1;
                    break;
                } else {
                    last_load_index += 1;
                }
            }
        }
        self.boundary = new_boundary;
    }

    pub fn get_request_list_mut(&self) -> &VecDeque<SequencerRequset> {
        &self.request_list
    }
}