/// Safe ceiling for turbo's thread pools (rayon and tokio).
///
/// Turbo creates two thread pools: rayon (data parallelism) and tokio
/// (async I/O), each sized to the available CPU count. On high-core
/// machines the combined thread count triggers a futex deadlock during
/// initialization — all threads block in `futex_wait_queue` and turbo
/// hangs indefinitely.
///
/// Testing on 96-core GitHub Actions runners ([issue]) showed:
/// - 24 total threads (12+12 via `taskset -c 0-11`): 0/120+ hangs
/// - ~120 total threads (60 cores, no cap): 0 hangs historically
/// - ~144 total threads (72+72 via sched_setaffinity): ~22% hang rate
/// - ~168 total threads (72 rayon + 96 tokio, stock): ~33% hang rate
///
/// A per-pool cap of 32 keeps the combined count at ~64, well below the
/// observed threshold while still providing ample parallelism. 32 worker
/// threads already saturate most CI and build workloads.
///
/// [issue]: https://github.com/vercel/turborepo/issues/12251
pub const MAX_RAYON_THREADS: usize = 32;

/// Scale a CPU count to a safe rayon thread pool size.
///
/// Uses all available cores up to [`MAX_RAYON_THREADS`], then caps.
/// Always returns at least 1.
pub fn scale_thread_count(cpus: usize) -> usize {
    cpus.clamp(1, MAX_RAYON_THREADS)
}

/// Run a blocking closure, notifying the tokio runtime when possible.
///
/// On a multi-threaded tokio runtime this calls `tokio::task::block_in_place`
/// so the runtime can spawn a replacement worker. On current-thread runtimes
/// (common in tests) or outside of tokio the closure runs directly.
pub fn block_in_place<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    use tokio::runtime::{Handle, RuntimeFlavor};
    match Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(RuntimeFlavor::MultiThread) => tokio::task::block_in_place(f),
        _ => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_thread_count_always_at_least_one() {
        assert_eq!(scale_thread_count(0), 1);
        assert_eq!(scale_thread_count(1), 1);
    }

    #[test]
    fn scale_thread_count_preserves_small_values() {
        assert_eq!(scale_thread_count(2), 2);
        assert_eq!(scale_thread_count(4), 4);
        assert_eq!(scale_thread_count(8), 8);
        assert_eq!(scale_thread_count(16), 16);
    }

    #[test]
    fn scale_thread_count_caps_at_max() {
        assert_eq!(scale_thread_count(32), MAX_RAYON_THREADS);
        assert_eq!(scale_thread_count(64), MAX_RAYON_THREADS);
        assert_eq!(scale_thread_count(96), MAX_RAYON_THREADS);
        assert_eq!(scale_thread_count(256), MAX_RAYON_THREADS);
    }

    #[test]
    fn block_in_place_works_outside_runtime() {
        let result = block_in_place(|| 42);
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn block_in_place_works_on_current_thread_runtime() {
        let result = block_in_place(|| 42);
        assert_eq!(result, 42);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn block_in_place_works_on_multi_thread_runtime() {
        let result = block_in_place(|| 42);
        assert_eq!(result, 42);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn block_in_place_with_rayon_completes() {
        let result = block_in_place(|| {
            use rayon::prelude::*;
            (0..1000).into_par_iter().map(|i| i * 2).sum::<i64>()
        });
        assert_eq!(result, 999_000);
    }
}
