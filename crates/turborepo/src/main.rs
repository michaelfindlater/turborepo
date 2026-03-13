// Bump all rust changes
#![deny(clippy::all)]

use std::{future::Future, pin::Pin, process, sync::Arc};

use anyhow::Result;
use miette::Report;

/// Concrete [`turborepo_query_api::QueryServer`] that delegates to
/// `turborepo_query`.
///
/// Lives in the binary crate because it's the only place that depends on both
/// `turborepo-lib` and `turborepo-query`, enabling the dependency inversion
/// that allows them to compile in parallel.
struct TurboQueryServer;

impl turborepo_query_api::QueryServer for TurboQueryServer {
    fn execute_query<'a>(
        &'a self,
        run: Arc<dyn turborepo_query_api::QueryRun>,
        query: &'a str,
        variables_json: Option<&'a str>,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<turborepo_query_api::QueryResult, turborepo_query_api::Error>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            turborepo_query::execute_query(run, query, variables_json)
                .await
                .map_err(Into::into)
        })
    }

    fn run_query_server(
        &self,
        run: Arc<dyn turborepo_query_api::QueryRun>,
        signal: turborepo_signals::SignalHandler,
    ) -> Pin<Box<dyn Future<Output = Result<(), turborepo_query_api::Error>> + Send + '_>> {
        Box::pin(async move {
            turborepo_query::run_query_server(run, signal)
                .await
                .map_err(Into::into)
        })
    }

    fn run_web_ui_server(
        &self,
        state: turborepo_ui::wui::query::SharedState,
        run: Arc<dyn turborepo_query_api::QueryRun>,
    ) -> Pin<Box<dyn Future<Output = Result<(), turborepo_query_api::Error>> + Send + '_>> {
        Box::pin(async move {
            turborepo_query::run_server(Some(state), run)
                .await
                .map_err(Into::into)
        })
    }
}

/// On Linux, restrict CPU affinity to at most `MAX_RAYON_THREADS` cores when
/// the system has more than that.
///
/// This is the earliest possible mitigation for a futex deadlock that occurs
/// during thread pool initialization on high-core-count machines (96+ cores).
/// By calling `sched_setaffinity` before any Rust runtime code runs, we ensure
/// that `std::thread::available_parallelism()`, rayon, and tokio all see a
/// reduced core count - exactly replicating the `taskset` workaround that
/// eliminates the hang in practice.
///
/// See <https://github.com/vercel/turborepo/issues/12251>
#[cfg(target_os = "linux")]
fn limit_cpu_affinity() {
    use turborepo_rayon_compat::MAX_RAYON_THREADS;

    // Allow users to opt out if they know their system is not affected.
    if std::env::var("TURBO_NO_AFFINITY_LIMIT").is_ok() {
        return;
    }

    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if cpus <= MAX_RAYON_THREADS {
        return;
    }

    // SAFETY: sched_setaffinity is a standard POSIX syscall. We pass a
    // zeroed cpu_set_t, enable the first MAX_RAYON_THREADS cores, and apply
    // to the current process (pid 0). This is identical to what `taskset`
    // does externally.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for i in 0..MAX_RAYON_THREADS {
            libc::CPU_SET(i, &mut set);
        }
        let ret = libc::sched_setaffinity(
            0, // current process
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
        if ret == 0 {
            eprintln!(
                "[turbo] restricted CPU affinity to {} cores (was {}) to avoid \
                 thread pool deadlock (vercel/turborepo#12251)",
                MAX_RAYON_THREADS, cpus
            );
        }
        // If it fails (e.g. in a container with seccomp), silently continue -
        // the rayon/tokio thread caps are still in place as a fallback.
    }
}

// This function should not expanded. Please add any logic to
// `turborepo_lib::main` instead
fn main() -> Result<()> {
    #[cfg(target_os = "linux")]
    limit_cpu_affinity();

    std::panic::set_hook(Box::new(turborepo_lib::panic_handler));

    let query_server: Arc<dyn turborepo_lib::QueryServer> = Arc::new(TurboQueryServer);
    let exit_code = turborepo_lib::main(Some(query_server)).unwrap_or_else(|err| {
        eprintln!("{:?}", Report::new(err));
        1
    });

    process::exit(exit_code)
}
