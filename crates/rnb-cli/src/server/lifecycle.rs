use rnb_llm::GenerationCancellation;
use std::io;
use std::net::{TcpListener, TcpStream};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(25);
const MIN_CONNECTION_HANDLERS: usize = 4;
const MAX_CONNECTION_HANDLERS: usize = 32;

fn connection_handler_limit(available_parallelism: usize) -> usize {
    available_parallelism
        .max(1)
        .saturating_mul(2)
        .clamp(MIN_CONNECTION_HANDLERS, MAX_CONNECTION_HANDLERS)
}

pub(super) fn install_shutdown_handler() -> Result<GenerationCancellation, String> {
    let shutdown = GenerationCancellation::new();
    let signal_shutdown = shutdown.clone();
    ctrlc::set_handler(move || signal_shutdown.cancel())
        .map_err(|error| format!("install shutdown signal handler: {error}"))?;
    Ok(shutdown)
}

pub(super) fn poll_accept(listener: &TcpListener) -> io::Result<Option<TcpStream>> {
    match listener.accept() {
        Ok((stream, _)) => {
            stream.set_nonblocking(false)?;
            Ok(Some(stream))
        }
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
            thread::sleep(ACCEPT_POLL_INTERVAL);
            Ok(None)
        }
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) struct ConnectionThreads {
    joins: Vec<JoinHandle<()>>,
    limit: usize,
}

impl ConnectionThreads {
    pub fn new() -> Self {
        let available_parallelism = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(MIN_CONNECTION_HANDLERS);
        Self::with_limit(connection_handler_limit(available_parallelism))
    }

    fn with_limit(limit: usize) -> Self {
        Self {
            joins: Vec::new(),
            limit: limit.max(1),
        }
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn is_full(&self) -> bool {
        self.joins.len() >= self.limit
    }

    pub fn spawn<F>(&mut self, run: F) -> Result<(), String>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.is_full() {
            return Err(format!(
                "HTTP connection handler limit reached ({})",
                self.limit
            ));
        }
        let join = thread::Builder::new()
            .name("rnb-http-connection".to_string())
            .spawn(run)
            .map_err(|error| format!("start HTTP connection handler: {error}"))?;
        self.joins.push(join);
        Ok(())
    }

    pub fn reap_finished(&mut self) -> Result<(), String> {
        let mut index = 0;
        while index < self.joins.len() {
            if self.joins[index].is_finished() {
                let join = self.joins.swap_remove(index);
                join.join()
                    .map_err(|_| "HTTP connection handler panicked".to_string())?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub fn join_all(mut self) -> Result<(), String> {
        for join in self.joins.drain(..) {
            join.join()
                .map_err(|_| "HTTP connection handler panicked".to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_handler_limit_scales_and_caps() {
        assert_eq!(connection_handler_limit(1), MIN_CONNECTION_HANDLERS);
        assert_eq!(connection_handler_limit(8), 16);
        assert_eq!(connection_handler_limit(64), MAX_CONNECTION_HANDLERS);
    }

    #[test]
    fn connection_threads_reject_work_at_capacity() {
        let mut threads = ConnectionThreads::with_limit(2);
        threads.spawn(|| {}).unwrap();
        threads.spawn(|| {}).unwrap();

        assert!(threads.is_full());
        assert!(threads.spawn(|| {}).is_err());
        threads.join_all().unwrap();
    }
}
