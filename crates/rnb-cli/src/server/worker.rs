use super::http::{write_json_response, HttpRequest};
use super::session_store::ResponseStore;
use rnb_llm::{Engine, EngineLoadConfig, GenerationCancellation};
use rnb_runtime::policy::response_session_cache_budget_bytes;
use rnb_runtime::scheduler::{fair_execution_queue, FairExecutionReceiver, FairExecutionSender};
use std::io::ErrorKind;
use std::net::{Shutdown as TcpShutdown, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub(super) struct WorkerRequest {
    pub stream: TcpStream,
    pub request: HttpRequest,
    pub cancellation: RequestCancellation,
}

pub(super) type WorkerSender = FairExecutionSender<WorkerRequest>;

pub(super) struct Worker {
    sender: Option<WorkerSender>,
    join: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn sender(&self) -> WorkerSender {
        self.sender
            .as_ref()
            .expect("worker sender unavailable")
            .clone()
    }

    pub fn shutdown(mut self) -> Result<(), String> {
        self.stop()
    }

    fn stop(&mut self) -> Result<(), String> {
        drop(self.sender.take());
        if let Some(join) = self.join.take() {
            join.join()
                .map_err(|_| "engine worker panicked during shutdown".to_string())?;
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

pub(super) fn start(
    model_path: PathBuf,
    load_config: EngineLoadConfig,
    model_name: String,
    explicit_cache_bytes: Option<u64>,
) -> Result<Worker, String> {
    let queue_capacity = request_queue_capacity();
    let (sender, receiver) = fair_execution_queue(queue_capacity);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let join = thread::Builder::new()
        .name("rnb-engine-worker".to_string())
        .spawn(move || {
            let mut engine = match Engine::from_gguf_with_config(&model_path, load_config) {
                Ok(engine) => engine,
                Err(error) => {
                    let _ = ready_tx.send(Err(format!("load model: {error}")));
                    return;
                }
            };
            if engine.tokenizer.chat_template().is_none() {
                let _ = ready_tx.send(Err(
                    "GGUF does not contain tokenizer.chat_template".to_string()
                ));
                return;
            }
            let cache_bytes = response_session_cache_budget_bytes(
                engine.host_memory_plan(),
                explicit_cache_bytes,
            );
            let mut store = ResponseStore::new(cache_bytes);
            if ready_tx.send(Ok(cache_bytes)).is_err() {
                return;
            }
            worker_loop(&receiver, &mut engine, &mut store, &model_name);
        })
        .map_err(|error| format!("start engine worker: {error}"))?;

    let cache_bytes = match ready_rx.recv() {
        Ok(Ok(cache_bytes)) => cache_bytes,
        Ok(Err(error)) => {
            drop(sender);
            let _ = join.join();
            return Err(error);
        }
        Err(_) => {
            drop(sender);
            let _ = join.join();
            return Err("engine worker stopped during startup".to_string());
        }
    };
    eprintln!("Response session cache budget: {cache_bytes} bytes");
    Ok(Worker {
        sender: Some(sender),
        join: Some(join),
    })
}

fn request_queue_capacity() -> usize {
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .saturating_mul(2)
}

fn worker_loop(
    receiver: &FairExecutionReceiver<WorkerRequest>,
    engine: &mut Engine,
    store: &mut ResponseStore,
    model_name: &str,
) {
    while let Some(mut work) = receiver.receive() {
        if work.cancellation.token().is_cancelled() {
            continue;
        }
        let result = super::handle_worker_request(
            &mut work.stream,
            engine,
            store,
            work.cancellation.token(),
            model_name,
            work.request,
        );
        if let Err(error) = result {
            if error.status != 499 && !work.cancellation.token().is_cancelled() {
                let _ = write_json_response(&mut work.stream, error.status, &error.body());
            }
        }
    }
}

pub(super) struct RequestCancellation {
    token: GenerationCancellation,
    finished: Arc<AtomicBool>,
}

impl RequestCancellation {
    pub fn monitor(stream: &TcpStream, shutdown: GenerationCancellation) -> Self {
        let token = GenerationCancellation::new();
        let finished = Arc::new(AtomicBool::new(false));
        if let Ok(probe) = stream.try_clone() {
            let monitor_token = token.clone();
            let monitor_finished = Arc::clone(&finished);
            thread::spawn(move || {
                monitor_connection(probe, monitor_token, monitor_finished, shutdown)
            });
        }
        Self { token, finished }
    }

    pub fn token(&self) -> &GenerationCancellation {
        &self.token
    }
}

impl Drop for RequestCancellation {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
    }
}

fn monitor_connection(
    probe: TcpStream,
    cancellation: GenerationCancellation,
    finished: Arc<AtomicBool>,
    shutdown: GenerationCancellation,
) {
    let _ = probe.set_read_timeout(Some(Duration::from_millis(100)));
    let mut byte = [0_u8; 1];
    while !finished.load(Ordering::Acquire) {
        if shutdown.is_cancelled() {
            cancellation.cancel();
            let _ = probe.shutdown(TcpShutdown::Both);
            return;
        }
        match probe.peek(&mut byte) {
            Ok(0) => {
                cancellation.cancel();
                return;
            }
            Ok(_) => thread::sleep(Duration::from_millis(25)),
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
                ) => {}
            Err(_) => {
                cancellation.cancel();
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::time::Instant;

    #[test]
    fn server_shutdown_cancels_request_and_closes_connection() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect(address).unwrap();
        let (server, _) = listener.accept().unwrap();
        let shutdown = GenerationCancellation::new();
        let request = RequestCancellation::monitor(&server, shutdown.clone());

        shutdown.cancel();
        let deadline = Instant::now() + Duration::from_secs(1);
        while !request.token().is_cancelled() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }

        assert!(request.token().is_cancelled());
        let mut byte = [0_u8; 1];
        assert_eq!(client.peek(&mut byte).unwrap(), 0);
    }
}
