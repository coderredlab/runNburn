use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};

/// Multi-producer FIFO admission queue for a single execution worker.
pub struct FairExecutionSender<T> {
    sender: SyncSender<T>,
}

impl<T> Clone for FairExecutionSender<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}
#[derive(Debug)]

pub enum FairExecutionSubmitError<T> {
    Full(T),
    Disconnected(T),
}

impl<T> FairExecutionSender<T> {
    pub fn submit(&self, request: T) -> Result<(), FairExecutionSubmitError<T>> {
        self.sender.try_send(request).map_err(|error| match error {
            TrySendError::Full(request) => FairExecutionSubmitError::Full(request),
            TrySendError::Disconnected(request) => FairExecutionSubmitError::Disconnected(request),
        })
    }
}

pub struct FairExecutionReceiver<T> {
    receiver: Receiver<T>,
}

impl<T> FairExecutionReceiver<T> {
    pub fn receive(&self) -> Option<T> {
        self.receiver.recv().ok()
    }
}

pub fn fair_execution_queue<T>(
    capacity: usize,
) -> (FairExecutionSender<T>, FairExecutionReceiver<T>) {
    let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
    (
        FairExecutionSender { sender },
        FairExecutionReceiver { receiver },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receives_requests_in_submission_order() {
        let (sender, receiver) = fair_execution_queue(3);
        let second_sender = sender.clone();
        sender.submit(1_u8).unwrap();
        second_sender.submit(2_u8).unwrap();
        sender.submit(3_u8).unwrap();

        assert_eq!(receiver.receive(), Some(1));
        assert_eq!(receiver.receive(), Some(2));
        assert_eq!(receiver.receive(), Some(3));
    }

    #[test]
    fn rejects_work_when_bounded_queue_is_full() {
        let (sender, _receiver) = fair_execution_queue(1);
        sender.submit(1_u8).unwrap();
        assert!(matches!(
            sender.submit(2_u8),
            Err(FairExecutionSubmitError::Full(2))
        ));
    }
}
