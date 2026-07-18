use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;
const REQUEST_READ_DEADLINE: Duration = Duration::from_secs(15);

#[derive(Debug)]
pub(super) struct HttpRequest {
    pub method: String,
    pub path: String,
    pub query: Option<String>,
    pub body: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct ApiError {
    pub status: u16,
    pub message: String,
    pub error_type: &'static str,
    pub param: Option<&'static str>,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn invalid(
        message: impl Into<String>,
        param: Option<&'static str>,
        code: Option<&'static str>,
    ) -> Self {
        Self {
            status: 400,
            message: message.into(),
            error_type: "invalid_request_error",
            param,
            code,
        }
    }

    pub fn unauthorized() -> Self {
        Self {
            status: 401,
            message: "Incorrect API key provided".to_string(),
            error_type: "invalid_request_error",
            param: None,
            code: Some("invalid_api_key"),
        }
    }

    pub fn model_not_found(model: &str) -> Self {
        Self {
            status: 404,
            message: format!("The model '{model}' does not exist"),
            error_type: "invalid_request_error",
            param: Some("model"),
            code: Some("model_not_found"),
        }
    }

    pub fn resource_not_found(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: 404,
            message: message.into(),
            error_type: "invalid_request_error",
            param,
            code: Some("not_found"),
        }
    }

    pub fn route_not_found() -> Self {
        Self {
            status: 404,
            message: "Not found".to_string(),
            error_type: "invalid_request_error",
            param: None,
            code: Some("not_found"),
        }
    }

    pub fn method_not_allowed() -> Self {
        Self {
            status: 405,
            message: "Method not allowed".to_string(),
            error_type: "invalid_request_error",
            param: None,
            code: Some("method_not_allowed"),
        }
    }

    pub fn cancelled() -> Self {
        Self {
            status: 499,
            message: "Request cancelled".to_string(),
            error_type: "invalid_request_error",
            param: None,
            code: Some("request_cancelled"),
        }
    }

    pub fn request_timeout() -> Self {
        Self {
            status: 408,
            message: "HTTP request body was not received within 15 seconds".to_string(),
            error_type: "invalid_request_error",
            param: None,
            code: Some("request_timeout"),
        }
    }

    pub fn overloaded() -> Self {
        Self {
            status: 429,
            message: "The server is handling too many queued requests".to_string(),
            error_type: "server_error",
            param: None,
            code: Some("server_overloaded"),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
            error_type: "server_error",
            param: None,
            code: Some("server_error"),
        }
    }

    pub fn body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        })
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.status, self.message)
    }
}

pub(super) fn read_request(
    stream: &mut TcpStream,
    after_headers: impl FnOnce(&HashMap<String, String>) -> Result<(), ApiError>,
) -> Result<HttpRequest, ApiError> {
    read_request_with_deadline(stream, REQUEST_READ_DEADLINE, after_headers)
}

fn read_request_with_deadline(
    stream: &mut TcpStream,
    read_deadline: Duration,
    after_headers: impl FnOnce(&HashMap<String, String>) -> Result<(), ApiError>,
) -> Result<HttpRequest, ApiError> {
    let mut bytes = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 8192];
    let deadline = Instant::now() + read_deadline;
    let header_end = loop {
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let header_end = index + 4;
            if header_end > MAX_HEADER_BYTES {
                return Err(ApiError::invalid(
                    "HTTP headers are too large",
                    None,
                    Some("headers_too_large"),
                ));
            }
            break header_end;
        }
        if bytes.len() >= MAX_HEADER_BYTES {
            return Err(ApiError::invalid(
                "HTTP headers are too large",
                None,
                Some("headers_too_large"),
            ));
        }
        let read = read_with_deadline(stream, &mut chunk, deadline, "HTTP request")?;
        if read == 0 {
            return Err(ApiError::invalid(
                "Incomplete HTTP request headers",
                None,
                Some("invalid_request"),
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
    };

    let header_text = std::str::from_utf8(&bytes[..header_end - 4]).map_err(|_| {
        ApiError::invalid("HTTP headers must be UTF-8", None, Some("invalid_request"))
    })?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or_default().to_string();
    let path = request_parts.next().unwrap_or_default().to_string();
    let version = request_parts.next().unwrap_or_default();
    if method.is_empty()
        || path.is_empty()
        || !matches!(version, "HTTP/1.0" | "HTTP/1.1")
        || request_parts.next().is_some()
    {
        return Err(ApiError::invalid(
            "Malformed HTTP request line",
            None,
            Some("invalid_request"),
        ));
    }

    let mut headers = HashMap::new();
    for line in lines {
        let (name, value) = line.split_once(':').ok_or_else(|| {
            ApiError::invalid("Malformed HTTP header", None, Some("invalid_request"))
        })?;
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() {
            return Err(ApiError::invalid(
                "Malformed HTTP header name",
                None,
                Some("invalid_request"),
            ));
        }
        if name == "content-length" && headers.contains_key(&name) {
            return Err(ApiError::invalid(
                "Duplicate Content-Length header",
                None,
                Some("invalid_request"),
            ));
        }
        headers.insert(name, value.trim().to_string());
    }
    after_headers(&headers)?;

    if headers.contains_key("transfer-encoding") {
        return Err(ApiError::invalid(
            "Transfer-Encoding is not supported; send Content-Length",
            None,
            Some("unsupported_transfer_encoding"),
        ));
    }

    let content_length = match headers.get("content-length") {
        Some(value) => value.parse::<usize>().map_err(|_| {
            ApiError::invalid(
                "Invalid Content-Length header",
                None,
                Some("invalid_request"),
            )
        })?,
        None => 0,
    };
    if content_length > MAX_BODY_BYTES {
        return Err(ApiError::invalid(
            "Request body is too large",
            None,
            Some("body_too_large"),
        ));
    }

    let target_len = header_end + content_length;
    while bytes.len() < target_len {
        let read = read_with_deadline(stream, &mut chunk, deadline, "HTTP body")?;
        if read == 0 {
            return Err(ApiError::invalid(
                "Incomplete HTTP request body",
                None,
                Some("invalid_request"),
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
    }

    Ok(HttpRequest {
        method,
        path: path.split('?').next().unwrap_or(&path).to_string(),
        query: path.split_once('?').map(|(_, query)| query.to_string()),
        body: bytes[header_end..target_len].to_vec(),
    })
}

fn read_with_deadline(
    stream: &mut TcpStream,
    buffer: &mut [u8],
    deadline: Instant,
    context: &str,
) -> Result<usize, ApiError> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(ApiError::request_timeout)?;
    stream
        .set_read_timeout(Some(remaining))
        .map_err(|error| ApiError::internal(format!("set request timeout: {error}")))?;
    stream.read(buffer).map_err(|error| {
        if matches!(
            error.kind(),
            io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
        ) {
            ApiError::request_timeout()
        } else {
            ApiError::invalid(format!("read {context}: {error}"), None, None)
        }
    })
}

pub(super) fn write_json_response(
    stream: &mut TcpStream,
    status: u16,
    body: &Value,
) -> io::Result<()> {
    let bytes = serde_json::to_vec(body).map_err(io::Error::other)?;
    write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_line(status),
        bytes.len()
    )?;
    stream.write_all(&bytes)
}

pub(super) fn write_sse_headers(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n",
    )?;
    stream.flush()
}

pub(super) fn write_sse_json(stream: &mut TcpStream, value: &Value) -> io::Result<()> {
    stream.write_all(b"data: ")?;
    serde_json::to_writer(&mut *stream, value).map_err(io::Error::other)?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

pub(super) fn write_sse_event(
    stream: &mut TcpStream,
    event: &str,
    value: &Value,
) -> io::Result<()> {
    stream.write_all(b"event: ")?;
    stream.write_all(event.as_bytes())?;
    stream.write_all(b"\ndata: ")?;
    serde_json::to_writer(&mut *stream, value).map_err(io::Error::other)?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

pub(super) fn write_sse_done(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()
}

fn status_line(status: u16) -> &'static str {
    match status {
        200 => "200 OK",
        400 => "400 Bad Request",
        401 => "401 Unauthorized",
        404 => "404 Not Found",
        408 => "408 Request Timeout",
        405 => "405 Method Not Allowed",
        429 => "429 Too Many Requests",
        _ => "500 Internal Server Error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn absolute_read_deadline_rejects_trickled_headers() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let writer = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            for byte in b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n" {
                if stream.write_all(&[*byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        });
        let (mut stream, _) = listener.accept().unwrap();
        let started = Instant::now();

        let error = read_request_with_deadline(&mut stream, Duration::from_millis(50), |_| Ok(()))
            .unwrap_err();

        assert_eq!(error.status, 408);
        assert!(started.elapsed() < Duration::from_millis(500));
        drop(stream);
        writer.join().unwrap();
    }

    #[test]
    fn header_gate_rejects_before_reading_declared_body() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let writer = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(
                    b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\n\r\n",
                )
                .unwrap();
        });
        let (mut stream, _) = listener.accept().unwrap();

        let error = read_request_with_deadline(&mut stream, Duration::from_secs(1), |_| {
            Err(ApiError::unauthorized())
        })
        .unwrap_err();

        assert_eq!(error.status, 401);
        writer.join().unwrap();
    }
}
