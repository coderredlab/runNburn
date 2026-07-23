use super::*;

fn identity() -> ResponseIdentity {
    ResponseIdentity {
        id: "resp_test".to_string(),
        suffix: "test".to_string(),
        created_at: 1,
    }
}

#[test]
fn builds_text_and_parallel_function_output_items() {
    let output = ParsedAssistantOutput {
        content: "checking".to_string(),
        tool_calls: vec![
            ParsedToolCall {
                name: "weather".to_string(),
                arguments: r#"{"city":"Seoul"}"#.to_string(),
            },
            ParsedToolCall {
                name: "time".to_string(),
                arguments: r#"{"zone":"UTC"}"#.to_string(),
            },
        ],
    };
    let items = output_items(&identity(), &output, "completed");

    assert_eq!(items.len(), 3);
    assert_eq!(items[0]["type"], "message");
    assert_eq!(items[1]["type"], "function_call");
    assert_eq!(items[2]["call_id"], "call_test_3");
}

#[test]
fn text_stream_opens_message_before_content_part() {
    let output = ParsedAssistantOutput {
        content: "4".to_string(),
        tool_calls: Vec::new(),
    };
    let events = buffered_output_events(&identity(), &output, "completed");

    assert_eq!(events[0]["type"], "response.output_item.added");
    assert_eq!(events[0]["item"]["content"], json!([]));
    assert_eq!(events[1]["type"], "response.content_part.added");
    assert_eq!(events[1]["part"]["text"], "");
}

#[test]
fn function_stream_events_have_open_delta_done_and_item_done_order() {
    let output = ParsedAssistantOutput {
        content: String::new(),
        tool_calls: vec![ParsedToolCall {
            name: "weather".to_string(),
            arguments: r#"{"city":"Seoul"}"#.to_string(),
        }],
    };
    let events = buffered_output_events(&identity(), &output, "completed");
    let kinds = events
        .iter()
        .map(|event| event["type"].as_str().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(
        kinds,
        vec![
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done"
        ]
    );
    assert_eq!(events[2]["name"], "weather");
}
