use futures::future::join_all;
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_template,
    llm::OpenAI,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::Message,
    template_jinja2,
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model("gpt-4-turbo-preview");
    let generate_explanation_chain = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_jinja2!(
            GENERATE_EXPLANATION,
            "question"
        )))
        .llm(llm.clone())
        .build()
        .unwrap();

    let question=String::from("Encuentra la expansión en serie de Fourier de la función f(x) = x, dentro de los límites [–π, π]");
    let result = generate_explanation_chain
        .invoke(prompt_args! {
        "question"=>&question})
        .await
        .unwrap();

    let steps = extract_and_append(&result);

    let code_steps = generate_code(&steps).await;

    println!("Code Step: {}", code_steps.len());
    let speech_steps = generate_speech(&question, &steps).await;
    println!("Speech Step: {}", speech_steps.len());

    for x in 0..steps.len() {
        println!("Code: {}", code_steps[x]);
        println!("\n");
        println!("Speech: {}", speech_steps[x]);
        println!("\n\n\n\n");
    }
}

fn extract_and_append(text: &str) -> Vec<String> {
    let mut vec = Vec::new();
    let marker = "```";
    let mut start = 0;
    let mut end = 0;
    let mut in_code_block = false;

    for (index, _) in text.match_indices(marker) {
        if !in_code_block {
            start = index + marker.len();
            in_code_block = true;
        } else {
            end = index;
            in_code_block = false;
            let code_block = &text[start..end].trim();
            vec.push(code_block.to_string());
        }
    }

    vec
}

async fn generate_code(steps: &[String]) -> Vec<String> {
    let llm = OpenAI::default().with_model("gpt-4-turbo-preview");
    let mut futures = Vec::new();

    for step in steps {
        let llm_clone = llm.clone();
        let step_clone = step.clone();
        let future = async move {
            let generate_code_chain = LLMChainBuilder::new()
                .prompt(message_formatter![
                    fmt_message!(Message::new_system_message(&GENERATE_CODE_SYSTEM)), // Assuming `code_system` is properly formatted for system-message use
                    fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                        GENERATE_CODE_USER,
                        "request"
                    ))),
                ])
                .llm(llm_clone)
                .build()
                .unwrap();

            let result = generate_code_chain
                .invoke(prompt_args! {
                    "request" => &step_clone
                })
                .await
                .unwrap();
            result // Return result of this future
        };
        futures.push(future); // Push the future into the vector.
    }

    // Await all futures to complete and collect their results.
    let code_user = join_all(futures).await;
    code_user
}

async fn generate_speech(question: &str, steps: &[String]) -> Vec<String> {
    let llm = OpenAI::default().with_model("gpt-4-turbo-preview");
    let mut futures = Vec::new();
    for step in steps {
        let generate_speech_chain = LLMChainBuilder::new()
            .prompt(HumanMessagePromptTemplate::new(template_jinja2!(
                GENERATE_SPEECH_EXPLANATION,
                "question",
                "step"
            )))
            .llm(llm.clone())
            .build()
            .unwrap();
        let future = async move {
            let result = generate_speech_chain
                .invoke(prompt_args! {
                    "question" => question,
                    "step" => step
                })
                .await
                .unwrap();
            result // Return result of this future
        };
        futures.push(future); // Push the future into the vector.
    }
    // Await all futures to complete and collect their results.
    let speech = join_all(futures).await;
    speech
}

const GENERATE_EXPLANATION: &str = r#"You should answer questions, step by step,each steap should be wrap in ```step```

Example:
question: Whats the result of 10+30/5
answer:

```
xxx
```

```
Division First
[ 30 / 5 = 6 ]
```

```
Then, Addition:
[ 10 + 6 = 16 ]
```

```
So, the answer is 16.
```


Generate step by step explanation for the following question: {{question}}
answer:
"#;

const GENERATE_CODE_SYSTEM: &str = r#"
Write Manim scripts for animations in Python. Generate code, not text. Never explain code. Never add functions. Never add comments. Never infinte loops. Never use other library than Manim/math. Only complete the code block. Use variables with length of maximum 2 characters. At the end use 'self.play'.

```
from manim import *
from math import *

class GenScene(Scene):
    def construct(self):
        # Write here
```
"#;

const GENERATE_CODE_USER: &str = r#"
Animation Request:{{request}}
"#;

const GENERATE_SPEECH_EXPLANATION: &str = r#"
You will resive a question, and a step of the resolution,you should generate text explaining the step in words.
The script should be just for that step. only text.
question: {{question}}
step: {{step}}

script:
"#;
