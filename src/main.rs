//! hoosh CLI — inference gateway server and management tool.
//!
//! Usage:
//!   hoosh serve                         # Start the gateway server (port 8088)
//!   hoosh serve --port 9000             # Custom port
//!   hoosh models                        # List available models across providers
//!   hoosh infer --model llama3 "hello"  # One-shot inference
//!   hoosh health                        # Check provider health
//!   hoosh info                          # System info (providers, hardware)
//!   hoosh --version

use clap::{Parser, Subcommand};
use hoosh::client::HooshClient;
use hoosh::config::HooshConfig;

#[derive(Parser)]
#[command(name = "hoosh", version, about = "AI inference gateway")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference gateway server
    Serve {
        /// Port to listen on (overrides config file)
        #[arg(short, long)]
        port: Option<u16>,
        /// Bind address (overrides config file)
        #[arg(long)]
        bind: Option<String>,
        /// Path to config file (default: hoosh.toml)
        #[arg(short, long)]
        config: Option<String>,
    },
    /// List available models across all providers
    Models {
        /// hoosh server URL
        #[arg(long, default_value = "http://127.0.0.1:8088")]
        server: String,
    },
    /// Run a one-shot inference
    Infer {
        /// Model identifier
        #[arg(short, long)]
        model: String,
        /// Prompt text
        prompt: String,
        /// Stream output token by token
        #[arg(long)]
        stream: bool,
        /// hoosh server URL
        #[arg(long, default_value = "http://127.0.0.1:8088")]
        server: String,
    },
    /// Check health of all configured providers
    Health {
        /// hoosh server URL
        #[arg(long, default_value = "http://127.0.0.1:8088")]
        server: String,
    },
    /// Transcribe an audio file (speech-to-text)
    Transcribe {
        /// Path to audio file (WAV)
        file: String,
        /// hoosh server URL
        #[arg(long, default_value = "http://127.0.0.1:8088")]
        server: String,
    },
    /// Show system info (providers, hardware, config)
    Info,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, bind, config } => {
            let hoosh_config = if let Some(path) = config {
                HooshConfig::load(&path)?
            } else {
                HooshConfig::load_or_default()
            };
            let server_config = hoosh_config.into_server_config(
                bind.as_deref(),
                port,
            );
            hoosh::server::run(server_config).await?;
        }
        Commands::Models { server } => {
            let client = HooshClient::new(&server);
            match client.list_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        println!("No models available.");
                    } else {
                        for m in &models {
                            println!(
                                "  {} (provider: {}, available: {})",
                                m.id, m.provider, m.available
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to connect to hoosh server at {}: {}", server, e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Infer {
            model,
            prompt,
            stream,
            server,
        } => {
            let client = HooshClient::new(&server);
            let req = hoosh::InferenceRequest {
                model,
                prompt,
                stream,
                ..Default::default()
            };
            if stream {
                use std::io::Write;
                match client.infer_stream(&req).await {
                    Ok(mut rx) => {
                        while let Some(result) = rx.recv().await {
                            match result {
                                Ok(token) => {
                                    print!("{token}");
                                    std::io::stdout().flush().ok();
                                }
                                Err(e) => {
                                    eprintln!("\nStream error: {e}");
                                    std::process::exit(1);
                                }
                            }
                        }
                        println!();
                    }
                    Err(e) => {
                        eprintln!("Inference failed: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                match client.infer(&req).await {
                    Ok(resp) => println!("{}", resp.text),
                    Err(e) => {
                        eprintln!("Inference failed: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Commands::Health { server } => {
            let client = HooshClient::new(&server);
            match client.health().await {
                Ok(true) => println!("hoosh server at {} is healthy", server),
                Ok(false) => {
                    println!("hoosh server at {} is unhealthy", server);
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Health check failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Transcribe { file, server } => {
            let audio = std::fs::read(&file).unwrap_or_else(|e| {
                eprintln!("Failed to read '{}': {}", file, e);
                std::process::exit(1);
            });

            let client = reqwest::Client::new();
            let url = format!("{}/v1/audio/transcriptions", server.trim_end_matches('/'));
            match client.post(&url).body(audio).send().await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let body: serde_json::Value = resp.json().await.unwrap_or_default();
                        if let Some(text) = body["text"].as_str() {
                            println!("{text}");
                        } else {
                            println!("{}", serde_json::to_string_pretty(&body).unwrap_or_default());
                        }
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        eprintln!("Transcription failed ({status}): {body}");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to connect to hoosh server at {server}: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Info => {
            println!("hoosh v{}", env!("CARGO_PKG_VERSION"));
            println!();

            #[cfg(feature = "hwaccel")]
            {
                let hw = hoosh::hardware::HardwareManager::detect();
                println!("Hardware:");
                for line in hw.summary() {
                    println!("{line}");
                }
                if hw.has_accelerator() {
                    let mem_gb =
                        hw.total_accelerator_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
                    println!("  Total accelerator memory: {mem_gb:.1} GB");
                }
                println!();
            }

            println!("Local providers:");
            #[cfg(feature = "ollama")]
            println!("  ollama (enabled)");
            #[cfg(feature = "llamacpp")]
            println!("  llamacpp (enabled)");
            #[cfg(feature = "synapse")]
            println!("  synapse (enabled)");
            #[cfg(feature = "lmstudio")]
            println!("  lmstudio (enabled)");
            #[cfg(feature = "localai")]
            println!("  localai (enabled)");
            println!();

            println!("Remote providers:");
            #[cfg(feature = "openai")]
            println!("  openai (enabled)");
            #[cfg(feature = "anthropic")]
            println!("  anthropic (enabled)");
            #[cfg(feature = "deepseek")]
            println!("  deepseek (enabled)");
            #[cfg(feature = "mistral")]
            println!("  mistral (enabled)");
            #[cfg(feature = "groq")]
            println!("  groq (enabled)");
            #[cfg(feature = "openrouter")]
            println!("  openrouter (enabled)");
            println!();

            #[cfg(feature = "whisper")]
            println!("Speech-to-text: whisper.cpp (enabled)");
            #[cfg(not(feature = "whisper"))]
            println!("Speech-to-text: disabled (enable 'whisper' feature)");
        }
    }

    Ok(())
}
